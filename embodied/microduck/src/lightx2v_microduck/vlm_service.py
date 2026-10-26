"""Pinned Transformers VLM reference service for MicroDuck planner experiments.

This is a model reference backend, not a LightX2V optimization. It preserves
raw model responses; invalid plans are rejected by the existing robot executor.
There is no rule-based answer, output repair, or privileged simulator input.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.metadata
import io
import json
import re
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from PIL import Image


MAX_BODY_BYTES = 8 * 1024 * 1024
MAX_IMAGE_PIXELS = 2048 * 2048
MAX_TEXT_CHARS = 64000


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical(value) + b"\n")
    temporary.replace(path)


def verify_assets(directory, lock_path):
    """Hash the actual local bytes before loading either processor or model."""
    directory, lock_path = Path(directory), Path(lock_path)
    lock_bytes = lock_path.read_bytes()
    lock = json.loads(lock_bytes)
    if (lock.get("schema") != "microduck_vlm_assets_v1"
            or lock.get("repository") != "Qwen/Qwen3.5-9B"
            or not re.fullmatch(r"[0-9a-f]{40}", lock.get("revision", ""))):
        raise ValueError("requires the pinned Qwen3.5-9B asset contract")
    names = [entry["name"] for entry in lock["files"]]
    if (len(names) != len(set(names)) or any(Path(n).name != n or n.startswith(".") for n in names)):
        raise ValueError("invalid or duplicate asset name")
    required = {"config.json", "chat_template.jinja", "preprocessor_config.json",
                "tokenizer.json", "tokenizer_config.json", "model.safetensors.index.json"}
    if not required.issubset(names):
        raise ValueError("asset lock lacks required processor/model files")
    # AutoProcessor/from_pretrained must not silently discover unpinned extras.
    extras = set(p.name for p in directory.iterdir()) - set(names) - {".download.lock"}
    if extras:
        raise ValueError("model directory contains unpinned files")
    verified = {}
    for entry in lock["files"]:
        path = directory / entry["name"]
        size = entry["bytes"]
        if path.is_symlink() or not path.is_file() or path.stat().st_size != size:
            raise ValueError("missing or mismatched asset: " + entry["name"])
        sha = hashlib.sha256()
        git = hashlib.sha1(b"blob " + str(size).encode() + b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                sha.update(chunk)
                git.update(chunk)
        expected = entry.get("sha256")
        matches = sha.hexdigest() == expected if expected else git.hexdigest() == entry.get("git_blob_sha1")
        if not matches:
            raise ValueError("asset content checksum mismatch: " + entry["name"])
        verified[entry["name"]] = {"sha256": sha.hexdigest(), "bytes": size}
    config = json.loads((directory / "config.json").read_text())
    if config.get("architectures") != ["Qwen3_5ForConditionalGeneration"]:
        raise ValueError("unexpected model architecture")
    weights = set(json.loads((directory / "model.safetensors.index.json").read_text())["weight_map"].values())
    if not weights or not weights.issubset(verified) or any(not n.endswith(".safetensors") for n in weights):
        raise ValueError("weight index refers to unpinned shards")
    return {"repository": lock["repository"], "revision": lock["revision"],
            "lock_sha256": hashlib.sha256(lock_bytes).hexdigest(), "files": verified}


def parse_request(payload, model_id):
    if not isinstance(payload, dict):
        raise ValueError("request must be an object")
    allowed = {"model", "messages", "temperature", "max_tokens", "stream", "metadata"}
    if set(payload) - allowed:
        raise ValueError("unsupported request fields")
    if payload.get("model") != model_id:
        raise ValueError("requested model does not match this pinned service")
    if payload.get("stream", False) is not False or payload.get("temperature", 0) != 0:
        raise ValueError("this reference protocol uses non-streamed greedy decoding")
    limit = payload.get("max_tokens", 512)
    if type(limit) is not int or not 1 <= limit <= 512:
        raise ValueError("max_tokens must be an integer in 1..512")
    messages = payload.get("messages")
    if not isinstance(messages, list) or not 1 <= len(messages) <= 8:
        raise ValueError("expected one to eight messages")
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict) or len(canonical(metadata)) > 2048:
        raise ValueError("metadata must be a bounded object")
    normalized, images, text_count = [], [], 0
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or set(message) != {"role", "content"}:
            raise ValueError("each message needs role and content")
        role = message["role"]
        if role not in ("system", "user", "assistant") or (role == "system" and index != 0):
            raise ValueError("invalid message role/order")
        content = message["content"]
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        if not isinstance(content, list) or len(content) > 16:
            raise ValueError("invalid message content")
        items = []
        for item in content:
            if not isinstance(item, dict):
                raise ValueError("invalid content item")
            if item.get("type") == "text" and set(item) == {"type", "text"} and isinstance(item["text"], str):
                text_count += len(item["text"])
                items.append(dict(item))
            elif item.get("type") == "image_url" and set(item) == {"type", "image_url"}:
                if role != "user" or len(images) >= 2:
                    raise ValueError("only two user observation images are supported")
                reference = item["image_url"]
                if not isinstance(reference, dict) or set(reference) != {"url"}:
                    raise ValueError("image_url must contain only an inline URL")
                url = reference["url"]
                if not isinstance(url, str):
                    raise ValueError("invalid image URL")
                match = re.fullmatch(r"data:image/(jpeg|png);base64,([A-Za-z0-9+/=]+)", url)
                if not match:
                    raise ValueError("only inline JPEG/PNG observations are allowed")
                data = base64.b64decode(match[2], validate=True)
                with Image.open(io.BytesIO(data)) as source:
                    if source.format != {"jpeg": "JPEG", "png": "PNG"}[match[1]]:
                        raise ValueError("image MIME type differs from its content")
                    if source.width * source.height > MAX_IMAGE_PIXELS:
                        raise ValueError("observation image is too large")
                    rgb = source.convert("RGB")
                images.append({"sha256": hashlib.sha256(data).hexdigest(), "size": list(rgb.size)})
                items.append({"type": "image", "image": rgb})
            else:
                raise ValueError("unsupported content item")
        normalized.append({"role": role, "content": items})
    if text_count > MAX_TEXT_CHARS or not images or messages[-1]["role"] != "user":
        raise ValueError("requires bounded text and a final user observation with images")
    return {"messages": normalized, "max_tokens": limit, "images": images, "metadata": metadata}


class InferenceCancelled(RuntimeError):
    pass


class InferenceDeadline(RuntimeError):
    pass


class QwenReferenceEngine:
    def __init__(self, directory, lock_path, *, device="cuda:0", deadline_s=4.5, cpu_threads=None):
        if cpu_threads is not None and (type(cpu_threads) is not int or not 1 <= cpu_threads <= 192):
            raise ValueError("cpu_threads must be an integer in 1..192 or unspecified")
        artifact = verify_assets(directory, lock_path)
        # This environment is isolated from both the host and LightX2V's base.
        if importlib.metadata.version("transformers") != "5.5.4":
            raise ValueError("the verified reference requires transformers==5.5.4")
        import torch
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

        # Explicit per-process execution configuration. The legacy default is
        # retained unless requested; record the actual setting in the identity.
        # This changes CPU scheduling, never input shape, decoding or weights.
        if cpu_threads is not None:
            torch.set_num_threads(cpu_threads)

        if not device.startswith("cuda:") or not torch.cuda.is_available():
            raise ValueError("the initial reference protocol requires one CUDA GPU")
        if not 0 < deadline_s <= 30:
            raise ValueError("invalid request deadline")
        self.device, self.deadline_s = torch.device(device), float(deadline_s)
        self.processor = AutoProcessor.from_pretrained(str(directory), local_files_only=True, trust_remote_code=False)
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            str(directory), local_files_only=True, trust_remote_code=False,
            dtype=torch.bfloat16, attn_implementation="sdpa",
        ).to(self.device).eval()
        self.model.requires_grad_(False)
        self.model_id = "Qwen3.5-9B@" + artifact["revision"][:12]
        self.identity = {
            "model_id": self.model_id, "backend": "transformers_reference", "lightx2v_acceleration": False,
            "artifact": artifact, "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "safetensors", "Pillow")},
            "precision": "bfloat16", "attention": "sdpa", "enable_thinking": False,
            "decoding": "greedy", "request_deadline_s": self.deadline_s,
            "maximum_input_tokens": 8192, "maximum_output_tokens": 512,
            "structured_output": "prompt_only_no_repair", "hardware_validated": False,
            "cpu_threads": torch.get_num_threads(),
            "cpu_interop_threads": torch.get_num_interop_threads(),
            "cpu_threads_requested": cpu_threads,
        }

    def generate(self, request, cancelled):
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList

        started = time.monotonic()
        deadline = started + self.deadline_s

        class Stop(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return cancelled.is_set() or time.monotonic() >= deadline

        inputs = self.processor.apply_chat_template(
            request["messages"], tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt", enable_thinking=False,
        ).to(self.device)
        count = inputs["input_ids"].shape[-1]
        if count > 8192:
            raise ValueError("model context exceeds the reference input limit")
        if cancelled.is_set():
            raise InferenceCancelled()
        if time.monotonic() >= deadline:
            raise InferenceDeadline()
        torch.cuda.synchronize(self.device)
        generation_start = time.monotonic()
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs, do_sample=False, max_new_tokens=request["max_tokens"],
                stopping_criteria=StoppingCriteriaList([Stop()]), use_cache=True,
            )
        torch.cuda.synchronize(self.device)
        end = time.monotonic()
        tokens = generated[0, count:]
        raw = self.processor.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        # Keep incomplete output in the evidence even if the deadline was hit.
        status = "cancelled" if cancelled.is_set() else "deadline" if end >= deadline else "complete"
        return {"content": raw, "status": status, "prompt_tokens": int(count),
                "completion_tokens": int(tokens.shape[0]), "total_ms": (end - started) * 1000,
                "generation_ms": (end - generation_start) * 1000,
                "cpu_threads": torch.get_num_threads(),
                "cpu_interop_threads": torch.get_num_interop_threads(),
                "finish_reason": "length" if tokens.shape[0] >= request["max_tokens"] else "stop"}


class VLMServer(ThreadingHTTPServer):
    # PyTorch/CUDA must not be destroyed while an HTTP thread is generating.
    # The job's outer process timeout remains the hard bound for stuck kernels.
    daemon_threads = False
    block_on_close = True

    def __init__(self, address, engine, records):
        self.engine, self.records = engine, Path(records)
        self.records.mkdir(parents=True, exist_ok=True)
        self.inference_lock = threading.Lock()
        self.requests_lock = threading.Lock()
        self.active = {}
        self.draining = False
        self.identity_sha256 = hashlib.sha256(canonical(engine.identity)).hexdigest()
        write_json(self.records / "service.json", {"identity": engine.identity, "identity_sha256": self.identity_sha256,
                                                  "started_unix": time.time()})
        super().__init__(address, VLMHandler)

    def server_close(self):
        started = time.time()
        with self.requests_lock:
            self.draining = True
            pending = list(self.active)
            for event in self.active.values():
                event.set()
        receipt = {"status": "draining", "started_unix": started,
                   "cancelled_request_ids": pending, "inference_threads_joined": False}
        write_json(self.records / "shutdown.json", receipt)
        # ThreadingMixIn joins non-daemon request threads, including their
        # evidence writes and finally blocks. Never report success before this.
        super().server_close()
        with self.requests_lock:
            remaining = list(self.active)
        receipt.update(status="complete" if not remaining else "failed",
                       finished_unix=time.time(), remaining_request_ids=remaining,
                       inference_threads_joined=not remaining)
        write_json(self.records / "shutdown.json", receipt)
        if remaining:
            raise RuntimeError("model requests remained active after service close")


class VLMHandler(BaseHTTPRequestHandler):
    def setup(self):
        super().setup()
        self.connection.settimeout(10)

    def log_message(self, *args):
        pass

    def reply(self, value, status=200):
        raw = canonical(value)
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(raw)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        if self.path != "/health":
            return self.reply({"error": "not_found"}, 404)
        with self.server.requests_lock:
            active = list(self.server.active)
            ready = not self.server.draining
        self.reply({"ready": ready, "identity": self.server.engine.identity,
                    "identity_sha256": self.server.identity_sha256, "active_requests": active})

    def do_POST(self):
        match = re.fullmatch(r"/v1/requests/([0-9a-f]{32})/cancel", self.path)
        if match:
            with self.server.requests_lock:
                cancel = self.server.active.get(match[1])
                if cancel is not None:
                    cancel.set()
                    with (self.server.records / "cancellations.jsonl").open("a") as log:
                        log.write(canonical({"request_id": match[1], "received_unix": time.time()}).decode() + "\n")
            return self.reply({"cancel_requested": cancel is not None}, 200 if cancel is not None else 404)
        if self.path != "/v1/chat/completions":
            return self.reply({"error": "not_found"}, 404)
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= MAX_BODY_BYTES or self.headers.get("Transfer-Encoding"):
                raise ValueError("invalid request length")
            raw = self.rfile.read(size)
            if len(raw) != size:
                raise ValueError("truncated request")
            payload = json.loads(raw)
            request = parse_request(payload, self.server.engine.model_id)
            identifier = self.headers.get("X-Microduck-Request-Id", uuid.uuid4().hex)
            if not re.fullmatch(r"[0-9a-f]{32}", identifier):
                raise ValueError("invalid request id")
        except (ValueError, TypeError, KeyError, OSError, Image.DecompressionBombError):
            return self.reply({"error": "invalid_request"}, 400)
        directory = self.server.records / identifier
        # Reserve the receipt before admission. A duplicate must not replace
        # an existing cancellation event or overwrite that request's evidence.
        try:
            directory.mkdir()
        except FileExistsError:
            return self.reply({"error": "request_id_already_used"}, 409)
        digest = hashlib.sha256(canonical(payload)).hexdigest()
        evidence = {"request_id": identifier, "request_sha256": digest,
                    "identity_sha256": self.server.identity_sha256,
                    "model_id": self.server.engine.model_id,
                    "backend": self.server.engine.identity["backend"]}
        write_json(directory / "request.json", payload)
        write_json(directory / "input.json", {"request_sha256": digest, "images": request["images"],
                                               "identity_sha256": self.server.identity_sha256,
                                               "received_unix": time.time()})
        cancelled = threading.Event()
        # Register admission atomically against shutdown. Closing the listener
        # alone cannot stop a request that has already been accepted.
        with self.server.requests_lock:
            rejected = (503 if self.server.draining else
                        429 if not self.server.inference_lock.acquire(blocking=False) else None)
            if rejected is None:
                self.server.active[identifier] = cancelled
        if rejected is not None:
            response = {"error": "service_draining" if rejected == 503 else "model_busy",
                        "microduck_evidence": evidence}
            write_json(directory / "error.json", {"type": "RequestNotAdmitted", "http_status": rejected,
                       "inference_started": False, "response": response, "finished_unix": time.time()})
            return self.reply(response, rejected)
        try:
            result = self.server.engine.generate(request, cancelled)
            evidence = {"request_id": identifier, "request_sha256": digest,
                        "identity_sha256": self.server.identity_sha256,
                        "model_id": self.server.engine.model_id, "backend": self.server.engine.identity["backend"],
                        "total_ms": result["total_ms"], "generation_ms": result["generation_ms"]}
            response = {
                "id": identifier, "object": "chat.completion", "model": self.server.engine.model_id,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": result["content"]},
                             "finish_reason": result["finish_reason"]}],
                "usage": {"prompt_tokens": result["prompt_tokens"], "completion_tokens": result["completion_tokens"]},
                "microduck_evidence": evidence,
            }
            write_json(directory / "result.json", {"result": result, "response": response, "finished_unix": time.time()})
            if result["status"] != "complete":
                return self.reply({"error": result["status"], "microduck_evidence": evidence},
                                  504 if result["status"] == "deadline" else 409)
            self.reply(response)
        except Exception as exc:
            status = 409 if isinstance(exc, InferenceCancelled) else 504 if isinstance(exc, InferenceDeadline) else 500
            response = {"error": "inference_failed", "request_id": identifier, "type": type(exc).__name__,
                        "microduck_evidence": evidence}
            write_json(directory / "error.json", {"type": type(exc).__name__, "http_status": status,
                       "response": response, "finished_unix": time.time()})
            self.reply(response, status)
        finally:
            with self.server.requests_lock:
                self.server.active.pop(identifier, None)
            self.server.inference_lock.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18769)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--request-deadline-s", type=float, default=4.5)
    parser.add_argument("--cpu-threads", type=int, help="Explicit CPU thread budget; legacy default when omitted")
    args = parser.parse_args()
    engine = QwenReferenceEngine(args.model_dir, args.lock, device=args.device,
                                deadline_s=args.request_deadline_s, cpu_threads=args.cpu_threads)
    with VLMServer(("127.0.0.1", args.port), engine, args.records) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()
