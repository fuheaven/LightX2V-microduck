"""Independent, standard-library gallery and process-isolated live simulation API.

No model, MuJoCo, training monitor or GPU is imported by this HTTP service.
Heavy dependencies live in a bounded child process, only after an explicit run.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import mimetypes
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

REPO = Path(__file__).resolve().parents[4]
MEDIA = REPO / "examples/microduck/visualization"
STATIC = Path(__file__).parent / "resources/visualization"
ACTIVE = {"starting", "verifying", "loading", "warming", "running", "cancelling", "saving"}
SCENES = ("room_clear", "room_curriculum_one", "room_randomized")


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False))
    temporary.replace(path)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default


def load_catalog(root=MEDIA):
    """Verify packaged media and bind each displayed outcome to its episode."""
    root = Path(root).resolve()
    catalog = read_json(root / "catalog.json")
    if not catalog or catalog.get("schema") != "microduck_visualization_v1":
        raise ValueError("missing or unsupported gallery catalog")
    for name, entry in catalog["files"].items():
        path = (root / name).resolve()
        if path.parent != root or path.name != name:
            raise ValueError("gallery paths must be flat filenames")
        if path.stat().st_size != entry["bytes"] or hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"gallery checksum mismatch: {name}")
    ids = set()
    for group in catalog["groups"]:
        if group["formal_acceptance"] or group["hardware_validated"]:
            raise ValueError("historical development recordings are not acceptance")
        assessment = read_json(root / group["assessment"])
        if assessment["checkpoint_sha256"] != catalog["checkpoint_sha256"]:
            raise ValueError("assessment model mismatch")
        if group["id"] == "clear-room":
            count, total = assessment["successes"]["optimized"], assessment["cases_per_variant"]
        elif group["id"] == "single-obstacle":
            # This assessment also contains clear-room cases: never count all 81.
            evaluation = read_json(root / group["evaluation"])
            rows = [e for e in evaluation["episodes"] if e.get("scene", evaluation.get("scene")) == group["scene"]]
            if not rows:
                rows = [e for e in evaluation["episodes"] if 2300 <= e["seed"] < 2320]
            count, total = sum(e["strict_model_success"] for e in rows), len(rows)
        else:
            count, total = assessment["realtime_successes"], assessment["episodes"]
        if (count, total) != (group["passed"], group["total"]):
            raise ValueError(f"cohort counts disagree: {group['id']}")
        for clip in group["clips"]:
            if clip["id"] in ids:
                raise ValueError("duplicate clip id")
            ids.add(clip["id"])
            for field in ("video", "poster", "report"):
                if clip[field] not in catalog["files"]:
                    raise ValueError("unverified clip asset")
            episode = read_json(root / clip["report"])
            success = episode["realtime_success"] if group["mode"] == "realtime" else episode["strict_model_success"]
            if ((clip["task"], clip["seed"], clip["passed"], clip["mode"], clip["fallback_ticks"])
                    != (episode["task"], episode["seed"], success, episode["mode"], episode["fallback_control_ticks"])
                    or episode["policy"]["artifact_sha256"] != catalog["checkpoint_sha256"]):
                raise ValueError("clip outcome differs from its episode")
    return catalog


def validate_run(value):
    if not isinstance(value, dict) or set(value) - {"policy", "scene", "color", "seed", "seconds", "variant"}:
        raise ValueError("unsupported run options")
    result = dict(policy="fastwam", scene="room_clear", color="red", seed=2000,
                  seconds=30, variant="optimized")
    result.update(value)
    if (result["policy"] not in ("fastwam", "official")
            or type(result["seed"]) is not int or not 0 <= result["seed"] <= 1000000
            or type(result["seconds"]) is not int or not 2 <= result["seconds"] <= 60):
        raise ValueError("invalid policy, scene, color, seed or duration")
    if result["policy"] == "official":
        result.update(scene="flat", color=None, variant=None)
    elif (result["scene"] not in SCENES or result["color"] not in ("red", "blue", "yellow")
          or result["variant"] not in ("plain", "optimized")):
        raise ValueError("invalid navigation scene, color or variant")
    return result


class BusyError(ValueError):
    pass


class LiveJobs:
    def __init__(self, data, *, enable_live=False, enable_fastwam=False):
        self.data = Path(data).resolve()
        self.enabled, self.fastwam = enable_live, enable_fastwam
        self.root = self.data / "results/visualization"
        self.lock = threading.RLock()
        self.jobs, self.processes = {}, {}
        if self.root.exists():
            # Only bounded, completed metadata is restored. Never resume a run.
            for directory in sorted(self.root.iterdir(), key=lambda p: p.stat().st_mtime)[-24:]:
                if not re.fullmatch(r"[a-f0-9]{12}", directory.name):
                    continue
                job = read_json(directory / "job.json")
                if job:
                    if job["status"] in ACTIVE:
                        job.update(status="interrupted", error="服务重启；此运行未完成。")
                    self.jobs[directory.name] = job

    def health(self):
        missing = [m for m in ("numpy", "PIL", "mujoco", "onnxruntime", "bam") if importlib.util.find_spec(m) is None]
        assets = all((self.data / "upstream" / n).is_dir() for n in ("microduck_rl", "microduck-policies"))
        official = self.enabled and not missing and assets
        lock = read_json(REPO / "deploy/microduck/fastwam/model-files.lock.json")
        files = all((self.data / name).is_file() and (self.data / name).stat().st_size == spec["bytes"]
                    for name, spec in lock["files"].items())
        native = importlib.util.find_spec("torch") is not None
        return {"status": "ok", "service": "microduck-visualization", "official_available": official,
                "fastwam_available": bool(official and self.fastwam and native and files),
                "live_enabled": self.enabled, "model_files_present": files,
                "model_integrity": "SHA256 checked in worker before every FastWAM load",
                "missing_dependencies": missing, "sim_assets_present": assets,
                "message": ("实时运行已启用，点击开始后加载。" if official else
                            "录屏可直接观看。实时仿真需要启用运行器、安装仿真依赖并准备官方资产。"),
                "fastwam_message": ("启动时校验权重；空闲时不加载模型。" if self.fastwam and files else
                                    "FastWAM 权重需另行提供，并启用 GPU 运行器。"),
                "active_id": next((k for k, v in self.jobs.items() if v["status"] in ACTIVE), None)}

    def start(self, config):
        config = validate_run(config)
        with self.lock:
            if any(job["status"] in ACTIVE for job in self.jobs.values()):
                raise BusyError("已有实时仿真正在运行，请先停止或等待结束。")
            if not self.health()[config["policy"] + "_available"]:
                raise ValueError("所选运行器尚未就绪，请按部署文档准备环境、资产与模型。")
            identifier = uuid.uuid4().hex[:12]
            folder = self.root / identifier
            folder.mkdir(parents=True)
            job = dict(id=identifier, status="starting", config=config, created_unix=time.time())
            atomic_json(folder / "job.json", job)
            atomic_json(folder / "request.json", config)
            command = [sys.executable, str(REPO / "scripts/microduck/fastwam_visual_worker.py"),
                       "--data", str(self.data), "--output", str(folder)]
            env = os.environ.copy()
            env["MICRODUCK_ASSET_ROOT"] = str(self.data / "upstream")
            env["PYTHONPATH"] = os.pathsep.join((str(REPO), str(REPO / "embodied/microduck/src"),
                                               str(REPO / "third_party/bam"), env.get("PYTHONPATH", "")))
            with (folder / "worker.log").open("wb") as log:
                process = subprocess.Popen(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True)
            self.jobs[identifier], self.processes[identifier] = job, process
            threading.Thread(target=self._watch, args=(identifier, process), daemon=True).start()
            return self.get(identifier)

    def _watch(self, identifier, process):
        try:
            code = process.wait(timeout=600)
        except subprocess.TimeoutExpired:
            # A driver/model hang must not keep an unattended UI job forever.
            with self.lock:
                self.jobs[identifier]["error"] = "运行超过 10 分钟墙钟上限，已终止。"
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            self._terminate_after_grace(process)
            code = process.wait()
        with self.lock:
            job = self.jobs[identifier]
            folder = self.root / identifier
            outcome = read_json(folder / "outcome.json", {})
            job.update(status="cancelled" if job.get("stop_requested") else
                       "complete" if code == 0 and outcome.get("status") == "complete" else "failed",
                       exit_code=code, finished_unix=time.time())
            if job["status"] == "failed":
                job["error"] = outcome.get("error", job.get("error", "仿真进程退出；请查看运行日志。"))
            atomic_json(folder / "job.json", job)
            self.processes.pop(identifier, None)

    def get(self, identifier=None):
        with self.lock:
            if identifier is None:
                identifier = next(reversed(self.jobs), None)
                if identifier is None:
                    return {"status": "idle"}
            job = dict(self.jobs[identifier])
            folder = self.root / identifier
            stage = read_json(folder / "stage.json", {})
            if job["status"] in ACTIVE and not job.get("stop_requested"):
                job["status"] = stage.get("status", job["status"])
            job["stage_message"] = stage.get("message", "")
            job["live"] = read_json(folder / "live.json", {})
            report = read_json(folder / "outcome.json", {}).get("report")
            if report:
                job["report"] = report
            job["frames"] = {name: f"/api/live/{identifier}/frame/{name}" for name in ("observer", "camera")
                             if (folder / f"{name}.jpg").is_file()}
            job["updated_unix"] = time.time()
            job["report_url"] = f"/api/live/{identifier}/report"
            job["log_url"] = f"/api/live/{identifier}/log"
            return job

    def file(self, identifier, name):
        with self.lock:
            if identifier not in self.jobs:
                raise KeyError(identifier)
        return self.root / identifier / name

    def stop(self, identifier):
        with self.lock:
            job = self.jobs[identifier]
            if job["status"] not in ACTIVE:
                return self.get(identifier)
            job.update(status="cancelling", stop_requested=True)
            folder = self.root / identifier
            (folder / "cancel.request").touch()
            for episode in folder.glob("episode*"):
                if episode.is_dir():
                    (episode / "cancel.request").touch()
            process = self.processes.get(identifier)
            if process:
                threading.Thread(target=self._terminate_after_grace, args=(process,), daemon=True).start()
            return self.get(identifier)

    @staticmethod
    def _terminate_after_grace(process):
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def close(self):
        with self.lock:
            processes = list(self.processes.values())
        for process in processes:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for process in processes:
            self._terminate_after_grace(process)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, message, *args):
        if self.command == "POST" or len(args) < 2 or str(args[1]) not in ("200", "206", "304"):
            super().log_message(message, *args)

    def reply(self, value, status=200, content_type="application/json; charset=utf-8"):
        raw = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(raw)

    def serve_file(self, path, *, ranged=False):
        size = path.stat().st_size
        start, end, status = 0, size - 1, 200
        if ranged and self.headers.get("Range"):
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", self.headers["Range"])
            if match and any(match.groups()):
                left, right = match.groups()
                if left:
                    start, end = int(left), min(int(right), size - 1) if right else size - 1
                elif int(right) > 0:
                    start = max(0, size - int(right))
                else:
                    start = size
            else:
                start = size
            if start >= size or start > end:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            status = 206
        self.send_response(status)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        if ranged:
            self.send_header("Accept-Ranges", "bytes")
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if self.command != "HEAD":
            with path.open("rb") as stream:
                stream.seek(start)
                remaining = end - start + 1
                while remaining:
                    raw = stream.read(min(remaining, 65536))
                    if not raw:
                        break
                    self.wfile.write(raw)
                    remaining -= len(raw)

    def proxy(self, path, payload=None):
        try:
            req = Request(self.server.worker_url + path,
                          data=None if payload is None else json.dumps(payload).encode(),
                          headers={"Content-Type": "application/json"})
            with urlopen(req, timeout=5) as response:
                return self.reply(response.read(4 * 1024 * 1024), response.status,
                                  response.headers.get("Content-Type", "application/json"))
        except HTTPError as exc:
            return self.reply(exc.read(65536), exc.code)
        except (URLError, TimeoutError, OSError):
            if path == "/api/health":
                return self.reply({"status": "worker_unavailable", "official_available": False,
                                   "fastwam_available": False, "message": "实时运行器暂时离线，录屏仍可观看。"})
            return self.reply({"error": "实时运行器连接中断；请检查后端服务。"}, 503)

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        path = urlsplit(self.path).path
        try:
            if self.server.worker_url and (path == "/api/health" or path == "/api/live" or re.fullmatch(
                    r"/api/live/[a-f0-9]{12}(?:/frame/(?:observer|camera)|/report|/log)?", path)):
                return self.proxy(path)
            if path == "/api/health":
                return self.reply(self.server.jobs.health())
            if path == "/api/catalog":
                return self.reply(self.server.catalog)
            if path == "/api/live":
                return self.reply(self.server.jobs.get())
            match = re.fullmatch(r"/api/live/([a-f0-9]{12})(?:/(frame/(observer|camera)|report|log))?", path)
            if match:
                identifier, suffix, camera = match.groups()
                if camera:
                    return self.serve_file(self.server.jobs.file(identifier, camera + ".jpg"))
                if suffix == "report":
                    return self.serve_file(self.server.jobs.file(identifier, "outcome.json"))
                if suffix == "log":
                    file = self.server.jobs.file(identifier, "worker.log")
                    with file.open("rb") as stream:
                        stream.seek(max(0, file.stat().st_size - 65536))
                        return self.reply(stream.read(), content_type="text/plain; charset=utf-8")
                return self.reply(self.server.jobs.get(identifier))
            if path.startswith("/media/") and path[7:] in self.server.catalog["files"]:
                return self.serve_file(self.server.media / path[7:], ranged=True)
            if path in ("/", "/live", "/index.html"):
                return self.serve_file(STATIC / "index.html")
            if path in ("/app.js", "/style.css"):
                return self.serve_file(STATIC / path[1:])
            return self.reply({"error": "not found"}, 404)
        except (FileNotFoundError, KeyError):
            self.reply({"error": "not found"}, 404)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self):
        path = urlsplit(self.path).path
        try:
            origin = self.headers.get("Origin")
            if origin and (urlsplit(origin).netloc != self.headers.get("Host") or urlsplit(origin).scheme not in ("http", "https")):
                return self.reply({"error": "cross-origin write rejected"}, 403)
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 4096 or self.headers.get_content_type() != "application/json":
                return self.reply({"error": "expected a small application/json request"}, 400)
            payload = json.loads(self.rfile.read(length))
            stop_match = re.fullmatch(r"/api/live/([a-f0-9]{12})/stop", path)
            if path != "/api/live" and not stop_match:
                return self.reply({"error": "not found"}, 404)
            if self.server.worker_url:
                return self.proxy(path, payload)
            if stop_match:
                return self.reply(self.server.jobs.stop(stop_match[1]))
            return self.reply(self.server.jobs.start(payload), 202)
        except BusyError as exc:
            self.reply({"error": str(exc)}, 409)
        except (ValueError, TypeError) as exc:
            self.reply({"error": str(exc)}, 400)
        except KeyError:
            self.reply({"error": "unknown run"}, 404)
        except (BrokenPipeError, ConnectionResetError):
            pass


def make_server(host, port, data, *, enable_live=False, enable_fastwam=False, media=MEDIA, worker_url=None):
    catalog = load_catalog(media)
    if worker_url:
        parsed = urlsplit(worker_url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc or parsed.path not in ("", "/"):
            raise ValueError("worker-url must be an HTTP(S) service origin")
    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    server.catalog, server.media = catalog, Path(media)
    server.worker_url = worker_url.rstrip("/") if worker_url else None
    server.jobs = LiveJobs(data, enable_live=enable_live, enable_fastwam=enable_fastwam)
    return server
