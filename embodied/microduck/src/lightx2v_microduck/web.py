"""Local experiment UI. Runs real CLI evaluations and displays their records."""

from __future__ import annotations

import io
import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image

from .web_catalog import experiment_catalog, record_category, validate_launch

WEB_RESOURCES = Path(__file__).parent / "resources" / "web"
PAGE = (WEB_RESOURCES / "index.html").read_text(encoding="utf-8")


class ExperimentServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, runs, host="127.0.0.1", port=8767, training_dir=None):
        self.runs = Path(runs).resolve()
        self.runs.mkdir(parents=True, exist_ok=True)
        self.jobs = {}
        self.jobs_lock = threading.Lock()
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.frame_lock = threading.Lock()
        self.frame_cache = None
        from .training_dashboard import default_directory
        self.training_dir = Path(training_dir) if training_dir else default_directory()
        from .school.service import SchoolService
        self.school = SchoolService(self.runs / "_school")
        super().__init__((host, port), Handler)
        self.fastwam_sync_stop = None
        if os.environ.get('MICRODUCK_FASTWAM_SYNC') == '1':
            from .fastwam_dashboard import start_sync
            self.fastwam_sync_stop = start_sync()

    def safe_path(self, value):
        path = (self.runs / value).resolve()
        if not path.is_relative_to(self.runs):
            raise ValueError("path outside experiment directory")
        return path

    def launch(self, payload):
        payload = validate_launch(payload)
        mode = payload["mode"]
        seed = payload["seed"]
        identifier = uuid.uuid4().hex[:12]
        output = self.runs / identifier
        args = [
            sys.executable,
            "-m",
            "lightx2v_microduck.cli",
            "realtime" if mode == "robotd_bam" else "live" if mode == "live" else "eval",
            "--task",
            payload["task"],
            "--scene",
            payload["scene"],
            "--policy",
            payload["policy"],
            "--output",
            str(output),
        ]
        if mode == "robotd_bam":
            args += ["--seed", str(seed), "--seconds", "30"]
        elif mode == "live":
            args += ["--seed", str(seed)]
        else:
            args += ["--seed-start", str(seed), "--seeds", "1"]
        for key in ("model_path", "backbone_path", "device", "precision", "endpoint", "planner_model", "planner_context", "initial_conditions"):
            if payload.get(key):
                args += ["--" + key.replace("_", "-"), str(payload[key])]
        with self.jobs_lock:
            if sum(j["status"] in ("queued", "running") for j in self.jobs.values()) >= 4:
                raise ValueError("four experiments are already queued; wait for completion")
            self.jobs[identifier] = {"id": identifier, "status": "queued", "mode": mode, "task": payload["task"], "scene": payload["scene"], "policy": payload["policy"]}

        def run():
            try:
                if self.jobs[identifier].get("cancel_requested"):
                    self.jobs[identifier]["status"] = "cancelled"
                    return
                self.jobs[identifier]["status"] = "running"
                output.mkdir(exist_ok=True)
                from .data import write_json
                write_json(output / "launch.json", payload)
                with (output / "process.log").open("w") as log:
                    if mode == "live":
                        process = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT)
                        cancelled_at = None
                        while process.poll() is None:
                            if self.jobs[identifier].get("cancel_requested"):
                                (output / "cancel.request").touch()
                                cancelled_at = cancelled_at or time.monotonic()
                                if time.monotonic() - cancelled_at > 3:
                                    process.terminate()
                                if time.monotonic() - cancelled_at > 5:
                                    process.kill()
                            time.sleep(0.05)
                        returncode = process.returncode
                    else:
                        returncode = subprocess.run(args, stdout=log, stderr=subprocess.STDOUT).returncode
                cancelled = self.jobs[identifier].get("cancel_requested", False)
                self.jobs[identifier]["status"] = "cancelled" if cancelled else "complete" if returncode == 0 else "failed"
                if returncode and not cancelled:
                    self.jobs[identifier]["error"] = (output / "process.log").read_text()[-2000:]
            except Exception as exc:
                self.jobs[identifier].update(status="failed", error=f"{type(exc).__name__}: {exc}")

        self.pool.submit(run)
        return self.jobs[identifier]

    def cancel(self, identifier):
        if not isinstance(identifier, str):
            raise ValueError("expected a job id")
        with self.jobs_lock:
            job = self.jobs[identifier]
            if job["mode"] != "live":
                raise ValueError("stop is available for live simulations")
            if job["status"] in ("queued", "running"):
                packet = self.runs / identifier / "live.json"
                if packet.is_file() and json.loads(packet.read_text()).get("status") in ("complete", "cancelled"):
                    return dict(job)
                job["cancel_requested"] = True
            return dict(job)

    def server_close(self):
        if getattr(self, 'fastwam_sync_stop', None) is not None:
            self.fastwam_sync_stop.set()
        try:
            super().server_close()
        finally:
            self.pool.shutdown(wait=True)


class Handler(BaseHTTPRequestHandler):
    def reply(self, value, status=200, content_type="application/json", headers=None):
        data = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False).encode()
        try:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            for name, value in (headers or {}).items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            # Refreshing a live viewer closes in-flight image requests.
            self.close_connection = True

    def video_reply(self, data, content_type):
        """Serve a verified recording with single-byte-range seeking support."""
        headers = {"Accept-Ranges": "bytes"}
        requested = self.headers.get("Range", "")
        # Unsupported/malformed ranges and unverifiable If-Range conditions
        # receive the complete representation. Multipart ranges are not needed.
        match = re.fullmatch(r"bytes=(\d*)-(\d*)", requested.strip()) if len(requested) <= 128 else None
        if not match or not any(match.groups()) or self.headers.get("If-Range"):
            return self.reply(data, content_type=content_type, headers=headers)
        first, last = match.groups()
        length = len(data)
        start = int(first) if first else max(0, length - int(last))
        end = min(int(last), length - 1) if first and last else length - 1
        if start >= length or end < start:
            return self.reply(b"", status=416, content_type=content_type,
                              headers={**headers, "Content-Range": f"bytes */{length}"})
        return self.reply(data[start:end + 1], status=206, content_type=content_type,
                          headers={**headers, "Content-Range": f"bytes {start}-{end}/{length}"})

    def do_GET(self):
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        try:
            if parsed.path in ('/fastwam-capabilities.js', '/fastwam-capabilities.css'):
                kind = 'text/javascript' if parsed.path.endswith('.js') else 'text/css'
                return self.reply((WEB_RESOURCES / parsed.path[1:]).read_bytes(), content_type=kind+'; charset=utf-8')
            if parsed.path == '/api/fastwam/capabilities':
                from .fastwam_showcase import manifest
                return self.reply(manifest())
            if parsed.path == '/api/fastwam/media':
                from .fastwam_showcase import artifact
                data, kind = artifact(query.get('id', [''])[0], query.get('kind', ['video'])[0])
                if kind == 'video/mp4':
                    return self.video_reply(data, kind)
                return self.reply(data, content_type=kind)
            if parsed.path in ("/training", "/training.js", "/fastwam-training.js"):
                name = ("tinyvla-training.html" if query.get('model') == ['tinyvla'] else "training.html") if parsed.path == "/training" else parsed.path[1:]
                return self.reply((WEB_RESOURCES / name).read_bytes(), content_type="text/html; charset=utf-8" if name.endswith("html") else "text/javascript; charset=utf-8")
            if parsed.path == '/api/training/fastwam':
                from .fastwam_dashboard import dashboard as fastwam_dashboard
                return self.reply(fastwam_dashboard())
            if parsed.path == "/api/training":
                from .training_dashboard import dashboard
                return self.reply(dashboard(self.server.training_dir, query.get("run", [None])[0]))
            if parsed.path == "/api/training/artifact":
                from .training_dashboard import evidence
                data, content_type = evidence(self.server.training_dir, query["id"][0], query.get("sha256", [None])[0])
                if content_type.startswith("video/"):
                    return self.video_reply(data, content_type)
                return self.reply(data, content_type=content_type)
            if parsed.path in ("/school", "/school.js"):
                name = "school.html" if parsed.path == "/school" else "school.js"
                return self.reply((WEB_RESOURCES / name).read_bytes(), content_type="text/html; charset=utf-8" if name.endswith("html") else "text/javascript; charset=utf-8")
            if parsed.path == "/api/school/catalog":
                from .school.courses import catalog
                return self.reply(catalog())
            if parsed.path == "/api/school/jobs":
                return self.reply(self.server.school.listing())
            if parsed.path == "/api/school/job":
                return self.reply(self.server.school.job(query["id"][0]))
            if parsed.path == "/api/school/live":
                return self.reply(self.server.school.live(query["id"][0]))
            if parsed.path in ("/", "/experiments", "/showcase.js", "/showcase.css"):
                name = {"/": "index.html", "/experiments": "experiments.html"}.get(parsed.path, parsed.path[1:])
                kind = "text/html" if name.endswith(".html") else "text/css" if name.endswith(".css") else "text/javascript"
                return self.reply((WEB_RESOURCES / name).read_bytes(), content_type=kind + "; charset=utf-8")
            if parsed.path == "/api/showcase":
                from .showcase import gallery
                return self.reply(gallery(self.server.training_dir))
            if parsed.path == "/health":
                return self.reply({"status": "ok", "service": "microduck-simulator"})
            if parsed.path == "/app.js":
                return self.reply((WEB_RESOURCES / "app.js").read_bytes(), content_type="text/javascript; charset=utf-8")
            if parsed.path == "/api/catalog":
                return self.reply(experiment_catalog())
            if parsed.path == "/api/tasks":
                from .evaluation import suite_tasks

                return self.reply([*suite_tasks("skills13"), *suite_tasks("visual_goal"), "community_follow", "ball_apex"])
            if parsed.path == "/api/episodes":
                rows = []
                paths = [*self.server.runs.rglob("episode.json"), *self.server.runs.rglob("realtime.json")]
                for path in sorted(paths, key=lambda item: item.stat().st_mtime_ns, reverse=True):
                    r = json.loads(path.read_text())
                    rows.append(
                        {
                            "path": str(path.relative_to(self.server.runs)),
                            "task": r.get("task", "realtime"),
                            "seed": r["seed"],
                            "policy": r.get("policy", {}).get("policy_id", "legacy-unrecorded"),
                            "has_camera": (path.parent / "observation.jpg").is_file(),
                            "has_observer": (path.parent / "observer.jpg").is_file(),
                            "mode": r.get("mode", "offline"),
                            "scene": r.get("scene", "unknown"),
                            "category": record_category(r),
                            "execution_scope": "project_tasks_v2" if r.get("policy", {}).get("metadata", {}).get("acceptance_scope") == "project_tasks_v2" or r.get("final", {}).get("acceptance_scope") == "project_tasks_v2" or r.get("task") == "ground_lower" else "original",
                            "recorded_at": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
                            "success": r.get("success", r.get("strict_success")),
                            "termination_reason": r.get("termination_reason", r.get("fault")),
                            "duration_ticks": r.get("final", {}).get("tick"),
                        }
                    )
                return self.reply(rows)
            if parsed.path == "/api/job":
                return self.reply(self.server.jobs[query["id"][0]])
            if parsed.path == "/api/live":
                identifier = query["id"][0]
                job = self.server.jobs[identifier]
                if job["mode"] != "live":
                    raise ValueError("expected a live simulation")
                path = self.server.runs / identifier / "live.json"
                packet = json.loads(path.read_text()) if path.is_file() else {"frames": {}, "status": "initializing"}
                return self.reply({**packet, "job": dict(job)})
            if parsed.path in ("/api/episode", "/api/frame"):
                path = self.server.safe_path(query["path"][0])
                if path.name not in ("episode.json", "realtime.json"):
                    raise ValueError("expected an episode record")
                if parsed.path == "/api/episode":
                    record = json.loads(path.read_text())
                    if path.name == "realtime.json":
                        record.update(
                            success=record["strict_success"],
                            termination_reason=record.get("fault") or record.get("final", {}).get("termination_reason") or "realtime_duration_complete",
                            inference_ms={"p95": record.get("inference_p95_ms")},
                            profile="robotd_bam",
                        )
                        record.setdefault("policy", {"policy_id": "legacy-unrecorded"})
                    return self.reply(record)
                with self.server.frame_lock:
                    view = query.get("view", ["camera"])[0]
                    if view not in ("camera", "observer"):
                        raise ValueError("view must be camera or observer")
                    cached = self.server.frame_cache
                    if cached is None or cached[0] != (path, view):
                        realtime = path.name == "realtime.json"
                        with np.load(path.parent / ("observations.npz" if realtime else "trajectory.npz"), allow_pickle=False) as trajectory:
                            frame_key = "observer_rgb" if view == "observer" else "rgb"
                            tick_key = "observer_frame_tick" if view == "observer" else "tick" if realtime else "frame_tick"
                            cached = ((path, view), trajectory[frame_key], trajectory[tick_key])
                        self.server.frame_cache = cached
                    if not len(cached[1]):
                        raise ValueError("episode was recorded without camera frames")
                    requested = int(query.get("tick", [0])[0])
                    index = int(np.argmin(np.abs(cached[2] - requested)))
                    output = io.BytesIO()
                    Image.fromarray(cached[1][index]).save(output, format="JPEG")
                return self.reply(output.getvalue(), content_type="image/jpeg")
            return self.reply({"error": "not found"}, 404)
        except (OSError, ValueError, KeyError) as exc:
            self.reply({"error": str(exc)}, 400)

    def do_POST(self):
        try:
            if self.path not in ("/api/run", "/api/cancel", "/api/school/preview", "/api/school/run", "/api/school/cancel"):
                return self.reply({"error": "not found"}, 404)
            # Local UI only. Cross-site form/fetch requests cannot create jobs.
            origin = self.headers.get("Origin")
            if origin and urlparse(origin).netloc != self.headers.get("Host"):
                raise ValueError("cross-origin requests refused")
            if not self.headers.get("Content-Type", "").startswith("application/json"):
                raise ValueError("application/json required")
            size = int(self.headers.get("Content-Length", 0))
            if not 0 < size <= 16384:
                raise ValueError("invalid request size")
            payload = json.loads(self.rfile.read(size))
            if not isinstance(payload, dict):
                raise ValueError("expected a JSON object")
            if self.path == "/api/school/preview":
                from .school.courses import parse_lesson
                return self.reply(parse_lesson(payload.get("instruction")))
            if self.path == "/api/school/run":
                return self.reply(self.server.school.launch(payload), 202)
            if self.path == "/api/school/cancel":
                return self.reply(self.server.school.cancel(payload["id"]), 202)
            result = self.server.cancel(payload["id"]) if self.path == "/api/cancel" else self.server.launch(payload)
            return self.reply(result, 202)
        except (ValueError, KeyError, OSError) as exc:
            self.reply({"error": str(exc)}, 400)

    def log_message(self, format, *args):
        return


def serve(runs, host="127.0.0.1", port=8767):
    server = ExperimentServer(runs, host, port)
    print(f"MicroDuck experiments: http://{host}:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
