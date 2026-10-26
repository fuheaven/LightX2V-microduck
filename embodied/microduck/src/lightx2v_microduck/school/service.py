"""Durable, single-worker School jobs; survives UI/web-server reconnects."""

import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from .courses import Course, parse_lesson
from .learning import load_artifact, write


class SchoolService:
    def __init__(self, root):
        self.root = Path(root)
        self.lock = threading.Lock()
        self.children = {}

    def path(self, identifier):
        if not isinstance(identifier, str) or not re.fullmatch(r"[0-9a-f]{12}", identifier):
            raise ValueError("invalid School id")
        path = (self.root / identifier).resolve()
        if not path.is_relative_to(self.root.resolve()):
            raise ValueError("invalid School directory")
        return path

    def job(self, identifier):
        folder = self.path(identifier)
        job = json.loads((folder / "job.json").read_text())
        child = self.children.get(identifier)
        if child is not None:
            child.poll()  # Reap finished child processes.
        if job["status"] in ("queued", "running"):
            pid_file = folder / "pid"
            alive = False
            if pid_file.is_file():
                try:
                    command = Path(f"/proc/{int(pid_file.read_text())}/cmdline").read_bytes().split(b"\0")
                    alive = b"lightx2v_microduck.school" in command and str(folder / "request.json").encode() in command
                except (OSError, ValueError):
                    pass
            # Startup is asynchronous; give the child time to write its PID.
            if not alive and time.time() - job["created_at"] > 5:
                job.update(status="interrupted", error="worker exited before completing; saved checkpoint can be resumed")
                write(folder / "job.json", job)
        for filename, key in (("progress.json", "progress"), ("evaluation.json", "evaluation")):
            path = folder / filename
            if path.is_file():
                job[key] = json.loads(path.read_text())
        if (folder / "skill.json").is_file():
            artifact = load_artifact(folder / "skill.json")
            job["skill"] = {k: artifact.get(k) for k in ("course", "qualification", "selected_source", "status", "seed", "parent_sha256")}
        return job

    def listing(self):
        result = []
        if not self.root.exists():
            return result
        for folder in sorted(self.root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if (folder / "job.json").is_file():
                job = self.job(folder.name)
                result.append({key: value for key, value in job.items() if key not in ("progress", "evaluation")})
        return result

    def launch(self, payload):
        mode = payload.get("mode", "train")
        if mode not in ("train", "resume", "evaluate", "live"):
            raise ValueError("invalid School mode")
        seed, budget = payload.get("seed", 0), payload.get("budget_s", 600)
        if type(seed) is not int or not 0 <= seed < 1000000 or type(budget) is not int or not 10 <= budget <= 1800:
            raise ValueError("seed must be 0..999999; budget must be 10..1800 seconds")
        parent = None
        if mode != "train":
            parent_folder = self.path(payload.get("skill_id"))
            parent = load_artifact(parent_folder / "skill.json")
            if self.job(parent_folder.name)["status"] in ("queued", "running"):
                raise ValueError("wait until the source skill is saved")
            course = Course.from_dict(parent["course"])
            instruction = parent_folder.joinpath("request.json")
            instruction = json.loads(instruction.read_text()).get("instruction", course.family)
            if mode == "resume":
                seed = parent["seed"]
        else:
            parsed = parse_lesson(payload.get("instruction"))
            course, instruction = Course.from_dict(parsed["course"]), parsed["instruction"]
        with self.lock:
            if any(j["status"] in ("queued", "running") for j in self.listing()):
                raise ValueError("鸭子学校已有任务运行；请先停止或等待完成")
            identifier = uuid.uuid4().hex[:12]
            folder = self.path(identifier)
            folder.mkdir(parents=True)
            request = {"mode": mode, "seed": seed, "budget_s": budget, "course": course.to_dict(), "instruction": instruction, "artifact": str(parent_folder / "skill.json") if parent else None}
            write(folder / "request.json", request)
            job = {
                "id": identifier,
                "mode": mode,
                "status": "queued",
                "created_at": time.time(),
                "instruction": instruction,
                "course": course.to_dict(),
                "seed": seed,
                "budget_s": budget,
                "parent_id": payload.get("skill_id"),
            }
            write(folder / "job.json", job)
            try:
                with (folder / "process.log").open("w") as log:
                    process = subprocess.Popen(
                        [sys.executable, "-m", "lightx2v_microduck.school", "worker", str(folder / "request.json")], stdout=log, stderr=subprocess.STDOUT, start_new_session=True
                    )
                    self.children[identifier] = process
            except OSError as exc:
                job.update(status="failed", error=str(exc))
                write(folder / "job.json", job)
                raise ValueError(str(exc)) from exc
            return job

    def cancel(self, identifier):
        job = self.job(identifier)
        if job["status"] in ("queued", "running"):
            (self.path(identifier) / "cancel.request").touch()
        return job

    def live(self, identifier):
        job = self.job(identifier)
        if job["mode"] != "live":
            raise ValueError("expected live School job")
        path = self.path(identifier) / "live.json"
        return {"job": job, **(json.loads(path.read_text()) if path.is_file() else {"status": "initializing", "frames": {}})}


def worker(request_path):
    import fcntl
    import traceback

    from .learning import execute_skill, train

    request_path = Path(request_path).resolve()
    folder = request_path.parent
    (folder / "pid").write_text(str(os.getpid()))
    job = json.loads((folder / "job.json").read_text())
    with (folder.parent / ".worker.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            job["status"] = "running"
            write(folder / "job.json", job)
            request = json.loads(request_path.read_text())
            if request["mode"] in ("train", "resume"):
                train(Course.from_dict(request["course"]), folder, seed=request["seed"], budget_s=request["budget_s"], population=4 if request["budget_s"] <= 60 else 8, resume=request["artifact"])
            else:
                execute_skill(request["artifact"], folder, seed=request["seed"], live=request["mode"] == "live", budget_s=request["budget_s"])
            job["status"] = "cancelled" if (folder / "cancel.request").exists() else "complete"
        except Exception as exc:  # noqa: BLE001 - worker boundary must persist unexpected failures
            traceback.print_exc()
            job.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        finally:
            job["finished_at"] = time.time()
            write(folder / "job.json", job)
