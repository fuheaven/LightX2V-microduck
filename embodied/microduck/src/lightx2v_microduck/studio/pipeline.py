"""Process entry point for one requested round; failed gates never trigger another round."""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import signal
import time
import traceback
import numpy as np
from .schema import OBSERVATION, ACTION, digest, file_hash, write_json
from .simulation import cases, run_episode
from .evidence import summarize


class GateBlocked(RuntimeError):
    pass


class Job:
    def __init__(self, root, run_id):
        self.root = Path(root).resolve()
        self.id = run_id
        self.out = self.root / "runs" / run_id
        self.snapshot = json.loads((self.out / "snapshot.json").read_text())
        self.skill = self.snapshot["skill"]
        self.request = self.snapshot["request"]
        self.cancelled = False
        self.started = time.monotonic()
        self.latest_validation = {}
        signal.signal(signal.SIGTERM, lambda *_: setattr(self, "cancelled", True))
        signal.signal(signal.SIGINT, lambda *_: setattr(self, "cancelled", True))

    def stop(self):
        return self.cancelled or (self.out / "cancel").exists() or time.monotonic() - self.started > self.request["recipe"]["max_hours"] * 3600

    def progress(self, payload):
        media = payload.get("media", "")
        if "/validation/" in media:
            self.latest_validation = {"latest_validation_media": media, "latest_validation_model": Path(media).parent.parent.name}
        write_json(self.out / "progress.json", {**self.latest_validation, **payload, "elapsed_s": time.monotonic() - self.started})

    def episode(self, case, path, **kwargs):
        self.progress({"phase": "simulation", "case": {k: v for k, v in case.items() if k != "scene"}, "media": str(Path(path).relative_to(self.root))})
        return run_episode(self.skill, case, path, stop=self.stop, progress=lambda p: self.progress({**p, "media": str(Path(path).relative_to(self.root))}), **kwargs)

    def freeze_cases(self, name, rows):
        write_json(self.out / (name + "-cases.json"), rows)
        return digest(rows)

    def reference_path(self, run, subpath):
        if not run or any(c not in "0123456789abcdef" for c in run):
            raise ValueError("invalid source run")
        return self.root / "runs" / run / subpath

    def teacher(self):
        rows = []
        for group in self.skill["task"]["groups"]:
            rows.extend(cases(self.skill, "teacher", self.request["teacher_cases"], self.request["seed"], groups=[group]))
        case_hash = self.freeze_cases("teacher", rows)
        episodes = []
        for n, case in enumerate(rows):
            episodes.append(self.episode(case, self.out / "teacher" / f"{n:06d}"))
            report = summarize(episodes, teacher=True, groups=self.skill["task"]["groups"], case_sha=case_hash)
            report.update(task_sha256=digest(self.skill), teacher_source_sha256=self.snapshot["provenance"]["source_sha256"])
            write_json(self.out / "teacher-report.json", report)
            self.progress({"phase": "teacher", "completed": n + 1, "total": len(rows), "passed": report["passed"]})
        return report

    def teacher_report(self):
        path = self.reference_path(self.request["teacher_run"], "teacher-report.json") if self.request["teacher_run"] else self.out / "teacher-report.json"
        report = json.loads(path.read_text())
        if not report["teacher_qualified"] or report["task_sha256"] != digest(self.skill) or report["teacher_source_sha256"] != self.snapshot["provenance"]["source_sha256"]:
            raise GateBlocked("当前任务/场景/教师版本尚未达到每难度组100集、95%的教师门槛")
        return report

    def collect(self):
        teacher_report = self.teacher_report()
        from .dataset import audit_dataset

        data = self.out / "data"
        data.mkdir(exist_ok=True)
        count = self.request["episodes"]
        split_counts = {"train": max(1, round(count * 0.8)), "validation": max(1, round(count * 0.1)), "test": max(1, count - round(count * 0.8) - round(count * 0.1))}
        from collections import Counter

        rows = []
        quotas = Counter()
        collected = Counter()
        key = lambda c: (c["split"], c["group"], c["category"], c["color"])
        for split, n in split_counts.items():
            quotas.update(key(c) for c in cases(self.skill, split, n, self.request["seed"]))
            rows.extend(cases(self.skill, split, max(n * 2, n + 36), self.request["seed"]))
        self.freeze_cases("collection", rows)
        entries = []
        for n, case in enumerate(rows):
            if collected[key(case)] >= quotas[key(case)]:
                continue
            path = data / f"episode-{n:06d}"
            report = self.episode(case, path)
            eligible = report["success"] and not report["interventions"]
            if eligible:
                collected[key(case)] += 1
            if (path / "sensors.npz").exists():
                with np.load(path / "sensors.npz") as d:
                    frames = len(d["tick"])
                entries.append(
                    {
                        "id": path.name,
                        "path": str((path / "sensors.npz").relative_to(data)),
                        "sha256": file_hash(path / "sensors.npz"),
                        **{k: case[k] for k in ("category", "color", "group", "split", "layout_id", "instance_id", "instruction_family")},
                        "instruction": report["instruction"],
                        "eligible": eligible,
                        "frames": frames,
                    }
                )
            write_json(
                data / "index.json",
                {"schema": "studio_dataset_v1", "observation": OBSERVATION, "action": ACTION, "task_sha256": digest(self.skill), "teacher_report_sha256": digest(teacher_report), "episodes": entries},
            )
            self.progress({"phase": "collection", "completed": sum(collected.values()), "total": sum(quotas.values()), "attempts": len(entries), "eligible": sum(r["eligible"] for r in entries)})
        report = audit_dataset(data)
        if any(collected[k] < v for k, v in quotas.items()):
            raise GateBlocked("部分类别、颜色或难度组的合格示范不足；已保留数据，需要检查教师后重新提交采集")
        return report

    def model_path(self):
        version = self.snapshot.get("model_version")
        if not version:
            raise ValueError("select a frozen model version")
        path = (self.root / version["model_path"]).resolve()
        if self.root not in path.parents:
            raise ValueError("model outside workspace")
        if file_hash(path / "model.json") != version["manifest_sha256"]:
            raise ValueError("published model was modified")
        return path

    def runtime(self, path=None, variant=None):
        policy = self.request["policy"]
        if policy in ("teacher", "rule"):
            return None
        if policy.startswith("vlm_"):
            from .vlm import VLMRuntime

            return VLMRuntime(policy)
        from .models import SmallRuntime
        from .fastwam import FastWAMRuntime

        path = path or self.model_path()
        if policy == "lightweight":
            return SmallRuntime(path, os.environ.get("STUDIO_DEVICE", "cpu"))
        version = self.snapshot.get("model_version") or {}
        dependency_root = self.root / version["dependency_root"] if version.get("dependency_root") else Path(os.environ["STUDIO_MODEL_ROOT"])
        return FastWAMRuntime(path, dependency_root / "models/Wan2.2-TI2V-5B", variant or self.request["variant"])

    def evaluate(self, runtime=None, *, path=None, split="test", count=None, variant=None):
        rows = cases(self.skill, split, count or self.request["evaluation_cases"], self.request["seed"])
        manifest_hash = self.freeze_cases("evaluation", rows)
        episodes = []
        runtime = runtime or self.runtime(path, variant)
        model_sha = file_hash(Path(path or self.model_path()) / "model.pt") if self.request["policy"] in ("fastwam", "lightweight") and split == "test" else None
        start = time.monotonic()
        for n, case in enumerate(rows):
            folder = self.out / ("evaluation-" + (variant or self.request["variant"])) / f"{n:06d}"
            report = self.episode(case, folder, policy_kind=self.request["policy"], runtime=runtime, realtime=self.request["mode"] == "realtime", ttl_ms=self.request["max_action_age_ms"])
            report["media"] = str(folder.relative_to(self.root))
            episodes.append(report)
            result = summarize(episodes, groups=self.skill["task"]["groups"], mode=self.request["mode"], model_sha=model_sha, case_sha=manifest_hash)
            delays = [t for r in episodes for t in r.get("inference_ms", [])]
            result.update(
                p95_ms=float(np.percentile(delays, 95)) if delays else 0,
                execution=f"nav5:ttl{self.request['max_action_age_ms']}",
                resources={"wall_seconds": time.monotonic() - start, "gpu_seconds": time.monotonic() - start if os.environ.get("STUDIO_DEVICE") == "cuda" else 0},
                variant=variant or self.request["variant"],
                task_sha256=digest(self.skill["task"]),
                model_version=self.request["version"],
            )
            if hasattr(runtime, "usage"):
                result["resources"].update(
                    api_calls=len(runtime.usage),
                    input_tokens=sum(u.get("prompt_tokens", 0) for u in runtime.usage),
                    output_tokens=sum(u.get("completion_tokens", 0) for u in runtime.usage),
                    external_inference_unmetered=True,
                )
            write_json(self.out / ("evaluation-" + (variant or self.request["variant"]) + ".json"), result)
        return result

    def training(self):
        from .training import train

        data = self.reference_path(self.request["data_run"], "data") if self.request["data_run"] else self.out / "data"
        resume = None
        if self.request["resume_run"]:
            old = self.reference_path(self.request["resume_run"], "")
            if not self.request["data_run"]:
                data = old / "data"
            resume = old / "training/resume.pt"
            receipt = json.loads((old / "training/resume.json").read_text())
            if file_hash(resume) != receipt["sha256"]:
                raise ValueError("resume checkpoint mismatch")
        index = json.loads((data / "index.json").read_text())
        if index["task_sha256"] != digest(self.skill):
            raise GateBlocked("dataset belongs to a different task or scene version")
        model_root = Path(os.environ.get("STUDIO_MODEL_ROOT", "/data"))
        if self.request["recipe"]["backend"] == "fastwam":
            # Fresh transfer always uses the immutable published MicroDuck initialization.
            lock = json.loads((Path(__file__).resolve().parents[5] / "deploy/microduck/fastwam/model-files.lock.json").read_text())
            for name in ("models/microduck/model.pt", "models/microduck/dataset_stats.json"):
                if file_hash(model_root / name) != lock["files"][name]["sha256"]:
                    raise ValueError("FastWAM parent differs from the pinned publication")
        dev = cases(self.skill, "validation", max(9, len(self.skill["task"]["groups"]) * 9), self.request["seed"] + 10000)
        self.freeze_cases("development", dev)

        def validate(candidate, model, stats):
            if self.request["recipe"]["backend"] == "fastwam":
                from .fastwam import ReferenceRuntime

                runtime = ReferenceRuntime(model, stats)
            else:
                from .models import SmallRuntime

                runtime = SmallRuntime(candidate, os.environ.get("STUDIO_DEVICE", "cpu"))
            episodes = []
            for i, c in enumerate(dev):
                folder = candidate / "validation" / f"{i:05d}"
                e = self.episode(c, folder, policy_kind=self.request["recipe"]["backend"], runtime=runtime)
                e["media"] = str(folder.relative_to(self.root))
                episodes.append(e)
            return summarize(episodes, case_sha=digest(dev))

        result = train(
            data,
            self.out / "training",
            self.request["recipe"],
            device=os.environ.get("STUDIO_DEVICE", "cpu"),
            backbone=model_root / "models/Wan2.2-TI2V-5B",
            parent=model_root / "models/microduck",
            resume=resume,
            stop=self.stop,
            progress=self.progress,
            validate=validate,
        )
        result["candidate"] = str(Path(result["candidate"]).relative_to(self.root))
        return result

    def execute(self):
        req = self.request
        kind = req["kind"]
        if kind == "historical":
            return self.historical()
        if kind == "teacher":
            return self.teacher()
        if kind == "collect":
            return self.collect()
        if kind == "train":
            return self.training()
        if kind == "pipeline":
            if not req["resume_run"]:
                report = self.teacher()
                if not report["teacher_qualified"]:
                    raise GateBlocked("教师未通过，流水线停在教师检查；未启动采集或GPU训练")
                self.collect()
            result = self.training()
            req["policy"] = req["recipe"]["backend"]
            evaluation = self.evaluate(path=self.root / result["candidate"])
            return {**result, "evaluation": evaluation}
        if kind == "evaluate":
            return self.evaluate()
        if kind == "optimize":
            if req["policy"] != "fastwam":
                raise ValueError("LightX2V cache comparison applies to FastWAM")
            reports = []
            for variant in ("plain", "optimized"):
                runtime = self.runtime(variant=variant)
                reports.append(self.evaluate(runtime=runtime, variant=variant))
                del runtime
                import torch

                torch.cuda.empty_cache()
            from .evidence import compare

            return {"reports": reports, "comparison": compare(reports)}
        case = cases(self.skill, "validation", 1, req["seed"], groups=[req["group"]])[0]
        from .scene import sample_scene

        case.update(category=req["category"], color=req["color"])
        case["scene"] = (
            self.skill["scene"] if kind == "preview" else sample_scene(self.skill["scene"], req["seed"], group=req["group"], category=req["category"], color=req["color"], split="validation")
        )
        case["layout_id"] = digest(case["scene"])
        runtime = None if kind == "preview" else self.runtime()

        def control():
            path = self.out / "control.json"
            return json.loads(path.read_text()) if path.exists() else None

        return self.episode(
            case,
            self.out / "live",
            policy_kind=req["policy"],
            runtime=runtime,
            realtime=kind != "preview",
            ttl_ms=req["max_action_age_ms"],
            preview=kind == "preview",
            control=control if kind == "correction" else None,
        )

    def historical(self):
        """Keep the published 48D runtime isolated from new 176D training contracts."""
        import sys, threading
        from types import SimpleNamespace

        scripts = Path(__file__).resolve().parents[5] / "scripts/microduck"
        sys.path.insert(0, str(scripts))
        from fastwam_visual_worker import run

        out = self.out / "historical"
        out.mkdir(exist_ok=True)
        write_json(
            out / "request.json",
            {
                "policy": "fastwam",
                "variant": self.request["variant"],
                "color": self.request["color"],
                "scene": "room_clear" if self.request["group"] == "clear" else "room_curriculum_one",
                "seed": self.request["seed"],
                "seconds": 45,
            },
        )
        done = threading.Event()

        def relay():
            while not done.wait(0.3):
                if self.stop():
                    (out / "cancel.request").touch()
                path = out / "live.json"
                if path.exists():
                    packet = json.loads(path.read_text())
                    self.progress(
                        {
                            "phase": "historical_realtime",
                            "media": str(out.relative_to(self.root)),
                            "head_file": "camera.jpg",
                            "tick": packet["tick"],
                            "sim_s": packet["sim_s"],
                            "rtf": packet["real_time_factor"],
                            "control_source": packet["execution"].get("execution_source", packet["execution"].get("source", "fastwam")),
                        }
                    )
                elif (out / "stage.json").exists():
                    self.progress(json.loads((out / "stage.json").read_text()))

        watcher = threading.Thread(target=relay, daemon=True)
        watcher.start()
        try:
            run(SimpleNamespace(output=out, data=Path(os.environ["STUDIO_MODEL_ROOT"])))
            result = json.loads((out / "outcome.json").read_text())["report"]
            result.update(historical=True, observation="history_rgb+proprio48", formal_acceptance=False)
            return result
        finally:
            done.set()
            watcher.join(timeout=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    job = Job(args.workspace, args.run)
    try:
        result = job.execute()
        status = "complete"
    except GateBlocked as e:
        result = {"reason": str(e)}
        status = "blocked"
    except InterruptedError as e:
        deadline = time.monotonic() - job.started > job.request["recipe"]["max_hours"] * 3600
        result = {"reason": "per-run time limit reached" if deadline else str(e), "resumable": (job.out / "training/resume.json").exists()}
        status = "cancelled"
    except Exception as e:
        traceback.print_exc()
        result = {"reason": f"{type(e).__name__}: {e}"}
        status = "failed"
    result["wall_seconds"] = time.monotonic() - job.started
    result["gpu_seconds"] = result["wall_seconds"] if os.environ.get("STUDIO_DEVICE") == "cuda" else 0
    write_json(job.out / "result.json", {"status": status, "result": result})


if __name__ == "__main__":
    main()
