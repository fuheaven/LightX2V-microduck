"""Small control plane. Heavy work is leased to independent workers."""

import argparse
import json
import os
from pathlib import Path
import secrets
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from .schema import SkillDraft, RunRequest, recipe_for, source_identity, digest, file_hash, write_json, SceneSpec
from .store import Store

REPO = Path(__file__).resolve().parents[5]


def create_app(workspace=None):
    store = Store(workspace or os.environ.get("STUDIO_WORKSPACE", "artifacts/studio"))

    @asynccontextmanager
    async def lifespan(app):
        yield
        await run_in_threadpool(store.close)

    app = FastAPI(title="MicroDuck Studio", version="1.0.0", lifespan=lifespan)
    app.state.store = store
    worker_token = os.environ.get("STUDIO_WORKER_TOKEN")
    app_token = os.environ.get("STUDIO_TOKEN")
    provenance = {
        "source_sha256": source_identity(REPO),
        "worker_image": os.environ.get("STUDIO_WORKER_IMAGE", "local-development"),
        "video_image": os.environ.get("STUDIO_VIDEO_IMAGE", "microduck-studio-video:v1"),
    }
    from .scene_api import install as install_scene_api

    install_scene_api(app, store, provenance)
    from .video_api import install as install_video_api

    install_video_api(app, store)

    @app.middleware("http")
    async def security(request, call_next):
        # Same-origin mutations prevent other websites submitting LAN training jobs.
        if request.method not in ("GET", "HEAD", "OPTIONS"):
            origin = request.headers.get("origin")
            if origin and origin.rstrip("/") != str(request.base_url).rstrip("/"):
                return JSONResponse({"detail": "cross-origin mutation rejected"}, 403)
        required = worker_token if request.url.path.startswith("/api/worker/") else app_token
        if request.url.path.startswith("/api/worker/") and not worker_token:
            return JSONResponse({"detail": "worker token is not configured"}, 503)
        protected = request.url.path.startswith(("/api/", "/media/")) and request.url.path != "/api/session"
        if protected and required and request.headers.get("authorization") != "Bearer " + required and request.cookies.get("studio_token") != required:
            return JSONResponse({"detail": "authentication required"}, 401)
        return await call_next(request)

    @app.post("/api/session")
    async def login(request: Request):
        body = await request.json()
        if app_token and not secrets.compare_digest(str(body.get("token", "")), app_token):
            raise HTTPException(401, "访问令牌不正确")
        response = JSONResponse({"ok": True})
        if app_token:
            response.set_cookie("studio_token", app_token, httponly=True, samesite="strict", secure=request.url.scheme == "https", max_age=8 * 3600)
        return response

    @app.exception_handler(ValueError)
    async def bad_value(request, exc):
        return JSONResponse({"detail": str(exc)}, 409)

    @app.exception_handler(KeyError)
    async def missing(request, exc):
        return JSONResponse({"detail": str(exc)}, 404)

    @app.get("/api/health")
    def health():
        from .store import sqlite3

        return {"status": "ok", "product": "MicroDuck Studio", "workers": store.workers(), "provenance": provenance, "sqlite_version": sqlite3.sqlite_version, "hardware_validated": False}

    @app.get("/api/bootstrap")
    def bootstrap():
        from ..visualization import load_catalog

        return {
            "skills": [s for s in store.skills() if s["id"] != "history_fastwam"],
            "versions": store.versions(),
            "workers": store.workers(),
            "recipes": {b: recipe_for(b).model_dump() for b in ("fastwam", "lightweight")},
            "template": SkillDraft().model_dump(),
            "history": load_catalog(),
            "scope": "simulation_only",
        }

    @app.get("/api/skills")
    def skills():
        return store.skills()

    @app.post("/api/history/runs", status_code=202)
    def history_run(body: RunRequest):
        if body.category != "ball":
            raise ValueError("historical 48D model supports colored-ball tasks only")
        if body.group not in ("clear", "obstacle"):
            raise ValueError("historical demo supports clear or single-obstacle scenes")
        if not store.skill("history_fastwam"):
            skill = SkillDraft().model_dump()
            skill["name"] = "历史 FastWAM · 彩色球"
            skill["task"].update(categories=["ball"], groups=["clear", "obstacle"])
            store.save_skill(skill, "history_fastwam")
        req = body.model_dump()
        req.update(kind="historical", policy="fastwam", mode="realtime", version=None)
        return store.submit("history_fastwam", req, provenance)

    @app.post("/api/skills")
    def create_skill(body: SkillDraft):
        return store.save_skill(body.model_dump())

    @app.put("/api/skills/{skill_id}")
    def save_skill(skill_id: str, body: SkillDraft):
        if not store.skill(skill_id):
            raise KeyError("unknown skill")
        return store.save_skill(body.model_dump(), skill_id)

    @app.post("/api/scenes/validate")
    def scene_validate(body: SceneSpec):
        from .scene import validate_scene

        scene = validate_scene(body.model_dump())
        return {"valid": True, "sha256": digest(scene)}

    @app.post("/api/skills/{skill_id}/runs", status_code=202)
    def run(skill_id: str, body: RunRequest):
        return store.submit(skill_id, body.model_dump(), provenance)

    @app.get("/api/runs")
    def runs(skill_id: str | None = None):
        return store.runs(skill_id)

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str, tool: bool = False):
        row = store.run(run_id)
        if row is None:
            raise KeyError("unknown run")
        if tool:
            # Deliberately exclude evaluator result, target/world geometry and success answers.
            return {
                "id": row["id"],
                "status": row["status"],
                "feedback": {k: v for k, v in row["progress"].items() if k in ("phase", "tick", "sim_s", "control_source")},
                "policy_reported_complete": row["result"].get("policy_reported_complete", row["progress"].get("policy_reported_complete")),
            }
        return row

    @app.post("/api/runs/{run_id}/cancel")
    def cancel(run_id: str):
        return store.cancel(run_id)

    @app.get("/api/runs/{run_id}/stream/{view}")
    async def stream(run_id: str, view: str, request: Request):
        if view not in ("head", "observer") or not await run_in_threadpool(store.run, run_id):
            raise HTTPException(404)
        import asyncio

        async def frames():
            last = None
            while not await request.is_disconnected():
                row = await run_in_threadpool(store.run, run_id)
                p = row["progress"]
                name = p.get("head_file", "head.jpg") if view == "head" else "observer.jpg"
                folder = p.get("media", f"runs/{run_id}/live")
                path = (store.root / folder / name).resolve()
                if store.root in path.parents and path.is_file():
                    stamp = path.stat().st_mtime_ns
                    if stamp != last:
                        data = path.read_bytes()
                        last = stamp
                        yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(data)).encode() + b"\r\n\r\n" + data + b"\r\n"
                if row["status"] in ("complete", "failed", "blocked", "cancelled", "interrupted"):
                    break
                await asyncio.sleep(0.1)

        return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame", headers={"Cache-Control": "no-store"})

    @app.get("/api/runs/{run_id}/dataset")
    def dataset_preview(run_id: str):
        row = store.run(run_id)
        if not row:
            raise KeyError("unknown run")
        root = store.root / "runs" / run_id / "data"
        if not (root / "index.json").is_file():
            return {"total": 0, "eligible": 0, "coverage": {}, "examples": []}
        from collections import Counter

        index = json.loads((root / "index.json").read_text())
        episodes = index["episodes"]
        return {
            "total": len(episodes),
            "eligible": sum(e["eligible"] for e in episodes),
            "coverage": dict(Counter(f"{e['split']} / {e['category']} / {e['color']} / {e['group']}" for e in episodes if e["eligible"])),
            "examples": [
                {**{k: e[k] for k in ("id", "split", "category", "color", "group", "eligible")}, "media": str((root / Path(e["path"]).parent).relative_to(store.root))} for e in episodes[-24:]
            ],
        }

    @app.get("/api/runs/{run_id}/metrics")
    def metrics(run_id: str):
        if not store.run(run_id):
            raise KeyError("unknown run")
        path = store.root / "runs" / run_id / "training/metrics.jsonl"
        if not path.is_file():
            return []
        points = []
        for line in path.read_text().splitlines()[-5000:]:
            try:
                points.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # A worker may be writing the last line.
        return points

    @app.post("/api/runs/{run_id}/resume", status_code=202)
    def resume(run_id: str):
        row = store.run(run_id)
        if not row or row["status"] not in ("cancelled", "failed", "interrupted"):
            raise ValueError("only interrupted/failed runs can resume")
        if not (store.root / "runs" / run_id / "training/resume.json").exists():
            raise ValueError("no verified training checkpoint available")
        req = {**row["snapshot"]["request"], "resume_run": run_id, "idempotency_key": None}
        if digest(store.skill(row["skill_id"])) != digest(row["snapshot"]["skill"]):
            raise ValueError("skill changed; restore the frozen task before resuming")
        return store.submit(row["skill_id"], req, provenance)

    @app.post("/api/runs/{run_id}/control")
    def control(run_id: str, body: dict):
        from ..navigation_action_contracts import action_contract

        row = store.run(run_id)
        if not row or row["status"] != "running" or row["snapshot"]["request"]["kind"] != "correction":
            raise ValueError("not an active correction session")
        action = action_contract().validate(body["action"])
        if action.shape != (5,):
            raise ValueError("one nav5 action required")
        write_json(store.root / "runs" / run_id / "control.json", {"action": action.tolist(), "expires": time.time() + 0.3, "source": "human_keyboard"})
        return {"accepted": True, "valid_for_ms": 300}

    @app.post("/api/worker/claim")
    def claim(body: dict):
        return store.claim(body["id"], body["capabilities"], body["provenance"])

    @app.post("/api/worker/runs/{run_id}")
    def worker_update(run_id: str, body: dict):
        lease = body.pop("lease")
        return store.update(run_id, lease, **body)

    @app.post("/api/skills/{skill_id}/publish")
    def publish(skill_id: str, body: dict):
        row = store.run(body["run_id"])
        if not row or row["skill_id"] != skill_id or row["status"] != "complete":
            raise ValueError("publish requires a completed training run")
        rel = row["result"].get("candidate")
        if not rel:
            raise ValueError("run produced no model candidate")
        model = (store.root / rel).resolve()
        if store.root not in model.parents:
            raise ValueError("candidate path escapes workspace")
        manifest = json.loads((model / "model.json").read_text())
        for name, sha in manifest["files"].items():
            path = (model / name).resolve()
            if model not in path.parents or file_hash(path) != sha:
                raise ValueError("candidate checksum mismatch")
        report = row["result"].get("evaluation", {})
        accepted = report.get("accepted", False) and report.get("model_sha256") == manifest["files"]["model.pt"]
        return store.add_version(
            skill_id,
            {
                "name": body.get("name") or f"{manifest['backend']} · step {manifest['step']}",
                "model_path": rel,
                "manifest_sha256": file_hash(model / "model.json"),
                "backend": manifest["backend"],
                "observation": manifest["observation"],
                "status": "accepted_" + report["mode"] if accepted else "experimental",
                "report": report,
                "run_id": row["id"],
                "task_contract": row["snapshot"]["skill"]["task"],
                "skill_snapshot": row["snapshot"]["skill"],
                "environment": row["snapshot"].get("environment"),
                "provenance": row["snapshot"]["provenance"],
                "recipe": row["snapshot"]["request"]["recipe"],
                "hardware_validated": False,
            },
        )

    @app.get("/api/versions/{version_id}/export")
    def export(version_id: str, offline: bool = False):
        from .packages import export_bundle

        path = export_bundle(store, version_id, offline=offline, dependency_root=os.environ.get("STUDIO_MODEL_ROOT"))
        return FileResponse(path, filename=path.name)

    @app.post("/api/versions/{version_id}/qualify")
    def qualify(version_id: str, body: dict):
        """Explicitly register an evaluated release; the old version stays immutable."""
        from .evidence import summarize

        version = store.version(version_id)
        row = store.run(body["run_id"])
        if not version or not row or row["status"] != "complete" or row["snapshot"]["request"]["kind"] != "evaluate":
            raise ValueError("qualification requires a completed independent evaluation")
        if row["skill_id"] != version["skill_id"] or row["snapshot"]["request"]["version"] != version_id:
            raise ValueError("evaluation belongs to another model version")
        model = (store.root / version["model_path"]).resolve()
        if store.root not in model.parents or file_hash(model / "model.json") != version["manifest_sha256"]:
            raise ValueError("published model manifest changed")
        manifest = json.loads((model / "model.json").read_text())
        for name, sha in manifest["files"].items():
            path = (model / name).resolve()
            if model not in path.parents or file_hash(path) != sha:
                raise ValueError("published model changed")
        report = row["result"].get("evaluation", row["result"])
        if report.get("model_sha256") != manifest["files"]["model.pt"] or report.get("task_sha256") != digest(version["task_contract"]):
            raise ValueError("report does not match frozen model/task")
        cases_path = store.root / "runs" / row["id"] / "evaluation-cases.json"
        if not cases_path.is_file() or digest(json.loads(cases_path.read_text())) != report.get("cases_sha256"):
            raise ValueError("frozen test cases missing or changed")
        recomputed = summarize(report["episodes"], groups=version["task_contract"]["groups"], mode=report["mode"], model_sha=report["model_sha256"], case_sha=report["cases_sha256"])
        if not recomputed["accepted"]:
            raise ValueError("independent skill acceptance gate has not passed")
        return store.add_version(
            version["skill_id"],
            {
                **{k: v for k, v in version.items() if k not in ("id", "skill_id")},
                "name": version["name"],
                "status": "accepted_" + report["mode"],
                "derived_from": version_id,
                "qualification_run": row["id"],
                "report": {**report, **recomputed},
            },
        )

    @app.post("/api/import")
    def import_skill(file: UploadFile = File(...)):
        from .packages import import_bundle

        path = store.root / ("upload-" + secrets.token_hex(12) + ".tar")
        try:
            total = 0
            with path.open("wb") as out:
                while block := file.file.read(8 * 1024 * 1024):
                    total += len(block)
                    if total > 40 * 1024**3:
                        raise ValueError("package exceeds 40 GiB")
                    out.write(block)
            return import_bundle(store, path)
        finally:
            path.unlink(missing_ok=True)

    @app.post("/api/compare")
    def comparison(body: dict):
        from .evidence import compare

        reports = []
        for key in body["runs"]:
            row = store.run(key)
            if not row or row["status"] != "complete":
                raise ValueError("comparison requires completed evaluations")
            reports.append(row["result"].get("evaluation", row["result"]))
        return compare(reports, body.get("prices"))

    @app.get("/api/files/{path:path}")
    def files(path: str):
        p = (store.root / path).resolve()
        # Model/data downloads use explicit exports; teacher truth is not a runtime feed.
        if store.root not in p.parents or p.suffix not in (".jpg", ".mp4", ".json", ".jsonl", ".log") or p.name in ("teacher_truth.json", "snapshot.json", "control.json"):
            raise HTTPException(404)
        if not p.is_file():
            raise HTTPException(404)
        return FileResponse(p, headers={"Cache-Control": "no-store"})

    app.mount("/media", StaticFiles(directory=REPO / "examples/microduck/visualization"), name="history-media")
    frontend = REPO / "apps/microduck-studio/dist"
    if frontend.is_dir():
        app.mount("/", StaticFiles(directory=frontend, html=True), name="studio")
    return app


def main():
    import faulthandler
    import signal
    import uvicorn

    # Allow diagnosing a stuck service without evaluating code or changing jobs.
    if hasattr(signal, "SIGUSR1"):
        faulthandler.register(signal.SIGUSR1, all_threads=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8771)
    parser.add_argument("--workspace", default="artifacts/studio")
    args = parser.parse_args()
    # SSE/MJPEG connections otherwise keep the old process and its SQLite
    # connection alive indefinitely during a restart. Clients reconnect to the
    # current episode; workers retain their lease across a brief Web outage.
    uvicorn.run(create_app(args.workspace), host=args.host, port=args.port, timeout_graceful_shutdown=3)


if __name__ == "__main__":
    main()
