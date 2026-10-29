"""Observer and authoring endpoints; never exported through the robot tool API."""

import asyncio
import json
import os

from fastapi import HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from starlette.concurrency import run_in_threadpool

from .schema import SceneSpec
from .store import TERMINAL


def install(app, store, provenance):
    @app.get("/api/assets")
    def assets():
        from .assets import catalog

        return catalog()

    @app.get("/api/assets/robot")
    def robot():
        p = store.root / "assets/robot.json"
        if not p.exists():
            raise HTTPException(503, "官方本体尚未准备好，请检查 CPU 运行器和资产配置")
        value = json.loads(p.read_text())
        if value["source_sha256"] != provenance["source_sha256"]:
            raise HTTPException(503, "本体可视缓存正在更新，请启动匹配版本的 CPU 运行器")
        import gzip

        return Response(gzip.compress(json.dumps(value).encode(), compresslevel=3), media_type="application/json", headers={"Content-Encoding": "gzip", "Vary": "Accept-Encoding"})

    @app.get("/api/authoring/status")
    def status():
        workers = [w for w in store.workers() if w["online"]]
        video = next((w for w in workers if "video" in w.get("capabilities", [])), None)
        return {
            "scene_model_configured": bool(os.getenv("STUDIO_SCENE_LLM_URL") and os.getenv("STUDIO_SCENE_LLM_MODEL")),
            "video_ready": bool(video and video.get("video_ready")),
            "video_reason": video.get("video_reason") if video else "尚未连接视频运行器与 H3 模型",
            "scene_draft_requires_apply": True,
        }

    @app.post("/api/environments/export")
    def export(body: SceneSpec):
        from .environments import export_environment

        return export_environment(body.model_dump())

    @app.post("/api/environments/import")
    def import_scene(body: dict):
        from .environments import import_environment

        return {"scene": import_environment(body)}

    @app.post("/api/runs/{run_id}/apply-scene")
    def apply(run_id: str):
        row = store.run(run_id)
        if not row or row["status"] != "complete" or row["snapshot"]["request"]["kind"] != "scene_draft":
            raise ValueError("草案尚未完成")
        return store.apply_scene_draft(row["skill_id"], row["result"]["scene"], row["result"]["base_sha256"])

    def observer_file(run_id, name):
        row = store.run(run_id)
        if not row:
            raise KeyError("unknown run")
        folder = row["progress"].get("media", row["result"].get("media", f"runs/{run_id}/live"))
        path = (store.root / folder / name).resolve()
        if store.root / "runs" / run_id not in path.parents:
            raise ValueError("observer record outside run")
        return row, path

    @app.get("/api/runs/{run_id}/visual")
    def visual(run_id: str):
        row, path = observer_file(run_id, "visual.json")
        if not path.exists():
            raise HTTPException(404, "此运行尚无三维位姿；历史实验保留原始录像")
        import gzip

        return Response(gzip.compress(path.read_bytes(), compresslevel=3), media_type="application/json", headers={"Content-Encoding": "gzip", "Vary": "Accept-Encoding"})

    @app.get("/api/runs/{run_id}/state-stream")
    async def state_stream(run_id: str, request: Request):
        if not await run_in_threadpool(store.run, run_id):
            raise HTTPException(404)

        def read():
            row, path = observer_file(run_id, "state.json")
            return row["status"], path.read_text() if path.exists() else None

        async def events():
            last = None
            count = 0
            while not await request.is_disconnected():
                status, data = await run_in_threadpool(read)
                if data and data != last:
                    value = json.loads(data)
                    last = data
                    yield f"id: {value['episode_id']}:{value['tick']}\nevent: state\ndata: {data.strip()}\n\n"
                if status in TERMINAL:
                    yield "event: ended\ndata: " + json.dumps({"status": status}) + "\n\n"
                    break
                count += 1
                if count % 30 == 0:
                    yield ": heartbeat\n\n"
                await asyncio.sleep(0.1)

        return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})
