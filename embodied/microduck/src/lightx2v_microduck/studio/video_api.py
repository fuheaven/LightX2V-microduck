import json
from pathlib import Path

from fastapi.responses import FileResponse

from .schema import file_hash
from .video import Review


def install(app, store):
    @app.get("/api/runs/{run_id}/recordings")
    def recordings(run_id: str):
        row = store.run(run_id)
        if not row:
            raise KeyError("unknown run")
        if row["status"] in ("queued", "running"):
            return []
        root = store.root / "runs" / run_id
        records = []
        for path in sorted(root.rglob("*.mp4")):
            if path.name not in ("head.mp4", "observer.mp4") or not (path.parent / "episode.json").exists():
                continue
            records.append({"file": str(path.relative_to(root)), "path": str(path.relative_to(store.root)), "view": path.stem})
        return records

    def completed(run_id):
        row = store.run(run_id)
        if not row or row["status"] != "complete" or row["snapshot"]["request"]["kind"] != "video_enhance":
            raise ValueError("视频增强作业尚未完成")
        root = store.root / "runs" / run_id / "video"
        receipt = json.loads((root / "provenance.json").read_text())
        if file_hash(root / "generated.mp4") != receipt["generated_sha256"]:
            raise ValueError("增强录像已变化，既有审核无效")
        return root, receipt

    @app.get("/api/runs/{run_id}/video-reviews")
    def reviews(run_id: str):
        completed(run_id)
        return store.video_reviews(run_id)

    @app.post("/api/runs/{run_id}/video-reviews")
    def review(run_id: str, body: Review):
        root, receipt = completed(run_id)
        if body.status == "passed" and not receipt["automatic_checks"]["passed"]:
            raise ValueError("文件自动检查未通过")
        if any(i.end_s > receipt["effective"]["duration_s"] for i in body.issues):
            raise ValueError("问题标注超出录像时长")
        return store.add_video_review(
            run_id, {**body.model_dump(), "generated_sha256": receipt["generated_sha256"], "scope": "visual_review_only", "training_eligible": False, "skill_acceptance": False}
        )

    @app.get("/api/runs/{run_id}/enhanced-export")
    def export(run_id: str):
        import tempfile
        import zipfile

        root, receipt = completed(run_id)
        reviews = store.video_reviews(run_id)
        export = root / "exports"
        export.mkdir(exist_ok=True)
        with tempfile.NamedTemporaryFile(suffix=".zip", dir=export, delete=False) as tmp:
            path = Path(tmp.name)
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as z:
            z.write(root / "generated.mp4", "AI_GENERATED.mp4")
            z.write(root / "provenance.json", "provenance.json")
            z.write(root / "time-map.json", "time-map.json")
            z.writestr("review.json", json.dumps({"history": reviews, "status": reviews[-1]["status"] if reviews else "pending"}, ensure_ascii=False, indent=2))
            z.writestr(
                "README.txt",
                "AI-generated visual enhancement. Source hash and frame time mapping are included. Review approval is visual only, not sensor fidelity or skill acceptance. Future reference frames may have been used. Not eligible for training or online policy input.\n",
            )
        from starlette.background import BackgroundTask

        return FileResponse(path, filename=f"microduck-AI-enhanced-{run_id}.zip", background=BackgroundTask(path.unlink))
