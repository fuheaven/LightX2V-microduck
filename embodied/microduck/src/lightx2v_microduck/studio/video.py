"""Offline video artifacts with immutable sources and explicit visual review.

Reference implementation: Diffusers MiniMax-H3 Ref2VA. Nothing in this module
feeds generated frames back to policies, datasets, or task scores.
"""

import json
import math
import os
import shutil
import subprocess
import time
from fractions import Fraction
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from .schema import Strict, digest, file_hash, write_json

DIFFUSERS_REVISION = "d42ebf9c6e0b46103b74283f5324357a67acaa04"


def probe(path):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries", "stream=width,height,avg_frame_rate,nb_read_frames,duration:format=duration", "-of", "json", str(path)],
        capture_output=True,
        text=True,
        timeout=45,
        check=True,
    )
    value = json.loads(result.stdout)
    stream = value["streams"][0]
    fps = float(Fraction(stream["avg_frame_rate"]))
    frames = int(stream["nb_read_frames"])
    if fps <= 0 or frames < 1:
        raise ValueError("invalid video timebase")
    return {"width": stream["width"], "height": stream["height"], "fps": fps, "frames": frames, "duration_s": frames / fps}


def freeze_source(store, skill_id, options):
    if not options:
        raise ValueError("选择原始头部或观察者录像")
    row = store.run(options["source_run"])
    if not row or row["skill_id"] != skill_id or row["status"] not in ("complete", "failed", "cancelled", "blocked", "interrupted"):
        raise ValueError("source must be a finished simulation run of this skill")
    root = store.root / "runs" / row["id"]
    source = (root / options["source_file"]).resolve()
    if root not in source.parents or source.name not in ("head.mp4", "observer.mp4") or not (source.parent / "episode.json").is_file():
        raise ValueError("only independent original simulation recordings can be enhanced")
    if row["snapshot"]["request"]["kind"] == "video_enhance":
        raise ValueError("generated video is not an original recording")
    info = probe(source)
    if options["start_s"] + options["duration_s"] > info["duration_s"] + 1e-6:
        raise ValueError("所选片段超出原始录像时长")
    return {
        "path": str(source.relative_to(store.root)),
        "sha256": file_hash(source),
        "metadata": info,
        "snapshot_sha256": digest(row["snapshot"]),
        "environment": row["snapshot"].get("environment"),
        "source_model": row["snapshot"].get("model_version"),
        "view": source.stem,
        "source_run": row["id"],
        "episode_sha256": file_hash(source.parent / "episode.json"),
    }


def frame_plan(options, metadata):
    # H3 decodes 17*n+5 frames at 24 fps. Use the nearest legal count, then
    # map each output sample explicitly to an actual source frame (no flow).
    count = min(345, max(124, 17 * round((options["duration_s"] * 24 - 5) / 17) + 5))
    indices = [min(metadata["frames"] - 1, int(math.floor((options["start_s"] + i * options["duration_s"] / count) * metadata["fps"] + 0.5))) for i in range(count)]
    return {
        "fps": 24,
        "frames": count,
        "duration_s": count / 24,
        "requested_start_s": options["start_s"],
        "requested_duration_s": options["duration_s"],
        "sampling": "nearest_source_frame",
        "source_frame_indices": indices,
        "source_times_s": [i / metadata["fps"] for i in indices],
        "output_times_s": [i / 24 for i in range(count)],
    }


def readiness():
    import importlib.util

    path = Path(os.getenv("STUDIO_H3_MODEL", "/unconfigured"))
    reason = None
    if not (path / "modular_model_index.json").is_file():
        reason = "H3 本地模型未配置或缺少 modular_model_index.json"
    elif not (path / "studio-model-manifest.json").is_file():
        reason = "H3 模型缺少固定文件校验清单"
    elif not importlib.util.find_spec("diffusers") or not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        reason = "视频运行器依赖未就绪"
    else:
        import torch

        if not torch.cuda.is_available():
            reason = "视频运行器没有可用 CUDA GPU"
        else:
            minimum = float(os.getenv("STUDIO_H3_MIN_FREE_GIB", "100"))
            free, _ = torch.cuda.mem_get_info()
            if free < minimum * 1024**3:
                reason = f"可用显存不足配置门槛 {minimum:g} GiB"
    return {"video_ready": reason is None, "video_reason": reason, "video_backend": "minimax_h3_ref2va_diffusers"}


def verify_model(path):
    manifest = json.loads((path / "studio-model-manifest.json").read_text())
    if not manifest.get("revision") or not manifest.get("files"):
        raise ValueError("model requires a pinned revision and SHA256 file inventory")
    actual = {str(p.relative_to(path)) for p in path.rglob("*") if p.is_file() and ".cache" not in p.parts and p.name != "studio-model-manifest.json"}
    if actual != set(manifest["files"]):
        raise ValueError("H3 model file inventory changed")
    for name, expected in manifest["files"].items():
        target = (path / name).resolve()
        if path.resolve() not in target.parents or file_hash(target) != expected:
            raise ValueError(f"H3 model checksum mismatch: {name}")
    return {"revision": manifest["revision"], "manifest_sha256": digest(manifest)}


def make_reference(source, output, plan, size, stop):
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image, ImageOps

    reader = imageio.get_reader(str(source))
    writer = imageio.get_writer(str(output), fps=plan["fps"], codec="libx264", macro_block_size=1)
    try:
        last, frame = -1, None
        for index in plan["source_frame_indices"]:
            if stop():
                raise InterruptedError("cancelled")
            if index != last:
                frame = reader.get_data(index)
                last = index
            writer.append_data(np.asarray(ImageOps.pad(Image.fromarray(frame), size, color=(0, 0, 0))))
    finally:
        reader.close()
        writer.close()


def validate_output(info, plan, options):
    checks = {
        "dimensions": (info["width"], info["height"]) == (options["width"], options["height"]),
        "frame_rate": abs(info["fps"] - plan["fps"]) < 1e-6,
        "frame_count": info["frames"] == plan["frames"],
        "duration": abs(info["duration_s"] - plan["duration_s"]) < 1 / 24,
        "time_mapping": len(plan["source_frame_indices"]) == info["frames"] and all(a <= b for a, b in zip(plan["source_frame_indices"], plan["source_frame_indices"][1:])),
    }
    return {"passed": all(checks.values()), "checks": checks, "semantic_consistency": "requires_human_review"}


def generate(job):
    state = readiness()
    if not state["video_ready"]:
        from .pipeline import GateBlocked

        raise GateBlocked(state["video_reason"])
    import torch
    from diffusers import ModularPipeline
    from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3VideoReference
    from diffusers.utils.export_utils import encode_video

    options, source = job.request["video"], job.snapshot["video_source"]
    path = Path(os.environ["STUDIO_H3_MODEL"]).resolve()
    job.progress({"phase": "video_verify", "message": "校验原始录像与模型文件"})
    if file_hash(job.root / source["path"]) != source["sha256"]:
        raise ValueError("original recording changed since submission")
    model = verify_model(path)
    if job.stop():
        raise InterruptedError("cancelled")
    plan = frame_plan(options, source["metadata"])
    out = job.out / "video"
    out.mkdir(exist_ok=True)
    write_json(out / "time-map.json", plan)
    reference = out / "reference.mp4"
    make_reference(job.root / source["path"], reference, plan, (options["width"], options["height"]), job.stop)
    job.progress({"phase": "video_loading", "message": "加载 H3 Ref2VA 参考后端"})
    pipe = ModularPipeline.from_pretrained(str(path), workflow="ref2va", local_files_only=True)
    pipe.load_components(dtype=torch.bfloat16, local_files_only=True)
    pipe.to("cuda")
    pipe.register_to_config(canvas_short_edge=min(options["width"], options["height"]), canvas_max_pixels=options["width"] * options["height"])
    # Count real denoiser forwards and support cancellation between them.
    calls = 0

    def before_step(_module, _args):
        nonlocal calls
        if job.stop():
            raise InterruptedError("cancelled")
        calls += 1
        job.progress({"phase": "video_generation", "step": calls, "total": options["steps"]})

    hook = pipe.transformer_ref.register_forward_pre_hook(before_step)
    started = time.monotonic()
    try:
        result = pipe(
            prompt=job.request["prompt"],
            references=[MiniMaxH3VideoReference.from_file(str(reference))],
            width=options["width"],
            height=options["height"],
            num_frames=plan["frames"],
            num_inference_steps=options["steps"],
            generator=torch.Generator(device="cpu").manual_seed(options["seed"]),
            output=["videos"],
        )
    finally:
        hook.remove()
    if job.stop():
        raise InterruptedError("cancelled")
    if calls != options["steps"]:
        raise ValueError(f"actual denoiser passes {calls} differ from requested steps {options['steps']}")
    generated = out / "generated.mp4"
    temporary = out / "unlabelled.partial.mp4"
    encode_video(result["videos"][0], fps=24, output_path=str(temporary))
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            str(temporary),
            "-vf",
            "drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:text='AI GENERATED':x=12:y=12:fontsize=16:fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=6",
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-metadata",
            "title=AI-generated visual enhancement",
            str(generated),
        ],
        check=True,
        timeout=120,
    )
    temporary.unlink()
    info = probe(generated)
    checks = validate_output(info, plan, options)
    receipt = {
        "schema": "studio_video_enhancement_v1",
        "ai_generated": True,
        "source": source,
        "backend": "minimax_h3_ref2va_diffusers",
        "diffusers_revision": DIFFUSERS_REVISION,
        "lightx2v_accelerated": False,
        "model": model,
        "prompt": job.request["prompt"],
        "requested": options,
        "effective": {**info, "steps": calls, "seed": options["seed"]},
        "generated_sha256": file_hash(generated),
        "reference_sha256": file_hash(reference),
        "time_map_sha256": digest(plan),
        "visible_ai_label": True,
        "generation_seconds": time.monotonic() - started,
        "automatic_checks": checks,
        "review_state": "pending",
        "future_reference_frames_possible": True,
        "policy_input": False,
        "training_eligible": False,
        "skill_acceptance": False,
        "media": str(out.relative_to(job.root)),
        "source_unchanged": file_hash(job.root / source["path"]) == source["sha256"],
    }
    write_json(out / "provenance.json", receipt)
    if not checks["passed"] or not receipt["source_unchanged"]:
        raise ValueError("generated file failed automatic consistency checks; see video/provenance.json")
    return receipt


class Issue(Strict):
    start_s: float = Field(ge=0)
    end_s: float = Field(ge=0)
    note: str = Field(min_length=1, max_length=1000)


class Review(Strict):
    status: Literal["pending", "passed", "failed", "uncertain"] = "pending"
    target: Literal["passed", "failed", "uncertain"] = "uncertain"
    layout: Literal["passed", "failed", "uncertain"] = "uncertain"
    passage: Literal["passed", "failed", "uncertain"] = "uncertain"
    occlusion: Literal["passed", "failed", "uncertain"] = "uncertain"
    continuity: Literal["passed", "failed", "uncertain"] = "uncertain"
    issues: list[Issue] = Field(default_factory=list, max_length=100)
    note: str = Field(default="", max_length=2000)

    @model_validator(mode="after")
    def consistent(self):
        checks = (self.target, self.layout, self.passage, self.occlusion, self.continuity)
        if self.status == "passed" and (any(c != "passed" for c in checks) or self.issues):
            raise ValueError("所有检查项通过且无未解决问题，才能标记视觉审核通过")
        if self.status == "uncertain" and "failed" in checks:
            raise ValueError("已有明确失败项，应标记不通过")
        if any(i.end_s < i.start_s for i in self.issues):
            raise ValueError("invalid issue time range")
        return self
