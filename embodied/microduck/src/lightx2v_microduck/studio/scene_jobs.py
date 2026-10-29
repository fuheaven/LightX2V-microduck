"""Data-only layout drafts. An LLM cannot execute code or save the user's scene."""

import json
import os
from copy import deepcopy

from pydantic import Field

from .schema import ObjectSpec, SceneSpec, Spawn, Strict, digest


class SceneChanges(Strict):
    objects: list[ObjectSpec] = Field(max_length=30)
    spawn: Spawn
    target_id: str
    size: float = Field(ge=2, le=6)


def apply_changes(scene, changes):
    from .scene import validate_scene

    change = SceneChanges.model_validate(changes).model_dump()
    result = SceneSpec.model_validate({**deepcopy(scene), **change, "version": 2}).model_dump()
    validate_scene(result)
    return result


def generate(job):
    import httpx

    from .assets import catalog

    endpoint, model = os.getenv("STUDIO_SCENE_LLM_URL"), os.getenv("STUDIO_SCENE_LLM_MODEL")
    if not endpoint or not model:
        from .pipeline import GateBlocked

        raise GateBlocked("自然语言布局未配置：需要 STUDIO_SCENE_LLM_URL / STUDIO_SCENE_LLM_MODEL")
    job.progress({"phase": "layout_draft", "message": "正在生成布局草案"})
    # URL and credentials are operator configuration, never model/user supplied.
    with httpx.Client(timeout=90) as client:
        response = client.post(
            endpoint.rstrip("/") + "/chat/completions",
            headers={"Authorization": "Bearer " + os.getenv("STUDIO_SCENE_LLM_KEY", "")},
            json={
                "model": model,
                "temperature": 0.1,
                "max_tokens": 6000,
                "response_format": {"type": "json_object"},
                "messages": [
                    {
                        "role": "system",
                        "content": "Edit a static robot room using data only. Return ONE JSON object matching this schema: "
                        + json.dumps(SceneChanges.model_json_schema())
                        + ". Units metres, Z up, yaw degrees. Preserve objects not requested to change and keep stable IDs. No code. Avoid overlaps, retain one unique target and a clear robot path. Registered furniture: "
                        + json.dumps(catalog()),
                    },
                    {"role": "user", "content": json.dumps({"scene": job.skill["scene"], "request": job.request["prompt"]}, ensure_ascii=False)},
                ],
            },
        )
        response.raise_for_status()
        payload = response.json()
    if job.stop():
        raise InterruptedError("cancelled")
    draft = apply_changes(job.skill["scene"], json.loads(payload["choices"][0]["message"]["content"]))
    return {
        "scene": draft,
        "base_sha256": digest(job.skill["scene"]),
        "scene_sha256": digest(draft),
        "requires_application": True,
        "model": model,
        "usage": payload.get("usage"),
        "prompt": job.request["prompt"],
    }
