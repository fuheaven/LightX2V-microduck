"""Portable, versioned public contracts; paths and credentials are worker configuration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator, model_serializer

CATEGORIES = ("ball", "box", "toy")
COLORS = ("red", "blue", "yellow")
GROUPS = ("clear", "outside_view", "obstacle", "occlusion")
OBSERVATION = "rgb_history_300_150_50_0+proprio48+tof64+valid64_v1"
ACTION = "microduck_navigation_v1"


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(canonical(value) + "\n")
    temp.replace(path)


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ObjectSpec(Strict):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,39}$")
    kind: Literal["ball", "box", "toy", "obstacle"] = "ball"
    color: Literal["red", "blue", "yellow", "gray"] = "red"
    x: float = Field(default=0.8, ge=-3, le=3)
    y: float = Field(default=0.3, ge=-3, le=3)
    yaw: float = Field(default=0, ge=-180, le=180)
    width: float = Field(default=0.08, ge=0.04, le=3)
    depth: float = Field(default=0.08, ge=0.04, le=3)
    height: float = Field(default=0.08, ge=0.04, le=2)
    jitter: float = Field(default=0.15, ge=0, le=1)
    asset_id: Literal["table", "chair", "sofa", "cabinet", "partition"] | None = None
    asset_version: Literal["1.0.0"] | None = None
    material: Literal["wood", "fabric", "slate"] | None = None

    @model_validator(mode="after")
    def asset(self):
        if self.asset_id:
            if self.kind != "obstacle" or self.asset_version != "1.0.0" or self.material is None:
                raise ValueError("furniture requires obstacle kind, asset version and material")
        elif self.asset_version or self.material or max(self.width, self.depth, self.height) > 0.8:
            raise ValueError("primitive objects keep the v1 dimensions and material contract")
        return self

    @model_serializer(mode="wrap")
    def portable(self, handler):
        data = handler(self)
        if self.asset_id is None:
            for k in ("asset_id", "asset_version", "material"):
                data.pop(k, None)
        return data


class Spawn(Strict):
    x: float = Field(default=0, ge=-3, le=3)
    y: float = Field(default=0, ge=-3, le=3)
    yaw: float = Field(default=0, ge=-180, le=180)
    jitter: float = Field(default=0.12, ge=0, le=0.5)
    yaw_jitter: float = Field(default=15, ge=0, le=180)


class SceneSpec(Strict):
    version: Literal[1, 2] = 1
    name: str = Field(default="我的房间", min_length=1, max_length=80)
    size: float = Field(default=3, ge=2, le=6)
    spawn: Spawn = Field(default_factory=Spawn)
    target_id: str = "target"
    objects: list[ObjectSpec] = Field(default_factory=lambda: [ObjectSpec(id="target")], min_length=1, max_length=30)

    @model_validator(mode="after")
    def unique(self):
        if self.version == 1 and any(o.asset_id for o in self.objects):
            raise ValueError("furniture requires scene version 2")
        ids = [o.id for o in self.objects]
        if len(set(ids)) != len(ids) or self.target_id not in ids:
            raise ValueError("object IDs must be unique and include target_id")
        target = next(o for o in self.objects if o.id == self.target_id)
        if target.kind == "obstacle" or target.color == "gray":
            raise ValueError("target must have a registered class and color")
        if sum((o.kind, o.color) == (target.kind, target.color) for o in self.objects) != 1:
            raise ValueError("ambiguous target: duplicate class/color")
        return self

    @model_serializer(mode="wrap")
    def portable(self, handler):
        data = handler(self)
        if self.version == 1:
            data.pop("version", None)
        return data


class TaskConfig(Strict):
    plugin: str = Field(default="find_object_v1", pattern=r"^[a-z][a-z0-9_]{0,63}$")
    categories: list[Literal["ball", "box", "toy"]] = Field(default_factory=lambda: list(CATEGORIES), min_length=1)
    colors: list[Literal["red", "blue", "yellow"]] = Field(default_factory=lambda: list(COLORS), min_length=1)
    groups: list[Literal["clear", "outside_view", "obstacle", "occlusion"]] = Field(default_factory=lambda: list(GROUPS), min_length=1)
    stop_min: float = Field(default=0.25, ge=0.1, le=0.8)
    stop_max: float = Field(default=0.40, ge=0.15, le=1)
    hold_s: float = Field(default=2, ge=2, le=10)
    timeout_s: float = Field(default=45, ge=3, le=180)

    @model_validator(mode="after")
    def bounds(self):
        if self.stop_min >= self.stop_max:
            raise ValueError("stop_min must be less than stop_max")
        for values in (self.categories, self.colors, self.groups):
            if len(set(values)) != len(values):
                raise ValueError("duplicate task dimensions")
        return self


class Recipe(Strict):
    backend: Literal["fastwam", "lightweight"] = "lightweight"
    learning_rate: float = Field(default=1e-4, gt=0, le=0.01)
    batch_size: int = Field(default=64, ge=1, le=512)
    micro_batch: int = Field(default=4, ge=1, le=64)
    max_steps: int = Field(default=5000, ge=1, le=1000000)
    validate_every: int = Field(default=250, ge=1, le=100000)
    max_hours: float = Field(default=12, gt=0, le=720)
    seed: int = Field(default=42, ge=0, le=2147483647)
    precision: Literal["bf16", "fp32"] = "bf16"

    @model_validator(mode="after")
    def batch(self):
        if self.batch_size % self.micro_batch:
            raise ValueError("effective batch must be divisible by micro_batch")
        return self


def recipe_for(backend):
    return Recipe(backend="fastwam", learning_rate=3e-6, batch_size=8, micro_batch=1, max_steps=1000) if backend == "fastwam" else Recipe()


class SkillDraft(Strict):
    name: str = Field(default="寻找并接近物体", min_length=1, max_length=80)
    description: str = Field(default="搜索 · 绕障 · 重新寻找 · 接近停稳", max_length=500)
    task: TaskConfig = Field(default_factory=TaskConfig)
    scene: SceneSpec = Field(default_factory=SceneSpec)


class VideoOptions(Strict):
    source_run: str = Field(pattern=r"^[a-f0-9]{16}$")
    source_file: str = Field(max_length=300)
    start_s: float = Field(default=0, ge=0)
    duration_s: float = Field(default=5, ge=5, le=15)
    width: int = Field(default=960, ge=256, le=1920, multiple_of=32)
    height: int = Field(default=544, ge=256, le=1920, multiple_of=32)
    steps: int = Field(default=30, ge=1, le=100)
    seed: int = Field(default=42, ge=0, le=2147483647)


class RunRequest(Strict):
    kind: Literal["preview", "teacher", "collect", "train", "pipeline", "evaluate", "live", "correction", "optimize", "historical", "scene_draft", "video_enhance"] = "preview"
    prompt: str = Field(default="", max_length=3000)
    video: VideoOptions | None = None
    recipe: Recipe = Field(default_factory=Recipe)
    version: str | None = None
    data_run: str | None = None
    teacher_run: str | None = None
    resume_run: str | None = None
    policy: Literal["teacher", "rule", "fastwam", "lightweight", "vlm_local", "vlm_cloud"] = "rule"
    mode: Literal["offline", "realtime"] = "offline"
    variant: Literal["plain", "optimized"] = "optimized"
    seed: int = Field(default=0, ge=0, le=2147483647)
    category: Literal["ball", "box", "toy"] = "ball"
    color: Literal["red", "blue", "yellow"] = "red"
    group: Literal["clear", "outside_view", "obstacle", "occlusion"] = "clear"
    teacher_cases: int = Field(default=100, ge=1, le=1000)
    episodes: int = Field(default=1000, ge=1, le=100000)
    evaluation_cases: int = Field(default=200, ge=1, le=10000)
    max_action_age_ms: int = Field(default=500, ge=100, le=2000)
    idempotency_key: str | None = Field(default=None, max_length=120)


def source_identity(repo):
    """Content identity includes dirty files, not just the last git commit."""
    paths = []
    for folder in ("embodied/microduck/src", "lightx2v", "scripts/microduck", "deploy/microduck/studio", "third_party/fastwam", "examples/microduck/studio"):
        paths.extend(p for p in (Path(repo) / folder).rglob("*") if p.is_file() and (p.suffix in (".py", ".json", ".yaml", ".lock") or p.name.endswith("Dockerfile")))
    return digest({str(p.relative_to(repo)): file_hash(p) for p in sorted(paths)})
