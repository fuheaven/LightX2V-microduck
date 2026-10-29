"""Editable scene descriptions, deterministic sampling and official-body composition."""

from __future__ import annotations
import math
from dataclasses import replace
import numpy as np

from .schema import SceneSpec, TaskConfig, digest
from .assets import parts


def collision_boxes(obj):
    """World XY projections of the very same boxes compiled into MuJoCo.

    All heights are conservatively occupied for the planar teacher; it will
    not plan under furniture. Physics still uses the actual multipart shape.
    """
    if not obj.get("asset_id"):
        return [{"id": obj["id"], "xy": [obj["x"], obj["y"]], "half_size": envelope(obj).tolist(), "height": obj["height"]}]
    a = math.radians(obj["yaw"])
    rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    return [
        {
            "id": f"{obj['id']}_{i}",
            "xy": (rot @ np.array(p["pos"][:2]) + [obj["x"], obj["y"]]).tolist(),
            "half_size": (np.abs(rot) @ (np.array(p["size"][:2]) / 2)).tolist(),
            "height": p["pos"][2] + p["size"][2] / 2,
        }
        for i, p in enumerate(parts(obj))
    ]


def envelope(obj):
    a = math.radians(obj["yaw"])
    return np.abs([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]]) @ np.array([obj["width"], obj["depth"]]) / 2


def surface_distance(obj, point):
    delta = np.asarray(point)[:2] - [obj["x"], obj["y"]]
    if obj["kind"] in ("ball", "toy"):
        return float(np.linalg.norm(delta) - max(obj["width"], obj["depth"]) / 2)
    a = math.radians(obj["yaw"])
    local = np.array([[math.cos(a), math.sin(a)], [-math.sin(a), math.cos(a)]]) @ delta
    q = np.abs(local) - np.array([obj["width"], obj["depth"]]) / 2
    return float(np.linalg.norm(np.maximum(q, 0)) + min(max(q), 0))


def layout_identity(scene):
    """Appearance/instance changes alone do not create a new spatial layout."""
    return digest(
        {
            "size": scene["size"],
            "spawn": {k: round(scene["spawn"][k], 6) for k in ("x", "y", "yaw")},
            "objects": sorted(
                [{"id": o["id"], **{k: round(o[k], 6) for k in ("x", "y", "yaw")}, **({k: o[k] for k in ("width", "depth", "height")} if o["kind"] == "obstacle" else {})} for o in scene["objects"]],
                key=lambda o: o["id"],
            ),
        }
    )


def validate_scene(value):
    scene = SceneSpec.model_validate(value).model_dump()
    half = scene["size"] / 2
    spawn = np.array([scene["spawn"]["x"], scene["spawn"]["y"]])
    if np.any(np.abs(spawn) > half - 0.2):
        raise ValueError("出生点过于接近墙面")
    for i, obj in enumerate(scene["objects"]):
        e, xy = envelope(obj), np.array([obj["x"], obj["y"]])
        if np.any(np.abs(xy) + e > half - 0.07):
            raise ValueError(f"{obj['id']} 超出房间边界")
        if surface_distance(obj, spawn) < 0.20:
            raise ValueError(f"{obj['id']} 与机器人出生区域重叠")
        for other in scene["objects"][:i]:
            if np.all(np.abs(xy - [other["x"], other["y"]]) < e + envelope(other) + 0.025):
                raise ValueError(f"{obj['id']} 与 {other['id']} 重叠")
    from ..teachers import plan_grid_path

    goal = next(o for o in scene["objects"] if o["id"] == scene["target_id"])
    obstacles = [b for o in scene["objects"] if o["id"] != scene["target_id"] for b in collision_boxes(o)]
    found = False
    for a in np.linspace(-math.pi, math.pi, 24, endpoint=False):
        stop = np.array([goal["x"], goal["y"]]) + (0.325 + max(goal["width"], goal["depth"]) / 2) * np.array([math.cos(a), math.sin(a)])
        if np.any(np.abs(stop) > half - 0.2) or any(np.all(np.abs(stop - o["xy"]) < np.array(o["half_size"]) + 0.15) for o in obstacles):
            continue
        if plan_grid_path(spawn, stop, obstacles, (-half, half, -half, half)):
            found = True
            break
    if not found:
        raise ValueError("出生点与目标停止区域之间没有可通行路径")
    return scene


def sample_scene(value, seed, *, group=None, category=None, color=None, split="train"):
    """Disjoint instance/phrasing families per split, actual geometry varies with seed."""
    base = SceneSpec.model_validate(value).model_dump()
    rng = np.random.default_rng(seed)
    for attempt in range(200):
        scene = SceneSpec.model_validate(base).model_dump()
        spawn = scene["spawn"]
        spawn["x"] += float(rng.uniform(-spawn["jitter"], spawn["jitter"]))
        spawn["y"] += float(rng.uniform(-spawn["jitter"], spawn["jitter"]))
        spawn["yaw"] = float(((spawn["yaw"] + rng.uniform(-spawn["yaw_jitter"], spawn["yaw_jitter"]) + 180) % 360) - 180)
        for obj in scene["objects"]:
            obj["x"] += float(rng.uniform(-obj["jitter"], obj["jitter"]))
            obj["y"] += float(rng.uniform(-obj["jitter"], obj["jitter"]))
        goal = next(o for o in scene["objects"] if o["id"] == scene["target_id"])
        goal["kind"] = category or goal["kind"]
        goal["color"] = color or goal["color"]
        # A parameterized target must remain unique after changing class/color.
        for other in scene["objects"]:
            if other["id"] != goal["id"] and (other["kind"], other["color"]) == (goal["kind"], goal["color"]):
                other["color"] = next(c for c in ("red", "blue", "yellow") if c != goal["color"])
        instance = {"train": (0.85, 1.05), "validation": (1.06, 1.15), "test": (1.16, 1.30), "teacher": (0.9, 1.2)}[split]
        scale = float(rng.uniform(*instance))
        for k in ("width", "depth", "height"):
            goal[k] = max(0.04, min(0.5, goal[k] * scale))
        if group in ("clear", "outside_view"):
            scene["objects"] = [o for o in scene["objects"] if o["kind"] != "obstacle" or o.get("asset_id")]
        if group in ("outside_view", "occlusion"):
            delta = np.array([goal["x"] - spawn["x"], goal["y"] - spawn["y"]])
            spawn["yaw"] = float(((math.degrees(math.atan2(delta[1], delta[0])) + (180 if group == "outside_view" else 0) + 180) % 360) - 180)
        if group in ("obstacle", "occlusion") and not any(o["kind"] == "obstacle" for o in scene["objects"]):
            mid = (np.array([spawn["x"], spawn["y"]]) + np.array([goal["x"], goal["y"]])) / 2
            scene["objects"].append(
                {"id": "curriculum_box", "kind": "obstacle", "color": "gray", "x": float(mid[0]), "y": float(mid[1]), "yaw": 0.0, "width": 0.16, "depth": 0.22, "height": 0.25, "jitter": 0.0}
            )
        try:
            validate_scene(scene)
            return scene
        except ValueError:
            continue
    raise ValueError("随机范围无法生成合法布局；请调整物体间距或随机范围")


def build_spec(scene, asset_root, profile, task):
    import mujoco
    from ..scenes import build_scene, Obstacle

    spec, description = build_scene(asset_root, profile, "official", replace(task, suite="studio_world"))
    half = scene["size"] / 2
    for name, xy, size in (("north", (0, half), (half, 0.025, 0.16)), ("south", (0, -half), (half, 0.025, 0.16)), ("east", (half, 0), (0.025, half, 0.16)), ("west", (-half, 0), (0.025, half, 0.16))):
        spec.worldbody.add_geom(name=f"studio_wall_{name}", type=mujoco.mjtGeom.mjGEOM_BOX, pos=(*xy, 0.16), size=size, rgba=(0.70, 0.73, 0.76, 1))
    colors = {"red": (0.9, 0.04, 0.04, 1), "blue": (0.04, 0.16, 0.95, 1), "yellow": (0.95, 0.80, 0.04, 1), "gray": (0.40, 0.43, 0.46, 1)}
    obstacles = []
    for obj in scene["objects"]:
        w, d, h = obj["width"], obj["depth"], obj["height"]
        a = math.radians(obj["yaw"]) / 2
        # Static task objects; no hidden teleports/reset during an episode.
        body = spec.worldbody.add_body(name="studio_" + obj["id"], pos=(obj["x"], obj["y"], 0), quat=(math.cos(a), 0, 0, math.sin(a)))
        kwargs = dict(rgba=colors[obj["color"]], friction=(0.8, 0.01, 0.001))
        if obj.get("asset_id"):
            for i, part in enumerate(parts(obj)):
                body.add_geom(
                    name=f"studio_geom_{obj['id']}_{i}", type=mujoco.mjtGeom.mjGEOM_BOX, size=tuple(v / 2 for v in part["size"]), pos=part["pos"], rgba=part["rgba"], friction=kwargs["friction"]
                )
        elif obj["kind"] == "ball":
            radius = max(w, d) / 2
            body.add_geom(name="studio_geom_" + obj["id"], type=mujoco.mjtGeom.mjGEOM_SPHERE, size=(radius, 0, 0), pos=(0, 0, radius), **kwargs)
        elif obj["kind"] == "toy":
            body.add_geom(type=mujoco.mjtGeom.mjGEOM_ELLIPSOID, size=(w / 2, d / 2, h * 0.34), pos=(0, 0, h * 0.34), **kwargs)
            body.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=(min(w, d) * 0.32, 0, 0), pos=(0, 0, h * 0.78), **kwargs)
            for sign in (-1, 1):
                body.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=(w * 0.12, 0, 0), pos=(sign * w * 0.25, 0, h * 0.98), **kwargs)
        else:
            body.add_geom(name="studio_geom_" + obj["id"], type=mujoco.mjtGeom.mjGEOM_BOX, size=(w / 2, d / 2, h / 2), pos=(0, 0, h / 2), **kwargs)
        if obj["id"] != scene["target_id"]:
            obstacles.extend(Obstacle(b["id"], tuple(b["xy"]), tuple(b["half_size"]), b["height"]) for b in collision_boxes(obj))
    return spec, replace(description, name="studio", room_bounds=(-half, half, -half, half), obstacles=tuple(obstacles), layout_id=digest(scene), geometry_hash=digest(scene))


def make_env(scene, task_config, *, asset_root=None, render=True):
    from ..env import MicroDuckEnv
    from ..contracts import TaskSpec

    class ObjectEnv(MicroDuckEnv):
        def _ball_positions(self):
            return {o["id"]: np.array([o["x"], o["y"], o["height"] / 2]) for o in scene["objects"] if o["kind"] != "obstacle"}

        def target_surface_distance(self, target, robot):
            return surface_distance(goal, robot)

    cfg = TaskConfig.model_validate(task_config)
    goal = next(o for o in scene["objects"] if o["id"] == scene["target_id"])
    task = TaskSpec(
        "studio_find_object",
        f"Find the {goal['color']} {goal['kind']} and approach it, then stop.",
        "visual_goal",
        duration_s=cfg.timeout_s,
        target_color=scene["target_id"],
        stop_distance_m=(cfg.stop_min, cfg.stop_max),
        stop_hold_s=cfg.hold_s,
        allowed_sensors=("rgb", "tof", "proprioception"),
    )
    env = ObjectEnv(asset_root=asset_root, render=render, scene="studio", scene_builder=lambda root, profile, _scene, t, **kw: build_spec(scene, root, profile, t))
    env.studio_task, env.studio_scene = task, scene
    return env
