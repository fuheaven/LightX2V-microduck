"""Task plugins are installed Python entry points, never executable browser payloads."""

import importlib.metadata
import math
import numpy as np
from .scene import make_env, surface_distance


class FindObject:
    name = "find_object_v1"
    version = "1.0.0"
    action_contract = "microduck_navigation_v1"

    def environment(self, scene, task, **kwargs):
        return make_env(scene, task, **kwargs)

    def teacher(self, env):
        return ObjectTeacher(env)

    def score(self, env, interventions=0):
        w = env.evaluator_state()
        reason = "collision" if w["collision"] else "fall" if w["ever_fallen"] else "takeover" if interventions or w["recovery_events"] else "success" if w["success"] else "timeout"
        return {"success": reason == "success", "reason": reason, "hold_s": w["goal_hold_ticks"] * 0.02, "tick": env.tick, "interventions": interventions, "hardware_validated": False}


def plugin(name):
    if name == "find_object_v1":
        return FindObject()
    entries = list(importlib.metadata.entry_points(group="microduck.studio.tasks", name=name))
    if len(entries) != 1:
        raise ValueError(f"task plugin {name} is not installed")
    return entries[0].load()()


class ObjectTeacher:
    """Privileged A* demonstrator; qualified only by separate closed-loop checks."""

    def __init__(self, env):
        self.env = env
        self.seen = False
        self.last_diagnostics = {}

    def predict(self, observation):
        from ..teachers import plan_grid_path, _angle

        env = self.env
        world = env.evaluator_state()
        target = next(o for o in env.studio_scene["objects"] if o["id"] == env.studio_scene["target_id"])
        # Visibility is teacher-only segmentation, never a student feature.
        renderer = env.physics.head_renderer()
        renderer.enable_segmentation_rendering()
        try:
            renderer.update_scene(env.physics.data, camera=env.physics.camera)
            seg = renderer.render()
        finally:
            renderer.disable_segmentation_rendering()
        mj, model = env.physics.mj, env.physics.model
        body = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "studio_" + target["id"])
        geoms = np.where(model.geom_bodyid == body)[0]
        visible = np.count_nonzero(np.isin(seg[:, :, 0], geoms) & (seg[:, :, 1] == int(mj.mjtObj.mjOBJ_GEOM))) >= 16
        self.seen |= visible
        self.last_diagnostics = {"teacher_uses_truth": True, "visible": bool(visible), "seen": bool(self.seen)}
        if not self.seen and not world["obstacles"]:
            return np.array([0, 0, 0.65, 0, 0], np.float32)
        pos = np.array(world["robot_position"][:2])
        distance = surface_distance(target, pos)
        lo, hi = env.task.stop_distance_m
        if lo + 0.02 <= distance <= hi - 0.03:
            return np.array([0, 0, 0, 0, 1], np.float32)
        if distance < lo + 0.02:
            return np.array([-0.07, 0, 0, 0, 0], np.float32)
        radius = max(target["width"], target["depth"]) / 2 + (lo + hi) / 2
        candidates = [np.array([target["x"], target["y"]]) + radius * np.array([math.cos(a), math.sin(a)]) for a in np.linspace(-math.pi, math.pi, 24, endpoint=False)]
        path = []
        for stop in sorted(candidates, key=lambda p: np.linalg.norm(p - pos)):
            if not lo <= surface_distance(target, stop) <= hi:
                continue
            path = plan_grid_path(pos, stop, world["obstacles"], world["room_bounds"])
            if path:
                break
        if not path:
            self.last_diagnostics["no_path"] = True
            return np.array([0, 0, 0, 0, 1], np.float32)
        waypoint = next((p for p in path[1:] if np.linalg.norm(p - pos) >= 0.14), path[-1])
        error = _angle(math.atan2(*(waypoint - pos)[::-1]) - world["robot_yaw"])
        return np.array([0.3 if abs(error) < 0.65 else 0, 0, np.clip(error * 2, -1.1, 1.1), 0, 0], np.float32)


class PixelRule:
    """Sensor-only color/search/ToF baseline. Shape ambiguity is reported, never oracle resolved."""

    def __init__(self, color):
        self.color = color

    def predict(self, observation):
        rgb = observation.rgb.astype(np.float32) / 255
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        mask = {"red": (r > g * 1.5) & (r > b * 1.5) & (r > 0.3), "blue": (b > r * 1.4) & (b > g * 1.3) & (b > 0.3), "yellow": (r > 0.4) & (g > 0.35) & (b < g * 0.6)}[self.color]
        y, x = np.nonzero(mask)
        if len(x) < 16:
            return np.array([0, 0, 0.6, 0, 0], np.float32)
        turn = float(np.clip((rgb.shape[1] / 2 - x.mean()) / rgb.shape[1] * 2, -0.7, 0.7))
        ranges = observation.tof
        near = float(np.median(ranges[2:6, 2:6])) if ranges is not None else 4
        if len(x) / mask.size > 0.018 and 0.25 <= near <= 0.40:
            return np.array([0, 0, 0, 0, 1], np.float32)
        if near < 0.25:
            return np.array([0, 0, 0.7, 0, 0], np.float32)
        return np.array([0.3 if abs(turn) < 0.35 else 0, 0, turn, 0, 0], np.float32)
