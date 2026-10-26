"""Real CPU MuJoCo + official BAM; sensor input and scoring truth are separate."""

import hashlib
import math
import time
import uuid
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ..assets import load_lock, resolve_asset_root, verify_source
from ..contracts import HOME_POLICY, Observation, RobotContract, TaskSpec, pack_observation
from ..physics import BAM_REVISION, Physics
from ..policies import PolicyBank
from ..scenes import build_scene
from .courses import REVISION, scenario


def wrap(value):
    return float(np.arctan2(np.sin(value), np.cos(value)))


class SchoolWorld:
    def __init__(self, course, *, asset_root=None, render=False):
        self.course = course
        self.render_enabled = render or not course.privileged
        self.asset_root = resolve_asset_root(asset_root)
        verified = verify_source("microduck_rl", self.asset_root)
        if not verified["ok"]:
            raise ValueError(f"official model asset verification failed: {verified['errors'][:3]}")
        files = load_lock()["sources"]["microduck-policies"]["files"]
        self.hashes = {e["path"]: e["sha256"] for e in files}
        self.bank = PolicyBank(self.asset_root / "microduck-policies", hashes=self.hashes, profile="official_bam")
        task = TaskSpec("school_" + course.family, "school", "visual_goal", target_color=course.color)
        spec, self.description = build_scene(self.asset_root, "official_bam", "official", task)
        import mujoco

        self.marker_names = []
        for k in range(course.waypoints if course.family == "goto" else 0):
            name = "school_marker_" + str(k)
            spec.worldbody.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=(0.15, 0.002, 0), pos=(0, 0, 0.003), rgba=(0.2, 0.8, 0.4, 0.5), contype=0, conaffinity=0)
            self.marker_names.append(name)
        if course.family == "circle":
            for k in range(48):
                name = "school_circle_" + str(k)
                spec.worldbody.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_SPHERE, size=(0.012, 0, 0), pos=(0, 0, 0.008), rgba=(0.2, 0.8, 0.4, 1), contype=0, conaffinity=0)
                self.marker_names.append(name)
        self.physics = Physics(spec, profile="official_bam", image_size=256)
        # Load and verify both frozen actors before running any candidate.
        for skill in ("walk", "stand"):
            self.bank.infer(skill, np.zeros(61, np.float32))
        sources = [Path(__file__).parent / name for name in ("simulation.py", "policy.py", "courses.py")]
        sources += [Path(__file__).parent.parent / name for name in ("pixel_policy.py", "contracts.py", "physics.py", "policies.py")]
        self.identity = {
            "revision": REVISION,
            "robot": asdict(RobotContract()),
            "bam_revision": BAM_REVISION,
            "profile": "official_bam",
            "policies": self.hashes,
            "robot_signature": self.physics.robot_signature(),
            "physics_config": asdict(self.physics.profile),
            "sources": {str(path.relative_to(Path(__file__).parent.parent)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        }

    def reset(self, seed):
        p = self.physics
        self.layout = scenario(self.course, seed)
        self.episode_id, self.tick = uuid.uuid4().hex, 0
        self.raw = np.zeros(14, np.float32)
        self.bank.reset(self.episode_id)
        for k, name in enumerate(self.marker_names):
            xy = (
                self.layout["waypoints"][k]
                if self.course.family == "goto"
                else np.asarray(self.layout["center"]) + self.course.radius_m * np.array([np.cos(k * 2 * np.pi / 48), np.sin(k * 2 * np.pi / 48)])
            )
            gid = p._id(p.mj.mjtObj.mjOBJ_GEOM, name)
            p.model.geom_pos[gid, :2] = xy
        p.reset(robot_xy=self.layout["origin"], robot_yaw=self.layout["yaw"], ball_positions=self.layout["balls"], seed=seed)
        # Fixed warmup retains official actuator and action history state.
        for _ in range(100):
            state = pack_observation(*p.sensors(), self.raw, np.zeros(13))
            self.raw = self.bank.infer("stand", state)
            p.step(np.asarray(HOME_POLICY, np.float32) + self.raw)
        self.origin_time = float(p.data.time)
        self.initial = self.truth()
        self.previous = self.initial
        self.yaw_total = self.orbit_total = self.radius_error = 0.0
        self.max_displacement = 0.0
        self.hold = self.fallen_ticks = self.max_hold = 0
        self.waypoint = 0
        self.collision = self.fallen = self.ball_contact = self.success = self.done = False
        self.rows, self.actions, self.commands = [], [], []
        self.frame_cache, self.frame_tick, self.frame_metadata = None, -1, {}
        return self.observe()

    def truth(self):
        p, d = self.physics, self.physics.data
        quat = d.xquat[p.trunk]
        w, x, y, z = quat
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        balls = {c: d.xpos[p._id(p.mj.mjtObj.mjOBJ_BODY, "lx_ball_" + c)].copy() for c in ("red", "blue", "yellow")}
        return {
            "xy": d.xpos[p.trunk, :2].copy(),
            "height": float(d.xpos[p.trunk, 2]),
            "yaw": yaw,
            "gravity_z": float(p.sensors()[1][2]),
            "speed": float(np.linalg.norm(d.qvel[p.trunk_qvel : p.trunk_qvel + 2])),
            "balls": balls,
        }

    def policy_truth(self):
        if not self.course.privileged:
            raise RuntimeError("RGB controllers cannot request simulator truth")
        return {**self.truth(), "initial": self.initial, "layout": self.layout, "waypoint": min(self.waypoint, len(self.layout["waypoints"]) - 1), "yaw_total": self.yaw_total}

    def observe(self):
        p = self.physics
        if self.render_enabled and self.frame_tick // 5 != self.tick // 5:
            self.frame_cache, self.frame_tick = p.render(), self.tick
            self.frame_metadata = p.camera_metadata()
        return Observation(
            self.episode_id,
            self.tick,
            self.tick * 0.02,
            pack_observation(*p.sensors(), self.raw, np.zeros(13)),
            rgb=self.frame_cache.copy() if self.render_enabled else None,
            instruction=f"find {self.course.color} ball and stop",
            task_id="school_" + self.course.family,
            camera_tick=self.frame_tick if self.render_enabled else None,
            monotonic_ns=time.monotonic_ns(),
            camera_metadata=self.frame_metadata,
        )

    def step(self, intent):
        if self.done:
            raise RuntimeError("episode terminated")
        if intent.episode_id != self.episode_id or intent.source_tick != self.tick or intent.skill not in ("walk", "stand"):
            raise ValueError("invalid School skill source")
        for _ in range(intent.duration_ticks):
            p = self.physics
            state = pack_observation(*p.sensors(), self.raw, intent.command)
            self.raw = self.bank.infer(intent.skill, state)
            p.step(np.asarray(HOME_POLICY, np.float32) + self.raw)
            self.tick += 1
            self.actions.append(self.raw.copy())
            self.commands.append(intent.command.copy())
            self._score()
            if self.done:
                break
        return self.observe()

    def _score(self):
        now, c, p = self.truth(), self.course, self.physics
        self.yaw_total += wrap(now["yaw"] - self.previous["yaw"])
        center = np.asarray(self.layout["center"])
        ray, old_ray = now["xy"] - center, self.previous["xy"] - center
        self.orbit_total += wrap(math.atan2(ray[1], ray[0]) - math.atan2(old_ray[1], old_ray[0]))
        self.radius_error += abs(float(np.linalg.norm(ray)) - c.radius_m)
        self.fallen_ticks = self.fallen_ticks + 1 if now["gravity_z"] > -0.58 or now["height"] < 0.073 else 0
        self.fallen |= self.fallen_ticks >= 10
        for a, b in p.last_contacts:
            body_a, body_b = int(p.model.geom_bodyid[a]), int(p.model.geom_bodyid[b])

            def belongs(body):
                while body > 0:
                    if body == p.trunk:
                        return True
                    body = int(p.model.body_parentid[body])
                return False

            if belongs(body_a) == belongs(body_b):
                continue
            other = b if belongs(body_a) else a
            name = p.mj.mj_id2name(p.model, p.mj.mjtObj.mjOBJ_BODY, int(p.model.geom_bodyid[other])) or ""
            if c.family == "ball" and name == "lx_ball_red":
                self.ball_contact = True
            elif p.model.geom_type[other] != p.mj.mjtGeom.mjGEOM_PLANE:
                self.collision = True
        displacement = float(np.linalg.norm(now["xy"] - self.initial["xy"]))
        self.max_displacement = max(self.max_displacement, displacement)
        target = np.array(self.layout["waypoints"][min(self.waypoint, len(self.layout["waypoints"]) - 1)])
        distance = float(np.linalg.norm(now["xy"] - target))
        surface = float(np.linalg.norm(now["xy"] - now["balls"][c.color][:2])) - 0.035
        ball_motion = float(np.linalg.norm(now["balls"]["red"][:2] - self.initial["balls"]["red"][:2]))
        upright = now["gravity_z"] < -0.82 and now["height"] > 0.09
        if c.family == "goto" and distance < 0.15 and self.waypoint < c.waypoints - 1:
            self.waypoint += 1
        recent_yaw = [row[3] for row in self.rows[-10:]] + [now["yaw"]]
        yaw_speed = abs(wrap(recent_yaw[-1] - recent_yaw[0])) / max(0.02, (len(recent_yaw) - 1) * 0.02)
        ready = (
            (c.family == "goto" and self.waypoint == c.waypoints - 1 and distance < 0.15 and now["speed"] < 0.06)
            or (c.family == "rgb" and 0.25 <= surface <= 0.40 and now["speed"] < 0.06 and yaw_speed < 0.15)
        ) and upright
        self.hold = self.hold + 1 if ready else 0
        self.max_hold = max(self.max_hold, self.hold)
        achieved = {
            "goto": self.hold >= 100,
            "rgb": self.hold >= 100,
            "explore": displacement >= c.distance_m,
            "spin": self.yaw_total * c.direction >= 2 * np.pi * c.turns and self.max_displacement < 0.25,
            "circle": self.orbit_total * c.direction >= 2 * np.pi * c.turns and self.radius_error / self.tick < 0.15,
            "ball": ball_motion >= c.distance_m and self.ball_contact,
        }[c.family]
        self.success = bool(achieved and not self.fallen and not self.collision)
        self.done = self.success or self.fallen or self.collision or self.tick >= round(c.duration_s * 50)
        self.rows.append([self.tick * 0.02, *now["xy"], now["yaw"], now["height"], surface, distance, ball_motion, now["speed"]])
        self.previous = now

    def result(self):
        c, now = self.course, self.previous
        displacement = float(np.linalg.norm(now["xy"] - self.initial["xy"]))
        distance = float(np.linalg.norm(now["xy"] - self.layout["waypoints"][min(self.waypoint, c.waypoints - 1)]))
        surface = float(np.linalg.norm(now["xy"] - now["balls"][c.color][:2])) - 0.035
        motion = float(np.linalg.norm(now["balls"]["red"][:2] - self.initial["balls"]["red"][:2]))
        initial_distance = float(np.linalg.norm(self.initial["xy"] - self.layout["waypoints"][0]))
        initial_surface = float(np.linalg.norm(self.initial["xy"] - self.initial["balls"][c.color][:2])) - 0.035
        rewards = {
            "goto": initial_distance - distance + self.waypoint + self.max_hold / 100,
            "rgb": abs(initial_surface - 0.325) - abs(surface - 0.325) + self.max_hold / 100,
            "explore": displacement / c.distance_m,
            "spin": self.yaw_total * c.direction / (2 * np.pi * c.turns) - 2 * self.max_displacement,
            "circle": self.orbit_total * c.direction / (2 * np.pi * c.turns) - 2 * self.radius_error / max(1, self.tick),
            "ball": motion / c.distance_m if self.ball_contact else -0.5 * surface,
        }
        return {
            "success": self.success,
            "reward": float(rewards[c.family] + 2 * self.success - 3 * (self.fallen or self.collision)),
            "termination_reason": "success" if self.success else "fall" if self.fallen else "collision" if self.collision else "timeout",
            "tick": self.tick,
            "sim_time_s": self.tick * 0.02,
            "layout": self.layout,
            "metrics": {
                "displacement_m": displacement,
                "surface_distance_m": surface,
                "target_distance_m": distance,
                "ball_displacement_m": motion,
                "ball_contact": self.ball_contact,
                "yaw_rad": self.yaw_total,
                "orbit_rad": self.orbit_total,
                "mean_radius_error_m": self.radius_error / max(1, self.tick),
                "max_hold_s": self.max_hold * 0.02,
            },
        }

    def close(self):
        self.physics.close()
