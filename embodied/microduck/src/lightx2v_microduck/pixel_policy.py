"""Transparent RGB colour + real ToF baseline. No simulator truth access."""

from __future__ import annotations

import hashlib
import math
import re
from collections import deque
from pathlib import Path

import numpy as np

from .contracts import Observation, PolicyManifest, SkillIntent

PIXEL_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def detect_colored_ball(rgb: np.ndarray, color: str, *, all_candidates=False) -> dict | list | None:
    """Find a compact saturated component, rejecting sky/floor-sized regions."""
    image = np.asarray(rgb, dtype=np.float32) / 255.0
    r, g, b = image[..., 0], image[..., 1], image[..., 2]
    if color == "red":
        mask = (r > 1.65 * g) & (r > 1.65 * b) & (r > 0.12)
    elif color == "blue":
        # The official checkerboard itself has B/G ~= 1.5. Require saturated
        # blue so its connected floor regions cannot merge into the ball mask.
        mask = (b > 2.5 * r) & (b > 2.1 * g) & (b > 0.12)
    elif color == "yellow":
        mask = (r > 1.7 * b) & (g > 1.7 * b) & (r > 0.12) & (g > 0.12) & (r < 1.7 * g) & (g < 1.7 * r)
    else:
        raise ValueError(f"unsupported target colour {color!r}")
    height, width = mask.shape
    seen = np.zeros_like(mask)
    candidates = []
    for y, x in zip(*np.nonzero(mask)):
        if seen[y, x]:
            continue
        todo = [(int(y), int(x))]
        seen[y, x] = True
        pixels = []
        while todo:
            cy, cx = todo.pop()
            pixels.append((cy, cx))
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    todo.append((ny, nx))
        if len(pixels) < max(16, round(height * width * 0.0015)):
            continue
        pixels = np.asarray(pixels)
        low, high = pixels.min(axis=0), pixels.max(axis=0)
        h, w = high - low + 1
        if not 0.75 <= w / h <= 1.33 or len(pixels) > height * width * 0.12:
            continue
        if min(low) == 0 or high[0] == height - 1 or high[1] == width - 1:
            continue
        fill = len(pixels) / (h * w)
        if fill < 0.60:
            continue
        candidates.append(
            {"u": float(pixels[:, 1].mean()), "v": float(pixels[:, 0].mean()), "radius_px": float((w + h) / 4), "area_px": len(pixels), "bbox": [int(low[1]), int(low[0]), int(high[1]), int(high[0])]}
        )
    return candidates if all_candidates else max(candidates, key=lambda c: c["area_px"], default=None)


def ball_in_trunk(detection, calibration, *, radius_m=0.035):
    """Monocular known-radius geometry with the public camera extrinsic TF."""
    u, v = detection["u"], detection["v"]
    rotation = int(calibration.get("canonical_rotation_cw", 0))
    width, height = int(calibration["width"]), int(calibration["height"])
    if rotation == 90:
        u, v = v, height - 1 - u
    elif rotation == 180:
        u, v = width - 1 - u, height - 1 - v
    elif rotation == 270:
        u, v = width - 1 - v, u
    elif rotation != 0:
        raise ValueError("unsupported image rotation")
    fx, fy = float(calibration["fx"]), float(calibration["fy"])
    depth = 0.5 * (fx + fy) * radius_m / detection["radius_px"]
    ray = np.array(((u - calibration["cx"]) / fx, -(v - calibration["cy"]) / fy, -1.0)) * depth
    rot = np.asarray(calibration["camera_to_trunk_rotation"], dtype=float)
    translation = np.asarray(calibration["camera_to_trunk_translation"], dtype=float)
    point = translation + rot @ ray
    if rot.shape != (3, 3) or translation.shape != (3,) or not np.isfinite(point).all():
        raise ValueError("invalid camera extrinsic calibration")
    return point


class PixelRulePolicy:
    def __init__(self, *, profile="official_bam", target_color=None, use_tof=True):
        self.profile, self.configured_color, self.use_tof = profile, target_color, bool(use_tof)
        self.manifest = PolicyManifest(
            "pixel_color_tof_fsm" if use_tof else "pixel_color_fsm",
            "skill_intent",
            "rgb_proprio_tof" if use_tof else "rgb_proprio",
            profile,
            horizon=5,
            execution_prefix=5,
            uses_privileged_state=False,
            metadata={"ball_radius_prior_m": 0.035, "learned": False, "source_sha256": PIXEL_SOURCE_SHA256, "controller_revision": "latched_stop_v2"},
        )
        self.reset("")

    def reset(self, episode_id=""):
        self.episode_id = str(episode_id)
        self.phase = "search"
        self._avoid_until = -1
        self._detour_until = -1
        self._avoid_direction = 1.0
        self._distances = deque(maxlen=3)
        self._head_pitch = 0.0
        self._stop_latched = False
        self._last_target_tick = -10000
        self._last_distance = math.inf
        self._bearing_estimate = 0.0
        self._previous_time = None
        self._previous_yaw_rate = 0.0
        self.last_diagnostics = {}

    def _color(self, instruction):
        if self.configured_color:
            return self.configured_color
        match = re.search(r"\b(red|blue|yellow)\b", instruction.lower())
        if match:
            return match.group(1)
        for word, color in (("红", "red"), ("蓝", "blue"), ("黄", "yellow")):
            if word in instruction:
                return color
        return None

    def detect(self, observation, color):
        return detect_colored_ball(observation.rgb, color)

    def predict(self, observation: Observation):
        if observation.episode_id != self.episode_id:
            self.reset(observation.episode_id)
        if self._previous_time is not None:
            dt = np.clip(observation.sim_time_s - self._previous_time, 0.0, 0.25)
            self._bearing_estimate -= 0.5 * (self._previous_yaw_rate + float(observation.state[2])) * dt
            self._bearing_estimate = math.atan2(math.sin(self._bearing_estimate), math.cos(self._bearing_estimate))
        self._previous_time = observation.sim_time_s
        self._previous_yaw_rate = float(observation.state[2])
        cmd = np.zeros(13, dtype=np.float32)
        color = self._color(observation.instruction)
        role = "stand"
        detection = None
        diagnostics = {"uses_privileged_state": False, "target_color": color}
        if self._stop_latched:
            # The instruction ends in "stop and wait". Losing the ball or
            # seeing its close-range ToF return cannot authorize new motion.
            self.phase = "hold"
        elif color is None or observation.rgb is None:
            self.phase = "missing_target_or_rgb"
        elif self.use_tof and (observation.tof is None or not observation.tof_metadata.get("available", False) or observation.tof_metadata.get("frame_age_s", math.inf) > 0.25):
            self.phase = "missing_or_stale_tof"
        else:
            detection = self.detect(observation, color)
            point, distance, bearing, filtered_distance = None, math.inf, 0.0, math.inf
            if detection is not None:
                try:
                    point = ball_in_trunk(detection, observation.camera_metadata)
                    distance = float(np.linalg.norm(point[:2])) - 0.035
                    bearing = float(math.atan2(point[1], point[0]))
                    self._distances.append(distance)
                    filtered_distance = float(np.median(self._distances))
                    self._last_target_tick, self._last_distance = observation.tick, filtered_distance
                    self._bearing_estimate = bearing
                    self._head_pitch = float(np.clip((0.8 - filtered_distance) * 0.6, 0.0, 0.3))
                    diagnostics.update(detection=detection, estimated_ball_surface_distance_m=distance, target_bearing_rad=bearing)
                except (KeyError, ValueError, FloatingPointError):
                    detection = None
                    diagnostics["camera_calibration_invalid"] = True
            nearest = 4.0
            if self.use_tof:
                status = np.asarray(observation.tof_metadata.get("statuses", np.zeros((8, 8))), dtype=np.uint8)
                distances = np.where(status == 5, observation.tof, 4.0)
                nearest = float(np.min(distances[1:6, 2:6]))
                left, right = float(np.mean(distances[1:6, :3])), float(np.mean(distances[1:6, 5:]))
                if nearest < 0.30 and observation.tick >= self._avoid_until:
                    self._avoid_direction = 1.0 if left >= right else -1.0
                    self._avoid_until = observation.tick + 60
                    self._detour_until = observation.tick + 140
            diagnostics["front_range_m"] = nearest
            if detection is not None and filtered_distance <= 0.35:
                self._stop_latched, self.phase = True, "hold"
            elif detection is None and self._last_distance < 0.6 and observation.tick - self._last_target_tick < 75:
                # Pause and lower the head to reacquire a nearby ball instead
                # of initiating a blind yaw sweep as it leaves the lower FOV.
                self.phase, self._head_pitch = "reacquire", 0.3
            elif observation.tick < self._avoid_until:
                role, self.phase = "walk", "avoid"
                cmd[2] = self._avoid_direction * 1.25
            elif observation.tick < self._detour_until:
                role, self.phase = "walk", "detour"
                cmd[0] = 0.3 if nearest > 0.32 else 0.0
                cmd[2] = self._avoid_direction * 0.3
            elif detection is None:
                role, self.phase = "walk", "search"
                # The pinned alpha gait remains nearly stationary at yaw=.9;
                # this command is calibrated against its actual BAM response.
                remembered = observation.tick - self._last_target_tick < 250
                cmd[2] = (1.1 if self._bearing_estimate >= 0 else -1.1) if remembered else 1.1
                self._distances.clear()
            else:
                role, self.phase = "walk", "approach"
                cmd[2] = np.clip(bearing * 2.0, -1.1, 1.1)
                cmd[0] = 0.3 if abs(bearing) < 0.65 and nearest > 0.30 else 0.0
        diagnostics["phase"] = self.phase
        diagnostics["stop_latched"] = self._stop_latched
        cmd[4] = self._head_pitch
        self.last_diagnostics = diagnostics
        return SkillIntent(role, cmd, observation.episode_id, observation.tick, 5)
