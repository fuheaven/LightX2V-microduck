"""Short-window heading-change evidence from actor IMU observations only.

This estimates relative body heading, not global pose, translation, clearance,
or task completion. The official gyro and projected gravity must share the
trunk frame; hardware use requires its own sensor calibration. No simulator or
evaluator object can enter the estimator.
"""

import hashlib
import json
import math

import numpy as np


MOTION_VERSION = "microduck_actor_heading_change_v1"
MOTION_CONTRACT = {
    "version": MOTION_VERSION,
    "source": "state48_gyro3_and_projected_gravity3",
    "gyro_and_gravity_frame": "trunk",
    "timebase": "consecutive_50hz_control_ticks",
    "integration": "trapezoid_of_zyx_heading_rate",
    "minimum_down_gravity_component": 0.58,
    "maximum_window_ticks": 100,
    "duplicate_same_tick": "idempotent_if_sensor_bytes_equal",
    "missing_or_changed_samples": "invalid_until_next_window",
    "world_pose_available": False,
    "translation_available": False,
    "hardware_calibrated": False,
}
MOTION_CONTRACT_SHA256 = hashlib.sha256(json.dumps(
    MOTION_CONTRACT, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def heading_rate(state):
    """ZYX heading derivative from body gyro and downward unit gravity.

    g = (sin(pitch), -sin(roll)*cos(pitch), -cos(roll)*cos(pitch)).
    Therefore yaw_dot = -(gyro_y*g_y + gyro_z*g_z)/(g_y**2+g_z**2).
    This accounts for body tilt; integrating gyro_z alone does not.
    """
    state = np.asarray(state, dtype=np.float64)
    if state.ndim != 1 or len(state) not in (48, 61) or not np.isfinite(state[:6]).all():
        raise ValueError("invalid_actor_imu_sample")
    gyro, gravity = state[:3], state[3:6]
    norm = float(np.linalg.norm(gravity))
    if not 0.9 <= norm <= 1.1:
        raise ValueError("invalid_projected_gravity_norm")
    gravity = gravity / norm
    if -gravity[2] < MOTION_CONTRACT["minimum_down_gravity_component"]:
        raise ValueError("body_tilt_outside_heading_estimator_contract")
    rate = float(-np.dot(gyro[1:], gravity[1:]) / np.dot(gravity[1:], gravity[1:]))
    if not math.isfinite(rate):
        raise ValueError("nonfinite_heading_rate")
    return rate


class ActorHeadingChange:
    """One action window. Missing samples cannot be silently interpolated."""

    def __init__(self, observation):
        self.episode_id = observation.episode_id
        self.started_tick = observation.tick
        self.last_tick = None
        self._last_sensor = None
        self._last_rate = None
        self._angle = 0.0
        self.samples = 0
        self.error = None
        self.observe(observation)

    def observe(self, observation):
        if self.error is not None:
            return
        if not self.episode_id or observation.episode_id != self.episode_id:
            self.error = "episode_mismatch"
            return
        tick = observation.tick
        if type(tick) is not int or tick < 0:
            self.error = "invalid_sensor_tick"
            return
        if tick - self.started_tick > MOTION_CONTRACT["maximum_window_ticks"]:
            self.error = "window_exceeds_validated_duration"
            return
        try:
            rate = heading_rate(observation.state)
        except (TypeError, ValueError) as error:
            self.error = str(error)
            return
        sensor = np.asarray(observation.state[:6]).copy()
        if self.last_tick is not None:
            if tick == self.last_tick:
                if not np.array_equal(sensor, self._last_sensor):
                    self.error = "same_tick_sensor_changed"
                return
            if tick != self.last_tick + 1:
                self.error = "missing_or_out_of_order_sensor_tick"
                return
            updated = self._angle + (0.5 * self._last_rate + 0.5 * rate) / 50.0
            if not math.isfinite(updated):
                self.error = "nonfinite_integrated_heading"
                return
            self._angle = updated
        self.last_tick, self._last_rate, self._last_sensor = tick, rate, sensor
        self.samples += 1

    def snapshot(self):
        valid = self.error is None and self.samples >= 2
        return {
            "version": MOTION_VERSION, "contract_sha256": MOTION_CONTRACT_SHA256,
            "source": MOTION_CONTRACT["source"],
            "started_tick": self.started_tick, "last_sensor_tick": self.last_tick,
            "sample_count": self.samples, "available": valid,
            "reason": self.error or (None if valid else "awaiting_next_sensor_tick"),
            "measured_duration_s": (self.last_tick - self.started_tick) / 50.0 if self.last_tick is not None else 0.0,
            "estimated_body_heading_change_rad": self._angle if valid else None,
            "estimated_body_heading_change_degrees": math.degrees(self._angle) if valid else None,
            "interpretation": "relative_inertial_estimate_not_verified_displacement_or_task_success",
        }
