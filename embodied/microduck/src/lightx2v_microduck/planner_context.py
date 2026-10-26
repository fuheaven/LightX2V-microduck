"""Bounded, actor-only context for a robot tool planner.

Command receipts use the next observation's raw-action history. They establish
which joint command was consumed, not displacement, contact or task success.
No environment object, evaluator, object pose or teacher enters this module.
"""

from collections import deque
from copy import deepcopy

import numpy as np


CONTEXT_VERSION = "microduck_planner_feedback_v1"
CONTEXT_MODES = ("current_frame", "feedback")


def camera_tick(observation):
    tick = observation.tick if observation.camera_tick is None else observation.camera_tick
    if not 0 <= tick <= observation.tick:
        raise ValueError("camera tick must not be negative or in the future")
    return int(tick)


def capture_monotonic_ns(observation):
    captured = observation.camera_metadata.get("capture_monotonic_s")
    if captured is not None:
        if not np.isfinite(captured) or captured <= 0:
            raise ValueError("invalid camera capture timestamp")
        return int(captured * 1e9)
    return observation.monotonic_ns


class PlannerContext:
    def __init__(self, history_limit=4, *, measured_heading=False):
        self.history_limit = history_limit
        self.measured_heading = bool(measured_heading)
        self.reset("")

    def reset(self, episode_id):
        self.episode_id = episode_id
        self.history = deque(maxlen=self.history_limit)
        self.events = deque(maxlen=self.history_limit)
        self.active = None
        self.pending_receipt = None
        self.reference = None
        self.motion = None
        self.memory = {"text": "", "source_tick": None, "provenance": "model_hypothesis"}

    def observe(self, observation):
        if observation.episode_id != self.episode_id:
            raise ValueError("context belongs to a different episode")
        if self.pending_receipt is not None:
            tick, raw, record = self.pending_receipt
            if observation.tick > tick:
                if observation.tick != tick + 1:
                    record["receipt_unknown_ticks"] += 1
                elif np.array_equal(observation.state[34:48], raw):
                    record["raw_history_confirmed_ticks"] += 1
                else:
                    record["raw_history_mismatch_ticks"] += 1
                self.pending_receipt = None
        if self.active is not None:
            if self.motion is not None:
                self.motion.observe(observation)
                self.active["observed_body_heading_change"] = self.motion.snapshot()
            self.active["last_observed_tick"] = observation.tick
            if self.reference is not None:
                self.active["joint_offset_change_l2"] = float(np.linalg.norm(
                    observation.state[6:20] - self.reference.state[6:20]))
                if observation.rgb is not None and observation.rgb.shape == self.reference.rgb.shape:
                    self.active["rgb_mean_absolute_change_0_1"] = float(np.mean(np.abs(
                        observation.rgb.astype(np.float32) - self.reference.rgb.astype(np.float32))) / 255)

    def begin(self, plan, source_observation, execution_observation):
        if (source_observation.episode_id != self.episode_id
                or execution_observation.episode_id != self.episode_id
                or source_observation.tick > execution_observation.tick):
            raise ValueError("action context needs an ordered source in the same episode")
        self.finish("superseded", execution_observation.tick)
        self.reference = execution_observation
        self.active = {
            "skill": plan["skill"], "command_mps_rps": list(plan["command"]),
            "head_yaw_rad": plan.get("head_yaw_rad", 0.0),
            "requested_duration_s": plan["duration_s"],
            "source_tick": source_observation.tick,
            "source_camera_tick": camera_tick(source_observation),
            "started_tick": execution_observation.tick,
            "last_observed_tick": execution_observation.tick,
            "status": "running", "issued_ticks": 0,
            "raw_history_confirmed_ticks": 0, "raw_history_mismatch_ticks": 0,
            "receipt_unknown_ticks": 0,
            "outcome": "not_inferred_from_command_or_image_change",
        }
        if self.measured_heading:
            from .planner_motion import ActorHeadingChange

            self.motion = ActorHeadingChange(execution_observation)
            self.active["observed_body_heading_change"] = self.motion.snapshot()
        if "tool_call" in plan:
            self.active.update(tool_call=deepcopy(plan["tool_call"]),
                               tool_protocol=plan["tool_protocol"],
                               tool_contract_sha256=plan["tool_contract_sha256"])
        if "model_assessment" in plan:
            self.active.update(model_assessment=deepcopy(plan["model_assessment"]),
                               decision_protocol=plan["decision_protocol"])
        if "memory" in plan:
            self.memory = {"text": plan["memory"], "source_tick": source_observation.tick,
                           "provenance": "model_hypothesis"}

    def issued(self, observation, raw):
        if self.active is not None:
            self.active["issued_ticks"] += 1
            self.pending_receipt = (observation.tick, np.array(raw, copy=True), self.active)

    def finish(self, status, tick):
        if self.active is not None:
            self.active.update(status=status, ended_tick=tick)
            self.history.append(self.active)
            self.active = None
            self.motion = None

    def failure(self, reason, tick, source_tick=None):
        self.events.append({"reason": reason, "observed_tick": tick, "source_tick": source_tick})

    def snapshot(self, observation, mode):
        if mode not in CONTEXT_MODES:
            raise ValueError("unknown planner context mode")
        if observation.episode_id != self.episode_id:
            raise ValueError("context belongs to a different episode")
        # command13 is intentionally excluded: it may contain a task program or
        # privileged teacher answer. Images are immutable Observation copies.
        context = {
            "schema": CONTEXT_VERSION, "mode": mode,
            "instruction": observation.instruction, "tick": observation.tick,
            "camera_tick": camera_tick(observation),
            "state48": observation.state[:48].tolist(),
            "state_layout": "gyro3,gravity3,joint_offset14,joint_velocity14,previous_raw_action14",
        }
        previous = None
        if mode == "feedback":
            history = [*self.history, *([self.active] if self.active is not None else [])]
            context.update(recent_actions=deepcopy(history[-self.history_limit:]),
                           execution_events=deepcopy(list(self.events)),
                           task_memory=deepcopy(self.memory))
            if self.reference is not None and camera_tick(self.reference) < camera_tick(observation):
                previous = self.reference
            if self.measured_heading:
                from .planner_motion import MOTION_CONTRACT_SHA256, MOTION_VERSION

                context.update(motion_feedback_version=MOTION_VERSION,
                               motion_feedback_contract_sha256=MOTION_CONTRACT_SHA256)
        return context, previous
