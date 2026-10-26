"""Bounded navigation tools over official locomotion, with no scene access.

These are timed commands, not Cartesian goals or guaranteed displacements.
The model chooses every tool and duration; this module never searches for a
target, reads evaluator truth, or substitutes a navigation decision.
"""

from copy import deepcopy
import hashlib
import json
import math


TOOL_PROTOCOL = "microduck_bounded_navigation_tools_v1"
BOUNDED_TOOL_PROFILES = ("bounded_tools_v1", "bounded_tools_named_sensors_v2")
MEASURED_DECISION_PROFILE = "perception_action_measured_v5"
DECISION_TOOL_PROFILES = ("perception_action_tools_v3", "perception_action_compact_v4", MEASURED_DECISION_PROFILE)
ACTOR_TOOL_PROFILES = (*BOUNDED_TOOL_PROFILES, *DECISION_TOOL_PROFILES)
# [vx m/s, vy m/s, yaw rad/s, head yaw rad, stand]. Turning follows a
# forward arc: a zero-speed yaw command is weak on the official walking actor.
TOOL_ACTIONS = {
    "advance": (.30, 0., 0., 0., 0.),
    "turn_left": (.15, 0., 1.1, 0., 0.),
    "turn_right": (.15, 0., -1.1, 0., 0.),
    "look_left": (0., 0., 0., .45, 1.),
    "look_right": (0., 0., 0., -.45, 1.),
    "stop": (0., 0., 0., 0., 1.),
}
TOOL_CONTRACT = {
    "protocol": TOOL_PROTOCOL, "profile": "official_bam", "embodiment": "legs",
    "actions": TOOL_ACTIONS, "duration_s_bounds": [.1, 2.0],
    "control_hz": 50, "memory_max_chars": 160,
    "effect": "timed_official_commands_not_guaranteed_displacement",
    "truth_access": False,
}
TOOL_CONTRACT_SHA256 = hashlib.sha256(json.dumps(
    TOOL_CONTRACT, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def decode_tool_call(value):
    """Reject unknown tools/arguments; translate exactly once to the old ABI."""
    if not isinstance(value, dict) or set(value) - {"tool", "duration_s", "memory"}:
        raise ValueError("tool call contains unsupported arguments")
    tool, duration = value.get("tool"), value.get("duration_s")
    if not isinstance(tool, str) or tool not in TOOL_ACTIONS:
        raise ValueError("unknown navigation tool")
    if (isinstance(duration, bool) or not isinstance(duration, (int, float))
            or not math.isfinite(duration) or not .1 <= duration <= 2.0):
        raise ValueError("tool duration must be a number in 0.1..2.0 seconds")
    # Avoid a rounded duration silently differing from the submitted command.
    if not math.isclose(duration * 50, round(duration * 50), abs_tol=1e-7):
        raise ValueError("tool duration must be a multiple of the 20ms control tick")
    if "memory" in value and (not isinstance(value["memory"], str) or len(value["memory"]) > 160):
        raise ValueError("tool memory must be at most 160 characters")
    vx, vy, yaw, head, stand = TOOL_ACTIONS[tool]
    result = {"skill": "stand" if stand else "walk", "command": [vx, vy, yaw],
              "head_yaw_rad": head, "duration_s": float(duration),
              "tool_call": deepcopy(value), "tool_protocol": TOOL_PROTOCOL,
              "tool_contract_sha256": TOOL_CONTRACT_SHA256}
    if "memory" in value:
        result["memory"] = value["memory"]
    return result


def decode_actor_reply(value, profile):
    if profile in DECISION_TOOL_PROFILES:
        from .planner_decisions import decode_decision_call

        return decode_decision_call(value)
    if profile not in BOUNDED_TOOL_PROFILES:
        raise ValueError("unknown actor tool profile")
    return decode_tool_call(value)


def tool_prompt(context_mode):
    memory = (" Optional memory: a short uncertain task note of at most 160 characters."
              if context_mode == "feedback" else " Omit memory.")
    return (
        "Control a MicroDuck using its current camera, task and sensor observations. "
        "Choose exactly one tool and its duration. Return only compact JSON, for example "
        '{"tool":"look_left","duration_s":1.0}. '
        "Tools: advance walks forward; turn_left and turn_right turn the BODY while moving "
        "slightly forward along an arc (they require clearance); look_left and look_right "
        "turn only the HEAD while standing; stop stands and centers the head. "
        "advance requests vx=0.30m/s; body turns request vx=0.15m/s and yaw=+/-1.1rad/s; "
        "head looks request +/-0.45rad. These are commands, not guaranteed motion. "
        "Duration is 0.1..2.0 seconds in multiples of 0.02 seconds. "
        "Use the requested target, not a different visible object. If it is outside the "
        "image, choose an observation or search motion and observe again; its location is unknown. "
        "Image left/right are relative to the current camera, which may be turned from the body. "
        "Compare before/after images and actual sensor feedback to decide whether an action helped. "
        "A command receipt does not establish successful movement, clearance or task completion. "
        "Do not add velocity arguments, explanations or Markdown."
        + memory
    )
