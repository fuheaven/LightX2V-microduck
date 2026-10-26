"""Explicit primitive timing matched to the existing observation-age budget.

This opt-in contract changes neither action values nor executor expiry. A
model must choose an admissible duration; invalid calls are never clipped.
Learned skills still produce fresh, separately timestamped action chunks.
"""
from copy import deepcopy
import math

from .planner_tools import TOOL_ACTIONS


PRIMITIVE_TIMING_PROTOCOL = "microduck_mission_primitive_timing_v1"


def primitive_timing_contract(max_duration_ms):
    if (type(max_duration_ms) is not int or not 100 <= max_duration_ms <= 2000
            or max_duration_ms % 100):
        raise ValueError("primitive duration limit must be 100..2000 ms in 100 ms increments")
    return dict(protocol=PRIMITIVE_TIMING_PROTOCOL, maximum_duration_ms=max_duration_ms,
                decision_interval_ms=100, source_clock="original_observation_preserved",
                duration_overflow="reject_without_execution_or_clipping")


def validate_primitive_timing(call, contract):
    # Caller first applies the complete tool/claim schema. Claims and explicit
    # goal selection have their own fixed 0.1s duration; they are not motion.
    if contract is not None and call["tool"] in TOOL_ACTIONS:
        maximum = contract["maximum_duration_ms"] / 1000
        duration = call["duration_s"]
        if duration > maximum or not math.isclose(duration * 10, round(duration * 10), abs_tol=1e-7):
            raise ValueError(f"primitive duration must be 0.1..{maximum:g} seconds in 0.1 second increments; "
                             "choose a new duration, then observe again before further motion")
    return call


def with_primitive_timing(payload, contract, *, learned_skills, registry_sha256):
    if contract is None:
        return payload
    result = deepcopy(payload)
    system = result["messages"][0]["content"]
    # Rewrite the known declaration instead of appending a contradictory rule.
    original = ("Duration is 0.1..2.0 seconds in multiples of 0.02. After this interval you observe again. "
                if learned_skills else "Duration is 0.1..2.0 seconds in multiples of 0.02 seconds. ")
    if system.count(original) != 1:
        raise ValueError("primitive timing requires the supported original duration declaration")
    maximum = contract["maximum_duration_ms"] / 1000
    timing = (f"Primitive tools advance, turn_left, turn_right, look_left, look_right and stop use "
              f"duration_s=0.1..{maximum:g} seconds in 0.1 second increments. "
              "Choose another action from a new observation when that duration ends. "
              "The source observation is never refreshed to keep an old command alive. ")
    if learned_skills:
        timing += ("Learned navigate_COLOR skills retain 0.1..2.0 second windows in 0.02 second increments; "
                   "their internal actions use fresh camera observations during that window. ")
    else:
        system = system.replace('{"tool":"look_left","duration_s":1.0}',
                                '{"tool":"look_left","duration_s":0.1}')
    result["messages"][0]["content"] = system.replace(original, timing)
    result["metadata"].update(primitive_timing_contract=deepcopy(contract),
                               tool_contract_sha256=registry_sha256)
    return result
