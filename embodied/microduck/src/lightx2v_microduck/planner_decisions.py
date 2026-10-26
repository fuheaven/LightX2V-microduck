"""Explicit model observations and task stages, never evaluator truth.

The model still chooses the tool and duration. Its reported perception is
recorded for diagnosis and subsequent feedback, not used as an execution oracle.
"""

from copy import deepcopy

from .planner_tools import DECISION_TOOL_PROFILES, decode_tool_call


DECISION_PROTOCOL = "microduck_visual_decision_v1"
DECISION_PROFILE, COMPACT_DECISION_PROFILE, MEASURED_DECISION_PROFILE = DECISION_TOOL_PROFILES
COMPACT_LAYOUT_ID = "single_line_memory_five_words_v1"
# The formatting-only experiment keeps semantic fields, actor inputs and tool
# execution unchanged. The word limit is a prompt, not an output repair rule.
COMPACT_LAYOUT_RULE = (
    " Output layout requirement: emit ONE compact JSON line, with no newline or indentation"
    " and no spaces outside strings. Keep memory to at most five words."
    " Preserve all required fields and their meanings."
)


def decode_decision_call(value):
    if not isinstance(value, dict) or set(value) != {"target", "stage", "tool", "duration_s", "memory"}:
        raise ValueError("visual decision requires target, stage, tool, duration_s and memory")
    target = value["target"]
    if (not isinstance(target, dict) or set(target) != {"visible", "image_side"}
            or type(target["visible"]) is not bool
            or target["image_side"] not in ("left", "center", "right", "absent")
            or target["visible"] != (target["image_side"] != "absent")):
        raise ValueError("inconsistent model-reported target visibility or image side")
    if value["stage"] not in ("search", "align", "approach", "stop"):
        raise ValueError("unknown model-reported task stage")
    result = decode_tool_call({k: value[k] for k in ("tool", "duration_s", "memory")})
    result.update(decision_call=deepcopy(value), decision_protocol=DECISION_PROTOCOL,
                  model_assessment={"target": deepcopy(target), "stage": value["stage"],
                                    "provenance": "model_hypothesis_not_verified"})
    return result


def decision_prompt(mode):
    return (
        "Control MicroDuck for the current visual task. First report what the CURRENT camera actually shows, "
        "then choose the task stage and one bounded motion. Return compact JSON with exactly: "
        '"target":{"visible":true/false,"image_side":"left|center|right|absent"}, '
        '"stage":"search|align|approach|stop", "tool":"...", "duration_s":number, "memory":"...". '
        "Use absent exactly when the requested target is not visible. A different colored ball is not the target. "
        "Your report is a hypothesis, not verified success. Memory is a brief uncertain note, at most 160 characters, "
        "about the last target observation and what changed after acting. "
        "Tools: advance requests forward walking at 0.30m/s with head centered. "
        "turn_left/turn_right turn the BODY along a forward arc at vx=0.15m/s, yaw=+/-1.1rad/s, "
        "with head centered; they require clearance. look_left/look_right turn only the HEAD to +/-0.45rad "
        "while standing. stop stands with head centered. Commands do not guarantee displacement. "
        "Choose duration_s between 0.1 and 2.0 seconds in multiples of 0.02. "
        "Image left/right follows the head camera; looking at a target does not align the body with it. "
        "Named joint readings describe the head relative to the body, not global target coordinates. "
        "Use current visual evidence, named sensor feedback and available history to choose search, alignment, "
        "approach or stopping. Reobserve after each bounded action. Do not infer off-camera coordinates. "
        "A command receipt is not proof of motion or task completion. No extra arguments, prose or Markdown. "
        + ("Your memory and earlier model assessments are retained as hypotheses for the next decision."
           if mode == "feedback" else "This ablation retains no history or memory for the next decision.")
    )
