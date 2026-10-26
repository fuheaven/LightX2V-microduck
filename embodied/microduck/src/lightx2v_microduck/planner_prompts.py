"""Versioned, actor-only prompt presentation; never repairs model actions."""

from copy import deepcopy
import json

from .planner_tools import ACTOR_TOOL_PROFILES, DECISION_TOOL_PROFILES, MEASURED_DECISION_PROFILE


PROMPT_PROFILES = ("legacy_v1", "task_first_v2", "calibrated_tools_v3", *ACTOR_TOOL_PROFILES)


def with_visual_decision(named_payload, *, compact=False, measured_motion=False):
    from .planner_decisions import (
        COMPACT_DECISION_PROFILE, COMPACT_LAYOUT_ID, COMPACT_LAYOUT_RULE,
        DECISION_PROFILE, DECISION_PROTOCOL, decision_prompt,
    )

    if named_payload.get("metadata", {}).get("prompt_profile") != "bounded_tools_named_sensors_v2":
        raise ValueError("visual decisions require the named actor input")
    result = deepcopy(named_payload)
    context = json.loads(result["messages"][-1]["content"][1]["text"])
    result["messages"][0]["content"] = decision_prompt(context["mode"])
    result["max_tokens"] = 160
    result["metadata"].update(prompt_profile=DECISION_PROFILE, decision_protocol=DECISION_PROTOCOL)
    if compact:
        result["messages"][0]["content"] += COMPACT_LAYOUT_RULE
        result["max_tokens"] = 96
        result["metadata"].update(prompt_profile=COMPACT_DECISION_PROFILE,
                                  diagnostic_output_layout=COMPACT_LAYOUT_ID)
    if measured_motion:
        from .planner_motion import MOTION_CONTRACT_SHA256, MOTION_VERSION

        if (not compact or context.get("mode") != "feedback"
                or context.get("motion_feedback_contract_sha256") != MOTION_CONTRACT_SHA256
                or context.get("motion_feedback_version") != MOTION_VERSION):
            raise ValueError("measured visual decisions require declared actor motion feedback")
        result["messages"][0]["content"] += (
            " Recent actions include observed_body_heading_change from consecutive gyro/gravity samples."
            " When available=true, estimated_body_heading_change_degrees estimates the BODY's actual"
            " relative heading change: positive is left and negative is right. It is an inertial estimate,"
            " not the requested yaw rate or a global pose. When available=false the effect is unknown;"
            " do not infer success from the command receipt. A short turn can initially retain prior motion."
            " Compare measured heading change, head encoders and before/after images before choosing"
            " another action. This feedback does not measure translation, obstacle clearance, target"
            " distance or task completion."
        )
        result["metadata"].update(prompt_profile=MEASURED_DECISION_PROFILE,
                                  motion_feedback_version=MOTION_VERSION,
                                  motion_feedback_contract_sha256=MOTION_CONTRACT_SHA256)
    return result


def with_named_sensors(tool_payload):
    """Add sensor semantics to a recorded tool input without changing actions."""
    from .planner_sensors import NAMED_SENSOR_CONTRACT_SHA256, NAMED_SENSOR_VERSION, name_actor_sensors

    if tool_payload.get("metadata", {}).get("prompt_profile") != "bounded_tools_v1":
        raise ValueError("named sensors require an unmodified bounded-tools input")
    result = deepcopy(tool_payload)
    content = result["messages"][-1]["content"]
    context = json.loads(content[1]["text"])
    if "named_proprioception" in context:
        raise ValueError("named actor sensors have already been added")
    context["named_proprioception"] = name_actor_sensors(context)
    content[1]["text"] = json.dumps(context, allow_nan=False)
    result["messages"][0]["content"] += (
        " The named_proprioception field translates the existing encoder observations."
        " Positive head_yaw means the head joint turns left relative to the body;"
        " a head-only look does not align the body with the viewed target."
        " advance, turn_left, turn_right and stop request a centered head."
        " Use the visible target and head-joint feedback to decide whether body alignment is needed before advancing."
        " Joint angles are not a calibrated camera pose or a target bearing."
        " If image_and_sensor_same_tick is false, the named angles are newer than the image."
    )
    result["metadata"].update(prompt_profile="bounded_tools_named_sensors_v2",
                              named_sensor_version=NAMED_SENSOR_VERSION,
                              named_sensor_contract_sha256=NAMED_SENSOR_CONTRACT_SHA256)
    return result


def with_calibrated_tools(task_first_payload):
    """Extend an existing paired input without repeating task/context text."""
    from .navigation_tool_response import CARD_SHA256, load_tool_response

    if task_first_payload.get("metadata", {}).get("prompt_profile") != "task_first_v2":
        raise ValueError("calibrated description requires an unmodified task-first input")
    card = load_tool_response()
    result = deepcopy(task_first_payload)
    result["messages"][0]["content"] += "\nMeasured locomotion tool behavior:\n" + "\n".join(card["actor_notes"])
    result["metadata"].update(prompt_profile="calibrated_tools_v3", tool_response_sha256=CARD_SHA256,
                              tool_response_report_sha256=card["report_sha256"])
    return result


def present_task(payload, profile):
    if profile not in PROMPT_PROFILES:
        raise ValueError("unknown planner prompt profile")
    if profile == "legacy_v1":
        return payload
    if profile in DECISION_TOOL_PROFILES:
        return with_visual_decision(present_task(payload, "bounded_tools_named_sensors_v2"),
                                    compact=profile in ("perception_action_compact_v4", MEASURED_DECISION_PROFILE),
                                    measured_motion=profile == MEASURED_DECISION_PROFILE)
    if profile == "bounded_tools_named_sensors_v2":
        return with_named_sensors(present_task(payload, "bounded_tools_v1"))
    if profile == "bounded_tools_v1":
        from .planner_tools import TOOL_CONTRACT_SHA256, TOOL_PROTOCOL, tool_prompt

        result = deepcopy(payload)
        content = result["messages"][-1]["content"]
        context = json.loads(content[0]["text"])
        if context.get("mode") not in ("current_frame", "feedback") or not context.get("instruction"):
            raise ValueError("bounded tools require actor context and instruction")
        result["messages"][0]["content"] = tool_prompt(context["mode"])
        # The actual sensor/context JSON and images stay byte-identical. The
        # new tool interface is a declared system variant, not a prompt ablation.
        content.insert(0, {"type": "text", "text": "Current task: " + context["instruction"]})
        result["max_tokens"] = 96
        result["metadata"].update(prompt_profile=profile, tool_protocol=TOOL_PROTOCOL,
                                  tool_contract_sha256=TOOL_CONTRACT_SHA256)
        return result
    if profile == "calibrated_tools_v3":
        return with_calibrated_tools(present_task(payload, "task_first_v2"))
    result = deepcopy(payload)
    content = result["messages"][-1]["content"]
    context = json.loads(content[0]["text"])
    instruction = context.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("task-first presentation requires the actor instruction")
    if context.get("mode") not in ("current_frame", "feedback"):
        raise ValueError("task-first presentation requires a declared context mode")
    # Preserve the original sensor/context JSON and image bytes. Repeating the
    # user's goal adds no target position, teacher decision or evaluator truth.
    content.insert(0, {"type": "text", "text": "Current task: " + instruction})
    content.append({"type": "text", "text": "Choose the next action for this task: " + instruction})
    result["messages"][0]["content"] += (
        " Follow the current task's requested target; a visible object of a different color is not a substitute."
        " When the requested target is outside the image, use bounded observation/search motions and observe again;"
        " do not invent its location or claim it is visible."
        " Output one compact JSON object without Markdown or code fences."
        " Include skill, command, head_yaw_rad and duration_s. Omit reason."
        + (" Optional memory must be brief (at most 12 words) and express uncertainty when needed."
           if context["mode"] == "feedback" else " Omit memory in current_frame mode.")
    )
    result["max_tokens"] = 128
    result["metadata"]["prompt_profile"] = profile
    return result
