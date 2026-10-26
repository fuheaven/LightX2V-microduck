"""Turn a rejected model tool call into bounded, identity-bound feedback.

This compiler never executes or replaces an action, refreshes observation
timestamps, retries transport errors, or selects a tool. The caller retains its
normal cancellation, source-age, request-budget and execution checks. A new
model response must pass the original validator before execution.
"""

from copy import deepcopy
import hashlib
import json
import re

from .planner_response import JSON_ONLY, JSON_OR_FENCE, decode_framed_object


VALIDATION_FEEDBACK_PROTOCOL = "microduck_tool_validation_feedback_v1"


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _source(payload):
    meta = payload["metadata"]
    episode = meta.get("episode_id")
    if not isinstance(episode, str) or not episode:
        raise ValueError("tool feedback requires an episode")
    values = []
    for key in ("source_tick", "camera_tick"):
        value = meta.get(key)
        if type(value) is int and value >= 0:
            values.append(value)
        elif isinstance(value, str) and re.fullmatch(r"0|[1-9][0-9]*", value):
            values.append(int(value))
        else:
            raise ValueError("tool feedback requires nonnegative source ticks")
    if values[1] > values[0]:
        raise ValueError("tool feedback camera cannot be from the future")
    return episode, *values


def rejected_tool_record(payload, content, evidence, validator, *, response_framing=JSON_ONLY):
    """Record an actual schema rejection, retaining the exact model text.

    Transport/identity errors and validator bugs are not recoverable tool
    rejections. The transport must already have authenticated the service.
    """
    episode, source_tick, camera_tick = _source(payload)
    if not isinstance(content, str) or len(content) > 4096:
        raise ValueError("rejected tool content must be bounded text")
    if (not isinstance(evidence, dict) or evidence.get("request_sha256") != _digest(payload)
            or evidence.get("model_id") != payload.get("model")
            or not re.fullmatch(r"[0-9a-f]{32}", str(evidence.get("request_id", "")))
            or not re.fullmatch(r"[0-9a-f]{64}", str(evidence.get("identity_sha256", "")))):
        raise ValueError("rejected tool response is bound to another request")
    if response_framing == JSON_ONLY:
        value = json.loads(content)
    elif response_framing == JSON_OR_FENCE:
        value, _ = decode_framed_object(content)
    else:
        raise ValueError("unsupported response framing")
    try:
        validator(value)
    except ValueError as error:
        reason = str(error)[:512]
    else:
        raise ValueError("a valid action cannot be recorded as rejected")
    return {
        "protocol": VALIDATION_FEEDBACK_PROTOCOL,
        "episode_id": episode, "source_tick": source_tick, "camera_tick": camera_tick,
        "model_id": payload["model"], "request_id": evidence["request_id"],
        "request_sha256": evidence["request_sha256"], "identity_sha256": evidence["identity_sha256"],
        "content": content, "response_framing": response_framing,
        "validation_error": reason, "action_executed": False,
    }


def with_validation_feedback(current_payload, rejection):
    """Append one rejected call and its result to the current observation.

    A fresh observation may be supplied, but this function cannot make an old
    observation fresh. The rejected call's source remains separate and visible.
    """
    episode, source_tick, camera_tick = _source(current_payload)
    if (rejection.get("protocol") != VALIDATION_FEEDBACK_PROTOCOL
            or rejection.get("action_executed") is not False
            or rejection.get("model_id") != current_payload.get("model")
            or rejection.get("episode_id") != episode
            or source_tick < rejection["source_tick"] or camera_tick < rejection["camera_tick"]):
        raise ValueError("tool feedback must stay in the same ordered episode and model")
    if "validation_feedback" in current_payload["metadata"]:
        raise ValueError("a correction request cannot recursively add another correction")
    if not 1 <= len(current_payload["messages"]) <= 6:
        raise ValueError("tool feedback exceeds the message budget")
    payload = deepcopy(current_payload)
    feedback = {
        "tool_result": {
            "status": "rejected", "action_executed": False,
            "request_id": rejection["request_id"], "source_tick": rejection["source_tick"],
            "error": rejection["validation_error"],
        },
        "instruction": "Reassess the current observations and choose one valid tool call using the original "
                       "tool definitions. No action from the rejected call was executed. Do not infer "
                       "movement or completion from it. Return only the required JSON.",
    }
    payload["messages"].extend([
        {"role": "assistant", "content": rejection["content"]},
        {"role": "user", "content": json.dumps(feedback, ensure_ascii=False, allow_nan=False)},
    ])
    payload["metadata"]["validation_feedback"] = {
        "protocol": VALIDATION_FEEDBACK_PROTOCOL, "attempt": 1,
        "rejected_request_id": rejection["request_id"], "rejection_sha256": _digest(rejection),
    }
    return payload


def request_with_validation_feedback(payload, transport, validator, *, check_current,
                                     response_framing=JSON_ONLY):
    """Allow one model-authored correction; return every attempt for recording.

    Only a successfully received, parsed but invalid tool call may be corrected.
    Transport, framing, session and freshness failures abort normally. The
    caller's check runs before and after every HTTP request. This function does
    not execute any action or advance/reset an observation's source time.
    """
    trace = []
    current = payload
    try:
        for attempt in range(2):
            check_current()
            row = {"attempt": attempt, "request_sha256": _digest(current),
                   "source_tick": current["metadata"]["source_tick"],
                   "camera_tick": current["metadata"]["camera_tick"],
                   "status": "request_started", "action_executed": False}
            trace.append(row)
            try:
                content, evidence = transport(current)
            except Exception as error:
                failure = getattr(error, "planner_transport_error", None)
                if failure is not None:
                    row["transport_error"] = deepcopy(failure)
                raise
            row.update(raw_content=content, serving_evidence=deepcopy(evidence), status="received")
            check_current()
            if response_framing == JSON_OR_FENCE:
                value, framing = decode_framed_object(content)
                if evidence is not None:
                    evidence = {**evidence, **framing}
            elif response_framing == JSON_ONLY:
                value = json.loads(content)
            else:
                raise ValueError("unsupported response framing")
            # Parsing/transport failures above never become a tool rejection.
            try:
                plan = validator(value)
            except ValueError as error:
                rejection = rejected_tool_record(current, content, evidence, validator,
                                                 response_framing=response_framing)
                row.update(status="tool_rejected", rejection=rejection)
                if attempt == 1:
                    raise
                current = with_validation_feedback(payload, rejection)
            else:
                row.update(status="validated", serving_evidence=deepcopy(evidence))
                return plan, evidence, trace
    except Exception as error:
        # The original exception type remains visible to the existing fallback.
        # Never store this in mutable shared policy state during inference: a
        # reset may have occurred while the HTTP response was in flight.
        error.planner_request_trace = deepcopy(trace)
        raise
