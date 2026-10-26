"""Explicit response framing for robot tools, without semantic output repair."""

import hashlib
import json
import math
import re


JSON_ONLY = "json_object_v1"
JSON_OR_FENCE = "json_object_or_fenced_json_v1"


def decode_framed_object(content):
    """Accept exactly one JSON object, optionally in one complete JSON fence.

    Never extract an object from prose, repair a partial reply, choose between
    multiple objects, or replace tool arguments. Callers still validate the
    decoded object against the robot's unchanged action protocol.
    """
    if not isinstance(content, str):
        raise ValueError("tool response content must be text")
    body = content.strip()
    envelope = "plain_json"
    if body.startswith("```"):
        match = re.fullmatch(r"```json[ \t]*\r?\n([\s\S]*?)\r?\n```", body)
        if match is None:
            raise ValueError("tool response needs one complete JSON fence and no surrounding prose")
        body = match[1]
        envelope = "json_fence"

    def unique_object(pairs):
        result = {}
        for name, value in pairs:
            if name in result:
                raise ValueError("duplicate key in tool response: " + name)
            result[name] = value
        return result

    def finite_float(value):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("nonfinite number in tool response")
        return number

    def reject_constant(value):
        raise ValueError("non-JSON constant in tool response: " + value)

    result = json.loads(body, object_pairs_hook=unique_object,
                        parse_float=finite_float, parse_constant=reject_constant)
    if not isinstance(result, dict):
        raise ValueError("tool response must contain one JSON object")
    return result, {
        "response_framing": JSON_OR_FENCE,
        "response_envelope": envelope,
        "response_content_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "response_json_sha256": hashlib.sha256(body.encode()).hexdigest(),
    }
