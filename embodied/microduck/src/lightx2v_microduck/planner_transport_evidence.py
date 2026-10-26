"""Record transport failures without retrying, executing, or inventing a reply."""

from copy import deepcopy
import hashlib
import json
import re
import urllib.error


PROTOCOL = "microduck_planner_transport_error_v1"
IDENTITY_KEYS = ("request_id", "request_sha256", "model_id", "identity_sha256")


def request_digest(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _response_fields(body):
    if not isinstance(body, dict):
        return {}
    fields = {}
    for key in ("error", "type"):
        if isinstance(body.get(key), str) and len(body[key]) <= 256:
            fields["server_error" if key == "error" else "server_error_type"] = body[key]
    evidence = body.get("microduck_evidence")
    if isinstance(evidence, dict):
        fields["serving_evidence"] = {k: evidence[k] for k in IDENTITY_KEYS
                                      if isinstance(evidence.get(k), str) and len(evidence[k]) <= 128}
    return fields


def capture_transport_error(error, payload, identifier, expected_identity):
    row = dict(protocol=PROTOCOL, request_id=identifier, request_sha256=request_digest(payload),
               model_id=payload["model"], expected_service_identity=expected_identity,
               exception_type=type(error).__name__, http_status=None, service_evidence_verified=False)
    if isinstance(error, urllib.error.HTTPError):
        row["http_status"] = error.code
        try:
            raw = error.read(8193)
            if len(raw) <= 8192:
                row.update(_response_fields(json.loads(raw)))
        except (OSError, ValueError, TypeError):
            # The original transport exception remains the failure. A missing
            # or invalid response body cannot be authenticated retroactively.
            pass
        expected = dict(request_id=identifier, request_sha256=row["request_sha256"],
                        model_id=payload["model"], identity_sha256=expected_identity)
        row["service_evidence_verified"] = bool(
            re.fullmatch(r"[0-9a-f]{64}", str(expected_identity))
            and row.get("serving_evidence") == expected)
    return row


def validate_recorded_transport_error(row, payload, identifier, identity, terminal):
    """Validate a historical client failure against its actual server receipt.

    A timeout's duration is not remeasured. Missing service receipts remain an
    audit failure even if a client-side digest exists.
    """
    expected = dict(protocol=PROTOCOL, request_id=identifier, request_sha256=request_digest(payload),
                    model_id=payload["model"], expected_service_identity=identity)
    if any(row.get(k) != v for k, v in expected.items()):
        raise ValueError("transport failure belongs to another request or service")
    kind = row.get("exception_type")
    if kind not in ("HTTPError", "URLError", "TimeoutError") or not terminal:
        raise ValueError("unsupported or unrecorded transport failure")
    if kind == "HTTPError":
        if "error.json" in terminal:
            stored = terminal["error.json"]
            status, response = stored.get("http_status"), stored.get("response")
        else:
            raw = terminal.get("result.json", {}).get("result", {})
            if raw.get("status") not in ("deadline", "cancelled"):
                raise ValueError("HTTP failure does not match recorded inference outcome")
            status = 504 if raw["status"] == "deadline" else 409
            response = dict(error=raw["status"], microduck_evidence=
                            terminal["result.json"]["response"]["microduck_evidence"])
        fields = _response_fields(response)
        identity_fields = dict(request_id=identifier, request_sha256=expected["request_sha256"],
                               model_id=payload["model"], identity_sha256=identity)
        if (type(status) is not int or not 400 <= status <= 599 or row.get("http_status") != status
                or fields.get("serving_evidence") != identity_fields
                or row.get("service_evidence_verified") is not True
                or any(row.get(k) != fields.get(k) for k in
                       ("server_error", "server_error_type", "serving_evidence"))):
            raise ValueError("HTTP failure status, response or identity differs from its receipt")
    elif (row.get("http_status") is not None or row.get("service_evidence_verified") is not False
          or any(k in row for k in ("server_error", "server_error_type", "serving_evidence"))):
        raise ValueError("a client timeout or connection failure cannot claim an HTTP response")


def raise_recorded_transport_error(row):
    kind = row["exception_type"]
    if kind == "HTTPError":
        error = urllib.error.HTTPError("recorded", row["http_status"], "recorded client failure", {}, None)
    elif kind == "URLError":
        error = urllib.error.URLError("recorded client failure")
    elif kind == "TimeoutError":
        error = TimeoutError("recorded client failure")
    else:
        raise ValueError("unsupported recorded transport error")
    error.planner_transport_error = deepcopy(row)
    raise error
