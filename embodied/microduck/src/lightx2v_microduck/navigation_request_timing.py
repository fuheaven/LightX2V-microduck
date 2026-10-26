"""Decompose each recorded request before summarizing latency distributions."""
from __future__ import annotations

import math

from .navigation_outcomes import require


TIMESTAMPS = ("prepare_started_ns", "prepare_finished_ns", "worker_started_ns",
              "predict_started_ns", "predict_finished_ns", "worker_finished_ns", "reply_received_ns")
STAGES = ("capture_to_preparation", "preparation", "queue_and_handoff", "worker_setup_and_injected_delay",
          "policy_call", "response_validation", "reply_handoff", "admission")


def distribution(values):
    if not values:
        return None
    values = sorted(values)
    def quantile(q):
        index = (len(values)-1)*q
        low, high = math.floor(index), math.ceil(index)
        return values[low] + (values[high]-values[low])*(index-low)
    return dict(count=len(values), p50_ms=quantile(.5), p95_ms=quantile(.95), max_ms=values[-1])


def audit_request_timing(episode, controls, predictions, camera_receipts):
    """Old traces remain usable but cannot claim a measured decomposition.

    This covers replies consumed by the control thread. Pending requests replaced
    by newer observations and replies left at task end are not inference samples.
    Each source's stages add exactly; their independent percentiles do not.
    """
    metadata = (episode.get("prediction_recording") or {}).get("request_timing")
    realtime = [row for row in predictions if row["phase"] == "realtime"]
    if metadata is None:
        require(not any("request_timing" in row for row in realtime), "request timestamps lack a declared contract")
        return dict(status="unavailable", measured_replies=0, legacy_replies=len(realtime),
                    additive_per_request_verified=False,
                    reason="Legacy recording has no model-call timestamps; do not infer queue delay.")
    require(metadata.get("schema") == "microduck_navigation_request_timing_v1"
            and metadata.get("clock") == "monotonic_ns"
            and metadata.get("timestamps") == list(TIMESTAMPS)
            and metadata.get("unconsumed_requests_and_replies_excluded") is True,
            "request timing contract differs")
    if not realtime:
        return dict(status="no_samples", measured_replies=0, additive_per_request_verified=False,
                    reason="No realtime replies were consumed during this measurement.")
    by_tick = {row["source_tick"]: row for row in camera_receipts}
    samples = {name: [] for name in STAGES}
    success_ages, error_reply_ages = [], []
    errors = 0
    for prediction in realtime:
        timing = prediction.get("request_timing")
        require(isinstance(timing, dict) and set(timing) == set(TIMESTAMPS),
                "request timestamps missing or unexpected")
        times = [timing[name] for name in TIMESTAMPS]
        require(all(type(value) is int and value > 0 for value in times), "invalid request timestamp")
        capture = prediction["source_monotonic_ns"]
        require(type(capture) is int and 0 < capture <= times[0] and times == sorted(times),
                "request timing order differs")
        row = by_tick[prediction["source_tick"]]
        require(row["status"] == "delivered" and row["delivery_phase"] == "measured"
                and row["capture_ns"] == capture and row["delivered_ns"] <= times[0],
                "request preparation precedes actual camera delivery")
        receipt_tick = prediction["tick"]
        require(type(receipt_tick) is int and 0 <= receipt_tick < len(controls)
                and times[-1] <= controls[receipt_tick]["execution_monotonic_ns"],
                "request reply is outside its receiving control tick")
        failed = prediction["error"] is not None
        admission = prediction["admission_monotonic_ns"]
        if failed:
            require(admission is None and prediction["accepted"] is False
                    and prediction["submitted_actions"] is None, "failed request fabricated action admission")
            errors += 1
            endpoint = times[-1]
            error_reply_ages.append((endpoint-capture)/1e6)
        else:
            require(type(admission) is int and times[-1] <= admission <= controls[receipt_tick]["execution_monotonic_ns"],
                    "request admission precedes its reply")
            endpoint = admission
            success_ages.append((endpoint-capture)/1e6)
        stamps = [capture, *times] + ([] if failed else [admission])
        intervals = [right-left for left, right in zip(stamps, stamps[1:])]
        require(sum(intervals) == endpoint-capture, "request latency decomposition differs")
        for name, duration in zip(STAGES, intervals):
            samples[name].append(duration/1e6)
    return dict(schema="microduck_navigation_request_timing_audit_v1", status="complete",
                measured_replies=len(realtime), inference_errors=errors,
                additive_per_request_verified=True,
                stages={name: distribution(values) for name, values in samples.items()},
                successful_admission_age=distribution(success_ages), error_reply_age=distribution(error_reply_ages),
                model_inference_replayed=False, task_success_verified=False,
                scope="Consumed realtime replies only; no inference about replaced or unconsumed requests.",
                percentile_warning="Do not sum stage percentiles; stages add only for each individual request.")
