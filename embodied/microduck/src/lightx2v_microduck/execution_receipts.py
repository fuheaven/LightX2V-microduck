"""Recompute transport and latency evidence from actual robotd receipts.

Consumption means the external buffer selected an action in controller.step;
Safety and motor writes still follow. It is not an actuator-applied receipt.
"""
from __future__ import annotations

from collections import Counter
import numpy as np

PROTOCOL = "robotd_execution_receipts_v1"
LEGACY_MODE = "source_tick_age_compensated"
PREFIX_MODE = "prefix_playback_v1"


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {"count": 0, "p50": None, "p95": None, "p99": None, "maximum": None}
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("timing evidence must be finite and nonnegative")
    return {"count": len(values), "p50": float(np.percentile(values, 50)), "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)), "maximum": float(values.max())}


def summarize_receipts(receipts, *, request_events=(), missing_events=0, unrecorded_control_ticks=0):
    sequences = [row["sequence"] for row in receipts]
    if sequences != sorted(set(sequences)):
        raise ValueError("execution receipts must be unique and ordered")
    violations, first_latencies, all_ages, periods, source_ages = [], [], [], [], []
    accepted, closed, first_consumed, first_consumption_times = {}, set(), set(), {}
    modes, consumed_modes = {}, set()
    last_accepted_tick, previous_source, fallback_transitions = {}, None, 0
    consume, fallback, submissions = [], [], []
    source_clock_offsets = []
    for row in receipts:
        kind = row["kind"]
        episode = row.get("episode_id")
        key = (episode, row.get("source_tick"))
        if kind == "open":
            closed.discard(episode)
            modes[episode] = row.get("execution_mode", LEGACY_MODE)
            if modes[episode] not in (LEGACY_MODE, PREFIX_MODE):
                violations.append({"sequence": row["sequence"], "reason": "unknown_execution_mode"})
        elif kind in ("close", "revoke"):
            closed.add(episode)
        elif kind == "submit":
            submissions.append(row)
            if row.get("result") == "accepted":
                if row.get("packet_episode_id", episode) != episode or episode in closed:
                    violations.append({"sequence": row["sequence"], "reason": "accepted_wrong_or_closed_episode"})
                if row["source_tick"] <= last_accepted_tick.get(episode, -1):
                    violations.append({"sequence": row["sequence"], "reason": "accepted_out_of_order_source"})
                last_accepted_tick[episode] = row["source_tick"]
                accepted[key] = row
                mode = row.get("execution_mode", LEGACY_MODE)
                if mode != modes.get(episode, LEGACY_MODE):
                    violations.append({"sequence": row["sequence"], "reason": "unnegotiated_execution_mode"})
                if mode == PREFIX_MODE:
                    admission, start, expiry = (row.get(name) for name in ("admission_tick", "playback_start_tick", "valid_until_tick"))
                    if (any(type(x) is not int for x in (admission, start, expiry, row.get("source_tick")))
                            or admission < row["source_tick"] or start != admission + 1
                            or not admission < expiry <= row["source_tick"] + 10
                            or row.get("rows") != 5 or row.get("execution_prefix") != 5 or row.get("prediction_horizon") != 50):
                        violations.append({"sequence": row["sequence"], "reason": "invalid_prefix_admission"})
        elif kind == "consume":
            consume.append(row)
            if row.get("control_period_ns") is not None:
                periods.append(row["control_period_ns"] / 1e6)
            current = row.get("control_capture")
            if current:
                source_clock_offsets.append(row["daemon_tick"] - current["body_tick"])
                source_ages.append(max(0., (row["monotonic_ns"] - current["capture_monotonic_ns"]) / 1e6))
            if row.get("execution_source") != "external":
                fallback_transitions += previous_source == "external"
                previous_source = "official_fallback"
                fallback.append(row)
                continue
            previous_source = "external"
            mode = row.get("execution_mode", LEGACY_MODE)
            consumed_modes.add(mode)
            if key not in accepted:
                violations.append({"sequence": row["sequence"], "reason": "consume_without_recorded_acceptance"})
            if episode in closed:
                violations.append({"sequence": row["sequence"], "reason": "consume_after_close_or_revoke"})
            origin = row["source_tick"]
            if mode != modes.get(episode, LEGACY_MODE) or mode not in (LEGACY_MODE, PREFIX_MODE):
                violations.append({"sequence": row["sequence"], "reason": "unnegotiated_execution_mode"})
            if mode == PREFIX_MODE:
                submission = accepted.get(key, {})
                fields = ("execution_mode", "admission_tick", "playback_start_tick", "execution_prefix",
                          "prediction_horizon", "valid_until_tick", "source_capture")
                if (any(row.get(field) != submission.get(field) for field in fields)
                        or type(row.get("action_index")) is not int or not 0 <= row["action_index"] < 5):
                    violations.append({"sequence": row["sequence"], "reason": "prefix_identity_changed"})
                origin = row.get("playback_start_tick")
            if type(origin) is not int or row.get("action_index") != row["daemon_tick"] - origin:
                violations.append({"sequence": row["sequence"], "reason": "wrong_action_index"})
            if row["daemon_tick"] >= row["valid_until_tick"]:
                violations.append({"sequence": row["sequence"], "reason": "expired_action_consumed"})
            capture = row.get("source_capture")
            if capture is None:
                violations.append({"sequence": row["sequence"], "reason": "missing_source_capture"})
                continue
            age = (row["monotonic_ns"] - capture["capture_monotonic_ns"]) / 1e6
            if age < 0 or age >= 200:
                violations.append({"sequence": row["sequence"], "reason": "invalid_capture_age"})
                continue
            all_ages.append(age)
            if key not in first_consumed:
                first_latencies.append(age)
                first_consumed.add(key)
                first_consumption_times[key] = row["monotonic_ns"]
    observations, image_observations, image_latencies = {}, {}, []
    for event in request_events:
        source = event.get("source_capture")
        if source and event.get("inference_completed", True):
            identity = (source.get("episode_id"), source.get("task_epoch"), source["body_tick"])
            observations.setdefault(identity, source["capture_monotonic_ns"])
            image_ns = event.get("image_capture_monotonic_ns")
            if image_ns is not None:
                image_observations.setdefault((source.get("episode_id"), event.get("image_source_body_tick")), image_ns)
                consumed_ns = first_consumption_times.get((source.get("episode_id"), event.get("source_tick")))
                if consumed_ns is not None:
                    image_latencies.append((consumed_ns - image_ns) / 1e6)
    captures = sorted(observations.values())
    observation_hz = (len(captures) - 1) * 1e9 / (captures[-1] - captures[0]) if len(captures) > 1 and captures[-1] > captures[0] else None
    image_captures = sorted(image_observations.values())
    image_hz = (len(image_captures) - 1) * 1e9 / (image_captures[-1] - image_captures[0]) if len(image_captures) > 1 and image_captures[-1] > image_captures[0] else None
    rejected = [row for row in submissions if row.get("result") == "rejected"]
    sequence_gaps = sum(max(0, second-first-1) for first, second in zip(sequences, sequences[1:]))
    complete = not (missing_events or unrecorded_control_ticks or sequence_gaps)
    return {"protocol": PROTOCOL, "receipt_count": len(receipts), "complete_receipt_history": complete,
            "execution_modes": sorted(consumed_modes),
            "missing_events": missing_events, "unrecorded_control_ticks": unrecorded_control_ticks, "sequence_gaps": sequence_gaps,
            "consume_boundary": "controller_external_buffer_before_safety_and_io", "actuator_applied_measured": False,
            "first_action_capture_to_consume_ms": distribution(first_latencies), "all_external_action_age_ms": distribution(all_ages),
            "first_action_image_capture_to_consume_ms": distribution(image_latencies),
            "controller_period_ms": distribution(periods), "controller_sensor_age_ms": distribution(source_ages),
            "completed_fresh_observations": len(captures), "completed_observation_hz": observation_hz,
            "completed_fresh_images": len(image_captures), "completed_image_hz": image_hz,
            "submitted_chunks": len(accepted), "consumed_chunks": len(first_consumed),
            "accepted_but_never_consumed": len(set(accepted) - first_consumed),
            "external_ticks": len(consume) - len(fallback), "fallback_ticks": len(fallback),
            "external_to_fallback_transitions": fallback_transitions,
            "revocation_reasons": dict(Counter(row.get("reason", "unknown") for row in receipts if row["kind"] in ("close", "revoke"))),
            "fallback_reasons": dict(Counter(row.get("reason", "unknown") for row in fallback)),
            "rejected_chunks": len(rejected), "rejection_reasons": dict(Counter(row.get("reason", "unknown") for row in rejected)),
            "client_dropped_replies": sum(event.get("result") in ("injected_drop", "discarded_after_cancel") for event in request_events),
            "clock_offset_span_ticks": max(source_clock_offsets) - min(source_clock_offsets) if source_clock_offsets else None,
            "violations": violations, "protocol_invariants_passed": complete and not violations,
            "offline_accepted": False, "realtime_accepted": False, "hardware_unvalidated": True}
