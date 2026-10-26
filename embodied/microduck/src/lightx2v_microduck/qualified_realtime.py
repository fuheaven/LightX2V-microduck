"""Serial, real-robotd evaluation of qualified project TinyVLA HTTP artifacts.

This entry point does not start a model server or allocate a GPU. Run that
service inside the accounted allocation. Final populations are isolated from
development, and individual daemon diagnostics cannot grant skill acceptance.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np

from .artifact_identity import bundle_sha256
from .data import file_sha256, write_json
from .execution_receipts import PREFIX_MODE, summarize_receipts
from .qualified_eval import (FINAL_SEEDS, episode_result_from_state, summarize_task,
                             validate_population, validate_qualified_manifest)
from .qualified_teachers import ACCEPTANCE_SCOPE

PROTOCOL = "project_qualified_robotd_evaluation_v2"
EVIDENCE_PROTOCOL = "qualified_robotd_episode_evidence_v2"


def _positive(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and np.isfinite(value) and value > 0


def prefix_execution(receipts):
    """Retain each accepted chunk's actual playback, including empty prefixes.

    A global min/max cannot establish that all five rows ran before the next
    source observation. No partial closing chunk is silently excused: a future
    task-boundary protocol must provide that evidence explicitly.
    """
    chunks, by_source = [], {}
    for row in receipts:
        key = (row.get("episode_id"), row.get("source_tick"))
        if row.get("kind") == "submit" and row.get("result") == "accepted":
            chunk = {"episode_id": key[0], "source_tick": key[1], "indices": [], "ticks": [],
                     "consume_monotonic_ns": [], "capture_monotonic_ns": None}
            chunks.append(chunk)
            by_source[key] = chunk
        elif row.get("kind") == "consume" and row.get("execution_source") == "external":
            chunk = by_source.get(key)
            if chunk is not None:
                chunk["indices"].append(row.get("action_index"))
                chunk["ticks"].append(row.get("daemon_tick"))
                chunk["consume_monotonic_ns"].append(row.get("monotonic_ns"))
                capture = (row.get("source_capture") or {}).get("capture_monotonic_ns")
                if chunk["capture_monotonic_ns"] is None:
                    chunk["capture_monotonic_ns"] = capture
                elif chunk["capture_monotonic_ns"] != capture:
                    chunk["inconsistent_capture"] = True
    return {"protocol": "five_action_prefix_execution_v1", "chunks": chunks,
            "accepted_chunks": len(chunks)}


def valid_prefix_execution(value, timing):
    if not isinstance(value, dict) or value.get("protocol") != "five_action_prefix_execution_v1":
        return False
    chunks = value.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        return False
    if (value.get("accepted_chunks") != len(chunks)
            or timing.get("submitted_chunks") != len(chunks)
            or timing.get("consumed_chunks") != len(chunks)
            or timing.get("external_ticks") != 5 * len(chunks)
            or timing.get("accepted_but_never_consumed") != 0):
        return False
    seen, previous_end = set(), None
    for chunk in chunks:
        if not isinstance(chunk, dict):
            return False
        indices, ticks, times = (chunk.get(key) for key in ("indices", "ticks", "consume_monotonic_ns"))
        key = (chunk.get("episode_id"), chunk.get("source_tick"))
        capture = chunk.get("capture_monotonic_ns")
        if (not isinstance(key[0], str) or type(key[1]) is not int or key[1] < 0
                or key in seen or indices != [0, 1, 2, 3, 4] or any(type(i) is not int for i in indices)
                or not isinstance(ticks, list) or len(ticks) != 5 or any(type(t) is not int for t in ticks)
                or ticks != list(range(ticks[0], ticks[0] + 5))
                or not isinstance(times, list) or len(times) != 5 or any(type(t) is not int for t in times)
                or any(b <= a for a, b in zip(times, times[1:]))
                or type(capture) is not int or capture <= 0 or capture > times[0]
                or chunk.get("inconsistent_capture")
                or (previous_end is not None and capture < previous_end)):
            return False
        seen.add(key)
        previous_end = times[-1]
    return True


def inspect_execution(report, receipt_record):
    """Recompute evidence from actual events; never trust a cached success flag."""
    status = receipt_record.get("status", {})
    receipts = receipt_record.get("receipts", [])
    events = receipt_record.get("request_events", [])
    timing = summarize_receipts(receipts, request_events=events,
                               missing_events=status.get("missing_events", 0),
                               unrecorded_control_ticks=status.get("unrecorded_control_ticks", 0))
    consume = [r for r in receipts if r.get("kind") == "consume"]
    ticks = [r.get("daemon_tick") for r in consume]
    continuous = bool(ticks) and all(type(t) is int for t in ticks) and ticks == list(range(ticks[0], ticks[-1] + 1))
    episodes = {r.get("episode_id") for r in consume}
    valid_sources = len(episodes) == 1 and None not in episodes and all(
        (r.get("source_capture") or {}).get("episode_id") == r.get("episode_id")
        for r in consume if r.get("execution_source") == "external")
    indices = [r.get("action_index") for r in consume if r.get("execution_source") == "external"]
    # Require complete ordered prefixes, not only a global range within 0..4.
    prefixes = prefix_execution(receipts)
    prefix_passed = valid_prefix_execution(prefixes, timing)
    health = report.get("health_samples", [])
    health_passed = len(health) >= 2 and all(s.get("health", {}).get("healthy") is True for s in health)
    final = report.get("final", {})
    body = final.get("backend", {})
    step_times = body.get("task_step_monotonic_ns", [])
    factor = None
    if len(step_times) >= 2 and all(type(t) is int for t in step_times) and all(b > a for a, b in zip(step_times, step_times[1:])):
        factor = (len(step_times) - 1) * .02 * 1e9 / (step_times[-1] - step_times[0])
    latencies = [timing[key].get("p95") for key in
                 ("first_action_capture_to_consume_ms", "first_action_image_capture_to_consume_ms")]
    end_to_end = max(latencies) if all(_positive(x) for x in latencies) else None
    observations = timing["completed_observation_hz"]
    images = timing["completed_image_hz"]
    checks = {
        "physics_never_waited": report.get("mode") == "robotd_bam" and report.get("physics_waits_for_model") is False,
        "complete_receipts": status.get("supported") is True and status.get("complete") is True and timing["complete_receipt_history"],
        "protocol_invariants": timing["protocol_invariants_passed"] and valid_sources and continuous,
        "external_actions_observed": timing["external_ticks"] > 0,
        "no_fallback": timing["fallback_ticks"] == 0 and final.get("external", {}).get("fallback_ticks") == 0,
        "prefix_playback_mode": timing["execution_modes"] == [PREFIX_MODE],
        "first_five_actions_only": prefix_passed,
        "daemon_health": health_passed and final.get("health", {}).get("healthy") is True and body.get("fault") is None,
        "near_real_time": _positive(factor) and .9 <= factor <= 1.1,
        "fresh_observation_cadence": _positive(observations) and observations >= 9 and _positive(images) and images >= 9,
        "end_to_end_p95": _positive(end_to_end) and end_to_end <= 100,
        "no_transport_fault": not report.get("fault") and report.get("cancelled") is False,
        "nominal_run": report.get("latency_ms") == 0 and report.get("drop_every") == 0 and report.get("reorder_every") == 0,
    }
    checks = {key: bool(value) for key, value in checks.items()}
    return {"protocol": EVIDENCE_PROTOCOL, "checks": checks, "passed": all(checks.values()),
            "timing": timing, "end_to_end_p95_ms": end_to_end, "real_time_factor": factor,
            "consumed_action_index_range": [min(indices), max(indices)] if indices and all(type(x) is int for x in indices) else None,
            "prefix_execution": prefixes,
            "health_sample_count": len(health), "health_all_passed": health_passed,
            "target_control_hz": 50, "target_observation_hz": 10, "requested_execution_prefix": 5,
            "execution_semantics": timing["execution_modes"][0] if len(timing["execution_modes"]) == 1 else "mixed_or_unobserved",
            "hardware_unvalidated": True}


def valid_episode_evidence(evidence):
    required = {"physics_never_waited", "complete_receipts", "protocol_invariants", "external_actions_observed",
                "no_fallback", "prefix_playback_mode", "first_five_actions_only", "daemon_health", "near_real_time",
                "fresh_observation_cadence", "end_to_end_p95", "no_transport_fault", "nominal_run"}
    if not isinstance(evidence, dict) or evidence.get("protocol") != EVIDENCE_PROTOCOL:
        return False
    checks = evidence.get("checks", {})
    timing = evidence.get("timing", {})
    indices = evidence.get("consumed_action_index_range", [])
    factor, latency = evidence.get("real_time_factor"), evidence.get("end_to_end_p95_ms")
    measured = [timing.get(key, {}).get("p95") for key in
                ("first_action_capture_to_consume_ms", "first_action_image_capture_to_consume_ms")]
    return (set(checks) == required and all(type(v) is bool and v for v in checks.values())
            and evidence.get("passed") is True and evidence.get("requested_execution_prefix") == 5
            and indices == [0, 4] and all(type(i) is int for i in indices)
            and valid_prefix_execution(evidence.get("prefix_execution"), timing)
            and evidence.get("health_sample_count", 0) >= 2 and evidence.get("health_all_passed") is True
            and timing.get("protocol_invariants_passed") is True and timing.get("complete_receipt_history") is True
            and timing.get("execution_modes") == [PREFIX_MODE] and evidence.get("execution_semantics") == PREFIX_MODE
            and timing.get("fallback_ticks") == 0 and timing.get("external_ticks", 0) > 0
            and all(_positive(value) for value in measured) and latency == max(measured)
            and all(_positive(timing.get(key)) and timing[key] >= 9 for key in ("completed_observation_hz", "completed_image_hz"))
            and _positive(factor) and .9 <= factor <= 1.1 and _positive(latency) and latency <= 100)


def summarize_realtime_task(rows, contract, *, purpose):
    result = summarize_task(rows, contract, purpose=purpose)
    execution = [row["realtime_evidence"] for row in rows]
    healthy = all(item["checks"]["daemon_health"] for item in execution)
    takeovers = sum(row.get("recovery_events", 0) + row["realtime_evidence"]["timing"]["fallback_ticks"] for row in rows)
    accepted = result["project_task_accepted"] and healthy and takeovers == 0 and all(valid_episode_evidence(e) for e in execution)
    latency = [e["end_to_end_p95_ms"] for e in execution]
    factors = [e["real_time_factor"] for e in execution]
    result.update(offline_accepted=False, realtime_accepted=accepted, project_task_accepted=accepted,
                  counts_as_project_task_success=accepted, daemon_health_passed=healthy, takeovers=takeovers,
                  end_to_end_p95_ms=max(latency) if all(_positive(x) for x in latency) else None,
                  minimum_real_time_factor=min(factors) if all(_positive(x) for x in factors) else None,
                  execution_evidence_protocol=EVIDENCE_PROTOCOL,
                  execution_checks_passed=all(valid_episode_evidence(e) for e in execution),
                  execution_failures=dict(Counter(k for e in execution for k, v in e["checks"].items() if not v)))
    return result


def evaluate_qualified_realtime(endpoint, *, tasks, seeds, record_dir, purpose="development", asset_root=None,
                                binary=None, timeout=5.0, deadline=None):
    from .daemon import RobotdBamBackend
    from .realtime import run_realtime
    from .remote import TinyVLAHTTPPolicy

    if not _positive(timeout):
        raise ValueError("HTTP timeout must be finite and positive")
    tasks, seeds = validate_population(tasks, seeds, purpose=purpose, record_dir=record_dir)
    if purpose == "teacher_reference":
        raise ValueError("real-time project evaluation requires the native student service")
    policy = TinyVLAHTTPPolicy(endpoint, timeout=timeout)
    contracts = validate_qualified_manifest(policy.artifact_manifest, tasks)
    if policy.manifest.horizon != 50 or policy.manifest.execution_prefix != 5:
        raise ValueError("qualified real-time evaluator requires the public 50-action, five-step-prefix model")
    precision = policy.manifest.metadata.get("precision")
    if precision not in ("fp32", "bf16"):
        raise ValueError("qualified real-time evaluator requires the effective service precision")
    artifact = dict(policy.manifest.artifacts)
    if artifact.get("artifact_bundle_sha256") != bundle_sha256(policy.artifact_manifest):
        raise ValueError("service artifact bundle identity is inconsistent")
    destination = Path(record_dir).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    rows = []
    report = {"schema_version": 1, "protocol": PROTOCOL, "acceptance_scope": ACCEPTANCE_SCOPE,
              "purpose": purpose, "evaluator_role": "student", "independent_final_test": purpose == "final_test",
              "complete": False, "mode": "robotd_bam", "physics_waits_for_model": False,
              "precision": precision, "runtime_inference": dict(policy.runtime_inference),
              "task_contracts": contracts, "task_contracts_sha256": bundle_sha256(contracts),
              "checkpoint_sha256": artifact["checkpoint_sha256"], "artifact_bundle_sha256": artifact["artifact_bundle_sha256"],
              "semantic_refresh_steps": 5, "requested_execution_prefix": 5, "execute_steps": None,
              "requested_execution_mode": PREFIX_MODE, "execution_semantics": None, "nominal_observation_hz": 10,
              "counts_as_skill_success": False, "complete_author13_accepted": False, "claims_visual_language_autonomy": False,
              "author_program_reproduction": False, "uses_prescribed_command13": True, "hardware_unvalidated": True,
              "offline_accepted": False, "realtime_accepted": False, "episodes": rows, "tasks": {},
              "evaluator_source_sha256": file_sha256(Path(__file__))}
    try:
        for task in tasks:
            for seed in seeds:
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError("real-time evaluation allocation deadline exceeded")
                # Detect a server restart or changed bundle between episodes.
                health = policy.call("/health")
                if (health.get("engine") != "lightx2v-native-tinyvla" or health.get("raw_artifacts_verified") is not True
                        or bundle_sha256(health.get("manifest", {})) != artifact["artifact_bundle_sha256"]
                        or health.get("runtime_inference") != policy.runtime_inference
                        or health.get("precision") != precision):
                    raise ValueError("model server identity changed during evaluation")
                folder = destination / f"{task}-{seed}"
                backend = RobotdBamBackend(asset_root, scene="flat", render=True, binary=binary,
                                          initial_conditions="qualified_initial_v2", execution_mode=PREFIX_MODE)
                try:
                    backend.reset(seed, task)
                    duration = backend.task.duration_s + 2
                    if deadline is not None:
                        duration = min(duration, max(.001, deadline - time.monotonic()))
                    episode = run_realtime(backend, policy, folder, duration_s=duration, seed=seed, record_health=True)
                    row = episode_result_from_state(episode["final"], task, seed, contracts[task])
                    receipts = folder / episode["execution_receipts"]["path"]
                    if file_sha256(receipts) != episode["execution_receipts"]["sha256"]:
                        raise ValueError("execution receipt file changed after recording")
                    evidence = inspect_execution(episode, json.loads(receipts.read_text()))
                    row.update(realtime_evidence=evidence, behavior_success=row["success"],
                               success=bool(row["success"] and valid_episode_evidence(evidence)),
                               realtime_report_path=str(folder / "realtime.json"), realtime_report_sha256=file_sha256(folder / "realtime.json"),
                               receipt_path=str(receipts), receipt_sha256=file_sha256(receipts))
                    with backend.body.lock:
                        trace = np.asarray(backend.env._ground_trace if task == "ground_lower" else backend.env._trajectory).copy()
                        trace_tick = backend.env.tick
                    np.savez_compressed(folder / "physical-trace.npz", trace=trace, captured_task_tick=trace_tick)
                    row.update(trajectory_path=str(folder / "physical-trace.npz"), trajectory_sha256=file_sha256(folder / "physical-trace.npz"))
                    write_json(folder / "result.json", row)
                    rows.append(row)
                    write_json(destination / "progress.json", report)
                finally:
                    backend.close()
            selected = [row for row in rows if row["task"] == task]
            report["tasks"][task] = summarize_realtime_task(selected, contracts[task], purpose=purpose)
        report.update(complete=True, realtime_accepted=all(s["realtime_accepted"] for s in report["tasks"].values()),
                      execute_steps=5 if all(r["realtime_evidence"]["checks"]["first_five_actions_only"] for r in rows) else None,
                      execution_semantics=PREFIX_MODE if all(r["realtime_evidence"]["checks"]["prefix_playback_mode"] for r in rows) else "mixed_or_unobserved",
                      mean_strict_rate=float(np.mean([s["strict_rate"] for s in report["tasks"].values()])))
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        write_json(destination / "report.json", report)
        write_json(destination / "progress.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--purpose", choices=("validation", "development", "final_test"), default="development")
    parser.add_argument("--seed-start", type=int)
    parser.add_argument("--seeds", type=int)
    parser.add_argument("--asset-root")
    parser.add_argument("--robotd-binary")
    parser.add_argument("--http-timeout", type=float, default=5.)
    parser.add_argument("--wall-seconds", type=float, required=True)
    parser.add_argument("--output", required=True, help="New evidence directory")
    args = parser.parse_args()
    if not _positive(args.wall_seconds):
        parser.error("wall-seconds must be finite and positive")
    start = args.seed_start if args.seed_start is not None else (FINAL_SEEDS[0] if args.purpose == "final_test" else 610000)
    count = args.seeds if args.seeds is not None else (100 if args.purpose == "final_test" else 20)
    report = evaluate_qualified_realtime(args.endpoint, tasks=args.tasks, seeds=range(start, start + count), record_dir=args.output,
                                        purpose=args.purpose, asset_root=args.asset_root, binary=args.robotd_binary,
                                        timeout=args.http_timeout, deadline=time.monotonic() + args.wall_seconds)
    print(json.dumps(report["tasks"], indent=2))


if __name__ == "__main__":
    main()
