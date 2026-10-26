"""Real daemon latency experiments. The physics thread never waits here."""

from __future__ import annotations

import time
import threading
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .contracts import JointResidualChunk
from .data import file_sha256, write_json
from .evaluation import serializable, validate_action_source, validate_policy_environment
from .execution_receipts import distribution, summarize_receipts


def run_realtime(backend, policy, output_dir, *, duration_s=30.0, latency_ms=0.0, drop_every=0, reorder_every=0, seed=0,
                 cancel_after_s=None, cancel_path=None, record_health=False):
    if duration_s <= 0 or latency_ms < 0 or drop_every < 0 or reorder_every < 0:
        raise ValueError("invalid duration/fault configuration")
    if cancel_after_s is not None and (not np.isfinite(cancel_after_s) or cancel_after_s < 0):
        raise ValueError("cancel_after_s must be finite and nonnegative")
    manifest = getattr(policy, "manifest", None)
    contract_validation = validate_policy_environment(manifest, backend.env)
    if manifest.uses_privileged_state:
        raise ValueError("real-time external VLA evaluation excludes privileged teacher policies; use the separate offline oracle group")
    if manifest.action_kind != "joint_residual":
        raise ValueError("real-time external policy requires raw joint residuals")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "realtime.json").exists():
        raise FileExistsError(output / "realtime.json")
    started = time.monotonic()
    policy.reset(backend.episode_id)
    events, frames, frame_ticks, latencies, end_to_end = [], [], [], [], []
    query_ticks, query_states, source_times, raw_chunks = [], [], [], []
    previous = None
    last_observation = -1
    index = 0
    fault = None
    cancelled, watcher_done = threading.Event(), threading.Event()
    cancellation = {}
    receipt_status = {"supported": False, "complete": False, "reason": "backend does not expose execution receipts"}
    health_samples = []

    def sample_health():
        if record_health:
            health_samples.append({"monotonic_ns": time.monotonic_ns(),
                                   "health": backend.control.call("robot.health")})

    def drain_receipts():
        nonlocal receipt_status
        if hasattr(backend, "drain_execution_receipts"):
            receipt_status = backend.drain_execution_receipts()

    def watch_cancellation():
        # Runs independently of blocking HTTP inference. Cancellation revokes
        # the daemon session immediately; late replies are never submitted.
        while not watcher_done.wait(.005):
            if ((cancel_after_s is not None and time.monotonic() - started >= cancel_after_s)
                    or (cancel_path is not None and Path(cancel_path).exists())):
                cancellation["requested_monotonic_ns"] = time.monotonic_ns()
                cancelled.set()
                try:
                    cancellation["response"] = backend.cancel()
                    cancellation["ack_monotonic_ns"] = time.monotonic_ns()
                except Exception as exc:
                    cancellation["error"] = f"{type(exc).__name__}: {exc}"
                return

    watcher = threading.Thread(target=watch_cancellation, name="robotd-evaluation-cancel", daemon=True)
    watcher.start()
    try:
        drain_receipts()
        sample_health()
        while time.monotonic() - started < duration_s and not backend.terminated and not cancelled.is_set():
            observation = backend.observe()
            if observation.tick <= last_observation:
                time.sleep(0.001)
                continue
            last_observation = observation.tick
            before = time.perf_counter()
            inference_started_ns = time.monotonic_ns()
            action = policy.predict(observation)
            inference_completed_ns = time.monotonic_ns()
            latencies.append((time.perf_counter() - before) * 1000)
            if not isinstance(action, JointResidualChunk):
                raise TypeError("real-time external policy must return JointResidualChunk")
            validate_action_source(action, observation)
            if latency_ms:
                time.sleep(latency_ms / 1000.0)
            index += 1
            event = {"source_tick": action.source_tick, "wall_time_s": time.monotonic() - started, "inference_ms": latencies[-1],
                     "inference_started_monotonic_ns": inference_started_ns, "inference_completed_monotonic_ns": inference_completed_ns,
                     "inference_completed": True,
                     "source_capture": observation.camera_metadata.get("source_capture"),
                     "clock_mapping": observation.camera_metadata.get("clock_mapping"),
                     "image_source_body_tick": observation.camera_metadata.get("image_source_body_tick"),
                     "image_capture_monotonic_ns": (int(observation.camera_metadata["capture_monotonic_s"] * 1e9)
                                                    if observation.camera_metadata.get("capture_monotonic_s") else None),
                     "image_and_proprioception_same_tick": observation.camera_metadata.get("image_and_proprioception_same_tick", False)}
            if cancelled.is_set():
                event["result"] = "discarded_after_cancel"
            elif drop_every and index % drop_every == 0:
                event["result"] = "injected_drop"
            else:
                try:
                    backend.step(action)
                    event["result"] = "accepted"
                    if event["source_capture"]:
                        age_ms = (time.monotonic_ns() - event["source_capture"]["capture_monotonic_ns"]) / 1e6
                        end_to_end.append(age_ms)
                        event["source_to_submission_ms"] = age_ms
                except RuntimeError as exc:
                    event["result"] = "rejected"
                    event["reason"] = str(exc)
            if not cancelled.is_set() and reorder_every and index % reorder_every == 0 and previous is not None:
                try:
                    backend.step(previous)
                    event["reorder_result"] = "unexpectedly_accepted"
                except RuntimeError as exc:
                    event["reorder_result"] = str(exc)
            previous = action
            events.append(event)
            query_ticks.append(observation.tick)
            query_states.append(observation.state.copy())
            source_times.append(observation.monotonic_ns)
            raw_chunks.append(action.actions.copy())
            drain_receipts()
            sample_health()
            if observation.rgb is not None:
                frames.append(observation.rgb)
                frame_ticks.append(observation.tick)
            # Replan at most every 5 control ticks. Waiting has no effect on the
            # separately running body and robotd loops.
            next_observation_ns = observation.monotonic_ns + 100_000_000
            remaining = min((next_observation_ns - time.monotonic_ns()) / 1e9, duration_s - (time.monotonic() - started))
            if remaining > 0:
                cancelled.wait(remaining)
    except Exception as exc:
        fault = f"{type(exc).__name__}: {exc}"
    finally:
        watcher_done.set()
        watcher.join(timeout=3.)
        # Every experiment relinquishes authority, also on model exceptions.
        # Physics remains running while the official controller takes over.
        try:
            if hasattr(backend, "cancel"):
                backend.cancel()
            drain_receipts()
        except Exception as exc:
            receipt_status = {**receipt_status, "complete": False, "finalization_error": f"{type(exc).__name__}: {exc}"}
            fault = fault or f"receipt_finalization: {exc}"
    final = backend.evaluator_state()
    if record_health:
        health_samples.append({"monotonic_ns": time.monotonic_ns(), "health": final["health"]})
    receipts = getattr(backend, "execution_receipts", [])
    receipt_summary = summarize_receipts(receipts, request_events=events, missing_events=receipt_status.get("missing_events", 0),
                                         unrecorded_control_ticks=receipt_status.get("unrecorded_control_ticks", 0))
    receipt_summary["complete_receipt_history"] &= receipt_status["complete"] and receipt_status["supported"]
    receipt_summary["protocol_invariants_passed"] &= receipt_summary["complete_receipt_history"]
    receipt_path = output / "execution-receipts.json"
    write_json(receipt_path, {"status": receipt_status, "receipts": receipts, "request_events": events, "cancellation": cancellation})
    report = {
        "schema_version": 2,
        "mode": "robotd_bam",
        "physics_waits_for_model": False,
        "execution_mode": getattr(backend, "execution_mode", "source_tick_age_compensated"),
        "task": backend.task.task_id,
        "scene": backend.scene,
        "policy": asdict(manifest),
        "policy_environment_validation": contract_validation,
        "latency_ms": latency_ms,
        "drop_every": drop_every,
        "reorder_every": reorder_every,
        "seed": seed,
        "events": events,
        "cancelled": cancelled.is_set(), "cancellation": cancellation,
        "fault": fault,
        "wall_time_s": time.monotonic() - started,
        "inference_p95_ms": float(np.percentile(latencies, 95)) if latencies else None,
        "inference_ms": distribution(latencies),
        "source_to_submission_p95_ms": float(np.percentile(end_to_end, 95)) if end_to_end else None,
        "source_to_consumption_measured": bool(receipt_summary["first_action_capture_to_consume_ms"]["count"]),
        "execution_receipts": {"path": receipt_path.name, "sha256": file_sha256(receipt_path), **receipt_status},
        "execution_timing": receipt_summary,
        "health_samples": health_samples,
        "execution_recipe": getattr(backend, "execution_recipe", None),
        "acceptance_scope": final.get("acceptance_scope", "task_behavior"),
        "success_protocol": final.get("success_protocol"),
        "offline_accepted": False, "realtime_accepted": False,
        "counts_as_skill_success": False,
        "note": "Single real-daemon diagnostic. Consumption is buffer selection before Safety/motor IO; no actuator-applied acknowledgement is claimed.",
        "strict_success": bool(final["success"] and fault is None and not cancelled.is_set()
                               and receipt_summary["protocol_invariants_passed"] and receipt_summary["fallback_ticks"] == 0),
        "final": serializable(final),
        "hardware_validated": False,
    }
    observations_path = output / "observations.npz"
    np.savez_compressed(observations_path, rgb=np.asarray(frames, np.uint8), tick=np.asarray(frame_ticks, np.int64),
                        query_tick=np.asarray(query_ticks, np.int64), state61=np.asarray(query_states, np.float32).reshape(-1, 61),
                        source_monotonic_ns=np.asarray(source_times, np.int64),
                        raw_action14=np.concatenate(raw_chunks) if raw_chunks else np.empty((0, 14), np.float32),
                        chunk_offsets=np.cumsum([0, *(len(chunk) for chunk in raw_chunks)], dtype=np.int64))
    report["observations"] = {"path": observations_path.name, "sha256": file_sha256(observations_path),
                              "semantics": "source observations and predicted raw chunks; consumption is recorded separately"}
    if frames:
        from PIL import Image

        Image.fromarray(frames[0]).save(output / "observation.jpg")
    write_json(output / "realtime.json", report)
    return report
