"""Compare navigation policies while the official 50 Hz controller keeps running."""

from __future__ import annotations

import json
from dataclasses import replace
import queue
import threading
import time
from pathlib import Path

import numpy as np

from .navigation import DECISION_TICKS, NAVIGATION_CONTRACT, NavigationBuffer, NavigationChunk, intent_from_action, navigation_expiry_limits
from .navigation_requests import request_methods
from .navigation_cancellation import FileCancellationMonitor, validate_cancel_polling
from .wam_data import write_json


def _latest(target, value):
    try:
        target.put_nowait(value)
    except queue.Full:
        try:
            target.get_nowait()
        except queue.Empty:
            pass
        target.put_nowait(value)


def run_navigation_episode(env, policy, output, *, task="find_red_ball", seed=0,
                           realtime=True, injected_delay_ms=0, max_ticks=1500,
                           warmup_ticks=50, initialize_plan=True,
                           warmup_predictions=0, record_frames=False,
                           record_predictions=False, reset_environment=True,
                           record_stage_timings=False, max_action_age_ms=500,
                           snapshot_camera_workers=0, cancel_poll_interval_ms=None,
                           training_admission=None, training_collection_plan_sha256=None):
    """No learned capability claim is inferred from successful process exit.

    Initialization is outside the measurement window and reported separately.
    Once started, model inference only runs on immutable snapshots in a worker.
    A slow response never blocks physical stepping or local ONNX inference.
    """
    if not np.isfinite(injected_delay_ms) or injected_delay_ms < 0 or max_ticks < 1 or warmup_predictions < 0:
        raise ValueError("invalid latency or episode duration")
    if (isinstance(snapshot_camera_workers, bool) or not isinstance(snapshot_camera_workers, int)
            or not 0 <= snapshot_camera_workers <= 8
            or (snapshot_camera_workers and not realtime)):
        raise ValueError("snapshot camera needs realtime mode and an integer worker count in 1..8")
    camera_prepare = getattr(policy, 'prepare_camera_request', None)
    if camera_prepare is not None and (not callable(camera_prepare) or not realtime or not snapshot_camera_workers):
        raise ValueError('declared camera loss recovery requires realtime snapshot camera delivery')
    if not reset_environment and (warmup_ticks != 0 or env.physics is None or env.tick != 0
                                  or env.seed != seed or env.task.task_id != task or env.terminated):
        raise ValueError("preserved-state evaluation needs a live matching task at tick zero and no physical warmup")
    expiry = navigation_expiry_limits(max_action_age_ms, realtime=realtime)
    cancel_poll_interval_ms = validate_cancel_polling(cancel_poll_interval_ms, realtime=realtime)
    training_metadata = None
    if training_admission is not None or training_collection_plan_sha256 is not None:
        from .navigation_realtime_collection import episode_metadata
        training_metadata = episode_metadata(training_admission,
            plan_sha256=training_collection_plan_sha256, policy=policy, env=env, task=task, seed=seed,
            realtime=realtime, warmup_ticks=warmup_ticks, reset_environment=reset_environment,
            max_ticks=max_ticks, record_frames=record_frames, record_predictions=record_predictions,
            record_stage_timings=record_stage_timings, snapshot_camera_workers=snapshot_camera_workers)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    requests, replies = queue.Queue(maxsize=1), queue.Queue(maxsize=1)
    stop = threading.Event()
    buffer = NavigationBuffer(**expiry,
        contract=getattr(policy, 'metadata', {}).get('contract', NAVIGATION_CONTRACT))
    prepare_request, predict_prepared = request_methods(policy)
    prediction_required = getattr(policy, "prediction_required", None)
    if prediction_required is not None and not callable(prediction_required):
        raise ValueError("prediction scheduling hook must be callable")
    scheduled_holds = []

    def should_predict(observed):
        required = True if prediction_required is None else prediction_required(observed)
        if type(required) is not bool:
            raise ValueError("prediction scheduling hook must return a boolean")
        if not required:
            scheduled_holds.append(observed.tick)
        return required
    observe_execution = getattr(policy, "observe_execution", None)
    if observe_execution is not None and not callable(observe_execution):
        raise ValueError("execution feedback hook must be callable")
    latencies, ages, lag, decisions, trace, camera_latencies = [], [], [], [], [], []
    preparation_ms = []
    worker = None
    camera = None
    error = None
    startup_ms = None
    warmup_ms = []
    worker_warmup_ms = []
    frames, frame_ticks, positions, position_ticks = [], [], [], []
    predictions = []
    stage_timings = []
    env.render_enabled = False
    if reset_environment:
        env.reset(seed, task)
        for _ in range(warmup_ticks):
            env.step(intent_from_action([0, 0, 0, 0, 1], episode_id=env.episode_id, tick=env.tick))
        env.begin_task_at_current_pose()
        if training_metadata is not None:
            # Qualified demonstrations start after the same physical standing
            # warmup. Compare that task origin, before any learned inference,
            # rather than the raw spawn pose one second earlier.
            training_admission.verify_initial(env)
            env.render_enabled = False
            write_json(output / 'collection-admission.json', dict(training_metadata,
                initial_training_observation_verified=True, admitted_before_inference=True,
                initialization='reset_and_warmup', physical_warmup_ticks=warmup_ticks,
                admission_tick=env.tick,
                admitted_unix=time.time()))
    env.render_enabled = True
    def capture():
        capture_ns = time.monotonic_ns()
        current = env.observe()
        if current.rgb is not None:
            camera_latencies.append((time.monotonic_ns() - capture_ns) / 1e6)
        # Include camera work in decision age, not just worker/transport time.
        return replace(current, monotonic_ns=capture_ns)

    observation = capture()
    policy.reset(observation.episode_id)
    buffer.reset(observation.episode_id)
    intervention = getattr(policy, "metadata", {}).get("input_intervention", {})
    override_color = intervention.get("instruction_color")
    requested_ball_body = (env.physics._id(env.physics.mj.mjtObj.mjOBJ_BODY, f"lx_ball_{override_color}")
                           if override_color is not None else None)
    requested_ball_positions = []

    def prepare(observed):
        before = time.monotonic()
        timing = {"prepare_started_ns": time.monotonic_ns()} if record_stage_timings else None
        try:
            if camera is not None and camera_prepare is not None:
                prepared = camera_prepare(observed, camera.receipts)
            else:
                prepared = prepare_request(observed)
        finally:
            preparation_ms.append((time.monotonic() - before) * 1000)
        if timing is not None:
            timing["prepare_finished_ns"] = time.monotonic_ns()
        return observed, prepared, timing

    def execution_observation(observed):
        if not realtime:
            return observed
        # The source timestamp describes capture, not the time at which an
        # action is admitted or consumed. Camera/preparation work may already
        # have exhausted its TTL. Only the buffer sees this fresh clock; do
        # not rewrite the immutable observation sent to the model. Omit image
        # arrays to avoid copying them for this metadata-only buffer call.
        return replace(observed, rgb=None, tof=None, monotonic_ns=time.monotonic_ns())

    def record_result(result, source, *, accepted, error, phase, duration_ms,
                      admission_monotonic_ns=None, request_timing=None, request_trace=None):
        receipt = {"tick": env.tick, "accepted": accepted, "error": error,
                   "reason": "prediction_error" if result is None else buffer.last_reason,
                   "source_tick": source.tick, "phase": phase,
                   "admission_monotonic_ns": admission_monotonic_ns}
        decisions.append(receipt)
        if record_predictions:
            # Retain sensor inputs and the immutable submitted chunk. Convert
            # arrays to JSON only after the measured physics interval. These
            # actions have adapter bounds applied, but precede stand decoding
            # and source-age indexing by NavigationBuffer.
            predictions.append({**receipt, "duration_ms": duration_ms,
                "source_episode_id": source.episode_id,
                "source_monotonic_ns": source.monotonic_ns,
                "camera_tick": source.camera_tick, "observed_instruction": source.instruction,
                "observed_proprio48": source.state[:48].copy(),
                "valid_until_tick": None if result is None else result.valid_until_tick,
                "action_interval_ticks": None if result is None else result.action_interval_ticks,
                "action_contract": None if result is None else result.contract,
                "submitted_actions": None if result is None else result.actions,
                "diagnostics": None if result is None else result.diagnostics,
                **({"planner_request_trace": request_trace} if request_trace is not None else {}),
                **({"request_timing": request_timing} if request_timing is not None else {})})

    def deliver_camera(*, to_policy):
        for frame in camera.poll():
            observed = frame.observation
            frame.receipt["delivery_phase"] = "measured" if to_policy else "after_measurement"
            if frame.error:
                # An unavailable frame never stops the physical clock or
                # substitutes an old image with a fresh capture timestamp.
                record_result(None, observed, accepted=False, error=frame.error,
                              phase="camera", duration_ms=0.)
                continue
            camera_latencies.append((frame.receipt["ready_ns"] - observed.monotonic_ns) / 1e6)
            if record_frames:
                frames.append(observed.rgb)
                frame_ticks.append(observed.camera_tick)
            if to_policy:
                try:
                    prepared = prepare(observed)
                except Exception as exc:
                    record_result(None, observed, accepted=False,
                                  error=f"{type(exc).__name__}: {exc}", phase="preparation",
                                  duration_ms=preparation_ms[-1])
                else:
                    _latest(requests, prepared)

    # Compilation/kernel warmup is outside the task window. Discard those
    # actions, reset session state and capture a fresh observation afterward;
    # never make a stale action look fresh by changing its source timestamp.
    for _ in range(warmup_predictions):
        before = time.monotonic()
        policy.predict(observation)
        warmup_ms.append((time.monotonic() - before) * 1000)
    if warmup_predictions:
        policy.reset(observation.episode_id)
        observation = capture()

    def infer():
        while not stop.is_set():
            try:
                request, prepared, timing = requests.get(timeout=.05)
            except queue.Empty:
                continue
            started = time.monotonic()
            request_trace = None
            if timing is not None:
                timing["worker_started_ns"] = time.monotonic_ns()
            try:
                if not should_predict(request):
                    continue
                if injected_delay_ms:
                    time.sleep(injected_delay_ms / 1000)
                if timing is not None:
                    timing["predict_started_ns"] = time.monotonic_ns()
                try:
                    result = predict_prepared(prepared)
                finally:
                    if timing is not None:
                        timing["predict_finished_ns"] = time.monotonic_ns()
                if not isinstance(result, NavigationChunk):
                    raise TypeError("navigation policy must return NavigationChunk")
                if (result.episode_id, result.source_tick, result.source_monotonic_ns) != (request.episode_id, request.tick, request.monotonic_ns):
                    raise ValueError("navigation response source differs from submitted observation")
                packet = (result, None, (time.monotonic() - started) * 1000, request)
            except Exception as exc:
                packet = (None, f"{type(exc).__name__}: {exc}", (time.monotonic() - started) * 1000, request)
                request_trace = getattr(exc, "planner_request_trace", None)
            if timing is not None:
                timing["worker_finished_ns"] = time.monotonic_ns()
            _latest(replies, (*packet, timing, request_trace))

    try:
        if snapshot_camera_workers:
            from .snapshot_camera import SnapshotCamera

            camera = SnapshotCamera(env.physics, env.episode_id, workers=snapshot_camera_workers)
            # Context warmup is complete before the task clock starts. Capture
            # again rather than carrying an initialization-aged source forward.
            observation = capture()
        if realtime:
            worker = threading.Thread(target=infer, name="microduck-navigation-inference", daemon=True)
            worker.start()
            if warmup_predictions:
                # CUDA/library state can be thread-local. Warm the actual
                # long-lived inference worker too, before the physical clock
                # starts. Its action is discarded, then re-observe honestly.
                requests.put(prepare(observation))
                _, warmup_error, duration_ms, _, _, warmup_trace = replies.get(timeout=120.)
                if warmup_error:
                    error = RuntimeError(f"inference worker warmup failed: {warmup_error}")
                    if warmup_trace is not None:
                        error.planner_request_trace = warmup_trace
                    raise error
                worker_warmup_ms.append(duration_ms)
                policy.reset(observation.episode_id)
                observation = capture()
        if initialize_plan:
            started = time.monotonic()
            first = policy.predict(observation)
            startup_ms = (time.monotonic() - started) * 1000
            if (first.episode_id, first.source_tick, first.source_monotonic_ns) != (observation.episode_id, observation.tick, observation.monotonic_ns):
                raise ValueError("startup response does not match its observation")
            admission = execution_observation(env.observe())
            accepted = buffer.submit(first, admission)
            record_result(first, observation, accepted=accepted, error=None,
                          phase="startup", duration_ms=startup_ms,
                          admission_monotonic_ns=admission.monotonic_ns)
    except Exception as exc:
        stop.set()
        if worker:
            worker.join(timeout=1.)
        if camera is not None:
            camera.close()
        write_json(output / "execution_failure.json", {
            "phase": "initialization", "error_type": type(exc).__name__, "tick": env.tick,
            "mode": "realtime" if realtime else "offline", "task": task, "seed": seed,
            "strict_model_success": False, "inference_worker_stopped": worker is None or not worker.is_alive(),
            **({"planner_request_trace": exc.planner_request_trace}
               if hasattr(exc, "planner_request_trace") else {}),
        })
        raise
    initial_observation = observation
    cancel_monitor = (FileCancellationMonitor(output / 'cancel.request', interval_ms=cancel_poll_interval_ms)
                      if cancel_poll_interval_ms is not None else None)
    cancellation_requested = (cancel_monitor.requested.is_set if cancel_monitor is not None
                              else (output / 'cancel.request').exists)
    started = time.monotonic()
    execution_failure = None
    try:
        if cancel_monitor is not None:
            cancel_monitor.start()
            started = time.monotonic()
        while not env.terminated and env.tick < max_ticks and not cancellation_requested():
            stage = ({"tick": env.tick, "capture_started_ns": time.monotonic_ns()}
                     if record_stage_timings else None)
            env.render_enabled = camera is None and env.tick % DECISION_TICKS == 0
            observation = capture()
            if camera is not None:
                deliver_camera(to_policy=True)
                if env.tick % DECISION_TICKS == 0 and not (env.tick == 0 and initialize_plan):
                    camera.submit(observation)
            if stage is not None:
                stage["capture_finished_ns"] = time.monotonic_ns()
            if record_frames:
                if camera is not None and env.tick == 0 and initialize_plan:
                    frames.append(initial_observation.rgb.copy())
                    frame_ticks.append(0)
                elif observation.rgb is not None and observation.camera_tick == env.tick:
                    frames.append(observation.rgb.copy())
                    frame_ticks.append(env.tick)
                positions.append(env.physics.data.qpos.copy())
                position_ticks.append(env.tick)
            if requested_ball_body is not None:
                # Evaluator-only copy, measured as recording overhead. The ball
                # is a free body and may move; its final position cannot stand
                # in for the positions throughout a two-second hold.
                requested_ball_positions.append(env.physics.data.xpos[requested_ball_body].copy())
            if stage is not None:
                stage["recording_finished_ns"] = time.monotonic_ns()
            if camera is None and env.tick % DECISION_TICKS == 0 and not (env.tick == 0 and initialize_plan):
                if realtime:
                    # Capture history at the camera cadence, not at model
                    # completion cadence. Dropped pending requests must not
                    # erase measured history; workers receive immutable input.
                    try:
                        prepared = prepare(observation)
                    except Exception as exc:
                        # A missing/invalid camera sample must not stop the
                        # physical clock. Existing plans expire normally and
                        # the local controller falls back if no fresh plan
                        # arrives; the failed preparation remains inspectable.
                        record_result(None, observation, accepted=False,
                                      error=f"{type(exc).__name__}: {exc}", phase="preparation",
                                      duration_ms=preparation_ms[-1])
                    else:
                        _latest(requests, prepared)
                else:
                    before = time.monotonic()
                    required = True
                    if prediction_required is None:
                        result = policy.predict(observation)
                    else:
                        # Keep histories at sensor cadence even while the
                        # model deliberately holds an existing finite plan.
                        prepared = prepare(observation)[1]
                        required = should_predict(observation)
                        result = predict_prepared(prepared) if required else None
                    if required:
                        if not isinstance(result, NavigationChunk):
                            raise TypeError("navigation policy must return NavigationChunk")
                        if (result.episode_id, result.source_tick, result.source_monotonic_ns) != (observation.episode_id, observation.tick, observation.monotonic_ns):
                            raise ValueError("offline response does not match its observation")
                        duration_ms = (time.monotonic() - before) * 1000
                        latencies.append(duration_ms)
                        # Offline reports algorithmic performance without inference
                        # wall-time expiry. It is never called realtime evidence.
                        accepted = buffer.submit(result, observation)
                        record_result(result, observation, accepted=accepted, error=None,
                                      phase="offline", duration_ms=duration_ms,
                                      admission_monotonic_ns=observation.monotonic_ns)
            if stage is not None:
                stage["request_submission_finished_ns"] = time.monotonic_ns()
            while True:
                try:
                    result, error, duration_ms, source, timing, request_trace = replies.get_nowait()
                except queue.Empty:
                    break
                if timing is not None:
                    timing["reply_received_ns"] = time.monotonic_ns()
                latencies.append(duration_ms)
                admission = execution_observation(observation)
                accepted = result is not None and buffer.submit(result, admission)
                record_result(result, source, accepted=accepted, error=error,
                              phase="realtime", duration_ms=duration_ms,
                              admission_monotonic_ns=admission.monotonic_ns if result is not None else None,
                              request_timing=timing, request_trace=request_trace)
            if stage is not None:
                stage["admission_finished_ns"] = time.monotonic_ns()
            execution = execution_observation(observation)
            checked_chunk = buffer.chunk
            intent, evidence = buffer.consume(execution)
            if "observation_age_ms" in evidence:
                ages.append(evidence["observation_age_ms"])
            trace.append({"tick": env.tick, "skill": intent.skill, "command": intent.command.tolist(),
                          "execution_monotonic_ns": execution.monotonic_ns,
                          "observation_capture_monotonic_ns": observation.monotonic_ns,
                          "checked_plan_source_monotonic_ns": None if checked_chunk is None else checked_chunk.source_monotonic_ns,
                          **evidence})
            if stage is not None:
                stage["consume_finished_ns"] = time.monotonic_ns()
            env.render_enabled = False
            after_execution = env.step(intent)
            if observe_execution is not None:
                # Actor observations and consumed intent only. Evaluator truth
                # is deliberately not accessible through this feedback hook.
                observe_execution(observation, after_execution, intent, dict(evidence), checked_chunk)
            if stage is not None:
                stage["physics_finished_ns"] = time.monotonic_ns()
            if realtime:
                remaining = started + env.tick * .02 - time.monotonic()
                if remaining > 0:
                    stop.wait(remaining)
                lag.append(max(0., time.monotonic() - started - env.tick * .02))
            if stage is not None:
                stage["pacing_finished_ns"] = time.monotonic_ns()
                stage_timings.append(stage)
        elapsed = time.monotonic() - started
        result = env.evaluator_state()
        cancelled = cancellation_requested()
        if requested_ball_body is not None:
            requested_ball_positions.append(env.physics.data.xpos[requested_ball_body].copy())
    except Exception as exc:
        # Preserve the real prefix and failure before propagating. A malformed
        # offline model response must not erase all frames/control evidence.
        execution_failure = exc
        elapsed = time.monotonic() - started
        result = env.evaluator_state()
        cancelled = cancellation_requested()
        if requested_ball_body is not None:
            requested_ball_positions.append(env.physics.data.xpos[requested_ball_body].copy())
    finally:
        stop.set()
        if cancel_monitor is not None:
            cancel_monitor.close()
        if worker:
            worker.join(timeout=1.)
        if camera is not None:
            try:
                # Finish recording already captured frames after measurement;
                # these cannot reach the model or extend the task/action clock.
                deadline = time.monotonic() + 5.
                while camera._pending and time.monotonic() < deadline:
                    deliver_camera(to_policy=False)
                    if camera._pending:
                        time.sleep(.002)
            finally:
                camera.close()
    if cancel_monitor is not None:
        cancelled = bool(cancelled or cancel_monitor.requested.is_set())
    model_success = bool(result["success"] and buffer.fallback_ticks == 0 and not cancelled and execution_failure is None)
    real_time_factor = env.tick * .02 / elapsed if elapsed > 0 else 0.
    timing_qualified = real_time_factor >= .95 and max(lag, default=0) <= .1 if realtime else None
    cancellation_metrics = cancel_monitor.metrics() if cancel_monitor is not None else None
    if cancellation_metrics is not None:
        timing_qualified = bool(timing_qualified and cancellation_metrics['worker_stopped']
                                and cancellation_metrics['error'] is None)
    camera_metrics = None
    if camera is not None:
        camera_metrics = camera.metrics()
        expected_frames = (env.tick + DECISION_TICKS - 1) // DECISION_TICKS
        camera_metrics["expected_frames"] = expected_frames
        camera_metrics["initial_synchronous_frame"] = bool(initialize_plan)
        camera_metrics["complete_camera_cadence"] = bool(
            camera_metrics["delivered_frames"] + int(initialize_plan) == expected_frames
            and not camera_metrics["failed_frames"] and not camera_metrics["dropped_frames"]
            and not camera_metrics["cancelled_ticks"] and not camera_metrics["worker_errors"]
            and camera_metrics["workers_stopped"])
        timing_qualified = bool(timing_qualified and camera_metrics["complete_camera_cadence"])
    report = {"mode": "realtime" if realtime else "offline", "task": task, "seed": seed,
              "action_contract": buffer.contract,
              "initialization": "reset_and_warmup" if reset_environment else "prepared_physical_state_new_task_clock",
              "policy": getattr(policy, "metadata", {}), "task_success": bool(result["success"]),
              "strict_model_success": model_success, "cancelled": cancelled,
              "realtime_timing_qualified": timing_qualified,
              "realtime_success": bool(model_success and timing_qualified) if realtime else None,
              "timing_gate": {"min_real_time_factor": .95, "max_clock_lag_s": .1} if realtime else None,
              "fallback_control_ticks": buffer.fallback_ticks, "rejected_plans": buffer.rejected,
              "execution_clock": {
                  "schema": "microduck_navigation_execution_clock_v2",
                  "expiry_basis": "fresh_monotonic_at_admission_and_consumption" if realtime else "offline_observation_clock",
                  "max_age_ticks": buffer.max_age_ticks,
                  "max_wall_age_ns": buffer.max_wall_age_ns,
                  **({"observation_age_rejection_enabled": False,
                      "ablation": "no_age_limit_finite_source_aligned_horizon"}
                     if not buffer.age_rejection_enabled else {}),
                  "source_capture_timestamp_preserved": True,
              },
              "warmup_inference_ms": warmup_ms,
              "worker_warmup_inference_ms": worker_warmup_ms,
              "startup_inference_ms": startup_ms, "injected_delay_ms": injected_delay_ms,
              "model_latency_p50_ms": float(np.percentile(latencies, 50)) if latencies else None,
              "model_latency_p95_ms": float(np.percentile(latencies, 95)) if latencies else None,
              "observation_age_p95_ms": float(np.percentile(ages, 95)) if ages else None,
              "camera_capture_p95_ms": float(np.percentile(camera_latencies, 95)) if camera_latencies else None,
              "request_preparation_p95_ms": float(np.percentile(preparation_ms, 95)) if preparation_ms else None,
              "request_preparation_errors": sum(d.get('phase') == 'preparation' for d in decisions),
              "scheduled_prediction_hold_ticks": scheduled_holds,
              "scheduled_holds_refresh_action_age": False,
              "wall_s": elapsed, "sim_s": env.tick * .02, "real_time_factor": real_time_factor,
              "maximum_clock_lag_s": max(lag, default=0), "control_overruns": sum(v > .02 for v in lag),
              "inference_worker_stopped": worker is None or not worker.is_alive(),
              "execution_error": None if execution_failure is None else {
                  "phase": "execution", "error_type": type(execution_failure).__name__, "tick": env.tick},
              "evaluator": result, "hardware_validated": False}
    report = json.loads(json.dumps(report, default=lambda value: value.tolist()))
    if training_metadata is not None:
        from .navigation_outcomes import sha
        report.update(training_metadata,
            training_recording_admission_sha256=sha(output / 'collection-admission.json'))
    if cancellation_metrics is not None:
        report['cancellation_monitor'] = cancellation_metrics
        report['timing_gate']['cancellation_monitor_stopped_required'] = True
        report['timing_gate']['cancellation_monitor_error_forbidden'] = True
    if hasattr(env, "camera_configuration"):
        report["camera_rendering"] = env.camera_configuration()
    if camera_metrics is not None:
        report["camera_delivery"] = camera_metrics
        write_json(output / "camera_delivery.json", camera.receipts)
        report["camera_delivery"]["receipts_path"] = "camera_delivery.json"
    if override_color is not None:
        from .navigation_instruction_outcome import requested_goal_diagnostic
        import hashlib

        # Scoring and serialization happen after the measured interval. Neither
        # robot nor ball truth is exposed to the policy or its input history.
        report["instruction_intervention_outcome"] = requested_goal_diagnostic(
            env._trajectory, target_positions=requested_ball_positions,
            requested_color=override_color, task=env.task, evaluator=result,
            fallback_control_ticks=buffer.fallback_ticks, cancelled=cancelled,
            realtime_timing_qualified=timing_qualified)
        diagnostic_path = output / "requested-goal-trace.npz"
        np.savez_compressed(diagnostic_path, robot_trajectory=np.asarray(env._trajectory),
                            target_positions=np.asarray(requested_ball_positions),
                            control_ticks=np.arange(env.tick + 1))
        report["instruction_intervention_outcome"]["trace_recording"] = {
            "path": diagnostic_path.name,
            "sha256": hashlib.sha256(diagnostic_path.read_bytes()).hexdigest(),
            "source": "evaluator_only_per_control_tick_truth",
            "position_copy_overhead_included": True,
            "scoring_and_serialization_outside_measurement": True,
        }
    if record_frames and frames:
        # Serialization happens after the measured physics interval. The
        # qpos trace is evaluator-only and never reaches the learned policy.
        np.savez_compressed(output / "recording.npz", rgb=np.stack(frames),
                            camera_ticks=np.asarray(frame_ticks), qpos=np.stack(positions),
                            physics_ticks=np.asarray(position_ticks))
        report["recording"] = "recording.npz"
    if record_predictions:
        write_json(output / "predictions.json", json.loads(json.dumps(
            predictions, default=lambda value: value.tolist())))
        report["prediction_recording"] = {
            "path": "predictions.json", "records": len(predictions),
            "actions": "bounded submitted chunk before stand decoding and age-based indexing",
            "observations": "original sensor observation; input interventions are declared in policy metadata",
            "warmup_predictions_excluded": True, "serialization_outside_measurement": True,
        }
        if realtime and record_stage_timings:
            report["prediction_recording"]["request_timing"] = {
                "schema": "microduck_navigation_request_timing_v1", "clock": "monotonic_ns",
                "scope": "Consumed realtime replies, including inference errors; startup and warmup excluded.",
                "timestamps": ["prepare_started_ns", "prepare_finished_ns", "worker_started_ns",
                               "predict_started_ns", "predict_finished_ns", "worker_finished_ns",
                               "reply_received_ns"],
                "unconsumed_requests_and_replies_excluded": True,
                "instrumentation_overhead_included": True,
            }
    if record_stage_timings:
        # Main-thread intervals only. Inference overlaps these intervals and
        # remains in predictions.json; do not add both as sequential latency.
        write_json(output / "stage_timings.json", stage_timings)
        report["stage_timing_recording"] = {
            "schema": "microduck_navigation_main_thread_stages_v1",
            "path": "stage_timings.json", "records": len(stage_timings),
            "clock": "monotonic_ns", "initialization_excluded": True,
            "serialization_outside_measurement": True,
            "scope": "Main control thread intervals; instrumentation overhead is included.",
            "stages": {
                "capture": "env.observe plus observation metadata; rendering only on sensor ticks",
                "recording": "optional RGB and qpos copies",
                "request_submission": "prepare and enqueue in realtime; predict and submit in offline mode",
                "admission": "drain available replies, validate and record admission",
                "consume": "fresh execution clock, action selection and control record",
                "physics": "env.step including local locomotion policy and physics",
                "pacing": "realtime wait and clock-lag accounting",
            },
            "inference_overlap": "Worker inference is separate and can overlap all main-thread stages.",
        }
    write_json(output / "report.json", report)
    write_json(output / "decisions.json", decisions)
    write_json(output / "control.json", trace)
    if execution_failure is not None:
        write_json(output / "execution_failure.json", {
            **report["execution_error"], "mode": report["mode"], "task": task, "seed": seed,
            "strict_model_success": False, "inference_worker_stopped": report["inference_worker_stopped"],
            "prefix_report": "report.json",
            **({"planner_request_trace": execution_failure.planner_request_trace}
               if hasattr(execution_failure, "planner_request_trace") else {}),
        })
        raise execution_failure
    return report
