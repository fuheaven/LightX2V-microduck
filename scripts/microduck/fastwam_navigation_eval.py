"""Run a real FastWAM navigation checkpoint in the official MuJoCo+BAM scene."""

import argparse
import json
from pathlib import Path
import time


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("reference", "native"), required=True)
    parser.add_argument("--inference-process", action="store_true",
                        help="Own the model in a spawned process separate from the renderer; IPC is included in measured latency")
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--backbone", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument('--training-collection-plan', type=Path,
                        help='Explicit prospective TRAIN-only roll-in plan; never treats evaluation cases as training')
    parser.add_argument('--expected-training-collection-sha256',
                        help='Required checksum for the prospective collection plan')
    parser.add_argument("--scene", default="room_clear")
    parser.add_argument("--tasks", nargs="+", default=["find_red_ball", "find_blue_ball", "find_yellow_ball"])
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--mode", choices=("offline", "realtime"), default="offline")
    parser.add_argument("--camera-offsamples", type=int, choices=(0, 4),
                        help="Explicit MuJoCo camera sampling; omitted keeps MJCF defaults. Use the same setting for all compared backends.")
    parser.add_argument("--snapshot-camera-workers", type=int, choices=range(1, 9), default=0,
                        help="Realtime OSMesa only: render frozen physics snapshots in bounded background workers, preserving capture time")
    parser.add_argument("--async-cancel-poll-ms", type=float,
                        help="Optional realtime file-cancel polling outside the control thread; explicit interval in (0,1000]ms")
    parser.add_argument("--recover-declared-camera-loss", action="store_true",
                        help="Experimental target-memory recovery from recorded camera loss; keeps original frames, timestamps and acceptance gates")
    parser.add_argument("--inference-steps", type=int, default=20)
    parser.add_argument("--cache-prompt", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--cache-action-context", action="store_true",
                        help="Native backend: cache condition projections within each plan")
    parser.add_argument("--action-cuda-graph", action="store_true")
    parser.add_argument("--action-rms-one-pass", action="store_true",
                        help="Experimental native CUDA BF16 action RMS fusion; report closed-loop quality separately")
    parser.add_argument("--delay-ms", type=float, default=0.)
    age = parser.add_mutually_exclusive_group()
    age.add_argument("--max-action-age-ms", type=int, default=500,
                     help="Explicit observation TTL, in 20 ms multiples; applied to wall time, ticks and chunk validity")
    age.add_argument("--no-action-age-limit", dest="max_action_age_ms", action="store_const", const=None,
                     help="Simulation ablation: disable observation-age rejection; keep the finite source-aligned action horizon")
    parser.set_defaults(max_action_age_ms=500)
    parser.add_argument("--max-ticks", type=int, default=1500)
    parser.add_argument("--warmup-predictions", type=int, default=1)
    parser.add_argument("--record-frames", action="store_true")
    parser.add_argument("--record-predictions", action="store_true",
                        help="Diagnostic sensor inputs and submitted action chunks; serialized after each episode")
    parser.add_argument("--record-model-inputs", action="store_true",
                        help="Hash RGB/state/language at the model core-call boundary; requires frames and predictions")
    parser.add_argument("--record-stage-timings", action="store_true",
                        help="Diagnostic per-control-tick main-thread timing; serialized after the measurement window")
    parser.add_argument("--history-target-keyframe", action="store_true", help="Explicit RGB-only target-keyframe input; requires a matching trained receipt")
    parser.add_argument("--history-target-keyframe-fallback-ticks", type=int, choices=(300,),
                        help="Explicit v2 memory contract: keep six-second history before a target detection")
    parser.add_argument("--history-offsets-ticks", nargs="+", type=int,
                        help="Explicit trained history-grid input; realtime captures immutable snapshots at sensor cadence")
    parser.add_argument("--checkpoint-receipt", type=Path,
                        help="Required with history input to verify training, weights and normalization")
    parser.add_argument("--navigation-contract", choices=("microduck_navigation_v1", "microduck_navigation_forward04_v1"),
                        help="Optional consistency check; execution otherwise follows the trained checkpoint receipt")
    parser.add_argument("--mask-rgb", action="store_true", help="Controlled ablation: replace only model RGB with black")
    parser.add_argument("--instruction-color", choices=("red", "blue", "yellow"),
                        help="Controlled ablation: override the policy instruction; keep the evaluator's original target")
    parser.add_argument("--input-ablation-stage", choices=("observation", "prepared_request"), default="observation",
                        help="Apply before history selection (default), or after preparation to isolate model-input changes")
    parser.add_argument("--project-constant-actions", nargs="+", default=[],
                        choices=("vx_mps", "vy_mps", "yaw_rate_rps", "head_yaw_rad", "stand_probability"),
                        help="Controlled action ablation: project selected dimensions only if constant in frozen training stats")
    return parser


def main():
    args = build_parser().parse_args()
    from lightx2v_microduck.navigation import navigation_expiry_limits

    expiry = navigation_expiry_limits(args.max_action_age_ms, realtime=args.mode == "realtime")
    from lightx2v_microduck.navigation_cancellation import validate_cancel_polling
    validate_cancel_polling(args.async_cancel_poll_ms, realtime=args.mode == "realtime")
    if args.snapshot_camera_workers:
        import os
        if args.mode != "realtime" or os.environ.get("MUJOCO_GL") != "osmesa":
            raise ValueError("snapshot camera requires realtime mode and explicit MUJOCO_GL=osmesa")
    if args.record_model_inputs and not (args.record_frames and args.record_predictions):
        raise ValueError("model input recording requires camera frames and prediction records")
    if args.recover_declared_camera_loss and (
            args.mode != 'realtime' or not args.snapshot_camera_workers
            or not args.history_target_keyframe or not args.record_model_inputs
            or args.mask_rgb or args.instruction_color or args.project_constant_actions):
        raise ValueError('declared camera recovery requires realtime snapshot target memory and model input records, without other interventions')
    if args.input_ablation_stage != "observation" and not (args.mask_rgb or args.instruction_color):
        raise ValueError("prepared_request stage requires an explicit RGB or instruction intervention")
    if args.backend == "native" and args.compile:
        raise ValueError("native compilation is not implemented; requested flags cannot be ignored")
    if args.cache_action_context and args.backend != "native":
        raise ValueError("action-context caching is a native backend option")
    if args.action_cuda_graph and args.backend != "native":
        raise ValueError("action CUDA graph is a native backend option")
    if args.action_rms_one_pass and args.backend != "native":
        raise ValueError("action RMS fusion is a native backend option")
    if args.history_target_keyframe and args.history_offsets_ticks is None:
        raise ValueError("target keyframe requires explicit recent offsets")
    if args.history_target_keyframe_fallback_ticks is not None and not args.history_target_keyframe:
        raise ValueError("target keyframe fallback requires explicit target memory")
    history_spec = None
    if args.history_offsets_ticks is not None:
        from lightx2v_microduck.navigation_history import history_spec as make_history_spec, verify_evaluation_history

        history_spec = make_history_spec(tuple(args.history_offsets_ticks), target_keyframe=args.history_target_keyframe,
                                         target_keyframe_fallback_ticks=args.history_target_keyframe_fallback_ticks)
        if args.checkpoint_receipt is None:
            raise ValueError("history evaluation requires its trained checkpoint receipt")
    elif args.checkpoint_receipt is not None:
        raise ValueError("checkpoint-receipt currently requires explicit history offsets")
    from lightx2v_microduck.wam_policy import FastWAMNavigationPolicy, artifact_sha256, native_navigation_config
    from lightx2v_microduck.navigation_evaluation_contract import checkpoint_execution

    stats = args.dataset / "dataset_stats.json"
    weights_sha256, stats_sha256 = artifact_sha256(args.checkpoint), artifact_sha256(stats)
    receipt = json.loads(args.checkpoint_receipt.read_text()) if args.checkpoint_receipt else None
    execution_contract, normalization_binding = checkpoint_execution(
        receipt, weights_sha256=weights_sha256, stats_sha256=stats_sha256, requested=args.navigation_contract)
    from lightx2v_microduck.navigation_realtime_collection import load_plan, seal_recording
    training_collection = load_plan(args, weights_sha256=weights_sha256,
                                    stats_sha256=stats_sha256, entrypoint=__file__)
    args.output.mkdir(parents=True, exist_ok=False)
    from lightx2v_microduck.env import MicroDuckEnv
    from lightx2v_microduck.navigation_eval import run_navigation_episode
    from lightx2v_microduck.wam_data import write_json
    if not args.inference_process:
        import torch

    # Pin the executor, adapters and native runtime for same-weight comparisons.
    from lightx2v_microduck import navigation, navigation_eval, navigation_requests, wam_policy, wam_history_policy, navigation_history, navigation_visual_memory, env as environment, physics, rendering
    modules = (navigation, navigation_eval, navigation_requests, wam_policy,
               wam_history_policy, navigation_history, navigation_visual_memory, environment, physics, rendering)
    repo = Path(__file__).resolve().parents[2]
    source_hashes = {m.__name__: artifact_sha256(Path(m.__file__)) for m in modules}
    if args.async_cancel_poll_ms is not None:
        from lightx2v_microduck import navigation_cancellation
        source_hashes[navigation_cancellation.__name__] = artifact_sha256(Path(navigation_cancellation.__file__))
    if args.recover_declared_camera_loss:
        from lightx2v_microduck import navigation_camera_recovery
        source_hashes[navigation_camera_recovery.__name__] = artifact_sha256(Path(navigation_camera_recovery.__file__))
    if args.snapshot_camera_workers:
        from lightx2v_microduck import snapshot_camera
        source_hashes[snapshot_camera.__name__] = artifact_sha256(Path(snapshot_camera.__file__))
    if args.inference_process:
        from lightx2v_microduck import navigation_core_process, navigation_model_runtime

        for module in (navigation_core_process, navigation_model_runtime):
            source_hashes[module.__name__] = artifact_sha256(Path(module.__file__))
    if args.record_model_inputs:
        from lightx2v_microduck import navigation_model_inputs, navigation_camera_recording, navigation_camera_audit
        for module in (navigation_model_inputs, navigation_camera_recording, navigation_camera_audit):
            source_hashes[module.__name__] = artifact_sha256(Path(module.__file__))
    if args.mask_rgb or args.instruction_color:
        from lightx2v_microduck import navigation_ablation, navigation_instruction_outcome

        source_hashes[navigation_ablation.__name__] = artifact_sha256(Path(navigation_ablation.__file__))
        source_hashes[navigation_instruction_outcome.__name__] = artifact_sha256(Path(navigation_instruction_outcome.__file__))
    for relative in ("lightx2v/models/runners/wan/fastwam_runner.py",
                     "lightx2v/models/networks/wan/fastwam_model.py",
                     "lightx2v/models/networks/wan/infer/fastwam/transformer_infer.py",
                     "lightx2v/models/networks/wan/infer/fastwam/action_graph.py",
                     "lightx2v/common/ops/norm/triton_ops.py",
                     "embodied/microduck/src/lightx2v_microduck/wam_reference.py"):
        source_hashes[relative] = artifact_sha256(repo / relative)
    device_provenance = {}
    if not args.inference_process:
        device = torch.cuda.get_device_properties(torch.cuda.current_device())
        device_provenance = dict(torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
                                 gpu_name=device.name, gpu_uuid=str(getattr(device, "uuid", "")))
    runtime_provenance = {
        "source_sha256": source_hashes,
        "evaluation_script_sha256": artifact_sha256(Path(__file__)),
        "model_config_sha256": artifact_sha256(args.model_config),
        **device_provenance,
        "model_execution": "spawn" if args.inference_process else "in_process",
        "async_cancel_poll_ms": args.async_cancel_poll_ms,
        **expiry,
        "warmup_predictions": args.warmup_predictions,
    }
    started = time.monotonic()
    report = {"status": "running", "stage": "loading", "started_unix": time.time(),
              "backend": args.backend, "mode": args.mode, "scene": args.scene,
              "camera_offsamples": args.camera_offsamples,
              "snapshot_camera_workers": args.snapshot_camera_workers,
              "async_cancel_poll_ms": args.async_cancel_poll_ms,
              "runtime_provenance": runtime_provenance,
              "checkpoint": str(args.checkpoint), "checkpoint_sha256": weights_sha256,
              "stats_sha256": stats_sha256, "inference_steps": args.inference_steps,
              "navigation_contract": execution_contract, "normalization_binding": normalization_binding,
              "cache_prompt": args.cache_prompt, "compiled": args.compile,
              "action_context_cache": args.cache_action_context,
              "action_cuda_graph": args.action_cuda_graph,
              "action_rms_one_pass": args.action_rms_one_pass,
              "record_predictions": args.record_predictions,
              "record_model_inputs": args.record_model_inputs,
              "input_intervention": {"mask_rgb": args.mask_rgb, "instruction_color": args.instruction_color},
              "action_intervention": None,
              "episodes": [], "hardware_validated": False,
              "evidence": "development closed loop; not final independent acceptance"}
    report['input_history'] = history_spec.as_dict() if history_spec else None
    if training_collection is not None:
        report.update(purpose='training_rollin', training_release_allowed=False,
            training_collection_plan_sha256=training_collection.plan_sha256,
            training_admissions=[training_collection.cases[k].binding() for k in sorted(training_collection.cases)],
            evidence='TRAIN-only realtime student recording; physics replay and teacher labels pending')
    write_json(args.output / "report.json", report)
    core = None
    try:
        if history_spec:
            verify_evaluation_history(receipt, history_spec,
                                      weights_sha256=report['checkpoint_sha256'],
                                      stats_sha256=report['stats_sha256'])
            report['checkpoint_receipt_sha256'] = artifact_sha256(args.checkpoint_receipt)
        if args.inference_process:
            core = navigation_core_process.ProcessNavigationCore(
                navigation_model_runtime.build_navigation_core, backend=args.backend,
                model_config=args.model_config, checkpoint=args.checkpoint, backbone=args.backbone,
                stats=stats, history_contract=None if history_spec is None else history_spec.as_dict(),
                navigation_contract=execution_contract, normalization_binding=normalization_binding,
                inference_steps=args.inference_steps, compile_action=args.compile,
                cache_prompt=args.cache_prompt, cache_action_context=args.cache_action_context,
                action_cuda_graph=args.action_cuda_graph, action_rms_one_pass=args.action_rms_one_pass)
            runtime_provenance.update(core.runtime_provenance, model_process_pid=core.worker_pid)
            if core.reference_provenance is not None:
                report['reference_provenance'] = core.reference_provenance
            if core.resolved_config is not None:
                write_json(args.output / "resolved_config.json", core.resolved_config)
        elif args.backend == "reference":
            import inspect
            import fastwam
            from lightx2v_microduck.wam_reference import ReferenceNavigationCore
            core = ReferenceNavigationCore.from_checkpoint(
                args.model_config, args.checkpoint, stats,
                compile_action_infer=args.compile, cache_prompt=args.cache_prompt,
                inference_steps=args.inference_steps, history_spec=history_spec,
                navigation_contract=execution_contract, normalization_binding=normalization_binding)
            reference_root = Path(fastwam.__file__).resolve().parent
            report['reference_provenance'] = {
                'source_sha256': {str(p.relative_to(reference_root.parent)): artifact_sha256(p)
                                  for p in sorted(reference_root.rglob('*.py'))},
                'model_source': str(Path(inspect.getfile(type(core.model))).resolve().relative_to(reference_root.parent)),
                'adapter_sha256': source_hashes['embodied/microduck/src/lightx2v_microduck/wam_reference.py'],
            }
        else:
            import lightx2v_platform.set_ai_device  # noqa: F401
            from lightx2v.models.runners.wan.fastwam_runner import FastWAMPolicy
            config = native_navigation_config(args.backbone, args.checkpoint, stats, history_spec=history_spec,
                                              contract=execution_contract, normalization_binding=normalization_binding)
            config.update(checkpoint_mmap=True, action_infer_steps=args.inference_steps,
                          fastwam_cache_prompt=args.cache_prompt,
                          fastwam_cache_action_context=args.cache_action_context,
                          fastwam_action_cuda_graph=args.action_cuda_graph,
                          fastwam_action_rms_one_pass=args.action_rms_one_pass)
            write_json(args.output / "resolved_config.json", config)
            core = FastWAMPolicy.from_config(config)
        policy_class = FastWAMNavigationPolicy
        if history_spec:
            from lightx2v_microduck.wam_history_policy import FastWAMNavigationHistoryPolicy
            policy_class = FastWAMNavigationHistoryPolicy
        if args.recover_declared_camera_loss:
            from lightx2v_microduck.navigation_camera_recovery import FastWAMDeclaredCameraLossPolicy, CAMERA_LOSS_RECOVERY
            policy_class = FastWAMDeclaredCameraLossPolicy
            report['camera_loss_recovery'] = dict(CAMERA_LOSS_RECOVERY)
        policy = policy_class(core, model_id=args.checkpoint.parent.parent.parent.name,
                              artifact_sha256=report["checkpoint_sha256"], validity_ticks=expiry["max_age_ticks"],
                              record_model_inputs=args.record_model_inputs)
        if args.mask_rgb or args.instruction_color:
            from lightx2v_microduck.navigation_ablation import NavigationInputAblation

            policy = NavigationInputAblation(policy, mask_rgb=args.mask_rgb, instruction_color=args.instruction_color,
                                            stage=args.input_ablation_stage)
            report["input_intervention"] = policy.intervention
            report["evidence"] = "controlled input intervention; evaluator target unchanged; not capability acceptance"
        if args.project_constant_actions:
            from lightx2v_microduck.navigation_ablation import NavigationConstantActionAblation

            policy = NavigationConstantActionAblation(policy, json.loads(stats.read_text()),
                                                       dimensions=args.project_constant_actions,
                                                       stats_sha256=report["stats_sha256"])
            report["action_intervention"] = policy.intervention
            report["evidence"] = "controlled action intervention; original model identified; evaluator and expiry unchanged; not capability acceptance"
        report.update(stage="closed_loop", load_s=time.monotonic()-started)
        write_json(args.output / "report.json", report)
        with MicroDuckEnv(profile="official_bam", scene=args.scene, image_size=256,
                          camera_offsamples=args.camera_offsamples) as env:
            for seed in args.seeds:
                for task in args.tasks:
                    training_case = (training_collection.cases[task, seed]
                                     if training_collection is not None else None)
                    episode = run_navigation_episode(
                        env, policy, args.output / f"{task}-seed{seed}", task=task, seed=seed,
                        realtime=args.mode == "realtime", injected_delay_ms=args.delay_ms,
                        max_ticks=args.max_ticks, warmup_predictions=args.warmup_predictions,
                        record_frames=args.record_frames, record_predictions=args.record_predictions,
                        record_stage_timings=args.record_stage_timings, max_action_age_ms=args.max_action_age_ms,
                        snapshot_camera_workers=args.snapshot_camera_workers,
                        cancel_poll_interval_ms=args.async_cancel_poll_ms,
                        training_admission=training_case,
                        training_collection_plan_sha256=(training_collection.plan_sha256
                            if training_collection is not None else None))
                    if args.record_model_inputs:
                        from lightx2v_microduck.navigation_model_inputs import audit_recorded_inputs

                        folder = args.output / f"{task}-seed{seed}"
                        audit = audit_recorded_inputs(folder, expected_history_contract=(
                            None if history_spec is None else history_spec.as_dict()),
                            # The ordinary report keeps its historical no-op
                            # label; only an installed wrapper has a complete
                            # intervention contract at the model-call boundary.
                            expected_input_intervention=(report["input_intervention"]
                                if args.mask_rgb or args.instruction_color else None),
                            expected_camera_loss_recovery=report.get('camera_loss_recovery'))
                        write_json(folder / "model-input-audit.json", audit)
                        report.setdefault("model_input_audits", []).append({
                            "task": task, "seed": seed,
                            "path": str((folder / "model-input-audit.json").relative_to(args.output)),
                            "sha256": artifact_sha256(folder / "model-input-audit.json"),
                            "status": audit["status"],
                            "submitted_predictions": audit["submitted_predictions"],
                        })
                    if training_collection is not None:
                        seal_recording(args.output / f'{task}-seed{seed}', training_collection, training_case)
                    report["episodes"].append(episode)
                    report["strict_successes"] = sum(e["strict_model_success"] for e in report["episodes"])
                    report["realtime_successes"] = sum(e["realtime_success"] is True for e in report["episodes"])
                    report["completed_episodes"] = len(report["episodes"])
                    write_json(args.output / "report.json", report)
                    print(json.dumps({"task": task, "seed": seed, "success": episode["strict_model_success"],
                                      "latency_p95_ms": episode["model_latency_p95_ms"]}), flush=True)
                    if not episode["inference_worker_stopped"]:
                        raise RuntimeError("previous inference is still alive; refusing to reset its session")
        report.update(status="complete", stage="complete")
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        try:
            if args.inference_process:
                report['peak_memory_bytes'] = None
                if core is not None:
                    try:
                        report.update(core.runtime_metrics())
                    except Exception as exc:
                        report['runtime_metrics_error'] = f"{type(exc).__name__}: {exc}"
                    finally:
                        try:
                            report['model_process_shutdown'] = core.close()
                        except Exception as exc:
                            report.update(status="failed", shutdown_error=f"{type(exc).__name__}: {exc}")
                            raise
            else:
                report['peak_memory_bytes'] = torch.cuda.max_memory_allocated()
        finally:
            report.update(finished_unix=time.time(), allocated_gpu_seconds=time.monotonic()-started)
            write_json(args.output / "report.json", report)


if __name__ == "__main__":
    main()
