"""Capture real reference/native navigation tensors on identical recorded inputs.

Run the two backends in their isolated environments. This is a numerical
diagnostic; neither an untrained model nor a tensor match proves task success.
"""

import argparse
from contextlib import ExitStack
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np


def benchmark_execution_modes(core, cases, instruction, expected, repeats, atol, rtol):
    """Separate inference-mode overhead from worker-thread overhead, no renderer."""
    import torch

    results = []
    for worker in (False, True):
        for inference in (False, True):
            core.reset()

            def predict(index):
                rgb, state = cases[index]
                with torch.inference_mode(inference):
                    torch.cuda.synchronize()
                    action = core.predict_action_chunk({"head_camera": rgb}, state, instruction, seed=0)
                    torch.cuda.synchronize()
                return action

            with ThreadPoolExecutor(max_workers=1) as executor:
                def call(index):
                    return executor.submit(predict, index).result() if worker else predict(index)
                for index in range(len(cases)):
                    call(index)
                durations, max_abs, matches = [], 0., True
                for repeat in range(repeats):
                    index = repeat % len(cases)
                    start = time.perf_counter()
                    action = call(index)
                    durations.append((time.perf_counter() - start) * 1000)
                    max_abs = max(max_abs, float(np.max(np.abs(action - expected[f"{index}.action"]))))
                    matches &= bool(np.allclose(action, expected[f"{index}.action"], atol=atol, rtol=rtol))
            row = {"worker_thread": worker, "inference_mode": inference, "calls": repeats,
                   "p50_ms": float(np.percentile(durations, 50)), "p95_ms": float(np.percentile(durations, 95)),
                   "milliseconds": durations, "action_max_abs": max_abs, "actions_within_tolerance": matches,
                   "scope": "fixed recorded input, no concurrent physics/rendering; includes worker submission when enabled"}
            results.append(row)
            print(json.dumps({key: value for key, value in row.items() if key != "milliseconds"}), flush=True)
            if not matches:
                raise ValueError("execution mode changed fixed-seed actions")
    return results


def profile_prediction(core, backend, case, instruction, output):
    """Inspect one warm call separately from the uninstrumented latency sample."""
    import torch

    if backend == "reference":
        methods = [(core.model, "encode_prompt", "text_encoder"),
                   (core.model, "_encode_input_image_latents_tensor", "image_vae"),
                   (core.model.video_expert, "prepare", "video_prepare"),
                   (core.model.mot, "prefill_video_cache_tensor", "video_prefill"),
                   (core.model, "_denoise_action_with_video_cache", "action_denoise")]
    else:
        methods = [(core, "encode_prompt", "text_encoder"),
                   (core, "encode_image_latents", "image_vae"),
                   (core.model, "prepare_action_inputs", "video_prefill_and_state"),
                   (core, "_run_action_denoising", "action_denoise_loop")]

    def instrument(original, name):
        def call(*args, **kwargs):
            with torch.profiler.record_function(f"navigation::{name}"):
                return original(*args, **kwargs)
        return call

    rgb, state = case
    torch.cuda.synchronize()
    with ExitStack() as stack, torch.inference_mode():
        for obj, method, name in methods:
            stack.enter_context(patch.object(obj, method, instrument(getattr(obj, method), name)))
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as profiler:
            action = core.predict_action_chunk({"head_camera": rgb}, state, instruction, seed=0)
            torch.cuda.synchronize()
    profiler.export_chrome_trace(str(output / "profile-trace.json"))
    events = profiler.key_averages()
    stages = {}
    for event in events:
        if event.key.startswith("navigation::"):
            # Profiler may emit CPU and CUDA groups with the same name. Keep
            # both: a dict keyed only by name would silently overwrite CPU time.
            stages.setdefault(event.key, []).append({
                "device_type": str(event.device_type), "calls": event.count,
                "cpu_total_ms": event.cpu_time_total / 1000,
                "device_total_ms": event.device_time_total / 1000})
    (output / "profile-operators.txt").write_text(events.table(sort_by="self_device_time_total", row_limit=30))
    return action, {"stages": stages, "trace": "profile-trace.json", "operators": "profile-operators.txt",
                    "instrumentation_changes_timing": True,
                    "scope": "one instrumented warm call; excludes latency percentiles and is not task evidence"}


def check_prompt_cache(core, cases, atol, rtol):
    """Compare real-model actions with caching off/on and after session reset."""
    from lightx2v_microduck.tasks import get_task

    comparisons = []
    try:
        uncached = []
        core.config["fastwam_cache_prompt"] = False
        for index, color in enumerate(("red", "blue", "yellow", "red")):
            rgb, state = cases[index % len(cases)]
            instruction = get_task(f"find_{color}_ball").instruction
            uncached.append(core.predict_action_chunk({"head_camera": rgb}, state, instruction, seed=index + 1))
        core.config["fastwam_cache_prompt"] = True
        for index, color in enumerate(("red", "blue", "yellow", "red")):
            rgb, state = cases[index % len(cases)]
            instruction = get_task(f"find_{color}_ball").instruction
            expected = uncached[index]
            for condition in ("new_instruction", "cache_hit", "session_reset"):
                if condition == "session_reset":
                    core.reset()
                actual = core.predict_action_chunk({"head_camera": rgb}, state, instruction, seed=index + 1)
                row = {"color": color, "seed": index + 1, "condition": condition,
                       "max_abs": float(np.max(np.abs(expected - actual))),
                       "allclose": bool(np.allclose(expected, actual, atol=atol, rtol=rtol))}
                comparisons.append(row)
                if not row["allclose"]:
                    raise ValueError("prompt cache changed fixed-input action after instruction/session change")
    finally:
        core.reset()
    return comparisons


def benchmark_prompt_cache(core, cases, instruction, expected, repeats, atol, rtol):
    """Interleave warm cache-on/off calls in one process with identical inputs."""
    import torch

    timings = {False: [], True: []}
    maximum_error = 0.
    try:
        with torch.inference_mode():
            for repeat in range(repeats):
                index = repeat % len(cases)
                rgb, state = cases[index]
                for enabled in ((False, True) if repeat % 2 == 0 else (True, False)):
                    core.config["fastwam_cache_prompt"] = enabled
                    # Both variants get an unmeasured warm call. The cache-on
                    # sample therefore measures a genuinely unchanged prompt.
                    core.predict_action_chunk({"head_camera": rgb}, state, instruction, seed=0)
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    action = core.predict_action_chunk({"head_camera": rgb}, state, instruction, seed=0)
                    torch.cuda.synchronize()
                    timings[enabled].append((time.perf_counter() - start) * 1000)
                    maximum_error = max(maximum_error, float(np.max(np.abs(action - expected[f"{index}.action"]))))
                    if not np.allclose(action, expected[f"{index}.action"], atol=atol, rtol=rtol):
                        raise ValueError("prompt-cache benchmark changed fixed-input actions")
    finally:
        core.config["fastwam_cache_prompt"] = True
        core.reset()
    variants = {"enabled" if enabled else "disabled": {
        "calls": len(values), "milliseconds": values,
        "p50_ms": float(np.percentile(values, 50)), "p95_ms": float(np.percentile(values, 95))}
        for enabled, values in timings.items()}
    return {"variants": variants, "action_max_abs": maximum_error, "all_actions_within_tolerance": True,
            "p95_reduction_fraction": 1 - variants["enabled"]["p95_ms"] / variants["disabled"]["p95_ms"],
            "scope": "alternating warm calls on recorded inputs; no continuous physics or cold-start claim"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("reference", "native"), required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--backbone", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--episode", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--frame-indices", nargs="+", type=int, default=[0, 30, 60])
    parser.add_argument("--history-target-keyframe", action="store_true")
    parser.add_argument("--history-target-keyframe-fallback-ticks", type=int, choices=(300,),
                        help="Explicit v2 memory contract: keep six-second history before a target detection")
    parser.add_argument("--history-offsets-ticks", nargs="+", type=int,
                        help="Explicit causal RGB grid experiment; old weights are initialization, not history-trained capability")
    # BF16 diagnostics are reported separately from any future FP32 gate.
    parser.add_argument("--atol", type=float, default=.01)
    parser.add_argument("--rtol", type=float, default=.01)
    parser.add_argument("--benchmark-repeats", type=int, default=0,
                        help="After alignment, measure repeated sensor-to-action latency on the recorded frames")
    parser.add_argument("--profile-call", action="store_true",
                        help="Separately profile a warm call to locate the inference bottleneck")
    parser.add_argument("--capture-network-inputs", action="store_true",
                        help="Diagnostic only: capture actual BF16 VAE/proprio encoder inputs during prediction")
    parser.add_argument("--cache-action-context", action="store_true",
                        help="Native backend: reuse condition projections only within each action plan")
    parser.add_argument("--cache-prompt", action="store_true",
                        help="Native backend: reuse one unchanged instruction within its session")
    parser.add_argument("--benchmark-execution-modes", action="store_true",
                        help="Native backend: compare main/worker threads with inference mode on/off")
    parser.add_argument("--action-cuda-graph", action="store_true",
                        help="Native backend: capture the unchanged action denoising loop")
    parser.add_argument("--action-rms-one-pass", action="store_true",
                        help="Experimental native CUDA BF16 action RMS fusion; requires numerical validation")
    args = parser.parse_args()
    if args.capture_network_inputs and (args.benchmark_repeats or args.profile_call or args.benchmark_execution_modes):
        raise ValueError("network-input capture changes timing; run it separately from performance measurements")
    if args.benchmark_repeats and args.benchmark_repeats < 20:
        raise ValueError("use zero or at least 20 calls for a latency diagnostic")
    if args.cache_action_context and args.backend != "native":
        raise ValueError("action-context caching is a native backend option")
    if args.cache_prompt and args.backend != "native":
        raise ValueError("this prompt-cache alignment diagnostic requires the native backend")
    if args.action_cuda_graph and args.backend != "native":
        raise ValueError("action CUDA graph is a native backend option")
    if args.action_rms_one_pass and args.backend != "native":
        raise ValueError("action RMS fusion is a native backend option")
    if args.benchmark_execution_modes and (args.backend != "native" or args.benchmark_repeats < 20):
        raise ValueError("execution-mode comparison requires native backend and at least 20 repeats")
    args.output.mkdir(parents=True, exist_ok=False)
    from lightx2v_microduck.wam_data import PROMPT, write_json
    from lightx2v_microduck.wam_policy import artifact_sha256, native_navigation_config
    from lightx2v_microduck.navigation_history import history_spec as make_history_spec, compose_history_grid
    from lightx2v_microduck.navigation_visual_memory import target_memory_ticks, compose_target_memory_grid
    import torch

    if args.history_target_keyframe and args.history_offsets_ticks is None:
        raise ValueError("target keyframe requires explicit recent offsets")
    if args.history_target_keyframe_fallback_ticks is not None and not args.history_target_keyframe:
        raise ValueError("target keyframe fallback requires explicit target memory")
    history_spec = None if args.history_offsets_ticks is None else make_history_spec(
        tuple(args.history_offsets_ticks), target_keyframe=args.history_target_keyframe,
        target_keyframe_fallback_ticks=args.history_target_keyframe_fallback_ticks)

    started = time.monotonic()
    episode = args.dataset / args.episode
    metadata = json.loads((episode / "episode.json").read_text())
    if artifact_sha256(episode / "navigation.npz") != metadata["files"]["navigation.npz"]:
        raise ValueError("recorded input checksum mismatch")
    stats = args.dataset / "dataset_stats.json"
    report = {"backend": args.backend, "status": "running", "started_unix": time.time(),
              "checkpoint": str(args.checkpoint), "checkpoint_bytes": args.checkpoint.stat().st_size,
              "checkpoint_sha256": artifact_sha256(args.checkpoint),
              "input_sha256": metadata["files"]["navigation.npz"],
              "stats_sha256": artifact_sha256(stats), "seed": 0, "inference_steps": 20,
              "dtype": "bfloat16", "compiled": False, "prompt_cache": args.cache_prompt,
              "action_context_cache": args.cache_action_context,
              "action_cuda_graph": args.action_cuda_graph,
              "action_rms_one_pass": args.action_rms_one_pass,
              "frame_indices": args.frame_indices, "not_capability_acceptance": True}
    report['input_history'] = None if history_spec is None else history_spec.as_dict()
    write_json(args.output / "report.json", report)
    try:
        if args.backend == "reference":
            from lightx2v_microduck.wam_reference import ReferenceNavigationCore

            core = ReferenceNavigationCore.from_checkpoint(args.model_config, args.checkpoint, stats, history_spec=history_spec)
        else:
            import lightx2v_platform.set_ai_device  # noqa: F401
            from lightx2v.models.runners.wan.fastwam_runner import FastWAMPolicy

            config = native_navigation_config(args.backbone, args.checkpoint, stats, history_spec=history_spec)
            config["checkpoint_mmap"] = True
            config["fastwam_cache_action_context"] = args.cache_action_context
            config["fastwam_cache_prompt"] = args.cache_prompt
            config["fastwam_action_cuda_graph"] = args.action_cuda_graph
            config["fastwam_action_rms_one_pass"] = args.action_rms_one_pass
            core = FastWAMPolicy.from_config(config)
            write_json(args.output / "resolved_config.json", config)
        report["load_s"] = time.monotonic() - started
        snapshots = {}
        network_snapshots = {}
        cases = []
        with np.load(episode / "navigation.npz") as data, torch.inference_mode():
            memory_ticks = target_memory_ticks(data["rgb"], data["ticks"], metadata["instruction"]) if args.history_target_keyframe else None
            for i, frame in enumerate(args.frame_indices):
                if frame < 0 or frame >= len(data["action"]):
                    raise ValueError(f"frame {frame} is outside the episode")
                rgb, state = data["rgb"][frame], data["proprio"][frame]
                if history_spec is not None:
                    if args.history_target_keyframe:
                        rgb, history_ticks, history_padding = compose_target_memory_grid(data['rgb'], data['ticks'], int(data['ticks'][frame]), history_spec, memory_ticks)
                    else:
                        rgb, history_ticks, history_padding = compose_history_grid(data['rgb'], data['ticks'], int(data['ticks'][frame]), history_spec)
                    report.setdefault('history_samples', []).append({'frame': frame, 'source_ticks': history_ticks.tolist(),
                                                                   'padding': history_padding.tolist()})
                cases.append((rgb.copy(), state.copy()))
                images = {"head_camera": rgb}
                prompt = PROMPT.format(task=metadata["instruction"])
                if args.backend == "reference":
                    image = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().div(127.5).sub(1)
                    image = image.to(core.model.device, dtype=core.model.torch_dtype)
                    latents = core.model._encode_input_image_latents_tensor(input_image=image.unsqueeze(0), tiled=False)
                    context, mask = core.model.encode_prompt(prompt)
                    proprio = core.processor.normalizer.normalizers["state"]["default"].forward(torch.from_numpy(state))
                else:
                    image = core.build_image_tensor(images)
                    latents = core.encode_image_latents(image)
                    context, mask = core.encode_prompt(prompt)
                    proprio = core.state_normalizer.forward(state)
                for key, tensor in (("image", image), ("latents", latents), ("context", context),
                                    ("context_mask", mask), ("state", proprio)):
                    value = tensor.detach().float().cpu().numpy()
                    if key in ("latents", "context", "context_mask") and value.shape[0] == 1:
                        value = value.squeeze(0)
                    snapshots[f"{i}.{key}"] = value
                if args.capture_network_inputs:
                    from lightx2v_microduck.navigation_network_inputs import capture_prediction

                    captured, tensors, action = capture_prediction(
                        core, args.backend, rgb, np.asarray(state, np.float32), metadata['instruction'],
                        json.loads(stats.read_text()), seed=0)
                    report.setdefault('network_input_captures', []).append(dict(frame=frame, **captured))
                    for name, tensor in tensors.items():
                        network_snapshots[f'{i}.{name}'] = tensor
                else:
                    action = core.predict_action_chunk(images, state, metadata["instruction"], seed=0)
                if not np.isfinite(action).all():
                    raise ValueError("nonfinite real-model navigation action")
                snapshots[f"{i}.action"] = action
                print(json.dumps({"backend": args.backend, "frame": frame, "finished_s": time.monotonic()-started}), flush=True)
        np.savez_compressed(args.output / "tensors.npz", **snapshots)
        if args.capture_network_inputs:
            np.savez_compressed(args.output / 'network_inputs.npz', **network_snapshots)
            report['network_input_capture'] = dict(
                sampled_predictions=len(report['network_input_captures']),
                normalized_encoder_inputs_match=all(x['normalized_encoder_inputs_match']
                    and x['all_inputs_on_cuda'] for x in report['network_input_captures']),
                tensors_sha256=artifact_sha256(args.output / 'network_inputs.npz'),
                diagnostic_changes_timing=True, historical_realtime_inputs_verified=False,
                fresh_realtime_measurement=False, formal_acceptance=False)
            from lightx2v_microduck import navigation_network_inputs
            report['network_input_capture']['capture_source_sha256'] = artifact_sha256(navigation_network_inputs.__file__)
        report["peak_memory_bytes"] = torch.cuda.max_memory_allocated()
        if args.benchmark_repeats:
            latencies, max_action_diff, repeat_actions_match = [], 0., True
            report["stage"] = "latency_diagnostic"
            write_json(args.output / "report.json", report)
            # Alignment already exercised every input. Keep the warm steady
            # timing separate from loading and from actual closed-loop time.
            with torch.inference_mode():
                for repeat in range(args.benchmark_repeats):
                    i = repeat % len(cases)
                    rgb, state = cases[i]
                    torch.cuda.synchronize()
                    tick = time.perf_counter()
                    action = core.predict_action_chunk({"head_camera": rgb}, state, metadata["instruction"], seed=0)
                    torch.cuda.synchronize()
                    latencies.append((time.perf_counter() - tick) * 1000)
                    if not np.isfinite(action).all():
                        raise FloatingPointError("nonfinite action during timing")
                    max_action_diff = max(max_action_diff, float(np.max(np.abs(action - snapshots[f"{i}.action"]))))
                    repeat_actions_match &= bool(np.allclose(action, snapshots[f"{i}.action"], atol=args.atol, rtol=args.rtol))
            report["latency_diagnostic"] = {
                "calls": len(latencies), "milliseconds": latencies,
                "p50_ms": float(np.percentile(latencies, 50)), "p95_ms": float(np.percentile(latencies, 95)),
                "max_ms": max(latencies), "repeat_action_max_abs": max_action_diff,
                "repeat_actions_within_tolerance": repeat_actions_match,
                "input_kind": "fixed recorded observations; not continuous-physics evidence",
                "gpu_name": torch.cuda.get_device_name(),
                "peak_memory_bytes": torch.cuda.max_memory_allocated(),
            }
            if not repeat_actions_match:
                raise ValueError("fixed-seed repeated actions changed during the latency diagnostic")
        if args.benchmark_execution_modes:
            report["stage"] = "execution_mode_diagnostic"
            write_json(args.output / "report.json", report)
            report["execution_mode_diagnostic"] = benchmark_execution_modes(
                core, cases, metadata["instruction"], snapshots, args.benchmark_repeats, args.atol, args.rtol)
        if args.action_cuda_graph:
            from lightx2v_microduck.tasks import get_task

            # Reuse captured addresses while changing language, proprioception,
            # image and sample noise. A cached old observation must not pass.
            paired = []
            for index, color in enumerate(("red", "blue", "yellow")):
                rgb, state = cases[index % len(cases)]
                instruction = get_task(f"find_{color}_ball").instruction
                core.config["fastwam_action_cuda_graph"] = False
                eager = core.predict_action_chunk({"head_camera": rgb}, state, instruction, seed=index + 1)
                core.config["fastwam_action_cuda_graph"] = True
                graphed = core.predict_action_chunk({"head_camera": rgb}, state, instruction, seed=index + 1)
                row = {"color": color, "seed": index + 1,
                       "max_abs": float(np.max(np.abs(eager - graphed))),
                       "allclose": bool(np.allclose(eager, graphed, atol=args.atol, rtol=args.rtol))}
                paired.append(row)
                if not row["allclose"]:
                    raise ValueError("CUDA graph changed updated observation/language/noise action")
            report["graph_fresh_input_comparison"] = paired
        if args.cache_prompt:
            report["prompt_cache_comparison"] = check_prompt_cache(core, cases, args.atol, args.rtol)
            if args.benchmark_repeats:
                report["prompt_cache_latency_comparison"] = benchmark_prompt_cache(
                    core, cases, metadata["instruction"], snapshots, args.benchmark_repeats, args.atol, args.rtol)
        if args.profile_call:
            report["stage"] = "profiling"
            write_json(args.output / "report.json", report)
            # Prime this exact instruction again: the fresh-input checks above
            # deliberately changed it and reset the session.
            core.predict_action_chunk({"head_camera": cases[0][0]}, cases[0][1], metadata["instruction"], seed=0)
            action, diagnostic = profile_prediction(core, args.backend, cases[0], metadata["instruction"], args.output)
            diagnostic["action_max_abs"] = float(np.max(np.abs(action - snapshots["0.action"])))
            diagnostic["actions_within_tolerance"] = bool(np.allclose(action, snapshots["0.action"], atol=args.atol, rtol=args.rtol))
            report["profile_diagnostic"] = diagnostic
            if not diagnostic["actions_within_tolerance"]:
                raise ValueError("profiling changed the fixed-seed navigation action")
        if args.reference:
            reference_report = json.loads((args.reference / "report.json").read_text())
            if args.capture_network_inputs:
                saved = reference_report.get('network_input_capture', {})
                if (saved.get('normalized_encoder_inputs_match') is not True
                        or saved.get('sampled_predictions') != len(args.frame_indices)
                        or saved.get('tensors_sha256') != artifact_sha256(args.reference / 'network_inputs.npz')):
                    raise ValueError('reference lacks matching actual network-input evidence')
                with np.load(args.reference / 'network_inputs.npz', allow_pickle=False) as captured:
                    if set(captured.files) != set(network_snapshots):
                        raise ValueError('actual encoder capture inventories differ')
                    for key, value in network_snapshots.items():
                        np.testing.assert_array_equal(value, captured[key], err_msg='actual encoder tensor: '+key)
                report['network_input_capture']['reference_inputs_bitwise_equal'] = True
            if reference_report.get('input_history') != report['input_history'] or reference_report.get('history_samples') != report.get('history_samples'):
                raise ValueError('reference comparison uses a different observed history')
            for key in ("checkpoint", "checkpoint_bytes", "checkpoint_sha256", "input_sha256", "stats_sha256", "seed", "inference_steps", "frame_indices"):
                if reference_report[key] != report[key]:
                    raise ValueError(f"reference comparison differs in {key}")
            comparisons = {}
            with np.load(args.reference / "tensors.npz") as reference:
                for key, value in snapshots.items():
                    expected = reference[key]
                    if value.shape != expected.shape:
                        raise ValueError(f"reference shape mismatch for {key}: {expected.shape} versus {value.shape}")
                    diff = np.abs(value-expected)
                    comparisons[key] = {"max_abs": float(diff.max()), "mean_abs": float(diff.mean()),
                                        "allclose": bool(np.allclose(value, expected, atol=args.atol, rtol=args.rtol))}
            report["comparison"] = comparisons
            report["diagnostic_tolerance"] = {"atol": args.atol, "rtol": args.rtol, "precision": "BF16"}
            report["all_tensors_within_diagnostic_tolerance"] = all(x["allclose"] for x in comparisons.values())
        report.update(status="complete", stage="complete")
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["allocated_gpu_seconds"] = time.monotonic() - started
        report["finished_unix"] = time.time()
        write_json(args.output / "report.json", report)


if __name__ == "__main__":
    main()
