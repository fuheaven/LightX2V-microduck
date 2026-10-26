"""Portable deployment and fixed-input profiling of the retained MicroDuck model.

No training or model download happens on import. Run --help for each command.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "embodied/microduck/src"))
sys.path.insert(0, str(REPO))
os.environ.setdefault("LIGHTX2V_IMPORT_MODE", "robotics")
os.environ.setdefault("DTYPE", "BF16")
DEPLOY = REPO / "deploy/microduck/fastwam"
LOCK = DEPLOY / "model-files.lock.json"


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def verify(data, *, include_backbone=True):
    """No silent fallback to public manipulation weights or replacement stats."""
    lock = json.loads(LOCK.read_text())
    checked = {}
    for name, spec in lock["files"].items():
        if not include_backbone and spec["distribution"] == "public_backbone":
            continue
        path = data / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}; see docs/microduck/deployment.md")
        if path.stat().st_size != spec["bytes"] or sha256(path) != spec["sha256"]:
            raise ValueError(f"Model file differs from the selected artifact: {path}")
        checked[name] = spec["sha256"]
    from lightx2v_microduck.navigation_history import history_spec, verify_evaluation_history
    receipt = json.loads((data / "models/microduck/navigation_checkpoint.json").read_text())
    verify_evaluation_history(receipt, history_spec((300, 150, 50, 0)),
                              weights_sha256=checked["models/microduck/model.pt"],
                              stats_sha256=checked["models/microduck/dataset_stats.json"])
    return checked


def fetch(args):
    lock = json.loads(LOCK.read_text())
    if args.component in ("model", "all"):
        bundle = lock.get("published_bundle", {})
        if not bundle.get("repo") or not re.fullmatch(r"[0-9a-f]{40}", bundle.get("revision", "")):
            raise ValueError("No pinned model release in model-files.lock.json; update the deployment checkout")
    if args.component in ("sim", "all"):
        from lightx2v_microduck.assets import fetch_source
        for source in ("microduck_rl", "microduck-policies"):
            result = fetch_source(source, args.data / "upstream")
            if not result["ok"]:
                raise ValueError(result)
            print(f"Verified {source}", flush=True)
    if args.component in ("model", "all"):
        from huggingface_hub import snapshot_download
        names = [name for name, spec in lock["files"].items()
                 if args.component == "all" or spec["distribution"] != "public_backbone"]
        snapshot_download(repo_id=bundle["repo"], revision=bundle["revision"],
                          allow_patterns=names, local_dir=args.data)
        checked = verify(args.data, include_backbone=args.component == "all")
        print(f"Verified {len(checked)} model files from {bundle['repo']}@{bundle['revision']}", flush=True)
    if args.component == "backbone":
        from huggingface_hub import snapshot_download
        names = [p.removeprefix("models/Wan2.2-TI2V-5B/") for p, spec in lock["files"].items()
                 if spec["distribution"] == "public_backbone"]
        snapshot_download(repo_id=lock["public_backbone"]["repo"],
                          revision=lock["public_backbone"]["revision"], allow_patterns=names,
                          local_dir=args.data / "models/Wan2.2-TI2V-5B")
        for name, spec in lock["files"].items():
            if spec["distribution"] == "public_backbone" and sha256(args.data / name) != spec["sha256"]:
                raise ValueError(f"Backbone checksum mismatch: {name}")
        print("Verified minimal public backbone. Use fetch --component model for the MicroDuck checkpoint.")


def inputs(args):
    """Capture real camera/state samples, not a task-success demonstration."""
    import numpy as np
    from lightx2v_microduck.env import MicroDuckEnv
    from lightx2v_microduck.navigation import student_state
    from lightx2v_microduck.navigation_history import compose_history_grid, history_spec
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    grids, states, instructions = [], [], []
    env = MicroDuckEnv(profile="official_bam", scene="room_clear", asset_root=args.data / "upstream")
    try:
        for color in ("red", "blue", "yellow"):
            observation = env.reset(seed=2000, task=f"find_{color}_ball")
            # Valid episode-start padding under the actual trained history contract.
            grid, _, _ = compose_history_grid(np.asarray([observation.rgb]), np.array([0]),
                                               0, history_spec((300, 150, 50, 0)))
            grids.append(grid)
            states.append(student_state(observation))
            instructions.append(observation.instruction)
        with args.output.open("xb") as stream:
            np.savez_compressed(stream, rgb=np.stack(grids), state=np.stack(states),
                                instruction=np.asarray(instructions))
        write_json(args.output.with_suffix(".json"), {
            "purpose": "real simulator episode-start inputs; performance/parity only, not task acceptance",
            "sha256": sha256(args.output), "seed": 2000, "source_tick": 0,
            "history": history_spec((300, 150, 50, 0)).as_dict(), "hardware_validated": False})
        print(f"Saved three real camera/state/language inputs: {args.output}")
    finally:
        env.close()


def config_for(data, variant):
    from lightx2v_microduck.navigation_history import history_spec
    from lightx2v_microduck.wam_policy import native_navigation_config
    config = native_navigation_config(data / "models/Wan2.2-TI2V-5B", data / "models/microduck/model.pt",
                                      data / "models/microduck/dataset_stats.json",
                                      history_spec=history_spec((300, 150, 50, 0)))
    config.update(checkpoint_mmap=True, action_infer_steps=20,
                  fastwam_cache_prompt=variant in ("prompt", "context", "optimized"),
                  fastwam_cache_action_context=variant in ("context", "optimized"),
                  fastwam_action_cuda_graph=variant == "optimized")
    return config


def benchmark(args):
    import numpy as np
    checked = verify(args.data)
    if args.repeats < 20:
        raise ValueError("Use at least 20 repeats per input for a reported p95")
    args.output.mkdir(parents=True, exist_ok=False)
    with np.load(args.inputs, allow_pickle=False) as sample:
        rgb, state, instructions = sample["rgb"], sample["state"], sample["instruction"]
    if (rgb.dtype != np.uint8 or rgb.shape != (len(state), 512, 512, 3)
            or state.shape != (len(rgb), 48) or len(instructions) != len(rgb)
            or not len(rgb) or not np.isfinite(state).all()):
        raise ValueError("Expected N real history RGB uint8[512,512,3], finite state[48], instruction strings")
    import torch
    import lightx2v_platform.set_ai_device  # noqa: F401
    from lightx2v.models.runners.wan.fastwam_runner import FastWAMPolicy
    config = config_for(args.data, args.variant)
    write_json(args.output / "resolved_config.json", config)
    started = time.monotonic()
    core = FastWAMPolicy.from_config(config)
    torch.cuda.synchronize()
    load_seconds = time.monotonic() - started
    outputs, timings, cold = [], [], []
    torch.cuda.reset_peak_memory_stats()
    try:
        for image, proprio, instruction in zip(rgb, state, instructions):
            core.reset()  # Invalidate prompt and session action state between instructions.
            def call():
                return core.predict_action_chunk({"head_camera": image}, proprio, str(instruction), seed=0)
            torch.cuda.synchronize()
            start = time.perf_counter()
            first = call()
            torch.cuda.synchronize()
            cold.append((time.perf_counter() - start) * 1000)
            call()  # second warm-up, not counted in steady samples
            torch.cuda.synchronize()
            times = []
            for _ in range(args.repeats):
                start = time.perf_counter()
                action = call()
                torch.cuda.synchronize()
                times.append((time.perf_counter() - start) * 1000)
                if action.shape != (32, 5) or not np.isfinite(action).all():
                    raise ValueError("Invalid action chunk")
                np.testing.assert_allclose(action, first, atol=1e-2, rtol=1e-2)
            outputs.append(action)
            timings.append(times)
        actions = np.stack(outputs)
        np.save(args.output / "actions.npy", actions, allow_pickle=False)
        report = {
            "schema": "microduck_fastwam_fixed_input_benchmark_v1", "variant": args.variant,
            "scope": "model call only; excludes simulation, network, queueing; not task acceptance",
            "inputs_sha256": sha256(args.inputs), "files": checked, "load_seconds": load_seconds,
            "cold_call_ms_per_input": cold, "steady_call_ms_per_input": timings,
            "steady_p50_ms": float(np.percentile(timings, 50)),
            "steady_p95_ms": float(np.percentile(timings, 95)),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__),
            "cuda": torch.version.cuda, "precision": os.environ["DTYPE"],
            "seed": 0, "inference_steps": 20, "hardware_validated": False,
            "source_sha256": {p: sha256(REPO / p) for p in (
                "scripts/microduck/fastwam_handoff.py", "lightx2v/models/runners/wan/fastwam_runner.py",
                "lightx2v/models/networks/wan/fastwam_model.py",
                "lightx2v/models/networks/wan/infer/fastwam/action_graph.py")},
        }
        if args.reference:
            previous = json.loads((args.reference / "report.json").read_text())
            for field in ("inputs_sha256", "files", "precision", "seed", "inference_steps"):
                if report[field] != previous[field]:
                    raise ValueError(f"Reference uses different {field}")
            reference = np.load(args.reference / "actions.npy", allow_pickle=False)
            report["parity"] = {"atol": 1e-2, "rtol": 1e-2,
                "max_abs_error": float(np.max(np.abs(actions - reference))),
                "passed": bool(np.allclose(actions, reference, atol=1e-2, rtol=1e-2))}
        write_json(args.output / "report.json", report)
        if args.profile:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as profiler:
                call()
                torch.cuda.synchronize()
            profiler.export_chrome_trace(str(args.output / "trace.json"))
        print(json.dumps({k: report[k] for k in ("variant", "steady_p50_ms", "steady_p95_ms", "peak_allocated_bytes")}))
        if report.get("parity", {}).get("passed") is False:
            raise ValueError("Numerical regression against reference; see report.json")
    finally:
        core.close()


def eval_command(args):
    command = [sys.executable, str(REPO / "scripts/microduck/fastwam_navigation_eval.py"),
        "--backend", "native", "--model-config", str(DEPLOY / "recipe.json"),
        "--checkpoint", str(args.data / "models/microduck/model.pt"),
        "--backbone", str(args.data / "models/Wan2.2-TI2V-5B"),
        "--dataset", str(args.data / "models/microduck"),
        "--checkpoint-receipt", str(args.data / "models/microduck/navigation_checkpoint.json"),
        "--history-offsets-ticks", "300", "150", "50", "0", "--inference-steps", "20",
        "--scene", args.scene, "--mode", args.mode, "--seeds", *map(str, args.seeds),
        "--tasks", *args.tasks, "--max-action-age-ms", str(args.max_action_age_ms),
        "--max-ticks", str(args.max_ticks), "--output", str(args.output),
        "--record-frames", "--record-predictions", "--record-model-inputs", "--record-stage-timings"]
    if args.variant in ("prompt", "context", "optimized"):
        command.append("--cache-prompt")
    if args.variant in ("context", "optimized"):
        command.append("--cache-action-context")
    if args.variant == "optimized":
        command.append("--action-cuda-graph")
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("/data"), help="Portable model/cache/result root")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("fetch", help="Download pinned simulator assets and the published inference bundle")
    p.add_argument("--component", choices=("sim", "model", "backbone", "all"), default="all")
    sub.add_parser("verify", help="SHA256 every required model file and check trained history/statistics")
    p = sub.add_parser("inputs", help="Generate real simulator startup inputs for optimization checks")
    p.add_argument("--output", type=Path, required=True)
    for name in ("benchmark", "eval"):
        p = sub.add_parser(name)
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--variant", choices=("plain", "prompt", "context", "optimized"), default="optimized")
        if name == "benchmark":
            p.add_argument("--inputs", type=Path, required=True)
            p.add_argument("--repeats", type=int, default=20)
            p.add_argument("--reference", type=Path, help="Previous benchmark result directory, same inputs and weights")
            p.add_argument("--profile", action="store_true", help="Extra profiler call after latency measurement")
        else:
            p.add_argument("--mode", choices=("offline", "realtime"), default="realtime")
            p.add_argument("--scene", choices=("room_clear", "room_curriculum_one", "room_randomized"), default="room_clear")
            p.add_argument("--seeds", nargs="+", type=int, default=[2000])
            p.add_argument("--tasks", nargs="+", choices=("find_red_ball", "find_blue_ball", "find_yellow_ball"), default=["find_red_ball"])
            p.add_argument("--max-ticks", type=int, default=1500)
            p.add_argument("--max-action-age-ms", type=int, default=500)
    args = parser.parse_args()
    args.data = args.data.expanduser().resolve()
    os.environ.setdefault("MICRODUCK_ASSET_ROOT", str(args.data / "upstream"))
    if args.command == "fetch":
        fetch(args)
    elif args.command == "verify":
        print(json.dumps(verify(args.data), indent=2))
    elif args.command == "inputs":
        inputs(args)
    elif args.command == "benchmark":
        benchmark(args)
    else:
        verify(args.data)
        subprocess.run(eval_command(args), check=True)


if __name__ == "__main__":
    main()
