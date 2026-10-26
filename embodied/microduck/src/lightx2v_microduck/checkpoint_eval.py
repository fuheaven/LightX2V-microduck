"""Closed-loop checkpoint selection; loss never grants a skill release."""
import argparse
import gc
import json
import time
import subprocess
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np

from .data import file_sha256, write_json
from .checkpoint_selection import SELECTION_PROTOCOL, checkpoint_selection_score
from .evaluation import serializable, validate_policy_environment
from .preset_campaign import CONDITIONS, PresetTrainingEnv, wilson


def _save_video(path, frames):
    if not frames:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    process = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", "10", "-i", "pipe:0", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)], stdin=subprocess.PIPE)
    try:
        for frame in frames:
            process.stdin.write(np.ascontiguousarray(frame).tobytes())
    finally:
        process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError("evaluation video encoding failed")


def evaluate_checkpoint(checkpoint, backbone, *, tasks, seeds, device="cuda", precision="bf16", ablation=None, record_dir=None, mode="offline"):
    from lightx2v.models.networks.tinyvla.runtime import MicroDuckTinyVLAPolicy, TinyVLARuntime
    runtime = TinyVLARuntime.from_directory(checkpoint, backbone_path=backbone, device=device, precision=precision)
    policy = MicroDuckTinyVLAPolicy(runtime)
    rows, latencies = [], []
    if mode not in ("offline", "live") or (mode == "live" and ablation):
        raise ValueError("invalid evaluation mode/ablation combination")
    if mode == "live" and record_dir is None:
        raise ValueError("live evaluation needs an output directory")
    record_dir = Path(record_dir) if record_dir else None
    failures_recorded = set()
    seed_list = list(seeds)
    try:
        initial_conditions = runtime.manifest.get("initial_conditions") or CONDITIONS
        with PresetTrainingEnv(render=True, condition_recipe=initial_conditions) as env:
            for task in tasks:
                for seed in seed_list:
                    obs = env.reset(seed, task)
                    validate_policy_environment(policy.manifest, env, obs)
                    policy.reset(obs.episode_id)
                    frames, cameras, actions, states = [], [], [], []
                    realtime = None
                    if mode == "live":
                        from .evaluation import EvaluationConfig, run_episode
                        from .live import LiveController
                        destination = record_dir / f"{task}-{seed}"
                        controller = LiveController(env, destination, async_inference=True, publish_every=5)
                        try:
                            episode = run_episode(env, policy, destination / "episode", EvaluationConfig((task,), record_every=5), task=task, seed=seed, live=controller)
                        finally:
                            controller.close()
                        realtime = episode["live_execution"]
                        realtime["strict_success"] = bool(episode["success"])
                        realtime["real_time_factor"] = env.tick * .02 / max(realtime["control_wall_time_s"], 1e-9)
                        latencies.extend(value / 1000 for value in realtime["model_inference_ms"])
                    while mode == "offline" and not env.terminated:
                        obs = env.observe()
                        capture = record_dir is not None and task not in failures_recorded
                        if capture:
                            frames.append(env.render_observer().copy())
                            states.append(obs.state.copy())
                        if ablation == "black_image":
                            obs = replace(obs, rgb=np.zeros_like(obs.rgb))
                        elif ablation == "language":
                            obs = replace(obs, instruction="Stand still.")
                        elif ablation == "command":
                            state = obs.state.copy()
                            state[48:] = 0
                            obs = replace(obs, state=state)
                        if capture:
                            cameras.append(obs.rgb.copy())
                        start = time.monotonic()
                        action = policy.predict(obs)
                        if capture:
                            actions.append(action.actions[:5].copy())
                        latencies.append(time.monotonic() - start)
                        env.render_enabled = False
                        for raw in action.actions[:5]:
                            if env.terminated:
                                break
                            env.step(raw)
                        env.render_enabled = True
                    state = env.evaluator_state()
                    row = serializable({"task": task, "seed": seed, **{key: state[key] for key in ("success", "author_success", "strict_preset_result", "initial_condition_sha256", "recovery_events", "collision", "fault")}})
                    if realtime is not None:
                        row["success"] = realtime["strict_success"]
                        row["realtime"] = realtime
                    rows.append(row)
                    if record_dir is not None:
                        if frames and (seed == seed_list[0] or not row["success"]):
                            destination = record_dir / f"{task}-{seed}"
                            _save_video(destination / "observer.mp4", frames)
                            _save_video(destination / "camera.mp4", cameras)
                            np.savez_compressed(destination / "trajectory.npz", state61=states, action_prefix=actions)
                            write_json(destination / "result.json", row)
                            if not row["success"]:
                                failures_recorded.add(task)
                        write_json(record_dir / "progress.json", {"mode": mode, "episodes": rows, "complete": False})
        summary = {}
        for task in tasks:
            selected = [row for row in rows if row["task"] == task]
            count = sum(row["success"] for row in selected)
            distinct = len({row["initial_condition_sha256"] for row in selected})
            accepted = ablation is None and len(selected) >= 100 and distinct == len(selected) and count / len(selected) >= .9
            takeovers = sum(row.get("realtime", {}).get("execution", {}).get("fallback_ticks", 0) for row in selected)
            summary[task] = {"successes": count, "episodes": len(selected), "distinct_conditions": distinct, "strict_rate": count / len(selected), "author_rate": sum(row["author_success"] for row in selected) / len(selected), "wilson_95": wilson(count, len(selected)), "offline_accepted": accepted and mode == "offline", "realtime_accepted": accepted and mode == "live" and takeovers == 0, "takeovers": takeovers, "failed_checks": dict(Counter(check for row in selected for check in row["strict_preset_result"].get("failed_checks", [])))}
            if mode == "live":
                factor = min(row["realtime"]["real_time_factor"] for row in selected)
                p95 = float(np.percentile(latencies, 95) * 1000) if latencies else None
                summary[task]["minimum_real_time_factor"] = factor
                summary[task]["realtime_accepted"] &= factor >= .9 and p95 is not None and p95 <= 100
        report = {"checkpoint": str(checkpoint), "checkpoint_sha256": file_sha256(Path(checkpoint) / "model.safetensors"), "condition_recipe": initial_conditions, "semantic_refresh_steps": runtime.refresh_steps, "action_padding": runtime.action_padding, "author_metric_source": "local_author_style_proxy_unverified_against_author_scorer", "tasks": summary, "episodes": rows, "mean_strict_rate": float(np.mean([v["strict_rate"] for v in summary.values()])), "inference_p95_ms": float(np.percentile(latencies, 95) * 1000) if latencies else None, "mode": mode, "ablation": ablation, "hardware_unvalidated": True}
        if mode == "live":
            report["physics_waits_for_model"] = False
        if record_dir is not None:
            write_json(record_dir / "progress.json", {"mode": mode, "episodes": rows, "complete": True})
        return report
    finally:
        del policy, runtime
        gc.collect()
        if device.startswith("cuda"):
            import torch
            torch.cuda.empty_cache()


def select_best(directory, checkpoint, report):
    target = Path(directory) / "best-validation.json"
    previous = json.loads(target.read_text()) if target.exists() else None
    score = checkpoint_selection_score(report)
    if previous is not None and set(report["tasks"]) != set(previous["tasks"]):
        raise ValueError("checkpoint selection cannot compare different task suites")
    if previous is None or score > checkpoint_selection_score(previous):
        write_json(target, {"checkpoint": str(Path(checkpoint).resolve()), "mean_strict_rate": report["mean_strict_rate"], "tasks": report["tasks"], "checkpoint_sha256": report["checkpoint_sha256"], "selection_protocol": SELECTION_PROTOCOL, "selection_score": list(score), "auxiliary_scores_grant_acceptance": False})
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed-start", type=int, default=900000)
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--ablation", choices=("black_image", "language", "command"))
    parser.add_argument("--mode", choices=("offline", "live"), default="offline")
    args = parser.parse_args()
    report = evaluate_checkpoint(args.checkpoint, args.backbone, tasks=args.tasks, seeds=range(args.seed_start, args.seed_start + args.seeds), ablation=args.ablation, mode=args.mode, record_dir=Path(args.output).with_suffix(""))
    write_json(args.output, report)
    print(json.dumps(report["tasks"], indent=2))


if __name__ == "__main__":
    main()
