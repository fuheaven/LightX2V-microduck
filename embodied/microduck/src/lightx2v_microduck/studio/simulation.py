"""Actual MuJoCo episodes. Continuous mode never waits for model inference."""

from __future__ import annotations
import concurrent.futures
import math
import time
from pathlib import Path
import numpy as np
from PIL import Image
from .dataset import sensor_state, historical_indices
from .schema import CATEGORIES, COLORS, digest, write_json
from .scene import sample_scene, layout_identity
from .plugins import plugin, PixelRule


def cases(skill, split, count, seed=0, groups=None):
    task = skill["task"]
    groups = groups or task["groups"]
    combinations = [(g, c, a) for g in groups for c in task["categories"] for a in task["colors"]]
    offset = {"train": 0, "validation": 1000000, "test": 2000000, "teacher": 3000000}[split]
    rows = []
    for i in range(count):
        group, category, color = combinations[i % len(combinations)]
        s = offset + seed + i
        scene = sample_scene(skill["scene"], s, group=group, category=category, color=color, split=split)
        rows.append(
            {
                "seed": s,
                "split": split,
                "group": group,
                "category": category,
                "color": color,
                "scene": scene,
                "layout_id": layout_identity(scene),
                "instance_id": f"{split}:{category}:{digest(next(o for o in scene['objects'] if o['id'] == scene['target_id']))}",
                "instruction_family": f"{split}:{i % 3}",
            }
        )
    return rows


def atomic_image(path, array):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    Image.fromarray(array).save(temp, format="JPEG", quality=85)
    temp.replace(path)


def instruction(case):
    templates = {
        "train": ["Find the {color} {category} and approach it, then stop.", "Search for the {color} {category}; get close and stay still.", "Go near the {color} {category} and hold your position."],
        "validation": [
            "Locate the {color} {category}, approach safely and stop nearby.",
            "Move toward the {color} {category}; stop at a safe distance.",
            "Look around for the {color} {category}, then settle beside it.",
        ],
        "test": [
            "Please seek out the {color} {category} and come to rest close to it.",
            "Your destination is the {color} {category}. Approach and remain stationary.",
            "Reach the vicinity of the {color} {category} without touching it, then wait.",
        ],
        "teacher": ["Find the {color} {category} and stop nearby."],
    }
    family = int(case["instruction_family"].split(":")[-1])
    choices = templates[case["split"]]
    return choices[family % len(choices)].format(**case)


def run_episode(skill, case, out, *, policy_kind="teacher", runtime=None, realtime=False, ttl_ms=500, stop=lambda: False, progress=lambda x: None, preview=False, control=None):
    from ..navigation import intent_from_action, NavigationBuffer, NavigationChunk

    task_plugin = plugin(skill["task"]["plugin"])
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    env = task_plugin.environment(case["scene"], skill["task"], render=False)
    frames = []
    states = []
    ticks = []
    actions = []
    tof_ticks = []
    truth = []
    times = []
    executed = []
    history_frames = []
    history_states = []
    history_ticks = []
    pending = None
    latency = []
    policy_errors = []
    interventions = 0
    policy_stop_ticks = 0
    used_control = False
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    buffer = NavigationBuffer(max_age_ticks=max(1, ttl_ms // 20), max_wall_age_s=ttl_ms / 1000)
    writer = None
    report = None

    def submit_prediction(values, source, current):
        from ..navigation_action_contracts import action_contract

        values = np.asarray(values, np.float32)
        if values.shape != (32, 5) or not np.isfinite(values).all():
            raise ValueError("invalid model action chunk")
        lo, hi = action_contract().bounds()
        return buffer.submit(NavigationChunk(np.clip(values, lo, hi), source.episode_id, source.tick, source.monotonic_ns, source.tick + 160), current)

    def inference_failed(exc):
        buffer.reset(env.episode_id)
        policy_errors.append({"tick": env.tick, "type": type(exc).__name__, "message": str(exc)[:250]})

    try:
        from dataclasses import replace

        if runtime and hasattr(runtime, "reset"):
            runtime.reset()

        env.studio_task = replace(env.studio_task, instruction=instruction(case))
        spawn = case["scene"]["spawn"]
        obs = env.reset(seed=case["seed"], task=env.studio_task, options={"robot_xy": (spawn["x"], spawn["y"]), "robot_yaw": math.radians(spawn["yaw"])})
        # Settling is outside the scored episode and never used to recover failures.
        for _ in range(100):
            if stop():
                raise InterruptedError("cancelled")
            if env.terminated:
                break
            obs = env.step(intent_from_action([0, 0, 0, 0, 1], episode_id=env.episode_id, tick=env.tick))
        env.begin_task_at_current_pose()
        buffer.reset(env.episode_id)
        teacher = task_plugin.teacher(env) if policy_kind == "teacher" else PixelRule(case["color"]) if policy_kind == "rule" else None
        held = np.array([0, 0, 0, 0, 1], np.float32)
        # Precondition the first chunk while the episode is still stationary.
        # Preserve its actual capture timestamp: an already stale result is rejected.
        startup_s = 0.0
        if runtime and realtime and not preview:
            env.render_enabled = True
            initial = env.observe()
            initial_state = sensor_state(initial)
            t = time.monotonic()
            try:
                values = runtime.predict(np.stack([initial.rgb] * 4), np.stack([initial_state] * 4), CATEGORIES.index(case["category"]), COLORS.index(case["color"]), initial.instruction)
                submit_prediction(values, initial, env.observe())
            except Exception as exc:
                inference_failed(exc)
            startup_s = time.monotonic() - t
        clock = time.monotonic()
        while not env.terminated:
            if stop():
                raise InterruptedError("cancelled")
            if env.tick % 5 == 0:
                env.render_enabled = True
                obs = env.observe()
                state = sensor_state(obs)
                view = env.render_observer()
                atomic_image(out / "observer.jpg", view)
                atomic_image(out / "head.jpg", obs.rgb)
                if writer is None:
                    import imageio.v2 as imageio

                    writer = imageio.get_writer(str(out / "recording.mp4"), fps=10, codec="libx264", macro_block_size=1)
                display = Image.fromarray(view).resize((round(view.shape[1] * 256 / view.shape[0] / 2) * 2, 256))
                writer.append_data(np.concatenate((np.asarray(display), obs.rgb), axis=1))
                if preview:
                    report = {"preview": True, "scene": case["scene"], "tick": env.tick, "hardware_validated": False}
                    break
                history_frames.append(obs.rgb)
                history_states.append(state)
                history_ticks.append(obs.tick)
                if len(history_ticks) > 65:
                    history_frames.pop(0)
                    history_states.pop(0)
                    history_ticks.pop(0)
                idx = historical_indices(history_ticks, obs.tick)
                if teacher:
                    held = teacher.predict(obs)
                elif runtime:
                    if pending is not None and pending.done():
                        try:
                            values, source, predict_s = pending.result()
                            latency.append(predict_s * 1000)
                            submit_prediction(values, source, obs)
                        except Exception as exc:
                            inference_failed(exc)
                        pending = None
                    if pending is None:
                        f = np.stack([history_frames[i] for i in idx])
                        s = np.stack([history_states[i] for i in idx])

                        def predict(f=f, s=s, source=obs):
                            start = time.monotonic()
                            values = runtime.predict(f, s, CATEGORIES.index(case["category"]), COLORS.index(case["color"]), source.instruction)
                            return values, source, time.monotonic() - start

                        if realtime:
                            pending = pool.submit(predict)
                        else:
                            try:
                                values, source, elapsed = predict()
                                latency.append(elapsed * 1000)
                                # Offline model time is excluded explicitly, never advertised as realtime.
                                submit_prediction(values, source, source)
                            except Exception as exc:
                                inference_failed(exc)
                frames.append(obs.rgb)
                states.append(state)
                ticks.append(obs.tick)
                tof_ticks.append(obs.tof_metadata["capture_tick"])
                actions.append(held.copy())
                times.append(obs.monotonic_ns)
                world = env.evaluator_state()
                truth.append(
                    {
                        "tick": env.tick,
                        "robot_position": world["robot_position"].tolist(),
                        "target": world["balls"][env.task.target_color].tolist(),
                        "teacher": getattr(teacher, "last_diagnostics", {}),
                    }
                )
            env.render_enabled = False
            current = env.observe()
            selected_control = control() if control else None
            human_active = bool(selected_control and selected_control.get("expires", 0) > time.time())
            if human_active:
                held = np.asarray(selected_control["action"], np.float32)
                used_control = True
                interventions += 1
                buffer.reset(env.episode_id)
            if runtime and not human_active:
                if realtime:
                    intent, source = buffer.consume(current)
                else:
                    from dataclasses import replace

                    # Evaluate at simulation time, but keep source-tick alignment and validity.
                    aligned = replace(current, monotonic_ns=buffer.chunk.source_monotonic_ns + (current.tick - buffer.chunk.source_tick) * 20000000) if buffer.chunk else current
                    intent, source = buffer.consume(aligned)
                interventions += int(source["execution_source"] != "navigation_model")
                if env.tick % 5 == 0:
                    actions[-1] = np.array([*intent.command[:3], intent.command[5], float(intent.skill == "stand")], np.float32)
            else:
                intent = intent_from_action(held, episode_id=env.episode_id, tick=env.tick, duration_ticks=1)
                source = {"execution_source": "human" if human_active else policy_kind}
                if env.tick % 5 == 0:
                    actions[-1] = held.copy()
            policy_stop_ticks = policy_stop_ticks + 1 if source["execution_source"] in ("navigation_model", "rule") and intent.skill == "stand" else 0
            from dataclasses import replace

            obs = env.step(replace(intent, duration_ticks=1))
            executed.append({"tick": env.tick, "raw14": env.last_action.tolist(), "target14": env.applied_action.tolist(), "source": source})
            if env.tick % 25 == 0:
                progress(
                    {
                        "phase": "live" if realtime else "simulation",
                        "tick": env.tick,
                        "sim_s": env.tick * 0.02,
                        "wall_s": time.monotonic() - clock,
                        "rtf": env.tick * 0.02 / max(time.monotonic() - clock, 0.001),
                        "control_source": source["execution_source"],
                        "interventions": interventions,
                        "policy_reported_complete": policy_stop_ticks * 0.02 >= skill["task"]["hold_s"],
                    }
                )
            if realtime:
                delay = clock + env.tick * 0.02 - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
        if report is None:
            report = {
                **task_plugin.score(env, interventions),
                **{k: v for k, v in case.items() if k != "scene"},
                "policy": policy_kind,
                "mode": "realtime" if realtime else "offline",
                "p95_ms": float(np.percentile(latency, 95)) if latency else 0,
                "inference_ms": latency,
                "policy_errors": policy_errors,
                "startup_conditioning_s": startup_s,
                "wall_s": time.monotonic() - clock,
                "rtf": env.tick * 0.02 / max(time.monotonic() - clock, 0.001),
                "instruction": env.task.instruction,
                "observation_contract": "rgb+tof+proprio176",
                "uses_truth": policy_kind == "teacher",
                "human_correction": used_control,
                "policy_reported_complete": policy_stop_ticks * 0.02 >= skill["task"]["hold_s"],
                "robot_signature": env._robot_signature,
                "scene_sha256": digest(case["scene"]),
            }
            if frames:
                np.savez_compressed(
                    out / "sensors.npz", rgb=np.stack(frames), state=np.stack(states), action=np.stack(actions), tick=np.array(ticks), tof_tick=np.array(tof_ticks), monotonic_ns=np.array(times)
                )
                write_json(out / "teacher_truth.json", truth)
                write_json(out / "execution.json", executed)
        write_json(out / "episode.json", report)
        return report
    finally:
        buffer.reset("")
        if writer is not None:
            writer.close()
        pool.shutdown(wait=True, cancel_futures=True)
        env.close()
