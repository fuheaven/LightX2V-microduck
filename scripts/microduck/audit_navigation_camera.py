"""Replay recorded decisions for diagnostic camera visibility, never training.

The legacy model signature includes absolute source paths. We substitute only
those recorded paths when recomputing it, then require an exact signature and
robot-contract match. Physics parameters and recorded poses are not adjusted.
RGB comparisons are reported because render drivers can differ across hosts.
Replaying physical steps also preserves MuJoCo's camera-transform update phase;
setting qpos then mj_forward would render a slightly different moving pose.
"""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_model(env, expected):
    description = replace(env.description,
        source_xml=expected["scene"]["source_xml"],
        robot_source_xml=expected["scene"]["robot_source_xml"],
        original_demo_root=expected["scene"]["original_demo_root"])
    result = hashlib.sha256(f"{env.profile}/{description}/{env.action_delay_ticks}".encode())
    for key in ("body_mass", "body_inertia", "body_pos", "geom_pos", "geom_size", "jnt_range", "cam_quat"):
        result.update(np.asarray(getattr(env.physics.model, key)).tobytes())
    if result.hexdigest() != expected["model_signature"]:
        raise ValueError("recorded model differs after substituting only absolute source paths")
    if env._robot_signature != expected["robot_signature"]:
        raise ValueError("recorded robot contract differs")
    return {"local_signature": env._signature, "recorded_signature": expected["model_signature"],
            "source_path_substituted_signature": result.hexdigest(),
            "robot_signature": env._robot_signature, "numeric_model_signature_matches": True}


def sample_camera(env):
    physics = env.physics
    rgb = physics.render()
    renderer = physics.renderer
    renderer.enable_segmentation_rendering()
    try:
        renderer.update_scene(physics.data, camera=physics.camera)
        segmentation = renderer.render().copy()
    finally:
        renderer.disable_segmentation_rendering()
    counts = {}
    for color in ("red", "blue", "yellow"):
        geom = physics.mj.mj_name2id(physics.model, physics.mj.mjtObj.mjOBJ_GEOM, f"lx_ball_{color}_geom")
        if geom < 0:
            raise ValueError(f"missing diagnostic ball geometry: {color}")
        counts[color] = int(np.count_nonzero((segmentation[:, :, 0] == geom) &
                           (segmentation[:, :, 1] == int(physics.mj.mjtObj.mjOBJ_GEOM))))
    matrix = physics.data.cam_xmat[physics.camera].reshape(3, 3)
    return rgb, counts, {"position": physics.data.cam_xpos[physics.camera].tolist(),
                        "forward": (-matrix[:, 2]).tolist(),
                        "raw_camera_fovy_degrees": float(physics.model.cam_fovy[physics.camera])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("diagnostic reports are immutable")
    protocol = json.loads(args.protocol.read_text())
    if len(set(args.seeds)) != len(args.seeds) or set(args.seeds) - set(protocol["development_seeds"]):
        raise ValueError("select unique frozen development seeds only")
    from lightx2v_microduck.env import MicroDuckEnv
    from lightx2v_microduck.navigation import intent_from_action

    report = {"status": "running", "script_sha256": digest(__file__),
        "protocol_sha256": digest(args.protocol), "diagnostic_truth_only": True,
        "eligible_training_data": False, "formal_acceptance": False, "hardware_validated": False,
        "visibility_rule": "actual head-camera MuJoCo segmentation, at least 16 ball pixels",
        "pose_reconstruction": "replay accepted first actions through original BAM/ONNX controller; check all qpos and decision proprioception",
        "cases": []}
    env = MicroDuckEnv(profile="official_bam", scene="room_clear", render=False)
    try:
        for seed in args.seeds:
            for task in protocol["tasks"]:
                directory = args.experiment / f"{task}-seed{seed}"
                result = json.loads((directory / "report.json").read_text())
                if result["mode"] != "offline" or result["seed"] != seed or result["task"] != task:
                    raise ValueError("camera audit requires matching completed offline records")
                predictions = json.loads((directory / "predictions.json").read_text())
                prediction_at = {p["source_tick"]: p for p in predictions if p["accepted"]}
                if len(prediction_at) != len(predictions):
                    raise ValueError("duplicate or rejected predictions need a separate execution audit")
                env.reset(seed, task)
                model_check = verify_model(env, result["evaluator"])
                if abs(result["evaluator"]["evaluation_origin_s"] - 1.) > 1e-8:
                    raise ValueError("this diagnostic expects the original 50-tick physical warmup")
                for _ in range(50):
                    env.step(intent_from_action([0, 0, 0, 0, 1], episode_id=env.episode_id, tick=env.tick))
                env.begin_task_at_current_pose()
                color = result["evaluator"]["target_color"]
                frames = []
                qpos_error = proprio_error = 0.
                with np.load(directory / "recording.npz", allow_pickle=False) as packed:
                    # NPZ indexing decompresses an entire array on every
                    # access. Cache each array once before the frame/tick
                    # loops; repeated RGB decompression dominated full-cohort
                    # diagnostics without contributing any new evidence.
                    data = {name: packed[name] for name in ("physics_ticks", "qpos", "camera_ticks", "rgb")}
                    indices = {int(t): i for i, t in enumerate(data["physics_ticks"])}
                    if (len(indices) != len(data["qpos"]) or
                            data["qpos"].shape[1] != env.physics.model.nq or not np.isfinite(data["qpos"]).all()):
                        raise ValueError("invalid physics recording")
                    action = None
                    for frame_index, tick in enumerate(data["camera_ticks"]):
                        tick = int(tick)
                        if tick not in indices or tick not in prediction_at:
                            raise ValueError("camera frame lacks an exact physics pose and accepted prediction")
                        while env.tick < tick:
                            qpos_error = max(qpos_error, float(np.abs(env.physics.data.qpos - data["qpos"][indices[env.tick]]).max()))
                            env.step(intent_from_action(action, episode_id=env.episode_id, tick=env.tick))
                        qpos_error = max(qpos_error, float(np.abs(env.physics.data.qpos - data["qpos"][indices[tick]]).max()))
                        proprio_error = max(proprio_error, float(np.abs(env.observe().state[:48] -
                            np.asarray(prediction_at[tick]["observed_proprio48"])).max()))
                        if max(qpos_error, proprio_error) > 1e-5:
                            raise ValueError("recorded actions failed to reproduce physics and sensor state")
                        rgb, counts, camera = sample_camera(env)
                        difference = np.abs(rgb.astype(np.int16) - data["rgb"][frame_index].astype(np.int16))
                        action = prediction_at[tick]["submitted_actions"][0]
                        frames.append({"tick": tick, "ball_visible_pixels": counts,
                            "requested_target_visible": counts[color] >= 16,
                            "stand": bool(action[4] >= .5), "first_action": action,
                            "render_mean_abs_difference": float(difference.mean()),
                            "render_max_abs_difference": int(difference.max()), "camera": camera})
                    while env.tick < len(data["qpos"]):
                        qpos_error = max(qpos_error, float(np.abs(env.physics.data.qpos - data["qpos"][indices[env.tick]]).max()))
                        env.step(intent_from_action(action, episode_id=env.episode_id, tick=env.tick))
                    final = env.evaluator_state()
                    final_position_error = float(np.abs(np.asarray(final["robot_position"]) -
                        np.asarray(result["evaluator"]["robot_position"])).max())
                    if (max(qpos_error, final_position_error) > 1e-5
                            or final["strict_success"] != result["evaluator"]["strict_success"]):
                        raise ValueError("replayed physical outcome differs from the recorded episode")
                seen = [f["tick"] for f in frames if f["requested_target_visible"]]
                stands = [f for f in frames if f["stand"]]
                report["cases"].append({"seed": seed, "task": task, "model_check": model_check,
                    "strict_success": result["strict_model_success"], "frame_count": len(frames),
                    "requested_target_visible_frames": len(seen),
                    "first_target_visible_tick": min(seen) if seen else None,
                    "last_target_visible_tick": max(seen) if seen else None,
                    "stand_frames": len(stands),
                    "stand_without_target_frames": sum(not f["requested_target_visible"] for f in stands),
                    "replay_qpos_max_abs": qpos_error, "replay_proprio_max_abs": proprio_error,
                    "replay_final_position_max_abs": final_position_error,
                    "maximum_frame_mean_abs_render_difference": max(f["render_mean_abs_difference"] for f in frames),
                    "files": {name: digest(directory / name) for name in ("report.json", "predictions.json", "recording.npz")},
                    "frames": frames})
                print(json.dumps({k: v for k, v in report["cases"][-1].items()
                                  if k not in ("frames", "files", "model_check")}), flush=True)
    finally:
        env.close()
    report["status"] = "complete"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
