"""One UI-requested simulation. Process exit releases all GPU/model resources."""
import argparse
import os
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "embodied/microduck/src")]
os.environ.setdefault("LIGHTX2V_IMPORT_MODE", "robotics")
os.environ.setdefault("DTYPE", "BF16")
os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
if os.environ.get("MUJOCO_GL") == "egl":
    # CUDA containers can receive NVIDIA libraries but lack its GLVND vendor
    # JSON, silently selecting Mesa llvmpipe. Choose NVIDIA explicitly.
    os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES",
                          str(REPO / "deploy/microduck/fastwam/10_nvidia.json"))


def run(args):
    from lightx2v_microduck.visualization import read_json, atomic_json, validate_run
    from lightx2v_microduck.navigation_viewer import NavigationViewer
    from lightx2v_microduck.env import MicroDuckEnv
    options = validate_run(read_json(args.output / "request.json"))
    def stage(status, message):
        atomic_json(args.output / "stage.json", dict(status=status, message=message))
        if (args.output / "cancel.request").exists():
            raise InterruptedError("run cancelled before physics started")
    viewer = NavigationViewer(args.output)
    if options["policy"] == "fastwam":
        stage("verifying", "正在校验模型、归一化统计与依赖工件。")
        from fastwam_handoff import verify, config_for
        checked = verify(args.data)
        stage("loading", "正在加载 FastWAM；首次加载需要等待。")
        import lightx2v_platform.set_ai_device  # noqa: F401
        from lightx2v.models.runners.wan.fastwam_runner import FastWAMPolicy
        from lightx2v_microduck.wam_history_policy import FastWAMNavigationHistoryPolicy
        config = config_for(args.data, options["variant"])
        atomic_json(args.output / "resolved_config.json", config)
        core = FastWAMPolicy.from_config(config)
        policy = FastWAMNavigationHistoryPolicy(core,
            model_id="clear-room-realtime-control-pilot-v1", artifact_sha256=checked["models/microduck/model.pt"])
        stage("warming", "正在初始化场景、相机与推理缓存；任务时钟尚未开始。")
        from lightx2v_microduck.navigation_eval import run_navigation_episode
        with MicroDuckEnv(profile="official_bam", scene=options["scene"], asset_root=args.data / "upstream") as env:
            try:
                report = run_navigation_episode(env, policy, args.output / "episode", task=f"find_{options['color']}_ball",
                    seed=options["seed"], realtime=True, max_ticks=options["seconds"] * 50,
                    warmup_predictions=1, max_action_age_ms=500, record_frames=True,
                    record_predictions=True, live_viewer=viewer,
                    snapshot_camera_workers=4 if os.environ["MUJOCO_GL"] == "osmesa" else 0)
            finally:
                viewer.close()
        report.update(model_sha256=checked["models/microduck/model.pt"], variant=options["variant"])
    else:
        stage("warming", "正在初始化官方 ONNX 行走演示；此模式不使用 FastWAM。")
        from lightx2v_microduck.navigation import intent_from_action
        with MicroDuckEnv(profile="official_bam", scene="flat", asset_root=args.data / "upstream") as env:
            env.render_enabled = False
            env.reset(options["seed"], "walk_forward")
            for _ in range(50):
                env.step(intent_from_action([0, 0, 0, 0, 1], episode_id=env.episode_id, tick=env.tick))
            env.begin_task_at_current_pose()
            viewer.start(env)
            viewer.begin()
            start = time.monotonic()
            try:
                while env.tick < options["seconds"] * 50 and not env.terminated and not (args.output / "cancel.request").exists():
                    # No task teacher or truth-based navigation. Just a visible
                    # standing/walking command to the official 50 Hz controller.
                    moving = 1 < env.tick * .02 < min(options["seconds"] - 2, 8)
                    action = [.1 if moving else 0, 0, 0, 0, 0 if moving else 1]
                    env.step(intent_from_action(action, episode_id=env.episode_id, tick=env.tick))
                    viewer.publish(env, wall_s=time.monotonic()-start,
                                   execution={"source": "official_onnx", "skill": "walk" if moving else "stand"})
                    time.sleep(max(0, start + env.tick*.02 - time.monotonic()))
            finally:
                viewer.close()
            elapsed = time.monotonic()-start
            report = dict(policy="official_onnx", mode="realtime", sim_s=env.tick*.02,
                          wall_s=elapsed, real_time_factor=env.tick*.02/max(elapsed, 1e-9),
                          strict_model_success=None, realtime_success=None,
                          scope="official ONNX walking demo; not a visual navigation capability test")
    import hashlib
    runtime_paths = ["scripts/microduck/fastwam_visual_worker.py",
        "embodied/microduck/src/lightx2v_microduck/navigation_viewer.py",
        "embodied/microduck/src/lightx2v_microduck/navigation_eval.py",
        "embodied/microduck/src/lightx2v_microduck/navigation.py",
        "embodied/microduck/src/lightx2v_microduck/env.py",
        "deploy/microduck/fastwam/10_nvidia.json"]
    report.update(viewer=viewer.metrics(), formal_acceptance=False, hardware_validated=False,
                  runtime_source_sha256={name: hashlib.sha256((REPO/name).read_bytes()).hexdigest()
                                         for name in runtime_paths})
    atomic_json(args.output / "outcome.json", dict(status="complete", report=report))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args)
    except Exception as exc:
        from lightx2v_microduck.visualization import atomic_json
        atomic_json(args.output / "outcome.json", dict(status="failed", error=f"{type(exc).__name__}: {exc}"))
        raise


if __name__ == "__main__":
    main()
