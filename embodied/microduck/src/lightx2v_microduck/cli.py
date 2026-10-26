"""Standalone entry points. ROS is never imported by these commands."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _environment_options(parser):
    parser.add_argument("--asset-root", default=os.environ.get("MICRODUCK_ASSET_ROOT"))
    parser.add_argument("--profile", choices=("official_bam", "tinyvla_author_compat"), default="official_bam")
    parser.add_argument("--scene", choices=("room", "room_clear", "room_randomized", "flat", "water_court", "ball_apex", "community_follow", "slalom_cones", "ball_tracking", "duck_detection", "climb_desk"), default=None)
    parser.add_argument("--randomization", choices=("none", "deployment_noise_v1"), default="none")
    from .execution_profiles import INITIAL_PROFILES
    parser.add_argument("--initial-conditions", choices=INITIAL_PROFILES, default="nominal")


def _policy_options(parser):
    parser.add_argument("--policy", default="lightx2v_tinyvla")
    parser.add_argument("--model-path")
    parser.add_argument("--backbone-path")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--compile-expert", action="store_true")
    parser.add_argument("--endpoint", help="native TinyVLA, author reference or VLM service URL")
    parser.add_argument("--planner-model", help="VLM model served by the skill planner endpoint")
    parser.add_argument("--planner-context", choices=("current_frame", "feedback"), default="feedback",
                        help="VLM current-frame ablation or bounded action feedback and task memory")
    from .planner_prompts import PROMPT_PROFILES
    parser.add_argument("--planner-prompt-profile", choices=PROMPT_PROFILES, default="legacy_v1",
                        help="Versioned VLM task and locomotion tool description")
    parser.add_argument("--planner-service-identity", help="Expected model service identity SHA256; reject changed or missing evidence")


def parser():
    root = argparse.ArgumentParser(description="Pinned MicroDuck simulation, VLA data and evaluation")
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("school", help="registered courses, CEM teaching, saved skills and live viewing")
    assets = commands.add_parser("assets")
    assets.add_argument("action", choices=("fetch", "verify", "list"))
    assets.add_argument("sources", nargs="*")
    assets.add_argument("--asset-root")
    assets.add_argument("--all", action="store_true")
    sim = commands.add_parser("sim")
    _environment_options(sim)
    sim.add_argument("--task", default="walk_forward")
    sim.add_argument("--seed", type=int, default=0)
    sim.add_argument("--steps", type=int, default=250)
    sim.add_argument("--output", default="runs/sim")
    sim.add_argument("--policy", choices=("official", "privileged", "pixel", "pixel_rgb"), default="official")
    sim.add_argument("--no-render", action="store_true")
    sim.add_argument("--no-observer", action="store_true", help="omit third-person replay frames")
    collect = commands.add_parser("collect")
    _environment_options(collect)
    collect.add_argument("--suite", default="skills13")
    collect.add_argument("--task", action="append")
    collect.add_argument("--teacher", choices=("official", "privileged"), required=True)
    collect.add_argument("--episodes", type=int, default=200, help="episodes per task; visual startup quota uses 334 per color")
    collect.add_argument("--steps", type=int)
    collect.add_argument("--seed", type=int, default=0)
    collect.add_argument("--horizon", type=int, default=50)
    collect.add_argument("--execute-steps", type=int, default=5)
    collect.add_argument("--output", required=True)
    collect.add_argument("--mode", choices=("teacher", "dagger"), default="teacher")
    _policy_options(collect)
    train = commands.add_parser("train")
    train.add_argument("--recipe", required=True, help="YAML path or a bundled recipe name")
    train.add_argument("--dataset-index")
    train.add_argument("--output")
    train.add_argument("--model-path")
    train.add_argument("--backbone-path")
    train.add_argument("--steps", type=int)
    evaluate = commands.add_parser("eval")
    _environment_options(evaluate)
    _policy_options(evaluate)
    evaluate.add_argument("--suite", default="skills13")
    evaluate.add_argument("--task", action="append")
    evaluate.add_argument("--seeds", type=int, default=100, help="number of seeds per task")
    evaluate.add_argument("--seed-start", type=int, default=10000)
    evaluate.add_argument("--steps", type=int)
    evaluate.add_argument("--execution-prefix", type=int, default=5)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--no-render", action="store_true", help="for proprioceptive/oracle baselines; image policies will refuse missing RGB")
    evaluate.add_argument("--no-observer", action="store_true", help="omit third-person replay frames")
    compare = commands.add_parser("compare")
    compare.add_argument("--experiment", required=True)
    compare.add_argument("--asset-root")
    replay = commands.add_parser("replay")
    _environment_options(replay)
    replay.add_argument("--dataset-index", required=True)
    replay.add_argument("--episode", type=int, default=0)
    realtime = commands.add_parser("realtime")
    realtime.add_argument("--backend", choices=("robotd_bam",), default="robotd_bam")
    _environment_options(realtime)
    _policy_options(realtime)
    realtime.add_argument("--task", default="find_red_ball")
    realtime.add_argument("--seed", type=int, default=0)
    realtime.add_argument("--seconds", type=float, default=30)
    realtime.add_argument("--latency-ms", type=float, default=0)
    realtime.add_argument("--drop-every", type=int, default=0)
    realtime.add_argument("--reorder-every", type=int, default=0)
    realtime.add_argument("--cancel-after-seconds", type=float, help="revoke robotd independently of an in-flight inference")
    realtime.add_argument("--cancel-path", help="revoke when this cancellation marker file appears")
    realtime.add_argument("--output", required=True)
    live = commands.add_parser("live", help="wall-clock simulation with browser frames and saved evaluation")
    _environment_options(live)
    _policy_options(live)
    live.add_argument("--task", default="walk_forward")
    live.add_argument("--seed", type=int, default=2)
    live.add_argument("--output", required=True)
    daemon = commands.add_parser("robotd")
    daemon.add_argument("action", choices=("prepare", "build"))
    daemon.add_argument("--asset-root")
    daemon.add_argument("--output")
    web = commands.add_parser("web")
    web.add_argument("--runs", default="runs")
    web.add_argument("--host", default="127.0.0.1")
    web.add_argument("--port", type=int, default=8767)
    return root


def execute(args):
    from .assets import RESOURCE_DIR, fetch_source, load_lock, verify_assets
    from .evaluation import EvaluationConfig, evaluate, make_policy, suite_tasks

    if args.command == "assets":
        from .community import load_catalog, verify_community_policy
        lock = load_lock()
        sources = [*lock["sources"], *('community:' + name for name in load_catalog())] if args.all else args.sources or lock["default_sources"]
        if args.action == "list":
            return {**lock, 'community_policies': load_catalog()}
        if args.action == "fetch":
            from .community import fetch_community_policy

            return {"sources": [fetch_community_policy(name.split(":", 1)[1], args.asset_root) if name.startswith("community:") else fetch_source(name, args.asset_root) for name in sources]}
        result = verify_assets(args.asset_root, [name for name in sources if not name.startswith('community:')])
        result['community_policies'] = [verify_community_policy(name.split(':', 1)[1], args.asset_root) for name in sources if name.startswith('community:')]
        result['ok'] = result['ok'] and all(record['ok'] for record in result['community_policies'])
        if not result["ok"]:
            raise ValueError(json.dumps(result))
        return result
    if args.command == "robotd":
        from .daemon import build_robotd, prepare_robotd

        return {"path": str((build_robotd if args.action == "build" else prepare_robotd)(args.asset_root, args.output))}
    if args.command == "web":
        from .web import serve

        return serve(args.runs, args.host, args.port)
    if args.command == "compare":
        from .evaluation import compare

        return compare(args.experiment, asset_root=args.asset_root)
    if args.command == "train":
        import yaml

        from .training import TrainingConfig, train

        path = Path(args.recipe)
        if not path.is_file():
            path = RESOURCE_DIR / "recipes" / (args.recipe + ".json")
        text = os.path.expandvars(path.read_text())
        if "${" in text:
            raise ValueError("recipe has unset environment variables; configure MICRODUCK_ASSET_ROOT, MICRODUCK_RUN_ROOT and MICRODUCK_BACKBONE_ROOT")
        settings = yaml.safe_load(text)
        settings = settings.get("training", settings)
        for name, value in (("dataset_index", args.dataset_index), ("output_dir", args.output), ("model_path", args.model_path), ("backbone_path", args.backbone_path), ("max_steps", args.steps)):
            if value is not None:
                settings[name] = value
        from dataclasses import asdict

        return asdict(train(TrainingConfig(**settings)))
    from .env import MicroDuckEnv

    scene = args.scene or ("flat" if args.initial_conditions != "nominal" or getattr(args, "suite", "").startswith("skills") else "room")
    required_sources = ["microduck-policies", "tinyvla-author-rl" if args.profile == "tinyvla_author_compat" else "microduck_rl"]
    if scene == "water_court":
        required_sources.append("microduck-ai-world")
    checked = verify_assets(args.asset_root, required_sources)
    if not checked["ok"]:
        raise ValueError(f"upstream assets failed verification: {checked}")

    def new_environment(render=True):
        from .execution_profiles import is_project_profile, make_execution_environment
        task = getattr(args, "task", None) or "walk_forward"
        if isinstance(task, list):
            if is_project_profile(args.initial_conditions) and len(task) != 1:
                raise ValueError("use qualified_eval for multi-task project evaluation")
            task = task[0]
        env = make_execution_environment(task, recipe=args.initial_conditions, profile=args.profile,
                                         asset_root=args.asset_root, scene=scene, render=render)
        if args.randomization == "deployment_noise_v1":
            from .randomization import RandomizedEnv

            env = RandomizedEnv(env)
        return env

    def env_factory():
        return new_environment(render=not getattr(args, "no_render", False))

    options = {name: getattr(args, name) for name in ("model_path", "backbone_path", "device", "precision", "compile_expert", "endpoint", "planner_model", "planner_context", "planner_prompt_profile", "planner_service_identity") if hasattr(args, name)}
    if args.command == "live":
        from .evaluation import run_episode
        from .live import LiveController

        output = Path(args.output)
        with env_factory() as env:
            env.reset(args.seed, args.task)
            policy = make_policy(args.policy, env, **options)
            controller = LiveController(env, output, async_inference=args.policy == "lightx2v_tinyvla")
            try:
                return run_episode(env, policy, output / args.task / f"seed-{args.seed}", EvaluationConfig((args.task,), record_every=2), task=args.task, seed=args.seed, live=controller)
            finally:
                controller.close()
    if args.command in ("sim", "eval"):
        tasks = (args.task,) if args.command == "sim" else tuple(args.task or suite_tasks(args.suite))
        seeds = (args.seed,) if args.command == "sim" else tuple(range(args.seed_start, args.seed_start + args.seeds))
        config = EvaluationConfig(tasks, seeds, max_steps=args.steps, execution_prefix=getattr(args, "execution_prefix", 5), record_observer=not args.no_observer)
        return evaluate(env_factory, lambda env: make_policy(args.policy, env, **options), args.output, config)
    if args.command == "collect":
        from .data import CollectionConfig, collect_dataset
        from .tasks import get_task
        from .teachers import ClassicalTeacher

        tasks = tuple(args.task or suite_tasks(args.suite))
        goals = tuple(t for t in tasks if get_task(t).suite == "visual_goal")
        if goals and args.teacher != "privileged":
            raise ValueError("visual teacher needs explicit --teacher privileged")
        student = None
        if args.mode == "dagger":
            probe = env_factory()
            try:
                probe.reset(args.seed, tasks[0])
                student = make_policy(args.policy, probe, **options)
            finally:
                probe.close()
        config = CollectionConfig(
            tasks=tasks,
            episodes=args.episodes,
            max_steps=args.steps or (1500 if goals else 250),
            action_horizon=args.horizon,
            execute_steps=args.execute_steps,
            seed=args.seed,
            mode=args.mode,
            goal_tasks=goals,
        )
        groups = {}
        for task in tasks:
            groups.setdefault(get_task(task).embodiment, []).append(task)
        if len(groups) > 1 and student is not None:
            raise ValueError("DAgger uses one embodiment checkpoint per invocation; select skills_legs10 or skills_rollers3")
        from dataclasses import replace

        results = {}
        for embodiment, group in groups.items():
            output = Path(args.output) / embodiment if len(groups) > 1 else Path(args.output)
            grouped = replace(config, tasks=tuple(group), goal_tasks=tuple(t for t in group if t in goals))
            results[embodiment] = collect_dataset(
                env_factory,
                ClassicalTeacher,
                output,
                config=grouped,
                student=student,
                teacher_env_factory=lambda: new_environment(render=False),
                provenance={"teacher": args.teacher, "profile": args.profile, "scene": scene, "randomization": args.randomization, "source_lock": load_lock()["lightx2v_revision"]},
            )
        return results
    if args.command == "replay":
        from .data import replay_episode

        env = env_factory()
        try:
            return replay_episode(env, args.dataset_index, args.episode)
        finally:
            env.close()
    if args.command == "realtime":
        if args.profile != "official_bam" or args.randomization != "none":
            raise ValueError("robotd_bam currently requires official_bam with randomization=none; offline deployment_noise_v1 is a separate experiment")
        from .daemon import RobotdBamBackend
        from .evaluation import validate_action_source, validate_policy_environment
        from .realtime import run_realtime

        # Loading/compiling the model must not consume the live task window.
        probe = new_environment(render=True)
        try:
            initial = probe.reset(args.seed, args.task)
            policy = make_policy(args.policy, probe, **options)
            validate_policy_environment(policy.manifest, probe)
            if policy.manifest.uses_privileged_state:
                raise ValueError("real-time external VLA evaluation excludes privileged teacher policies")
            if policy.manifest.action_kind != "joint_residual":
                raise ValueError("real-time external policy requires raw joint residuals")
            policy.reset(initial.episode_id)
            warmup = policy.predict(initial)
            validate_action_source(warmup, initial)
            policy.reset(initial.episode_id)
        finally:
            probe.close()
        backend = RobotdBamBackend(args.asset_root, scene=scene, initial_conditions=args.initial_conditions)
        try:
            backend.reset(args.seed, args.task)
            return run_realtime(backend, policy, args.output, duration_s=args.seconds, latency_ms=args.latency_ms, drop_every=args.drop_every, reorder_every=args.reorder_every, seed=args.seed,
                                cancel_after_s=args.cancel_after_seconds, cancel_path=args.cancel_path)
        finally:
            backend.close()
    raise ValueError(f"unhandled command {args.command}")


def main(argv=None):
    import sys
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "school":
        from .school.__main__ import main as school_main
        return school_main(arguments[1:])
    args = parser().parse_args(argv)
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("LIGHTX2V_IMPORT_MODE", "robotics")
    try:
        result = execute(args)
        if result is not None:
            from .evaluation import serializable

            print(json.dumps(serializable(result), indent=2, ensure_ascii=False, allow_nan=False))
    except (ValueError, FileNotFoundError, RuntimeError, ImportError) as exc:
        raise SystemExit(f"microduck: {exc}") from exc


if __name__ == "__main__":
    main()
