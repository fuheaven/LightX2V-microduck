"""Versioned preset initial conditions, teacher gates and held-out evaluation.

No score thresholds or author command programs are modified here. All physical
rollouts use the same public task clock without truth-triggered command resets.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

from .data import CollectionConfig, collect_dataset, file_sha256, merge_datasets, write_json
from .env import MicroDuckEnv
from .evaluation import serializable
from .tasks import PROGRAMS
from .teachers import ClassicalTeacher

CONDITIONS = {"version": "preset_initial_v1", "xy_half_range_m": .02, "yaw_half_range_rad": .04, "joint_half_range_rad": .003, "light_half_range": .05, "warmup_ticks": 50, "clock": "program"}
BAM_KICK_CONDITIONS = {**CONDITIONS, "version": "preset_initial_bam_kick_v2", "kick_ball_forward_offset_m": .09}
CONDITION_RECIPES = {recipe["version"]: recipe for recipe in (CONDITIONS, BAM_KICK_CONDITIONS)}


def resolve_condition_recipe(recipe=None):
    """Resolve an immutable named protocol, preserving the original default.

    v2 only corrects BAM kick initialization to the official RL training ball
    offset. It changes neither the author commands nor any scoring threshold.
    Exact mappings are accepted so checkpoint/dataset provenance can be used
    without silently discarding a changed calibration field.
    """
    if recipe is None:
        recipe = CONDITIONS["version"]
    version = recipe if isinstance(recipe, str) else recipe.get("version")
    if version not in CONDITION_RECIPES:
        raise ValueError(f"unknown preset initial condition recipe: {version!r}")
    expected = CONDITION_RECIPES[version]
    if not isinstance(recipe, str) and recipe != expected:
        raise ValueError("preset initial condition recipe differs from its registered version")
    return copy.deepcopy(expected)


def dataset_condition_recipe(index):
    """Use dataset provenance as the training condition authority.

    Older preset datasets may omit the field and keep the legacy default.
    Per-episode and merged-shard provenance must never contradict the index.
    """
    recipe = resolve_condition_recipe(index.get("provenance", {}).get("initial_conditions"))
    for row in index.get("episodes", []):
        recorded = (row.get("initial_condition") or {}).get("recipe")
        shard = row.get("source_provenance", {}).get("initial_conditions")
        for value in (recorded, shard):
            if value is not None and resolve_condition_recipe(value) != recipe:
                raise ValueError("dataset contains mixed initial condition recipes")
    return recipe


def manifest_condition_recipe(manifest, requested=None):
    recipe = resolve_condition_recipe(manifest.get("initial_conditions"))
    if requested is not None and resolve_condition_recipe(requested) != recipe:
        raise ValueError("requested initial condition recipe differs from the checkpoint manifest")
    return recipe


def _validate_condition_profile(recipe, profile):
    if recipe["version"] == BAM_KICK_CONDITIONS["version"] and profile != "official_bam":
        raise ValueError("preset_initial_bam_kick_v2 requires official_bam; author compatibility retains its original initialization")


def _validated_teacher_gate(gate_path, task, condition_recipe):
    recipe = resolve_condition_recipe(condition_recipe)
    gate = json.loads(Path(gate_path).read_text())
    row = gate.get("tasks", {}).get(task, {})
    episodes, successes = row.get("episodes", 0), row.get("successes", 0)
    if (
        gate.get("condition_recipe") != recipe
        or gate.get("profile") != "official_bam"
        or gate.get("validation_count", 0) < 100
        or episodes < 100
        or row.get("distinct_conditions") != episodes
        or not row.get("eligible_for_collection")
        or successes / max(episodes, 1) < .95
        or row.get("strict_rate") != successes / max(episodes, 1)
    ):
        raise ValueError("task has not passed the matching official_bam 95% teacher gate and condition recipe")
    return recipe


class PresetTrainingEnv(MicroDuckEnv):
    def __init__(self, *, condition_recipe=None, **kwargs):
        self.condition_recipe = resolve_condition_recipe(condition_recipe)
        _validate_condition_profile(self.condition_recipe, kwargs.get("profile", "official_bam"))
        super().__init__(scene="flat", **kwargs)
        self.preset_clock = "program"
        self.initial_condition = None

    def reset(self, seed=0, task="walk_forward", options=None):
        super().reset(seed, task, options)
        rng = np.random.default_rng(seed)
        recipe = self.condition_recipe
        xy = rng.uniform(-recipe["xy_half_range_m"], recipe["xy_half_range_m"], 2)
        yaw = float(rng.uniform(-recipe["yaw_half_range_rad"], recipe["yaw_half_range_rad"]))
        joint = rng.uniform(-recipe["joint_half_range_rad"], recipe["joint_half_range_rad"], 14)
        light = float(rng.uniform(1 - recipe["light_half_range"], 1 + recipe["light_half_range"]))
        self.physics.reset(robot_xy=xy, robot_yaw=yaw, kick_side=PROGRAMS[task].ball_side, seed=seed)
        side = PROGRAMS[task].ball_side
        if side and "kick_ball_forward_offset_m" in recipe:
            # Match pinned microduck_rl/scripts/infer_policy.py BALL_OFFSET_X
            # and BallKick reset_ball_in_front_of_foot. Applied only at reset,
            # before warmup; task rollouts never reposition the ball.
            c, s = math.cos(yaw), math.sin(yaw)
            offset = np.array((recipe["kick_ball_forward_offset_m"], .042 if side == "left" else -.042))
            position = np.array(((c, -s), (s, c))) @ offset + xy
            self.physics.set_free_body("ball_free", (*position, .035))
        self.physics.data.qpos[self.physics.qpos_indices] += joint
        if self.physics.bam_controller is not None:
            self.physics.bam_controller.reset(self.physics.data.qpos)
            self.physics.bam_controller.last_ts = self.physics.data.time
            self.physics.bam_controller.q_target[:] = self.physics.last_targets
        self.physics.model.light_diffuse[:] *= light
        self.physics.mj.mj_forward(self.physics.model, self.physics.data)
        rendering = self.render_enabled
        self.render_enabled = False
        for _ in range(recipe["warmup_ticks"]):
            state = self.observe().state.copy()
            state[48:] = 0
            self.step(self.bank.infer("roller" if self.task.embodiment == "rollers" else "stand", state))
        self.begin_task_at_current_pose()
        self.initial_condition = {"recipe": copy.deepcopy(recipe), "task": task, "xy": xy.tolist(), "yaw": yaw, "joint_offset": joint.tolist(), "light_scale": light}
        self.render_enabled = rendering
        return self.observe()

    def evaluator_state(self):
        result = super().evaluator_state()
        if self.initial_condition is not None:
            result["initial_condition"] = copy.deepcopy(self.initial_condition)
            result["initial_condition_sha256"] = hashlib.sha256(json.dumps(self.initial_condition, sort_keys=True).encode()).hexdigest()
        return result

    def clone_state(self):
        result = super().clone_state()
        result["preset_condition_recipe"] = copy.deepcopy(self.condition_recipe)
        result["preset_initial_condition"] = copy.deepcopy(self.initial_condition)
        return result

    def restore_state(self, snapshot):
        if resolve_condition_recipe(snapshot.get("preset_condition_recipe")) != self.condition_recipe:
            raise ValueError("preset snapshot belongs to a different initial condition recipe")
        super().restore_state(snapshot)
        self.initial_condition = copy.deepcopy(snapshot["preset_initial_condition"])


def wilson(successes, total):
    if total == 0:
        return [0., 1.]
    z = 1.959963984540054
    p, d = successes / total, 1 + z * z / total
    center = (p + z * z / (2 * total)) / d
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / d
    return [max(0., center - half), min(1., center + half)]


def teacher_episode(args):
    task, seed, profile = args[:3]
    condition_recipe = args[3] if len(args) > 3 else None
    with PresetTrainingEnv(profile=profile, render=False, condition_recipe=condition_recipe) as env:
        env.reset(seed, task)
        teacher = ClassicalTeacher(env)
        while not env.terminated:
            env.step(teacher.predict(env.observe()))
        state = env.evaluator_state()
        return serializable({"task": task, "seed": seed, **{key: state[key] for key in ("success", "author_success", "strict_preset_result", "initial_condition", "initial_condition_sha256", "recovery_events", "collision", "fault", "model_signature")}})


def teacher_gate(output, tasks, *, workers=8, diagnostic=20, validation=100, profile="official_bam", condition_recipe=None):
    recipe = resolve_condition_recipe(condition_recipe)
    _validate_condition_profile(recipe, profile)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "teacher-gate.json").exists():
        raise FileExistsError("teacher report exists; choose a new immutable run directory")
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = [(task, seed, profile, recipe) for task in tasks for seed in (*range(1000, 1000 + diagnostic), *range(2000, 2000 + validation))]
        with (output / "episodes.jsonl").open("w") as stream:
            for row in pool.map(teacher_episode, jobs, chunksize=1):
                rows.append(row)
                stream.write(json.dumps(row) + "\n")
                stream.flush()
    summary = {}
    for task in tasks:
        selected = [row for row in rows if row["task"] == task and row["seed"] >= 2000]
        successes = sum(row["success"] for row in selected)
        distinct = len({row["initial_condition_sha256"] for row in selected})
        summary[task] = {"successes": successes, "episodes": len(selected), "distinct_conditions": distinct, "strict_rate": successes / len(selected), "author_rate": sum(row["author_success"] for row in selected) / len(selected), "wilson_95": wilson(successes, len(selected)), "eligible_for_collection": validation >= 100 and distinct == validation and successes / validation >= .95, "failed_checks": sorted({check for row in selected for check in row["strict_preset_result"].get("failed_checks", [])})}
    report = {"condition_recipe": recipe, "profile": profile, "tasks": summary, "diagnostic_count": diagnostic, "validation_count": validation, "hardware_unvalidated": True}
    write_json(output / "teacher-gate.json", report)
    return report


def collect(output, gate_path, task, episodes=200, seed=10000, *, condition_recipe=None):
    recipe = _validated_teacher_gate(gate_path, task, condition_recipe)
    return collect_dataset(lambda: PresetTrainingEnv(render=True, condition_recipe=recipe), ClassicalTeacher, output, config=CollectionConfig(tasks=(task,), episodes=episodes, seed=seed), teacher_env_factory=lambda: PresetTrainingEnv(render=False, condition_recipe=recipe), provenance={"teacher_gate_sha256": file_sha256(gate_path), "initial_conditions": recipe})


def dagger(output, gate_path, task, checkpoint, backbone, *, episodes=100, seed=1000000, condition_recipe=None):
    recipe = _validated_teacher_gate(gate_path, task, condition_recipe)
    from lightx2v.models.networks.tinyvla.runtime import MicroDuckTinyVLAPolicy, TinyVLARuntime
    if not 1 <= episodes <= 100:
        raise ValueError("at most 100 DAgger episodes per skill and round")
    runtime = TinyVLARuntime.from_directory(checkpoint, backbone_path=backbone, device="cuda", precision="bf16")
    student = MicroDuckTinyVLAPolicy(runtime)
    result = collect_dataset(lambda: PresetTrainingEnv(render=True, condition_recipe=recipe), ClassicalTeacher, output, config=CollectionConfig(tasks=(task,), episodes=episodes, seed=seed, mode="dagger"), student=student, teacher_env_factory=lambda: PresetTrainingEnv(render=False, condition_recipe=recipe), provenance={"teacher_gate_sha256": file_sha256(gate_path), "initial_conditions": recipe, "student_checkpoint_sha256": file_sha256(Path(checkpoint) / "model.safetensors")})
    # Keep fixed validation/test populations from the original BC dataset.
    # New DAgger validation/test rollouts remain on disk for diagnosis and are
    # excluded from the training aggregation, never reassigned to train.
    result["episodes"] = [row for row in result["episodes"] if row["split"] == "train"]
    write_json(Path(output) / "train-only-index.json", result)
    return result


def merge(output, indices):
    """Publish immutable shards as one dataset; reject cross-split overlap."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if output.name != "index.json":
        raise ValueError("merged dataset must be named index.json")
    recipes = [dataset_condition_recipe(json.loads(Path(path).read_text())) for path in indices]
    if not recipes or any(recipe != recipes[0] for recipe in recipes):
        raise ValueError("cannot merge preset datasets with different initial condition recipes")
    result = merge_datasets(indices, output.parent)
    result["provenance"]["initial_conditions"] = recipes[0]
    write_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("teachers", "collect", "merge", "dagger"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--tasks", nargs="+", default=list(PROGRAMS))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--diagnostic", type=int, default=20)
    parser.add_argument("--validation", type=int, default=100)
    parser.add_argument("--gate")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--indices", nargs="+")
    parser.add_argument("--checkpoint")
    parser.add_argument("--backbone")
    parser.add_argument("--condition-recipe", choices=tuple(CONDITION_RECIPES), default=CONDITIONS["version"])
    args = parser.parse_args()
    if args.command == "teachers":
        result = teacher_gate(args.output, args.tasks, workers=args.workers, diagnostic=args.diagnostic, validation=args.validation, condition_recipe=args.condition_recipe)
    elif args.command == "collect":
        if len(args.tasks) != 1 or not args.gate:
            parser.error("collect requires exactly one task and --gate")
        result = collect(args.output, args.gate, args.tasks[0], args.episodes, args.seed, condition_recipe=args.condition_recipe)
    elif args.command == "dagger":
        if len(args.tasks) != 1 or not args.gate or not args.checkpoint or not args.backbone:
            parser.error("dagger needs one task, gate, checkpoint and backbone")
        result = dagger(args.output, args.gate, args.tasks[0], args.checkpoint, args.backbone, episodes=args.episodes, seed=args.seed, condition_recipe=args.condition_recipe)
    else:
        result = merge(args.output, args.indices)
    print(json.dumps(result if args.command == "teachers" else {"episodes": len(result["episodes"]), "output": args.output}, indent=2))


if __name__ == "__main__":
    main()
