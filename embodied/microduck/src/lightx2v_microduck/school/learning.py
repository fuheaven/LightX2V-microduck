"""Resumable CEM, matched rule comparison, and held-out physical evaluation."""

import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from ..evaluation import serializable
from ..live import atomic_json, image_uri
from .courses import REVISION, UPSTREAM, Course, split_seeds
from .policy import SchoolPolicy, initial_weights
from .simulation import SchoolWorld


class Interrupted(Exception):
    pass


def digest(value):
    return hashlib.sha256(json.dumps(serializable(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write(path, value):
    atomic_json(Path(path), serializable(value))


def save_artifact(path, content):
    value = serializable(content)
    value.pop("sha256", None)
    value["sha256"] = digest(value)
    write(path, value)


def load_artifact(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict) or not {"revision", "course", "weights"}.issubset(value):
        raise ValueError("invalid School artifact schema")
    checksum = value.pop("sha256", None)
    if checksum != digest(value) or value.get("revision") != REVISION:
        raise ValueError("invalid or incompatible School artifact")
    SchoolPolicy(Course.from_dict(value["course"]), value["weights"])
    value["sha256"] = checksum
    return value


class CEM:
    def __init__(self, mean, seed=0, checkpoint=None):
        self.mean = np.asarray(mean, dtype=float)
        self.std = np.full(self.mean.shape, 0.25)
        self.rng = np.random.default_rng(seed)
        self.generation = 0
        if checkpoint:
            self.mean = np.asarray(checkpoint["mean"], dtype=float)
            self.std = np.asarray(checkpoint["std"], dtype=float)
            self.rng.bit_generator.state = checkpoint["rng"]
            self.generation = int(checkpoint["generation"])
        if self.mean.shape != (15,) or self.std.shape != (15,) or not np.isfinite(self.mean).all() or not np.isfinite(self.std).all() or np.any(self.std <= 0):
            raise ValueError("invalid CEM checkpoint")

    def sample(self, count, incumbent):
        return np.vstack([incumbent, np.clip(self.rng.normal(self.mean, self.std, (count - 1, 15)), -5, 5)])

    def update(self, population, scores, elites=2):
        selected = np.asarray(population)[np.argsort(scores)[-elites:]]
        self.mean = selected.mean(axis=0)
        self.std = np.maximum(selected.std(axis=0), 0.08)
        self.generation += 1

    def state(self):
        return serializable({"mean": self.mean, "std": self.std, "generation": self.generation, "rng": self.rng.bit_generator.state})


def run_episode(world, weights, seed, *, guard=lambda: None, record=None, live=None):
    guard()
    if live is not None:
        Path(live).parent.mkdir(parents=True, exist_ok=True)
    policy = SchoolPolicy(world.course, weights)
    observation = world.reset(seed)
    rgb, observer, ticks, states = [], [], [], []
    started = time.monotonic()

    def publish(status):
        if live is not None:
            write(
                live,
                {
                    "status": status,
                    "tick": world.tick,
                    "sim_time_s": world.tick * 0.02,
                    "wall_time_s": time.monotonic() - started,
                    "real_time_factor": world.tick * 0.02 / max(time.monotonic() - started, 1e-9),
                    "frames": {"camera": image_uri(observation.rgb), "observer": image_uri(world.physics.render_observer())},
                    "diagnostics": policy.last_diagnostics,
                    "result": world.result() if world.tick else None,
                },
            )

    try:
        publish("running")
        while not world.done:
            guard()
            if record is not None and world.tick % 5 == 0:
                states.append(observation.state.copy())
                ticks.append(world.tick)
                if observation.rgb is not None:
                    rgb.append(observation.rgb.copy())
                if live is not None:
                    observer.append(world.physics.render_observer())
            if live is None or world.tick % 5 == 0:
                action = policy.predict(observation, truth=world.policy_truth() if world.course.privileged else None)
            if live is not None:
                action = replace(action, source_tick=world.tick, duration_ticks=1)
            observation = world.step(action)
            if live is not None:
                deadline = started + world.tick * 0.02
                while time.monotonic() < deadline:
                    guard()
                    time.sleep(max(0.0, min(0.02, deadline - time.monotonic())))
                if world.tick % 5 == 0 or world.done:
                    publish("running")
        result = world.result()
        publish("complete")
    except Interrupted:
        result = {**world.result(), "success": False, "termination_reason": "interrupted", "partial": True}
        publish("cancelled")
        raise
    finally:
        if record is not None:
            record = Path(record)
            record.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                record / "trajectory.npz",
                state=states,
                observation_tick=ticks,
                rgb=np.asarray(rgb, dtype=np.uint8),
                observer_rgb=np.asarray(observer, dtype=np.uint8),
                raw_action=world.actions,
                command=world.commands,
                truth_trajectory=world.rows,
                action_source_tick=np.arange(len(world.actions)),
            )
            saved = dict(locals().get("result", {"success": False, "termination_reason": "error"}))
            saved.update(
                revision=REVISION,
                course=world.course.to_dict(),
                identity_sha256=digest(world.identity),
                weights_sha256=digest(weights),
                policy_sensors=["proprioception", "simulator_truth"] if world.course.privileged else ["rgb", "proprioception"],
                trajectory_columns=["sim_time_s", "x", "y", "yaw", "height", "ball_surface_distance", "waypoint_distance", "ball_displacement", "speed"],
                hardware_validated=False,
            )
            write(record / "result.json", saved)
    result.update(wall_time_s=time.monotonic() - started, policy_sensors=["proprioception", "simulator_truth"] if world.course.privileged else ["rgb", "proprioception"])
    return result


def summary(rows):
    return {
        "episodes": len(rows),
        "success_rate": sum(row["success"] for row in rows) / len(rows) if rows else None,
        "mean_reward": float(np.mean([row["reward"] for row in rows])) if rows else None,
        "failures": {reason: sum(r["termination_reason"] == reason for r in rows) for reason in sorted({r["termination_reason"] for r in rows if not r["success"]})},
    }


def train(course, output, *, seed=0, budget_s=600, generations=20, population=8, train_count=3, test_count=10, resume=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if not 1 <= generations <= 100 or not 4 <= population <= 32 or not 1 <= train_count <= 12 or not 1 <= test_count <= 100 or not 10 <= budget_s <= 1800:
        raise ValueError("invalid teaching budget/population/split")
    started = time.monotonic()
    deadline = started + budget_s
    training_deadline = started + budget_s * 0.65

    def guard(until=deadline):
        if (output / "cancel.request").exists() or time.monotonic() >= until:
            raise Interrupted()

    parent = load_artifact(resume) if resume else None
    if parent and (parent["course"] != course.to_dict() or parent["seed"] != seed):
        raise ValueError("resume requires the same course and split seed")
    world = SchoolWorld(course)
    try:
        if parent and digest(parent["identity"]) != digest(world.identity):
            raise ValueError("parent physics or official policies differ; cannot resume")
        weights = np.asarray(parent["weights"] if parent else initial_weights(course))
        cem = CEM(weights, seed, parent.get("cem") if parent else None)
        history = list(parent.get("history", [])) if parent else []
        incumbent_score = parent.get("validation_reward") if parent else None
        selected_source = parent.get("selected_source", "rule_initialization") if parent else "rule_initialization"
        artifact = {
            "revision": REVISION,
            "course": course.to_dict(),
            "seed": seed,
            "identity": world.identity,
            "upstream_revision": UPSTREAM,
            "parent_sha256": parent["sha256"] if parent else None,
            "hardware_validated": False,
            "controller": "linear_3x5_with_registered_features_and_stop_logic",
            "training_inputs": "simulator_truth" if course.privileged else "rgb_proprioception",
            "perception": "fixed_color_detector_and_known_ball_radius" if not course.privileged else None,
        }

        def checkpoint(status):
            artifact.update(
                weights=weights.tolist(),
                cem=cem.state(),
                history=history,
                validation_reward=incumbent_score,
                selected_source=selected_source,
                qualification="not_evaluated",
                status=status,
                elapsed_s=time.monotonic() - started,
            )
            save_artifact(output / "skill.json", artifact)

        def progress(phase, **extra):
            write(output / "progress.json", {"phase": phase, "generation": cem.generation, "history": history, "elapsed_s": time.monotonic() - started, "budget_s": budget_s, **extra})

        def evaluate_set(candidate, seeds, prefix, *, training=False):
            rows = []
            for index, current in enumerate(seeds):
                check = (lambda: guard(training_deadline)) if training else guard
                check()
                row = run_episode(world, candidate, current, guard=check)
                rows.append(row)
                # Candidate evidence is persisted as each actual episode completes.
                path = output / "trials"
                path.mkdir(exist_ok=True)
                write(path / f"{prefix}-{index}.json", row)
                progress(prefix, completed=index + 1, total=len(seeds))
            return rows

        checkpoint("initializing")
        stopped_training = False
        try:
            if incumbent_score is None:
                incumbent_score = summary(evaluate_set(weights, split_seeds(seed, "validation", 3), "initial-validation", training=True))["mean_reward"]
            for _ in range(generations):
                guard(training_deadline)
                # Commit only complete generations; partial generations are replayed on resume.
                before = cem.state()
                candidates = cem.sample(population, weights)
                pool = split_seeds(seed, "train", 24)
                start = (cem.generation * train_count) % len(pool)
                seeds = [pool[(start + k) % len(pool)] for k in range(train_count)]
                scores = []
                try:
                    for k, candidate in enumerate(candidates):
                        rows = evaluate_set(candidate, seeds, f"generation-{cem.generation}-candidate-{k}", training=True)
                        scores.append(summary(rows)["mean_reward"])
                    winner = candidates[int(np.argmax(scores))]
                    validation = summary(evaluate_set(winner, split_seeds(seed, "validation", 3), f"validation-{cem.generation}", training=True))
                except Interrupted:
                    cem = CEM(weights, checkpoint=before)
                    raise
                if validation["mean_reward"] > incumbent_score:
                    weights, incumbent_score = winner.copy(), validation["mean_reward"]
                    selected_source = "cem"
                cem.update(candidates, scores)
                history.append(
                    {
                        "generation": cem.generation,
                        "train_mean": float(np.mean(scores)),
                        "train_best": float(max(scores)),
                        "validation_reward": incumbent_score,
                        "candidate_validation": validation,
                        "train_seeds": seeds,
                    }
                )
                checkpoint("training")
                progress("training")
        except Interrupted:
            stopped_training = True
        checkpoint("trained" if history else "no_complete_generation")
        # Test episodes never rank candidates. A time budget can yield a partial test.
        comparison = {"learned": [], "rule": []}
        try:
            for current in split_seeds(seed, "test", test_count):
                for name, candidate in (("learned", weights), ("rule", initial_weights(course))):
                    guard()
                    progress("testing", completed=sum(map(len, comparison.values())), total=test_count * 2)
                    result = run_episode(world, candidate, current, guard=guard, record=output / "evaluation" / f"{name}-{current}")
                    comparison[name].append(result)
        except Interrupted:
            pass
        report = {
            "revision": REVISION,
            "split": "test",
            "seed": seed,
            "course": course.to_dict(),
            "results": comparison,
            "summaries": {k: summary(v) for k, v in comparison.items()},
            "expected_episodes_per_policy": test_count,
            "complete": all(len(v) == test_count for v in comparison.values()),
            "training_cut_short": stopped_training,
            "independent_test": True,
            "hardware_validated": False,
        }
        report["qualified"] = bool(history and report["complete"] and test_count >= 10 and report["summaries"]["learned"]["success_rate"] >= 0.8)
        write(output / "evaluation.json", report)
        artifact.update(
            qualification="no_complete_generation" if not history else "passed_small_test" if report["qualified"] else "not_passed" if report["complete"] else "incomplete_test",
            evaluation_sha256=digest(report),
            status="cancelled" if (output / "cancel.request").exists() else "complete",
            elapsed_s=time.monotonic() - started,
        )
        save_artifact(output / "skill.json", artifact)
        progress(artifact["status"], qualification=artifact["qualification"], report=report["summaries"])
        return report
    finally:
        world.close()


def execute_skill(artifact_path, output, *, seed=0, live=False, count=10, budget_s=600):
    artifact = load_artifact(artifact_path)
    course = Course.from_dict(artifact["course"])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + budget_s

    def guard():
        if (output / "cancel.request").exists() or time.monotonic() > deadline:
            raise Interrupted()

    world = SchoolWorld(course, render=live)
    try:
        if digest(world.identity) != digest(artifact["identity"]):
            raise ValueError("skill physics or official policy hashes are incompatible")
        rows = []
        seeds = split_seeds(seed, "demo", 1) if live else split_seeds(seed, "test", count)
        try:
            for current in seeds:
                write(output / "progress.json", {"phase": "live" if live else "testing", "completed": len(rows), "total": len(seeds)})
                rows.append(run_episode(world, artifact["weights"], current, guard=guard, record=output / str(current), live=output / "live.json" if live else None))
        except Interrupted:
            pass
        report = {
            "skill_sha256": artifact["sha256"],
            "course": artifact["course"],
            "results": rows,
            "summary": summary(rows),
            "complete": len(rows) == len(seeds),
            "mode": "live" if live else "evaluation",
            "split": "demo" if live else "test",
            "hardware_validated": False,
        }
        write(output / "evaluation.json", report)
        return report
    finally:
        world.close()
