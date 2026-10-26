"""Episode datasets and branched teacher labels for MicroDuck imitation learning.

The teacher sees its own simulator branch. DAgger defaults to student execution;
optional teacher roll-ins are recorded separately and cannot establish student
success. RGB and state always come from the actor before the executed action.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

import numpy as np

from .contracts import POLICY_INDICES, JointResidualChunk, Observation, RobotContract


class CollectionEnvironment(Protocol):
    def reset(self, seed: int = 0, task: str = "walk_forward") -> Observation: ...
    def observe(self) -> Observation: ...
    def step(self, action: np.ndarray) -> Observation: ...
    def clone_state(self) -> Any: ...
    def restore_state(self, snapshot: Any) -> None: ...
    def evaluator_state(self) -> Mapping[str, Any]: ...
    def close(self) -> None: ...


@dataclass(frozen=True)
class CollectionConfig:
    tasks: tuple[str, ...] = ("walk_forward",)
    episodes: int = 10
    max_steps: int = 250
    action_horizon: int = 50
    execute_steps: int = 5
    seed: int = 0
    mode: str = "teacher"
    teacher_rollin_probability: float = 0.0
    teacher_prefix_ticks: int = 0
    verify_teacher_completion: bool = True
    goal_tasks: tuple[str, ...] = ()
    split_fractions: tuple[float, float, float] = (0.8, 0.1, 0.1)
    instruction_templates: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {
            "train": ("{instruction}", "Please {instruction_lower}"),
            "val": ("Your task is to {instruction_lower}", "Now, {instruction_lower}"),
            "test": ("I would like you to {instruction_lower}", "Follow this request: {instruction}"),
        }
    )

    def __post_init__(self):
        if not self.tasks or len(set(self.tasks)) != len(self.tasks):
            raise ValueError("tasks must be nonempty and unique")
        if self.episodes < 1 or self.max_steps < 1:
            raise ValueError("episodes and max_steps must be positive")
        if not 1 <= self.execute_steps <= self.action_horizon <= 256:
            raise ValueError("require 1 <= execute_steps <= action_horizon <= 256")
        if self.mode not in ("teacher", "dagger"):
            raise ValueError("mode must be teacher or dagger")
        probability = self.teacher_rollin_probability
        if (isinstance(probability, bool) or not isinstance(probability, (int, float))
                or not np.isfinite(probability) or not 0 <= probability < 1
                or probability > 0 and self.mode != "dagger"):
            raise ValueError("teacher roll-in probability must be finite in [0, 1) and requires DAgger")
        prefix = self.teacher_prefix_ticks
        if isinstance(prefix, bool) or not isinstance(prefix, int) or prefix < 0:
            raise ValueError("teacher prefix ticks must be a nonnegative integer")
        if prefix and (self.mode != "dagger" or probability != 0
                       or prefix % self.execute_steps or prefix > self.max_steps - self.execute_steps):
            raise ValueError("teacher prefix requires DAgger, zero random roll-in, whole execution chunks and a remaining student chunk")
        fractions = np.asarray(self.split_fractions, dtype=float)
        if fractions.shape != (3,) or not np.isfinite(fractions).all() or np.any(fractions <= 0) or not np.isclose(fractions.sum(), 1):
            raise ValueError("three positive split fractions must sum to one")
        if not set(self.goal_tasks).issubset(self.tasks):
            raise ValueError("goal_tasks must be present in tasks")
        if set(self.instruction_templates) != {"train", "val", "test"} or not all(self.instruction_templates.values()):
            raise ValueError("instruction_templates must define nonempty train/val/test templates")
        owners = {}
        for split, templates in self.instruction_templates.items():
            for template in templates:
                if not isinstance(template, str) or ("{instruction}" not in template and "{instruction_lower}" not in template):
                    raise ValueError("instruction templates must preserve the task instruction")
                if owners.setdefault(template, split) != split:
                    raise ValueError("instruction templates must not overlap across splits")


def validate_split_domains(index: Mapping[str, Any]) -> dict[str, Any]:
    """Reject claimed held-out layouts leaking between train/validation/test."""
    owners, templates = {}, {}
    checked = 0
    for episode in index["episodes"]:
        split = episode["split"]
        identity = episode.get("initial_condition_sha256")
        if identity and owners.setdefault(("initial_condition", identity), split) != split:
            raise ValueError("initial condition occurs in different splits")
        template = episode.get("instruction_template")
        if template and templates.setdefault(template, split) != split:
            raise ValueError("instruction template occurs in different splits")
        if episode.get("layout_generalization_eligible"):
            for field_name in ("layout_id", "geometry_hash"):
                identity = episode.get(field_name)
                if not identity:
                    raise ValueError("held-out layout episodes require layout_id and geometry_hash")
                key = (field_name, identity)
                if owners.setdefault(key, split) != split:
                    raise ValueError(f"{field_name} occurs in different splits")
            checked += 1
    texture_splits = {split: sorted({episode["texture_id"] for episode in index["episodes"] if episode["split"] == split and episode.get("texture_id")}) for split in ("train", "val", "test")}
    texture_sets = [set(texture_splits[split]) for split in ("train", "val", "test")]
    texture_disjoint = all(texture_sets) and not any(texture_sets[i] & texture_sets[j] for i in range(3) for j in range(i + 1, 3))
    return {
        "checked_randomized_layout_episodes": checked,
        "layout_generalization_available": checked == len(index["episodes"]) and checked > 0,
        "instruction_templates_disjoint": bool(templates),
        "texture_ids_by_split": texture_splits,
        "texture_generalization_available": bool(texture_disjoint),
    }


class DataDeadlineExceeded(TimeoutError):
    """The caller's cooperative data-preparation allowance is exhausted."""


def check_deadline(deadline: float | None, phase: str = "data preparation") -> None:
    """Cooperatively stop work without publishing incomplete training evidence.

    The deadline uses this process's monotonic clock. Individual native calls
    or filesystem operations are not preempted; callers check their boundaries.
    """
    if deadline is None:
        return
    if isinstance(deadline, bool) or not isinstance(deadline, (int, float)) or not np.isfinite(deadline):
        raise ValueError("data deadline must be a finite monotonic timestamp")
    if time.monotonic() >= deadline:
        raise DataDeadlineExceeded(f"data preparation deadline reached during {phase}")


def file_sha256(path: str | Path, *, deadline: float | None = None) -> str:
    check_deadline(deadline, "file hashing")
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            check_deadline(deadline, "file hashing")
            h.update(block)
    check_deadline(deadline, "file hashing")
    return h.hexdigest()


def write_json(path: str | Path, value: Any) -> None:
    """Publish an index only after every referenced episode is durable."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=destination.name + ".", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _actor_observation(observation: Observation, neutral_command: bool) -> Observation:
    if not neutral_command:
        return observation
    state = observation.state.copy()
    state[48:61] = 0
    return replace(observation, state=state)


def _raw_chunk(action: Any) -> np.ndarray:
    raw = getattr(action, "actions", action)
    result = np.asarray(raw, dtype=np.float32)
    if result.ndim == 1:
        result = result[None]
    if result.ndim != 2 or result.shape[1] != 14 or not len(result) or not np.isfinite(result).all():
        raise ValueError("collector requires finite raw joint actions shaped [H,14]")
    return result.copy()


def _prediction_rows(action: Any, observation: Observation) -> np.ndarray:
    if isinstance(action, JointResidualChunk):
        if action.action_profile != "raw_residual":
            raise ValueError("collector needs denormalized raw residuals")
        if action.episode_id != observation.episode_id or action.source_tick != observation.tick or abs(action.source_time_s - observation.sim_time_s) > 1e-8:
            raise ValueError("collector action source differs from the current observation")
    return _raw_chunk(action)


def _done(state: Mapping[str, Any]) -> bool:
    return bool(state.get("terminated", False) or state.get("truncated", False))


@contextmanager
def _without_intermediate_rendering(environment):
    """Native collection needs images at decisions, not discarded substeps."""
    enabled = getattr(environment, "render_enabled", None)
    if enabled is not None:
        environment.render_enabled = False
    try:
        yield
    finally:
        if enabled is not None:
            environment.render_enabled = enabled


def _applied_targets(environment: CollectionEnvironment) -> np.ndarray:
    status = environment.evaluator_state()
    value = status.get("joint_targets", status.get("applied_action"))
    if value is None:
        raise ValueError("environment must report joint_targets after action postprocessing")
    targets = np.asarray(value, dtype=np.float32)
    if targets.shape == (15,):
        targets = targets[list(POLICY_INDICES)]
    if targets.shape != (14,) or not np.isfinite(targets).all():
        raise ValueError("applied joint targets must be finite 14D or mouth-inclusive 15D")
    return targets.copy()


def _splits(count: int, fractions: tuple[float, float, float], seed: int) -> list[str]:
    counts = np.floor(np.asarray(fractions) * count).astype(int)
    counts[0] += count - int(counts.sum())
    # A small smoke corpus still has episode-disjoint validation/test data.
    if count >= 3:
        for index in (1, 2):
            if counts[index] == 0:
                donor = int(np.argmax(counts))
                counts[donor] -= 1
                counts[index] += 1
    labels = np.repeat(["train", "val", "test"], counts)
    np.random.default_rng(seed).shuffle(labels)
    return labels.tolist()


def collect_dataset(
    env_factory: Callable[[], CollectionEnvironment],
    teacher_factory: Callable[[CollectionEnvironment], Any],
    output_dir: str | Path,
    *,
    config: CollectionConfig | None = None,
    student: Any | None = None,
    provenance: Mapping[str, Any] | None = None,
    teacher_env_factory: Callable[[], CollectionEnvironment] | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    deadline: float | None = None,
    publish_index: bool = True,
) -> dict[str, Any]:
    """Collect teacher demonstrations or branched DAgger labels.

    ``episodes`` is per task. A teacher must provide ``predict(obs)`` returning
    raw residuals, not skill intents. It can read privileged state only through
    the separate environment passed to its factory. Labels are recomputed from
    a full clone at every decision, never borrowed from an earlier teacher run.
    The index records actual post-filter joint targets separately from raw
    residuals; they are not interchangeable training targets.
    Optional mixed-policy or fixed-prefix roll-ins log each execution decision;
    assisted task completion is not independent student success.
    Qualified callers defer index publication until their provenance is stamped.
    """
    check_deadline(deadline, "collection initialization")
    config = config or CollectionConfig()
    if config.mode == "dagger" and student is None:
        raise ValueError("DAgger requires a student policy")
    destination = Path(output_dir)
    if (destination / "index.json").exists():
        raise FileExistsError("refusing to replace an existing dataset; use a new iteration directory")
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.rglob("episode_*.npz")):
        raise FileExistsError("episode files already exist; use a new dataset directory")
    (destination / "student_data").mkdir(exist_ok=True)
    (destination / "teacher_data").mkdir(exist_ok=True)
    main_env = teacher_env = None
    records: list[dict[str, Any]] = []
    embodiment = None
    task = episode_seed = None
    try:
        main_env = env_factory()
        check_deadline(deadline, "actor environment initialization")
        teacher_env = (teacher_env_factory or env_factory)()
        if main_env is teacher_env:
            raise ValueError("teacher and student must have distinct environment instances")
        check_deadline(deadline, "teacher environment initialization")
        teacher = teacher_factory(teacher_env)
        for task_index, task in enumerate(config.tasks):
            splits = _splits(config.episodes, config.split_fractions, config.seed + task_index)
            for episode_index in range(config.episodes):
                check_deadline(deadline, "episode initialization")
                episode_seed = config.seed + task_index * config.episodes + episode_index
                rollin_rng = np.random.default_rng(np.random.SeedSequence([episode_seed, 0xDA66E2]))
                observation = main_env.reset(seed=episode_seed, task=task)
                check_deadline(deadline, "actor reset")
                teacher_env.reset(seed=episode_seed, task=task)
                check_deadline(deadline, "teacher reset")
                split = splits[episode_index]
                templates = config.instruction_templates[split]
                template = templates[episode_index % len(templates)]
                instruction = template.format(instruction=observation.instruction, instruction_lower=observation.instruction[:1].lower() + observation.instruction[1:])
                for environment in (main_env, teacher_env):
                    if is_dataclass(getattr(environment, "task", None)):
                        environment.task = replace(environment.task, instruction=instruction)
                episode_embodiment = getattr(getattr(main_env, "task", None), "embodiment", "legs")
                if embodiment is not None and episode_embodiment != embodiment:
                    raise ValueError("collect legs and rollers into separate dataset directories")
                embodiment = episode_embodiment
                if student is not None and hasattr(student, "reset"):
                    student.reset(observation.episode_id)
                neutral = task in config.goal_tasks or getattr(getattr(main_env, "task", None), "suite", None) == "visual_goal"
                samples: dict[str, list] = {
                    key: []
                    for key in (
                        "rgb",
                        "state",
                        "tick",
                        "camera_tick",
                        "sim_time_s",
                        "monotonic_ns",
                        "teacher_raw_actions",
                        "action_is_pad",
                        "executed_raw_actions",
                        "executed_joint_targets",
                        "executed_ticks",
                        "executed_is_pad",
                        "executed_action_source",
                    )
                }
                teacher_samples = {key: [] for key in ("state", "command13", "tick", "monotonic_ns", "raw_actions", "action_is_pad", "failure")}
                branch_outcomes = []
                remaining = config.max_steps
                while remaining > 0:
                    check_deadline(deadline, "student decision")
                    actor_obs = replace(_actor_observation(observation, neutral), instruction=instruction)
                    if actor_obs.rgb is None:
                        raise ValueError("VLA collection requires a rendered RGB camera")
                    # This snapshot is taken after the actor's preceding steps.
                    # restore_state must restore physics, previous action, BAM,
                    # command, task state and RNG, not just qpos/qvel.
                    teacher_env.restore_state(main_env.clone_state())
                    if hasattr(teacher, "reset"):
                        teacher.reset(actor_obs.episode_id)
                    labels = np.zeros((config.action_horizon, 14), np.float32)
                    label_pad = np.ones(config.action_horizon, bool)
                    teacher_states = np.zeros((config.action_horizon, 61), np.float32)
                    teacher_commands = np.zeros((config.action_horizon, 13), np.float32)
                    teacher_ticks = np.full(config.action_horizon, -1, np.int64)
                    teacher_times = np.zeros(config.action_horizon, np.int64)
                    teacher_failure = False
                    for label_index in range(config.action_horizon):
                        check_deadline(deadline, "teacher action labels")
                        if _done(teacher_env.evaluator_state()):
                            break
                        teacher_obs = teacher_env.observe()
                        check_deadline(deadline, "teacher observation")
                        proposal = teacher.predict(teacher_obs)
                        check_deadline(deadline, "teacher inference")
                        raw = _prediction_rows(proposal, teacher_obs)[0]
                        diagnostics = getattr(proposal, "diagnostics", getattr(teacher, "last_diagnostics", {}))
                        teacher_commands[label_index] = diagnostics.get("teacher_command", teacher_obs.state[48:61])
                        teacher_states[label_index] = teacher_obs.state
                        teacher_states[label_index, 48:61] = teacher_commands[label_index]
                        teacher_ticks[label_index], teacher_times[label_index] = teacher_obs.tick, teacher_obs.monotonic_ns
                        teacher_failure |= diagnostics.get("phase") == "no_path"
                        labels[label_index] = raw
                        label_pad[label_index] = False
                        teacher_env.step(raw)
                    branch_terminal = teacher_env.evaluator_state()
                    if config.mode == "dagger" and config.verify_teacher_completion:
                        # A valid 50-step prefix does not prove the teacher can
                        # finish from the student's disturbed state. Complete
                        # the branch without adding future images to inputs.
                        with _without_intermediate_rendering(teacher_env):
                            for _ in range(config.max_steps):
                                check_deadline(deadline, "teacher task completion")
                                if _done(branch_terminal):
                                    break
                                tail_obs = teacher_env.observe()
                                check_deadline(deadline, "teacher completion observation")
                                tail_action = _prediction_rows(teacher.predict(tail_obs), tail_obs)[0]
                                check_deadline(deadline, "teacher completion inference")
                                teacher_env.step(tail_action)
                                branch_terminal = teacher_env.evaluator_state()
                        teacher_failure |= not _done(branch_terminal) or not bool(branch_terminal.get("success"))
                    teacher_failure |= bool(branch_terminal.get("fault")) or (_done(branch_terminal) and not bool(branch_terminal.get("success")))
                    # Diagnostic teacher-side evidence only. Keep the existing
                    # full-task supervision gate; a failed task must never be
                    # relabeled as a successful demonstration. These reasons
                    # distinguish future repairs from uninformative booleans.
                    strict = branch_terminal.get("strict_preset_result") or branch_terminal.get("foundation_result") or {}
                    branch_outcomes.append({
                        "source_tick": int(actor_obs.tick),
                        "branch_end_tick": int(branch_terminal.get("tick", actor_obs.tick)),
                        "branch_terminated": bool(_done(branch_terminal)),
                        "branch_task_success": bool(branch_terminal.get("success")),
                        "branch_failed_checks": list(strict.get("failed_checks", [])),
                        "branch_fault": branch_terminal.get("fault"),
                        "branch_recovery_events": int(branch_terminal.get("recovery_events", 0)),
                        "teacher_label_rejected": bool(teacher_failure),
                        "acceptance_scope": "complete_original_task" if config.mode == "dagger" and config.verify_teacher_completion else "teacher_prefix",
                    })
                    if label_pad.all():
                        branch_outcomes.pop()
                        break
                    # Count actual executed ticks since reset, not the task's
                    # absolute clock (which may include an initialization phase).
                    # Prefix scheduling is collector-only; it is never added to
                    # the observation or passed to the student policy.
                    prefix_remaining = max(0, config.teacher_prefix_ticks - (config.max_steps - remaining))
                    teacher_rollin = (prefix_remaining > 0 or
                                      (config.mode == "dagger" and config.teacher_rollin_probability > 0
                                       and rollin_rng.random() < config.teacher_rollin_probability))
                    if config.mode == "dagger" and not teacher_rollin:
                        check_deadline(deadline, "student inference admission")
                        proposed = _prediction_rows(student.predict(actor_obs), actor_obs)
                        check_deadline(deadline, "student inference")
                        action_source = 1
                    else:
                        proposed = labels[~label_pad]
                        action_source = 0
                    execute = min(config.execute_steps, len(proposed), remaining)
                    if prefix_remaining:
                        execute = min(execute, prefix_remaining)
                    raw_executed = np.zeros((config.execute_steps, 14), np.float32)
                    targets = np.zeros_like(raw_executed)
                    execute_ticks = np.full(config.execute_steps, -1, np.int64)
                    execute_pad = np.ones(config.execute_steps, bool)
                    with _without_intermediate_rendering(main_env):
                        for offset in range(execute):
                            check_deadline(deadline, "actor execution")
                            if _done(main_env.evaluator_state()):
                                break
                            execute_ticks[offset] = observation.tick
                            raw_executed[offset] = proposed[offset]
                            observation = main_env.step(proposed[offset])
                            targets[offset] = _applied_targets(main_env)
                            execute_pad[offset] = False
                            remaining -= 1
                    observation = main_env.observe()
                    check_deadline(deadline, "actor observation")
                    if execute_pad.all():
                        branch_outcomes.pop()
                        break
                    for key, value in {
                        "rgb": actor_obs.rgb,
                        "state": actor_obs.state,
                        "tick": actor_obs.tick,
                        "camera_tick": actor_obs.camera_tick if actor_obs.camera_tick is not None else actor_obs.tick,
                        "sim_time_s": actor_obs.sim_time_s,
                        "teacher_raw_actions": labels,
                        "monotonic_ns": actor_obs.monotonic_ns,
                        "action_is_pad": label_pad,
                        "executed_raw_actions": raw_executed,
                        "executed_joint_targets": targets,
                        "executed_ticks": execute_ticks,
                        "executed_is_pad": execute_pad,
                        "executed_action_source": np.uint8(action_source),
                    }.items():
                        samples[key].append(value)
                    for key, value in {
                        "state": teacher_states,
                        "command13": teacher_commands,
                        "tick": teacher_ticks,
                        "monotonic_ns": teacher_times,
                        "raw_actions": labels,
                        "action_is_pad": label_pad,
                        "failure": teacher_failure,
                    }.items():
                        teacher_samples[key].append(value)
                    if _done(main_env.evaluator_state()):
                        break
                if not samples["state"]:
                    raise ValueError(f"episode {task}/{episode_seed} produced no training samples")
                check_deadline(deadline, "episode finalization")
                filename = f"student_data/episode_{len(records):06d}.npz"
                teacher_filename = f"teacher_data/episode_{len(records):06d}.npz"
                branch_filename = f"teacher_data/episode_{len(records):06d}_branches.json"
                arrays = {name: np.asarray(values) for name, values in samples.items()}
                np.savez_compressed(destination / filename, **arrays)
                check_deadline(deadline, "actor archive write")
                terminal = main_env.evaluator_state()
                if config.mode == "teacher" and _done(terminal) and not terminal.get("success", False):
                    teacher_samples["failure"] = [True] * len(samples["state"])
                    for outcome in branch_outcomes:
                        outcome["teacher_label_rejected"] = True
                        outcome["episode_teacher_failed"] = True
                np.savez_compressed(destination / teacher_filename, **{key: np.asarray(values) for key, values in teacher_samples.items()})
                check_deadline(deadline, "teacher archive write")
                write_json(destination / branch_filename, {"schema_version": 1, "policy_input": False, "branches": branch_outcomes})
                scene_description = terminal.get("scene", {})
                metadata = scene_description.get("metadata", scene_description)
                records.append(
                    {
                        "file": filename,
                        "sha256": file_sha256(destination / filename, deadline=deadline),
                        "teacher_file": teacher_filename,
                        "teacher_sha256": file_sha256(destination / teacher_filename, deadline=deadline),
                        "teacher_branch_evidence_file": branch_filename,
                        "teacher_branch_evidence_sha256": file_sha256(destination / branch_filename, deadline=deadline),
                        "teacher_failure_rows": [i for i, failure in enumerate(teacher_samples["failure"]) if failure],
                        "episode_id": actor_obs.episode_id,
                        "seed": episode_seed,
                        "task_id": task,
                        "instruction": actor_obs.instruction,
                        "split": split,
                        "samples": len(samples["state"]),
                        "instruction_template": template,
                        "control_steps": int(sum((~p).sum() for p in samples["executed_is_pad"])),
                        "neutral_command": neutral,
                        "initial_condition": terminal.get("initial_condition"),
                        "initial_condition_sha256": terminal.get("initial_condition_sha256"),
                        "success": bool(terminal.get("success", False)),
                        "student_execution_decisions": sum(int(value) == 1 for value in samples["executed_action_source"]),
                        "teacher_execution_decisions": sum(int(value) == 0 for value in samples["executed_action_source"]),
                        "teacher_rollin_probability": config.teacher_rollin_probability,
                        "teacher_prefix_ticks": config.teacher_prefix_ticks,
                        "student_control_steps": sum(int((~pad).sum()) for source, pad in zip(samples["executed_action_source"], samples["executed_is_pad"]) if int(source) == 1),
                        "teacher_control_steps": sum(int((~pad).sum()) for source, pad in zip(samples["executed_action_source"], samples["executed_is_pad"]) if int(source) == 0),
                        "student_valid_label_rows": sum(int(source) == 1 and not failure for source, failure in zip(samples["executed_action_source"], teacher_samples["failure"])),
                        "student_independent_success": bool(config.mode == "dagger" and terminal.get("success", False)
                            and not terminal.get("recovery_events") and all(int(value) == 1 for value in samples["executed_action_source"])),
                        "fallen": bool(terminal.get("fallen", False)),
                        "camera_metadata": actor_obs.camera_metadata,
                        "embodiment": embodiment,
                        "model_signature": terminal.get("model_signature"),
                        "layout_id": metadata.get("layout_id", scene_description.get("layout_id")),
                        "geometry_hash": metadata.get("geometry_hash", scene_description.get("geometry_hash")),
                        "texture_id": metadata.get("texture_id", scene_description.get("texture_id")),
                        "layout_generalization_eligible": getattr(main_env, "scene", None) == "room_randomized",
                    }
                )
                if progress_callback is not None:
                    progress_callback(dict(records[-1]))
        index = {
            "schema_version": 1,
            "format": "microduck_episode_npz_v1",
            "contract_sha256": RobotContract(embodiment=embodiment).sha256,
            "collection": asdict(config),
            "embodiment": embodiment,
            "environment": {
                "profile": getattr(main_env, "profile", None),
                "scene": getattr(main_env, "scene", None),
                "actuator_profile": terminal.get("actuator_profile"),
                "action_delay_ticks": terminal.get("action_delay_ticks"),
            },
            "provenance": dict(provenance or {}),
            "teacher": asdict(teacher.manifest) if is_dataclass(getattr(teacher, "manifest", None)) else {"class": type(teacher).__qualname__, "artifact_manifest_available": False},
            "student": asdict(student.manifest)
            if is_dataclass(getattr(student, "manifest", None))
            else {"class": type(student).__qualname__ if student is not None else None, "artifact_manifest_available": False},
            "observation": "RGB + official 61D; goal command slots are neutral",
            "label_semantics": "teacher raw joint residuals from cloned student visited state",
            "executed_semantics": "raw residuals and applied absolute joint targets stored separately",
            "episodes": records,
        }
        index["split_audit"] = validate_split_domains(index)
        index["teacher_failure_policy"] = "retain all recordings; exclude flagged teacher-label rows from training by default; student failure alone is not teacher failure"
        index["privileged_data_boundary"] = "teacher_data stores teacher state/command13; student_data stores actor observations and supervised raw-action targets only"
        index["action_source_codes"] = {"0": "teacher", "1": "student"}
        if config.teacher_prefix_ticks:
            index["rollin"] = {"protocol": "teacher_prefix_then_student_v1", "prefix_ticks": config.teacher_prefix_ticks,
                               "selection": "fixed executed ticks since actor reset; student alone after handoff",
                               "counts_as_skill_success": False}
        elif config.teacher_rollin_probability:
            index["rollin"] = {"protocol": "teacher_student_chunk_mixture_v1", "probability": config.teacher_rollin_probability,
                               "selection": "independent Bernoulli per decision from an episode-local RNG", "counts_as_skill_success": False}
        if config.teacher_prefix_ticks or config.teacher_rollin_probability:
            index["counts_as_skill_success"] = False
            index["label_semantics"] = "teacher raw joint residuals from cloned mixed-policy visited states; not autonomous student evaluation"
        check_deadline(deadline, "collection index publication")
        if publish_index:
            write_json(destination / "index.json", index)
        return index
    except DataDeadlineExceeded as error:
        write_json(destination / "collection-incomplete.json", {
            "complete": False, "reason": "deadline", "error": str(error),
            "collection": asdict(config), "episodes": records,
            "interrupted_task": task, "interrupted_seed": episode_seed,
            "index_published": False, "counts_as_skill_success": False,
        })
        raise
    finally:
        try:
            if main_env is not None:
                main_env.close()
        finally:
            if teacher_env is not None and teacher_env is not main_env:
                teacher_env.close()


PREFIX_SUPERVISION = "after_student_action_v1"


def prefix_supervision_rows(path, samples):
    """Queries reached by actual student actions, not by teacher handoff alone."""
    with np.load(path, allow_pickle=False) as actor:
        sources = actor["executed_action_source"]
        padding = actor["executed_is_pad"]
        if (sources.shape != (samples,) or padding.ndim != 2 or padding.shape[0] != samples
                or not np.isin(sources, [0, 1]).all()):
            raise ValueError("invalid execution evidence for prefix supervision")
        preceding_student = np.r_[False, (sources[:-1] == 1) & (~padding[:-1]).any(axis=1)]
        return np.flatnonzero(~((sources == 1) & preceding_student)).tolist()


def supervision_exclusions(record, path=None):
    """Validate exclusions independently of the teacher-failure diagnostic flag."""
    scope = record.get("supervision_scope")
    rows = record.get("supervision_excluded_rows", [])
    if scope is None and not rows:
        return set()
    if (scope != PREFIX_SUPERVISION or type(record.get("teacher_prefix_ticks")) is not int
            or record["teacher_prefix_ticks"] <= 0 or not isinstance(rows, list)
            or any(type(i) is not int or not 0 <= i < record["samples"] for i in rows)
            or rows != sorted(set(rows))):
        raise ValueError("invalid fixed-prefix supervision selection")
    if path is not None and rows != prefix_supervision_rows(path, record["samples"]):
        raise ValueError("prefix supervision selection differs from recorded execution")
    return set(rows)


class EpisodeDataset:
    """A torch-compatible dataset without requiring torch for data tools."""

    def __init__(
        self,
        index_path: str | Path,
        split: str = "train",
        action_horizon: int | None = None,
        *,
        verify_hashes: bool = True,
        cache_dir: str | Path | None = None,
        include_teacher_failures: bool = False,
        mmap_cache_episodes: int = 128,
        deadline: float | None = None,
    ):
        # Used only during eager validation, not retained by training loaders.
        check_deadline(deadline, "dataset validation")
        self.index_path = Path(index_path)
        if self.index_path.is_dir():
            self.index_path /= "index.json"
        self.index = json.loads(self.index_path.read_text())
        if self.index.get("format") != "microduck_episode_npz_v1":
            raise ValueError("unsupported MicroDuck dataset format")
        self.split_audit = validate_split_domains(self.index)
        if split not in ("train", "val", "test", "all"):
            raise ValueError("split must be train, val, test, or all")
        self.episodes = [e for e in self.index["episodes"] if split == "all" or e["split"] == split]
        available = int(self.index["collection"]["action_horizon"])
        self.action_horizon = available if action_horizon is None else action_horizon
        if not 1 <= self.action_horizon <= available:
            raise ValueError("requested horizon exceeds the collected teacher label horizon")
        self.sample_map: list[tuple[int, int]] = []
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_index = -1
        self._cache: dict[str, np.ndarray] = {}
        if type(mmap_cache_episodes) is not int or mmap_cache_episodes < 1:
            raise ValueError("mmap_cache_episodes must be a positive integer")
        # Five array mappings per episode; leave headroom for both the training
        # and validation datasets plus the model/runtime's file descriptors.
        try:
            import resource
            soft, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
            if soft != resource.RLIM_INFINITY:
                mmap_cache_episodes = min(mmap_cache_episodes, max(1, (soft - 128) // 10))
        except ImportError:
            mmap_cache_episodes = min(mmap_cache_episodes, 64)
        self.mmap_cache_episodes = mmap_cache_episodes
        self._mmap_cache: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()
        for episode_index, record in enumerate(self.episodes):
            check_deadline(deadline, "dataset episode validation")
            path = self.index_path.parent / record["file"]
            if not path.resolve().is_relative_to(self.index_path.parent.resolve()):
                raise ValueError("episode paths must stay within their dataset directory")
            if verify_hashes and file_sha256(path, deadline=deadline) != record["sha256"]:
                raise ValueError(f"episode hash mismatch: {record['file']}")
            if record.get("teacher_file"):
                teacher_path = (self.index_path.parent / record["teacher_file"]).resolve()
                if not teacher_path.is_relative_to(self.index_path.parent.resolve()):
                    raise ValueError("teacher paths must stay within their dataset directory")
                if verify_hashes and file_sha256(teacher_path, deadline=deadline) != record["teacher_sha256"]:
                    raise ValueError("teacher archive hash mismatch")
            if record.get("teacher_branch_evidence_file"):
                evidence_path = (self.index_path.parent / record["teacher_branch_evidence_file"]).resolve()
                if not evidence_path.is_relative_to(self.index_path.parent.resolve()):
                    raise ValueError("teacher evidence paths must stay within their dataset directory")
                if verify_hashes and file_sha256(evidence_path, deadline=deadline) != record["teacher_branch_evidence_sha256"]:
                    raise ValueError("teacher branch evidence hash mismatch")
                evidence = json.loads(evidence_path.read_text())
                if evidence.get("policy_input") is not False or len(evidence.get("branches", [])) != record["samples"]:
                    raise ValueError("invalid teacher branch evidence boundary or sample count")
            if split == "train" and record.get("teacher_prefix_ticks", 0) and record.get("supervision_scope") != PREFIX_SUPERVISION:
                raise ValueError("prefix training must explicitly exclude teacher-reached queries")
            excluded = supervision_exclusions(record, path)
            failures = excluded | (set() if include_teacher_failures else set(record.get("teacher_failure_rows", [])))
            self.sample_map.extend((episode_index, row) for row in range(record["samples"]) if row not in failures)
        check_deadline(deadline, "dataset validation completion")

    def __len__(self) -> int:
        return len(self.sample_map)

    def __getstate__(self):
        # Spawned data-loader workers reopen mappings, never pickle whole arrays.
        state = self.__dict__.copy()
        state.update(_cache_index=-1, _cache={}, _mmap_cache=OrderedDict())
        return state

    def close(self):
        self._cache_index, self._cache = -1, {}
        for values in self._mmap_cache.values():
            for value in values.values():
                value._mmap.close()
        self._mmap_cache.clear()

    def __getitem__(self, item: int) -> dict[str, Any]:
        episode_index, row = self.sample_map[item]
        record = self.episodes[episode_index]
        if self._cache_index != episode_index:
            source = self.index_path.parent / record["file"]
            fields = ("rgb", "state", "teacher_raw_actions", "action_is_pad", "tick")
            if self.cache_dir is None:
                with np.load(source, allow_pickle=False) as archive:
                    self._cache = {key: archive[key] for key in fields}
            elif record["sha256"] in self._mmap_cache:
                self._cache = self._mmap_cache.pop(record["sha256"])
                self._mmap_cache[record["sha256"]] = self._cache
            else:
                # Compressed episode NPZ is convenient for transfer, but
                # repeatedly decompressing a whole episode for each shuffled
                # sample starves a GPU trainer. Materialize a hash-addressed,
                # shared, read-only mmap cache once, without changing the source.
                cache = self.cache_dir / record["sha256"]
                if not cache.exists():
                    with tempfile.TemporaryDirectory(prefix=".episode-", dir=self.cache_dir) as temporary:
                        staging = Path(temporary) / "arrays"
                        staging.mkdir()
                        with np.load(source, allow_pickle=False) as archive:
                            for key in fields:
                                np.save(staging / f"{key}.npy", archive[key], allow_pickle=False)
                        try:
                            staging.rename(cache)
                        except OSError:
                            # Another DDP rank can publish the same complete
                            # directory while this rank is unpacking it.
                            if not cache.is_dir():
                                raise
                self._cache = {key: np.load(cache / f"{key}.npy", mmap_mode="r", allow_pickle=False) for key in fields}
                self._mmap_cache[record["sha256"]] = self._cache
                while len(self._mmap_cache) > self.mmap_cache_episodes:
                    _, evicted = self._mmap_cache.popitem(last=False)
                    for value in evicted.values():
                        value._mmap.close()
            self._cache_index = episode_index
        values = self._cache
        state = values["state"][row].astype(np.float32, copy=True)
        if record["neutral_command"] and np.any(state[48:61] != 0):
            raise ValueError("goal dataset leaks nonneutral teacher command slots")
        return {
            "image_rgb": values["rgb"][row].copy(),
            "state61": state,
            "instruction": record["instruction"],
            "task_id": record["task_id"],
            "action": values["teacher_raw_actions"][row, : self.action_horizon].copy(),
            "action_is_pad": values["action_is_pad"][row, : self.action_horizon].copy(),
            "episode_id": record["episode_id"],
            "tick": int(values["tick"][row]),
        }


def load_dataset(index_path: str | Path, split: str = "train", action_horizon: int | None = None) -> EpisodeDataset:
    return EpisodeDataset(index_path, split, action_horizon)


def merge_datasets(index_paths: list[str | Path], output_dir: str | Path, *, deadline: float | None = None,
                   publish_index: bool = True) -> dict[str, Any]:
    """Aggregate DAgger rounds without relabeling splits or losing provenance."""
    check_deadline(deadline, "merge initialization")
    if not index_paths:
        raise ValueError("at least one source dataset is required")
    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("aggregate dataset output must be empty")
    sources = []
    for value in index_paths:
        path = Path(value)
        if path.is_dir():
            path /= "index.json"
        dataset = EpisodeDataset(path, "all", deadline=deadline)  # validates every source hash
        sources.append((path, dataset.index))
    reference = sources[0][1]
    for _, index in sources[1:]:
        dynamics = ("profile", "actuator_profile", "action_delay_ticks")
        mismatch = any(index.get("environment", {}).get(key) != reference.get("environment", {}).get(key) for key in dynamics)
        if index["contract_sha256"] != reference["contract_sha256"] or index["collection"]["action_horizon"] != reference["collection"]["action_horizon"] or mismatch:
            raise ValueError("cannot aggregate datasets with different robot contracts or label horizons")
    # Identical initial conditions must not migrate across splits as a result of
    # a new DAgger round's seed/split configuration.
    split_by_scenario: dict[tuple[str, int], str] = {}
    for _, index in sources:
        for episode in index["episodes"]:
            scenario = (episode["task_id"], episode["seed"])
            previous = split_by_scenario.setdefault(scenario, episode["split"])
            if previous != episode["split"]:
                raise ValueError("same task/seed occurs in different splits across DAgger rounds")
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    (destination / "student_data").mkdir(exist_ok=True)
    (destination / "teacher_data").mkdir(exist_ok=True)
    for path, index in sources:
        source_hash = file_sha256(path, deadline=deadline)
        for record in index["episodes"]:
            check_deadline(deadline, "episode merge")
            filename = f"student_data/episode_{len(records):06d}.npz"
            shutil.copyfile(path.parent / record["file"], destination / filename)
            check_deadline(deadline, "actor archive merge")
            teacher_record = {}
            if record.get("teacher_file"):
                teacher_filename = f"teacher_data/episode_{len(records):06d}.npz"
                shutil.copyfile(path.parent / record["teacher_file"], destination / teacher_filename)
                check_deadline(deadline, "teacher archive merge")
                teacher_record["teacher_file"] = teacher_filename
            if record.get("teacher_branch_evidence_file"):
                evidence_filename = f"teacher_data/episode_{len(records):06d}_branches.json"
                shutil.copyfile(path.parent / record["teacher_branch_evidence_file"], destination / evidence_filename)
                teacher_record["teacher_branch_evidence_file"] = evidence_filename
            records.append(
                {**record, **teacher_record, "file": filename, "source_index_sha256": source_hash, "source_provenance": index.get("provenance", {}), "source_environment": index.get("environment", {})}
            )
    result = {**reference, "episodes": records, "provenance": {"aggregation": [{"index_sha256": file_sha256(path, deadline=deadline), "source": str(path.resolve())} for path, _ in sources]}}
    result["split_audit"] = validate_split_domains(result)
    check_deadline(deadline, "merged index publication")
    if publish_index:
        write_json(destination / "index.json", result)
    return result


def compute_normalization(dataset: EpisodeDataset, *, deadline: float | None = None) -> dict[str, Any]:
    """Compute train-only moments, excluding padded actions and held-out episodes."""
    check_deadline(deadline, "normalization initialization")
    if any(e["split"] != "train" for e in dataset.episodes):
        raise ValueError("normalization must be fitted on the training split only")
    sums = {"state": np.zeros(61, np.float64), "action": np.zeros(14, np.float64)}
    squares = {name: np.zeros_like(value) for name, value in sums.items()}
    counts = {"state": 0, "action": 0}
    for sample_index in range(len(dataset)):
        check_deadline(deadline, "normalization sample loading")
        sample = dataset[sample_index]
        check_deadline(deadline, "normalization moments")
        for name, values in (("state", sample["state61"][None]), ("action", sample["action"][~sample["action_is_pad"]])):
            values = values.astype(np.float64)
            if not np.isfinite(values).all():
                raise ValueError("normalization encountered nonfinite data")
            sums[name] += values.sum(axis=0)
            squares[name] += np.square(values).sum(axis=0)
            counts[name] += len(values)
    result: dict[str, Any] = {}
    for name in sums:
        if not counts[name]:
            raise ValueError("training split contains no valid data")
        mean = sums[name] / counts[name]
        std = np.sqrt(np.maximum(squares[name] / counts[name] - mean**2, 0))
        # Constant channels (especially neutral command slots) stay well scaled.
        std = np.where(std < 1e-6, 1.0, std)
        result[name] = {"mean": mean.tolist(), "std": std.tolist(), "count": counts[name]}
    result["provenance"] = {"fit_split": "train", "index_sha256": file_sha256(dataset.index_path, deadline=deadline), "episode_hashes": [e["sha256"] for e in dataset.episodes]}
    return result


def replay_episode(env: CollectionEnvironment, index_path: str | Path, episode: int = 0, *, atol: float = 2e-5, deadline: float | None = None) -> dict[str, Any]:
    """Replay the recorded executed actions, checking observations and targets."""
    check_deadline(deadline, "actor replay initialization")
    index_path = Path(index_path)
    if index_path.is_dir():
        index_path /= "index.json"
    index = json.loads(index_path.read_text())
    record = index["episodes"][episode]
    path = index_path.parent / record["file"]
    if file_sha256(path, deadline=deadline) != record["sha256"]:
        raise ValueError("episode hash mismatch")
    observation = env.reset(seed=record["seed"], task=record["task_id"])
    state_error = target_error = 0.0
    steps = 0
    with np.load(path, allow_pickle=False) as archive:
        for row in range(record["samples"]):
            check_deadline(deadline, "actor replay observation")
            observation = _actor_observation(observation, record["neutral_command"])
            if observation.tick != int(archive["tick"][row]):
                raise AssertionError("replay tick mismatch")
            state_error = max(state_error, float(np.max(np.abs(observation.state - archive["state"][row]))))
            for offset, raw in enumerate(archive["executed_raw_actions"][row]):
                if archive["executed_is_pad"][row, offset]:
                    continue
                check_deadline(deadline, "actor replay step")
                observation = env.step(raw)
                target_error = max(target_error, float(np.max(np.abs(_applied_targets(env) - archive["executed_joint_targets"][row, offset]))))
                steps += 1
    check_deadline(deadline, "actor replay completion")
    if max(state_error, target_error) > atol:
        raise AssertionError(f"replay divergence: state={state_error:g}, target={target_error:g}")
    return {"steps": steps, "state_max_abs_error": state_error, "target_max_abs_error": target_error, "atol": atol}
