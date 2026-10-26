"""Train the native LightX2V TinyVLA from recorded MicroDuck episodes.

Run under torchrun for DDP. This module does not launch remote jobs or claim
that a checkpoint is behaviorally competent; closed-loop evaluation is a
separate required step after fitting the flow-matching objective.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import signal
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import HOME_POLICY, POLICY_JOINT_NAMES, RobotContract
from .data import EpisodeDataset, compute_normalization, file_sha256, write_json


@dataclass(frozen=True)
class TrainingConfig:
    dataset_index: str
    output_dir: str
    model_path: str | None = None
    backbone_path: str | None = None
    backbone_model_id: str = "Qwen/Qwen3.5-0.8B"
    backbone_revision: str | None = None
    normalization_path: str | None = None
    dataset_cache_dir: str | None = None
    dataset_mmap_cache_episodes: int = 256
    embodiment: str = "legs"
    action_horizon: int = 50
    learning_rate: float = 3e-5
    language_learning_rate: float | None = None
    weight_decay: float = 0.01
    global_batch_size: int = 128
    micro_batch_size: int = 4
    max_steps: int = 5000
    warmup_steps: int = 100
    save_every: int = 500
    eval_every: int = 100
    max_val_batches: int = 20
    grad_clip_norm: float = 1.0
    seed: int = 0
    precision: str = "bf16"
    device: str = "auto"
    resume_from: str | None = None
    ddp_find_unused_parameters: bool = True
    include_teacher_failures: bool = False
    unfreeze_lm_last_n: int = 0
    freeze_vision: bool = True
    max_wall_seconds: float | None = None
    closed_loop_every: int = 0
    closed_loop_seeds: int = 5
    closed_loop_patience: int = 3
    closed_loop_reserve_seconds: float = 0.
    semantic_refresh_steps: int = 5
    action_padding: str = "free"
    dagger_sampling_fraction: float | None = None
    validation_sampling: str = "sequential"
    executed_prefix_weight: float = 1.0
    flow_loss: str = "mse"
    flow_huber_delta: float = 1.0

    def __post_init__(self):
        if type(self.dataset_mmap_cache_episodes) is not int or self.dataset_mmap_cache_episodes < 1:
            raise ValueError("dataset_mmap_cache_episodes must be a positive integer")
        if min(self.global_batch_size, self.micro_batch_size, self.max_steps, self.save_every, self.eval_every, self.max_val_batches) < 1:
            raise ValueError("batch sizes, steps, and intervals must be positive")
        if self.global_batch_size % self.micro_batch_size:
            raise ValueError("global batch size must divide into complete micro batches")
        if not 1 <= self.action_horizon <= 256:
            raise ValueError("invalid action horizon")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")
        if self.warmup_steps < 0 or not math.isfinite(self.weight_decay) or self.weight_decay < 0 or not math.isfinite(self.grad_clip_norm) or self.grad_clip_norm <= 0:
            raise ValueError("invalid optimizer configuration")
        if self.precision not in ("bf16", "float32"):
            raise ValueError("precision must be bf16 or float32")
        if self.embodiment not in ("legs", "rollers"):
            raise ValueError("embodiment must be legs or rollers")
        if self.unfreeze_lm_last_n < 0:
            raise ValueError("unfreeze_lm_last_n must be nonnegative")
        if self.language_learning_rate is not None:
            if (type(self.language_learning_rate) not in (int, float)
                    or not math.isfinite(self.language_learning_rate) or self.language_learning_rate <= 0):
                raise ValueError("language_learning_rate must be positive and finite")
            if not self.unfreeze_lm_last_n:
                raise ValueError("language_learning_rate requires explicitly unfrozen language layers")
        if self.closed_loop_every < 0 or self.closed_loop_seeds < 1 or self.closed_loop_patience < 1 or (self.max_wall_seconds is not None and self.max_wall_seconds <= 0):
            raise ValueError("invalid validation interval or wall-time limit")
        if (type(self.closed_loop_reserve_seconds) not in (int, float)
                or not math.isfinite(self.closed_loop_reserve_seconds) or self.closed_loop_reserve_seconds < 0
                or self.closed_loop_reserve_seconds and (not self.closed_loop_every or self.max_wall_seconds is None
                                                         or not math.isfinite(self.max_wall_seconds)
                                                         or self.closed_loop_reserve_seconds >= self.max_wall_seconds)):
            raise ValueError("closed-loop reserve requires a positive remaining optimization allowance and bounded validation")
        if isinstance(self.semantic_refresh_steps, bool) or not isinstance(self.semantic_refresh_steps, int) or not 1 <= self.semantic_refresh_steps <= 5:
            raise ValueError("fresh-observation training requires semantic_refresh_steps between 1 and 5; cached semantics need a separate training recipe")
        if self.action_padding not in ("free", "training_noise_path"):
            raise ValueError("unsupported action padding trajectory")
        if self.dagger_sampling_fraction is not None and not 0 < self.dagger_sampling_fraction < 1:
            raise ValueError("DAgger sampling must retain both original and corrective data")
        if self.validation_sampling not in ("sequential", "stratified_v1"):
            raise ValueError("unsupported validation sampling")
        if isinstance(self.executed_prefix_weight, bool) or not math.isfinite(self.executed_prefix_weight) or self.executed_prefix_weight < 1:
            raise ValueError("executed_prefix_weight must be finite and at least one")
        if self.flow_loss not in ("mse", "huber"):
            raise ValueError("flow_loss must be mse or huber")
        if isinstance(self.flow_huber_delta, bool) or not math.isfinite(self.flow_huber_delta) or self.flow_huber_delta <= 0:
            raise ValueError("flow_huber_delta must be positive and finite")


@dataclass(frozen=True)
class TrainingResult:
    checkpoint_dir: str
    optimizer_steps: int
    samples_seen: int
    final_loss: float
    validation_loss: float | None
    world_size: int
    device: str
    precision: str
    stop_reason: str = "unknown"
    requested_steps: int | None = None
    step_target_reached: bool = False


def _complete_qualified_validation(report, checkpoint, tasks, seeds):
    """Only a complete fixed population may enter or remain in best selection."""
    rows = report.get("episodes", [])
    if (report.get("protocol") != "project_qualified_evaluation_v2" or report.get("complete") is not True
            or report.get("purpose") != "validation" or report.get("evaluator_role") != "student"
            or report.get("ablation") is not None or set(report.get("tasks", {})) != set(tasks)
            or len(rows) != len(tasks) * len(seeds)
            or report.get("checkpoint_sha256") != file_sha256(Path(checkpoint) / "model.safetensors")):
        raise ValueError("qualified checkpoint needs complete, matching fixed-validation evidence")
    for task in tasks:
        selected = [row for row in rows if row.get("task") == task]
        if ([row.get("seed") for row in selected] != list(seeds)
                or len({row.get("initial_condition_sha256") for row in selected}) != len(seeds)):
            raise ValueError("qualified validation population is incomplete or changed")


def _require_complete_qualified_best(output_dir, tasks, seeds):
    output_dir = Path(output_dir).resolve()
    try:
        receipt = json.loads((output_dir / "best-validation.json").read_text())
        checkpoint = Path(receipt["checkpoint"]).resolve()
        if output_dir not in checkpoint.parents:
            raise ValueError("best checkpoint is outside its training stage")
        report = json.loads((checkpoint / "closed-loop-validation.json").read_text())
        _complete_qualified_validation(report, checkpoint, tasks, seeds)
        if receipt["checkpoint_sha256"] != report["checkpoint_sha256"]:
            raise ValueError("best checkpoint hash changed")
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise RuntimeError("no complete qualified validation candidate remains; development cannot start") from error
    return checkpoint


def _run_qualified_validation(output_dir, checkpoint, evaluator, selector, *, deadline, **kwargs):
    """Return None on a budget stop, preserving only an already complete best."""
    tasks, seeds = kwargs["tasks"], list(kwargs["seeds"])
    kwargs["seeds"] = seeds
    # Keep every control state, executed action and physical trace so failed
    # fixed-validation episodes can be replayed without another GPU inference.
    kwargs["record_dir"] = Path(checkpoint) / "closed-loop-trajectories"
    try:
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("qualified training deadline reached before validation")
        report = evaluator(checkpoint, **kwargs, deadline=deadline)
    except TimeoutError as error:
        write_json(Path(checkpoint) / "closed-loop-validation-incomplete.json", {
            "protocol": "qualified_validation_budget_stop_v1", "complete": False,
            "error": str(error), "deadline_monotonic": deadline, "counts_as_skill_success": False,
            "checkpoint": str(Path(checkpoint).resolve()), "used_for_checkpoint_selection": False})
        _require_complete_qualified_best(output_dir, tasks, seeds)
        return None
    _complete_qualified_validation(report, checkpoint, tasks, seeds)
    write_json(Path(checkpoint) / "closed-loop-validation.json", report)
    return selector(output_dir, checkpoint, report)


def _qualified_on_rank_zero(callback, *, rank, world_size, distributed):
    """Share completion or failure before any rank can wait at a later barrier."""
    outcome = [None, None]
    if rank == 0:
        try:
            outcome[0] = callback()
        except Exception as error:
            outcome[1] = f"{type(error).__name__}: {error}"
    if world_size > 1:
        distributed.broadcast_object_list(outcome, src=0)
    if outcome[1] is not None:
        raise RuntimeError(f"qualified validation failed on rank zero: {outcome[1]}")
    return outcome[0]


class TinyVLACollator:
    """Use the same image/token preprocessing as native inference."""

    def __init__(self, preprocessor: Any, normalization: dict[str, Any]):
        self.preprocessor = preprocessor
        self.normalization = normalization.get("norm_stats", normalization)

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        state_stats = self.normalization["state"]
        action_stats = self.normalization["action"]
        state_mean = np.asarray(state_stats["mean"], np.float32)
        state_std = np.asarray(state_stats["std"], np.float32)
        action_mean = np.asarray(action_stats["mean"], np.float32)
        action_std = np.asarray(action_stats["std"], np.float32)
        if state_mean.shape != (61,) or action_mean.shape != (14,) or state_std.shape != (61,) or action_std.shape != (14,):
            raise ValueError("MicroDuck normalization must have state61/action14 statistics")
        if not np.isfinite(np.concatenate((state_mean, state_std, action_mean, action_std))).all() or np.any(state_std < 0) or np.any(action_std < 0):
            raise ValueError("normalization statistics must be finite with nonnegative standard deviations")
        state_std, action_std = np.maximum(state_std, 1e-6), np.maximum(action_std, 1e-6)
        rows = [
            self.preprocessor.prepare(
                sample["image_rgb"],
                (sample["state61"] - state_mean) / state_std,
                sample["instruction"],
                normalized=True,
            )
            for sample in samples
        ]
        if not rows or any(set(row) != set(rows[0]) for row in rows):
            raise ValueError("preprocessing produced inconsistent batch keys")
        batch: dict[str, Any] = {}
        for key in rows[0]:
            values = [row[key] for row in rows]
            if not all(torch.is_tensor(value) for value in values):
                raise TypeError(f"preprocessor field {key} must be a batched tensor")
            if all(value.shape[1:] == values[0].shape[1:] for value in values):
                batch[key] = torch.cat(values, dim=0)
            elif all(value.ndim == 2 and value.shape[0] == 1 for value in values):
                # Language tokens and masks can have different prompt lengths.
                width = max(value.shape[1] for value in values)
                padded = values[0].new_zeros((len(values), width))
                for index, value in enumerate(values):
                    padded[index, : value.shape[1]] = value[0]
                batch[key] = padded
            else:
                raise ValueError(f"incompatible preprocessed shapes for {key}")
        actions = np.stack([(sample["action"] - action_mean) / action_std for sample in samples])
        is_pad = np.stack([sample["action_is_pad"] for sample in samples]).astype(bool)
        padded_actions = np.zeros((*actions.shape[:2], 64), np.float32)
        padded_actions[..., :14] = actions
        padded_actions[is_pad] = 0.0
        batch["action"] = torch.from_numpy(padded_actions)
        batch["action_is_pad"] = torch.from_numpy(is_pad)
        mask = torch.zeros((len(samples), 64), dtype=torch.bool)
        mask[:, :14] = True
        batch["action_dim_mask"] = mask
        return batch


def expand_recipe_environment(value):
    """Expand portable recipe paths and reject an accidentally literal $VAR."""
    if isinstance(value, dict):
        return {key: expand_recipe_environment(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_recipe_environment(item) for item in value]
    if isinstance(value, str):
        expanded = os.path.expandvars(value)
        if re.search(r"\$\{?[A-Za-z_][A-Za-z0-9_]*", expanded):
            raise ValueError(f"unresolved recipe environment variable: {value}")
        return expanded
    return value


def resolve_checkpoint_source(source):
    """An explicit best-validation receipt replaces guessed final-step names."""
    path = Path(source) if source else None
    if path is not None and path.is_file() and path.name == "best-validation.json":
        record = json.loads(path.read_text())
        checkpoint = Path(record["checkpoint"])
        if file_sha256(checkpoint / "model.safetensors") != record["checkpoint_sha256"]:
            raise ValueError("selected best checkpoint changed")
        return str(checkpoint)
    return source


def _loss(model: Any, batch: dict[str, Any], *, flow_loss="mse", flow_huber_delta=1.0):
    options = {"flow_loss": flow_loss, "flow_huber_delta": flow_huber_delta} if flow_loss != "mse" else {}
    output = model(batch, **options)
    loss = output[0] if isinstance(output, tuple) else output["loss"] if isinstance(output, dict) else output
    if loss.ndim != 0:
        loss = loss.mean()
    return loss


def action_error_metrics(prediction, batch, normalization):
    """Measure executed dimensions in radians, excluding padded future rows."""
    return {name: math.sqrt(total / count) for name, (total, count) in action_error_totals(prediction, batch, normalization).items() if count}


def action_error_totals(prediction, batch, normalization):
    """Sufficient statistics for correctly pooling differently padded batches."""
    import torch

    stats = normalization.get("norm_stats", normalization)["action"]
    scale = torch.as_tensor(stats["std"], device=prediction.device, dtype=torch.float32).clamp_min(1e-6)
    error = (prediction[..., :14].float() - batch["action"][..., :14].float()) * scale
    valid = ~batch["action_is_pad"].bool()
    metrics = {}
    for name, horizon in (("first", 1), ("prefix5", 5), ("full", error.shape[1])):
        selected = error[:, :horizon][valid[:, :horizon]]
        if selected.numel():
            metrics[name + "_rmse_rad"] = (float(selected.square().sum()), selected.numel())
    return metrics


def stratified_validation_indices(dataset, maximum):
    """Fixed task-balanced samples spanning every task's ordered episode timeline."""
    groups = {}
    for index, (episode, row) in enumerate(dataset.sample_map):
        groups.setdefault(dataset.episodes[episode]["task_id"], []).append(index)
    if maximum < len(groups):
        raise ValueError("validation budget must cover every task")
    selected = []
    ordered = sorted(groups)
    for position, task in enumerate(ordered):
        quota = maximum // len(ordered) + (position < maximum % len(ordered))
        values = groups[task]
        positions = np.linspace(0, len(values) - 1, min(len(values), quota)).round().astype(int)
        selected.extend(values[int(item)] for item in positions)
    return selected


def dagger_sample_weights(dataset, fraction):
    """Change sampling probability without duplicating data or moving held-out rows."""
    modes = [dataset.episodes[episode].get("collection_mode") for episode, _ in dataset.sample_map]
    if not modes or set(modes) != {"teacher", "dagger"} or any(row["split"] != "train" for row in dataset.episodes):
        raise ValueError("balanced DAgger requires explicitly tagged teacher and dagger train samples")
    counts = {mode: modes.count(mode) for mode in ("teacher", "dagger")}
    return [(fraction if mode == "dagger" else 1 - fraction) / counts[mode] for mode in modes]


def configure_trainable_parameters(model, config):
    """Freeze the backbone first, then opt into its final semantic layers."""
    semantic = getattr(model, "semantic", None)
    vlm = getattr(semantic, "vlm", None)
    if vlm is not None:
        language = vlm.language_model
        layers = language.layers
        if config.unfreeze_lm_last_n > len(layers):
            raise ValueError("unfreeze_lm_last_n exceeds the retained Qwen language layers")
        language.requires_grad_(False)
        if config.unfreeze_lm_last_n:
            for layer in layers[-config.unfreeze_lm_last_n :]:
                layer.requires_grad_(True)
        vlm.get_input_embeddings().requires_grad_(False)
        vlm.visual.requires_grad_(not config.freeze_vision)
    elif config.unfreeze_lm_last_n:
        raise ValueError("cannot unfreeze language layers in a model without a Qwen backbone")
    model.config.freeze_lm = config.unfreeze_lm_last_n == 0
    model.config.freeze_vision_encoder = config.freeze_vision
    model.config.lm_unfreeze_last_n = config.unfreeze_lm_last_n
    names = {name: parameter.numel() for name, parameter in model.named_parameters() if parameter.requires_grad}
    return {
        "trainable_parameters": sum(names.values()),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "unfreeze_lm_last_n": config.unfreeze_lm_last_n,
        "freeze_vision": config.freeze_vision,
        "trainable_parameter_names": names,
    }


def build_optimizer_groups(model, config: TrainingConfig):
    """Use parameter identity to separate unfrozen LM blocks from policy layers.

    Call after configure_trainable_parameters. The default keeps the original
    single group and parameter order, including when a legacy recipe unfreezes
    language layers at the same learning rate as the rest of the model.
    """
    named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not named:
        raise ValueError("model has no trainable parameters")
    if config.language_learning_rate is None:
        partitions = [("all_trainable", config.learning_rate, named)]
    else:
        vlm = getattr(getattr(model, "semantic", None), "vlm", None)
        language = getattr(vlm, "language_model", None)
        if language is None:
            raise ValueError("separate language learning rate requires a language backbone")
        language_ids = {id(parameter) for parameter in language.parameters() if parameter.requires_grad}
        partitions = [
            ("policy", config.learning_rate, [(name, p) for name, p in named if id(p) not in language_ids]),
            ("language", config.language_learning_rate, [(name, p) for name, p in named if id(p) in language_ids]),
        ]
        if not all(parameters for _, _, parameters in partitions):
            raise ValueError("separate language learning rate requires trainable policy and language parameters")
    groups, records = [], []
    for name, rate, parameters in partitions:
        groups.append({"params": [parameter for _, parameter in parameters], "lr": rate, "weight_decay": config.weight_decay})
        records.append({"name": name, "learning_rate": rate, "weight_decay": config.weight_decay,
                        "parameter_count": sum(parameter.numel() for _, parameter in parameters),
                        "parameters": [{"name": key, "shape": list(parameter.shape)} for key, parameter in parameters]})
    return groups, {"protocol": "tinyvla_optimizer_groups_v1", "groups": records,
                    "unfreeze_lm_last_n": config.unfreeze_lm_last_n, "freeze_vision": config.freeze_vision}


def validate_optimizer_resume(config, previous_config, record, saved_record, sidecar_record):
    """Do not attach Adam moments to a changed multi-rate parameter grouping."""
    if config.language_learning_rate is None and previous_config.get("language_learning_rate") is None:
        return  # Legacy single-group checkpoint behavior remains available.
    fields = ("language_learning_rate", "learning_rate", "weight_decay", "warmup_steps",
              "unfreeze_lm_last_n", "freeze_vision")
    if (any(previous_config.get(name) != getattr(config, name) for name in fields)
            or saved_record != record or sidecar_record != record):
        raise ValueError("resume optimizer groups differ or lack provenance; use model_path for a separately recorded experiment")


def train(config: TrainingConfig, *, model: Any | None = None, preprocessor: Any | None = None) -> TrainingResult:
    """Run real optimizer steps; injected components are for bounded CPU tests."""
    import torch
    import torch.distributed as distributed
    from torch.nn.parallel import DistributedDataParallel
    from torch.utils.data import DataLoader, DistributedSampler, Subset, WeightedRandomSampler

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if config.dagger_sampling_fraction is not None and world_size != 1:
        raise ValueError("balanced DAgger sampler currently requires the authorized single-GPU configuration")
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device_name = config.device
    if device_name == "auto":
        device_name = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", local_rank)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if config.global_batch_size % (world_size * config.micro_batch_size):
        raise ValueError("global_batch_size must be divisible by world_size * micro_batch_size")
    accumulation = config.global_batch_size // (world_size * config.micro_batch_size)
    started_group = False
    if world_size > 1 and not distributed.is_initialized():
        group_options = {}
        if config.closed_loop_reserve_seconds:
            from datetime import timedelta
            # Rank-zero physical evaluation can exceed the backend's default
            # collective timeout. Its propagated training deadline is the
            # relevant bound; peers must still be present to receive failure.
            group_options["timeout"] = timedelta(seconds=max(1800., config.max_wall_seconds + 120.))
        distributed.init_process_group("nccl" if device.type == "cuda" else "gloo", **group_options)
        started_group = True
    random.seed(config.seed + rank)
    np.random.seed(config.seed + rank)
    torch.manual_seed(config.seed + rank)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed + rank)
    output_dir = Path(config.output_dir)
    if not config.resume_from and ((output_dir / "training_result.json").exists() or (output_dir / "metrics.jsonl").exists() or any(output_dir.glob("checkpoint-*"))):
        raise FileExistsError("training outputs already exist; use a new output directory or resume")
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        distributed.barrier()
    started_at = time.monotonic()
    stop_requested = [False]
    stop_signal = [None]
    previous_handlers = {}
    if __import__("threading").current_thread() is __import__("threading").main_thread():
        def request_stop(signum, _frame):
            stop_signal[0] = signum
            stop_requested[0] = True
        for signum in (signal.SIGTERM, signal.SIGINT):
            previous_handlers[signum] = signal.signal(signum, request_stop)
    try:
        cache_dir = config.dataset_cache_dir or str(output_dir / "data_cache")
        train_data = EpisodeDataset(config.dataset_index, "train", config.action_horizon, cache_dir=cache_dir, include_teacher_failures=config.include_teacher_failures, mmap_cache_episodes=config.dataset_mmap_cache_episodes)
        val_data = EpisodeDataset(config.dataset_index, "val", config.action_horizon, cache_dir=cache_dir, include_teacher_failures=config.include_teacher_failures, mmap_cache_episodes=config.dataset_mmap_cache_episodes)
        if (train_data.index.get("provenance", {}).get("training_eligible") is False
                or any(row.get("source_provenance", {}).get("training_eligible") is False for row in train_data.index["episodes"])):
            raise ValueError("dataset is explicitly marked ineligible for training; diagnostic fixtures cannot enter optimization")
        if rank == 0:
            write_json(output_dir / "data_cache.json", {"protocol": "bounded_readonly_mmap_lru_v1", "requested_episodes": config.dataset_mmap_cache_episodes,
                       "train_effective_limit": train_data.mmap_cache_episodes, "val_effective_limit": val_data.mmap_cache_episodes,
                       "sample_order_changed": False, "source_hash_verification": True})
        if train_data.index.get("embodiment", "legs") != config.embodiment or any(e.get("embodiment", "legs") != config.embodiment for e in train_data.index["episodes"]):
            raise ValueError("training embodiment does not match the dataset; legs and rollers require separate models")
        if not len(train_data):
            raise ValueError("training split is empty")
        if not len(val_data):
            raise ValueError("validation split has no eligible teacher labels; collect valid held-out episodes before training")
        command_modes = {bool(row.get("neutral_command", False)) for row in train_data.index["episodes"]}
        if len(command_modes) != 1:
            raise ValueError("preset command and neutral goal datasets require separate model artifacts")
        observation_profile = "microduck_rgb_proprio_neutral_goal_v1" if command_modes == {True} else "microduck_rgb_proprio_preset_command_v1"
        foundation = train_data.index.get("protocol") == "foundation_stability_v1"
        foundation_walk = train_data.index.get("protocol") == "foundation_walk_v1"
        from .qualified_training import DATASET_PROTOCOL, dataset_artifact_contract
        qualified = train_data.index.get("protocol") == DATASET_PROTOCOL
        qualified_contract = dataset_artifact_contract(train_data.index, config.embodiment) if qualified else {}
        training_deadline = started_at + config.max_wall_seconds if config.max_wall_seconds is not None else None
        optimization_deadline = (training_deadline - config.closed_loop_reserve_seconds
                                 if qualified and training_deadline is not None else training_deadline)
        if qualified:
            observation_profile = qualified_contract["observation_profile"]
        if foundation:
            if command_modes != {True} or {row["task_id"] for row in train_data.index["episodes"]} != {"stand_still"}:
                raise ValueError("foundation data requires neutral stand_still observations")
            observation_profile = "microduck_rgb_proprio_neutral_foundation_v1"
        if foundation_walk:
            from .foundation_walk import ACCEPTANCE_SCOPE, COMMAND_PROGRAM, CONDITIONS, OBSERVATION_PROFILE, TASK
            if (command_modes != {False} or {row["task_id"] for row in train_data.index["episodes"]} != {TASK.task_id}
                    or train_data.index.get("provenance", {}).get("initial_conditions") != CONDITIONS
                    or train_data.index.get("provenance", {}).get("command_program") != COMMAND_PROGRAM
                    or train_data.index.get("acceptance_scope") != ACCEPTANCE_SCOPE or train_data.index.get("counts_as_skill_success") is not False):
                raise ValueError("walk curriculum requires its versioned same-tick command and initial-condition contract")
            observation_profile = OBSERVATION_PROFILE
        norm_path = output_dir / "normalization.json"
        if config.normalization_path:
            normalization = json.loads(Path(config.normalization_path).read_text())
        elif config.resume_from and (Path(config.resume_from) / "normalization.json").exists():
            normalization = json.loads((Path(config.resume_from) / "normalization.json").read_text())
        else:
            norm_result: list[Any] = [None, None]
            if rank == 0:
                try:
                    norm_result[0] = compute_normalization(train_data)
                except Exception as error:
                    norm_result[1] = f"normalization failed: {type(error).__name__}: {error}"
            if world_size > 1:
                distributed.broadcast_object_list(norm_result, src=0)
            if norm_result[1]:
                raise ValueError(norm_result[1])
            normalization = norm_result[0]
        if rank == 0:
            write_json(norm_path, normalization)
            write_json(output_dir / "training_config.json", asdict(config))
        source = resolve_checkpoint_source(config.resume_from or config.model_path)
        if foundation_walk and source and (Path(source) / "artifact_manifest.json").is_file():
            source_manifest = json.loads((Path(source) / "artifact_manifest.json").read_text())
            if source_manifest.get("acceptance_scope") == "foundation_stability":
                raise ValueError("standing-to-walk normalized flow migration is not implemented; initialize the independent walk BC from public ckpt_v4_duckft")
        if model is None:
            if not source:
                raise ValueError("model_path or resume_from is required")
            from lightx2v.models.networks.tinyvla.modeling_tinyvla import TinyVLAModel

            model = TinyVLAModel.from_pretrained(source, backbone_path=config.backbone_path, backbone_revision=config.backbone_revision)
        if preprocessor is None:
            from transformers import AutoTokenizer

            from lightx2v.models.networks.tinyvla.processing import TinyVLAPreprocessor

            if not config.backbone_path:
                raise ValueError("backbone_path is required for native tokenizer preprocessing")
            tokenizer = AutoTokenizer.from_pretrained(config.backbone_path, revision=config.backbone_revision)
            preprocessor = TinyVLAPreprocessor(model.config, tokenizer, normalization=normalization, device="cpu")
        trainable = configure_trainable_parameters(model, config)
        if rank == 0:
            write_json(output_dir / "trainable_parameters.json", trainable)
        model.to(device)
        if model.config.chunk_size != config.action_horizon:
            raise ValueError("model chunk_size must match the collected/trained action_horizon")
        raw_model = model
        if world_size > 1:
            model = DistributedDataParallel(model, device_ids=[device.index] if device.type == "cuda" else None, find_unused_parameters=config.ddp_find_unused_parameters)
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        groups, optimizer_record = build_optimizer_groups(raw_model, config)
        optimizer = torch.optim.AdamW(groups, lr=config.learning_rate, weight_decay=config.weight_decay)
        if rank == 0:
            write_json(output_dir / "optimizer_groups.json", optimizer_record)
        step = samples_seen = epoch = 0
        if config.resume_from:
            state_path = Path(config.resume_from) / "training_state.pt"
            previous_config = json.loads((state_path.parent / "training_config.json").read_text())
            for name, legacy_default in (("flow_loss", "mse"), ("flow_huber_delta", 1.0), ("executed_prefix_weight", 1.0)):
                if previous_config.get(name, legacy_default) != getattr(config, name):
                    raise ValueError("resume objective differs; use model_path for a separately recorded experiment")
            saved = torch.load(state_path, map_location="cpu", weights_only=False)
            sidecar = state_path.parent / "optimizer_groups.json"
            validate_optimizer_resume(config, previous_config, optimizer_record, saved.get("optimizer_groups"),
                                      json.loads(sidecar.read_text()) if sidecar.is_file() else None)
            if saved["dataset_sha256"] != file_sha256(train_data.index_path):
                raise ValueError("resume dataset differs; start a new DAgger iteration with model_path instead")
            optimizer.load_state_dict(saved["optimizer"])
            step, samples_seen, epoch = saved["step"], saved["samples_seen"], saved["epoch"]
            if step >= config.max_steps:
                raise ValueError("max_steps must exceed the saved optimizer step to continue training")
            # Resume restores optimizer/model state at an epoch boundary; it is
            # deliberately not advertised as a bit-exact distributed replay.
        collator = TinyVLACollator(preprocessor, normalization)
        sampler = DistributedSampler(train_data, num_replicas=world_size, rank=rank, shuffle=True, seed=config.seed) if world_size > 1 else None
        generator = torch.Generator().manual_seed(config.seed + rank)
        if config.dagger_sampling_fraction is not None:
            weights = dagger_sample_weights(train_data, config.dagger_sampling_fraction)
            sampler = WeightedRandomSampler(weights, len(weights), replacement=True, generator=generator)
            write_json(output_dir / "sampling.json", {"protocol": "teacher_dagger_mixture_v1", "dagger_probability": config.dagger_sampling_fraction,
                       "eligible_samples": len(weights), "original_validation_unchanged": True})
        if config.resume_from and "rng" in saved:
            rng = saved["rng"]
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch"])
            generator.set_state(rng["loader"])
            if device.type == "cuda" and rng.get("cuda") is not None:
                torch.cuda.set_rng_state_all(rng["cuda"])
        loader = DataLoader(train_data, batch_size=config.micro_batch_size, sampler=sampler, shuffle=sampler is None, collate_fn=collator, num_workers=0, generator=generator, drop_last=True)
        if not len(loader):
            raise ValueError("training split per rank is smaller than micro_batch_size; reduce it for a smoke run")
        validation_data = val_data
        if len(val_data) and config.validation_sampling == "stratified_v1":
            indices = stratified_validation_indices(val_data, config.micro_batch_size * config.max_val_batches)
            validation_data = Subset(val_data, indices)
            if rank == 0:
                write_json(output_dir / "validation-population.json", {"protocol": config.validation_sampling,
                           "dataset_sha256": file_sha256(val_data.index_path), "indices": indices,
                           "samples": len(indices), "available_validation_samples": len(val_data),
                           "sampling": "task balanced; evenly spaced across each task's episode timeline", "full_validation_population": len(indices) == len(val_data)})
        val_loader = DataLoader(validation_data, batch_size=config.micro_batch_size, shuffle=False, collate_fn=collator, num_workers=0) if len(val_data) else None
        use_bf16 = config.precision == "bf16" and device.type == "cuda"
        if use_bf16 and not torch.cuda.is_bf16_supported():
            raise RuntimeError("requested BF16 is unavailable on this CUDA device")
        effective_precision = "bf16" if use_bf16 else "float32"

        def amp():
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_bf16 else nullcontext()

        def move(batch):
            moved = {name: value.to(device) for name, value in batch.items()}
            if config.executed_prefix_weight != 1:
                weights = torch.ones(config.action_horizon, device=device)
                weights[:5] = config.executed_prefix_weight
                moved["action_time_weights"] = weights
            return moved

        validation_actions = {}

        def validate() -> float | None:
            nonlocal validation_actions
            if val_loader is None:
                return None
            raw_model.eval()
            total = count = 0
            action_totals = {}
            # A fixed RNG context keeps validation noise comparable between steps
            # and does not consume the training flow-noise stream.
            cuda_devices = [device.index] if device.type == "cuda" else []
            with torch.random.fork_rng(devices=cuda_devices), torch.no_grad():
                torch.manual_seed(config.seed + 991)
                for batch_index, batch in enumerate(val_loader):
                    if batch_index >= config.max_val_batches:
                        break
                    batch = move(batch)
                    with amp():
                        value = _loss(raw_model, batch)
                        if batch_index == 0 or config.validation_sampling == "stratified_v1":
                            padding_kwargs = {"active_action_dim": 14} if config.action_padding == "training_noise_path" else {}
                            predicted = raw_model.predict_action_chunk(batch, **padding_kwargs)
                            for name, (squared, elements) in action_error_totals(predicted, batch, normalization).items():
                                previous = action_totals.get(name, (0., 0))
                                action_totals[name] = (previous[0] + squared, previous[1] + elements)
                    total += float(value) * len(batch["action"])
                    count += len(batch["action"])
            validation_actions = {name: math.sqrt(squared / elements) for name, (squared, elements) in action_totals.items() if elements}
            raw_model.train()
            return total / count if count else None

        checkpoint = output_dir / f"checkpoint-{step:06d}"

        def save() -> Path:
            path = output_dir / f"checkpoint-{step:06d}"
            if rank == 0:
                path.mkdir(parents=True, exist_ok=True)
                raw_model.save_pretrained(path)
                if config.backbone_path:
                    # Runtime caches are machine-specific; released model
                    # config retains the portable model identity and true pin.
                    model_config = json.loads((path / "config.json").read_text())
                    model_config["lm_model_name"] = config.backbone_model_id
                    model_config["backbone_revision"] = config.backbone_revision or getattr(raw_model.config, "backbone_revision", None)
                    write_json(path / "config.json", model_config)
                write_json(path / "normalization.json", normalization)
                write_json(path / "training_config.json", asdict(config))
                write_json(path / "trainable_parameters.json", trainable)
                write_json(path / "optimizer_groups.json", optimizer_record)
                torch.save(
                    {"optimizer": optimizer.state_dict(), "optimizer_groups": optimizer_record, "step": step, "samples_seen": samples_seen, "epoch": epoch, "dataset_sha256": file_sha256(train_data.index_path), "rng": {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(), "loader": generator.get_state(), "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None}, "resume_semantics": "optimizer and random state restored; loader resumes at a new epoch"}, path / "training_state.pt"
                )
                write_json(
                    path / "provenance.json",
                    {
                        "dataset_index": str(train_data.index_path.resolve()),
                        "dataset_sha256": file_sha256(train_data.index_path),
                        "world_size": world_size,
                        "precision": effective_precision,
                        "source_model": source,
                        "hardware_validated": False,
                        "closed_loop_validated": False,
                        "docker_image_id": os.environ.get("MICRODUCK_IMAGE_ID"),
                        "implementation_sha256": file_sha256(os.environ["MICRODUCK_SOURCE_MANIFEST"]) if os.environ.get("MICRODUCK_SOURCE_MANIFEST") and Path(os.environ["MICRODUCK_SOURCE_MANIFEST"]).is_file() else None,
                    },
                )
                backbone_revision = config.backbone_revision or getattr(raw_model.config, "backbone_revision", None)
                if backbone_revision and re.fullmatch(r"[0-9a-f]{40}", backbone_revision) and config.action_horizon == 50:
                    write_json(
                        path / "artifact_manifest.json",
                        {
                            "format": "lightx2v-tinyvla-v1",
                            "model_variant": "lightx2v_microduck_50",
                            "source_revision": "603f5045b45126af011418befc899e3c67d04346",
                            "implementation_sha256": file_sha256(os.environ["MICRODUCK_SOURCE_MANIFEST"]) if os.environ.get("MICRODUCK_SOURCE_MANIFEST") and Path(os.environ["MICRODUCK_SOURCE_MANIFEST"]).is_file() else None,
                            "backbone_revision": backbone_revision,
                            "observation_profile": observation_profile,
                            "simulation_profile": train_data.index.get("environment", {}).get("profile"),
                            "trained_tasks": sorted({row["task_id"] for row in train_data.index["episodes"]}),
                            "acceptance_scope": "foundation_walk" if foundation_walk else "foundation_stability" if foundation else "task_behavior",
                            **({"command_program": COMMAND_PROGRAM, "uses_preset_command13": True, "claims_visual_language_autonomy": False} if foundation_walk else {}),
                            "counts_as_skill_success": False,
                            "initial_conditions": train_data.index.get("provenance", {}).get("initial_conditions"),
                            **qualified_contract,
                            "preset_clock": "program" if command_modes == {False} else None,
                            "embodiment_digest": RobotContract(embodiment=config.embodiment).sha256,
                            "embodiment": config.embodiment,
                            "control_profile": "raw_residual",
                            **{
                                name: {"path": filename, "sha256": file_sha256(path / filename)}
                                for name, filename in (("checkpoint", "model.safetensors"), ("config", "config.json"), ("normalization", "normalization.json"))
                            },
                            "support": {"mode": "none_trained"},
                            "inference": {"semantic_refresh_steps": config.semantic_refresh_steps, "semantic_training": "fresh_observation", "action_padding": config.action_padding,
                                          "semantic_padding_policy": getattr(raw_model.config, "semantic_padding_policy", "legacy")},
                            "action_contract": {
                                "kind": "joint_residual",
                                "units": "rad",
                                "control_hz": 50,
                                "observation_hz": 10,
                                "execute_steps": 5,
                                "scale": 1.0,
                                "joint_names": list(POLICY_JOINT_NAMES),
                                "home_pose": list(HOME_POLICY),
                            },
                        },
                    )
            if world_size > 1:
                distributed.barrier()
            return path

        def validate_checkpoint():
            from .checkpoint_eval import evaluate_checkpoint, select_best
            if foundation:
                from .foundation_eval import evaluate_foundation_checkpoint as evaluate_checkpoint
            elif foundation_walk:
                from .foundation_walk_eval import evaluate_walk_checkpoint as evaluate_checkpoint
            elif qualified:
                from .qualified_eval import evaluate_qualified_checkpoint as evaluate_checkpoint
            kwargs = {"tasks": sorted({row["task_id"] for row in train_data.index["episodes"]}),
                      "seeds": range(500000, 500000 + config.closed_loop_seeds),
                      "device": str(device), "precision": "bf16" if use_bf16 else "fp32"}
            with torch.random.fork_rng(devices=[device.index] if device.type == "cuda" else []):
                torch.manual_seed(config.seed + 500000)
                if qualified:
                    return _run_qualified_validation(output_dir, checkpoint, evaluate_checkpoint, select_best,
                        deadline=training_deadline, backbone=config.backbone_path, **kwargs)
                report = evaluate_checkpoint(checkpoint, config.backbone_path, **kwargs)
                write_json(checkpoint / "closed-loop-validation.json", report)
                return select_best(output_dir, checkpoint, report)

        def qualified_rank_zero(callback):
            return _qualified_on_rank_zero(callback, rank=rank, world_size=world_size, distributed=distributed)

        final_loss = 0.0
        validation_loss = None
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pending = 0
        accumulated_loss = 0.0
        stagnant_checks = 0
        qualified_validation_timed_out = False
        stop_reason = None
        stop_acknowledged = False
        while step < config.max_steps:
            if isinstance(sampler, DistributedSampler):
                sampler.set_epoch(epoch)
            for batch_index, batch in enumerate(loader):
                deadline_reached = (not stop_requested[0] and optimization_deadline is not None
                                    and time.monotonic() >= optimization_deadline)
                should_stop = stop_requested[0] or deadline_reached
                if world_size > 1:
                    stop_tensor = torch.tensor(int(should_stop), device=device)
                    distributed.all_reduce(stop_tensor, op=distributed.ReduceOp.MAX)
                    should_stop = bool(stop_tensor.item())
                if should_stop:
                    stop_acknowledged = True
                    if stop_signal[0] is not None:
                        stop_reason = "signal"
                    elif stop_reason is None:
                        # Another DDP rank may request a stop before this rank
                        # reaches its own deadline. Do not invent its cause.
                        stop_reason = "optimization_deadline" if deadline_reached else "peer_requested_stop"
                    stop_requested[0] = True
                    optimizer.zero_grad(set_to_none=True)
                    checkpoint = save()
                    break
                pending += 1
                should_step = pending == accumulation
                sync = model.no_sync() if world_size > 1 and not should_step else nullcontext()
                with sync:
                    with amp():
                        loss = _loss(model, move(batch), flow_loss=config.flow_loss, flow_huber_delta=config.flow_huber_delta)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"nonfinite flow loss at step {step}")
                    (loss / accumulation).backward()
                accumulated_loss += float(loss.detach())
                samples_seen += len(batch["action"]) * world_size
                if not should_step:
                    continue
                torch.nn.utils.clip_grad_norm_(parameters, config.grad_clip_norm)
                scale = min(1.0, (step + 1) / max(1, config.warmup_steps))
                for group, record in zip(optimizer.param_groups, optimizer_record["groups"], strict=True):
                    group["lr"] = record["learning_rate"] * scale
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                final_loss = accumulated_loss / pending
                if world_size > 1:
                    metric = torch.tensor(final_loss, device=device)
                    distributed.all_reduce(metric)
                    final_loss = float(metric / world_size)
                pending = 0
                accumulated_loss = 0.0
                if step % config.eval_every == 0 or step == config.max_steps:
                    validation_loss = validate()
                if rank == 0:
                    with (output_dir / "metrics.jsonl").open("a") as stream:
                        stream.write(
                            json.dumps(
                                {"step": step, "train_loss": final_loss, "train_loss_objective": config.flow_loss,
                                 "flow_huber_delta": config.flow_huber_delta if config.flow_loss == "huber" else None,
                                 "validation_loss": validation_loss, "validation_loss_objective": "mse",
                                 "validation_action_metrics": validation_actions, "samples_seen": samples_seen, "lr": optimizer.param_groups[0]["lr"],
                                 "learning_rates": {record["name"]: group["lr"] for group, record in
                                                    zip(optimizer.param_groups, optimizer_record["groups"], strict=True)}}, allow_nan=False
                            )
                            + "\n"
                        )
                if step % config.save_every == 0 or step == config.max_steps or (config.closed_loop_every and step % config.closed_loop_every == 0):
                    checkpoint = save()
                if config.closed_loop_every and (step % config.closed_loop_every == 0 or step == config.max_steps):
                    improved = qualified_rank_zero(validate_checkpoint) if qualified else (validate_checkpoint() if rank == 0 else None)
                    if rank == 0 or qualified:
                        if improved is None:
                            qualified_validation_timed_out = True
                            stop_requested[0] = True
                            stop_reason = "qualified_validation_deadline"
                            if rank == 0:
                                write_json(output_dir / "stop_reason.json", {"reason": "qualified_validation_deadline", "step": step})
                        else:
                            stagnant_checks = 0 if improved else stagnant_checks + 1
                        if improved is not None and stagnant_checks >= config.closed_loop_patience:
                            stop_requested[0] = True
                            stop_reason = "closed_loop_plateau"
                            if rank == 0:
                                write_json(output_dir / "stop_reason.json", {"reason": "closed_loop_plateau", "checks_without_improvement": stagnant_checks, "step": step})
                    if world_size > 1:
                        distributed.barrier()
                if step == config.max_steps:
                    break
            epoch += 1
            # A signal can arrive in the last micro-batch of an epoch. Only
            # leave after the shared stop check has cleared partial gradients
            # and saved the committed optimizer state on every DDP rank.
            if stop_acknowledged:
                break
        if stop_signal[0] is not None:
            stop_reason = "signal"
        elif stop_reason is None:
            stop_reason = "max_steps" if step >= config.max_steps else "unknown"
        stop_record = {
            "reason": stop_reason, "step": step, "requested_steps": config.max_steps,
            "step_target_reached": step >= config.max_steps,
            "signal_number": stop_signal[0], "checks_without_improvement": stagnant_checks,
            "optimization_window_elapsed_seconds": time.monotonic() - started_at,
            "training_max_wall_seconds": config.max_wall_seconds,
            "closed_loop_reserve_seconds": config.closed_loop_reserve_seconds,
            "optimization_window_limit_seconds": (optimization_deadline - started_at
                                                   if optimization_deadline is not None else None),
            "counts_as_skill_success": False,
        }
        # Persist before the final physical check: the optimizer may have
        # stopped even if validation subsequently fails or runs out of time.
        if rank == 0:
            write_json(output_dir / "stop_reason.json", stop_record)
        elif world_size > 1:
            write_json(output_dir / f"stop_reason.rank-{rank}.json", stop_record)
        def needs_final_validation():
            return (config.closed_loop_every and step > 0 and not qualified_validation_timed_out
                    and not (checkpoint / "closed-loop-validation.json").exists())

        if qualified and config.closed_loop_every:
            if qualified_rank_zero(needs_final_validation):
                qualified_rank_zero(validate_checkpoint)
            qualified_rank_zero(lambda: _require_complete_qualified_best(output_dir,
                sorted({row["task_id"] for row in train_data.index["episodes"]}),
                range(500000, 500000 + config.closed_loop_seeds)))
        elif rank == 0 and needs_final_validation():
            validate_checkpoint()
        result = TrainingResult(str(checkpoint), step, samples_seen, final_loss, validation_loss, world_size,
                                str(device), effective_precision, stop_reason, config.max_steps, step >= config.max_steps)
        if rank == 0:
            write_json(output_dir / "training_result.json", asdict(result))
        return result
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        if started_group:
            distributed.destroy_process_group()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train native TinyVLA from a MicroDuck episode dataset (torchrun-compatible)")
    parser.add_argument("--config", required=True, help="TrainingConfig JSON")
    parser.add_argument("--dataset-index")
    parser.add_argument("--output-dir")
    parser.add_argument("--model-path")
    parser.add_argument("--backbone-path")
    parser.add_argument("--resume-from")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--micro-batch-size", type=int)
    parser.add_argument("--global-batch-size", type=int)
    arguments = parser.parse_args()
    payload = json.loads(Path(arguments.config).read_text())
    for key, value in vars(arguments).items():
        if key != "config" and value is not None:
            payload[key] = value
    result = train(TrainingConfig(**expand_recipe_environment(payload)))
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
