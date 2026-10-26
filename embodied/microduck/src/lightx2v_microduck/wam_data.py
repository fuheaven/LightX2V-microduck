"""Real MuJoCo navigation demonstrations and FastWAM joint video/action batches.

The dataset's public model tensors contain RGB, language, and proprioception.
Planner diagnostics and evaluator truth stay in episode reports, never tensors.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from .navigation import ACTION_NAMES, DECISION_TICKS, NAVIGATION_CONTRACT, NAVIGATION_SOURCE_SHA256, NavigationTeacher, VisualSearchTeacher, intent_from_action, student_state
from .navigation_search import SEARCH_SOURCE_SHA256
from .teachers import TEACHER_SOURCE_SHA256
from .navigation_dataset_metadata import read_indexed_metadata


PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"
COLLECTOR_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def split_for_seed(seed):
    # All instructions using one layout seed stay in the same split.
    bucket = int(hashlib.sha256(f"microduck-navigation-v1:{int(seed)}".encode()).hexdigest()[:8], 16) % 10
    return "train" if bucket < 8 else "validation" if bucket == 8 else "test"


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def reference_processor(stats, horizon=32):
    """Use the pinned author's normalizer for reference inference and training."""
    import torch
    from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
    from fastwam.datasets.lerobot.transforms.action_state_merger import ConcatLeftAlign

    shape = {key: [{"key": "default", "shape": dim, "raw_shape": dim}]
             for key, dim in (("action", 5), ("state", 48))}
    processor = FastWAMProcessor(
        shape_meta=shape, num_obs_steps=horizon + 1, num_output_cameras=1,
        action_output_dim=5, proprio_output_dim=48, action_state_transforms=None,
        use_stepwise_action_norm=False, norm_default_mode="min/max", norm_exception_mode=None,
        action_state_merger=ConcatLeftAlign(), train_transforms=None, val_transforms=None)
    tensors = {key: {"default": {name: torch.as_tensor(value, dtype=torch.float32)
                               for name, value in stats[key]["default"].items()}}
               for key in ("state", "action")}
    processor.set_normalizer_from_stats(tensors)
    return processor


def collect_episode(env, output, *, task, seed, warmup_ticks=50, teacher_kind="visual_search", teacher_options=None):
    calibrated_teachers = ("visual_search_measured_motion", "visual_search_trajectory_gain", "visual_search_long_trajectory", "visual_search_persistent_trajectory", "visual_search_early_persistent", "visual_search_pose_checked_persistent", "visual_search_pose_diverse_persistent")
    if teacher_options and teacher_kind not in calibrated_teachers:
        raise ValueError("teacher options are supported only for measured-motion calibration")
    if teacher_kind in calibrated_teachers and set(teacher_options or {}) != {"calibration", "expected_calibration_sha256"}:
        raise ValueError("measured-motion collection requires a pinned calibration")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    env.render_enabled = False
    env.reset(seed, task)
    for _ in range(warmup_ticks):
        env.step(intent_from_action([0, 0, 0, 0, 1], episode_id=env.episode_id, tick=env.tick))
    env.begin_task_at_current_pose()
    env.render_enabled = True
    observation = env.observe()
    if teacher_kind not in ("privileged", "visual_search", "visual_search_obstacle_aware",
                            "visual_search_initial_scan", "visual_search_local_views", "visual_search_latched_view",
                            "visual_search_pivot_view", "visual_search_camera_frustum", "visual_search_occluded_pockets",
                            "visual_search_measured_motion", "visual_search_observed_gain", "visual_search_trajectory_gain",
                            "visual_search_opportunity_weighted", "visual_search_spawn_support",
                            "visual_search_progress_limited", "visual_search_two_view_route", "visual_search_long_trajectory", "visual_search_persistent_trajectory", "visual_search_early_persistent", "visual_search_pose_checked_persistent", "visual_search_pose_diverse_persistent"):
        raise ValueError("unknown navigation teacher")
    visual_search = teacher_kind.startswith("visual_search")
    teacher = (VisualSearchTeacher(env, obstacle_aware=teacher_kind == "visual_search_obstacle_aware")
               if visual_search else NavigationTeacher(env))
    experiment_sha256 = None
    if teacher_kind in ("visual_search_initial_scan", "visual_search_local_views", "visual_search_latched_view",
                        "visual_search_pivot_view", "visual_search_camera_frustum", "visual_search_occluded_pockets",
                        "visual_search_measured_motion", "visual_search_observed_gain", "visual_search_trajectory_gain",
                        "visual_search_opportunity_weighted", "visual_search_spawn_support",
                        "visual_search_progress_limited", "visual_search_two_view_route", "visual_search_long_trajectory", "visual_search_persistent_trajectory", "visual_search_early_persistent", "visual_search_pose_checked_persistent", "visual_search_pose_diverse_persistent"):
        from .navigation_teacher_experiments import (
            EXPERIMENT_SOURCE_SHA256, EXPERIMENTAL_TEACHERS)

        teacher = EXPERIMENTAL_TEACHERS[teacher_kind](env, **(teacher_options or {}))
        experiment_sha256 = EXPERIMENT_SOURCE_SHA256
    frames, states, ticks = [observation.rgb], [student_state(observation)], [observation.tick]
    actions, diagnostics, trace_states, trace_actions, trace_ticks = [], [], [], [], []
    initial_world = env.evaluator_state()
    while not env.terminated:
        action = teacher.predict(observation)
        start_tick = env.tick
        env.render_enabled = False
        for _ in range(DECISION_TICKS):
            trace_states.append(observation.state)
            trace_ticks.append(observation.tick)
            observation = env.step(intent_from_action(action, episode_id=env.episode_id, tick=env.tick))
            trace_actions.append(env.last_action.copy())
            if env.terminated:
                break
        env.render_enabled = True
        observation = env.observe()
        if env.tick - start_tick == DECISION_TICKS:
            actions.append(action)
            diagnostics.append(dict(teacher.last_diagnostics))
            frames.append(observation.rgb)
            states.append(student_state(observation))
            ticks.append(observation.tick)
    result = env.evaluator_state()
    if not actions:
        raise ValueError("episode contains no complete navigation transition")
    recovery = any(d.get("planner_phase", d.get("phase")) in ("recover", "no_path") for d in diagnostics)
    eligible = bool(result["success"] and not recovery and not result["collision"]
                    and not result["ever_fallen"] and result["recovery_events"] == 0)
    np.savez_compressed(output / "navigation.npz", rgb=np.stack(frames), proprio=np.stack(states),
                        action=np.stack(actions), ticks=np.asarray(ticks, dtype=np.int64))
    np.savez_compressed(output / "control_trace.npz", state61=np.stack(trace_states),
                        raw_action=np.stack(trace_actions), ticks=np.asarray(trace_ticks, dtype=np.int64))
    manifest = {
        "schema": NAVIGATION_CONTRACT, "task": task, "instruction": env.task.instruction,
        "seed": int(seed), "split": split_for_seed(seed), "profile": env.profile,
        "scene": env.description.as_dict(), "action_names": ACTION_NAMES,
        "action_interval_ticks": DECISION_TICKS, "control_hz": 50, "camera_hz": 10,
        "student_state": "obs61[:48]; command13 excluded", "student_camera": "head_rgb_256",
        "teacher": teacher_kind + "_astar_with_official_onnx", "eligible_training_demonstration": eligible,
        "teacher_uses_privileged_obstacle_map": True,
        "search_visibility_gated": visual_search,
        "teacher_visibility": "privileged_head_camera_segmentation_min16pixels" if visual_search else "omniscient",
        "teacher_source_sha256": NAVIGATION_SOURCE_SHA256,
        "teacher_dependency_sha256": {"search": SEARCH_SOURCE_SHA256, "classical_planner": TEACHER_SOURCE_SHA256},
        "teacher_experiment_sha256": experiment_sha256,
        "teacher_options": teacher_options or {},
        "collector_source_sha256": COLLECTOR_SOURCE_SHA256,
        "result": result, "warmup_ticks": warmup_ticks, "decision_count": len(actions),
        "model_signature": env._signature, "robot_signature": env._robot_signature,
        "initial_teacher_truth": initial_world, "teacher_diagnostics": diagnostics,
        "hardware_validated": False,
        "files": {name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                  for name in ("navigation.npz", "control_trace.npz")},
    }
    # Some evaluator values are numpy scalars/arrays. Materialize for provenance.
    manifest = json.loads(json.dumps(manifest, default=lambda value: value.tolist()))
    write_json(output / "episode.json", manifest)
    return manifest


def build_index(root):
    """Audit episode artifacts and derive normalization exclusively from train."""
    root = Path(root)
    episodes, arrays = [], {"state": [], "action": []}
    for path in sorted(root.glob("*/episode.json")):
        meta = json.loads(path.read_text())
        if meta.get("schema") != NAVIGATION_CONTRACT:
            raise ValueError(f"wrong navigation schema: {path}")
        for filename, expected in meta["files"].items():
            if hashlib.sha256((path.parent / filename).read_bytes()).hexdigest() != expected:
                raise ValueError(f"episode checksum mismatch: {path.parent / filename}")
        with np.load(path.parent / "navigation.npz", allow_pickle=False) as data:
            n = len(data["action"])
            if (data["action"].shape != (n, 5) or data["proprio"].shape != (n + 1, 48)
                    or data["rgb"].shape != (n + 1, 256, 256, 3)
                    or data["rgb"].dtype != np.uint8
                    or not np.all(np.diff(data["ticks"]) == DECISION_TICKS)
                    or not np.isfinite(data["action"]).all() or not np.isfinite(data["proprio"]).all()):
                raise ValueError(f"invalid navigation tensor alignment: {path}")
            if meta["split"] != split_for_seed(meta["seed"]):
                raise ValueError(f"split does not match immutable layout assignment: {path}")
            if meta["eligible_training_demonstration"] and meta["split"] == "train":
                arrays["state"].append(data["proprio"][:-1].copy())
                arrays["action"].append(data["action"].copy())
        episodes.append({"path": str(path.parent.relative_to(root)), "task": meta["task"],
                         "seed": meta["seed"], "split": meta["split"],
                         "eligible": meta["eligible_training_demonstration"], "frames": n + 1,
                         "search_visibility_gated": meta.get("search_visibility_gated", False),
                         "success": bool(meta["result"]["success"]), "sha256": meta["files"]["navigation.npz"],
                         "metadata_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    keys = [(e["task"], e["seed"]) for e in episodes]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate task/layout seed in dataset")
    stats = {"schema": NAVIGATION_CONTRACT, "normalization_split": "train"}
    if arrays["state"]:
        for key, parts in arrays.items():
            values = np.concatenate(parts)
            stats[key] = {"default": {
                "global_min": values.min(0).tolist(), "global_max": values.max(0).tolist(),
                "global_mean": values.mean(0).tolist(), "global_std": values.std(0).tolist(),
                "global_q01": np.quantile(values, .01, axis=0).tolist(),
                "global_q99": np.quantile(values, .99, axis=0).tolist(),
            }}
        stats["source_episodes"] = [e["sha256"] for e in episodes if e["eligible"] and e["split"] == "train"]
        write_json(root / "dataset_stats.json", stats)
    index = {"schema": NAVIGATION_CONTRACT, "episodes": episodes, "metadata_pinned": True,
             "train_stats_available": bool(arrays["state"]),
             "teacher_successes": sum(e["success"] for e in episodes), "teacher_episodes": len(episodes),
             "evidence": "teacher_dataset_diagnostic; not learned-model acceptance"}
    write_json(root / "index.json", index)
    return index


class FastWAMNavigationDataset:
    """Hydra target compatible with pinned FastWAM Trainer's sample contract."""

    def __init__(self, root, split="train", *, horizon=32, video_stride=4,
                 sample_stride=4, text_embedding_cache_dir=None, context_len=128,
                 require_visual_search=True, pretrained_norm_stats=None,
                 augmentation_roots=None, correction_repeats=1,
                 retarget_roots=None, retarget_repeats=1,
                 recovery_roots=None, recovery_repeats=1, recovery_sampling=None,
                 qualified_teacher_roots=None, qualified_teacher_sha256=None,
                 sample_source_repeats=None):
        self.root = Path(root)
        self.split = split
        self.horizon, self.video_stride = int(horizon), int(video_stride)
        if split not in ("train", "validation", "test") or horizon < 1 or video_stride < 1 or sample_stride < 1:
            raise ValueError("invalid split/horizon/stride")
        if horizon % video_stride or (horizon // video_stride) % 4:
            raise ValueError("FastWAM video must contain 1+4k frames with aligned actions")
        index = json.loads((self.root / "index.json").read_text())
        if index["schema"] != NAVIGATION_CONTRACT:
            raise ValueError("wrong dataset contract")
        # Legacy immutable releases remain readable, but never acquire a claim
        # of pinned language retroactively. New releases require every digest.
        self.metadata_pinned = index.get("metadata_pinned", False)
        for entry in index["episodes"]:
            read_indexed_metadata(self.root, entry, require_digest=self.metadata_pinned)
        stats_path = self.root / "dataset_stats.json"
        if pretrained_norm_stats is not None:
            # Upstream build_datasets forwards this path to its validation
            # dataset. Only the frozen training artifact is acceptable here.
            if hashlib.sha256(Path(pretrained_norm_stats).read_bytes()).digest() != hashlib.sha256(stats_path.read_bytes()).digest():
                raise ValueError("validation normalization differs from the frozen train statistics")
        self.stats = json.loads(stats_path.read_text())
        if self.stats["normalization_split"] != "train" or self.stats["schema"] != NAVIGATION_CONTRACT:
            raise ValueError("normalization must use matching train-only statistics")
        self.entries = [e for e in index["episodes"] if e["eligible"] and e["split"] == split
                        and (e.get("search_visibility_gated", False) or not require_visual_search)]
        self.samples = [(entry, start) for entry in self.entries for start in range(0, entry["frames"] - 1, sample_stride)]
        self.base_samples = len(self.samples)
        self.correction_samples = 0
        self.retarget_samples = 0
        self.recovery_samples = 0
        self.recovery_student_anchor_samples = 0
        self.recovery_teacher_continuation_samples = 0
        self.qualified_teacher_samples = 0
        teacher_roots, teacher_hashes = qualified_teacher_roots or [], qualified_teacher_sha256 or []
        if (isinstance(teacher_roots, (str, Path)) or isinstance(teacher_hashes, str)
                or len(teacher_roots) != len(teacher_hashes)):
            raise ValueError("additional teacher releases require explicit roots and matching index hashes")
        if teacher_roots:
            from .navigation_teacher_data import load_qualified_teacher_release

            if not require_visual_search:
                raise ValueError("qualified teacher episodes require visibility-gated observations")
            identities = {(e['task'], e['seed']) for e in index['episodes']}
            for teacher_root, digest in zip(teacher_roots, teacher_hashes):
                released = load_qualified_teacher_release(teacher_root, self.root, digest)
                keys = {(e['task'], e['seed']) for e in released}
                if identities.intersection(keys):
                    raise ValueError("duplicate full teacher episode across releases")
                identities.update(keys)
                for entry in released:
                    if entry['split'] != split:
                        continue
                    entry = dict(entry, _root=str(Path(teacher_root).resolve()), _qualified_teacher_episode=True)
                    self.entries.append(entry)
                    starts = [(entry, start) for start in range(0, entry['frames'] - 1, sample_stride)]
                    self.samples.extend(starts)
                    self.qualified_teacher_samples += len(starts)
        if isinstance(correction_repeats, bool) or int(correction_repeats) != correction_repeats or correction_repeats < 1:
            raise ValueError("correction_repeats must be a positive integer")
        if augmentation_roots:
            from .navigation_augmentation import load_augmentation

            if split != "train" or not require_visual_search:
                raise ValueError("corrections may only augment qualified training data")
            if isinstance(augmentation_roots, (str, Path)):
                raise ValueError("augmentation_roots must be an explicit list of releases")
            identities = set()
            for correction_root in augmentation_roots:
                for entry in load_augmentation(correction_root, self.root):
                    identity = (entry["task"], entry["seed"], entry["source_tick"], entry["student_checkpoint_sha256"])
                    if identity in identities:
                        raise ValueError("duplicate correction across releases")
                    identities.add(identity)
                    entry = dict(entry, _root=str(Path(correction_root).resolve()))
                    self.entries.append(entry)
                    self.samples.extend([(entry, 0)] * int(correction_repeats))
                    self.correction_samples += int(correction_repeats)
        if isinstance(retarget_repeats, bool) or int(retarget_repeats) != retarget_repeats or retarget_repeats < 1:
            raise ValueError("retarget_repeats must be a positive integer")
        if retarget_roots:
            from .navigation_retarget_data import identity, load_retargeted

            if split != "train" or not require_visual_search:
                raise ValueError("retarget demonstrations may only augment qualified training data")
            if isinstance(retarget_roots, (str, Path)):
                raise ValueError("retarget_roots must be an explicit list of releases")
            identities = set()
            for retarget_root in retarget_roots:
                for entry in load_retargeted(retarget_root, self.root):
                    key = identity(entry)
                    if key in identities:
                        raise ValueError("duplicate retarget source/task identity across releases")
                    identities.add(key)
                    entry = dict(entry, _root=str(Path(retarget_root).resolve()))
                    self.entries.append(entry)
                    self.samples.extend([(entry, 0)] * int(retarget_repeats))
                    self.retarget_samples += int(retarget_repeats)
        if isinstance(recovery_repeats, bool) or int(recovery_repeats) != recovery_repeats or recovery_repeats < 1:
            raise ValueError("recovery_repeats must be a positive integer")
        anchor_repeats = continuation_repeats = int(recovery_repeats)
        if recovery_sampling is not None:
            if (not isinstance(recovery_sampling, Mapping)
                    or set(recovery_sampling) != {"student_anchor_repeats", "teacher_continuation_repeats"}):
                raise ValueError("recovery_sampling requires explicit student-anchor and teacher-continuation repeats")
            if recovery_repeats != 1:
                raise ValueError("recovery_sampling cannot be combined with non-default recovery_repeats")
            if not recovery_roots:
                raise ValueError("recovery_sampling requires recovery_roots")
            anchor_repeats = recovery_sampling["student_anchor_repeats"]
            continuation_repeats = recovery_sampling["teacher_continuation_repeats"]
            if (type(anchor_repeats) is not int or anchor_repeats < 1
                    or type(continuation_repeats) is not int or continuation_repeats < 0):
                raise ValueError("recovery_sampling repeats must be integers: anchor >= 1, continuation >= 0")
        self.recovery_sampling = {"student_anchor_repeats": anchor_repeats,
                                  "teacher_continuation_repeats": continuation_repeats}
        if recovery_roots:
            from .navigation_recovery_data import identity, load_recoveries

            if split != "train" or not require_visual_search:
                raise ValueError("teacher recoveries may only augment qualified training data")
            if isinstance(recovery_roots, (str, Path)):
                raise ValueError("recovery_roots must be an explicit list of releases")
            identities = set()
            for recovery_root in recovery_roots:
                for entry in load_recoveries(recovery_root, self.root,
                                             qualified_teacher_roots=teacher_roots,
                                             qualified_teacher_sha256=teacher_hashes):
                    key = identity(entry)
                    if key in identities:
                        raise ValueError("duplicate recovery source across releases")
                    identities.add(key)
                    entry = dict(entry, _root=str(Path(recovery_root).resolve()))
                    self.entries.append(entry)
                    if anchor_repeats == continuation_repeats:
                        # Preserve legacy sample order, including explicit equal
                        # weights, so existing seeds produce the same batches.
                        self.samples.extend([(entry, start) for start in entry["sample_starts"]] * anchor_repeats)
                    else:
                        self.samples.extend([(entry, start) for start in entry["actual_student_anchor_starts"]] * anchor_repeats)
                        self.samples.extend([(entry, start) for start in entry["teacher_continuation_starts"]] * continuation_repeats)
                    anchor_count = len(entry["actual_student_anchor_starts"]) * anchor_repeats
                    continuation_count = len(entry["teacher_continuation_starts"]) * continuation_repeats
                    self.recovery_samples += anchor_count + continuation_count
                    self.recovery_student_anchor_samples += anchor_count
                    self.recovery_teacher_continuation_samples += continuation_count
        self.source_sample_exposure = None
        if sample_source_repeats is not None:
            if augmentation_roots or retarget_roots:
                raise ValueError("source repetition currently supports full teachers and recoveries")
            from .navigation_source_sampling import repeat_samples
            self.samples, self.source_sample_exposure = repeat_samples(
                self.samples, self.root, split, sample_source_repeats)
            self.base_samples = sum(e.get('_root') is None for e, _ in self.samples)
            self.qualified_teacher_samples = sum(bool(e.get('_qualified_teacher_episode')) for e, _ in self.samples)
            recovery_samples = [(e, start) for e, start in self.samples
                                if e.get('_root') is not None and not e.get('_qualified_teacher_episode')]
            self.recovery_samples = len(recovery_samples)
            self.recovery_student_anchor_samples = sum(start in e['actual_student_anchor_starts']
                                                      for e, start in recovery_samples)
            self.recovery_teacher_continuation_samples = self.recovery_samples - self.recovery_student_anchor_samples
        if not self.samples:
            raise ValueError(f"no eligible {split} samples")
        self.cache = None
        self.cache_path = None
        self.text_embedding_cache_dir = Path(text_embedding_cache_dir) if text_embedding_cache_dir else None
        self.context_len = int(context_len)

    def __len__(self):
        return len(self.samples)

    @property
    def lerobot_dataset(self):
        # Upstream Trainer uses only its processor for denormalized validation
        # metrics. Expose that API while keeping our native episode storage.
        return self

    @property
    def processor(self):
        if not hasattr(self, "_processor"):
            self._processor = reference_processor(self.stats, self.horizon)
        return self._processor

    def _normalize(self, values, key):
        stats = self.stats[key]["default"]
        low, high = np.asarray(stats["global_min"], dtype=np.float32), np.asarray(stats["global_max"], dtype=np.float32)
        width = high - low
        constant = width < 1e-4
        scale = 2 / np.where(constant, 2, width)
        offset = np.where(constant, -low, -1 - scale * low)
        return np.clip(values * scale + offset, -5, 5).astype(np.float32)

    def __getitem__(self, index):
        import torch
        entry, start = self.samples[index]
        path = Path(entry.get("_root", self.root)) / entry["path"]
        if self.cache_path != path:
            if hashlib.sha256((path / "navigation.npz").read_bytes()).hexdigest() != entry["sha256"]:
                raise ValueError("navigation dataset changed after indexing")
            with np.load(path / "navigation.npz", allow_pickle=False) as data:
                self.cache = {key: data[key].copy() for key in ("rgb", "proprio", "action", "ticks")}
            self.cache_path = path
        data = self.cache
        n = len(data["action"])
        action_indices = start + np.arange(self.horizon)
        video_indices = start + np.arange(0, self.horizon + 1, self.video_stride)
        rgb = data["rgb"][np.minimum(video_indices, n)]
        video = torch.from_numpy(rgb.copy()).permute(3, 0, 1, 2).float().div(127.5).sub(1)
        meta = read_indexed_metadata(entry.get("_root", self.root), entry,
                                     require_digest=self.metadata_pinned and "_root" not in entry)
        prompt = PROMPT.format(task=meta["instruction"])
        sample = {
            "video": video,
            "action": torch.from_numpy(self._normalize(data["action"][np.minimum(action_indices, n - 1)], "action")),
            "proprio": torch.from_numpy(self._normalize(data["proprio"][np.minimum(action_indices, n)], "state")),
            "prompt": prompt, "action_is_pad": torch.from_numpy(action_indices >= n),
            "image_is_pad": torch.from_numpy(video_indices > n),
            "proprio_is_pad": torch.from_numpy(action_indices > n),
        }
        if self.text_embedding_cache_dir:
            digest = hashlib.sha256(prompt.encode()).hexdigest()
            cache = self.text_embedding_cache_dir / f"{digest}.t5_len{self.context_len}.wan22ti2v5b.pt"
            text = torch.load(cache, map_location="cpu", weights_only=True)
            context, mask = text["context"].clone(), text["mask"].bool()
            if context.shape != (self.context_len, 4096) or mask.shape != (self.context_len,):
                raise ValueError("T5 cache dimensions differ from FastWAM")
            context[~mask] = 0
            sample["context"], sample["context_mask"] = context, torch.ones_like(mask)
        return sample


def main():
    import argparse
    from .env import MicroDuckEnv

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--scene", default="room")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--tasks", nargs="+", default=["find_red_ball", "find_blue_ball", "find_yellow_ball"])
    parser.add_argument("--teacher", choices=("privileged", "visual_search"), default="visual_search")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with MicroDuckEnv(profile="official_bam", scene=args.scene, image_size=256) as env:
        for seed in args.seeds:
            for task in args.tasks:
                path = args.output / f"{task}-seed{seed}"
                if (path / "episode.json").is_file():
                    continue
                report = collect_episode(env, path, task=task, seed=seed, teacher_kind=args.teacher)
                print(json.dumps({"task": task, "seed": seed, "success": report["result"]["success"],
                                  "eligible": report["eligible_training_demonstration"]}), flush=True)
    print(json.dumps(build_index(args.output)), flush=True)


if __name__ == "__main__":
    main()
