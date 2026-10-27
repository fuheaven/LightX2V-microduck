"""Sensor-only training tensors and separately stored teacher/evaluator records."""

import json
from pathlib import Path
import numpy as np
from .schema import OBSERVATION, ACTION, file_hash, write_json, CATEGORIES, COLORS

HISTORY = (300, 150, 50, 0)


def sensor_state(obs):
    if obs.rgb is None or obs.tof is None or not obs.tof_metadata.get("available"):
        raise ValueError("RGB and official ToF are required; missing sensors are not zero-filled")
    mask = np.asarray(obs.tof_metadata["statuses"]) == 5
    if mask.shape != (8, 8) or obs.tof_metadata.get("units") != "m":
        raise ValueError("ToF must use the pinned 8x8 metric contract")
    state = np.concatenate((obs.state[:48], obs.tof.ravel(), mask.astype(np.float32).ravel())).astype(np.float32)
    if state.shape != (176,) or not np.isfinite(state).all():
        raise ValueError("invalid observation176")
    return state


def historical_indices(ticks, tick):
    return [max(0, int(np.searchsorted(ticks, tick - offset, side="right")) - 1) for offset in HISTORY]


def grid(frames):
    return np.concatenate((np.concatenate(frames[:2], axis=1), np.concatenate(frames[2:], axis=1)), axis=0)


def audit_dataset(root):
    root = Path(root)
    index = json.loads((root / "index.json").read_text())
    if index["observation"] != OBSERVATION or index["action"] != ACTION:
        raise ValueError("dataset contracts differ")
    keys, seen = {k: {} for k in ("layout_id", "instance_id", "instruction_family")}, set()
    train_states, train_actions = [], []
    for row in index["episodes"]:
        if row["id"] in seen:
            raise ValueError("duplicate episode")
        seen.add(row["id"])
        for key, splits in keys.items():
            if row[key] in splits and splits[row[key]] != row["split"]:
                raise ValueError(f"cross-split leakage: {key}")
            splits[row[key]] = row["split"]
        path = root / row["path"]
        if root.resolve() not in path.resolve().parents or file_hash(path) != row["sha256"]:
            raise ValueError("episode checksum/path mismatch")
        with np.load(path, allow_pickle=False) as d:
            n = len(d["action"])
            if (
                d["state"].shape != (n, 176)
                or d["rgb"].shape != (n, 256, 256, 3)
                or d["action"].shape != (n, 5)
                or d["rgb"].dtype != np.uint8
                or not np.isfinite(d["state"]).all()
                or not np.isfinite(d["action"]).all()
                or np.any(np.diff(d["tick"]) != 5)
                or np.any(d["tof_tick"] > d["tick"])
                or any(k in d for k in ("world", "target_position", "success", "teacher_command"))
            ):
                raise ValueError("invalid/aligned sensor tensor or privileged data in student stream")
            if row["eligible"] and row["split"] == "train":
                train_states.append(d["state"].copy())
                train_actions.append(d["action"].copy())
    if not train_states:
        raise ValueError("no qualified training episodes")
    stats = {
        "schema": ACTION,
        "observation": OBSERVATION,
        "normalization_split": "train",
        "index_sha256": file_hash(root / "index.json"),
        "source_episodes": [e["sha256"] for e in index["episodes"] if e["eligible"] and e["split"] == "train"],
    }
    for key, values in (("state", train_states), ("action", train_actions)):
        a = np.concatenate(values)
        stats[key] = {"default": {"global_min": a.min(0).tolist(), "global_max": a.max(0).tolist(), "global_mean": a.mean(0).tolist(), "global_std": a.std(0).tolist()}}
    write_json(root / "dataset_stats.json", stats)
    result = {
        "passed": True,
        "index_sha256": file_hash(root / "index.json"),
        "stats_sha256": file_hash(root / "dataset_stats.json"),
        "episodes": len(index["episodes"]),
        "eligible": sum(e["eligible"] for e in index["episodes"]),
        "splits": {s: sum(e["split"] == s and e["eligible"] for e in index["episodes"]) for s in ("train", "validation", "test")},
        "formal_acceptance": False,
    }
    write_json(root / "audit.json", result)
    return result


def affine(stats, key):
    s = stats[key]["default"]
    lo, hi = np.array(s["global_min"], np.float32), np.array(s["global_max"], np.float32)
    diff = hi - lo
    scale = 2 / np.where(diff < 1e-4, 2, diff)
    offset = np.where(diff < 1e-4, -lo, -1 - lo * scale)
    return scale, offset


class SkillDataset:
    def __init__(self, root, split="train", *, stride=4, include_video=False):
        self.root = Path(root)
        self.include_video = include_video
        self.index = json.loads((self.root / "index.json").read_text())
        self.stats = json.loads((self.root / "dataset_stats.json").read_text())
        audit = json.loads((self.root / "audit.json").read_text())
        if (
            self.stats["normalization_split"] != "train"
            or self.stats["observation"] != OBSERVATION
            or audit["index_sha256"] != file_hash(self.root / "index.json")
            or audit["stats_sha256"] != file_hash(self.root / "dataset_stats.json")
        ):
            raise ValueError("dataset audit/statistics binding mismatch")
        self.rows = [r for r in self.index["episodes"] if r["eligible"] and r["split"] == split]
        for row in self.rows:
            if file_hash(self.root / row["path"]) != row["sha256"]:
                raise ValueError("dataset episode changed")
        self.samples = [(r, t) for r in self.rows for t in range(0, r["frames"], stride)]
        if not self.samples:
            raise ValueError(f"no qualified {split} samples")
        self.cache_key = None
        self.cache = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        row, t = self.samples[index]
        if row["path"] != self.cache_key:
            with np.load(self.root / row["path"], allow_pickle=False) as data:
                self.cache = {k: data[k].copy() for k in data.files}
            self.cache_key = row["path"]
        d = self.cache
        n = len(d["action"])
        indices = historical_indices(d["tick"], d["tick"][t])
        future = np.minimum(t + np.arange(32), n - 1)
        item = {
            "rgb": d["rgb"][indices].copy(),
            "state": d["state"][indices].copy(),
            "action": d["action"][future].copy(),
            "valid": (t + np.arange(32) < n),
            "category": CATEGORIES.index(row["category"]),
            "color": COLORS.index(row["color"]),
            "instruction": row["instruction"],
        }
        if self.include_video:
            item.update(video=np.stack([grid(d["rgb"][historical_indices(d["tick"], d["tick"][min(t + i, n - 1)])]) for i in range(0, 33, 4)]), video_valid=t + np.arange(0, 33, 4) < n)
        return item
