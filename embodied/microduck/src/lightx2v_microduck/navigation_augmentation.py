"""Audited training-only corrections; preserve the qualified teacher release.

Correction releases contain only RGB, proprio48, actions and simulator ticks.
Teacher truth remains in the source diagnostics. Frozen statistics and held-out
layouts cannot change when corrections are added to the original demonstrations.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from .navigation import ACTION_HIGH, ACTION_LOW, NAVIGATION_CONTRACT


PUBLIC_KEYS = {"rgb", "proprio", "action", "ticks"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def contained(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("correction path leaves its artifact")
    return path


def qualified_base(root, protocol):
    root = Path(root)
    index = json.loads((root / "index.json").read_text())
    qualification = json.loads((root / "qualification_report.json").read_text())
    if (qualification.get("teacher_qualified") is not True or qualification["scene"] != "room_clear"
            or qualification["index_sha256"] != sha256(root / "index.json")
            or qualification["stats_sha256"] != sha256(root / "dataset_stats.json")
            or protocol["seed_assignment_sha256"] != qualification["index_sha256"]):
        raise ValueError("base qualification, statistics or reserved protocol changed")
    return index, qualification


def validate_tensors(data, count, source_tick, *, public_only=False, allow_empty_diagnostic=False):
    # A branch can finish within its first five motor ticks, leaving no complete
    # navigation label. Inspect that diagnostic, but never release it as training.
    empty_diagnostic = count == 0 and allow_empty_diagnostic and not public_only
    if (public_only and set(data) != PUBLIC_KEYS) or (count < 1 and not empty_diagnostic):
        raise ValueError("training corrections must contain only public tensors and real actions")
    if (data["action"].shape != (count, 5) or data["proprio"].shape != (count + 1, 48)
            or data["rgb"].shape != (count + 1, 256, 256, 3) or data["rgb"].dtype != np.uint8
            or not np.array_equal(data["ticks"], source_tick + np.arange(count + 1) * 5)
            or not np.isfinite(data["action"]).all() or not np.isfinite(data["proprio"]).all()
            or np.any(data["action"] < ACTION_LOW) or np.any(data["action"] > ACTION_HIGH)):
        raise ValueError("invalid navigation correction tensors or clock")


def audit_collection(source, base_dataset, protocol_path):
    from .wam_data import split_for_seed

    source, base_dataset, protocol_path = [Path(p).resolve() for p in (source, base_dataset, protocol_path)]
    report = json.loads((source / "report.json").read_text())
    protocol = json.loads(protocol_path.read_text())
    index, qualification = qualified_base(base_dataset, protocol)
    training = {(e["task"], e["seed"]): e for e in index["episodes"] if e["split"] == "train" and e["eligible"]}
    reserved = set(protocol["development_seeds"]) | set(protocol["independent_final_seeds"])
    if (report["status"] != "complete" or report["protocol_sha256"] != sha256(protocol_path)
            or report["stats_sha256"] != qualification["stats_sha256"]
            or report["qualification_sha256"] != sha256(base_dataset / "qualification_report.json")):
        raise ValueError("incomplete collection or mismatched frozen data/protocol")
    accepted, rejected, identities, rollouts = [], [], set(), set()
    for row in report["rollouts"]:
        seed, task = row["seed"], row["task"]
        if (task, seed) not in training or seed in reserved or split_for_seed(seed) != "train":
            raise ValueError("held-out or unqualified layout entered collection")
        if (task, seed) in rollouts:
            raise ValueError("duplicate student rollout")
        rollouts.add((task, seed))
        folder = contained(source, f"{task}-seed{seed}")
        rollout = json.loads((folder / "report.json").read_text())
        if (rollout["status"] != "complete" or (rollout["task"], rollout["seed"], rollout["split"]) != (task, seed, "train")
                or rollout["policy"]["artifact_sha256"] != report["checkpoint_sha256"]
                or rollout["policy"]["uses_privileged_state"] is not False):
            raise ValueError("student checkpoint/input provenance changed")
        base_entry = training[(task, seed)]
        instruction = json.loads((contained(base_dataset, base_entry["path"]) / "episode.json").read_text())["instruction"]
        with np.load(folder / "recording.npz", allow_pickle=False) as record:
            camera_ticks, rgb = record["camera_ticks"], record["rgb"]
            for entry in rollout["branches"]:
                path = contained(folder, entry["path"])
                branch = json.loads((path / "report.json").read_text())
                checksum = sha256(path / "navigation.npz")
                if checksum != entry["navigation_sha256"] or checksum != branch["navigation_sha256"]:
                    raise ValueError("branch data checksum changed")
                identity = (task, seed, branch["source_tick"])
                if identity in identities:
                    raise ValueError("duplicate student-state label")
                identities.add(identity)
                if ((branch["task"], branch["seed"], branch["split"]) != (task, seed, "train")
                        or branch["instruction"] != instruction
                        or branch["student_model"]["artifact_sha256"] != report["checkpoint_sha256"]):
                    raise ValueError("branch belongs to another task/layout/split/checkpoint")
                with np.load(path / "navigation.npz", allow_pickle=False) as data:
                    count = branch["label_decisions"]
                    validate_tensors(data, count, branch["source_tick"],
                        allow_empty_diagnostic=(count == 0 and not branch["eligible_training_supervision"]
                                                and not entry["eligible_training_supervision"]))
                    source_index = np.flatnonzero(camera_ticks == branch["source_tick"])
                    if len(source_index) != 1 or not np.array_equal(data["rgb"][0], rgb[source_index[0]]):
                        raise ValueError("branch did not start at the student's actual camera observation")
                    if not np.array_equal(data["proprio"][0], data["continuation_state61"][0, :48]):
                        raise ValueError("branch state differs from its first control state")
                result = branch["result"]
                unsafe = any(d.get("planner_phase", d.get("phase")) in ("recover", "no_path") for d in branch["teacher_diagnostics"])
                eligible = bool(result["success"] and not unsafe and not result["collision"] and not result["ever_fallen"]
                                and result["recovery_events"] == 0 and count > 0)
                if eligible != branch["eligible_training_supervision"] or eligible != entry["eligible_training_supervision"]:
                    raise ValueError("claimed eligibility does not match strict teacher result")
                item = {"path": str(path.relative_to(source)), "task": task, "seed": seed,
                        "source_tick": branch["source_tick"], "label_decisions": count, "sha256": checksum}
                if eligible:
                    accepted.append(item)
                else:
                    item["failure"] = {k: result.get(k) for k in ("success", "collision", "ever_fallen", "recovery_events", "tick")}
                    if count == 0:
                        item["failure"]["reason"] = "no_complete_navigation_decision"
                    rejected.append(item)
    return {"status": "complete", "schema": NAVIGATION_CONTRACT, "accepted": accepted, "rejected": rejected,
            "accepted_count": len(accepted), "rejected_count": len(rejected),
            "accepted_label_decisions": sum(item["label_decisions"] for item in accepted),
            "source_report_sha256": sha256(source / "report.json"),
            "frozen_stats_sha256": report["stats_sha256"], "checkpoint_sha256": report["checkpoint_sha256"],
            "qualified_training_layouts_only": True, "first_frame_matches_student_recording": True,
            "training_tensor_keys": sorted(PUBLIC_KEYS),
            "teacher_trace_keys_excluded_from_training": ["continuation_state61", "continuation_raw_action", "continuation_action"],
            "ready_for_aggregation": bool(accepted), "weights_updated": False,
            "scope": "data integrity only; no new teacher qualification or learned capability acceptance"}


def export_augmentation(source, base_dataset, protocol_path, output):
    from .wam_data import write_json

    source, base_dataset, protocol_path, output = [Path(p).resolve() for p in (source, base_dataset, protocol_path, output)]
    if output.exists():
        raise FileExistsError("correction releases are immutable")
    audit = audit_collection(source, base_dataset, protocol_path)
    if not audit["ready_for_aggregation"]:
        raise ValueError("no qualified correction labels")
    output.mkdir(parents=True)
    entries = []
    for entry in audit["accepted"]:
        original = source / entry["path"]
        target = output / entry["path"]
        target.mkdir(parents=True)
        branch = json.loads((original / "report.json").read_text())
        with np.load(original / "navigation.npz", allow_pickle=False) as data:
            np.savez_compressed(target / "navigation.npz", **{key: data[key] for key in sorted(PUBLIC_KEYS)})
        meta = {"schema": NAVIGATION_CONTRACT, "task": entry["task"], "seed": entry["seed"], "split": "train",
                "instruction": branch["instruction"], "source_tick": entry["source_tick"],
                "student_checkpoint_sha256": audit["checkpoint_sha256"], "source_navigation_sha256": entry["sha256"],
                "eligible_training_supervision": True, "search_visibility_gated": True,
                "sampling": "one clip starts at the actual student state; teacher futures supply video/action labels"}
        write_json(target / "episode.json", meta)
        entries.append({"path": entry["path"], "task": entry["task"], "seed": entry["seed"], "split": "train",
                        "eligible": True, "search_visibility_gated": True, "frames": entry["label_decisions"] + 1,
                        "source_tick": entry["source_tick"], "student_checkpoint_sha256": audit["checkpoint_sha256"],
                        "sample_starts": [0], "sha256": sha256(target / "navigation.npz"),
                        "metadata_sha256": sha256(target / "episode.json")})
    shutil.copyfile(base_dataset / "dataset_stats.json", output / "dataset_stats.json")
    shutil.copyfile(protocol_path, output / "protocol.json")
    write_json(output / "source_audit.json", audit)
    index = {"schema": NAVIGATION_CONTRACT, "kind": "student_state_corrections", "episodes": entries,
             "base_index_sha256": sha256(base_dataset / "index.json"),
             "base_qualification_sha256": sha256(base_dataset / "qualification_report.json"),
             "stats_sha256": sha256(output / "dataset_stats.json"), "protocol_sha256": sha256(output / "protocol.json"),
             "audit_sha256": sha256(output / "source_audit.json"), "source_report_sha256": audit["source_report_sha256"],
             "scope": "train-only augmentation; base qualification and normalization unchanged"}
    write_json(output / "index.json", index)
    receipt = {"status": "complete", "index_sha256": sha256(output / "index.json"),
               "corrections": len(entries), "label_decisions": audit["accepted_label_decisions"],
               "public_tensors_only": True, "weights_updated": False, "capability_accepted": False}
    write_json(output / "release.json", receipt)
    load_augmentation(output, base_dataset)
    return receipt


def load_augmentation(root, base_dataset):
    from .wam_data import split_for_seed

    root, base_dataset = Path(root), Path(base_dataset)
    receipt = json.loads((root / "release.json").read_text())
    index = json.loads((root / "index.json").read_text())
    protocol = json.loads((root / "protocol.json").read_text())
    base, _ = qualified_base(base_dataset, protocol)
    if (receipt["status"] != "complete" or receipt["index_sha256"] != sha256(root / "index.json")
            or index["schema"] != NAVIGATION_CONTRACT or index["kind"] != "student_state_corrections"
            or index["base_index_sha256"] != sha256(base_dataset / "index.json")
            or index["base_qualification_sha256"] != sha256(base_dataset / "qualification_report.json")
            or index["stats_sha256"] != sha256(root / "dataset_stats.json")
            or index["stats_sha256"] != sha256(base_dataset / "dataset_stats.json")
            or index["protocol_sha256"] != sha256(root / "protocol.json")
            or index["audit_sha256"] != sha256(root / "source_audit.json")):
        raise ValueError("correction release or frozen base artifact changed")
    audit = json.loads((root / "source_audit.json").read_text())
    approved = {(e["task"], e["seed"], e["source_tick"]): e for e in audit["accepted"]}
    training = {(e["task"], e["seed"]): e for e in base["episodes"] if e["eligible"] and e["split"] == "train"}
    reserved = set(protocol["development_seeds"]) | set(protocol["independent_final_seeds"])
    identities = set()
    for entry in index["episodes"]:
        identity = (entry["task"], entry["seed"], entry["source_tick"])
        if (identity in identities or identity not in approved or entry["split"] != "train"
                or (entry["task"], entry["seed"]) not in training or entry["seed"] in reserved
                or split_for_seed(entry["seed"]) != "train" or not entry["eligible"]
                or not entry["search_visibility_gated"] or entry["sample_starts"] != [0]
                or entry["student_checkpoint_sha256"] != audit["checkpoint_sha256"]):
            raise ValueError("duplicate, held-out or unqualified correction")
        identities.add(identity)
        path = contained(root, entry["path"])
        if sha256(path / "navigation.npz") != entry["sha256"] or sha256(path / "episode.json") != entry["metadata_sha256"]:
            raise ValueError("correction episode changed after export")
        meta = json.loads((path / "episode.json").read_text())
        parent = contained(base_dataset, training[(entry["task"], entry["seed"])]["path"])
        if (meta["instruction"] != json.loads((parent / "episode.json").read_text())["instruction"]
                or meta["source_navigation_sha256"] != approved[identity]["sha256"]):
            raise ValueError("correction instruction or source label changed")
        with np.load(path / "navigation.npz", allow_pickle=False) as data:
            validate_tensors(data, entry["frames"] - 1, entry["source_tick"], public_only=True)
    return index["episodes"]
