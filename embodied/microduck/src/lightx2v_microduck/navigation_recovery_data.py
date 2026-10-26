"""Portable full teacher recoveries from audited, recorded training student states.

Only the first state was visited by the student. Later sample starts are
explicitly teacher continuations. The student's recorded first RGB is retained
verbatim; physically replayed future frames remain simulator observations.
Frozen base normalization and all held-out layouts remain unchanged.
"""

import json
from pathlib import Path
import shutil

import numpy as np

from .navigation import NAVIGATION_CONTRACT
from .navigation_augmentation import PUBLIC_KEYS, contained, qualified_base, sha256, validate_tensors
from .wam_data import split_for_seed, write_json


KIND = "teacher_recovery_sequences"


def identity(entry):
    return entry["task"], entry["seed"], entry["source_tick"], entry["student_checkpoint_sha256"]


def _training_sources(base_dataset, base, teacher_root=None, teacher_sha256=None):
    """Keep normalization anchored to base while explicitly qualifying new layouts."""
    if (teacher_root is None) != (teacher_sha256 is None):
        raise ValueError("recovery teacher source requires both root and index hash")
    sources = [(Path(base_dataset), base["episodes"])]
    if teacher_root is not None:
        from .navigation_teacher_data import load_qualified_teacher_release

        sources.append((Path(teacher_root), load_qualified_teacher_release(
            teacher_root, base_dataset, teacher_sha256)))
    seen, training = set(), {}
    for root, entries in sources:
        for entry in entries:
            key = entry["task"], entry["seed"]
            if key in seen:
                raise ValueError("duplicate recovery source layout across datasets")
            seen.add(key)
            if entry["split"] == "train" and entry["eligible"]:
                training[key] = dict(entry, _source_root=root)
    return training


def export_recoveries(source, source_protocol, assessment, expected_assessment_sha256, base_dataset,
                      evaluation_protocol, output, *, teacher="aware", sample_stride=4,
                      qualified_teacher_root=None, qualified_teacher_sha256=None):
    source, source_protocol, assessment, base_dataset, evaluation_protocol, output = map(
        Path, (source, source_protocol, assessment, base_dataset, evaluation_protocol, output))
    if output.exists():
        raise FileExistsError("recovery releases are immutable")
    if teacher not in ("original", "aware", "qualified") or type(sample_stride) is not int or sample_stride < 1:
        raise ValueError("explicit paired teacher and positive sample stride required")
    if sha256(assessment) != expected_assessment_sha256:
        raise ValueError("recovery assessment changed")
    reviewed = json.loads(assessment.read_text())
    collection_protocol = json.loads(source_protocol.read_text())
    source_report = json.loads((source / "report.json").read_text())
    evaluation = json.loads(evaluation_protocol.read_text())
    base, qualification = qualified_base(base_dataset, evaluation)
    if (reviewed["status"] != "complete" or source_report["status"] != "complete"
            or reviewed["source_report_sha256"] != sha256(source / "report.json")
            or reviewed["all_selected_student_physical_replays_exact"] is not True
            or reviewed["paired_first_rgb_and_proprio_exact"] is not True
            or reviewed["protocol_sha256"] != source_report["protocol_sha256"]
            or reviewed["protocol_sha256"] != sha256(source_protocol)
            or collection_protocol["evaluation_protocol_sha256"] != sha256(evaluation_protocol)
            or source_report["original_task_deadline_preserved"] is not True):
        raise ValueError("complete exact physical replay assessment required")
    binding = None
    if "qualified_teacher_release" in collection_protocol:
        from .navigation_replay_binding import load_replay_binding
        binding = load_replay_binding(collection_protocol, verify_runtime=False)
        if (teacher != "qualified" or qualified_teacher_sha256 != binding["qualified_teacher_index_sha256"]
                or reviewed.get("qualified_replay_binding") != binding
                or source_report.get("qualified_replay_binding") != binding):
            raise ValueError("only the bound qualified teacher may supply additional-scene recovery labels")
    elif teacher == "qualified":
        raise ValueError("qualified teacher labels require a qualified replay protocol")
    training = _training_sources(base_dataset, base, qualified_teacher_root, qualified_teacher_sha256)
    forbidden = set(evaluation["development_seeds"]) | set(evaluation["independent_final_seeds"])
    rows = {(r["task"], r["seed"]): r for r in source_report["cases"] if r["branch_tick"] is not None}
    cases = reviewed["cases"]
    pinned_sources = {(e["task"], e["seed"]): e for e in collection_protocol["sources"]}
    if (len(cases) != len(rows) or {(r["task"], r["seed"]) for r in cases} != set(rows)
            or len(cases) != reviewed["paired_recoveries"]):
        raise ValueError("recovery cohort is incomplete or duplicated")
    accepted, rejected, payloads = [], [], []
    for case in cases:
        task, seed = case["task"], case["seed"]
        if (task, seed) not in training or seed in forbidden or split_for_seed(seed) != "train":
            raise ValueError("held-out or unqualified recovery source")
        row = rows[task, seed]
        selected = [b for b in row["branches"] if b["teacher"] == teacher]
        if len(selected) != 1:
            raise ValueError("paired teacher branch is missing or duplicated")
        selected = selected[0]
        folder = contained(source, f"{task}-seed{seed}/{teacher}")
        checksums = case["files"][teacher]
        if (sha256(folder / "report.json") != checksums["report_sha256"]
                or sha256(folder / "navigation.npz") != checksums["navigation_sha256"]):
            raise ValueError("reviewed recovery artifacts changed")
        branch = json.loads((folder / "report.json").read_text())
        if binding is not None and (
                branch.get("teacher_kind") != binding["teacher_kinds"][teacher]
                or branch.get("qualified_teacher_index_sha256") != binding["qualified_teacher_index_sha256"]):
            raise ValueError("recovery branch teacher identity differs from its qualified source")
        if ((branch["task"], branch["seed"], branch["split"], branch["source_tick"]) !=
                (task, seed, "train", case["source_tick"])
                or branch["student_model"]["artifact_sha256"] != source_report["source_checkpoint_sha256"]
                or branch["student_model"]["uses_privileged_state"] is not False
                or row["original_physical_replay_max_error"] != 0):
            raise ValueError("recovery source identity or physical replay differs")
        actual_source = Path(branch["source_rollout"])
        pinned = pinned_sources[task, seed]
        if (actual_source.resolve() != Path(pinned["path"]).resolve()
                or pinned["branch_tick"] != case["source_tick"]):
            raise ValueError("original student recording differs from the frozen cohort")
        for name, digest in pinned["files"].items():
            if sha256(actual_source / name) != digest:
                raise ValueError("original student recording changed after assessment")
        if sha256(actual_source / "report.json") != branch["source_rollout_sha256"]:
            raise ValueError("original student recording metadata changed")
        original = json.loads((actual_source / "report.json").read_text())
        if ((original["task"], original["seed"], original["split"]) != (task, seed, "train")
                or original["policy"]["artifact_sha256"] != source_report["source_checkpoint_sha256"]):
            raise ValueError("original student identity differs")
        with np.load(folder / "navigation.npz", allow_pickle=False) as data:
            arrays = {k: data[k].copy() for k in data.files}
        n = branch["label_decisions"]
        validate_tensors(arrays, n, case["source_tick"])
        if (n != branch["continuation_decisions"] or not np.array_equal(arrays["action"], arrays["continuation_action"])
                or not np.array_equal(arrays["proprio"][0], arrays["continuation_state61"][0, :48])):
            raise ValueError("full teacher recovery or its initial sensor state differs")
        result = branch["result"]
        unsafe = any(d.get("planner_phase", d.get("phase")) in ("recover", "no_path") for d in branch["teacher_diagnostics"])
        eligible = bool(result["strict_success"] and result["success"] and not unsafe
                        and not result["collision"] and not result["ever_fallen"] and result["recovery_events"] == 0)
        if (eligible != branch["eligible_training_supervision"] or eligible != selected["eligible"]
                or eligible != case[f"{teacher}_success"] or result["tick"] > 1500
                or not 0 <= result["tick"] - int(arrays["ticks"][-1]) < 5):
            raise ValueError("claimed recovery eligibility differs from strict outcome")
        source_entry = training[task, seed]
        source_meta = json.loads((contained(source_entry["_source_root"], source_entry["path"]) / "episode.json").read_text())
        instruction = source_meta["instruction"]
        if (source_entry["_source_root"].resolve() != base_dataset.resolve()
                and original.get("evaluator", {}).get("scene", {}).get("name") != source_meta["scene"]["name"]):
            raise ValueError("student recovery scene differs from its qualified source")
        if branch["instruction"] != instruction:
            raise ValueError("recovery instruction changed")
        item = {"task": task, "seed": seed, "source_tick": case["source_tick"],
                "source_navigation_sha256": checksums["navigation_sha256"],
                "source_branch_report_sha256": checksums["report_sha256"],
                "source_rollout_report_sha256": branch["source_rollout_sha256"]}
        if source_entry["_source_root"].resolve() != base_dataset.resolve():
            item.update(source_scene=source_meta["scene"]["name"],
                        source_teacher_index_sha256=qualified_teacher_sha256)
        if not eligible:
            rejected.append(item)
            continue
        with np.load(actual_source / "recording.npz", allow_pickle=False) as recorded:
            at = np.flatnonzero(recorded["camera_ticks"] == case["source_tick"])
            if len(at) != 1:
                raise ValueError("original student camera observation is missing")
            first_rgb = recorded["rgb"][int(at[0])].copy()
        diff = np.abs(first_rgb.astype(float) - arrays["rgb"][0])
        if diff.max() > 1 or diff.mean() > .01:
            raise ValueError("replayed camera differs beyond the audited renderer rounding bound")
        # This is the actual source capture, not a synthetic reconstruction.
        arrays["rgb"][0] = first_rgb
        import hashlib

        item.update(first_recorded_rgb_sha256=hashlib.sha256(first_rgb.tobytes()).hexdigest(),
                    rerender_max_abs=float(diff.max()), rerender_mean_abs=float(diff.mean()),
                    decisions=n, source_recording_sha256=sha256(actual_source / "recording.npz"))
        accepted.append(item)
        payloads.append((item, instruction, {key: arrays[key] for key in PUBLIC_KEYS}))
    if not accepted:
        raise ValueError("no successful training recoveries to release")
    return _write_recovery_release(
        source, source_protocol, assessment, expected_assessment_sha256, base_dataset,
        evaluation_protocol, output, accepted, rejected, payloads, qualification,
        source_report["source_checkpoint_sha256"], teacher=teacher, sample_stride=sample_stride,
        qualified_teacher_root=qualified_teacher_root, qualified_teacher_sha256=qualified_teacher_sha256)


def _write_recovery_release(source, source_protocol, assessment, expected_assessment_sha256,
                            base_dataset, evaluation_protocol, output, accepted, rejected,
                            payloads, qualification, checkpoint_sha256, *, teacher, sample_stride,
                            qualified_teacher_root=None, qualified_teacher_sha256=None,
                            source_report_path=None):
    """Shared serialization after the caller has independently audited its source format."""
    output.mkdir(parents=True)
    entries = []
    for item, instruction, arrays in payloads:
        relative = f"{item['task']}-seed{item['seed']}/recovery-{item['source_tick']:04d}"
        folder = output / relative
        folder.mkdir(parents=True)
        np.savez_compressed(folder / "navigation.npz", **arrays)
        starts = list(range(0, item["decisions"], sample_stride))
        meta = {"schema": NAVIGATION_CONTRACT, "kind": KIND, "task": item["task"], "seed": item["seed"],
                "split": "train", "instruction": instruction, "source_tick": item["source_tick"],
                "teacher_variant": teacher, "student_checkpoint_sha256": checkpoint_sha256,
                "initial_rgb_source": "original_student_recording", "actual_student_anchor_starts": [0],
                "teacher_continuation_starts": starts[1:], "search_visibility_gated": True,
                "source_navigation_sha256": item["source_navigation_sha256"]}
        # Optional source scope preserves legacy releases while distinguishing a
        # diagnostic startup prefix from a completed student task.
        meta.update({k: item[k] for k in ("source_scope", "full_student_task_outcome",
                                         "source_scene", "source_teacher_index_sha256") if k in item})
        write_json(folder / "episode.json", meta)
        entries.append({**item, "path": relative, "split": "train", "eligible": True,
                        "search_visibility_gated": True, "frames": item["decisions"] + 1,
                        "sample_starts": starts, "actual_student_anchor_starts": [0],
                        "teacher_continuation_starts": starts[1:], "teacher_variant": teacher,
                        "student_checkpoint_sha256": checkpoint_sha256,
                        "sha256": sha256(folder / "navigation.npz"), "metadata_sha256": sha256(folder / "episode.json")})
    shutil.copyfile(base_dataset / "dataset_stats.json", output / "dataset_stats.json")
    shutil.copyfile(evaluation_protocol, output / "evaluation_protocol.json")
    shutil.copyfile(assessment, output / "source_assessment.json")
    shutil.copyfile(source_protocol, output / "source_protocol.json")
    audit = {"status": "complete", "accepted": accepted, "rejected": rejected,
             "teacher_variant": teacher,
             "source_report_sha256": sha256(source_report_path or source / "report.json"),
             "source_assessment_sha256": expected_assessment_sha256,
             "source_protocol_sha256": sha256(source_protocol),
             "student_checkpoint_sha256": checkpoint_sha256,
             "public_tensor_keys": sorted(PUBLIC_KEYS), "original_first_student_rgb_preserved": True,
             "later_sample_origins": "teacher continuation; not additional independent student visits",
             "sample_stride": sample_stride, "model_weights_changed": False, "capability_accepted": False}
    write_json(output / "source_audit.json", audit)
    index = {"schema": NAVIGATION_CONTRACT, "kind": KIND, "episodes": entries,
             "base_index_sha256": sha256(base_dataset / "index.json"),
             "base_qualification_sha256": sha256(base_dataset / "qualification_report.json"),
             "stats_sha256": qualification["stats_sha256"], "protocol_sha256": sha256(output / "evaluation_protocol.json"),
             "assessment_sha256": expected_assessment_sha256, "audit_sha256": sha256(output / "source_audit.json")}
    index["source_protocol_sha256"] = sha256(output / "source_protocol.json")
    if qualified_teacher_sha256 is not None:
        index["qualified_teacher_index_sha256"] = qualified_teacher_sha256
    write_json(output / "index.json", index)
    receipt = {"status": "complete", "kind": KIND, "index_sha256": sha256(output / "index.json"),
               "trajectories": len(entries), "samples": sum(len(e["sample_starts"]) for e in entries),
               "actual_student_anchors": len(entries), "teacher_continuation_samples": sum(len(e["teacher_continuation_starts"]) for e in entries),
               "decisions": sum(e["decisions"] for e in entries), "public_tensors_only": True,
               "original_first_student_rgb_preserved": True, "model_weights_changed": False, "capability_accepted": False}
    write_json(output / "release.json", receipt)
    load_recoveries(output, base_dataset,
                    qualified_teacher_roots=[qualified_teacher_root] if qualified_teacher_root is not None else [],
                    qualified_teacher_sha256=[qualified_teacher_sha256] if qualified_teacher_sha256 is not None else [])
    return receipt


def load_recoveries(root, base_dataset, *, qualified_teacher_roots=None, qualified_teacher_sha256=None):
    import hashlib

    root, base_dataset = Path(root), Path(base_dataset)
    receipt = json.loads((root / "release.json").read_text())
    index = json.loads((root / "index.json").read_text())
    protocol = json.loads((root / "evaluation_protocol.json").read_text())
    base, _ = qualified_base(base_dataset, protocol)
    roots, hashes = qualified_teacher_roots or [], qualified_teacher_sha256 or []
    if (isinstance(roots, (str, Path)) or isinstance(hashes, str) or len(roots) != len(hashes)
            or len(set(hashes)) != len(hashes)):
        raise ValueError("recovery teacher roots require unique matching index hashes")
    source_hash = index.get("qualified_teacher_index_sha256")
    source_root = None
    if "qualified_teacher_index_sha256" in index:
        if not isinstance(source_hash, str) or source_hash not in hashes:
            raise ValueError("recovery requires its explicitly pinned qualified teacher source")
        source_root = roots[list(hashes).index(source_hash)]
    if (receipt["status"] != "complete" or receipt["kind"] != KIND or index["kind"] != KIND
            or index["schema"] != NAVIGATION_CONTRACT or receipt["index_sha256"] != sha256(root / "index.json")
            or index["base_index_sha256"] != sha256(base_dataset / "index.json")
            or index["base_qualification_sha256"] != sha256(base_dataset / "qualification_report.json")
            or index["stats_sha256"] != sha256(root / "dataset_stats.json")
            or index["stats_sha256"] != sha256(base_dataset / "dataset_stats.json")
            or index["protocol_sha256"] != sha256(root / "evaluation_protocol.json")
            or index["assessment_sha256"] != sha256(root / "source_assessment.json")
            or index["source_protocol_sha256"] != sha256(root / "source_protocol.json")
            or index["audit_sha256"] != sha256(root / "source_audit.json")):
        raise ValueError("recovery release or frozen base artifact changed")
    audit = json.loads((root / "source_audit.json").read_text())
    approved = {(e["task"], e["seed"], e["source_tick"]): e for e in audit["accepted"]}
    training = _training_sources(base_dataset, base, source_root, source_hash)
    forbidden = set(protocol["development_seeds"]) | set(protocol["independent_final_seeds"])
    seen = set()
    for entry in index["episodes"]:
        key = entry["task"], entry["seed"], entry["source_tick"]
        if (key in seen or key not in approved or (entry["task"], entry["seed"]) not in training
                or entry["seed"] in forbidden or split_for_seed(entry["seed"]) != "train"
                or entry["split"] != "train" or not entry["eligible"] or not entry["search_visibility_gated"]
                or entry["student_checkpoint_sha256"] != audit["student_checkpoint_sha256"]
                or entry["teacher_variant"] != audit["teacher_variant"]):
            raise ValueError("duplicate, held-out or unaudited recovery")
        seen.add(key)
        starts = list(range(0, entry["frames"] - 1, audit["sample_stride"]))
        if (entry["sample_starts"] != starts or entry["actual_student_anchor_starts"] != [0]
                or entry["teacher_continuation_starts"] != starts[1:]):
            raise ValueError("student anchor and teacher continuation origins changed")
        folder = contained(root, entry["path"])
        if sha256(folder / "navigation.npz") != entry["sha256"] or sha256(folder / "episode.json") != entry["metadata_sha256"]:
            raise ValueError("recovery episode changed after release")
        meta = json.loads((folder / "episode.json").read_text())
        if "source_scope" in approved[key]:
            if (approved[key]["source_scope"] != "three_second_startup_prefix"
                    or any(entry.get(k) != approved[key].get(k) or meta.get(k) != approved[key].get(k)
                           for k in ("source_scope", "full_student_task_outcome"))
                    or approved[key].get("full_student_task_outcome") is not None):
                raise ValueError("startup source scope or unknown full-student outcome changed")
        source_entry = training[entry["task"], entry["seed"]]
        original = contained(source_entry["_source_root"], source_entry["path"])
        if source_entry["_source_root"].resolve() != base_dataset.resolve():
            source_scene = json.loads((original / "episode.json").read_text())["scene"]["name"]
            for value in (entry, meta, approved[key]):
                if (value.get("source_scene") != source_scene
                        or value.get("source_teacher_index_sha256") != source_hash):
                    raise ValueError("recovery source scene or qualified teacher identity changed")
        if (meta["instruction"] != json.loads((original / "episode.json").read_text())["instruction"]
                or meta["source_navigation_sha256"] != approved[key]["source_navigation_sha256"]
                or meta["initial_rgb_source"] != "original_student_recording"
                or meta["actual_student_anchor_starts"] != [0] or meta["teacher_continuation_starts"] != starts[1:]):
            raise ValueError("recovery instruction, origin or source label changed")
        with np.load(folder / "navigation.npz", allow_pickle=False) as data:
            validate_tensors(data, entry["frames"] - 1, entry["source_tick"], public_only=True)
            if hashlib.sha256(data["rgb"][0].tobytes()).hexdigest() != approved[key]["first_recorded_rgb_sha256"]:
                raise ValueError("first RGB is not the recorded student observation")
    if seen != set(approved):
        raise ValueError("released recovery cohort omits audited trajectories")
    return index["episodes"]
