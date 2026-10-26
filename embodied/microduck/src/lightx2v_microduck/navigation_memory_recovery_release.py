"""Publish independently replayed clear-room teacher branches from current students.

The caller pins the completed label execution and anchor diagnosis. This module
checks every selected source/branch against those inputs, including failures,
and exports only successful TRAIN supervision through the existing loaders.
Multiple anchors share a student episode; they are not independent layouts.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from .navigation_augmentation import PUBLIC_KEYS, contained, qualified_base, sha256, validate_tensors
from .navigation_history_recovery import export_prefixes, load_prefixes, validate_prefix
from .navigation_recovery_data import _training_sources, _write_recovery_release, load_recoveries
from .wam_data import split_for_seed, write_json

SCHEMA = "microduck_current_memory_clear_recovery_publication_v1"
SOURCE_FILES = {"report.json", "control.json", "predictions.json", "recording.npz"}


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def _unique(rows, keys, description):
    result = {tuple(row[key] for key in keys): row for row in rows}
    require(len(result) == len(rows), "duplicate " + description)
    return result


def _branch(folder, row, case, original, recorded, instruction, checkpoint):
    """Check saved independent replay evidence and public/private tensor identity."""
    require(sha256(folder / "report.json") == row["report_sha256"], "branch report changed")
    branch = read(folder / "report.json")
    tick, n = row["source_tick"], branch["label_decisions"]
    require((branch["task"], branch["seed"], branch["source_tick"], branch["split"]) ==
            (case["task"], case["seed"], tick, "train")
            and branch["student_checkpoint_sha256"] == checkpoint
            and branch["instruction"] == instruction
            and branch["source_files"] == case["files"]
            and Path(branch["source_rollout"]).resolve() == original.resolve()
            and branch["source_recording_sha256"] == case["files"]["recording.npz"],
            "branch source identity differs")
    require(type(n) is int and n >= 0 and n == branch["continuation_decisions"],
            "full teacher continuation required")
    expected_files = {"trajectory.npz", "navigation.npz", "camera_prefix.npz"} if n else {"trajectory.npz"}
    require(set(branch["files"]) == expected_files, "missing or unexpected branch payload")
    for name, digest in branch["files"].items():
        require(sha256(contained(folder, name)) == digest, "teacher payload changed")
    with np.load(folder / "trajectory.npz", allow_pickle=False) as z:
        raw = {key: z[key] for key in z.files}
    validate_tensors(raw, n, tick, allow_empty_diagnostic=n == 0)
    result, replay = branch["result"], branch["replay"]
    controls = result["tick"] - tick
    require(0 < controls and result["tick"] <= 1500
            and 0 <= result["tick"] - int(raw["ticks"][-1]) < 5
            and controls == replay["exact_control_ticks"]
            and controls == len(raw["continuation_state61"])
            and replay["qualified_teacher_decisions_exact"] == (controls + 4) // 5
            and replay["complete_branch_evaluator_exact"] is True
            and replay["original_student_outcome_exact"] is True,
            "incomplete independent replay or changed original task deadline")
    np.testing.assert_array_equal(raw["action"], raw["continuation_action"])
    np.testing.assert_array_equal(raw["proprio"][0], raw["continuation_state61"][0, :48])
    require(all(len(raw[key]) == controls for key in
                ("continuation_qpos", "continuation_raw_action", "continuation_command13", "continuation_skill_id")),
            "incomplete physical control trace")
    at = tick // 5
    require(int(recorded["camera_ticks"][at]) == tick, "student anchor timestamp missing")
    np.testing.assert_array_equal(raw["rgb"][0], recorded["rgb"][at])
    np.testing.assert_array_equal(raw["proprio"][0], recorded["proprio48"][at])
    require(replay["exact_teacher_rgb_frames"] == n + 1, "terminal teacher image not replayed")
    if n:
        with np.load(folder / "navigation.npz", allow_pickle=False) as z:
            public = {key: z[key] for key in z.files}
        validate_tensors(public, n, tick, public_only=True)
        for key in PUBLIC_KEYS:
            np.testing.assert_array_equal(public[key], raw[key])
        with np.load(folder / "camera_prefix.npz", allow_pickle=False) as z:
            require(set(z.files) == {"rgb", "ticks"}, "privileged prefix tensors")
            validate_prefix(z["rgb"], z["ticks"], tick, public["rgb"][0])
            np.testing.assert_array_equal(z["rgb"], recorded["rgb"][:at + 1])
            np.testing.assert_array_equal(z["ticks"], recorded["camera_ticks"][:at + 1])
        require(branch["public_tensor_keys"] == sorted(PUBLIC_KEYS)
                and branch["prefix_tensor_keys"] == ["rgb", "ticks"]
                and replay["exact_public_rgb_frames"] == replay["exact_public_proprio_frames"] == n + 1
                and row["public_frames"] == n + 1, "public tensors were not independently replayed")
    else:
        require(branch["public_tensor_keys"] == branch["prefix_tensor_keys"] == []
                and row["public_frames"] == replay["exact_public_rgb_frames"] ==
                replay["exact_public_proprio_frames"] == 0, "empty branch has public supervision")
    unsafe = any(d.get("planner_phase", d.get("phase")) in ("recover", "no_path")
                 for d in branch["teacher_diagnostics"])
    success = bool(result["strict_success"] and result["success"] and not unsafe
                   and not result["collision"] and not result["ever_fallen"]
                   and result["recovery_events"] == 0 and not result.get("fault"))
    eligible = success and n > 0
    require(branch["teacher_continuation_success"] is success and row["teacher_success"] is success
            and branch["eligible_training_supervision"] is eligible
            and row["eligible_training_supervision"] is eligible, "claimed eligibility differs from strict teacher result")
    item = dict(task=case["task"], seed=case["seed"], source_tick=tick, decisions=n,
                source_branch_path=str(folder), source_branch_report_sha256=row["report_sha256"],
                source_navigation_sha256=branch["files"].get("navigation.npz"),
                source_rollout_report_sha256=case["files"]["report.json"],
                source_recording_sha256=case["files"]["recording.npz"],
                first_recorded_rgb_sha256=hashlib.sha256(recorded["rgb"][at].tobytes()).hexdigest(),
                original_student_success=False, teacher_success=success,
                eligible_training_supervision=eligible, instruction=instruction,
                failure=None if eligible else result)
    return item


def audit_clear_labels(source, rows, source_results, selected_sources, collection_root,
                       base_dataset, evaluation_protocol, checkpoint_sha256):
    """No publication; audit all anchors, including rejected and zero-action branches."""
    source, collection_root, base_dataset = map(Path, (source, collection_root, base_dataset))
    evaluation = read(evaluation_protocol)
    base, _ = qualified_base(base_dataset, evaluation)
    training = _training_sources(base_dataset, base)
    cases = _unique(selected_sources, ("task", "seed"), "selected source")
    summaries = _unique(source_results, ("task", "seed"), "source result")
    branches = _unique(rows, ("task", "seed", "source_tick"), "teacher branch")
    require(cases and set(cases) == set(summaries), "source cohort is incomplete")
    expected = {(t, s, tick) for (t, s), case in cases.items() for tick in case["anchors"]}
    require(set(branches) == expected, "selected branch cohort is incomplete")
    forbidden = set(evaluation["development_seeds"]) | set(evaluation["independent_final_seeds"])
    accepted, rejected, protocols, original_outcomes = [], [], {}, []
    for key, case in cases.items():
        task, seed = key
        require(case["scene"] == "room_clear" and key in training and seed not in forbidden
                and split_for_seed(seed) == "train", "held-out or unqualified clear-room source")
        anchors = case["anchors"]
        require(anchors == sorted(set(anchors)) and all(type(t) is int and 0 <= t <= 1400 and t % 5 == 0 for t in anchors),
                "invalid or duplicated selected anchors")
        require(set(case["files"]) == SOURCE_FILES, "incomplete student source binding")
        original = contained(collection_root, f"{task}-seed{seed}")
        for name, digest in case["files"].items():
            require(sha256(original / name) == digest, "original student recording changed")
        meta = read(original / "report.json")
        summary = summaries[key]
        require((meta["task"], meta["seed"], meta["split"]) == (task, seed, "train")
                and meta["policy"]["artifact_sha256"] == checkpoint_sha256
                and meta["policy"]["uses_privileged_state"] is False
                and meta["evaluator"]["scene"]["name"] == "room_clear"
                and type(case["original_student_success"]) is bool
                and meta["student_task_success"] is case["original_student_success"]
                and summary["original_student_success"] is case["original_student_success"]
                and summary["source_files"] == case["files"] and summary["anchors"] == len(anchors),
                "source outcome, identity or label count changed")
        original_outcomes.append(dict(task=task, seed=seed, success=meta["student_task_success"], anchors=len(anchors)))
        if not anchors:
            continue
        require(meta["student_task_success"] is False and summary.get("source_outcome_exact") is True,
                "source failure was not independently preserved")
        instruction = read(contained(base_dataset, training[key]["path"]) / "episode.json")["instruction"]
        require(meta["instruction"] == instruction, "student instruction differs from qualified training source")
        with np.load(original / "recording.npz", allow_pickle=False) as recorded:
            for ordinal, tick in enumerate(anchors):
                row = branches[task, seed, tick]
                require(row["path"] == f"{task}-seed{seed}/anchor-{tick:04d}", "unexpected teacher branch path")
                item = _branch(contained(source, row["path"]), row, case, original, recorded,
                               instruction, checkpoint_sha256)
                item["anchor_ordinal"] = ordinal
                (accepted if item["eligible_training_supervision"] else rejected).append(item)
                protocols.setdefault(ordinal, []).append(dict(task=task, seed=seed, path=str(original),
                    branch_tick=tick, files=case["files"]))
    return dict(schema=SCHEMA, status="complete", checkpoint_sha256=checkpoint_sha256,
                selected_cases=len(cases), physical_layouts=len({s for _, s in cases}),
                selected_anchors=len(expected), accepted=accepted, rejected=rejected,
                original_student_outcomes=original_outcomes, sources_by_anchor_ordinal=protocols,
                public_tensor_keys=sorted(PUBLIC_KEYS), independent_teacher_replay_verified=True,
                full_original_student_outcomes_preserved=True, capability_accepted=False)


def publish_clear_labels(source, rows, source_results, selected_sources, collection_root,
                         base_dataset, evaluation_protocol, checkpoint_sha256, output, *,
                         provenance_files, source_report_path, sample_stride=4):
    """Export ordinal groups with exact original camera history and frozen statistics.

    Ordinals only avoid duplicate task/seed identities in legacy prefix loaders.
    Each release retains actual source ticks and the complete batch assessment.
    A caller must distinguish a single-case preflight from a production cohort.
    """
    source, base_dataset, evaluation_protocol, output = map(Path,
        (source, base_dataset, evaluation_protocol, output))
    require(type(sample_stride) is int and sample_stride > 0, "positive sample stride required")
    if output.exists():
        raise FileExistsError("current-memory releases are immutable")
    require(provenance_files and str(source_report_path) in provenance_files,
            "source report requires an explicit provenance hash")
    for path, digest in provenance_files.items():
        require(sha256(path) == digest, "publication provenance changed: " + path)
    audit = audit_clear_labels(source, rows, source_results, selected_sources, collection_root,
                              base_dataset, evaluation_protocol, checkpoint_sha256)
    audit["provenance_files"] = provenance_files
    # No writes occur until every selected branch, including failures, passes.
    output.mkdir(parents=True)
    assessment = output / "assessment.json"
    write_json(assessment, audit)
    _, qualification = qualified_base(base_dataset, read(evaluation_protocol))
    releases = []
    for ordinal, sources in audit["sources_by_anchor_ordinal"].items():
        accepted = [item for item in audit["accepted"] if item["anchor_ordinal"] == ordinal]
        rejected = [item for item in audit["rejected"] if item["anchor_ordinal"] == ordinal]
        if not accepted:
            continue
        protocol = output / f"source-protocol-{ordinal:02d}.json"
        write_json(protocol, dict(schema=SCHEMA, evaluation_protocol_sha256=sha256(evaluation_protocol),
            sources=sources, label_assessment_sha256=sha256(assessment), anchor_ordinal=ordinal))
        payloads = []
        for item in accepted:
            path = Path(item["source_branch_path"]) / "navigation.npz"
            require(sha256(path) == item["source_navigation_sha256"], "audited public tensors changed")
            with np.load(path, allow_pickle=False) as z:
                payloads.append((item, item["instruction"], {key: z[key].copy() for key in z.files}))
        root, prefix = output / "releases" / f"anchor-{ordinal:02d}", output / "prefixes" / f"anchor-{ordinal:02d}"
        _write_recovery_release(source, protocol, assessment, sha256(assessment), base_dataset,
            evaluation_protocol, root, accepted, rejected, payloads, qualification, checkpoint_sha256,
            teacher="aware", sample_stride=sample_stride, source_report_path=Path(source_report_path))
        history = export_prefixes(root, base_dataset, prefix, current_checkpoint_sha256=checkpoint_sha256)
        entries = load_recoveries(root, base_dataset)
        load_prefixes(prefix, history["index_sha256"], root, entries)
        releases.append(dict(kind="clear_recovery", root=str(root), index_sha256=sha256(root / "index.json"),
            student_checkpoint_sha256=checkpoint_sha256, history_prefix_root=str(prefix),
            history_prefix_sha256=history["index_sha256"]))
    for path, digest in provenance_files.items():
        require(sha256(path) == digest, "publication provenance changed during export")
    receipt = dict(status="complete", schema=SCHEMA, assessment_sha256=sha256(assessment),
        selected_cases=audit["selected_cases"], physical_layouts=audit["physical_layouts"],
        selected_anchors=audit["selected_anchors"], accepted_branches=len(audit["accepted"]),
        rejected_branches=len(audit["rejected"]), correction_sources=releases,
        model_weights_changed=False, capability_accepted=False, hardware_validated=False)
    write_json(output / "report.json", receipt)
    return receipt
