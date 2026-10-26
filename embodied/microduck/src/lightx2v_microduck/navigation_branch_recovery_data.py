"""Audit expanded recorded teacher branches, then export the common recovery format."""

import hashlib
import json
from pathlib import Path

import numpy as np

from .navigation_augmentation import PUBLIC_KEYS, contained, qualified_base, sha256, validate_tensors
from .navigation_recovery_data import _write_recovery_release
from .wam_data import split_for_seed


def _eligible(branch):
    result = branch["result"]
    unsafe = any(d.get("planner_phase", d.get("phase")) in ("recover", "no_path")
                 for d in branch["teacher_diagnostics"])
    return bool(result["strict_success"] and result["success"] and not unsafe
                and not result["collision"] and not result["ever_fallen"]
                and result["recovery_events"] == 0 and branch["label_decisions"] > 0)


def _audit(source, source_protocol, base_dataset, evaluation_protocol):
    source, source_protocol, base_dataset, evaluation_protocol = map(
        Path, (source, source_protocol, base_dataset, evaluation_protocol))
    protocol = json.loads(source_protocol.read_text())
    report = json.loads((source / "report.json").read_text())
    evaluation = json.loads(evaluation_protocol.read_text())
    base, qualification = qualified_base(base_dataset, evaluation)
    script = Path(protocol["source_snapshot"]) / "scripts/microduck/replay_navigation_full_branches.py"
    if (report["status"] != "complete" or report["protocol_sha256"] != sha256(source_protocol)
            or report["checkpoint_sha256"] != protocol["student_checkpoint_sha256"]
            or report["script_sha256"] != sha256(script)
            or protocol["evaluation_protocol_sha256"] != sha256(evaluation_protocol)):
        raise ValueError("complete pinned branch expansion required")
    training = {(e["task"], e["seed"]): e for e in base["episodes"] if e["eligible"] and e["split"] == "train"}
    forbidden = set(evaluation["development_seeds"]) | set(evaluation["independent_final_seeds"])
    pinned = {(e["task"], e["seed"]): e for e in protocol["sources"]}
    rows = {(e["task"], e["seed"]): e for e in report["cases"]}
    if (len(pinned) != len(protocol["sources"]) or len(rows) != len(report["cases"])
            or set(rows) != set(pinned) or report["planned_sources"] != len(rows)):
        raise ValueError("incomplete or duplicate expansion cohort")
    accepted, rejected, payloads, maximum_rgb_error = [], [], [], 0.
    for key, row in rows.items():
        task, seed = key
        if key not in training or seed in forbidden or split_for_seed(seed) != "train":
            raise ValueError("held-out or unqualified student source")
        entry = pinned[key]
        directory = Path(entry["path"])
        if set(entry["files"]) != {"report.json", "control.json", "recording.npz"}:
            raise ValueError("complete original recording required")
        for name, digest in entry["files"].items():
            if sha256(directory / name) != digest:
                raise ValueError("original recording changed")
        original = json.loads((directory / "report.json").read_text())
        if (original["policy"]["artifact_sha256"] != report["checkpoint_sha256"]
                or original["policy"]["uses_privileged_state"] is not False
                or original["fallback_control_ticks"] != 0 or original["split"] != "train"
                or row["original_student_physical_replay_exact"] is not True):
            raise ValueError("student provenance or physical replay differs")
        old_entries = {b["source_tick"]: b for b in original["branches"]}
        expanded = {b["source_tick"]: b for b in row["branches"]}
        if (len(old_entries) != len(original["branches"]) or len(expanded) != len(row["branches"])
                or set(entry["branches"]) != {b["path"] for b in original["branches"]}):
            raise ValueError("duplicate or changed original branch inventory")
        expected_expanded = set()
        instruction = json.loads((contained(base_dataset, training[key]["path"]) / "episode.json").read_text())["instruction"]
        with np.load(directory / "recording.npz", allow_pickle=False) as recording:
            camera_ticks, original_rgb = recording["camera_ticks"], recording["rgb"]
        for tick, old_entry in old_entries.items():
            old_folder = contained(directory, old_entry["path"])
            for name, digest in entry["branches"][old_entry["path"]].items():
                if sha256(old_folder / name) != digest:
                    raise ValueError("original teacher branch changed")
            old_meta = json.loads((old_folder / "report.json").read_text())
            if (_eligible(old_meta) != old_meta["eligible_training_supervision"]
                    or _eligible(old_meta) != old_entry["eligible_training_supervision"]):
                raise ValueError("original branch eligibility disagrees with outcome")
            item = {"task": task, "seed": seed, "source_tick": tick}
            if not _eligible(old_meta):
                rejected.append({**item, "source_branch_report_sha256": sha256(old_folder / "report.json")})
                continue
            expected_expanded.add(tick)
            if tick not in expanded:
                raise ValueError("a successful original branch was omitted")
            selected = expanded[tick]
            folder = contained(source, selected["path"])
            if (sha256(folder / "report.json") != selected["report_sha256"]
                    or sha256(folder / "navigation.npz") != selected["navigation_sha256"]):
                raise ValueError("expanded artifacts changed")
            meta = json.loads((folder / "report.json").read_text())
            if ((meta["task"], meta["seed"], meta["split"], meta["source_tick"]) != (task, seed, "train", tick)
                    or meta["student_model"]["artifact_sha256"] != report["checkpoint_sha256"]
                    or meta["instruction"] != instruction or meta["original_branch_sha256"] != sha256(old_folder / "report.json")
                    or meta["original_navigation_sha256"] != sha256(old_folder / "navigation.npz")
                    or meta["source_rollout_sha256"] != entry["files"]["report.json"]
                    or meta["teacher_full_control_trace_exact"] is not True or not _eligible(meta)):
                raise ValueError("expanded branch identity or outcome differs")
            final = meta["result"]
            for name in ("tick", "goal_hold_ticks", "strict_success", "success", "robot_position", "robot_velocity",
                         "evaluator_source_sha256", "tasks_source_sha256", "success_protocol", "action_delay_ticks"):
                if final[name] != old_meta["result"][name]:
                    raise ValueError("expanded evaluator or outcome changed")
            if final["tick"] > 1500 or final["goal_hold_ticks"] < 100:
                raise ValueError("original deadline or two-second stop not met")
            with np.load(folder / "navigation.npz", allow_pickle=False) as data:
                arrays = {name: data[name].copy() for name in data.files}
            count = meta["label_decisions"]
            validate_tensors(arrays, count, tick)
            if (count != meta["continuation_decisions"] or not np.array_equal(arrays["action"], arrays["continuation_action"])
                    or not np.array_equal(arrays["proprio"][:-1], arrays["continuation_state61"][::5, :48][:count])
                    or not 0 <= final["tick"] - arrays["ticks"][-1] < 5):
                raise ValueError("full actions, proprioception or time alignment differs")
            with np.load(old_folder / "navigation.npz", allow_pickle=False) as old:
                n = old_meta["label_decisions"]
                for name in ("continuation_action", "continuation_state61", "continuation_raw_action"):
                    if not np.array_equal(arrays[name], old[name]):
                        raise ValueError("full original control trace differs")
                for name, length in (("action", n), ("proprio", n + 1), ("ticks", n + 1)):
                    if not np.array_equal(arrays[name][:length], old[name]):
                        raise ValueError("original prefix labels differ")
                difference = np.abs(arrays["rgb"][:n + 1].astype(float) - old["rgb"]).mean(axis=(1, 2, 3))
                maximum_rgb_error = max(maximum_rgb_error, float(difference.max()))
                if float(difference.max()) > protocol["max_mean_rgb_rerender_error"]:
                    raise ValueError("camera rerender error exceeds pinned bound")
            match = np.flatnonzero(camera_ticks == tick)
            if len(match) != 1 or not np.array_equal(arrays["rgb"][0], original_rgb[int(match[0])]):
                raise ValueError("first frame is not the actual student's observation")
            item.update(source_navigation_sha256=sha256(folder / "navigation.npz"),
                source_branch_report_sha256=sha256(folder / "report.json"),
                source_rollout_report_sha256=entry["files"]["report.json"],
                first_recorded_rgb_sha256=hashlib.sha256(arrays["rgb"][0].tobytes()).hexdigest(),
                source_recording_sha256=entry["files"]["recording.npz"],
                rerender_max_abs=0., rerender_mean_abs=0., decisions=count,
                original_label_decisions=n, stop_decisions=int((arrays["action"][:, 4] >= .5).sum()))
            accepted.append(item)
            payloads.append((item, instruction, {name: arrays[name] for name in PUBLIC_KEYS}))
        if (set(expanded) != expected_expanded or row["eligible_branches"] != len(expected_expanded)
                or row["rejected_branches"] != len(old_entries) - len(expected_expanded)):
            raise ValueError("expanded cohort differs from independently computed eligibility")
    assessment = {"status": "complete", "kind": "expanded_recorded_teacher_branches",
        "source_report_sha256": sha256(source / "report.json"), "protocol_sha256": sha256(source_protocol),
        "evaluation_protocol_sha256": sha256(evaluation_protocol), "source_checkpoint_sha256": report["checkpoint_sha256"],
        "sources": len(rows), "accepted": accepted, "rejected": rejected,
        "accepted_branches": len(accepted), "rejected_branches": len(rejected),
        "full_label_decisions": sum(e["decisions"] for e in accepted),
        "original_label_decisions": sum(e["original_label_decisions"] for e in accepted),
        "stop_label_decisions": sum(e["stop_decisions"] for e in accepted),
        "maximum_prefix_rgb_mean_error": maximum_rgb_error,
        "all_full_control_traces_exact": True, "all_original_student_first_rgb_exact": True,
        "train_only": True, "training_release_allowed": False, "learned_capability_accepted": False}
    return assessment, payloads, qualification


def audit_expanded_branches(source, source_protocol, base_dataset, evaluation_protocol):
    return _audit(source, source_protocol, base_dataset, evaluation_protocol)[0]


def export_expanded_recoveries(source, source_protocol, assessment, expected_assessment_sha256,
                               base_dataset, evaluation_protocol, output, *, sample_stride=4):
    source, source_protocol, assessment, base_dataset, evaluation_protocol, output = map(
        Path, (source, source_protocol, assessment, base_dataset, evaluation_protocol, output))
    if (output.exists() or type(sample_stride) is not int or sample_stride < 1
            or sha256(assessment) != expected_assessment_sha256):
        raise ValueError("existing output, invalid stride or changed independent assessment")
    fresh, payloads, qualification = _audit(source, source_protocol, base_dataset, evaluation_protocol)
    if fresh != json.loads(assessment.read_text()) or not fresh["accepted"]:
        raise ValueError("assessment differs or no successful full recovery available")
    return _write_recovery_release(source, source_protocol, assessment, expected_assessment_sha256,
        base_dataset, evaluation_protocol, output, fresh["accepted"], fresh["rejected"], payloads,
        qualification, fresh["source_checkpoint_sha256"], teacher="original", sample_stride=sample_stride)
