"""Release qualified teacher recoveries from explicitly incomplete student tasks.

The source is a complete startup *recording*, not a complete student episode.
Only recorded student anchors and successful full teacher continuations become
public training tensors; no full student outcome is inferred from the prefix.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from .navigation_augmentation import PUBLIC_KEYS, contained, qualified_base, sha256, validate_tensors
from .navigation_recovery_data import _write_recovery_release
from .wam_data import split_for_seed


SCOPE = "three_second_startup_prefix"


def audit_startup_recoveries(source, source_protocol, base_dataset, evaluation_protocol):
    """Recompute provenance, cohort membership and eligibility before serialization."""
    return _audit(source, source_protocol, base_dataset, evaluation_protocol)[0]


def _audit(source, source_protocol, base_dataset, evaluation_protocol):
    source, source_protocol, base_dataset, evaluation_protocol = map(
        Path, (source, source_protocol, base_dataset, evaluation_protocol))
    protocol = json.loads(source_protocol.read_text())
    report = json.loads((source / "report.json").read_text())
    evaluation = json.loads(evaluation_protocol.read_text())
    base, qualification = qualified_base(base_dataset, evaluation)
    if (report["status"] != "complete" or report["protocol_sha256"] != sha256(source_protocol)
            or report["source_checkpoint_sha256"] != protocol["student_checkpoint_sha256"]
            or protocol["evaluation_protocol_sha256"] != sha256(evaluation_protocol)
            or protocol["source_scope"] != SCOPE or protocol["branch_offsets_ticks"] != [0, 50]
            or protocol["hold_ticks"] != 100 or protocol["max_ticks_unchanged"] != 1500
            or report["original_task_deadline_preserved"] is not True):
        raise ValueError("complete pinned startup recovery audit required")
    teacher_qualification = Path(protocol["teacher_qualification_path"])
    if sha256(teacher_qualification) != protocol["teacher_qualification_sha256"]:
        raise ValueError("teacher qualification changed")
    teacher = json.loads(teacher_qualification.read_text())
    if (teacher["status"] != "complete" or teacher["teacher_qualified"] is not True
            or teacher["teacher"] != "visual_search_obstacle_aware"):
        raise ValueError("qualified aware teacher required")
    training = {(e["task"], e["seed"]): e for e in base["episodes"] if e["eligible"] and e["split"] == "train"}
    forbidden = set(evaluation["development_seeds"]) | set(evaluation["independent_final_seeds"])
    rows = {(e["task"], e["seed"]): e for e in report["cases"]}
    if len(rows) != len(report["cases"]) or len(rows) != len(protocol["sources"]) or len(rows) != report["planned_sources"]:
        raise ValueError("incomplete or duplicate startup cohort")
    seen, accepted, rejected, payloads = set(), [], [], []
    for entry in protocol["sources"]:
        directory = Path(entry["path"])
        if set(entry["files"]) != {"report.json", "control.json", "recording.npz", "predictions.json"}:
            raise ValueError("complete original sensor/control recording required")
        for name, digest in entry["files"].items():
            if sha256(directory / name) != digest:
                raise ValueError("original startup recording changed")
        original = json.loads((directory / "report.json").read_text())
        key = original["task"], original["seed"]
        task, seed = key
        if key in seen or key not in rows or key not in training or seed in forbidden or split_for_seed(seed) != "train":
            raise ValueError("duplicate, held-out or unqualified startup source")
        seen.add(key)
        row = rows[key]
        e, policy = original["evaluator"], original["policy"]
        if (original["mode"] != "offline" or original["initialization"] != "reset_and_warmup"
                or original["cancelled"] is not False or original["inference_worker_stopped"] is not True
                or original["fallback_control_ticks"] or original["rejected_plans"]
                or policy["artifact_sha256"] != protocol["student_checkpoint_sha256"]
                or policy["uses_privileged_state"] is not False or policy["sampling_noise_seed"] != protocol["noise_seed"]
                or policy["contract"] != "microduck_navigation_v1"
                or e["scene"]["name"] != "room_clear" or e["tick"] != 150 or e["terminated"] or e["strict_success"]
                or e["collision"] or e["ever_fallen"] or e["recovery_events"]
                or abs(e["evaluation_origin_s"] - 1.) > 1e-8 or abs(e["evaluation_elapsed_s"] - 3.) > 1e-8
                or row["source_scope"] != SCOPE or row["full_student_task_outcome"] is not None):
            raise ValueError("invalid student prefix provenance or falsely assigned full-task outcome")
        controls = json.loads((directory / "control.json").read_text())
        if ([c["tick"] for c in controls] != list(range(150))
                or any(c["execution_source"] != "navigation_model" or c["skill"] not in ("walk", "stand") for c in controls)):
            raise ValueError("incomplete or intervened student control")
        onset = next((t for t in range(0, 51, 5) if all(c["skill"] == "stand" for c in controls[t:t+100])), None)
        expected = [None, None] if onset is None else [onset, onset + 50]
        if expected != entry["branch_ticks"] or expected != row["branch_ticks"]:
            raise ValueError("action-only branch criterion changed")
        if onset is None:
            if row["branches"] or row["skipped_reason"] != "no_sustained_stand":
                raise ValueError("nonselected prefix has teacher branches")
            continue
        if row["original_physical_replay_max_error"] != 0 or row["original_proprio_replay_max_error"] != 0:
            raise ValueError("startup physical or sensor replay differs")
        if [b["tick"] for b in row["branches"]] != expected:
            raise ValueError("missing, duplicate or reordered startup branch")
        predictions = json.loads((directory / "predictions.json").read_text())
        if len(predictions) != 30 or [p["source_tick"] for p in predictions if p["accepted"]] != list(range(0, 150, 5)):
            raise ValueError("startup prediction recording incomplete")
        with np.load(directory / "recording.npz", allow_pickle=False) as packed:
            recorded = {k: packed[k] for k in ("camera_ticks", "rgb", "physics_ticks")}
        if (not np.array_equal(recorded["camera_ticks"], np.arange(0, 150, 5))
                or not np.array_equal(recorded["physics_ticks"], np.arange(150)) or len(recorded["rgb"]) != 30):
            raise ValueError("startup sensor clock incomplete")
        instruction = json.loads((contained(base_dataset, training[key]["path"]) / "episode.json").read_text())["instruction"]
        for selected in row["branches"]:
            tick = selected["tick"]
            folder = contained(source, f"{task}-seed{seed}/branch-{tick:04d}")
            if sha256(folder / "report.json") != selected["report_sha256"]:
                raise ValueError("branch report changed")
            branch = json.loads((folder / "report.json").read_text())
            if (sha256(folder / "navigation.npz") != branch["navigation_sha256"]
                    or (branch["task"], branch["seed"], branch["source_tick"], branch["split"]) != (task, seed, tick, "train")
                    or branch["student_model"] != policy or branch["teacher_variant"] != "aware"
                    or branch["source_rollout_sha256"] != entry["files"]["report.json"]
                    or Path(branch["source_rollout"]).resolve() != directory.resolve()
                    or branch["instruction"] != instruction or branch["source_scope"] != SCOPE
                    or branch["full_student_task_outcome"] is not None):
                raise ValueError("branch identity, sensors or source scope changed")
            with np.load(folder / "navigation.npz", allow_pickle=False) as packed:
                arrays = {k: packed[k] for k in packed.files}
            count = branch["label_decisions"]
            validate_tensors(arrays, count, tick)
            if (count != branch["continuation_decisions"] or not np.array_equal(arrays["action"], arrays["continuation_action"])
                    or not np.array_equal(arrays["proprio"][:-1], arrays["continuation_state61"][::5, :48][:count])
                    or not np.array_equal(arrays["proprio"][0], np.asarray(predictions[tick//5]["observed_proprio48"], np.float32))
                    or not np.array_equal(arrays["rgb"][0], recorded["rgb"][tick//5])):
                raise ValueError("labels differ from full teacher trace or actual student sensor anchor")
            result = branch["result"]
            for name in ("evaluator_source_sha256", "tasks_source_sha256", "success_protocol", "action_delay_ticks", "evaluation_origin_s"):
                if result[name] != e[name]:
                    raise ValueError("teacher evaluator or task clock changed")
            if (result["tick"] > 1500 or not 0 <= result["tick"] - arrays["ticks"][-1] < 5
                    or result["scene"]["layout_id"] != e["scene"]["layout_id"]):
                raise ValueError("teacher continuation left the original task")
            diagnostics = branch["teacher_diagnostics"]
            if any(d.get("hidden_target_coordinates_used", False) for d in diagnostics):
                raise ValueError("teacher used hidden target coordinates")
            unsafe = any(d.get("planner_phase", d.get("phase")) in ("recover", "no_path") for d in diagnostics)
            eligible = bool(result["success"] and result["strict_success"] and result["goal_hold_ticks"] >= 100
                            and not result["collision"] and not result["ever_fallen"] and result["recovery_events"] == 0 and not unsafe)
            if not (eligible == branch["eligible_training_supervision"] == branch["teacher_continuation_success"]
                    == selected["teacher_success"] == selected["eligible"]):
                raise ValueError("strict teacher outcome contradicts eligibility")
            item = dict(task=task, seed=seed, source_tick=tick, source_scope=SCOPE, full_student_task_outcome=None,
                        source_navigation_sha256=branch["navigation_sha256"], source_branch_report_sha256=selected["report_sha256"],
                        source_rollout_report_sha256=entry["files"]["report.json"], source_recording_sha256=entry["files"]["recording.npz"],
                        first_recorded_rgb_sha256=hashlib.sha256(arrays["rgb"][0].tobytes()).hexdigest(),
                        rerender_max_abs=0., rerender_mean_abs=0., decisions=count)
            (accepted if eligible else rejected).append(item)
            if eligible:
                payloads.append((item, instruction, {k: arrays[k] for k in PUBLIC_KEYS}))
    if seen != set(rows):
        raise ValueError("unverified source in startup cohort")
    assessment = dict(status="complete", kind="startup_prefix_recoveries", source_scope=SCOPE,
                      source_report_sha256=sha256(source / "report.json"), protocol_sha256=sha256(source_protocol),
                      source_checkpoint_sha256=protocol["student_checkpoint_sha256"], accepted=accepted, rejected=rejected,
                      independent_training_layouts=len({s for _, s in seen}), recorded_prefixes=len(seen),
                      actual_student_anchors=len(accepted), decisions=sum(i["decisions"] for i in accepted),
                      full_student_task_outcomes_unknown=True, original_first_student_sensors_exact=True,
                      training_release_allowed=False, learned_capability_accepted=False)
    return assessment, payloads, qualification


def export_startup_recoveries(source, source_protocol, assessment, expected_assessment_sha256,
                              base_dataset, evaluation_protocol, output, *, sample_stride=4):
    output, assessment = Path(output), Path(assessment)
    if output.exists():
        raise FileExistsError("recovery releases are immutable")
    if type(sample_stride) is not int or sample_stride < 1:
        raise ValueError("positive integer sample stride required")
    reviewed, payloads, qualification = _audit(source, source_protocol, base_dataset, evaluation_protocol)
    if sha256(assessment) != expected_assessment_sha256 or json.loads(assessment.read_text()) != reviewed:
        raise ValueError("startup recovery assessment differs from current source evidence")
    if not reviewed["accepted"]:
        raise ValueError("no successful startup recoveries to release")
    return _write_recovery_release(*map(Path, (source, source_protocol, assessment)), expected_assessment_sha256,
        Path(base_dataset), Path(evaluation_protocol), output, reviewed["accepted"], reviewed["rejected"], payloads,
        qualification, reviewed["source_checkpoint_sha256"], teacher="aware", sample_stride=sample_stride)
