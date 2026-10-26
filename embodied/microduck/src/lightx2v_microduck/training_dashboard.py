"""Read-only dashboard views over an atomically synchronized training snapshot."""
from __future__ import annotations

import hashlib
from functools import lru_cache
import json
import mimetypes
import os
from pathlib import Path
import time
from urllib.parse import quote

SCHEMA = "microduck_training_snapshot_v1"
TASK_LABELS = {
    "walk_forward": "向前走并停下", "circle_left": "向左转圈", "figure_eight": "走八字",
    "zigzag": "之字行走", "sway_circle": "摇摆后转圈", "kick_left": "左脚踢球",
    "kick_right": "右脚踢球", "ground_lower": "官方低头近地后起身", "sit_walk": "坐下、起身、行走",
    "roll_circle": "翻滚后转圈", "roller_glide": "滚轮滑行", "roller_crouch": "滚轮蹲伏", "roller_slalom": "滚轮蛇形",
}


def default_directory():
    configured = os.environ.get("MICRODUCK_TRAINING_DIR")
    if configured:
        return Path(configured).resolve()
    # In the source checkout. Installed packages should set the environment variable.
    return Path(__file__).resolve().parents[4] / "artifacts/training-monitor"


def read_json(path, default=None):
    try:
        if path.stat().st_size > 40 * 1024 * 1024:
            raise ValueError("training snapshot exceeds 40 MiB")
        return json.loads(path.read_text())
    except FileNotFoundError:
        return default


def artifact_path(directory, artifact):
    relative = Path(artifact["path"])
    release_manifest = (len(relative.parts) == 3 and relative.parts[0] == "releases"
                        and valid_digest(relative.parts[1]) and relative.parts[2] == "source-manifest.json"
                        and artifact.get("kind") == "report")
    if not relative.parts or relative.is_absolute() or ".." in relative.parts or "\\" in str(relative) or (relative.parts[0] not in {"runs", "diagnostics", "reports", "logs"} and not release_manifest):
        raise ValueError("invalid training evidence path")
    root = (directory / "evidence").resolve()
    if artifact.get("content_addressed"):
        if not valid_digest(artifact.get("sha256")):
            raise ValueError("invalid evidence version")
        relative = Path(".versions") / artifact["sha256"] / relative
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("training evidence escapes mirror")
    if path.suffix.lower() not in {".json", ".log", ".txt", ".mp4", ".webm"}:
        raise ValueError("unsupported training evidence type")
    return path


def valid_digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


@lru_cache(maxsize=4096)
def file_digest(path, size, modified_ns, changed_ns):
    # Cache by file identity metadata, never by name alone: mutable reports
    # otherwise appear available while the API correctly rejects old bytes.
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact_available(path, item):
    if not path.is_file():
        return False
    stat = path.stat()
    limit = 30 * 1024 * 1024 if item["kind"] == "video" else 65536 if item["kind"] == "log" else 8 * 1024 * 1024
    return stat.st_size <= limit and (not item.get("sha256") or
        file_digest(path, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns) == item["sha256"])


def scoped_acceptance(report, score, task, mode):
    """Display acceptance only with an independent, scoped acceptance report."""
    allowed_scopes = {"task_behavior", "microduck_project_behaviors_v2", "project_behaviors_v2", "project_tasks_v2"}
    if task == "ground_lower":
        allowed_scopes.add("official_ground_lower")
    project = report.get("acceptance_scope") == "project_tasks_v2"
    project_protocol = "project_qualified_evaluation_v2" if mode == "offline" else "project_qualified_robotd_evaluation_v2"
    if project and not (report.get("protocol") == project_protocol and report.get("schema_version") == 1
                        and report.get("complete") is True
                        and report.get("purpose") == "final_test" and report.get("evaluator_role") == "student"
                        and report.get("independent_final_test") is True
                        and score.get("project_task_accepted") is True
                        and score.get("counts_as_project_task_success") is True
                        and score.get(f"{mode}_accepted") is True):
        return False
    for evidence in (report, score):
        if evidence.get("acceptance_scope", "task_behavior") not in allowed_scopes:
            return False
        # Project qualifications intentionally keep the original-author-skill
        # flag false, including the ground_lower replacement. Only the explicit
        # project evaluator's independent-final-test flags can override it.
        if not project and evidence.get("counts_as_skill_success") is False:
            return False
        if evidence.get("counts_as_student_skill_success") is False:
            return False
        # The project report's aggregate flag is all(tasks); another task's
        # failure must not revoke an independently accepted individual task.
        if evidence.get(f"{mode}_accepted") is False and not (project and evidence is report):
            return False
    count = score.get("episodes", 0)
    if not (isinstance(count, int) and count >= 100 and score.get("distinct_conditions") == count and score.get("strict_rate", 0) >= .9):
        return False
    if report.get("ablation") or score.get("takeovers", 0):
        return False
    if mode == "offline":
        return report.get("mode") == "offline"
    if project:
        from .qualified_realtime import EVIDENCE_PROTOCOL
        latency = score.get("end_to_end_p95_ms")
        return (report.get("mode") == "robotd_bam" and report.get("physics_waits_for_model") is False
                and score.get("daemon_health_passed") is True and score.get("takeovers") == 0
                and report.get("execute_steps") == 5 and report.get("semantic_refresh_steps") == 5
                and score.get("execution_checks_passed") is True and score.get("execution_evidence_protocol") == EVIDENCE_PROTOCOL
                and isinstance(latency, (int, float)) and not isinstance(latency, bool) and 0 < latency <= 100
                and .9 <= score.get("minimum_real_time_factor", 0) <= 1.1)
    latency = report.get("inference_p95_ms")
    return (report.get("mode") in {"live", "realtime"} and report.get("physics_waits_for_model") is False
            and latency is not None and latency <= 100 and score.get("minimum_real_time_factor", 0) >= .9)


def score_view(report, score, task=None):
    return {"strict_rate": score.get("strict_rate"), "successes": score.get("successes"),
            "episodes": score.get("episodes"), "distinct_conditions": score.get("distinct_conditions"),
            "wilson_95": score.get("wilson_95"), "failed_checks": score.get("failed_checks"),
            "source": report.get("source"), "artifact_id": report.get("artifact_id"),
            "protocol": score.get("success_protocol", score.get("protocol", report.get("protocol"))),
            "conditions": report.get("initial_conditions_by_task", {}).get(task, report.get("condition_recipe", report.get("conditions"))),
            "teacher_version": score.get("teacher_version", report.get("teacher_version")),
            "scope": score.get("acceptance_scope", report.get("acceptance_scope", "task_behavior"))}


def task_matrix(snapshot, selected):
    teachers = sorted(snapshot.get("teacher_reports", []), key=lambda row: row.get("modified_at", 0))
    reports = list(selected.get("checkpoints", [])) if selected else []
    if selected:
        reports += selected.get("evaluation_reports", [])
        reports += [selected[key] for key in ("development_report", "pilot_report") if selected.get(key)]
        # Final reports must match a checkpoint belonging to the selected run.
        checkpoints = {row.get("checkpoint_sha256") for row in reports if row.get("checkpoint_sha256")}
        reports += [row for row in snapshot.get("other_reports", []) if row.get("checkpoint_sha256") in checkpoints]
        best = selected.get("best_validation") or {}
        if best.get("checkpoint_sha256"):
            reports = [row for row in reports if row.get("checkpoint_sha256") == best["checkpoint_sha256"]]
            reports.append(best)
    reports.sort(key=lambda row: row.get("modified_at", 0))
    matrix = []
    for task, label in TASK_LABELS.items():
        row = {"task": task, "label": label, "teacher": None, "student": None,
               "offline_accepted": False, "realtime_accepted": False}
        for report in teachers:
            score = report.get("tasks", {}).get(task)
            if score is None and report.get("task") == task:
                score = {"episodes": report.get("validation_episodes"), "strict_rate": report.get("validation_success_rate"),
                         "successes": report.get("validation_successes"), "distinct_conditions": report.get("distinct_validation_conditions"), "wilson_95": report.get("wilson_95")}
            if score is not None:
                row["teacher"] = score_view(report, score, task)
                row["teacher"]["eligible_for_collection"] = score.get("eligible_for_collection", report.get("eligible_for_ground_lower_collection"))
        for report in reports:
            score = report.get("tasks", {}).get(task)
            if score is None:
                continue
            if report.get("evaluator_role") not in (None, "student") or report.get("counts_as_student_skill_success") is False:
                continue
            row["student"] = score_view(report, score, task)
            for mode in ("offline", "realtime"):
                if scoped_acceptance(report, score, task, mode):
                    row[f"{mode}_accepted"] = True
                    row[f"{mode}_evidence"] = score_view(report, score, task)
        matrix.append(row)
    return matrix


def dashboard(directory, run_id=None, now=None):
    directory = Path(directory)
    now = time.time() if now is None else now
    try:
        snapshot = read_json(directory / "latest.json")
        if snapshot and snapshot.get("schema_version") != SCHEMA:
            raise ValueError("unsupported training snapshot schema")
        sync = read_json(directory / "sync.json", {})
    except (OSError, ValueError) as error:
        return {"available": False, "error": str(error), "freshness": {"state": "invalid"}, "matrix": task_matrix({}, None), "runs": []}
    if not snapshot:
        return {"available": False, "error": "尚未同步训练快照", "freshness": {"state": "missing"}, "matrix": task_matrix({}, None), "runs": []}
    generated = snapshot.get("generated_at")
    age = max(0, now - generated) if isinstance(generated, (int, float)) else None
    state = "disconnected" if sync.get("connection_ok") is False else "stale" if age is None or age > 180 else "fresh"
    runs = snapshot.get("runs", [])
    selected_id = run_id or snapshot.get("active_job", {}).get("id")
    selected = next((row for row in runs if row["id"] == selected_id), runs[-1] if runs else None)
    for artifact in snapshot.get("artifacts", []):
        try:
            path = artifact_path(directory, artifact)
            artifact["available"] = artifact_available(path, artifact)
        except (OSError, ValueError, KeyError):
            artifact["available"] = False
        artifact["url"] = f"/api/training/artifact?id={quote(artifact['id'])}" if artifact["available"] else None
        if artifact["available"] and artifact.get("content_addressed"):
            artifact["url"] += "&sha256=" + artifact["sha256"]
    campaigns = snapshot.get("acceptance_campaigns", [])
    registered = max(campaigns, key=lambda c: c.get("completed_at", 0)) if campaigns else None
    registered_matrix = task_matrix({}, {"evaluation_reports": registered["reports"]}) if registered else None
    return {**snapshot, "available": True, "selected_run_id": selected["id"] if selected else None,
            "matrix": task_matrix(snapshot, selected),
            "registered_acceptance": {"campaign": registered, "matrix": registered_matrix} if registered else None,
            "freshness": {"state": state, "age_seconds": age, "generated_at": generated,
                          "last_attempt_at": sync.get("last_attempt_at"), "last_success_at": sync.get("last_success_at"),
                          "error": sync.get("error")}}


def evidence(directory, identifier, sha256=None):
    directory = Path(directory)
    snapshot = read_json(directory / "latest.json", {})
    item = next((item for item in snapshot.get("artifacts", []) if item.get("id") == identifier), None)
    if item is None:
        raise ValueError("unknown training evidence")
    if sha256 is not None:
        if not valid_digest(sha256) or not item.get("content_addressed"):
            raise ValueError("unsupported evidence version")
        item = {**item, "sha256": sha256}
    path = artifact_path(directory, item)
    limit = 30 * 1024 * 1024 if item["kind"] == "video" else 65536 if item["kind"] == "log" else 8 * 1024 * 1024
    if path.stat().st_size > limit:
        raise ValueError("mirrored evidence exceeds size limit")
    data = path.read_bytes()
    if item.get("sha256") and hashlib.sha256(data).hexdigest() != item["sha256"]:
        raise ValueError("evidence is still synchronizing; retry after the next refresh")
    content_type = "application/json" if path.suffix == ".json" else "text/plain; charset=utf-8" if item["kind"] == "log" else mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return data, content_type
