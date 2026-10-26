"""Operator-owned model registry. Browser clients select IDs, never GPU paths."""
import json
import os
import re
from pathlib import Path
from urllib.parse import urlparse

from .data import file_sha256
from .tasks import TASKS


def _task_spec(row, task):
    if row.get("acceptance_scope") == "project_tasks_v2":
        from .qualified_teachers import training_contracts
        from .contracts import TaskSpec
        contract = training_contracts().get(task)
        if contract is None:
            raise ValueError("task is not part of the registered project behavior suite")
        return TaskSpec(**contract["task_spec"])
    if task not in TASKS:
        raise ValueError("unknown registry task; project tasks require an explicit acceptance scope")
    return TASKS[task]


def _validate_project_registration(row):
    from .qualified_teachers import training_contracts
    digest = row.get("artifact_bundle_sha256", "")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("project registry requires the complete artifact bundle digest")
    expected = training_contracts()
    contracts = row.get("task_contracts", {})
    if set(contracts) != set(row["tasks"]) or any(
            task not in expected or json.dumps(contracts[task], sort_keys=True) != json.dumps(expected[task], sort_keys=True)
            for task in row["tasks"]):
        raise ValueError("project registry task programs or initial conditions changed")


def entries():
    location = os.environ.get("MICRODUCK_MODEL_REGISTRY")
    if not location:
        return []
    payload = json.loads(Path(location).read_text())
    rows = payload["models"]
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("duplicate registered model ID")
    for row in rows:
        if row["embodiment"] not in ("legs", "rollers"):
            raise ValueError("invalid registry embodiment")
        if any(_task_spec(row, task).embodiment != row["embodiment"] for task in row["tasks"]):
            raise ValueError("registry task/embodiment mismatch")
        if row.get("acceptance_scope") == "project_tasks_v2":
            _validate_project_registration(row)
    return rows


def _allows_skill_acceptance(report, score, mode):
    """Only task-behavior reports may grant a skill release.

    Historical reports omit these fields and retain the numerical checks below.
    Explicit report/task denials must not be reconstructed into acceptance from
    a high success rate. Do not read a model manifest's pre-evaluation defaults:
    a checkpoint can later earn independent task-behavior evidence.
    """
    accepted_key = "offline_accepted" if mode == "offline" else "realtime_accepted"
    for evidence in (report, score):
        if evidence.get("acceptance_scope", "task_behavior") != "task_behavior":
            return False
        if evidence.get("counts_as_skill_success") is False:
            return False
        if accepted_key in evidence and evidence[accepted_key] is not True:
            return False
    return True


def evidence(row):
    paths = [(row.get("report_path"), row.get("report_sha256")), (row.get("realtime_report_path"), row.get("realtime_report_sha256"))]
    if not any(path for path, _ in paths):
        return {}
    result = {task: {"offline_accepted": False, "realtime_accepted": False} for task in row["tasks"]}
    for path, digest in paths:
        if not path:
            continue
        if file_sha256(path) != digest:
            raise ValueError("registered acceptance report changed")
        report = json.loads(Path(path).read_text())
        if report["checkpoint_sha256"] != row["checkpoint_sha256"]:
            raise ValueError("acceptance report belongs to a different checkpoint")
        project = row.get("acceptance_scope") == "project_tasks_v2"
        if project:
            _validate_project_registration(row)
            if report.get("artifact_bundle_sha256") != row["artifact_bundle_sha256"]:
                raise ValueError("acceptance report belongs to different statistics, backbone or task programs")
            precision = row.get("precision", "bf16")
            if precision not in ("fp32", "bf16") or report.get("precision") != precision:
                raise ValueError("acceptance report uses a different registered inference precision")
        for task in row["tasks"]:
            score = report.get("tasks", {}).get(task, {})
            passed = score.get("episodes", 0) >= 100 and score.get("distinct_conditions") == score.get("episodes") and score.get("strict_rate", 0) >= .9 and not report.get("ablation")
            passed = passed and (_allows_project_acceptance(row, report, score, task) if project
                                 else _allows_skill_acceptance(report, score, report.get("mode")))
            entry = result[task]
            if report.get("mode") == "offline":
                entry.update(strict_rate=score.get("strict_rate"), episodes=score.get("episodes", 0), wilson_95=score.get("wilson_95"), offline_accepted=passed)
            if report.get("mode") in (("robotd_bam",) if project else ("live",)):
                latency = score.get("end_to_end_p95_ms") if project else report.get("inference_p95_ms")
                timing_ok = latency is not None and latency <= 100 and score.get("minimum_real_time_factor", 0) >= .9
                entry.update(realtime_strict_rate=score.get("strict_rate"), realtime_episodes=score.get("episodes", 0), realtime_wilson_95=score.get("wilson_95"), realtime_accepted=passed and timing_ok and report.get("physics_waits_for_model") is False and score.get("takeovers", 1) == 0)
    return result


def catalog_models():
    return [{**{key: row[key] for key in ("id", "label", "embodiment", "tasks")},
             "acceptance_scope": row.get("acceptance_scope", "task_behavior"),
             "ready": bool(row.get("ready")), "scores": evidence(row), "hardware_unvalidated": True} for row in entries()]


def resolve(model_id, task):
    row = next((row for row in entries() if row["id"] == model_id), None)
    if row is None or not row.get("ready"):
        raise ValueError("模型尚未部署；请选择可用版本或官方对照")
    if task not in row["tasks"] or _task_spec(row, task).embodiment != row["embodiment"]:
        raise ValueError("模型与任务或本体不匹配")
    evidence(row)
    endpoint = row.get("endpoint", "")
    parsed = urlparse(endpoint)
    if parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost", "::1") or parsed.username or parsed.password or parsed.query:
        raise ValueError("registered model service must use a local SSH tunnel")
    from .remote import TinyVLAHTTPPolicy
    probe = TinyVLAHTTPPolicy(endpoint, timeout=5.)
    if probe.manifest.embodiment != row["embodiment"] or probe.manifest.artifacts["checkpoint_sha256"] != row["checkpoint_sha256"] or probe.manifest.observation_profile != row["observation_profile"]:
        raise ValueError("服务当前加载的模型与登记版本不一致")
    precision = row.get("precision", "bf16")
    if precision not in ("fp32", "bf16") or probe.manifest.metadata.get("precision") != precision:
        raise ValueError("服务实际推理精度与登记版本不一致")
    if row.get("acceptance_scope") == "project_tasks_v2":
        from .qualified_eval import validate_qualified_manifest
        validate_qualified_manifest(probe.artifact_manifest, row["tasks"])
        if probe.manifest.artifacts.get("artifact_bundle_sha256") != row["artifact_bundle_sha256"]:
            raise ValueError("服务权重配套的统计量、任务程序或推理设置与登记版本不一致")
        return {"endpoint": endpoint, "model_id": row["id"], "device": "cuda", "precision": row.get("precision", "bf16"),
                "initial_conditions": "qualified_initial_v2", "acceptance_scope": "project_tasks_v2"}
    return {"endpoint": endpoint, "model_id": row["id"], "device": "cuda", "precision": row.get("precision", "bf16"), "initial_conditions": row.get("initial_conditions", "preset_initial_v1")}


def _allows_project_acceptance(row, report, score, task):
    """A project final report may release individual skills; author13 stays separate."""
    mode = report.get("mode")
    if mode not in ("offline", "robotd_bam"):
        return False
    if (report.get("acceptance_scope") != "project_tasks_v2" or report.get("purpose") != "final_test"
            or report.get("schema_version") != 1
            or report.get("evaluator_role") != "student" or report.get("independent_final_test") is not True
            or report.get("complete") is not True or report.get("ablation")
            or report.get("semantic_refresh_steps") != 5 or report.get("execute_steps") != 5):
        return False
    contract = row["task_contracts"][task]
    if json.dumps(report.get("task_contracts", {}).get(task), sort_keys=True) != json.dumps(contract, sort_keys=True):
        return False
    for key, expected in (("acceptance_scope", contract["acceptance_scope"]),
                          ("success_protocol", contract["protocol"]),
                          ("teacher_version", contract["teacher_version"]),
                          ("command_program_sha256", contract["command_program_sha256"])):
        if score.get(key) != expected:
            return False
    if score.get("counts_as_project_task_success") is not True or score.get("project_task_accepted") is not True:
        return False
    if score.get("offline_accepted" if mode == "offline" else "realtime_accepted") is not True:
        return False
    from .qualified_eval import FINAL_SEEDS
    from .artifact_identity import bundle_sha256
    import numpy as np
    episodes = [item for item in report.get("episodes", []) if item.get("task") == task]
    if ([item.get("seed") for item in episodes] != list(FINAL_SEEDS)
            or score.get("episodes") != len(FINAL_SEEDS) or score.get("distinct_conditions") != len(FINAL_SEEDS)):
        return False
    required_checks = _project_check_names(task)
    condition_hashes, successes = set(), 0
    for item in episodes:
        initial = item.get("initial_condition")
        recipe = contract["initial_conditions"]
        rng = np.random.default_rng(item["seed"])
        expected_initial = {"recipe": recipe, "task": task,
            "xy": rng.uniform(-recipe["xy_half_range_m"], recipe["xy_half_range_m"], 2).tolist(),
            "yaw": float(rng.uniform(-recipe["yaw_half_range_rad"], recipe["yaw_half_range_rad"])),
            "joint_offset": rng.uniform(-recipe["joint_half_range_rad"], recipe["joint_half_range_rad"], 14).tolist(),
            "light_scale": float(rng.uniform(1-recipe["light_half_range"], 1+recipe["light_half_range"]))}
        if (not isinstance(initial, dict) or initial.get("recipe") != contract["initial_conditions"]
                or bundle_sha256(initial) != item.get("initial_condition_sha256")
                or bundle_sha256(initial) != bundle_sha256(expected_initial)):
            return False
        condition_hashes.add(item["initial_condition_sha256"])
        result = item.get("task_result", {})
        checks = result.get("checks", {})
        if (result.get("protocol") != contract["protocol"] or not required_checks.issubset(checks)
                or any(not isinstance(value, bool) for value in checks.values())
                or result.get("success") != all(checks.values())):
            return False
        success = all(checks.values()) and not (item.get("recovery_events") or item.get("fault") or item.get("collision"))
        if mode == "robotd_bam":
            from .qualified_realtime import valid_episode_evidence
            success = success and valid_episode_evidence(item.get("realtime_evidence"))
        if item.get("success") is not success:
            return False
        successes += success
    if (len(condition_hashes) != 100 or score.get("successes") != successes
            or score.get("strict_rate") != successes / 100):
        return False
    if mode == "robotd_bam":
        from .qualified_realtime import EVIDENCE_PROTOCOL
        latency = score.get("end_to_end_p95_ms")
        return (report.get("protocol") == "project_qualified_robotd_evaluation_v2"
                and report.get("physics_waits_for_model") is False
                and score.get("daemon_health_passed") is True and score.get("takeovers") == 0
                and score.get("execution_evidence_protocol") == EVIDENCE_PROTOCOL
                and score.get("execution_checks_passed") is True
                and isinstance(latency, (int, float)) and not isinstance(latency, bool) and 0 < latency <= 100
                and .9 <= score.get("minimum_real_time_factor", 0) <= 1.1
                and all(valid_episode_evidence(item.get("realtime_evidence")) for item in episodes))
    return report.get("protocol") == "project_qualified_evaluation_v2"


def _project_check_names(task):
    """Read scorer predicate names without constructing a simulator or running a policy."""
    import numpy as np
    if task == "ground_lower":
        from .ground_lower import score_ground_lower
        trace = np.zeros((2, 13))
        trace[1, 0] = .02
        return set(score_ground_lower(trace)["checks"])
    from .tasks import strict_preset_result
    trace = np.zeros((2, 12))
    trace[1, 0] = 5.
    checks = set(strict_preset_result(task, trace, set(), collision=False, recoveries=0)["checks"])
    checks.update({"initialization_valid", "no_fault", "no_unintended_fall"})
    if task == "sit_walk":
        checks.update({"walk_phase_started", "rise_complete_before_walk", "forward_not_path_only", "no_backward_excursion"})
    if task in {"roller_crouch", "roller_slalom"}:
        checks.add("no_unintended_tip")
    if task == "roller_glide":
        checks.update({"forward_not_path_only", "no_backward_excursion"})
    return checks
