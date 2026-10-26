"""Recompute full retarget-teacher qualification from recorded evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assess(directory, protocol_path, expected_protocol_sha256, launch_path, source_manifest):
    if digest(protocol_path) != expected_protocol_sha256:
        raise ValueError("qualification protocol changed after declaration")
    protocol = json.loads(protocol_path.read_text())
    launch = json.loads(launch_path.read_text())
    manifest = json.loads(source_manifest.read_text())
    report = json.loads((directory / "report.json").read_text())
    if launch["protocol_sha256"] != expected_protocol_sha256 or launch["source_archive_sha256"] != manifest["sha256"]:
        raise ValueError("launch does not match the frozen protocol and source")
    if (report["status"] != "complete" or launch["status"] != "complete" or launch["exit_code"] != 0):
        raise ValueError("complete finite collection required before qualification")
    if (report["source_index_sha256"] != protocol["source_index_sha256"]
            or report["script_sha256"] != manifest["files"]["scripts/microduck/fastwam_instruction_retarget.py"]
            or report["anchor"] != protocol["anchor"]
            or report.get("teacher_variant", "coverage") != protocol.get("teacher_variant", "coverage")
            or report["obstacle_aware_teacher"] != protocol["obstacle_aware_teacher"]):
        raise ValueError("collection differs from the frozen source, teacher or anchor")
    requirements = protocol["requirements"]
    if (requirements["minimum_changed_instruction_episodes_per_target"] < 100
            or requirements["minimum_strict_teacher_success_rate_per_target"] < .95
            or requirements["minimum_changed_instruction_success_rate_per_target"] < .95):
        raise ValueError("protocol weakens the agreed teacher qualification gate")
    expected = {(f"find_{source}_ball", seed) for seed in protocol["seeds"] for source in protocol["source_colors"]}
    actual = [(case["source_task"], case["seed"]) for case in report["cases"]]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("missing, extra or duplicate source states in fixed cohort")
    tasks = {f"find_{color}_ball" for color in protocol["target_colors"]}
    counters = {task: {"total": 0, "successes": 0, "changed_total": 0, "changed_successes": 0,
                       "collisions": 0, "falls": 0, "recoveries": 0} for task in tasks}
    different_action_groups = 0
    for case in report["cases"]:
        branch_tasks = [b["task"] for b in case["branches"]]
        if len(branch_tasks) != len(set(branch_tasks)) or set(branch_tasks) != tasks:
            raise ValueError("each source state must have every declared color instruction")
        if max(case[k] for k in ("replay_state_max_abs", "replay_actor_max_abs", "anchor_state_max_abs")) > requirements["source_replay_max_abs_tolerance"]:
            raise ValueError("source state replay differs from its qualified demonstration")
        input_hashes, first_actions = set(), set()
        for entry in case["branches"]:
            branch_path = (directory / entry["path"]).resolve()
            if not branch_path.is_relative_to(directory.resolve()):
                raise ValueError("branch path escapes collection")
            branch = json.loads((branch_path / "report.json").read_text())
            if (branch["task"] != entry["task"] or branch["seed"] != case["seed"]
                    or branch["original_task"] != case["source_task"]
                    or branch["original_source_tick"] != case["source_tick"]
                    or branch["source_episode_sha256"] != case["source_episode_sha256"]
                    or branch["split"] != "train" or not branch["teacher_history_reset"]
                    or branch["initial_teacher_target_previously_observed"]
                    or branch["physics_and_controller_reset"]
                    or branch["branch_kind"] != "counterfactual_instruction_retargeting"
                    or not branch["branch_task_clock_restarted"]
                    or branch["obstacle_aware_teacher"] != protocol["obstacle_aware_teacher"]
                    or branch.get("teacher_variant", "coverage") != protocol.get("teacher_variant", "coverage")
                    or branch["eligible_training_supervision"]):
                raise ValueError("branch provenance or diagnostic-only boundary changed")
            if branch.get("teacher_variant") == "initial_scan" and branch["teacher_experiment_source_sha256"] != manifest["files"]["embodied/microduck/src/lightx2v_microduck/navigation_teacher_experiments.py"]:
                raise ValueError("initial-scan teacher differs from its declared source")
            data_path = branch_path / "navigation.npz"
            if digest(data_path) != branch["navigation_sha256"]:
                raise ValueError("branch tensors changed after collection")
            with np.load(data_path, allow_pickle=False) as data:
                n = len(data["action"])
                if (data["rgb"].shape != (n + 1, 256, 256, 3) or data["rgb"].dtype != np.uint8
                        or data["proprio"].shape != (n + 1, 48) or data["action"].shape != (n, 5)
                        or n < 1 or not np.isfinite(data["proprio"]).all() or not np.isfinite(data["action"]).all()
                        or not np.array_equal(data["ticks"], np.arange(n + 1) * 5)
                        or not np.array_equal(data["action"], data["continuation_action"][:n])):
                    raise ValueError("invalid branch sensor/action alignment")
                pair = tuple(hashlib.sha256(data[k][0].tobytes()).hexdigest() for k in ("rgb", "proprio"))
                if pair != (branch["first_rgb_sha256"], branch["first_proprio_sha256"]):
                    raise ValueError("first sensor frame differs from recorded snapshot")
                input_hashes.add(pair)
                first_actions.add(tuple(data["action"][0].tolist()))
            state = branch["result"]
            unsafe = any(d.get("planner_phase", d.get("phase")) in ("recover", "no_path") for d in branch["teacher_diagnostics"])
            success = bool(state["success"] and not state["collision"] and not state["ever_fallen"]
                           and state["recovery_events"] == 0 and not unsafe)
            if success != branch["teacher_continuation_success"] or success != entry["teacher_continuation_success"]:
                raise ValueError("reported teacher score disagrees with evaluator evidence")
            count = counters[entry["task"]]
            changed = case["source_task"] != entry["task"]
            count["total"] += 1
            count["successes"] += int(success)
            count["changed_total"] += int(changed)
            count["changed_successes"] += int(changed and success)
            count["collisions"] += int(state["collision"])
            count["falls"] += int(state["ever_fallen"])
            count["recoveries"] += int(state["recovery_events"] > 0)
        if len(input_hashes) != 1:
            raise ValueError("color instructions did not receive identical first RGB and proprio")
        different_action_groups += int(len(first_actions) > 1)
    for count in counters.values():
        count["strict_success_rate"] = count["successes"] / count["total"]
        count["changed_instruction_success_rate"] = count["changed_successes"] / count["changed_total"]
        count["qualified"] = bool(count["changed_total"] >= requirements["minimum_changed_instruction_episodes_per_target"]
                                  and count["strict_success_rate"] >= requirements["minimum_strict_teacher_success_rate_per_target"]
                                  and count["changed_instruction_success_rate"] >= requirements["minimum_changed_instruction_success_rate_per_target"])
    return {"status": "complete", "teacher_qualified": all(c["qualified"] for c in counters.values()),
            "source_states": len(actual), "instruction_discriminating_source_states": different_action_groups,
            "per_target": counters, "protocol_sha256": expected_protocol_sha256,
            "collection_report_sha256": digest(directory / "report.json"),
            "source_archive_sha256": manifest["sha256"], "scope": protocol["scope"],
            "data_released_for_training": False, "training_started": False,
            "not_learned_model_acceptance": True, "hardware_validated": False}
