"""Audit saved episodes against versioned gates, without inventing missing evidence.

Run ``python -m lightx2v_microduck.acceptance --runs DIR --output FILE``.
Every count is recomputed from hash-checked episode trajectories; aggregate
success_rate, primary_success, and author_success are never acceptance inputs.

For visual holdout evidence, add an object to evaluation.json::

    "heldout_evidence": {
      "schema_version": 1,
      "checkpoint_sha256": "<same checkpoint as the policy manifest>",
      "exhaustive_for_training_and_tuning": true,
      "dataset_indices": [{"path": "../../data/index.json", "sha256": "..."}]
    }

Paths resolve relative to evaluation.json. Each index uses the collection
schema (episodes with split, layout_id, geometry_hash). Train AND validation
layouts are excluded. This audits the declared history; it cannot prove that
an undeclared dataset or manual tuning run never existed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from importlib.resources import files
from pathlib import Path


def _digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _json(path):
    def reject(value):
        raise ValueError(f"nonfinite JSON constant {value}")

    return json.loads(Path(path).read_text(), parse_constant=reject)


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def load_protocol():
    payload = files("lightx2v_microduck").joinpath("resources/acceptance_v1.json").read_bytes()
    return json.loads(payload), hashlib.sha256(payload).hexdigest()


def _state(count, minimum, successes, threshold, issues):
    if count == 0:
        return "incomplete" if issues else "not_evaluated"
    if issues or count < minimum:
        return "incomplete"
    return "passed" if successes / count >= threshold else "failed"


def _holdout(index_path, evaluation, checkpoint):
    evidence = evaluation.get("heldout_evidence")
    if not isinstance(evidence, dict):
        return None, "missing_heldout_evidence"
    if evidence.get("schema_version") != 1 or not _sha(checkpoint) or evidence.get("checkpoint_sha256") != checkpoint:
        return None, "heldout_checkpoint_or_schema_mismatch"
    if evidence.get("exhaustive_for_training_and_tuning") is not True:
        return None, "training_and_tuning_history_not_declared_complete"
    references = evidence.get("dataset_indices")
    if not isinstance(references, list) or not references:
        return None, "missing_training_dataset_indices"
    layouts, geometries, receipts = set(), set(), []
    try:
        for reference in references:
            path = (index_path.parent / reference["path"]).resolve()
            if not _sha(reference["sha256"]) or _digest(path) != reference["sha256"]:
                raise ValueError("training index digest mismatch")
            dataset = _json(path)
            episodes = dataset["episodes"]
            if not isinstance(episodes, list) or not episodes:
                raise ValueError("training index has no episodes")
            training_count = 0
            for episode in episodes:
                if episode["split"] not in ("train", "val", "test"):
                    raise ValueError("unknown dataset split")
                if episode["split"] in ("train", "val"):
                    if not episode.get("layout_id") or not _sha(episode.get("geometry_hash")):
                        raise ValueError("training/tuning layout identity is missing")
                    layouts.add(episode["layout_id"])
                    geometries.add(episode["geometry_hash"])
                    training_count += 1
            if training_count == 0:
                raise ValueError("index has no training or validation layout history")
            receipts.append({"path": str(path), "sha256": reference["sha256"]})
    except (OSError, KeyError, TypeError, ValueError) as exc:
        return None, f"invalid_heldout_evidence:{exc}"
    return {"layouts": layouts, "geometries": geometries, "indices": receipts}, None


def _role(episode, protocol):
    policy = episode["policy"]
    metadata = policy.get("metadata", {})
    validation = episode.get("policy_environment_validation", {})

    def field(name, default=None):
        return episode.get(name, validation.get(name, default))

    reasons = []
    observation_profile = str(policy.get("observation_profile", "")).lower()
    sensors = set(field("policy_sensors", []))
    if (
        policy.get("uses_privileged_state") is True
        or field("uses_privileged_state") is True
        or metadata.get("teacher_only")
        or metadata.get("oracle_recovery")
        or "simulator_truth" in sensors
        or any(word in observation_profile for word in ("oracle", "privileged"))
        or policy.get("policy_id") in ("official", "privileged")
    ):
        return "teacher_or_oracle", ["teacher_or_privileged_state"]
    if field("extra_sensors", []) or sensors - {"rgb", "proprioception"} or "tof" in observation_profile:
        return "extra_sensor", ["extra_sensor_input"]
    if policy.get("uses_privileged_state") is not False or field("uses_privileged_state") is not False:
        reasons.append("privileged_state_absence_not_declared")
    if field("sensor_contract_declared") is not True:
        reasons.append("missing_sensor_contract")
    if field("eligible_for_primary_benchmark") is not True:
        reasons.append("primary_sensor_eligibility_not_verified")
    if not {"rgb", "proprioception"}.issubset(sensors):
        reasons.append("rgb_proprioception_inputs_not_declared")
    availability = episode.get("sensor_availability", {})
    if availability.get("rgb") is not True or availability.get("proprioception") is not True:
        reasons.append("required_sensor_availability_missing")
    if episode.get("config", {}).get("ablation"):
        reasons.append("ablation_run")
    if episode.get("profile") not in protocol["primary_profiles"]:
        reasons.append("non_primary_simulation_profile")
    return ("unverified" if reasons else "primary_rgb_proprio"), reasons


def _episode_checks(episode, path, protocol):
    issues = []
    final = episode.get("final", {})
    if episode.get("schema_version") != 1 or episode.get("evidence") != "simulation_evaluation":
        issues.append("unsupported_episode_evidence_schema")
    if type(episode.get("success")) is not bool or type(final.get("strict_success")) is not bool:
        issues.append("missing_boolean_strict_outcome")
    if final.get("terminated") is not True:
        issues.append("partial_episode")
    if not _sha(final.get("tasks_source_sha256")) or not _sha(final.get("evaluator_source_sha256")):
        issues.append("missing_evaluator_source_identity")
    try:
        if not _sha(episode.get("trajectory_sha256")) or _digest(path.parent / "trajectory.npz") != episode["trajectory_sha256"]:
            issues.append("trajectory_hash_mismatch")
    except OSError:
        issues.append("missing_trajectory")
    task = episode.get("task")
    strict = final.get("strict_preset_result")
    if task in protocol["skills13"]["legs"] + protocol["skills13"]["rollers"]:
        if not isinstance(strict, dict) or strict.get("protocol") != protocol["skills13"]["success_protocol"]:
            issues.append("missing_strict_preset_protocol")
        elif not isinstance(strict.get("checks"), dict) or not strict["checks"] or any(type(value) is not bool for value in strict["checks"].values()):
            issues.append("missing_strict_predicates")
        elif strict.get("success") is not all(strict["checks"].values()) or strict.get("success") is not final.get("strict_success"):
            issues.append("contradictory_strict_predicates")
    elif task in protocol["visual_goal"]["tasks"]:
        if final.get("success_protocol") != protocol["visual_goal"]["success_protocol"]:
            issues.append("unsupported_visual_success_protocol")
        if type(final.get("ever_fallen")) is not bool:
            issues.append("missing_visual_fall_evidence")
    for key in ("collision", "recovery_active"):
        if type(final.get(key)) is not bool:
            issues.append(f"missing_{key}_evidence")
    recovery = final.get("recovery_events")
    if type(recovery) is not int or recovery < 0:
        issues.append("missing_recovery_event_count")
    intervention = bool(final.get("recovery_active") or (isinstance(recovery, int) and recovery > 0) or final.get("interventions") or episode.get("interventions"))
    visual_fall = task in protocol["visual_goal"]["tasks"] and final.get("ever_fallen")
    success = bool(
        episode.get("success") is True and final.get("strict_success") is True and not intervention and not visual_fall and not final.get("collision") and not final.get("fault") and not issues
    )
    reasons = (
        []
        if success
        else [
            "intervention"
            if intervention
            else "collision"
            if final.get("collision")
            else "fault"
            if final.get("fault")
            else "fall"
            if visual_fall
            else episode.get("termination_reason") or "strict_failure"
        ]
    )
    if strict and isinstance(strict, dict):
        reasons.extend(f"predicate:{key}" for key, value in strict.get("checks", {}).items() if value is False)
    return issues, success, sorted(set(reasons))


def _suite_score(rows, tasks, *, minimum, threshold, common_issues):
    scores = {}
    for task in tasks:
        records = [row for row in rows if row["task"] == task]
        problems = sorted(set(common_issues + [issue for row in records for issue in row["issues"]]))
        successes = sum(row["success"] for row in records)
        seeds = len({row["seed"] for row in records})
        scores[task] = {
            "status": _state(seeds, minimum, successes, threshold, problems),
            "count": len(records),
            "unique_seeds": seeds,
            "successes": successes,
            "success_rate": successes / len(records) if records else None,
            "failure_breakdown": dict(Counter(reason for row in records for reason in row["failure_reasons"])),
            "issues": problems,
        }
    states = [score["status"] for score in scores.values()]
    status = "passed" if all(state == "passed" for state in states) else "failed" if "failed" in states else "incomplete" if any(state != "not_evaluated" for state in states) else "not_evaluated"
    return {"status": status, "tasks": scores}


def audit_runs(runs):
    """Read-only audit. Missing/invalid receipts remain visible and never pass."""
    root = Path(runs).resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    protocol, protocol_hash = load_protocol()
    groups, errors, sources, holdout_cache = {}, [], [], {}
    candidates = sorted(root.rglob("evaluation.json"))
    # An interrupted evaluate() writes progress.json but never evaluation.json.
    candidates += sorted(path for path in root.rglob("progress.json") if not (path.parent / "evaluation.json").exists())
    for index_path in candidates:
        try:
            evaluation = _json(index_path)
            records = evaluation["episodes"]
            if not isinstance(records, list):
                raise ValueError("episodes must be an array")
            config = evaluation.get("config", {})
            complete = index_path.name == "evaluation.json" and evaluation.get("complete") is True
            run_issues = []
            if not complete:
                run_issues.append("partial_evaluation")
            if complete:
                tasks, seeds = config.get("tasks", []), config.get("seeds", [])
                if evaluation.get("schema_version") != 1 or not tasks or not seeds or any(type(seed) is not int for seed in seeds) or len(set(tasks)) != len(tasks) or len(set(seeds)) != len(seeds):
                    run_issues.append("invalid_evaluation_config")
                expected = {(task, seed) for task in tasks for seed in seeds}
                actual = [(record.get("task"), record.get("seed")) for record in records]
                if set(actual) != expected or len(actual) != len(expected) or evaluation.get("count") != len(records):
                    run_issues.append("incomplete_or_duplicate_evaluation_grid")
            sources.append({"path": str(index_path), "sha256": _digest(index_path), "complete": complete, "issues": run_issues})
            seen_groups = set()
            for reference in records:
                try:
                    path = (index_path.parent / reference["path"]).resolve()
                    if not path.is_relative_to(index_path.parent.resolve()) or path.name != "episode.json":
                        raise ValueError("episode path escapes evaluation directory or is not episode.json")
                    episode = _json(path)
                    if not isinstance(episode, dict) or not isinstance(episode.get("final"), dict) or not isinstance(episode.get("config"), dict):
                        raise ValueError("episode, final and config must be objects")
                    task, seed = episode["task"], episode["seed"]
                    if type(seed) is not int or task != reference["task"] or seed != reference["seed"]:
                        raise ValueError("episode task/seed does not match evaluation index")
                    policy = episode["policy"]
                    if not isinstance(policy, dict) or not isinstance(policy.get("policy_id"), str) or policy.get("embodiment") not in ("legs", "rollers"):
                        raise ValueError("invalid policy manifest")
                    if reference.get("policy") != policy["policy_id"]:
                        raise ValueError("indexed policy does not match episode manifest")
                    final = episode.get("final", {})
                    role, role_reasons = _role(episode, protocol)
                    identity = {
                        "policy": policy,
                        "profile": episode.get("profile"),
                        "ablation": episode.get("config", {}).get("ablation"),
                        "execution_prefix": episode["config"].get("execution_prefix"),
                        "max_steps": episode["config"].get("max_steps"),
                        "role": role,
                        "tasks_source_sha256": final.get("tasks_source_sha256"),
                        "evaluator_source_sha256": final.get("evaluator_source_sha256"),
                    }
                    identifier = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
                    group = groups.setdefault(
                        identifier, {"id": identifier, **identity, "eligibility_reasons": role_reasons, "rows": [], "issues": [], "seen": {}, "sources": set(), "holdout_indices": {}}
                    )
                    seen_groups.add(identifier)
                    group["sources"].add(str(index_path))
                    issues, success, failure_reasons = _episode_checks(episode, path, protocol)
                    issues.extend(run_issues)
                    key = (task, seed)
                    episode_sha = _digest(path)
                    if key in group["seen"]:
                        if group["seen"][key] != episode_sha:
                            group["issues"].append(f"conflicting_repeated_task_seed:{task}:{seed}")
                        # Copying an identical receipt cannot increase sample size.
                        continue
                    group["seen"][key] = episode_sha
                    row = {
                        "task": task,
                        "seed": seed,
                        "success": success,
                        "failure_reasons": failure_reasons,
                        "issues": issues,
                        "path": str(path),
                        "sha256": episode_sha,
                        "layout_id": final.get("scene", {}).get("layout_id"),
                        "geometry_hash": final.get("scene", {}).get("geometry_hash"),
                    }
                    if task in protocol["visual_goal"]["tasks"]:
                        checkpoint = policy.get("artifacts", {}).get("checkpoint_sha256")
                        cache_key = (str(index_path), checkpoint)
                        if cache_key not in holdout_cache:
                            holdout_cache[cache_key] = _holdout(index_path, evaluation, checkpoint)
                        heldout, problem = holdout_cache[cache_key]
                        if problem:
                            row["issues"].append(problem)
                        elif not row["layout_id"] or not _sha(row["geometry_hash"]):
                            row["issues"].append("missing_test_layout_identity")
                        elif row["layout_id"] in heldout["layouts"] or row["geometry_hash"] in heldout["geometries"]:
                            row["issues"].append("test_layout_seen_in_training_or_tuning")
                        else:
                            for entry in heldout["indices"]:
                                group["holdout_indices"][entry["sha256"]] = entry
                    group["rows"].append(row)
                except (OSError, KeyError, TypeError, ValueError, AttributeError) as exc:
                    errors.append({"path": str(index_path), "episode": reference.get("path") if isinstance(reference, dict) else None, "error": str(exc)})
                    run_issues.append("unreadable_or_invalid_episode")
            for identifier in seen_groups:
                groups[identifier]["issues"].extend(run_issues)
        except (OSError, KeyError, TypeError, ValueError) as exc:
            errors.append({"path": str(index_path), "error": str(exc)})
    output_groups = []
    for group in groups.values():
        rows = group.pop("rows")
        group.pop("seen")
        group["sources"] = sorted(group["sources"])
        group["holdout_indices"] = list(group["holdout_indices"].values())
        group["issues"] = sorted(set(group["issues"]))
        policy = group["policy"]
        native_target = any(policy["policy_id"].startswith(prefix) for prefix in protocol["target_policy_prefixes"])
        metadata = policy.get("metadata", {})
        variant = metadata.get("model_variant")
        artifact_verified = (
            metadata.get("checkpoint_verified") is True
            and metadata.get("engine") == "lightx2v-native-tinyvla"
            and isinstance(variant, str)
            and bool(variant)
            and policy["policy_id"].endswith(":" + variant)
        )
        group["target_model"] = native_target
        group["primary_gate_eligible"] = group["role"] == "primary_rgb_proprio" and native_target and artifact_verified and _sha(policy.get("artifacts", {}).get("checkpoint_sha256"))
        if not native_target:
            group["eligibility_reasons"] += ["comparison_baseline_not_lightx2v_vla_target"]
        if not _sha(policy.get("artifacts", {}).get("checkpoint_sha256")):
            group["eligibility_reasons"] += ["missing_checkpoint_identity"]
        if native_target and not artifact_verified:
            group["eligibility_reasons"] += ["missing_verified_native_model_variant"]
        common = group["issues"]
        for embodiment in ("legs", "rollers"):
            group[f"skills_{embodiment}"] = _suite_score(
                rows,
                protocol["skills13"][embodiment],
                minimum=protocol["skills13"]["minimum_unique_seeds_per_task"],
                threshold=protocol["skills13"]["minimum_success_rate_per_task"],
                common_issues=common,
            )
        visual = protocol["visual_goal"]
        visual_rows = [row for row in rows if row["task"] in visual["tasks"]]
        score = _suite_score(visual_rows, visual["tasks"], minimum=1, threshold=0.0, common_issues=common)
        layouts = {row["layout_id"] for row in visual_rows if row["layout_id"] and not row["issues"]}
        geometries = {row["geometry_hash"] for row in visual_rows if _sha(row["geometry_hash"]) and not row["issues"]}
        visual_issues = common + [issue for row in visual_rows for issue in row["issues"]]
        if visual_rows and any(not any(row["task"] == task for row in visual_rows) for task in visual["tasks"]):
            visual_issues.append("missing_visual_task_coverage")
        for task in visual["tasks"]:
            task_rows = [row for row in visual_rows if row["task"] == task]
            unique = len({row["geometry_hash"] for row in task_rows if _sha(row["geometry_hash"]) and not row["issues"]})
            score["tasks"][task]["unique_unseen_layouts"] = unique
            # Per-task rates are descriptive; the v1 formal rate gate is pooled.
            score["tasks"][task]["status"] = "incomplete" if score["tasks"][task]["issues"] else "evaluated" if task_rows else "not_evaluated"
            if unique < visual["require_min_layouts_per_task"]:
                visual_issues.append(f"insufficient_layouts_for_task:{task}")
            seen_geometry = [row["geometry_hash"] for row in task_rows if _sha(row["geometry_hash"])]
            if len(seen_geometry) != len(set(seen_geometry)):
                visual_issues.append(f"repeated_geometry_within_task:{task}")
        successes = sum(row["success"] for row in visual_rows)
        independent = min(len(layouts), len(geometries))
        if visual_rows and independent < visual["minimum_unique_unseen_layouts"]:
            visual_issues.append("insufficient_independent_unseen_layouts")
        score.update(
            {
                "status": _state(len(visual_rows), 1, successes, visual["minimum_overall_success_rate"], visual_issues),
                "count": len(visual_rows),
                "successes": successes,
                "success_rate": successes / len(visual_rows) if visual_rows else None,
                "unique_unseen_layouts": independent,
                "unique_layout_ids": len(layouts),
                "unique_geometry_hashes": len(geometries),
                "issues": sorted(set(visual_issues)),
            }
        )
        group["visual_goal"] = score
        group["episode_receipts"] = [{key: row[key] for key in ("path", "sha256", "task", "seed")} for row in rows]
        output_groups.append(group)

    def gate(name, embodiment=None):
        entries = [group for group in output_groups if group["primary_gate_eligible"] and (embodiment is None or group["policy"]["embodiment"] == embodiment)]
        rejected = [
            group
            for group in output_groups
            if group["target_model"]
            and not group["primary_gate_eligible"]
            and group["role"] not in ("teacher_or_oracle", "extra_sensor")
            and (embodiment is None or group["policy"]["embodiment"] == embodiment)
            and any(task["count"] for task in group[name]["tasks"].values())
        ]
        states = [group[name]["status"] for group in entries]
        state = "passed" if "passed" in states else "failed" if "failed" in states else "incomplete" if "incomplete" in states or rejected else "not_evaluated"
        return {
            "status": state,
            "candidates": [{"group": group["id"], "status": group[name]["status"]} for group in entries],
            "rejected_candidates": [{"group": group["id"], "reasons": group["eligibility_reasons"]} for group in rejected],
        }

    legs, rollers, visual_gate = gate("skills_legs", "legs"), gate("skills_rollers", "rollers"), gate("visual_goal", "legs")
    skills_state = (
        "passed"
        if legs["status"] == rollers["status"] == "passed"
        else "failed"
        if "failed" in (legs["status"], rollers["status"])
        else "incomplete"
        if any(state != "not_evaluated" for state in (legs["status"], rollers["status"]))
        else "not_evaluated"
    )
    behavior_states = (skills_state, visual_gate["status"])
    behavior = (
        "passed"
        if all(state == "passed" for state in behavior_states)
        else "failed"
        if "failed" in behavior_states
        else "incomplete"
        if errors or any(state != "not_evaluated" for state in behavior_states)
        else "not_evaluated"
    )
    return {
        "schema_version": 1,
        "protocol": protocol["protocol"],
        "protocol_sha256": protocol_hash,
        "requirements": protocol,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runs": str(root),
        "sources": sources,
        "errors": errors,
        "groups": output_groups,
        "simulation_behavior_status": behavior,
        "gates": {
            "skills13": {"status": skills_state, "legs10": legs, "rollers3": rollers},
            "visual_goal": visual_gate,
            **{key: {"status": "pending", "reason": reason} for key, reason in protocol["external_evidence"].items()},
        },
        "overall_status": "failed" if behavior == "failed" else "incomplete" if behavior != "not_evaluated" or errors else "not_evaluated",
        "project_completion": {
            "status": "pending",
            "required_model": "native_lightx2v_tinyvla_with_verified_checkpoint_and_model_variant",
            "native_behavior_status": behavior,
            "pending_requirements": (["native_vla_training_or_behavior_evaluation"] if behavior != "passed" else []) + ["target_gpu_performance", "real_hardware_validation"],
        },
        "hardware_validated": False,
        "limits": [
            "Hash receipts establish file consistency, not authenticity of the simulator implementation or undeclared training history.",
            "Separate checkpoint/profile/ablation/evaluator identities are never pooled; legs and rollers may use separately accepted checkpoints.",
            "Historical reports missing sensor/protocol/holdout evidence remain incomplete. External performance and hardware evidence is pending.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--require-pass", action="store_true", help="exit 2 unless all simulation behavior gates pass; external gates remain pending")
    args = parser.parse_args()
    report = audit_runs(args.runs)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "output": str(output.resolve()),
                "simulation_behavior_status": report["simulation_behavior_status"],
                "overall_status": report["overall_status"],
                "groups": len(report["groups"]),
                "errors": len(report["errors"]),
            }
        )
    )
    if args.require_pass and report["simulation_behavior_status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
