"""Replay mission physics and model claims, including preserved failure prefixes.

No neural model is loaded. Recorded neural actions are reused while rebuilding
their inputs, command admission, physical motion, actor memory and task scores.
This establishes trace integrity, not held-out capability or realtime latency.
"""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np

from .contracts import SkillIntent
from .navigation_control_audit import audit_recording
from .navigation_execution_summary import EXECUTION_SUMMARY_PROTOCOL
from .navigation_hybrid_feedback_audit import FeedbackHybridObserver
from .navigation_mission import MISSION_PROTOCOL, MissionEnv, MissionNavigationPolicy, MissionSpec
from .navigation_mission_selection import SELECTED_GOAL_PROTOCOL, SelectedGoalMissionPolicy
from .navigation_mission_visual_memory import VISUAL_MEMORY_PROTOCOL
from .navigation_mission_timing import primitive_timing_contract
from .navigation_outcomes import read, require, sha
from .navigation_recording_replay import canonical
from .navigation_realtime_labeling import _same_tree
from .navigation_task_switch import _hazards, _motion_state
from .tasks import get_task


class MissionObserver(FeedbackHybridObserver):
    protocol = MISSION_PROTOCOL
    families = ("VLM", "VLM+WAM")
    policy_class = MissionNavigationPolicy

    def policy_options(self, metadata):
        options = {}
        if "execution_summary_protocol" in metadata:
            require(metadata["execution_summary_protocol"] == EXECUTION_SUMMARY_PROTOCOL,
                    "unsupported execution summary protocol")
            options["execution_summary"] = True
        if "primitive_timing_contract" in metadata:
            timing = metadata["primitive_timing_contract"]
            limit = timing["maximum_duration_ms"]
            require(timing == primitive_timing_contract(limit), "unsupported primitive timing contract")
            options["primitive_max_duration_ms"] = limit
        return options

    def __init__(self, episode, requests, identity, mission_id, instruction, prior_claims):
        self.mission_id, self.instruction = mission_id, instruction
        self.prior_claims = deepcopy(prior_claims)
        super().__init__(episode, requests, identity)

    def initialize_session(self):
        self.policy.begin_mission(self.mission_id, self.instruction)
        for claim in self.prior_claims:
            self.policy.mission_memory.append(claim)
        self.policy.reset(self.episode_id)

    def observe_prediction(self, observation, prediction):
        if self.original_generation is None:
            # Session numbers are recorded provenance, not CPU wall time. Use
            # the original session counter so committed claim IDs reconstruct
            # the next stage's exact HTTP context. Each later tick is checked.
            generation = prediction["diagnostics"]["hybrid"]["generation"]
            require(type(generation) is int and generation > 0, "invalid mission session")
            self.policy.generation = generation
            self.prepared = replace(self.prepared, generation=generation)
        super().observe_prediction(observation, prediction)


class SelectedGoalMissionObserver(MissionObserver):
    protocol = SELECTED_GOAL_PROTOCOL
    policy_class = SelectedGoalMissionPolicy

    def policy_options(self, metadata):
        options = super().policy_options(metadata)
        if "visual_memory_protocol" in metadata:
            require(metadata["visual_memory_protocol"] == VISUAL_MEMORY_PROTOCOL,
                    "unsupported mission visual memory protocol")
            options["visual_memory"] = True
        return options


def recorded_segments(folder, report):
    """Index an old unlisted exception prefix only from its original manifest.

    Earlier mission writers saved the shared executor's failed recording but
    did not append its segment to mission.json. Return a separate audit index;
    never rewrite or promote the original incomplete mission.
    """
    folder = Path(folder)
    segments = deepcopy(report["segments"])
    recovered = False
    if report["status"] == "incomplete":
        indexed_failure = bool(segments and (folder / segments[-1]["path"] / "execution_failure.json").is_file())
        if not indexed_failure:
            name = f"step-{len(segments):02d}"
            for item in ("report.json", "execution_failure.json", "recording.npz", "control.json", "predictions.json"):
                key = name + "/" + item
                require(key in report["files"] and sha(folder / key) == report["files"][key],
                        "incomplete mission has no authenticated execution prefix")
            episode = read(folder / name / "report.json")
            target = report["expected_targets"][min(len(segments), len(report["expected_targets"]) - 1)]
            segments.append(dict(path=name, report_sha256=sha(folder / name / "report.json"), claim=None,
                expected_target=target, physical_goal_satisfied=bool(episode["task_success"]),
                hazards=_hazards(episode), claim_target_matches=False))
            recovered = True
        terminal = read(folder / segments[-1]["path"] / "execution_failure.json")
        require(terminal["phase"] == "execution" and terminal["error_type"] == report["error_type"]
                and report["strict_success"] is False and report.get("reason") in (None, "execution_exception"),
                "incomplete mission differs from its recorded execution failure")
    return segments, recovered


def audit_mission(folder, *, requests, expected_report_sha256, environment_factory):
    folder, requests = Path(folder), Path(requests)
    require(sha(folder / "mission.json") == expected_report_sha256, "mission report changed")
    report = read(folder / "mission.json")
    require(report["schema"] == MISSION_PROTOCOL and report["source_sha256"] == sha(
                Path(__file__).with_name("navigation_mission.py"))
            and report["status"] in ("complete", "incomplete") and report["mode"] == "offline",
            "only matching terminal offline mission records can be replayed")
    spec = MissionSpec(report["instruction"], tuple(report["expected_targets"]),
                       report["max_claims"], report["max_ticks_per_claim"])
    for name, digest in report["files"].items():
        path = folder / name
        require(not path.is_symlink() and path.resolve().is_relative_to(folder.resolve())
                and sha(path) == digest, "mission recording changed or escaped")
    segments, recovered_prefix = recorded_segments(folder, report)
    require(0 < len(segments) <= spec.max_claims, "missing or excess mission segments")
    identity = read(requests / "service.json")["identity_sha256"]
    claims, audits, scored = [], [], []
    reason = None
    with environment_factory() as env:
        require(isinstance(env, MissionEnv) and env.mission_instruction == spec.instruction
                and env.profile == "official_bam" and env.scene == report["scene"],
                "replay environment differs from the mission")
        for index, segment in enumerate(segments):
            require(segment["path"] == f"step-{index:02d}", "mission order differs")
            path = folder / segment["path"]
            require(sha(path / "report.json") == segment["report_sha256"], "segment report changed")
            episode = read(path / "report.json")
            target = spec.expected_targets[min(index, len(spec.expected_targets) - 1)]
            task, episode_id = "find_" + target + "_ball", report["mission_id"] + f"/step-{index}"
            require(episode["mode"] == "offline" and episode["seed"] == report["seed"]
                    and episode["task"] == task and episode["evaluator"]["episode_id"] == episode_id
                    and episode["initialization"] == "prepared_physical_state_new_task_clock",
                    "segment identity or initialization differs")
            if index == 0:
                env.render_enabled = False
                env.reset(report["seed"], task, {"episode_id": episode_id})
                for _ in range(report["warmup_ticks"]):
                    env.step(SkillIntent("stand", np.zeros(13, np.float32), env.episode_id, env.tick, 1))
                env.begin_task_at_current_pose()
            else:
                require(reason is None and claims[-1]["tool"] == "complete_subgoal",
                        "mission continued after a terminal declaration or failure")
                before = env.clone_state()
                env.task, env.episode_id = get_task(task), episode_id
                env.begin_task_at_current_pose()
                _same_tree(_motion_state(before), _motion_state(env.clone_state()))
            observer_class = {MISSION_PROTOCOL: MissionObserver,
                              SELECTED_GOAL_PROTOCOL: SelectedGoalMissionObserver}.get(episode["policy"]["protocol"])
            require(observer_class is not None, "unknown mission policy protocol")
            observer = observer_class(episode, requests, identity, report["mission_id"], spec.instruction, claims)
            try:
                if index:
                    boundary = dict(previous_claim=claims[-1], next_episode_id=episode_id,
                        full_motion_state_unchanged=True, truth_feedback_to_policy=False,
                        mission_memory=observer.policy.mission_memory.snapshot())
                    require(boundary == read(folder / f"boundary-{index-1:02d}.json"),
                            "mission boundary or actor memory differs")
                controls, predictions = read(path / "control.json"), read(path / "predictions.json")
                control_audit = audit_recording(path)
                by_source = {}
                for prediction in predictions:
                    by_source.setdefault(prediction["source_tick"], []).append(prediction)
                with np.load(path / "recording.npz", allow_pickle=False) as recording:
                    cameras = {int(t): i for i, t in enumerate(recording["camera_ticks"])}
                    require(len(cameras) == len(recording["rgb"]), "duplicate camera frames")
                    for tick, control in enumerate(controls):
                        require(env.tick == tick and not env.terminated
                                and observer.policy.completion_claim() is None,
                                "actions continued after mission completion")
                        np.testing.assert_array_equal(env.physics.data.qpos, recording["qpos"][tick])
                        env.render_enabled = tick in cameras
                        before = env.observe()
                        if tick in cameras:
                            np.testing.assert_array_equal(before.rgb, recording["rgb"][cameras[tick]])
                        observer.observe_sensors(deepcopy(before), deepcopy(control), deepcopy(by_source.get(tick, [])))
                        for prediction in by_source.get(tick, []):
                            require(prediction["observed_instruction"] == spec.instruction,
                                    "evaluator target replaced the full user instruction")
                            np.testing.assert_array_equal(before.state[:48], prediction["observed_proprio48"])
                            observer.observe_prediction(deepcopy(before), deepcopy(prediction))
                        env.render_enabled = False
                        intent = SkillIntent(control["skill"], np.asarray(control["command"], np.float32),
                                             episode_id, tick, 1)
                        after = env.step(intent)
                        observer.observe_execution(deepcopy(before), deepcopy(after), intent, deepcopy(control))
                    if (path / "execution_failure.json").exists():
                        failure = read(path / "execution_failure.json")
                        require(report["status"] == "incomplete" and index == len(segments) - 1
                                and failure["tick"] == env.tick and env.tick in cameras,
                                "failure has no terminal source image")
                        env.render_enabled = True
                        before = env.observe()
                        np.testing.assert_array_equal(before.rgb, recording["rgb"][cameras[env.tick]])
                        observer.observe_failure(before, failure)
                require(canonical(env.evaluator_state()) == canonical(episode["evaluator"]),
                        "mission physical score differs")
                claim = observer.policy.completion_claim()
                row = dict(path=segment["path"], report_sha256=sha(path / "report.json"), claim=claim,
                           expected_target=target, physical_goal_satisfied=bool(env.evaluator_state()["success"]),
                           hazards=_hazards(episode), claim_target_matches=bool(claim and claim["target"] == target))
                require(row == segment, "model claim or physical result differs")
                sources = observer.finish()
                require(sources["submitted_predictions_verified"] == len(predictions)
                        and sources["execution_feedback_ticks"] == len(controls), "unreplayed mission inputs")
                audits.append(dict(path=segment["path"], control_ticks=len(controls), rgb_frames=len(cameras),
                                   sources=sources, control_audit=control_audit))
                scored.append(row)
                if claim is not None:
                    claims.append(claim)
                if (path / "execution_failure.json").exists():
                    require(claim is None, "failed prefix also claims completion")
                    reason = "execution_exception"
                elif row["hazards"] or claim is None:
                    reason = "execution_hazard" if row["hazards"] else "no_model_completion_claim"
                elif claim["tool"] == "finish_mission":
                    reason = "model_finished_mission"
                elif index + 1 == spec.max_claims:
                    reason = "claim_limit_without_model_finish"
                if reason is not None:
                    require(index == len(segments) - 1, "mission continued after termination")
            finally:
                observer.close()
    strict = bool(reason == "model_finished_mission"
        and [c["target"] for c in claims] == list(spec.expected_targets)
        and all(r["physical_goal_satisfied"] and r["claim_target_matches"] and not r["hazards"] for r in scored))
    expected_reason = "execution_exception" if report["status"] == "incomplete" else report["reason"]
    require(reason == expected_reason and strict == report["strict_success"], "mission score or final memory differs")
    final_memory_recorded = "actor_memory" in report
    if final_memory_recorded:
        require(report["actor_memory"]["claims"] == claims, "mission score or final memory differs")
    else:
        require(recovered_prefix, "indexed mission omitted final actor memory")
    require(all(sha(folder / n) == d for n, d in report["files"].items()), "mission changed during audit")
    return dict(schema=MISSION_PROTOCOL + "_audit", status="complete", segments=audits,
        mission_status=report["status"], scored_segments=scored, recovered_unindexed_prefix=recovered_prefix,
        final_memory_recorded=final_memory_recorded, reconstructed_claims=claims,
        strict_success=strict, reason=reason, physics_and_model_inputs_verified=True,
        model_claims_independently_scored=True, neural_outputs_recomputed=False,
        realtime_validated=False, formal_acceptance=False, hardware_validated=False,
        report_sha256=expected_report_sha256, audit_source_sha256=sha(__file__))
