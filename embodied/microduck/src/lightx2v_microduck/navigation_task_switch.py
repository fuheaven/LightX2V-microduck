"""Task replacement at a retained physical state, with a fresh policy session.

This diagnostic is explicitly offline: physics waits for inference and between
the two recorded segments. It tests a new user instruction during motion, not
autonomous subgoal discovery, retained task memory, or realtime interruption.
The old goal is superseded at a fixed tick, never after a privileged success
signal. Both segments use the existing official actor and navigation executor.
"""

import hashlib
from pathlib import Path
import time

import numpy as np

from .contracts import SkillIntent
from .navigation import DECISION_TICKS
from .navigation_control_audit import audit_recording
from .navigation_eval import run_navigation_episode
from .navigation_outcomes import read, require, sha
from .navigation_recording_replay import canonical
from .navigation_realtime_labeling import _same_tree
from .tasks import get_task
from .wam_data import write_json


PROTOCOL = "microduck_offline_task_replacement_v1"
GOALS = ("find_red_ball", "find_blue_ball", "find_yellow_ball")


def _motion_state(snapshot):
    # Full MjData includes contacts/solver warm starts; physics also contains
    # BAM and ToF state. Keep the official actor/RNG and actuator history too.
    return {
        **{key: snapshot[key] for key in (
            "profile", "scene", "signature", "action_delay_ticks",
            "camera_offsamples", "physics", "policies", "rng")},
        "control": {key: snapshot["env"][key] for key in (
            "last_action", "applied_action", "_delay", "_filtered_action")},
    }


def _hazards(report):
    final = report["evaluator"]
    return bool(report.get("cancelled") or report.get("execution_error")
                or report["fallback_control_ticks"] or report["rejected_plans"]
                or final["collision"] or final["ever_fallen"]
                or final.get("recovery_events", 0) or final.get("fault"))


def replace_goal(env, task, episode_id):
    """Reset only task/session accounting; do not reset/restore the robot."""
    if (env.profile != "official_bam" or env.daemon_backend is not None
            or env.task.task_id not in GOALS or task not in GOALS
            or task == env.task.task_id or env.terminated or env._recovery_active
            or env.tick % DECISION_TICKS or not isinstance(episode_id, str)
            or not episode_id or episode_id == env.episode_id):
        raise ValueError("task replacement requires a live aligned navigation state and a new session")
    before = env.clone_state()
    original = dict(task=env.task.task_id, episode_id=env.episode_id,
                    tick=env.tick, sim_time_s=env.sim_time_s)
    env.task = get_task(task)
    env.episode_id = episode_id
    env.begin_task_at_current_pose()
    after = env.clone_state()
    _same_tree(_motion_state(after), _motion_state(before))
    # Arrays in the evidence are evaluator-only, never planner inputs.
    return dict(protocol=PROTOCOL, previous=original, task=task,
                episode_id=episode_id, tick=env.tick, sim_time_s=env.sim_time_s,
                physical_reset=False, full_motion_state_unchanged=True,
                comparison="MjData+BAM+ToF+official_policy+RNG+raw_filtered_delayed_actions",
                previous_task_superseded=True, task_clock_restarted=True,
                policy_session="reset_by_shared_executor_before_new_prediction",
                old_action_buffer="discarded_new_episode_rejects_old_chunks",
                task_memory="cleared_at_user_goal_replacement",
                preserved_qpos_sha256=hashlib.sha256(env.physics.data.qpos.tobytes()).hexdigest(),
                preserved_qvel_sha256=hashlib.sha256(env.physics.data.qvel.tobytes()).hexdigest())


def run_task_switch(env, policy, output, *, initial_task, replacement_task,
                    seed, switch_tick=200, replacement_ticks=1500,
                    warmup_ticks=50, warmup_predictions=0):
    if (initial_task not in GOALS or replacement_task not in GOALS
            or initial_task == replacement_task
            or type(seed) is not int or seed < 0
            or type(switch_tick) is not int or not 5 <= switch_tick < 1500
            or switch_tick % DECISION_TICKS
            or type(replacement_ticks) is not int or not 1 <= replacement_ticks <= 1500
            or type(warmup_ticks) is not int or warmup_ticks < 0
            or type(warmup_predictions) is not int or not 0 <= warmup_predictions <= 3):
        raise ValueError("invalid fixed task replacement schedule")
    metadata = getattr(policy, "metadata", {})
    if metadata.get("uses_privileged_state", metadata.get("truth_access")) is not False:
        raise ValueError("task replacement comparison requires an explicitly sensor-only policy")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = dict(schema=PROTOCOL, status="running", mode="offline", seed=seed,
                  initial_task=initial_task, replacement_task=replacement_task,
                  switch_tick=switch_tick, replacement_ticks=replacement_ticks,
                  warmup_ticks=warmup_ticks, warmup_predictions=warmup_predictions,
                  started_unix=time.time(), segments=[], strict_success=False,
                  physical_success=False, independent_acceptance=False,
                  hardware_validated=False, realtime_success=None,
                  physics_waits_for_inference=True, replacement_preparation_wall_s=None,
                  truth_visible_to_policy=False, eligible_training_data=False,
                  source_sha256=sha(__file__))
    write_json(output / "task-switch.json", report)
    try:
        first = run_navigation_episode(env, policy, output / "before", task=initial_task,
            seed=seed, realtime=False, max_ticks=switch_tick, warmup_ticks=warmup_ticks,
            warmup_predictions=warmup_predictions, record_frames=True, record_predictions=True)
        report["segments"].append(dict(path="before", report_sha256=sha(output / "before/report.json")))
        # An early terminal event is preserved, never cleared to force a switch.
        if env.tick != switch_tick or env.terminated or _hazards(first):
            report.update(status="stopped_before_replacement", reason="prefix_terminal_or_intervention")
            return report
        started = time.monotonic()
        boundary = replace_goal(env, replacement_task, env.episode_id + "/revision-1")
        write_json(output / "boundary.json", boundary)
        report["replacement_preparation_wall_s"] = time.monotonic() - started
        # A newly allocated executor buffer and policy.reset use the new ID.
        # No physical warmup, placement reset, controller reset or teleport.
        second = run_navigation_episode(env, policy, output / "after", task=replacement_task,
            seed=seed, realtime=False, max_ticks=replacement_ticks, warmup_ticks=0,
            warmup_predictions=0, record_frames=True, record_predictions=True,
            reset_environment=False)
        report["segments"].append(dict(path="after", report_sha256=sha(output / "after/report.json")))
        report.update(status="complete", boundary_sha256=sha(output / "boundary.json"),
                      physical_success=bool(second["task_success"]),
                      strict_success=bool(second["strict_model_success"] and not _hazards(second)),
                      prefix_goal_completion_required=False,
                      total_control_ticks=switch_tick + second["evaluator"]["tick"])
        return report
    except BaseException as error:
        report.update(status="incomplete", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report["finished_unix"] = time.time()
        report["files"] = {str(p.relative_to(output)): sha(p)
                           for p in output.rglob("*") if p.is_file() and p.name != "task-switch.json"}
        write_json(output / "task-switch.json", report)


def audit_task_switch_physics(folder, *, environment_factory, expected_report_sha256):
    """Replay both segments in one environment from their recorded commands.

    Checks real physics/cameras, source observations, buffers and the retained
    boundary. Does not call a model or independently authenticate HTTP replies.
    """
    folder = Path(folder)
    require(sha(folder / "task-switch.json") == expected_report_sha256, "task-switch report changed")
    report = read(folder / "task-switch.json")
    require(report["schema"] == PROTOCOL and report["status"] == "complete"
            and report["mode"] == "offline" and report["source_sha256"] == sha(__file__),
            "only matching complete offline replacement records can be replayed")
    require(len(report["segments"]) == 2
            and [s["path"] for s in report["segments"]] == ["before", "after"],
            "replacement needs exactly two ordered segments")
    files = report["files"]
    for name, digest in files.items():
        p = folder / name
        require(not p.is_symlink() and p.resolve().is_relative_to(folder.resolve())
                and sha(p) == digest, "task-switch recording changed or escaped")
    results = []
    with environment_factory() as env:
        for index, segment in enumerate(report["segments"]):
            path = folder / segment["path"]
            require(sha(path / "report.json") == segment["report_sha256"], "segment report changed")
            episode = read(path / "report.json")
            require(episode["mode"] == "offline" and episode["seed"] == report["seed"], "segment identity differs")
            task = report["replacement_task"] if index else report["initial_task"]
            require(episode["task"] == task, "recorded replacement task differs")
            control_audit = audit_recording(path)
            if not index:
                require(episode["initialization"] == "reset_and_warmup", "prefix initialization differs")
                env.reset(report["seed"], task, {"episode_id": episode["evaluator"]["episode_id"]})
                for _ in range(report["warmup_ticks"]):
                    env.step(SkillIntent("stand", np.zeros(13, np.float32), env.episode_id, env.tick, 1))
                env.begin_task_at_current_pose()
            else:
                require(episode["initialization"] == "prepared_physical_state_new_task_clock"
                        and env.tick == report["switch_tick"], "replacement boundary differs")
                boundary = replace_goal(env, task, episode["evaluator"]["episode_id"])
                require(boundary == read(folder / "boundary.json"), "retained physical boundary differs")
            controls, predictions = read(path / "control.json"), read(path / "predictions.json")
            with np.load(path / "recording.npz", allow_pickle=False) as recording:
                np.testing.assert_array_equal(recording["physics_ticks"], np.arange(len(controls)))
                camera_indices = {int(t): i for i, t in enumerate(recording["camera_ticks"])}
                require(len(camera_indices) == len(recording["rgb"]), "duplicate camera ticks")
                by_source = {}
                for prediction in predictions:
                    by_source.setdefault(prediction["source_tick"], []).append(prediction)
                observed = 0
                for tick, control in enumerate(controls):
                    require(env.tick == tick and not env.terminated, "replay terminated before recorded action")
                    np.testing.assert_array_equal(env.physics.data.qpos, recording["qpos"][tick])
                    env.render_enabled = tick in camera_indices
                    observation = env.observe()
                    if tick in camera_indices:
                        np.testing.assert_array_equal(observation.rgb, recording["rgb"][camera_indices[tick]])
                    for prediction in by_source.get(tick, []):
                        require(prediction["source_episode_id"] == env.episode_id
                                and prediction["observed_instruction"] == observation.instruction
                                and prediction["camera_tick"] == tick, "model source belongs to another task")
                        np.testing.assert_array_equal(observation.state[:48], prediction["observed_proprio48"])
                        observed += 1
                    env.render_enabled = False
                    env.step(SkillIntent(control["skill"], np.asarray(control["command"], np.float32),
                                         env.episode_id, tick, 1))
                require(observed == len(predictions), "unreplayed model source")
            require(canonical(env.evaluator_state()) == canonical(episode["evaluator"]), "physical task outcome differs")
            results.append(dict(path=segment["path"], control_ticks=len(controls),
                                rgb_frames=len(camera_indices), source_observations=observed,
                                control_audit=control_audit))
    require(len(results) == 2 and report["segments"][0]["path"] == "before"
            and report["segments"][1]["path"] == "after", "replacement needs exactly two ordered segments")
    first, second = (read(folder / name / "report.json") for name in ("before", "after"))
    require(report["strict_success"] == bool(second["strict_model_success"] and not _hazards(first)
            and not _hazards(second)), "task replacement success incorrectly ignores failures")
    require(all(sha(folder / name) == digest for name, digest in files.items()), "files changed during replay")
    return dict(schema=PROTOCOL + "_physics_audit", segments=results, boundary_preserved=True,
                strict_success=report["strict_success"], physics_and_source_observations_verified=True,
                neural_outputs_recomputed=False, neural_input_construction_verified=False,
                service_receipts_authenticated=False, fresh_realtime_measurement=False,
                independent_acceptance=False, hardware_validated=False)
