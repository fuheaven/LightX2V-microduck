"""Post-episode diagnostics for a changed language target, never policy input."""

from __future__ import annotations

import numpy as np


def requested_goal_diagnostic(trajectory, *, target_positions, requested_color, task,
                              evaluator, fallback_control_ticks, cancelled,
                              realtime_timing_qualified):
    """Apply the visual-goal stopping conditions to the recorded requested ball.

    The caller records the ball's actual position at every control tick. The
    original task still determines termination; this describes its time window.
    Events anywhere in that window disqualify independent completion, even if
    they occur after a diagnostic hold. Nothing here changes the task score.
    """
    values = np.asarray(trajectory, dtype=np.float64)
    target = np.asarray(target_positions, dtype=np.float64)
    if (task.suite != "visual_goal" or values.ndim != 2 or values.shape[1] != 12
            or len(values) != evaluator["tick"] + 1 or not np.isfinite(values).all()
            or target.shape != (len(values), 3) or not np.isfinite(target).all()
            or not np.isclose(values[0, 0], 0., rtol=0., atol=1e-6)
            or not np.allclose(np.diff(values[:, 0]), .02, rtol=0., atol=1e-6)):
        raise ValueError("requested-goal diagnostic requires the complete 50 Hz visual-task trajectory")
    if type(fallback_control_ticks) is not int or fallback_control_ticks < 0:
        raise ValueError("invalid fallback count")
    if type(cancelled) is not bool or realtime_timing_qualified not in (None, True, False):
        raise ValueError("invalid cancellation or timing evidence")
    required_ticks = round(task.stop_hold_s * 50)
    hold = longest = 0
    first_completion = None
    distances = np.linalg.norm(values[:, 1:3] - target[:, :2], axis=1) - .035
    # Row zero is the initial pose, not a completed control interval. The
    # 200 ms heading window and speed column match MicroDuckEnv._update_task.
    for tick in range(1, len(values)):
        origin = max(0, tick - 10)
        yaw_delta = values[tick, 5] - values[origin, 5]
        yaw_speed = abs(np.arctan2(np.sin(yaw_delta), np.cos(yaw_delta))) / max(
            values[tick, 0] - values[origin, 0], .02)
        ready = (task.stop_distance_m[0] <= distances[tick] <= task.stop_distance_m[1]
                 and values[tick, 9] < .05 and yaw_speed < .15
                 and values[tick, 4] < -.82 and values[tick, 3] > .09)
        hold = hold + 1 if ready else 0
        longest = max(longest, hold)
        if hold >= required_ticks and first_completion is None:
            first_completion = tick
    disqualifiers = [name for name in ("collision", "ever_fallen", "fault", "recovery_events")
                     if evaluator[name]]
    if fallback_control_ticks:
        disqualifiers.append("fallback_control")
    if cancelled:
        disqualifiers.append("cancelled")
    completed = first_completion is not None and not disqualifiers
    return {
        "schema": "microduck_requested_goal_diagnostic_v1",
        "model_requested_color": requested_color,
        "evaluated_target_color": evaluator["target_color"],
        "evaluator_target_unchanged": True,
        "termination_still_follows_original_task": True,
        "observed_control_ticks": evaluator["tick"],
        "required_goal_hold_ticks": required_ticks,
        "longest_requested_goal_hold_ticks": longest,
        "final_requested_goal_hold_ticks": hold,
        "first_requested_goal_completion_tick": first_completion,
        "final_distance_to_model_requested_ball_surface_m": float(distances[-1]),
        "minimum_distance_to_model_requested_ball_surface_m": float(distances.min()),
        "strict_requested_target_success": completed,
        "realtime_requested_target_success": (completed and realtime_timing_qualified
                                               if realtime_timing_qualified is not None else None),
        "disqualifiers": disqualifiers,
        "task_rules": {"stop_surface_distance_m": list(task.stop_distance_m),
                       "stop_hold_s": task.stop_hold_s, "max_planar_speed_mps": .05,
                       "max_heading_speed_rps": .15, "heading_window_ticks": 10},
        "evaluator_source_sha256": evaluator["evaluator_source_sha256"],
        "tasks_source_sha256": evaluator["tasks_source_sha256"],
        "scope": "Requested-target diagnostic over the observed episode; not capability acceptance.",
        "formal_acceptance": False,
    }
