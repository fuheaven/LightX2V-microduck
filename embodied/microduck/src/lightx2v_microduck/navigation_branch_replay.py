"""Independently execute a recorded teacher continuation from its student state.

The verifier never calls a planner. Recorded high-level commands pass through
the actual official ONNX and BAM simulator, including an incomplete last
navigation decision. Diagnostic tensors never enter student inputs.
"""
import json

import numpy as np

from .contracts import SkillIntent
from .env import MicroDuckEnv
from .navigation_augmentation import validate_tensors


def canonical(value):
    return {k: v for k, v in json.loads(json.dumps(value, default=lambda x: x.tolist())).items()
            if k != "episode_id"}


def select_anchors(controls, *, student_success, fixed_ticks=(250, 750), stand_ticks=100):
    """Frozen action-only sampling; at most two motion states and one late stop."""
    if (type(student_success) is not bool or stand_ticks != 100
            or tuple(fixed_ticks) != (250, 750)
            or [c["tick"] for c in controls] != list(range(len(controls)))
            or any(c["execution_source"] != "navigation_model" or c["skill"] not in ("walk", "stand")
                   for c in controls)):
        raise ValueError("complete student controls and the fixed anchor rule are required")
    if student_success:
        return []
    selected = {t for t in fixed_ticks if t < len(controls)}
    onset = len(controls)
    for c in reversed(controls):
        if c["skill"] != "stand":
            break
        onset -= 1
    if len(controls) - onset >= stand_ticks:
        if onset % 5:
            raise ValueError("terminal stand is not decision aligned")
        selected.add(onset)
    return sorted(selected)


def replay_branch(student, arrays, report):
    """Check every physical tick and captured frame against a second execution."""
    start, count = report["source_tick"], report["label_decisions"]
    end = report["result"]["tick"]
    n = end - start
    validate_tensors(arrays, count, start, allow_empty_diagnostic=(count == 0))
    if (student.tick != start or student.terminated or not 0 < n <= 1500 - start
            or report["branch_isolation"] != "separate_environment_and_renderer"
            or count != report["continuation_decisions"]
            or not np.array_equal(arrays["action"], arrays["continuation_action"])
            or not 0 <= end - int(arrays["ticks"][-1]) < 5
            or arrays["continuation_command13"].shape != (n, 13)
            or arrays["continuation_skill_id"].shape != (n,)
            or arrays["continuation_state61"].shape != (n, 61)
            or arrays["continuation_raw_action"].shape != (n, 14)
            or arrays["continuation_qpos"].shape != (n, student.physics.model.nq)
            or not np.isin(arrays["continuation_skill_id"], [0, 1]).all()):
        raise ValueError("complete teacher trace including its terminal partial decision is required")
    for key in ("continuation_command13", "continuation_state61", "continuation_raw_action", "continuation_qpos"):
        if not np.isfinite(arrays[key]).all():
            raise ValueError("nonfinite teacher replay trace")
    captured = student.observe()
    np.testing.assert_array_equal(arrays["rgb"][0], captured.rgb)
    np.testing.assert_array_equal(arrays["proprio"][0], captured.state[:48])
    before = canonical(student.evaluator_state())
    snapshot = student.clone_state()
    with MicroDuckEnv(profile=student.profile, scene=student.scene, asset_root=student.asset_root,
                      image_size=student.physics.image_size, render=False,
                      action_delay_ticks=student.action_delay_ticks) as env:
        env.restore_state(snapshot)
        env.render_enabled = True
        observation = env.observe()
        # restore_state preserves the measured student RGB but discards the GL
        # context. Initialize its scene at the restored pose before advancing:
        # a cold first draw at the next pose can differ by a few pixel values.
        # Discard this draw; the captured student frame remains the input.
        env.physics.render()
        np.testing.assert_array_equal(env.observe().rgb, captured.rgb)
        for i in range(n):
            if env.terminated or env.tick != start + i:
                raise ValueError("teacher replay terminated at a different tick")
            np.testing.assert_array_equal(env.physics.data.qpos, arrays["continuation_qpos"][i])
            np.testing.assert_array_equal(observation.state, arrays["continuation_state61"][i])
            command = arrays["continuation_command13"][i]
            skill = "walk" if arrays["continuation_skill_id"][i] else "stand"
            env.render_enabled = False
            observation = env.step(SkillIntent(command=command.copy(), skill=skill, episode_id=env.episode_id,
                                               source_tick=env.tick, duration_ticks=1))
            np.testing.assert_array_equal(env.last_action, arrays["continuation_raw_action"][i])
            if (i + 1) % 5 == 0:
                env.render_enabled = True
                observation = env.observe()
                at = (i + 1) // 5
                np.testing.assert_array_equal(observation.rgb, arrays["rgb"][at])
                np.testing.assert_array_equal(observation.state[:48], arrays["proprio"][at])
        if canonical(env.evaluator_state()) != canonical(report["result"]):
            raise ValueError("recorded-command replay changed the final teacher evaluator")
    if before != canonical(student.evaluator_state()):
        raise ValueError("teacher replay mutated the original student")
    np.testing.assert_array_equal(student.observe().rgb, captured.rgb)
    return {"control_ticks_exact": n, "rgb_frames_exact": count + 1,
            "terminal_partial_ticks": n % 5, "complete_final_evaluator_exact": True,
            "teacher_queried_during_replay": False, "student_untouched": True,
            "renderer_initialized_at_source_tick": True, "captured_student_rgb_preserved": True}
