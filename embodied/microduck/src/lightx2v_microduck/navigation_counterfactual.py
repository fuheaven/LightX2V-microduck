"""Diagnose language grounding with task retargeting at one physical state.

These are new instructions with new scoring windows, not DAgger continuations
of the original task. The robot, camera, BAM, ONNX and RNG state are retained.
The original environment is restored even if a branch fails. Samples remain
diagnostic until a separate dataset readiness audit qualifies them.
"""

from __future__ import annotations

import hashlib

import numpy as np

from .navigation import DECISION_TICKS, VisualSearchTeacher, student_state
from .navigation_dagger import teacher_branch
from .tasks import get_task
from .wam_data import split_for_seed


def instruction_branches(env, tasks, *, max_decisions=300, label_horizon=32, obstacle_aware=False,
                         teacher_variant="coverage"):
    """Return teacher branches whose first RGB/proprio inputs are identical.

    A fresh teacher observes each new task. It cannot inherit the previous
    teacher's target locations, search coverage or stop latch. The ordinary
    evaluator still requires the complete approach/stop behavior.
    """
    if env.physics is None or env.daemon_backend is not None:
        raise ValueError("retargeting requires an initialized offline simulator")
    if env.profile != "official_bam" or env.scene != "room_clear":
        raise ValueError("retargeting currently uses the qualified clear-room teacher only")
    if split_for_seed(env.seed) != "train" or 10000 <= env.seed < 10200:
        raise ValueError("held-out layouts cannot enter instruction-retargeting diagnostics")
    if env.tick % DECISION_TICKS:
        raise ValueError("retarget at a navigation decision boundary")
    allowed = {f"find_{color}_ball" for color in ("red", "blue", "yellow")}
    tasks = tuple(tasks)
    if len(tasks) < 2 or len(set(tasks)) != len(tasks) or not set(tasks) <= allowed:
        raise ValueError("at least two distinct supported color instructions are required")
    if env.task.task_id not in allowed:
        raise ValueError("source task must use the same three-ball navigation scene")
    if max_decisions < 1 or label_horizon < 1:
        raise ValueError("positive branch and label horizons are required")
    if teacher_variant not in ("coverage", "initial_scan"):
        raise ValueError("unknown retarget teacher variant")
    if teacher_variant == "initial_scan" and not obstacle_aware:
        raise ValueError("the initial-scan experiment requires the visibility-gated obstacle guard")
    # Capture the sensor sample once, before branching. A state clone must
    # reuse that measured frame: re-rendering it can introduce GPU rasterization
    # differences even with a fresh context and identical physical state.
    first_rgb = env.render().copy()
    first_state = student_state(env.observe()).copy()
    snapshot = env.clone_state()
    render_before = env.render_enabled
    source_renderer = env.physics.renderer
    branches = []
    try:
        env.physics.renderer = None
        for task in tasks:
            # Isolate rendering history as well as physical state. Reusing a
            # renderer after a long continuation can change a few edge pixels
            # at an otherwise identical pose. Never relax the input-equality
            # gate or alter the caller's original renderer/context.
            if env.physics.renderer is not None:
                env.physics.renderer.close()
                env.physics.renderer = None
            env.render_enabled = False
            env.restore_state(snapshot)
            env.task = get_task(task)
            env.begin_task_at_current_pose()
            branch_render = env.physics.render()  # Initialize segmentation too.
            render_difference = np.abs(branch_render.astype(np.int16) - first_rgb.astype(np.int16))
            # This is the real sensor frame captured at the snapshot's physical
            # time, not a synthetic image or a frame from a later teacher state.
            env._rgb_cache, env._rgb_tick = first_rgb.copy(), env.tick
            env.render_enabled = True
            observation = env.observe()
            state = student_state(observation)
            if not np.array_equal(observation.rgb, first_rgb) or not np.array_equal(state, first_state):
                rgb_error = np.abs(observation.rgb.astype(np.int16) - first_rgb.astype(np.int16))
                state_error = np.abs(state - first_state)
                raise ValueError(f"retargeting changed first input for {task}: "
                                 f"rgb max={rgb_error.max()}, channels={np.count_nonzero(rgb_error)}, "
                                 f"proprio max={state_error.max()}, indices={np.flatnonzero(state_error).tolist()}")
            experiment_sha256 = None
            if teacher_variant == "initial_scan":
                from .navigation_teacher_experiments import (
                    EXPERIMENT_SOURCE_SHA256, InitialScanVisualSearchTeacher,
                )

                teacher = InitialScanVisualSearchTeacher(env)
                experiment_sha256 = EXPERIMENT_SOURCE_SHA256
            else:
                teacher = VisualSearchTeacher(env, obstacle_aware=obstacle_aware)
            arrays, report = teacher_branch(env, teacher, max_decisions=max_decisions,
                                             label_horizon=label_horizon)
            if not np.array_equal(arrays["rgb"][0], first_rgb) or not np.array_equal(arrays["proprio"][0], first_state):
                raise ValueError("teacher labels do not start at the shared observed state")
            report.update(
                branch_kind="counterfactual_instruction_retargeting",
                original_task=snapshot["env"]["task"].task_id,
                original_source_tick=snapshot["env"]["tick"],
                original_episode_id=snapshot["env"]["episode_id"],
                original_source_terminated=snapshot["env"]["_terminated"],
                branch_task_clock_restarted=True,
                physics_and_controller_reset=False,
                teacher_history_reset=True,
                obstacle_aware_teacher=bool(obstacle_aware),
                teacher_variant=teacher_variant,
                teacher_experiment_source_sha256=experiment_sha256,
                first_rgb_source="single_real_sensor_capture_retained_with_physical_snapshot",
                independent_rerender_max_abs=int(render_difference.max()),
                independent_rerender_different_channels=int(np.count_nonzero(render_difference)),
                independent_branch_renderer=True,
                first_rgb_sha256=hashlib.sha256(first_rgb.tobytes()).hexdigest(),
                first_proprio_sha256=hashlib.sha256(first_state.tobytes()).hexdigest(),
                eligible_training_supervision=False,
                readiness="diagnostic_only_requires_separate_dataset_qualification",
                scope="new task from unchanged physical state; not DAgger or student task success",
            )
            branches.append((arrays, report))
    finally:
        if env.physics.renderer is not None and env.physics.renderer is not source_renderer:
            env.physics.renderer.close()
        env.physics.renderer = None
        env.render_enabled = False
        env.restore_state(snapshot)
        # This context was parked at the original pose, never rendered by a
        # branch. Reattach it only after restore discards branch contexts.
        env.physics.renderer = source_renderer
        env.render_enabled = render_before
    return branches
