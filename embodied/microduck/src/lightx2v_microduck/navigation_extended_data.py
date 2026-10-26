"""Align recorded extended-command traces without changing their physical labels.

This conversion alone never qualifies a teacher or admits an episode to training.
The source must still pass its independent physical replay and dataset split gate.
"""
from __future__ import annotations

import numpy as np

from .navigation_action_contracts import FORWARD04_NAVIGATION_CONTRACT, action_contract


def supervised_transitions(trace, *, terminal_rgb=None):
    """Return public tensors and tail metadata for fully executed 5-tick actions.

    Collection records the camera before each action. When termination lands
    exactly on a decision boundary, training also needs a newly captured final
    image. Repeating the preceding image would fabricate the next observation.
    A partial last action stays in the private control trace, not as a full label.
    """
    actions, rgb = np.asarray(trace['action']), np.asarray(trace['rgb'])
    before, after = np.asarray(trace['state61']), np.asarray(trace['next_state61'])
    raw = np.asarray(trace['raw_action'])
    ticks = len(before)
    decisions, full = (ticks + 4) // 5, ticks // 5
    if full < 1 or before.shape != (ticks, 61) or after.shape != (ticks, 61) or raw.shape != (ticks, 14):
        raise ValueError('complete raw control states and at least one full decision are required')
    if actions.shape != (decisions, 5) or rgb.shape != (decisions, 256, 256, 3) or rgb.dtype != np.uint8:
        raise ValueError('teacher camera/action sequence does not match the control clock')
    for values in (actions, before, after, raw):
        if values.dtype != np.float32 or not np.isfinite(values).all():
            raise ValueError('physical labels and states must retain finite FP32 coordinates')
    action_contract(FORWARD04_NAVIGATION_CONTRACT).validate(actions)
    if not np.array_equal(before[1:], after[:-1]):
        raise ValueError('recorded control states are discontinuous')
    if not np.array_equal(before[1:, 34:48], raw[:-1]):
        raise ValueError('state history differs from the previous raw official action')
    if ticks % 5 == 0:
        if terminal_rgb is None:
            raise ValueError('a measured terminal camera frame is required; do not repeat the previous image')
        terminal = np.asarray(terminal_rgb)
        if terminal.shape != (256, 256, 3) or terminal.dtype != np.uint8:
            raise ValueError('invalid terminal camera frame')
        images = np.concatenate((rgb, terminal[None]), axis=0)
        states = np.concatenate((before[::5], after[-1:]), axis=0)
    else:
        if terminal_rgb is not None:
            raise ValueError('a partial-tail image is not a decision-boundary observation')
        images, states = rgb.copy(), before[::5].copy()
    assert len(images) == len(states) == full + 1
    tensors = dict(rgb=images, proprio=states[:, :48].copy(), action=actions[:full].copy(),
                   ticks=np.arange(full + 1, dtype=np.int64) * 5)
    metadata = dict(contract=FORWARD04_NAVIGATION_CONTRACT, control_ticks=ticks,
                    complete_decisions=full, tail_control_ticks=ticks % 5,
                    student_state='obs61[:48]; command13 excluded',
                    labels_clipped=False, training_release_allowed=False)
    return tensors, metadata


def branch_supervision(trace, *, source_tick, student_rgb, student_camera_ticks,
                       student_proprio, terminal_rgb=None):
    """Preserve absolute student time, causal history and forward-0.4 labels.

This is tensor alignment only. Qualification, independent physical replay and
training-split admission are still required before these tensors can be released.
"""
    from .navigation_history_recovery import validate_prefix

    if type(source_tick) is not int or not 0 <= source_tick < 1500 or source_tick % 5:
        raise ValueError('branch supervision requires an original decision-boundary tick')
    if len(trace['state61']) > 1500 - source_tick:
        raise ValueError('teacher continuation extends the original task deadline')
    tensors, metadata = supervised_transitions(trace, terminal_rgb=terminal_rgb)
    tensors['ticks'] += source_tick
    prefix_rgb, prefix_ticks = np.asarray(student_rgb), np.asarray(student_camera_ticks)
    validate_prefix(prefix_rgb, prefix_ticks, source_tick, tensors['rgb'][0])
    state = np.asarray(student_proprio)
    if state.shape != (48,) or state.dtype != np.float32 or not np.array_equal(state, tensors['proprio'][0]):
        raise ValueError('teacher continuation does not start at the actual student state')
    metadata.update(source_tick=source_tick, original_deadline_tick=1500,
                    first_student_rgb_exact=True, first_student_proprio_exact=True,
                    history_starts_at_episode_zero=True, training_release_allowed=False)
    return tensors, dict(rgb=prefix_rgb.copy(), ticks=prefix_ticks.copy()), metadata
