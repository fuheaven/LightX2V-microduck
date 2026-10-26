"""Qualified guarded-teacher continuations without executing its shadow plans.

Factories are supplied by the pinned teacher runtime. This module changes no
planner, physics, action bounds or qualification result. The caller must verify
training-layout admission and independently replay every returned control trace
before releasing labels. No model capability follows from a teacher continuation.
"""
import copy

import numpy as np


def _require(value, message):
    if not value:
        raise ValueError(message)


def observe_guarded_history(teacher, observation, coverage_factory):
    """Remember actual observations without predicting an unexecuted route."""
    env = teacher.env
    _require(observation.tick == observation.camera_tick == env.tick,
             'guarded teacher history requires a current captured observation')
    coverage = teacher.coverage
    _require(coverage is None or (coverage.frontier_controls is None
                                 and coverage.route_start_tick is None),
             'observation-only teacher already contains an execution plan')
    world = env.evaluator_state()
    pixels = {color: teacher._visible_pixels(color) for color in world['balls']}
    for color, count in pixels.items():
        if count >= 16:  # The qualified teacher's unchanged visual-memory rule.
            teacher.observed_balls[color] = np.asarray(world['balls'][color]).copy()
    target = env.task.target_color
    if pixels[target] >= 16:
        teacher.last_seen_position = teacher.observed_balls[target].copy()
    if teacher.obstacle_aware:
        world['obstacles'] = list(world['obstacles']) + [
            dict(name='observed_ball_' + color, xy=point[:2].tolist(),
                 half_size=[.035, .035], height=.07)
            for color, point in teacher.observed_balls.items() if color != target]
    if coverage is None:
        teacher.coverage = coverage_factory(world, teacher.motion_controller, env)
    teacher.coverage.obstacles = world['obstacles']
    physics = env.physics
    teacher.coverage.update_camera(physics.data.cam_xpos[physics.camera],
        physics.data.cam_xmat[physics.camera], physics.model.cam_fovy[physics.camera])
    _require(teacher.coverage.frontier_controls is None
             and teacher.coverage.route_start_tick is None,
             'observing the camera unexpectedly created an execution plan')
    return dict(tick=env.tick, pixels=pixels,
                observed_colors=sorted(teacher.observed_balls),
                target_previously_observed=teacher.last_seen_position is not None)


def fork_guarded_teacher(teacher, branch, teacher_factory):
    """Clone observed memory while owning a fresh predictor environment."""
    fork = teacher_factory(branch)
    guard = fork.speed_guard
    _require(fork is not teacher and fork.env is branch and guard is not teacher.speed_guard
             and guard.env is not teacher.speed_guard.env and guard.env is not branch,
             'teacher factory must create isolated branch and predictor resources')
    try:
        fork.__dict__ = copy.deepcopy(teacher.__dict__,
                                     {id(teacher.env): branch, id(teacher.speed_guard): guard})
        _require(fork.env is fork.planner.env is branch
                 and (fork.coverage is None or fork.coverage.source_env is branch)
                 and fork.speed_guard is guard,
                 'cloned teacher still points to the student environment')
        return fork
    except BaseException:
        fork.close()
        raise


def collect_guarded_branch(env, teacher, *, env_factory, teacher_factory,
                           intent_factory, validate_action, snapshot_digest):
    """Record a complete branch within the original 30-second task clock.

The returned raw trace includes partial final decisions. Public supervision is
constructed separately, including an actually rendered final image when the
episode ends exactly on a decision boundary. Never reset the branch task clock.
"""
    _require(teacher.env is env and not env.terminated
             and env.daemon_backend is None and 0 <= env.tick < 1500 and env.tick % 5 == 0,
             'branch requires a live local student decision within its original deadline')
    _require(teacher.coverage is not None and teacher.coverage.frontier_controls is None
             and teacher.coverage.route_start_tick is None,
             'guarded branch requires observation-only history')
    source_tick = env.tick
    before_digest = snapshot_digest(env)
    snapshot = env.clone_state()
    trace = {name: [] for name in ('action', 'rgb', 'state61', 'next_state61', 'qpos',
                                  'raw_action', 'command13', 'skill_id')}
    diagnostics = []
    terminal_rgb = None
    try:
        with env_factory() as branch:
            branch.restore_state(snapshot)
            # Recreate renderer history at the same source pose as independent replay.
            # Preserve the already captured student RGB in the environment cache.
            branch.physics.render()
            _require(branch.tick == source_tick and not branch.terminated,
                     'restoring the student snapshot changed the task clock')
            fork = fork_guarded_teacher(teacher, branch, teacher_factory)
            try:
                while not branch.terminated and branch.tick < 1500:
                    branch.render_enabled = True
                    observation = branch.observe()
                    _require(observation.tick == observation.camera_tick == branch.tick,
                             'teacher branch camera is stale')
                    action = validate_action(fork.predict(observation)).copy()
                    _require(action.shape == (5,) and action.dtype == np.float32,
                             'teacher action must retain five physical FP32 coordinates')
                    trace['action'].append(action)
                    trace['rgb'].append(observation.rgb.copy())
                    diagnostics.append(copy.deepcopy(fork.last_diagnostics))
                    branch.render_enabled = False
                    for _ in range(5):
                        tick = branch.tick
                        trace['qpos'].append(branch.physics.data.qpos.copy())
                        trace['state61'].append(branch.observe().state.copy())
                        intent = intent_factory(action, branch)
                        trace['command13'].append(intent.command.copy())
                        trace['skill_id'].append(int(intent.skill == 'walk'))
                        branch.step(intent)
                        _require(branch.tick == tick + 1, 'a control trace step must advance one tick')
                        trace['next_state61'].append(branch.observe().state.copy())
                        trace['raw_action'].append(branch.last_action.copy())
                        if branch.terminated or branch.tick == 1500:
                            break
                if (branch.tick - source_tick) % 5 == 0:
                    branch.render_enabled = True
                    terminal_rgb = branch.observe().rgb.copy()
                final = copy.deepcopy(branch.evaluator_state())
            finally:
                fork.close()
    finally:
        _require(snapshot_digest(env) == before_digest,
                 'teacher branching changed the original student simulation')
    arrays = {name: np.asarray(values) for name, values in trace.items()}
    report = dict(source_tick=source_tick, original_deadline_tick=1500,
                  control_ticks=len(arrays['state61']),
                  terminal_frame_measured=terminal_rgb is not None,
                  teacher_diagnostics=diagnostics, result=final,
                  original_student_unchanged=True, training_release_allowed=False,
                  formal_acceptance=False)
    return arrays, terminal_rgb, report
