"""Collect separate guarded-teacher demonstrations with replayable raw evidence.

Qualification cases are never training cases. A recorded success only becomes
eligible for a later dataset release after its independent physical replay.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .navigation_action_contracts import FORWARD04_NAVIGATION_CONTRACT, action_contract
from .navigation_extended_data import supervised_transitions
from .navigation_extended_qualification import RAW_CONTRACT, TASKS, layout_sha256, require


SCHEMA = 'guarded_navigation_separate_collection_v1'
PREFLIGHT = 'separate_collection_preflight'
COLLECTION = 'separate_teacher_demonstrations'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False,
                              default=lambda v: v.tolist()) + '\n')
    tmp.replace(path)


def verify_collection_plan(plan, certificate=None):
    """Check a pinned plan; the caller must verify its certificate file hash."""
    require(plan.get('schema') == SCHEMA and plan.get('profile') == 'official_bam'
            and plan.get('scene') == 'room_curriculum_one'
            and plan.get('teacher_contract') == RAW_CONTRACT
            and plan.get('student_contract') == FORWARD04_NAVIGATION_CONTRACT,
            'collection dynamics or action semantics changed')
    require(plan.get('max_ticks') == 1500 and plan.get('warmup_ticks') == 50
            and plan.get('decision_ticks') == 5 and plan.get('strict_stop_ticks') == 100
            and plan.get('strict_stop_surface_m') == [.25, .4]
            and plan.get('training_release_allowed') is False,
            'collection cannot change timing, task thresholds or release data by itself')
    cases = plan.get('cases', [])
    if plan.get('purpose') == PREFLIGHT:
        require(certificate is None and cases == [dict(seed=2300, task='find_red_ball', split='diagnostic')],
                'unqualified preflight is restricted to the existing diagnostic case')
        return
    require(plan.get('purpose') == COLLECTION and isinstance(certificate, dict),
            'separate demonstration collection requires a qualified teacher certificate')
    require(certificate.get('schema') == 'guarded_teacher_qualification_certificate_v1'
            and certificate.get('eligible_for_separate_collection') is True
            and certificate.get('qualification_episodes_may_train') is False
            and certificate.get('training_release_allowed') is False
            and certificate.get('teacher_contract') == RAW_CONTRACT
            and certificate.get('source_manifest_sha256') == plan.get('qualification_source_manifest_sha256'),
            'the collection teacher differs from its qualification certificate')
    colors = certificate.get('per_color', {})
    require(set(colors) == TASKS and all(c.get('total') == 100 and c.get('successes', 0) >= 95 for c in colors.values()),
            'a color failed the unchanged 95/100 teacher gate')
    qualified_seeds = certificate.get('qualification_seeds', [])
    qualified_geometry = certificate.get('qualification_geometry_sha256', [])
    require(len(qualified_seeds) == len(set(qualified_seeds)) == 100
            and len(qualified_geometry) == len(set(qualified_geometry)) == 100,
            'qualification must identify 100 distinct seeds and physical layouts')
    require(len(cases) == 300 and all(type(c['seed']) is int for c in cases),
            'the initial separate collection requires all 300 planned cases')
    seeds = {c['seed'] for c in cases}
    require(len(seeds) == 100 and {(c['seed'], c['task']) for c in cases}
            == {(s, t) for s in seeds for t in TASKS}, 'omitted or duplicated collection case')
    assignments = {}
    for c in cases:
        assignments.setdefault(c['seed'], set()).add(c['split'])
    require(all(len(v) == 1 for v in assignments.values()), 'one physical layout crosses dataset splits')
    counts = {split: sum(v == {split} for v in assignments.values()) for split in ('train', 'validation', 'test')}
    require(counts == {'train': 80, 'validation': 10, 'test': 10}, 'freeze an 80/10/10 split before collection')
    require(not seeds.intersection(qualified_seeds) and not seeds.intersection(range(10000, 10200)),
            'qualification or reserved final seeds entered collection')
    for start, stop in plan.get('excluded_seed_ranges', []):
        require(not seeds.intersection(range(start, stop)), 'excluded development layouts entered collection')


def verify_layout(plan, certificate, world):
    digest = layout_sha256(world)
    excluded = set(plan.get('excluded_geometry_sha256', []))
    if certificate is not None:
        excluded.update(certificate['qualification_geometry_sha256'])
    require(digest not in excluded, 'collection physically duplicates a reserved layout')
    return digest


def initialize(env, case, plan, intent):
    env.render_enabled = False
    env.reset(case['seed'], case['task'])
    require(env.profile == plan['profile'] and env.scene == plan['scene']
            and env.task.duration_s == 30. and env.task.stop_hold_s == 2.
            and tuple(env.task.stop_distance_m) == (.25, .4), 'actual environment task thresholds changed')
    for _ in range(plan['warmup_ticks']):
        env.step(intent(np.array([0, 0, 0, 0, 1], np.float32), env))
    env.begin_task_at_current_pose()
    require(env.tick == 0 and not env.terminated, 'task clock did not reset after warmup')


def independent_success(world, diagnostics):
    return bool(world['success'] and world['strict_success'] and world['goal_hold_ticks'] >= 100
        and not world['collision'] and not world['ever_fallen'] and not world['fault']
        and world['recovery_events'] == 0
        and not any(d.get('planner_phase', d.get('phase')) in ('recover', 'no_path') for d in diagnostics))


def _physical(world):
    value = json.loads(json.dumps(world, default=lambda v: v.tolist()))
    value.pop('episode_id', None)
    return value


def collect(env, teacher_factory, intent, output, *, case, plan, plan_sha256, certificate=None):
    """Record fresh physics; keep failures and private evidence out of tensors."""
    verify_collection_plan(plan, certificate)
    require(case in plan['cases'], 'unregistered collection case')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = dict(schema=SCHEMA, status='running', purpose=plan['purpose'], **case,
        plan_sha256=plan_sha256, started_unix=time.time(), profile=plan['profile'], scene=plan['scene'],
        teacher_contract=RAW_CONTRACT, student_contract=FORWARD04_NAVIGATION_CONTRACT,
        teacher_qualified=certificate is not None, qualification_only=False,
        eligible_training_demonstration=False, training_release_allowed=False,
        formal_acceptance=False, hardware_validated=False, files={})
    write_json(output / 'episode.json', report)
    trace = {k: [] for k in ('action', 'rgb', 'state61', 'next_state61', 'raw_action', 'positions', 'qpos')}
    diagnostics, teacher, terminal = [], None, None
    try:
        initialize(env, case, plan, intent)
        initial = copy.deepcopy(env.evaluator_state())
        geometry = verify_layout(plan, certificate, initial)
        report.update(initial=initial, layout_sha256=geometry, model_signature=env._signature,
                      robot_signature=env._robot_signature, instruction=env.task.instruction)
        teacher = teacher_factory(env)
        while not env.terminated:
            require(env.tick < plan['max_ticks'] and env.tick % 5 == 0, 'unexpected collection clock')
            env.render_enabled = True
            obs = env.observe()
            require(obs.tick == obs.camera_tick == env.tick, 'camera is not aligned to its decision tick')
            action = np.asarray(teacher.predict(obs)).copy()
            require(action.dtype == np.float32 and action.shape == (5,), 'teacher must preserve FP32 labels')
            action_contract(FORWARD04_NAVIGATION_CONTRACT).validate(action)
            trace['rgb'].append(obs.rgb.copy()); trace['action'].append(action)
            diagnostics.append(copy.deepcopy(teacher.last_diagnostics))
            env.render_enabled = False
            for _ in range(5):
                trace['state61'].append(env.observe().state.copy())
                trace['qpos'].append(env.physics.data.qpos.copy())
                env.step(intent(action, env))
                trace['raw_action'].append(env.last_action.copy())
                trace['next_state61'].append(env.observe().state.copy())
                world = env.evaluator_state()
                trace['positions'].append([env.tick, *world['robot_position'], world['goal_hold_ticks']])
                if env.terminated:
                    break
        result = copy.deepcopy(env.evaluator_state())
        require(0 < env.tick <= plan['max_ticks'], 'termination exceeded the frozen task clock')
        if env.tick % 5 == 0:
            env.render_enabled = True
            obs = env.observe()
            require(obs.camera_tick == env.tick, 'terminal camera frame is stale')
            terminal = obs.rgb.copy()
            np.testing.assert_array_equal(obs.state, trace['next_state61'][-1])
            np.save(output / 'terminal-rgb.npy', terminal)
        arrays = {k: np.asarray(v) for k, v in trace.items()}
        if env.tick >= 5:
            tensors, alignment = supervised_transitions(arrays, terminal_rgb=terminal)
            np.savez_compressed(output / 'navigation.npz', **tensors)
            report.update(alignment=alignment)
        else:
            report.update(alignment=None, no_complete_decision=True)
        report.update(status='complete', result=result, independent_success=independent_success(result, diagnostics),
                      terminal_frame_measured=terminal is not None, control_ticks=env.tick)
    except Exception as exc:
        report.update(status='failed', error=type(exc).__name__ + ': ' + str(exc))
        raise
    finally:
        # Even interrupted or failed cases retain their executed prefix. It is
        # deliberately not made eligible or repaired into a successful episode.
        try:
            if teacher is not None:
                teacher.close()
        finally:
            np.savez_compressed(output / 'trajectory.npz', **{k: np.asarray(v) for k, v in trace.items()})
            report.update(diagnostics=diagnostics, finished_unix=time.time(),
                files={p.name: sha256(p) for p in output.iterdir() if p.suffix in ('.npz', '.npy')})
            write_json(output / 'episode.json', report)
    return report


def verify_guard_diagnostics(actions, diagnostics, *, source_tick=0):
    require(type(source_tick) is int and 0 <= source_tick < 1500 and source_tick % 5 == 0,
            'guarded labels require the original task clock')
    require(len(actions) == len(diagnostics), 'teacher diagnostics do not match decision labels')
    for i, (action, d) in enumerate(zip(actions, diagnostics)):
        extended = bool(action[0] > np.float32(.3))
        require(extended == bool(d['approach_speed_probe_applied']), 'unguarded extended forward label')
        if not extended:
            continue
        guard = d['physical_speed_guard']
        require(action[0] == np.float32(.4) and abs(float(action[2])) <= .55
            and action[4] < .5 and source_tick + i * 5 < 1400 and d['phase'] == 'approach'
            and d['target_previously_observed'] and d['speed_proposal_surface_distance_m'] > .55
            and guard['safe'] and guard['source_state_unchanged'] and guard['task_neutralized']
            and guard['simulated_ticks'] == 150 and guard['movement_ticks'] == 50 and guard['braking_ticks'] == 100
            and guard['minimum_clearance_m'] >= .14 and guard['minimum_target_surface_distance_m'] >= .25
            and guard['object_source'] == 'explicit_visual_observation_memory',
            'extended label lacks its qualified physical guard')


def replay(env, intent, visible_pixels, output, *, case, plan, plan_sha256, certificate=None):
    """Independently replay all motor ticks, RGB frames and public transitions."""
    verify_collection_plan(plan, certificate)
    require(case in plan['cases'], 'unregistered collection case')
    output = Path(output)
    require(not (output / 'audit.json').exists(), 'do not overwrite an independent audit')
    report = json.loads((output / 'episode.json').read_text())
    require(report['status'] == 'complete' and report['plan_sha256'] == plan_sha256
            and report['schema'] == SCHEMA and report['purpose'] == plan['purpose']
            and all(report[k] == case[k] for k in ('seed', 'task', 'split'))
            and report['teacher_contract'] == RAW_CONTRACT
            and report['student_contract'] == FORWARD04_NAVIGATION_CONTRACT
            and report['training_release_allowed'] is False
            and report['eligible_training_demonstration'] is False,
            'collection identity changed or data was released before replay')
    for name, digest in report['files'].items():
        require(Path(name).name == name and sha256(output / name) == digest, 'recording bytes changed')
    with np.load(output / 'trajectory.npz', allow_pickle=False) as z:
        trace = {k: z[k].copy() for k in z.files}
    ticks = report['result']['tick']
    require(0 < ticks <= 1500 and ticks == report['control_ticks'], 'invalid full episode duration')
    require(all(len(trace[k]) == ticks for k in ('state61', 'next_state61', 'raw_action', 'positions', 'qpos'))
            and len(trace['action']) == len(trace['rgb']) == (ticks + 4) // 5,
            'recording omits executed ticks or camera frames')
    action_contract(FORWARD04_NAVIGATION_CONTRACT).validate(trace['action'])
    verify_guard_diagnostics(trace['action'], report['diagnostics'])
    initialize(env, case, plan, intent)
    require(_physical(env.evaluator_state()) == _physical(report['initial']), 'initial physics differs')
    require(env._signature == report['model_signature'] and env._robot_signature == report['robot_signature']
            and env.task.instruction == report['instruction'], 'robot or instruction changed')
    require(verify_layout(plan, certificate, env.evaluator_state()) == report['layout_sha256'], 'layout identity changed')
    for tick in range(ticks):
        require(env.tick == tick, 'replay control clock changed')
        np.testing.assert_array_equal(env.physics.data.qpos, trace['qpos'][tick])
        if tick % 5 == 0:
            env.render_enabled = True
            obs = env.observe()
            require(obs.camera_tick == tick, 'replay camera clock changed')
            np.testing.assert_array_equal(obs.rgb, trace['rgb'][tick // 5])
            require(visible_pixels(env) == report['diagnostics'][tick // 5]['target_visible_pixels'],
                    'recorded visibility differs from physical camera segmentation')
            env.render_enabled = False
        np.testing.assert_array_equal(env.observe().state, trace['state61'][tick])
        env.step(intent(trace['action'][tick // 5], env))
        np.testing.assert_array_equal(env.last_action, trace['raw_action'][tick])
        np.testing.assert_array_equal(env.observe().state, trace['next_state61'][tick])
        world = env.evaluator_state()
        np.testing.assert_array_equal([env.tick, *world['robot_position'], world['goal_hold_ticks']], trace['positions'][tick])
    require(env.terminated and _physical(env.evaluator_state()) == _physical(report['result']), 'final evaluation differs')
    terminal = None
    if ticks % 5 == 0:
        terminal = np.load(output / 'terminal-rgb.npy', allow_pickle=False)
        env.render_enabled = True
        obs = env.observe()
        require(obs.camera_tick == ticks, 'terminal replay image is stale')
        np.testing.assert_array_equal(obs.rgb, terminal)
    require(report['terminal_frame_measured'] == (terminal is not None), 'terminal frame provenance changed')
    full = ticks // 5
    if full:
        expected, alignment = supervised_transitions(trace, terminal_rgb=terminal)
        require(report['alignment'] == alignment, 'transition alignment metadata changed')
        with np.load(output / 'navigation.npz', allow_pickle=False) as z:
            require(set(z.files) == set(expected), 'private teacher data entered public tensors')
            for key, value in expected.items():
                np.testing.assert_array_equal(z[key], value)
    else:
        require(not (output / 'navigation.npz').exists(), 'partial action cannot become a full training label')
    success = independent_success(report['result'], report['diagnostics'])
    require(success == report['independent_success'], 'saved teacher outcome changed')
    row = dict(**case, purpose=plan['purpose'], plan_sha256=plan_sha256,
        metadata_sha256=sha256(output / 'episode.json'), files=report['files'],
        layout_sha256=report['layout_sha256'], exact_control_ticks=ticks,
        exact_rgb_frames=len(trace['rgb']) + int(terminal is not None),
        complete_decisions=full, tail_control_ticks=ticks % 5,
        strict_success=success, final_evaluator_exact=True, public_tensors_exact=bool(full),
        candidate_for_release=bool(certificate is not None and success and full),
        training_release_allowed=False, formal_acceptance=False)
    write_json(output / 'audit.json', row)
    return row
