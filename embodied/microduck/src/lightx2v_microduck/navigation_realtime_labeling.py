"""Teacher branches at actual states of an admitted realtime TRAIN recording.

The student is reconstructed from its executed 50 Hz commands, including
source-age indexing and fallback. Teachers only receive detached snapshots in
a separate environment. This module produces diagnostic labels; dataset
publication still requires the qualified teacher and release audits.
"""
from __future__ import annotations

import copy
import pickle
from collections import deque
from contextlib import ExitStack

import numpy as np

from .navigation_outcomes import read, require, sha
from .navigation_recording_replay import replay_recording
from .navigation_rollin_admission import GuardedTrainingCase


def _same_tree(actual, expected, path='snapshot'):
    """Include RNG, BAM, evaluator and policy state, not just body positions."""
    if isinstance(expected, np.ndarray):
        require(isinstance(actual, np.ndarray) and actual.dtype == expected.dtype,
                'teacher branch changed snapshot array type: '+path)
        np.testing.assert_array_equal(actual, expected, err_msg=path)
    elif isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(),
                'teacher branch changed snapshot fields: '+path)
        for key in expected:
            _same_tree(actual[key], expected[key], path+'.'+str(key))
    elif isinstance(expected, (list, tuple, deque)):
        require(type(actual) is type(expected) and len(actual) == len(expected),
                'teacher branch changed snapshot sequence: '+path)
        for index, (a, e) in enumerate(zip(actual, expected)):
            _same_tree(a, e, path+'['+str(index)+']')
    elif type(expected).__module__ == 'mujoco._structs' and type(expected).__name__ == 'MjData':
        # clone_state creates a new MjData every time; Python object equality
        # compares identity. MuJoCo's serialized state compares the full data,
        # including solver warm starts and contacts, without dropping fields.
        require(type(actual) is type(expected)
                and actual.__getstate__() == expected.__getstate__(),
                'teacher branch changed snapshot physics data: '+path)
    elif type(expected).__module__.startswith('bam.'):
        # BAM snapshots contain backend objects and a cyclic model/actuator
        # graph. Compare its complete serialized state, preserving reference
        # sharing, instead of object identity or a hand-picked parameter list.
        require(type(actual) is type(expected)
                and pickle.dumps(actual, protocol=5) == pickle.dumps(expected, protocol=5),
                'teacher branch changed snapshot actuator state: '+path)
    else:
        require(actual == expected, 'teacher branch changed snapshot state: '+path)


def label_realtime_recording(folder, case, *, admission, expected_files, anchors,
                             environment_factory, teacher_factory, observe_teacher,
                             collect_branch, verify_branch, _replay=replay_recording):
    """Return pending labels only after exact replay of the original episode.

    Factories and teacher callbacks must come from the caller's pinned,
    qualified runtime. ``collect_branch(env, teacher)`` returns tensors and a
    report with the existing full-continuation schema; ``verify_branch`` must
    independently consume recorded commands, without querying the teacher.
    The caller may then run the separate guarded-data publisher. This function
    never changes a checkpoint, extends a task, or publishes optimizer input.
    """
    from pathlib import Path

    folder = Path(folder)
    require(isinstance(admission, GuardedTrainingCase),
            'realtime labeling requires a qualified TRAIN layout admission')
    require(case.get('mode') == 'realtime' and case.get('warmup_ticks') == 50,
            'labeling requires the original realtime reset-and-warmup contract')
    admission.validate_request(task=case['task'], seed=case['seed'], scene=case['scene'],
        profile=case['profile'], contract=case['contract'], record_only=True)
    required = {'report.json', 'recording.npz', 'control.json', 'predictions.json', 'decisions.json'}
    require(required <= set(expected_files), 'complete realtime recording hashes are required')
    for name, digest in expected_files.items():
        require(Path(name).name == name and not (folder/name).is_symlink()
                and sha(folder/name) == digest, 'recording changed or escaped its folder')
    meta = read(folder/'report.json')
    require((meta['task'], meta['seed'], meta['mode']) == (case['task'], case['seed'], 'realtime')
            and meta['initialization'] == 'reset_and_warmup'
            and meta['policy']['artifact_sha256'] == case['weights_sha256']
            and meta['policy'].get('uses_privileged_state') is False
            and meta['evaluator']['scene']['name'] == case['scene'],
            'recording identity or sensor-only student contract differs')
    controls = read(folder/'control.json')
    require(isinstance(anchors, (list, tuple)) and anchors
            and len(anchors) == len(set(anchors))
            and all(type(t) is int and 0 <= t < min(1500, len(controls)) and t % 5 == 0
                    for t in anchors), 'anchors must be actual live navigation decision ticks')
    with np.load(folder/'recording.npz', allow_pickle=False) as recording:
        captured_ticks = set(int(t) for t in recording['camera_ticks'])
    require(set(anchors) <= captured_ticks,
            'teacher anchors require actual recorded camera captures; missing frames cannot be invented')
    admission._initial_observation()  # Recheck pinned TRAIN metadata before allocating physics.
    wanted = set(anchors)
    visited, branches = [], []
    teacher = branch_env = None

    with ExitStack() as cleanup:
        def visit(snapshot, observation):
            nonlocal teacher, branch_env
            tick = snapshot['env']['tick']
            require(type(tick) is int and tick % 5 == 0
                    and (not visited and tick == 0 or visited and tick > visited[-1])
                    and observation.tick == observation.camera_tick == tick,
                    'teacher observations must preserve causal capture ticks')
            if branch_env is None:
                branch_env = cleanup.enter_context(environment_factory())
            branch_env.render_enabled = True
            branch_env.restore_state(snapshot)
            restored = branch_env.observe()
            np.testing.assert_array_equal(restored.rgb, observation.rgb)
            np.testing.assert_array_equal(restored.state, observation.state)
            require(restored.instruction == observation.instruction
                    and branch_env.tick == tick and not branch_env.terminated
                    and branch_env.task.duration_s == 30.
                    and branch_env.task.stop_hold_s == 2.
                    and tuple(branch_env.task.stop_distance_m) == (.25, .4),
                    'branch initialization or original task predicates changed')
            if teacher is None:
                admission.verify_initial(branch_env)
                teacher = teacher_factory(branch_env)
                close_teacher = getattr(teacher, 'close', None)
                if callable(close_teacher):
                    # Qualified obstacle teachers own a separate physics
                    # predictor. Release it before closing the branch env,
                    # including when replay or label validation fails.
                    cleanup.callback(close_teacher)
            # Updating teacher memory is confined to its own environment.
            before_history = branch_env.clone_state()
            observe_teacher(teacher, restored)
            _same_tree(branch_env.clone_state(), before_history)
            if tick in wanted:
                before = branch_env.clone_state()
                arrays, report = collect_branch(branch_env, teacher)
                _same_tree(branch_env.clone_state(), before)
                require(report['source_tick'] == tick
                        and tick < report['result']['tick'] <= 1500,
                        'teacher continuation reset or extended the original deadline')
                np.testing.assert_array_equal(arrays['rgb'][0], observation.rgb)
                np.testing.assert_array_equal(arrays['proprio'][0], observation.state[:48])
                require(int(arrays['ticks'][0]) == tick,
                        'teacher labels do not begin at the actual student state')
                proof = verify_branch(branch_env, arrays, report)
                _same_tree(branch_env.clone_state(), before)
                require(proof.get('complete_final_evaluator_exact') is True
                        and proof.get('teacher_queried_during_replay') is False
                        and proof.get('student_untouched') is True,
                        'independent command-only teacher replay is incomplete')
                branches.append(dict(source_tick=tick, tensors=copy.deepcopy(arrays),
                    report=copy.deepcopy(report), independent_replay=copy.deepcopy(proof),
                    training_release_allowed=False))
            visited.append(tick)

        original = _replay(folder, case, snapshot_observer=visit)
    require(original.get('status') == 'complete'
            and original.get('physical_replay') is True
            and original.get('camera_replay') is True
            and original.get('final_evaluator_exact') is True,
            'original student replay is incomplete; no labels can be released')
    require(wanted == {b['source_tick'] for b in branches}
            and original['snapshot_visitation']['ticks'] == visited,
            'not every requested teacher branch was visited and verified: '
            f'requested={sorted(wanted)}, branches={[b["source_tick"] for b in branches]}, '
            f'observed={visited}, replayed={original["snapshot_visitation"]["ticks"]}')
    require(all(sha(folder/name) == digest for name, digest in expected_files.items()),
            'source recording changed during labeling')
    return dict(schema='microduck_realtime_teacher_branches_v1', status='complete',
        training_admission=admission.binding(), input_files_sha256=dict(expected_files),
        student_checkpoint_sha256=case['weights_sha256'], original_student_replay=original,
        observed_camera_ticks=visited, branches=branches, original_deadline_tick=1500,
        training_release_allowed=False, gpu_allocated=False, new_training=False,
        formal_acceptance=False, model_promoted=False, hardware_validated=False)
