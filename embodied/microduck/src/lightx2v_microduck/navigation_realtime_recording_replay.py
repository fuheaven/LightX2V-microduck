"""Replay every case of a sealed realtime job without filtering failed episodes.

The caller binds source/image/device provenance separately. This module checks
the complete recording inventory and independently reconstructs control,
physics, camera observations and task results. It never reruns model inference
or treats this verification as a new latency measurement or acceptance test.
"""
from __future__ import annotations

import json
from pathlib import Path

from .navigation_outcomes import read, require, sha
from .navigation_recording_replay import replay_recording
from .rendering import replay_camera_delivery, replay_camera_options, validate_snapshot_camera_workers


RECORD_FILES = ('report.json', 'decisions.json', 'predictions.json', 'control.json',
                'recording.npz', 'stage_timings.json')


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def verify_inputs(spec):
    require(spec['schema'] == 'microduck_realtime_recording_replay_job_v1', 'unknown replay specification')
    folder = Path(spec['report']).parent
    require(folder.is_absolute() and spec['profile'] == 'official_bam', 'unsupported recording location/profile')
    expected = {(task, seed) for task in spec['tasks'] for seed in spec['seeds']}
    require(len(expected) == len(spec['tasks']) * len(spec['seeds']) == spec['expected_cases'] > 0,
            'duplicate or empty expected case inventory')
    report = read(spec['report'])
    require(report['mode'] == 'realtime' and report['checkpoint_sha256'] == spec['weights_sha256']
            and report['completed_episodes'] == len(report['episodes']) == spec['expected_cases'],
            'realtime job is incomplete or uses different weights')
    require(validate_snapshot_camera_workers(report.get('snapshot_camera_workers', 0))
            == validate_snapshot_camera_workers(spec.get('snapshot_camera_workers', 0)),
            'realtime camera delivery differs from its replay specification')
    observed = [(e['task'], e['seed']) for e in report['episodes']]
    require(len(set(observed)) == len(observed) and set(observed) == expected,
            'recorded cases differ from the complete requested cohort')
    files = {str(folder/'report.json')}
    for task, seed in expected:
        files.update(str(folder/(task+'-seed'+str(seed))/name) for name in RECORD_FILES)
    # Native jobs additionally retain resolved configuration. Extra sealed
    # files are checked as well, but cannot replace any required recording.
    require(files <= set(spec['files']), 'sealed inventory omits an episode artifact')
    for name, digest in spec['files'].items():
        path = Path(name)
        require(path.is_relative_to(folder) and path.resolve().is_relative_to(folder.resolve())
                and not path.is_symlink(), 'recording escapes its job directory')
        require(sha(path) == digest, 'recording changed: '+name)
    for episode in report['episodes']:
        replay_camera_options(episode, spec)
        replay_camera_delivery(episode, spec)
        local = folder/(episode['task']+'-seed'+str(episode['seed']))
        require(read(local/'report.json') == episode, 'aggregate and per-case reports disagree')
        require(episode['execution_clock'] == spec['execution_clock']
                and episode['policy']['contract'] == spec['contract']
                and episode['policy']['artifact_sha256'] == spec['weights_sha256']
                and episode['evaluator']['scene']['name'] == spec['scene'],
                'case execution contract differs')
    return report


def replay_job(spec, output, *, diagnose_camera_mismatch=False, _replay=replay_recording):
    """Retain verification failures in the denominator and continue other cases."""
    report = verify_inputs(spec)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for episode in report['episodes']:
        name = episode['task']+'-seed'+str(episode['seed'])
        case = dict(profile=spec['profile'], mode='realtime', scene=spec['scene'],
                    task=episode['task'], seed=episode['seed'], contract=spec['contract'],
                    weights_sha256=spec['weights_sha256'], execution_clock=spec['execution_clock'],
                    image_size=spec['image_size'], warmup_ticks=spec['warmup_ticks'])
        if 'camera_offsamples' in spec:
            case['camera_offsamples'] = spec['camera_offsamples']
        if 'snapshot_camera_workers' in spec:
            case['snapshot_camera_workers'] = spec['snapshot_camera_workers']
        try:
            kwargs = {'diagnose_camera_mismatch': True} if diagnose_camera_mismatch else {}
            replay = _replay(Path(spec['report']).parent/name, case, **kwargs)
            require(replay['status'] in ('complete', 'incomplete_camera_replay')
                    and replay['physical_replay'] is True and replay['final_evaluator_exact'] is True
                    and replay['control_audit']['control_replay_exact'] is True
                    and replay['strict_success'] == episode['strict_model_success'],
                    'independent physical replay is incomplete')
            exact = replay['status'] == 'complete' and replay['camera_replay'] is True
            row = dict(task=episode['task'], seed=episode['seed'], verified=exact,
                       physical_verified=True,
                       recorded_strict_success=episode['strict_model_success'], replay=replay)
        except Exception as exc:
            row = dict(task=episode['task'], seed=episode['seed'], verified=False, physical_verified=False,
                       recorded_strict_success=episode['strict_model_success'],
                       error=type(exc).__name__+': '+str(exc))
        write_new(output/(name+'.json'), row)
        rows.append(dict(task=row['task'], seed=row['seed'], verified=row['verified'],
                         physical_verified=row['physical_verified'],
                         recorded_strict_success=row['recorded_strict_success'],
                         result_file=name+'.json', result_sha256=sha(output/(name+'.json'))))
    verify_inputs(spec)
    physical = [read(output/row['result_file'])['replay'] for row in rows if row['physical_verified']]
    verified = [r for r in physical if r['camera_replay']]
    complete = len(verified) == spec['expected_cases']
    result = dict(schema='microduck_realtime_recording_replay_result_v1',
        status='complete' if complete else 'incomplete', job=spec['job'],
        weights_sha256=spec['weights_sha256'], expected_cases=spec['expected_cases'],
        attempted_cases=len(rows), verified_cases=len(verified), failed_verification_cases=len(rows)-len(verified),
        physically_verified_cases=len(physical),
        recorded_strict_successes=sum(row['recorded_strict_success'] for row in rows),
        verified_strict_successes=sum(r['strict_success'] for r in verified),
        verified_task_failures=sum(not r['strict_success'] for r in verified),
        physically_recomputed_strict_successes=sum(r['strict_success'] for r in physical),
        exact_control_ticks=sum(r['exact_control_ticks'] for r in physical),
        exact_rgb_frames=sum(r['exact_rgb_frames'] for r in physical),
        attempted_rgb_frames=sum(r['attempted_rgb_frames'] for r in physical),
        camera_mismatch_frames=sum(len(r['camera_mismatches']) for r in physical),
        fallback_ticks=sum(r['fallback_control_ticks'] for r in physical),
        rejected_plans=sum(r['rejected_plans'] for r in physical),
        cancelled_cases=sum(r['cancelled'] for r in physical), cases=rows,
        physical_replay=len(physical)==spec['expected_cases'], camera_replay=complete,
        camera_mismatch_diagnostic_enabled=diagnose_camera_mismatch,
        task_success_independently_verified=complete,
        model_inference_replayed=False, model_rgb_tensor_verified=False,
        fresh_realtime_measurement=False, formal_acceptance=False, model_promoted=False,
        hardware_validated=False)
    write_new(output/'report.json', result)
    return result
