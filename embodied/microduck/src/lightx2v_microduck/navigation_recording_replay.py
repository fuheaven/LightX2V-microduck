"""Reconstruct recorded navigation physics, including rejected plans/fallbacks.

This replays recorded commands with CPU MuJoCo after auditing plan admission
and consumption. It does not rerun the model, measure fresh realtime latency,
prove historical runtime identity or issue capability acceptance.
"""
import argparse
import copy
from collections import defaultdict
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .contracts import SkillIntent
from .navigation_control_audit import audit_recording, integer
from .navigation_outcomes import read, require, sha
from .rendering import replay_camera_delivery, replay_camera_options


def canonical(world):
    return {k:v for k,v in json.loads(json.dumps(world, default=lambda x:x.tolist())).items()
            if k != 'episode_id'}


def replay_recording(folder, case, *, diagnose_camera_mismatch=False, _environment=None,
                     snapshot_observer=None, policy_observer=None):
    """The caller supplies the expected case, clock, weights and initialization."""
    if snapshot_observer is not None:
        require(callable(snapshot_observer) and not diagnose_camera_mismatch,
                'snapshot visitation requires exact camera verification and a callable observer')
    if policy_observer is not None:
        require(not diagnose_camera_mismatch and case['mode'] == 'offline'
                and callable(getattr(policy_observer, 'observe_prediction', None))
                and callable(getattr(policy_observer, 'observe_execution', None)),
                'policy replay requires exact offline camera and execution observers')
    folder = Path(folder)
    require(case['profile'] == 'official_bam' and case['mode'] in ('offline', 'realtime'),
            'unsupported replay profile/mode')
    integer(case['warmup_ticks'], 'warmup ticks')
    integer(case['image_size'], 'camera size', minimum=1)
    integer(case['seed'], 'seed')
    control_audit = audit_recording(folder)
    report = read(folder/'report.json')
    camera_options = replay_camera_options(report, case)
    camera_delivery = replay_camera_delivery(report, case)
    controls, predictions = (read(folder/name) for name in ('control.json', 'predictions.json'))
    for field in ('mode', 'task', 'seed', 'execution_clock'):
        require(report.get(field) == case[field], 'replay case differs: '+field)
    if 'hybrid_policy_sha256' in case:
        from .vlm_service import canonical as canonical_policy

        require('weights_sha256' not in case and report['policy'].get('family') == 'VLM+WAM'
                and hashlib.sha256(canonical_policy(report['policy'])).hexdigest() == case['hybrid_policy_sha256']
                and report['policy'].get('expected_service_identity') == case.get('service_identity_sha256'),
                'hybrid supervisor, skill bindings or policy identity differs')
    elif 'service_identity_sha256' in case:
        require('weights_sha256' not in case and report['policy'].get('family') == 'VLM'
                and report['policy'].get('expected_service_identity') == case['service_identity_sha256']
                and isinstance(case['service_identity_sha256'], str)
                and len(case['service_identity_sha256']) == 64,
                'VLM service identity differs')
    else:
        require(report['policy'].get('family') not in ('VLM', 'VLM+WAM')
                and report['policy'].get('artifact_sha256') == case['weights_sha256'],
                'model artifact differs')
    require(report['policy']['contract'] == case['contract']
            and report['evaluator']['scene']['name'] == case['scene']
            and report.get('initialization') == 'reset_and_warmup',
            'model, scene or initialization differs')
    files = dict(control_audit['input_files_sha256'])
    files['recording.npz'] = sha(folder/'recording.npz')
    with np.load(folder/'recording.npz', allow_pickle=False) as archive:
        rgb, camera_ticks, qpos, ticks = (archive[k].copy() for k in
            ('rgb', 'camera_ticks', 'qpos', 'physics_ticks'))
    count = len(controls)
    require(count > 0, 'empty physical recording requires separate initialization evidence')
    from .navigation_camera_recording import verify_camera_clock
    camera_clock = verify_camera_clock(folder, report, camera_ticks)
    terminal_tick = camera_clock.get('terminal_unexecuted_observation_tick')
    physical_samples = count + int(terminal_tick is not None)
    np.testing.assert_array_equal(ticks, np.arange(physical_samples))
    if terminal_tick is not None:
        files['execution_failure.json'] = camera_clock['terminal_failure_sha256']
    camera_indices = {int(tick): index for index, tick in enumerate(camera_ticks)}
    require(qpos.ndim == 2 and len(qpos) == physical_samples and np.isfinite(qpos).all(),
            'physical recording is incomplete/nonfinite')
    require(rgb.shape == (len(camera_ticks), case['image_size'], case['image_size'], 3)
            and rgb.dtype == np.uint8, 'camera recording is incomplete')
    sources = defaultdict(list)
    for prediction in predictions:
        source = prediction['source_tick']
        camera_failure = (prediction.get('phase') == 'camera'
                          and source in camera_clock['missing_ticks']
                          and prediction.get('camera_tick') is None
                          and prediction.get('accepted') is False
                          and prediction.get('submitted_actions') is None)
        require(((source in camera_indices and prediction['camera_tick'] == source) or camera_failure)
                and prediction['source_episode_id'] == report['evaluator']['episode_id'],
                'prediction source is not a recorded observation')
        sources[source].append(prediction)
    if _environment is None:
        from .env import MicroDuckEnv
        _environment = MicroDuckEnv
    observed_predictions = 0
    snapshot_ticks = []
    camera_mismatches = []
    with _environment(profile=case['profile'], scene=case['scene'], render=False,
                      image_size=case['image_size'], **camera_options) as env, ExitStack() as cleanup:
        env.reset(case['seed'], case['task'])
        for _ in range(case['warmup_ticks']):
            env.step(SkillIntent('stand', np.zeros(13, np.float32), env.episode_id, env.tick, 1))
        env.begin_task_at_current_pose()
        initial_world, initial_qpos = canonical(env.evaluator_state()), env.physics.data.qpos.copy()
        camera = None
        for tick, control in enumerate(controls):
            require(env.tick == tick and not env.terminated, 'replay terminated before recorded execution')
            np.testing.assert_array_equal(env.physics.data.qpos, qpos[tick], err_msg=f'physical state at tick {tick}')
            env.render_enabled = tick in camera_indices and camera is None
            observation = env.observe()
            if tick == 0:
                if report.get('camera_rendering') is not None:
                    require(env.camera_configuration() == report['camera_rendering'],
                            'actual replay camera configuration differs from recording')
                initial_proprio = observation.state[:48].copy()
                initial_instruction = observation.instruction
                if camera_delivery['workers']:
                    from .snapshot_camera import SnapshotCamera

                    camera = SnapshotCamera(env.physics, env.episode_id, workers=camera_delivery['workers'])
                    def close_camera():
                        require(camera.close(), 'replay camera worker did not stop')
                    cleanup.callback(close_camera)
            elif tick in camera_indices and camera is not None:
                # Replay has no realtime claim. Wait for the independently
                # reconstructed snapshot rather than substituting a different
                # OpenGL context or relaxing the recorded pixel equality check.
                require(camera.submit(observation), 'replay camera unexpectedly busy')
                deadline = time.monotonic() + 30.
                frames = []
                while not frames and time.monotonic() < deadline:
                    frames = camera.poll()
                    if not frames:
                        time.sleep(.001)
                require(len(frames) == 1 and frames[0].error is None,
                        'replay snapshot capture failed or timed out')
                require(frames[0].observation.tick == tick
                        and frames[0].observation.monotonic_ns == observation.monotonic_ns,
                        'replay snapshot source changed')
                observation = frames[0].observation
            if tick in camera_indices:
                recorded_rgb = rgb[camera_indices[tick]]
                if not np.array_equal(observation.rgb, recorded_rgb):
                    if not diagnose_camera_mismatch:
                        np.testing.assert_array_equal(observation.rgb, recorded_rgb,
                                                      err_msg=f'camera at tick {tick}')
                    difference = observation.rgb.astype(np.int16) - recorded_rgb.astype(np.int16)
                    camera_mismatches.append(dict(tick=tick,
                        differing_coordinates=int(np.count_nonzero(difference)),
                        maximum_absolute_difference=int(np.abs(difference).max()),
                        replayed_rgb_sha256=hashlib.sha256(observation.rgb.tobytes()).hexdigest(),
                        recorded_rgb_sha256=hashlib.sha256(recorded_rgb.tobytes()).hexdigest()))
            sensor_observer = getattr(policy_observer, 'observe_sensors', None)
            if sensor_observer is not None:
                sensor_observer(copy.deepcopy(observation), copy.deepcopy(control), copy.deepcopy(sources[tick]))
            for prediction in sources[tick]:
                np.testing.assert_array_equal(observation.state[:48],
                    np.asarray(prediction['observed_proprio48'], np.float32),
                    err_msg=f'prediction proprio at source tick {tick}')
                require(observation.instruction == prediction['observed_instruction'],
                        'prediction instruction differs from replayed task')
                if policy_observer is not None:
                    policy_observer.observe_prediction(copy.deepcopy(observation), copy.deepcopy(prediction))
                observed_predictions += 1
            if snapshot_observer is not None and tick in camera_indices:
                # A background camera capture belongs to this physical tick,
                # but is not stored in the main environment's RGB cache. Put
                # that independently verified frame into a detached snapshot.
                # Never expose the live replay environment to a label worker.
                snapshot = copy.deepcopy(env.clone_state())
                snapshot['camera_frames'] = {
                    **snapshot.get('camera_frames', {}),
                    'tick': tick, 'head': observation.rgb.copy(),
                }
                snapshot_observer(snapshot, copy.deepcopy(observation))
                snapshot_ticks.append(tick)
            env.render_enabled = False
            intent = SkillIntent(control['skill'], np.asarray(control['command'], np.float32),
                                 env.episode_id, tick, 1)
            after_execution = env.step(intent)
            if policy_observer is not None:
                policy_observer.observe_execution(copy.deepcopy(observation), copy.deepcopy(after_execution),
                                                  copy.deepcopy(intent), copy.deepcopy(control))
        if terminal_tick is not None:
            require(env.tick == terminal_tick and camera is None,
                    'terminal observation is not the unexecuted offline source tick')
            np.testing.assert_array_equal(env.physics.data.qpos, qpos[-1], err_msg='terminal physical state')
            env.render_enabled = True
            terminal_observation = env.observe()
            np.testing.assert_array_equal(terminal_observation.rgb, rgb[camera_indices[terminal_tick]],
                                          err_msg='terminal camera observation')
            if policy_observer is not None:
                require(callable(getattr(policy_observer, 'observe_failure', None)),
                        'failed model request requires a failure observer')
                policy_observer.observe_failure(copy.deepcopy(terminal_observation),
                                                read(folder / 'execution_failure.json'))
        final = canonical(env.evaluator_state())
        require(final == canonical(report['evaluator']), 'independent physical/evaluator result differs')
    require(observed_predictions == len(predictions), 'some prediction sources were not replayed')
    require(all(sha(folder/name) == digest for name,digest in files.items()),
            'recording changed during physical replay')
    result = dict(schema='microduck_navigation_recording_replay_v1',
        status='incomplete_camera_replay' if camera_mismatches else 'complete',
        case=case, input_files_sha256=files, control_audit=control_audit,
        exact_control_ticks=count, exact_rgb_frames=len(rgb)-len(camera_mismatches),
        attempted_rgb_frames=len(rgb), camera_mismatches=camera_mismatches,
        camera_mismatch_diagnostic_enabled=diagnose_camera_mismatch,
        camera_replay_delivery=camera_delivery,
        camera_clock=camera_clock,
        exact_prediction_source_observations=observed_predictions,
        terminal_unexecuted_observation_tick=terminal_tick,
        physical_replay=True, camera_replay=not camera_mismatches, final_evaluator_exact=True,
        initial_world=initial_world, initial_qpos=initial_qpos.tolist(),
        initial_proprio48=initial_proprio.tolist(), initial_instruction=initial_instruction,
        strict_success=report['strict_model_success'], task_success=final['success'],
        cancelled=report['cancelled'], fallback_control_ticks=control_audit['fallback_control_ticks'],
        rejected_plans=control_audit['rejected_plans'],
        model_inference_replayed=False, model_rgb_tensor_verified=False,
        historical_runtime_identity_verified=False, fresh_realtime_measurement=False,
        cancellation_cause_verified=False, formal_acceptance=False, hardware_validated=False,
        replay_module_sha256=sha(__file__))
    if snapshot_observer is not None:
        result['snapshot_visitation'] = {
            'ticks': snapshot_ticks, 'detached_physics_and_observation': True,
            'captured_rgb_preserved': True, 'training_release_allowed': False,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode-folder', type=Path, required=True)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--case-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--diagnose-camera-mismatch', action='store_true',
        help='Continue physics diagnostics while retaining camera mismatch as incomplete verification')
    args = parser.parse_args()
    require(not args.output.exists(), 'replay output already exists')
    require(sha(args.case) == args.case_sha256, 'replay case changed')
    case = read(args.case)
    result = replay_recording(args.episode_folder, case,
                              diagnose_camera_mismatch=args.diagnose_camera_mismatch)
    require(sha(args.case) == args.case_sha256, 'replay case changed during execution')
    result['case_sha256'] = args.case_sha256
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k:result[k] for k in ('status', 'exact_control_ticks', 'exact_rgb_frames',
                                         'strict_success', 'fallback_control_ticks', 'formal_acceptance')}))


if __name__ == '__main__':
    main()
