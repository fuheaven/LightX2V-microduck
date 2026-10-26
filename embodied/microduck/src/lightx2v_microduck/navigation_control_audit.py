"""Replay recorded navigation admission and consumption, including fallbacks.

This verifies control records against the declared buffer/clock contract.
It does not rerun physics, camera observations or model inference, verify the
historical runtime identity, or certify task success/final acceptance.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .navigation import NavigationBuffer, NavigationChunk, NAVIGATION_SOURCE_SHA256
from .navigation_action_contracts import action_contract
from .navigation_outcomes import require


RECEIPT_FIELDS = frozenset(('tick', 'accepted', 'error', 'reason', 'source_tick', 'phase',
                            'admission_monotonic_ns'))
EXECUTION_FIELDS = frozenset(('execution_source', 'source_tick', 'action_index',
                              'observation_age_ms', 'reason'))


def integer(value, name, *, minimum=0):
    require(type(value) is int and value >= minimum, 'invalid integer: '+name)
    return value


def finite(value, name):
    require(type(value) in (int, float) and math.isfinite(value), 'nonfinite or invalid: '+name)
    return value


def declared_buffer(episode):
    contract = episode['policy']['contract']
    action_contract(contract)
    require(episode.get('action_contract', contract) == contract, 'episode action contract differs')
    clock = episode['execution_clock']
    require(episode['mode'] in ('offline', 'realtime')
            and clock.get('schema') == 'microduck_navigation_execution_clock_v2'
            and clock.get('source_capture_timestamp_preserved') is True,
            'unsupported execution clock')
    basis = ('offline_observation_clock' if episode['mode'] == 'offline'
             else 'fresh_monotonic_at_admission_and_consumption')
    require(clock.get('expiry_basis') == basis, 'execution mode/clock differs')
    ticks, ns = clock['max_age_ticks'], clock['max_wall_age_ns']
    if ticks is None or ns is None:
        require(ticks is ns is None and clock.get('observation_age_rejection_enabled') is False
                and clock.get('ablation') == 'no_age_limit_finite_source_aligned_horizon',
                'disabled expiry requires an explicit no-age ablation')
        seconds = None
    else:
        integer(ticks, 'max_age_ticks', minimum=1)
        integer(ns, 'max_wall_age_ns', minimum=1)
        require(clock.get('observation_age_rejection_enabled', True) is True
                and 'ablation' not in clock, 'finite expiry cannot be labelled disabled')
        seconds = ns / 1e9
    buffer = NavigationBuffer(max_age_ticks=ticks, max_wall_age_s=seconds, contract=contract)
    require(buffer.max_wall_age_ns == ns, 'wall-time expiry cannot be represented exactly')
    return buffer


def audit_navigation_controls(episode, decisions, predictions, controls):
    """Use recorded timestamps; never replace late observations with fresh ones."""
    require(all(isinstance(v, list) for v in (decisions, predictions, controls))
            and len(decisions) == len(predictions), 'prediction/decision inventory differs')
    buffer = declared_buffer(episode)
    episode_id = episode['evaluator']['episode_id']
    require(isinstance(episode_id, str) and episode_id, 'recorded episode identity missing')
    buffer.reset(episode_id)
    count = len(controls)
    require(integer(episode['evaluator']['tick'], 'final tick') == count,
            'control trace is incomplete')
    require(type(episode.get('cancelled')) is bool, 'cancellation evidence missing')
    require(count > 0 or episode['cancelled'] or bool(episode['evaluator'].get('fault')),
            'empty trace has no recorded failure/cancellation')
    events = defaultdict(list)
    previous_tick = -1
    for decision, prediction in zip(decisions, predictions):
        require(isinstance(decision, dict) and isinstance(prediction, dict)
                and set(decision) == RECEIPT_FIELDS
                and {k: prediction.get(k) for k in RECEIPT_FIELDS} == decision,
                'prediction differs from its admission receipt')
        tick = integer(prediction['tick'], 'admission tick')
        camera_error = prediction.get('phase') == 'camera'
        require(tick >= previous_tick and (tick < count or count == tick == 0
                                          or camera_error and tick == count),
                'admission order or tick falls outside the trace')
        previous_tick = tick
        integer(prediction['source_tick'], 'source tick')
        integer(prediction['source_monotonic_ns'], 'source timestamp', minimum=1)
        if not camera_error:
            integer(prediction['camera_tick'], 'camera tick')
        require(isinstance(prediction.get('source_episode_id'), str) and prediction['source_episode_id'],
                'prediction source episode missing')
        require(type(prediction['accepted']) is bool
                and prediction['phase'] in ('startup', 'offline', 'realtime', 'preparation', 'camera'),
                'invalid prediction receipt')
        if camera_error:
            camera = episode.get('camera_delivery') or {}
            require(camera.get('schema') == 'microduck_snapshot_camera_v1'
                    and prediction['submitted_actions'] is None
                    and prediction['camera_tick'] is None and prediction['source_tick'] < count
                    and prediction['source_episode_id'] == episode_id,
                    'camera error lacks its snapshot source or submitted actions')
        require(finite(prediction['duration_ms'], 'inference duration') >= 0, 'negative inference duration')
        if prediction['phase'] == 'startup':
            require(tick == prediction['source_tick'] == 0 and not events[0],
                    'startup must be the first tick-zero admission')
        if episode['mode'] == 'offline':
            require(prediction['phase'] in ('startup', 'offline'), 'realtime event in an offline trace')
        else:
            require(prediction['phase'] != 'offline', 'offline event in a realtime trace')
        events[tick].append(prediction)

    rejected, fallback, errors = Counter(), Counter(), Counter()
    model_ticks = 0
    previous_execution_ns = 0
    previous_admission_ns = 0

    def admit(prediction, control=None):
        nonlocal previous_admission_ns
        actions = prediction['submitted_actions']
        if actions is None:
            require(prediction['accepted'] is False and prediction['reason'] == 'prediction_error'
                    and isinstance(prediction['error'], str) and prediction['error']
                    and prediction['admission_monotonic_ns'] is None
                    and prediction['valid_until_tick'] is None
                    and prediction['action_interval_ticks'] is None
                    and prediction.get('action_contract') is None
                    and prediction.get('diagnostics') is None,
                    'inference error cannot submit a plan')
            errors[prediction['phase']] += 1
            return
        require(prediction['error'] is None, 'a failed inference cannot also submit actions')
        require(prediction['phase'] not in ('preparation', 'camera'), 'observation failure cannot submit actions')
        contract = prediction.get('action_contract', buffer.contract)
        admission_ns = integer(prediction['admission_monotonic_ns'], 'admission timestamp', minimum=1)
        if control is not None:
            if prediction['phase'] == 'startup':
                require(admission_ns <= control['observation_capture_monotonic_ns'],
                        'startup admission follows the first control observation')
            else:
                require(control['observation_capture_monotonic_ns'] <= admission_ns
                        <= control['execution_monotonic_ns'], 'admission is outside its control tick')
        require(admission_ns >= previous_execution_ns, 'admission clock moved backward')
        require(admission_ns >= previous_admission_ns, 'admission receipt order moved backward')
        previous_admission_ns = admission_ns
        chunk = NavigationChunk(np.asarray(actions, np.float32), prediction['source_episode_id'],
            prediction['source_tick'], prediction['source_monotonic_ns'],
            integer(prediction['valid_until_tick'], 'valid-until tick', minimum=1),
            action_interval_ticks=integer(prediction['action_interval_ticks'], 'action interval', minimum=1),
            contract=contract)
        observation = SimpleNamespace(episode_id=episode_id, tick=prediction['tick'], monotonic_ns=admission_ns)
        accepted = buffer.submit(chunk, observation)
        require(accepted == prediction['accepted'] and buffer.last_reason == prediction['reason'],
                'recorded plan admission differs from buffer replay')
        if not accepted:
            rejected[buffer.last_reason] += 1

    for tick, control in enumerate(controls):
        require(isinstance(control, dict) and integer(control['tick'], 'control tick') == tick,
                'missing or reordered control tick')
        capture_ns = integer(control['observation_capture_monotonic_ns'], 'capture timestamp', minimum=1)
        execution_ns = integer(control['execution_monotonic_ns'], 'execution timestamp', minimum=1)
        require(previous_execution_ns <= capture_ns <= execution_ns, 'control clock moved backward')
        if episode['mode'] == 'offline':
            require(capture_ns == execution_ns, 'offline consumption must retain the observation clock')
        for prediction in events[tick]:
            admit(prediction, control)
        checked = buffer.chunk
        expected_source = None if checked is None else checked.source_monotonic_ns
        require(control['checked_plan_source_monotonic_ns'] == expected_source,
                'checked plan identity differs')
        intent, evidence = buffer.consume(SimpleNamespace(
            episode_id=episode_id, tick=tick, monotonic_ns=execution_ns))
        require({k: v for k, v in control.items() if k in EXECUTION_FIELDS} == evidence,
                'source, age, horizon index or fallback reason differs')
        require(intent.skill == control['skill'], 'executed skill differs')
        command = np.asarray(control['command'], np.float32)
        require(command.shape == (13,) and np.isfinite(command).all()
                and np.array_equal(command, intent.command), 'executed command differs')
        if evidence['execution_source'] == 'navigation_model':
            model_ticks += 1
        else:
            fallback[evidence['reason']] += 1
        previous_execution_ns = execution_ns
    # A camera failure discovered while draining after the final control tick
    # is still recorded, but cannot admit an action or alter execution.
    for prediction in events[count]:
        admit(prediction)
    require(integer(episode['fallback_control_ticks'], 'fallback count') == buffer.fallback_ticks
            and integer(episode['rejected_plans'], 'rejected-plan count') == buffer.rejected,
            'summary intervention counts differ from recorded execution')
    require(integer(episode.get('request_preparation_errors', 0), 'preparation error count')
            == errors['preparation'], 'preparation error count differs')
    camera = episode.get('camera_delivery')
    if camera is not None:
        require(integer(camera['failed_frames'], 'failed camera frames') == errors['camera'],
                'camera failure count differs from recorded receipts')
    require(type(episode.get('strict_model_success')) is bool
            and type(episode.get('task_success')) is bool, 'reported outcome is not boolean')
    require(type(episode['evaluator'].get('success')) is bool
            and episode['task_success'] == episode['evaluator']['success'],
            'reported outcome differs from evaluator')
    hazards = (buffer.fallback_ticks or episode['cancelled'] or episode['evaluator'].get('collision')
               or episode['evaluator'].get('ever_fallen') or episode['evaluator'].get('fault')
               or episode['evaluator'].get('recovery_events', 0) or episode.get('execution_error'))
    expected_strict = bool(episode['task_success'] and not hazards)
    require(episode['strict_model_success'] == expected_strict, 'reported strict success ignores an intervention')
    return dict(schema='microduck_navigation_control_audit_v1', status='complete',
        mode=episode['mode'], contract=buffer.contract, execution_clock=episode['execution_clock'],
        control_ticks=count, prediction_records=len(predictions), model_control_ticks=model_ticks,
        fallback_control_ticks=buffer.fallback_ticks, rejected_plans=buffer.rejected,
        rejections_by_reason=dict(rejected), fallbacks_by_reason=dict(fallback),
        prediction_errors_by_phase=dict(errors), cancelled=episode['cancelled'],
        reported_strict_model_success=episode['strict_model_success'], control_replay_exact=True,
        buffer_implementation_sha256=NAVIGATION_SOURCE_SHA256,
        physical_replay=False, camera_replay=False, model_inference_replayed=False,
        historical_runtime_identity_verified=False, cancellation_cause_verified=False,
        task_success_independently_verified=False, formal_acceptance=False, hardware_validated=False)


def audit_recording(folder):
    folder = Path(folder)
    files = {}
    loaded = {}
    for name in ('report.json', 'decisions.json', 'predictions.json', 'control.json'):
        data = (folder/name).read_bytes()
        files[name] = hashlib.sha256(data).hexdigest()
        def reject(value):
            raise ValueError('nonfinite JSON: '+value)
        loaded[name] = json.loads(data, parse_constant=reject)
    result = audit_navigation_controls(*(loaded[name] for name in
        ('report.json', 'decisions.json', 'predictions.json', 'control.json')))
    require(all(hashlib.sha256((folder/name).read_bytes()).hexdigest() == digest
                for name, digest in files.items()), 'recording changed during audit')
    result.update(input_files_sha256=files, audit_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode-folder', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'audit output already exists')
    result = audit_recording(args.episode_folder)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k:result[k] for k in ('status', 'control_ticks', 'fallback_control_ticks',
                                         'rejected_plans', 'formal_acceptance')}))


if __name__ == '__main__':
    main()
