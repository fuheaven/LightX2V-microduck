"""Bind recorded sensor frames to the arguments of submitted model calls.

This verifies RGB grids, raw proprioception and language at the core-call
boundary. It does not rerender images, reproduce neural inference, verify
normalized GPU tensors, or establish the authenticity of a historical runtime.
The caller must bind runtime sources and independently verify physical replay.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .navigation_history import compose_history_grid, history_spec_from_contract
from .navigation_outcomes import read, require, sha


SCHEMA = 'microduck_navigation_core_inputs_v1'
RECORDING_CONTRACT = dict(schema=SCHEMA, scope='core_predict_action_chunk_arguments',
                          before_and_after_call_hashes=True,
                          instrumentation_included_in_latency=True,
                          normalized_gpu_tensors_recorded=False)


def argument_receipt(observation, state):
    """Hash the exact immutable RGB and raw state passed to the model adapter."""
    rgb, state = np.asarray(observation.rgb), np.asarray(state)
    require(rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[-1] == 3
            and min(rgb.shape) > 0, 'core input must be HWC uint8 RGB')
    require(state.dtype == np.float32 and state.shape == (48,) and np.isfinite(state).all(),
            'core input must contain finite raw proprio48')
    require(isinstance(observation.instruction, str) and bool(observation.instruction),
            'core input requires language')
    return dict(schema=SCHEMA, source_episode_id=observation.episode_id,
        source_tick=observation.tick, source_monotonic_ns=observation.monotonic_ns,
        camera_tick=observation.camera_tick,
        rgb=dict(shape=list(rgb.shape), dtype='uint8',
                 sha256=hashlib.sha256(rgb.tobytes(order='C')).hexdigest()),
        proprio48=dict(shape=[48], dtype='float32_le',
                       sha256=hashlib.sha256(state.astype('<f4', copy=False).tobytes(order='C')).hexdigest()),
        instruction_sha256=hashlib.sha256(observation.instruction.encode('utf-8')).hexdigest())


def configure_recording(policy, enabled):
    require(type(enabled) is bool, 'model input recording must be explicitly boolean')
    policy.record_model_inputs = enabled
    if enabled:
        policy.metadata['model_input_recording'] = dict(RECORDING_CONTRACT)


def audit_recorded_inputs(folder, *, expected_history_contract, expected_input_intervention=None,
                          expected_camera_loss_recovery=None):
    """Check every submitted prediction, including rejected/expired plans.

Prediction errors retain their denominator and are not claimed as verified
model inputs. Warmups and unsubmitted worker replies are outside this scope.
GPU rasterization differences cannot be hidden by this separate input audit.
"""
    from types import SimpleNamespace
    from .navigation_visual_memory import TargetMemoryGridSpec, compose_target_memory_grid, target_memory_ticks

    folder = Path(folder)
    files = {name: sha(folder/name) for name in ('report.json', 'predictions.json', 'recording.npz')}
    episode, predictions = read(folder/'report.json'), read(folder/'predictions.json')
    metadata = episode['policy']
    require(metadata.get('model_input_recording') == RECORDING_CONTRACT,
            'recording lacks explicit core input receipts')
    require(metadata.get('history_contract') == expected_history_contract,
            'recorded input history differs from the required contract')
    require(metadata.get('input_intervention') == expected_input_intervention,
            'intervened inputs require the explicitly expected audit contract')
    require(metadata.get('camera_loss_recovery') == expected_camera_loss_recovery,
            'camera loss recovery requires its explicitly expected runtime contract')
    recovery = expected_camera_loss_recovery
    if recovery is not None:
        from .navigation_camera_recovery import CAMERA_LOSS_RECOVERY
        require(recovery == CAMERA_LOSS_RECOVERY and expected_input_intervention is None,
                'unsupported camera recovery or combined input intervention')
    intervention = expected_input_intervention
    if intervention is not None:
        require(isinstance(intervention, dict) and set(intervention) == {
            'mask_rgb', 'instruction_color', 'evaluator_target_unchanged', 'sensor_recording_unchanged',
            'application_stage', 'input_preparation_unchanged', 'not_capability_acceptance'},
            'unsupported model input intervention')
        stage = intervention['application_stage']
        require(type(intervention['mask_rgb']) is bool
                and intervention['instruction_color'] in (None, 'red', 'blue', 'yellow')
                and stage in ('observation', 'prepared_request')
                and intervention['input_preparation_unchanged'] is (stage == 'prepared_request')
                and all(intervention[key] is True for key in ('evaluator_target_unchanged',
                    'sensor_recording_unchanged', 'not_capability_acceptance')),
                'intervention changed sensor, evaluator or preparation semantics')
    spec = None if expected_history_contract is None else history_spec_from_contract(expected_history_contract)
    require(recovery is None or isinstance(spec, TargetMemoryGridSpec),
            'camera recovery requires target memory history')
    with np.load(folder/'recording.npz', allow_pickle=False) as archive:
        rgb, ticks = archive['rgb'].copy(), archive['camera_ticks'].copy()
    from .navigation_camera_recording import verify_camera_clock
    camera_clock = verify_camera_clock(folder, episode, ticks)
    require(rgb.shape == (len(ticks), 256, 256, 3) and rgb.dtype == np.uint8,
            'recording lacks original 256px head-camera images')
    require(isinstance(predictions, list) and bool(predictions), 'no recorded prediction inventory')
    submitted = [row for row in predictions if row.get('submitted_actions') is not None]
    errors = [row for row in predictions if row.get('submitted_actions') is None]
    for row in errors:
        require(row.get('accepted') is False and row.get('reason') == 'prediction_error'
                and isinstance(row.get('error'), str) and bool(row['error']),
                'missing prediction is not a declared error')
    instruction = predictions[0]['observed_instruction']
    model_instruction = instruction
    if intervention is not None and intervention['instruction_color'] is not None:
        from .tasks import get_task

        model_instruction = get_task('find_' + intervention['instruction_color'] + '_ball').instruction
    before_preparation = intervention is not None and intervention['application_stage'] == 'observation'
    history_rgb = np.zeros_like(rgb) if before_preparation and intervention['mask_rgb'] else rgb
    history_instruction = model_instruction if before_preparation else instruction
    history_ticks = ticks
    if isinstance(spec, TargetMemoryGridSpec) and not camera_clock['complete'] and recovery is None:
        # This memory contract rejects every later request after a missing
        # capture. Verify the submitted prefix and retain subsequent errors;
        # neither fill the gap nor silently introduce a different memory rule.
        first_missing = camera_clock['missing_ticks'][0]
        require(all(type(row['source_tick']) is int and row['source_tick'] < first_missing
                    for row in submitted),
                'target-memory prediction crossed an unavailable camera prefix')
        prefix = ticks < first_missing
        history_rgb, history_ticks = history_rgb[prefix], ticks[prefix]
    memory = (target_memory_ticks(history_rgb, history_ticks, history_instruction)
              if isinstance(spec, TargetMemoryGridSpec) and recovery is None else None)
    recovered = None
    if recovery is not None:
        from .navigation_camera_recovery import DeclaredLossTargetCameraHistory, loss_diagnostics
        require(episode.get('mode') == 'realtime' and episode.get('camera_delivery', {}).get('schema')
                == 'microduck_snapshot_camera_v1', 'camera recovery needs recorded snapshot delivery')
        files['camera_delivery.json'] = sha(folder/'camera_delivery.json')
        receipts = read(folder/'camera_delivery.json')
        captured = {r['source_tick']: r['capture_ns'] for r in receipts if r['status'] == 'delivered'}
        startup = [row for row in submitted if row['source_tick'] == 0]
        require(len(startup) == 1, 'camera recovery audit needs the measured initial input')
        captured[0] = startup[0]['source_monotonic_ns']
        require(all(type(row['source_tick']) is int for row in submitted), 'invalid camera source tick')
        recovered = DeclaredLossTargetCameraHistory(spec)
        recovered.reset(episode['evaluator']['episode_id'])
        frame_index = 0
    checked = []
    for row in sorted(submitted, key=lambda r: r['source_tick']) if recovered is not None else submitted:
        tick = row['source_tick']
        require(type(tick) is int and tick in ticks and row['camera_tick'] == tick,
                'prediction does not identify a measured source frame')
        require(row['source_episode_id'] == episode['evaluator']['episode_id']
                and row['observed_instruction'] == instruction,
                'prediction source episode or instruction changed')
        if recovered is not None:
            while frame_index < len(ticks) and ticks[frame_index] <= tick:
                at = int(ticks[frame_index])
                require(at in captured, 'recorded frame lacks its camera capture timestamp')
                recovered.ingest(SimpleNamespace(episode_id=episode['evaluator']['episode_id'],
                    tick=at, camera_tick=at, monotonic_ns=captured[at], rgb=history_rgb[frame_index],
                    instruction=history_instruction), receipts)
                frame_index += 1
            grid, selected, padding = recovered.compose(tick)
            require(row['source_monotonic_ns'] == captured[tick]
                    and row['diagnostics'].get('camera_loss_recovery') == loss_diagnostics(recovered.declared_losses),
                    'camera recovery changed capture time or declared loss history')
        elif spec is None:
            grid, selected, padding = history_rgb[int(np.searchsorted(ticks, tick))], None, None
        elif memory is None:
            grid, selected, padding = compose_history_grid(history_rgb, history_ticks, tick, spec)
        else:
            grid, selected, padding = compose_target_memory_grid(history_rgb, history_ticks, tick, spec, memory)
        require(row['diagnostics'].get('input_intervention') == intervention,
                'prediction intervention differs from the expected input contract')
        if intervention is not None and intervention['application_stage'] == 'prepared_request' and intervention['mask_rgb']:
            grid = np.zeros_like(grid)
        observation = SimpleNamespace(rgb=grid, instruction=model_instruction,
            episode_id=row['source_episode_id'], tick=tick, camera_tick=row['camera_tick'],
            monotonic_ns=row['source_monotonic_ns'])
        expected = argument_receipt(observation, np.asarray(row['observed_proprio48'], np.float32))
        require(row['diagnostics'].get('model_core_inputs') == expected,
                'recorded frames, state or language differ from actual core-call arguments')
        if spec is not None:
            require(row['diagnostics']['history_source_ticks'] == selected.tolist()
                    and row['diagnostics']['history_padding'] == padding.tolist(),
                    'recorded causal history selection differs')
        checked.append(dict(source_tick=tick, receipt_tick=row['tick'],
                            accepted=row['accepted'], rgb_sha256=expected['rgb']['sha256']))
    require(all(sha(folder/name) == digest for name,digest in files.items()),
            'recording changed during model input audit')
    result = dict(schema='microduck_navigation_recorded_core_input_audit_v1',
        status=('complete' if camera_clock['complete'] else 'complete_with_recorded_camera_loss')
            if submitted else 'no_submitted_predictions',
        camera_clock=camera_clock,
        input_files_sha256=files, prediction_records=len(predictions),
        submitted_predictions=len(submitted), accepted_predictions=sum(row['accepted'] for row in submitted),
        rejected_predictions=sum(not row['accepted'] for row in submitted),
        prediction_errors_without_input_verification=len(errors), checked=checked,
        submitted_core_call_inputs_verified=bool(submitted),
        warmup_inputs_verified=False, unsubmitted_worker_replies_verified=False,
        model_rgb_tensor_verified=False, neural_inference_replayed=False,
        physical_replay=False, camera_rerender_verified=False,
        historical_runtime_identity_verified=False, formal_acceptance=False,
        audit_module_sha256=sha(__file__))
    if intervention is not None:
        result.update(input_intervention=dict(intervention),
                      submitted_input_intervention_verified=bool(submitted))
    if recovery is not None:
        result.update(camera_loss_recovery=dict(recovery),
                      submitted_camera_loss_recovery_verified=bool(submitted))
    return result
