"""Strict physical/RGB/model-input replay for complete offline cohort records.

All submitted predictions, including rejected ones, are bound to their raw
outputs. Prediction errors and cancellations remain in the record. Missing records
or any physical/camera mismatch make verification fail; there is no diagnostic
camera-tolerance mode on this cohort path.
"""
import hashlib
import importlib.util
from pathlib import Path

import numpy as np

from .navigation_action_contracts import action_contract
from .navigation_history import HistoryGridSpec, compose_history_grid, history_spec_from_contract, verify_evaluation_history
from .navigation_outcomes import read, require, recording_inventory_valid, sha
from .navigation_recording_replay import replay_recording


def replay_history(protocol):
    """Legacy cohorts keep fixed timestamps; a changed input needs its trained receipt."""
    history = protocol.get('input_history')
    if history is None:
        return HistoryGridSpec((300, 150, 50, 0))
    spec = history_spec_from_contract(history)
    receipt_path = protocol.get('checkpoint_receipt')
    require(isinstance(receipt_path, str) and receipt_path
            and sha(receipt_path) == protocol.get('checkpoint_receipt_sha256'),
            'history replay requires its bound trained receipt')
    receipt = read(receipt_path)
    require(isinstance(protocol.get('contracts'), list) and len(protocol['contracts']) == 1,
            'history replay requires one execution contract')
    verify_evaluation_history(receipt, spec, weights_sha256=protocol['weights_sha256'],
                              stats_sha256=protocol['stats_sha256'])
    from .navigation_evaluation_contract import checkpoint_execution
    contract, _ = checkpoint_execution(receipt, weights_sha256=protocol['weights_sha256'],
        stats_sha256=protocol['stats_sha256'], requested=protocol['contracts'][0])
    require(protocol['contracts'] == [contract], 'history replay execution contract differs')
    return spec


def bind_model_traces(folder, episode, predictions, physical, *, history_spec=None):
    """Bind one warmup and all recorded raw predictions to measured sensors."""
    spec = HistoryGridSpec((300,150,50,0)) if history_spec is None else history_spec
    require(episode['policy']['history_contract'] == spec.as_dict(), 'model history differs')
    with np.load(folder/'recording.npz', allow_pickle=False) as archive:
        rgb, ticks = archive['rgb'].copy(), archive['camera_ticks'].copy()
    from .navigation_visual_memory import TargetMemoryGridSpec, compose_target_memory_grid, target_memory_ticks
    memory = (target_memory_ticks(rgb, ticks, physical['initial_instruction'])
              if isinstance(spec, TargetMemoryGridSpec) else None)

    def compose(tick):
        return (compose_target_memory_grid(rgb, ticks, tick, spec, memory) if memory is not None
                else compose_history_grid(rgb, ticks, tick, spec))
    with np.load(folder/'unbounded-model-predictions.npz', allow_pickle=False) as archive:
        actions, states = archive['action'].copy(), archive['state'].copy()
    inputs = read(folder/'unbounded-model-inputs.json')
    submitted = [p for p in predictions if p['submitted_actions'] is not None]
    for prediction in predictions:
        if prediction['submitted_actions'] is None:
            require(prediction.get('accepted') is False
                    and prediction.get('reason') == 'prediction_error'
                    and isinstance(prediction.get('error'), str) and prediction['error'],
                    'missing raw output lacks a recorded prediction error')
    require(actions.shape == (len(submitted)+1,32,5) and states.shape == (len(submitted)+1,48)
            and actions.dtype == states.dtype == np.float32
            and np.isfinite(actions).all() and np.isfinite(states).all()
            and isinstance(inputs,list) and len(inputs) == len(submitted)+1,
            'model traces omit warmup or submitted predictions')
    require(all(isinstance(x,dict) and set(x)=={'rgb_sha256','instruction'} for x in inputs),
            'raw model input inventory differs')
    grid, _, _ = compose(0)
    require(inputs[0] == dict(rgb_sha256=hashlib.sha256(grid.tobytes()).hexdigest(),
                              instruction=physical['initial_instruction']),
            'warmup input differs from the initial measured observation')
    np.testing.assert_array_equal(states[0], np.asarray(physical['initial_proprio48'],np.float32))
    for index, prediction in enumerate(submitted,1):
        contract = prediction['action_contract']
        require(contract == episode['policy']['contract'] or
                (prediction['accepted'] is False and prediction['reason']=='wrong_contract'),
                'different action semantics were executed')
        lower,upper = action_contract(contract).bounds()
        np.testing.assert_array_equal(np.clip(actions[index],lower,upper),
            np.asarray(prediction['submitted_actions'],np.float32))
        np.testing.assert_array_equal(states[index],np.asarray(prediction['observed_proprio48'],np.float32))
        require(prediction['observed_instruction'] == physical['initial_instruction'],
                'task instruction changed without a new episode')
        grid,history,padding = compose(prediction['source_tick'])
        require(inputs[index] == dict(rgb_sha256=hashlib.sha256(grid.tobytes()).hexdigest(),
                                       instruction=prediction['observed_instruction']),
                'model image or language differs from the source observation')
        diagnostics=prediction['diagnostics']
        require(diagnostics['history_source_ticks']==history.tolist()
                and diagnostics['history_padding']==padding.tolist(), 'history selection differs')
    return dict(exact_history_grids=len(submitted), model_input_trace_records=len(inputs),
        submitted_prediction_records=len(submitted), prediction_records=len(predictions),
        prediction_error_records=len(predictions)-len(submitted),
        raw_forward_predictions_above_legacy=int((actions[1:,:,0]>np.float32(.3)).sum()),
        initial_action=actions[0],initial_input=inputs[0])


def replay(root, row, protocol):
    require(protocol.get('mode','offline') == 'offline', 'cohort replay requires its offline protocol')
    relative=row['path'];root=Path(root).resolve();folder=(root/relative).resolve()
    require(not Path(relative).is_absolute() and root in folder.parents, 'episode path escapes root')
    require(recording_inventory_valid(row['files']), 'episode file inventory differs')
    for name,digest in row['files'].items():
        file=(folder/name).resolve()
        require(folder in file.parents and sha(file)==digest, 'recording changed: '+name)
    episode=read(folder/'report.json')
    require('cancel.request' not in row['files'] or episode['cancelled'] is True,
            'retained cancellation request lacks a cancelled outcome')
    require(episode['policy']['uses_privileged_state'] is False, 'privileged policy cannot enter this cohort')
    count=episode['evaluator']['tick']
    require(type(count) is int and 1<=count<=1500 and row['ticks']==count,
            'recorded episode duration differs')
    expected_clock=dict(schema='microduck_navigation_execution_clock_v2',
        expiry_basis='offline_observation_clock',max_age_ticks=25,max_wall_age_ns=10**18,
        source_capture_timestamp_preserved=True)
    case=dict(profile='official_bam',mode='offline',scene=row['scene'],seed=row['seed'],task=row['task'],
        contract=row['contract'],weights_sha256=protocol['weights_sha256'],execution_clock=expected_clock,
        image_size=256,warmup_ticks=protocol['warmup_ticks'])
    physical=replay_recording(folder,case)
    require(physical['status']=='complete' and physical['physical_replay'] and physical['camera_replay']
            and physical['final_evaluator_exact'], 'physical/camera replay is incomplete')
    predictions=read(folder/'predictions.json')
    binding=bind_model_traces(folder,episode,predictions,physical,history_spec=replay_history(protocol))
    require(binding['raw_forward_predictions_above_legacy']==row['raw_forward_predictions_above_legacy'],
            'raw horizon diagnostic differs')
    final=episode['evaluator']
    require(physical['strict_success']==row['strict_success']
            and row['collision']==final['collision'] and row['fallen']==final['ever_fallen'],
            'recorded outcome differs')
    require(all(sha(folder/name)==digest for name,digest in row['files'].items()),
            'recording changed during cohort replay')
    controls=read(folder/'control.json')
    result=dict(schema='microduck_navigation_cohort_case_replay_v2',
        scene=row['scene'],seed=row['seed'],task=row['task'],contract=row['contract'],
        exact_control_ticks=count,exact_rgb_frames=physical['exact_rgb_frames'],
        exact_history_grids=binding['exact_history_grids'],
        model_input_trace_records=binding['model_input_trace_records'],
        submitted_prediction_records=binding['submitted_prediction_records'],
        prediction_records=binding['prediction_records'],prediction_error_records=binding['prediction_error_records'],
        model_input_trace_complete=True,final_evaluator_exact=True,submitted_actions_exact=True,
        buffer_execution_exact=True,strict_success=physical['strict_success'],
        collision=final['collision'],fallen=final['ever_fallen'],recovery_events=final['recovery_events'],
        fallback_ticks=physical['fallback_control_ticks'],rejected_plans=physical['rejected_plans'],
        cancelled=episode['cancelled'],input_files_sha256=row['files'],
        input_history=episode['policy']['history_contract'],
        physically_executed_forward_above_legacy_ticks=sum(c['command'][0]>float(np.float32(.3)) for c in controls),
        all_horizon_forward_above_legacy_coordinates=binding['raw_forward_predictions_above_legacy'])
    initial=dict(world=physical['initial_world'],qpos=np.asarray(physical['initial_qpos']),
        state=np.asarray(physical['initial_proprio48'],np.float32),action=binding['initial_action'],
        input=binding['initial_input'])
    return result,initial


def replay_rows(root,rows,protocol,workers):
    # Reuse the existing bounded scheduling/failure-order implementation. Its
    # source remains unchanged and is included in the frozen source manifest.
    file=Path(__file__).resolve().parents[4]/'scripts/microduck/assess_navigation_contract_cohort.py'
    spec=importlib.util.spec_from_file_location('_microduck_cohort_scheduler',file)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    yield from module.replay_rows(root,rows,protocol,workers,_replay=replay)
