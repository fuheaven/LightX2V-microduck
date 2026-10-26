"""Admission and completion for a fresh current-correction candidate stage."""
import copy
from collections import Counter
import math
from pathlib import Path

from .navigation_correction_model import execution_parent_sources


def require(ok, message):
    if not ok:
        raise ValueError(message)


def preflight_plan_gate(plan, queue):
    require(queue.get('maximum_optimizer_steps') == 250 and queue.get('maximum_gpu_allocations') == 1
            and queue.get('maximum_concurrent_gpus') == 1 and queue.get('model_promotion_allowed') is False,
            'one bounded candidate without promotion required')
    require(plan.get('schema') == 'microduck_realtime_correction_model_preflight_v1'
            and isinstance(plan.get('parent_recipe_snapshot'), str)
            and isinstance(plan.get('data_admission_sha256'), str),
            'complete current-cohort model preflight required')
    require(plan.get('parent_weights_sha256') == queue['initializer_sha256']
            and plan.get('retained_weights_sha256') == queue['retained_sha256']
            and plan.get('image') == queue['image']
            and plan.get('train_output') == queue['train_output'], 'model roles or output changed')
    require(plan.get('training_steps') == 250 and plan.get('maximum_optimizer_steps') == 2
            and plan.get('maximum_gpu_allocations') == 2 and plan.get('maximum_concurrent_gpus') == 1
            and plan.get('model_promotion_allowed') is False and plan.get('formal_acceptance') is False,
            'preflight execution scope differs')


def candidate_plan(previous, queue, files):
    preflight_plan_gate(previous, queue)
    result = copy.deepcopy(previous)
    result.update(schema='microduck_realtime_correction_candidate_training_v1',
                  maximum_optimizer_steps=250, maximum_gpu_allocations=1,
                  candidate_training_registered=True, preflight_job=queue['parent'],
                  train_checkpoint_state=str(Path(previous['train_output']) / 'checkpoints/state/step_000250'))
    for path, digest in files.items():
        require(path not in result['files'] or result['files'][path] == digest,
                'a verified preflight dependency would be replaced')
        result['files'][path] = digest
    return result


def allocation_gate(actual, launch, plan, plan_path, digest, role, name):
    devices = [device for row in actual.get('HostConfig', {}).get('DeviceRequests') or []
               if any('gpu' in group for group in row.get('Capabilities', []))
               for device in row.get('DeviceIDs') or []]
    require(actual['Id'] == launch['container_id'] and actual['Image'] == plan['image']
            and actual['Config']['User'] == '1031:1031'
            and actual['Name'].lstrip('/') == name
            and devices == [launch['gpu_uuid']] and str(devices[0]).startswith('GPU-'),
            'actual allocation identity, image, user or GPU differs')
    expected = ['accelerate', 'launch', '--config_file', plan['accelerate_config'],
                '--main_process_port', '29642', plan['entrypoint'], '--plan', str(plan_path),
                '--plan-sha256', digest, '--config', plan[role + '_recipe']]
    require(actual['Config']['Cmd'] == expected and launch['execution_plan_sha256'] == digest,
            'actual allocation did not execute the bound recipe')
    state = actual['State']
    require(state['Status'] == 'exited' and not state['Running'] and state['ExitCode'] == 0
            and not state.get('OOMKilled') and not state.get('Error'), 'allocation did not finish successfully')


def candidate_training_gate(report, receipt, manifest, admission, plan, metrics):
    require(report.get('status') == 'complete' and report.get('optimizer_steps') == 250
            and receipt.get('step') == 250 and report.get('retention') == receipt.get('retention') == manifest,
            '250 actual updates with the admitted retention objective required')
    init = report['initialization']
    require(init == receipt['initialization'] and init['kind'] == 'new_stage_from_complete_weights'
            and init['weights_sha256'] == plan['parent_weights_sha256']
            and init['optimizer_restored'] is False and init['step_counter_restarted'] is True
            and report['retention_teacher_initialization']['weights_sha256'] == plan['retained_weights_sha256']
            == manifest['teacher_weights_sha256'], 'initializer and retention teacher must remain distinct bound roles')
    require(report['training_objective'] == receipt['training_objective'] == manifest['supervised_axis_objective']
            and report['input_history'] == receipt['input_history'] == manifest['input_pair']['student_history']
            and report['extended_navigation_training'] == receipt['extended_navigation_training']
            == admission['data_audit']['extended_training_contract'], 'training input or objective changed')
    for split in ('train', 'validation'):
        require(report['retention_data_binding'][split] == admission['data_audit']['splits'][split]['binding'],
                'actual dataset membership differs from the admitted full cohort')
    require(report['train_samples'] == admission['data_audit']['splits']['train']['samples']
            and report['validation_samples'] == admission['data_audit']['splits']['validation']['samples'], 'dataset sizes changed')
    windows = [r for r in metrics if 'train/loss' in r]
    require([r.get('step') for r in windows] == list(range(1, 251)), 'actual optimizer windows missing or duplicated')
    require(not any('train/loss' not in r and any(k.startswith('train_window/') for k in r) for r in metrics),
            'non-training row entered optimizer metrics')
    for row in windows:
        require(row.get('train_window/microbatches') == row.get('train_window/samples') == 8,
                'gradient accumulation differs')
        keys = ['train/loss'] + ['train_window/' + k for k in
                ('loss_microbatch_mean', 'loss_video_microbatch_mean', 'loss_action_microbatch_mean',
                 'loss_retention_microbatch_mean')]
        require(all(type(row.get(k)) in (int, float) and math.isfinite(row[k]) for k in keys),
                'nonfinite or missing training losses')
    return dict(status='verified', optimizer_steps=250, training_microbatches=2000,
                full_data_binding_exact=True, model_promoted=False, formal_acceptance=False)


def candidate_usage_gate(report, receipt, catalog, admission, metrics, source_sha256,
                         parent_recipe=None, forecast=None):
    """Audit actual source/task counts without claiming gradient attribution."""
    require(report.get('training_sample_usage') == receipt.get('training_sample_usage') == catalog
            and catalog.get('schema') == 'microduck_actual_training_sample_usage_v1'
            and catalog.get('source_sha256') == source_sha256,
            'actual training and checkpoint sample catalogs differ')
    require(catalog.get('counting') == 'actual_main_process_grad_enabled_loss_microbatches'
            and catalog.get('interval') == 'since_previous_training_log'
            and catalog.get('unit') == 'sample_uses_including_repeated_weighted_items'
            and catalog.get('metadata_removed_before_model') is True
            and all(catalog.get(k) is False for k in ['validation_included','prefetch_counted',
                'optimizer_warmup_counted','distributed_global_counts','gradient_contributions_measured',
                'resampled','tensor_values_changed']), 'sample counting scope differs')
    train = admission['data_audit']['splits']['train']
    require(catalog['weighted_items'] == train['samples'] and train['old_full_sequence_exact'] is True, 'complete previous training data was not retained')
    sources = catalog['sources']; inventory = {}
    for sid, spec in sources.items():
        require(spec['root'] not in inventory and spec['weighted_inventory_by_task']
                and all(type(v) is int and v > 0 for v in spec['weighted_inventory_by_task'].values()),
                'invalid or duplicated source inventory')
        inventory[spec['root']] = sum(spec['weighted_inventory_by_task'].values())
    require(inventory == {k:v['weighted_samples'] for k,v in train['source_exposure'].items()}
            and sum(inventory.values()) == train['samples'], 'source inventory differs from the data audit')
    tail = admission['data_audit']['extended_training_contract']['correction_tail']
    old_sources = execution_parent_sources(admission, parent_recipe)
    require(tail['preserved_tail_sources'] == old_sources, 'old correction sources were dropped')
    new_roots = {s['root'] for s in tail['sources'][old_sources:]}
    require(new_roots and new_roots <= inventory.keys(), 'new correction inventory is missing')
    rows = [r for r in metrics if 'train/loss' in r]
    require([r.get('step') for r in rows] == list(range(1,251)), '250 actual optimizer windows required')
    totals, tasks = Counter(), Counter()
    for row in metrics:
        if 'train/loss' not in row:
            require(not any(k.startswith('train_window/') for k in row), 'non-training sample counts present')
            continue
        parts = {}
        for kind in ['source_samples','task_samples','source_task_samples']:
            prefix = 'train_window/'+kind+'/'
            values = {k[len(prefix):]:v for k,v in row.items() if k.startswith(prefix)}
            require(values and all(type(v) in (int,float) and math.isfinite(v) and v > 0 and v == int(v)
                                   for v in values.values()), 'invalid sample-use count')
            parts[kind] = {k:int(v) for k,v in values.items()}
        require(row.get('train_window/samples') == row.get('train_window/microbatches') == 8
                and all(sum(v.values()) == 8 for v in parts.values()), 'incomplete accumulated sample counts')
        by_source, by_task = Counter(), Counter()
        for pair, count in parts['source_task_samples'].items():
            labels = pair.split('/')
            require(len(labels) == 2, 'invalid source/task pair')
            sid, task = labels
            require(sid in sources and task in sources[sid]['weighted_inventory_by_task'],
                    'observed task or source is absent from catalog')
            by_source[sid] += count; by_task[task] += count
        require(dict(by_source) == parts['source_samples'] and dict(by_task) == parts['task_samples'],
                'source/task marginals differ')
        totals.update({sources[sid]['root']:n for sid,n in by_source.items()}); tasks.update(by_task)
    require(sum(totals.values()) == 2000 and sum(totals[r] for r in new_roots) > 0,
            'new corrections did not enter actual model updates')
    current_uses = sum(totals[r] for r in new_roots)
    if forecast is not None:
        require(forecast['sample_uses'] == 2000 and forecast['sampler_order_changed'] is False
                and forecast['executed_training'] is False, 'prospective sampler forecast differs')
        require(current_uses == forecast['counts']['new_realtime']
                and sum(row['train_window/retained_items'] for row in rows) == forecast['counts']['retained'],
                'actual new-data or retention exposure differs from the frozen sampler forecast')
    return dict(status='verified', actual_sample_uses=2000,
                current_correction_uses=current_uses,
                current_correction_fraction=current_uses/2000,
                sampler_forecast_matched=forecast is not None,
                source_uses=dict(totals), task_uses=dict(tasks),
                per_item_trace_recorded=False, gradient_contributions_measured=False,
                capability_acceptance=False)
