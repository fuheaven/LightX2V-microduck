"""Verify actual model updates against each completed correction-data admission.

Parent membership comes from its pinned recipe, never a previous experiment's
hard-coded sample count. The runtime report, checkpoint, real loss calls and
sampler prefix must agree. These checks are not task acceptance.
"""
from collections import Counter
import hashlib
import math
from pathlib import Path


def require(value, message):
    if not value:
        raise ValueError(message)


def preparation_runtime_gate(container, *, expected_image, expected_command,
                             required_bindings=(), required_environment=None):
    """Verify the CPU producer independently of the frozen GPU model image.

    A validated simulator binding can require a different CPU image. Its
    identity must be explicit; accepting any image or changing the model image
    to match the producer would lose the existing model-runtime comparison.
    """
    require(isinstance(expected_image, str) and expected_image.startswith('sha256:')
            and len(expected_image) == 71
            and not set(expected_image[7:]) - set('0123456789abcdef'),
            'CPU preparation image must be pinned by digest')
    state = container['State']
    require(state['Status'] == 'exited' and state['ExitCode'] == 0
            and not state['Running'] and not state['OOMKilled'] and not state['Error'],
            'CPU preparation did not complete successfully')
    require(container['Image'] == expected_image
            and container['Config']['Cmd'] == expected_command
            and container['Config']['User'] == '1031:1031',
            'CPU preparation image, command or UID differs')
    require(not container['HostConfig'].get('DeviceRequests')
            and container['HostConfig']['NetworkMode'] == 'none',
            'CPU preparation execution scope differs')
    bindings = container['HostConfig'].get('Binds') or []
    require(all(binding in bindings for binding in required_bindings),
            'CPU preparation native binding missing or changed')
    values = container['Config']['Env']
    for name, value in (required_environment or {}).items():
        require([v for v in values if v.startswith(name + '=')] == [name + '=' + value],
                'CPU preparation renderer environment missing or ambiguous')
    return dict(status='verified', container_id=container['Id'], image=expected_image,
                gpu_allocated=False, command_exact=True,
                required_bindings=list(required_bindings),
                required_environment=dict(required_environment or {}))


def execution_parent_sources(admission, parent_recipe=None):
    audit = admission['data_audit']
    tail = audit['extended_training_contract']['correction_tail']
    path = Path(tail['parent_recipe_path'])
    require(hashlib.sha256(path.read_bytes()).hexdigest() == tail['parent_recipe_sha256']
            == audit['parent_recipe_sha256'], 'parent recipe identity changed')
    if parent_recipe is None:
        import yaml
        parent_recipe = yaml.safe_load(path.read_text())
    parent = parent_recipe
    old = parent['data']['train'].get('correction_sources', [])
    require(tail['sources'][:len(old)] == old and len(tail['sources']) > len(old),
            'an earlier corrective source was dropped or changed')
    require(audit['status'] == 'complete' and audit['normalization_unchanged'] is True
            and audit['validation_unchanged'] is True,
            'frozen normalization or validation changed')
    previous_train = parent['retention']['data']['train']
    require(tail['preserved_tail_sources'] == len(old)
            and tail['parent_sample_binding'] == {
                key: previous_train[key] for key in ('total_samples', 'complete_sampling_sha256')},
            'parent sample sequence or correction-source count changed')
    for split in ('train', 'validation'):
        item = audit['splits'][split]; previous = parent['retention']['data'][split]
        require(item['old_full_sequence_exact'] is True
                and item['old_samples'] == previous['total_samples']
                and item['samples'] == item['binding']['total_samples']
                and item['new_samples'] == item['samples'] - item['old_samples']
                and item['binding']['retained_prefix_samples'] == previous['retained_prefix_samples']
                and item['binding']['retained_sampling_sha256'] == previous['retained_sampling_sha256'],
                'complete parent sequence or retained membership differs')
        if split == 'train':
            require(item['new_samples'] > 0 and item['distinct_new_windows'] > 0
                    and item['sampled_windows_verified'] >= item['distinct_new_windows'],
                    'new training windows were not fully checked')
        if split == 'validation':
            require(item['new_samples'] == 0 and item['samples'] == previous['total_samples']
                    and item['binding']['complete_sampling_sha256'] == previous['complete_sampling_sha256'],
                    'validation sample sequence changed')
    return len(old)


def preparation_gate(prepared, admission, recipe, manifest, exposure):
    """Admit only a complete prospective realtime cohort with unchanged parent data."""
    require(prepared.get('status') == 'complete' and prepared.get('stage') == 'ready_for_model_preflight'
            and prepared.get('actual_complete_new_cohort_audited') is True
            and prepared.get('gpu_allocated') is False and prepared.get('optimizer_steps') == 0
            and admission.get('status') == 'complete'
            and admission.get('actual_complete_new_cohort_audited') is True
            and admission.get('scope') == 'complete_realtime_obstacle_training_corrections',
            'complete realtime cohort preparation is required')
    require(prepared['data_audit'] == admission['data_audit'], 'prepared data admission differs')
    old_sources = execution_parent_sources(admission)
    audit = admission['data_audit']; train = audit['splits']['train']
    tail = audit['extended_training_contract']['correction_tail']
    require(all(s['kind'] == 'realtime_recovery'
                and s['student_checkpoint_sha256'] == recipe['initialize_weights']['sha256']
                for s in tail['sources'][old_sources:]), 'new supervision belongs to another cohort or student')
    require(recipe['max_steps'] == recipe['save_every'] == recipe['eval_every'] == 250
            and recipe['resume'] is None and recipe['verify_resume_only'] is False
            and recipe['data']['train']['correction_sources'] == tail['sources']
            and recipe['data']['val']['correction_sources'] == [], 'candidate scope or data sources differ')
    require(exposure['exercises_retention_and_current_correction'] is True
            and exposure['sampler_order_changed'] is False and len(exposure['indices']) == 16,
            'model preflight must cover both retained and new examples without changing sampler order')
    indices = exposure['indices']; retained = train['binding']['retained_prefix_samples']
    require(all(type(i) is int and 0 <= i < train['samples'] for i in indices), 'sampler indices outside data')
    counts = dict(retained=sum(i < retained for i in indices),
                  old_other=sum(retained <= i < train['old_samples'] for i in indices),
                  current_correction=sum(i >= train['old_samples'] for i in indices))
    require(counts == exposure['counts'] and counts['retained'] > 0 and counts['current_correction'] > 0,
            'preflight exposure does not match actual sampler prefix')
    for split in ('train', 'validation'):
        require(all(audit['splits'][split]['binding'][k] == value
                    for k, value in manifest['data'][split].items()), 'retention manifest membership differs')
    require(recipe['retention']['teacher']['sha256'] == manifest['teacher_weights_sha256']
            and recipe['retention']['weight'] == manifest['weight'] == 16., 'retained teacher or weight changed')


def actual_training_gate(report, receipt, manifest, admission, execution, exposure, metrics):
    require(report['initialization'] == receipt['initialization']
            and report['initialization']['kind'] == 'new_stage_from_complete_weights'
            and report['initialization']['weights_sha256'] == execution['parent_weights_sha256']
            and report['initialization']['optimizer_restored'] is False
            and report['initialization']['step_counter_restarted'] is True
            and report['retention_teacher_initialization']['weights_sha256'] == execution['retained_weights_sha256'],
            'actual initializer, optimizer reset, or retained teacher differs')
    require(report['training_objective'] == receipt['training_objective'] == manifest['supervised_axis_objective']
            and report['input_history'] == receipt['input_history'] == manifest['input_pair']['student_history']
            and report['extended_navigation_training'] == receipt['extended_navigation_training']
                == admission['data_audit']['extended_training_contract'], 'actual model input or supervision differs')
    for split in ('train', 'validation'):
        require(report['retention_data_binding'][split] == admission['data_audit']['splits'][split]['binding'],
                'actual model data loader differs from admitted full sequence')
    require(report['train_samples'] == admission['data_audit']['splits']['train']['samples']
            and report['validation_samples'] == admission['data_audit']['splits']['validation']['samples'], 'actual model dataset size differs')
    rows = [row for row in metrics if 'train/loss' in row]
    require(len(rows) == 2, 'two actual optimizer windows required')
    prefix = admission['data_audit']['splits']['train']['binding']['retained_prefix_samples']
    for n, row in enumerate(rows):
        expected = sum(i < prefix for i in exposure['indices'][n * 8:(n + 1) * 8])
        require(row['train_window/retained_items'] == expected,
                'actual retention exposure differs from the bound sampler')
    return dict(status='verified', actual_optimizer_windows=2, actual_microbatches=16,
                retained_items=sum(r['train_window/retained_items'] for r in rows),
                sampler_projected_current_correction_items=exposure['counts']['current_correction'],
                per_item_trace_recorded=False, full_data_binding_exact=True)


def sample_usage_gate(report, receipt, catalog, admission, exposure, metrics, source_sha256,
                      parent_recipe=None):
    """Cross-check main-process loss counts; do not claim per-item trace/gradients."""
    require(report.get('training_sample_usage') == receipt.get('training_sample_usage') == catalog,
            'model report and checkpoint must bind the actual sample catalog')
    require(catalog.get('schema') == 'microduck_actual_training_sample_usage_v1'
            and catalog.get('source_sha256') == source_sha256
            and catalog.get('counting') == 'actual_main_process_grad_enabled_loss_microbatches'
            and catalog.get('unit') == 'sample_uses_including_repeated_weighted_items'
            and catalog.get('interval') == 'since_previous_training_log', 'sample counting contract differs')
    require(all(catalog.get(k) is False for k in (
        'validation_included', 'prefetch_counted', 'optimizer_warmup_counted', 'distributed_global_counts',
        'gradient_contributions_measured', 'resampled', 'tensor_values_changed'))
        and catalog.get('metadata_removed_before_model') is True, 'sample counting scope differs')
    digest = catalog.get('complete_sampling_sha256')
    require(isinstance(digest, str) and len(digest) == 64 and not set(digest) - set('0123456789abcdef'),
            'complete sample catalog fingerprint missing')
    train = admission['data_audit']['splits']['train']
    require(catalog['weighted_items'] == train['samples'], 'catalog size differs from admitted training data')
    tail = admission['data_audit']['extended_training_contract']['correction_tail']
    require(tail['preserved_tail_sources'] == execution_parent_sources(admission, parent_recipe),
            'previous correction sources were not fully retained')
    new_roots = {x['root'] for x in tail['sources'][tail['preserved_tail_sources']:]}
    require(new_roots, 'current correction roots missing')
    sources = catalog['sources']; inventory = {}
    for sid, spec in sources.items():
        require(isinstance(sid, str) and sid.startswith('source_') and '/' not in sid,
                'invalid source identifier')
        root = spec['root']; tasks = spec['weighted_inventory_by_task']
        require(root not in inventory and tasks and all(type(v) is int and v > 0 for v in tasks.values()),
                'source inventory is repeated or invalid')
        inventory[root] = sum(tasks.values())
    expected = {root: spec['weighted_samples'] for root, spec in train['source_exposure'].items()}
    require(inventory == expected and sum(inventory.values()) == train['samples']
            and new_roots <= inventory.keys(), 'catalog inventory differs from actual data audit')

    def counts(row, kind):
        prefix = 'train_window/' + kind + '/'
        result = {}
        for key, value in row.items():
            if key.startswith(prefix):
                require(type(value) in (int, float) and math.isfinite(value)
                        and value > 0 and value == int(value), 'invalid sample-use count')
                result[key[len(prefix):]] = int(value)
        return result

    rows = [r for r in metrics if 'train/loss' in r]
    require(len(rows) == 2 and [r.get('step') for r in rows] == [1, 2], 'two training windows required')
    for row in metrics:
        if 'train/loss' not in row:
            require(not any(k.startswith(('train_window/source_samples/', 'train_window/task_samples/',
                                         'train_window/source_task_samples/')) for k in row),
                    'validation or warmup entered sample-use counts')
    total_by_root, total_by_task = Counter(), Counter()
    current_total = 0
    for n, row in enumerate(rows):
        sc, tc, pairs = (counts(row, k) for k in ('source_samples', 'task_samples', 'source_task_samples'))
        require(row.get('train_window/samples') == row.get('train_window/microbatches') == 8
                and sum(sc.values()) == sum(tc.values()) == sum(pairs.values()) == 8,
                'actual sample-use counts must cover all eight accumulated loss calls')
        paired_sources, paired_tasks = Counter(), Counter()
        for key, value in pairs.items():
            parts = key.split('/')
            require(len(parts) == 2, 'invalid source/task count')
            sid, task = parts
            require(sid in sources and task in sources[sid]['weighted_inventory_by_task'],
                    'observed source/task is absent from the bound inventory')
            paired_sources[sid] += value; paired_tasks[task] += value
        require(dict(paired_sources) == sc and dict(paired_tasks) == tc,
                'source/task marginals disagree with joint counts')
        actual_current = sum(value for sid, value in sc.items() if sources[sid]['root'] in new_roots)
        projected_current = sum(i >= train['old_samples'] for i in exposure['indices'][n * 8:(n + 1) * 8])
        require(actual_current == projected_current, 'actual current-data use differs from bound sampler')
        current_total += actual_current
        total_by_task.update(tc)
        total_by_root.update({sources[sid]['root']: value for sid, value in sc.items()})
    require(current_total == exposure['counts']['current_correction'] > 0,
            'current correction data did not enter model training')
    return dict(status='verified', actual_sample_uses=16, actual_current_correction_uses=current_total,
                source_uses=dict(total_by_root), task_uses=dict(total_by_task),
                catalog_sha256=catalog['complete_sampling_sha256'],
                per_item_trace_recorded=False, gradient_contributions_measured=False,
                validation_excluded=True, prefetch_excluded=True)
