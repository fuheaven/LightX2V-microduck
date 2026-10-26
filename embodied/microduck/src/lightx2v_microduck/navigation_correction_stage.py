"""Build a fresh optimizer stage from a completed correction-data admission.

The initializer and frozen retention teacher deliberately have different roles.
Appending data invalidates optimizer/sampler continuation from the parent run.
This module prepares recipes only; it never allocates a device or promotes a model.
"""
import copy


def require(condition, message):
    if not condition:
        raise ValueError(message)


def choose_correction_repeats(old_samples, new_weighted_samples, *, target_fraction=.20,
                              maximum_repeat=32):
    """Increase exposure to new supervision without removing any old sample.

    Repetition changes optimizer exposure, never the number of independent
    trajectories. Refuse an extremely small cohort instead of hiding it under
    unlimited duplication. A fresh sampler forecast must follow this choice.
    """
    require(type(old_samples) is int and old_samples > 0
            and type(new_weighted_samples) is int and new_weighted_samples > 0,
            'positive old and newly admitted sample counts are required')
    require(type(target_fraction) in (int, float) and 0 < target_fraction <= .5
            and type(maximum_repeat) is int and 1 <= maximum_repeat <= 32,
            'invalid prospective correction exposure bounds')
    for repeat in range(1, maximum_repeat + 1):
        new_count = new_weighted_samples * repeat
        fraction = new_count / (old_samples + new_count)
        if fraction >= target_fraction:
            return dict(sample_repeat=repeat, old_samples=old_samples,
                unweighted_source_samples=new_weighted_samples, new_samples=new_count,
                total_samples=old_samples + new_count, expected_fraction=fraction,
                target_fraction=target_fraction, maximum_repeat=maximum_repeat,
                independent_data_increased_by_repetition=False)
    raise ValueError('insufficient new corrections for the requested exposure within repetition limit')


def build_correction_stage(parent, admitted, report, *, parent_sha256,
                           admission_sha256, initializer, output_dir,
                           training_steps=250, isolated_probe=False,
                           cohort='clear_and_obstacle'):
    """Keep the prior objective/data intact while appending audited supervision."""
    require(type(training_steps) is int and training_steps >= 250,
            'a separately bounded candidate stage is required')
    probe_scopes = {'actual_single_case_four_branch_data_preflight',
                    'actual_isolated_training_data_preflight'}
    scopes = {
        'clear_and_obstacle': ('complete_current_clear_and_obstacle_training_corrections',
                              {'clear_recovery', 'guarded_recovery'}),
        'realtime_obstacle': ('complete_realtime_obstacle_training_corrections',
                             {'realtime_recovery'}),
    }
    require(cohort in scopes and not (isolated_probe and cohort != 'clear_and_obstacle'),
            'unknown or ambiguous correction cohort')
    scope, required_kinds = scopes[cohort]
    require(report['status'] == ('passed' if isolated_probe else 'complete')
            and report['actual_complete_new_cohort_audited'] is (not isolated_probe)
            and (report['scope'] in probe_scopes if isolated_probe else
                 report['scope'] == scope),
            'a complete data admission for the declared cohort is required')
    audit = report['data_audit']
    require(audit['status'] == 'complete' and audit['parent_recipe_sha256'] == parent_sha256
            and audit['normalization_unchanged'] is True and audit['validation_unchanged'] is True,
            'parent data or frozen normalization was not verified')
    require(admitted['schema'] == 'microduck_memory_correction_data_audit_v1'
            and admitted['audit']['stage_preparation_only'] is True
            and admitted['audit']['stats_sha256'] == parent['audit']['stats_sha256'],
            'unexpected admitted data recipe')
    require(set(initializer) == {'path', 'sha256', 'parent_receipt'},
            'initialize complete navigation weights using their receipt')
    sources = admitted['data']['train']['correction_sources']
    prior_sources = parent['data']['train'].get('correction_sources', [])
    new_sources = sources[len(prior_sources):]
    require(sources[:len(prior_sources)] == prior_sources and new_sources
            and all(s['student_checkpoint_sha256'] == initializer['sha256'] for s in new_sources),
            'new supervision must come from the selected initializer')
    if not isolated_probe:
        require({s['kind'] for s in new_sources} == required_kinds,
                'correction sources differ from the declared cohort')
    require(admitted['data']['val']['correction_sources'] == [],
            'validation cannot receive correction data')
    for split in ('train', 'val'):
        expected = copy.deepcopy(parent['data'][split])
        expected.update(correction_parent_recipe_path=admitted['data'][split]['correction_parent_recipe_path'],
                        correction_parent_recipe_sha256=parent_sha256,
                        correction_sources=sources if split == 'train' else [])
        require(admitted['data'][split] == expected,
                'old sources, sampling, or observation configuration changed')

    expected_retention = copy.deepcopy(parent['retention'])
    for split in ('train', 'validation'):
        item = audit['splits'][split]
        old = parent['retention']['data'][split]
        bound = item['binding']
        require(item['old_samples'] == old['total_samples'] and item['old_full_sequence_exact'] is True
                and item['samples'] == bound['total_samples']
                and item['new_samples'] == item['samples'] - item['old_samples']
                and bound['retained_prefix_samples'] == old['retained_prefix_samples']
                and bound['retained_sampling_sha256'] == old['retained_sampling_sha256'],
                'old weighted sequence or retention membership changed')
        if split == 'train':
            require(item['new_samples'] > 0 and item['distinct_new_windows'] > 0
                    and item['sampled_windows_verified'] >= item['distinct_new_windows'],
                    'new model-visible windows were not verified')
        else:
            require(item['new_samples'] == 0
                    and bound['complete_sampling_sha256'] == old['complete_sampling_sha256'],
                    'validation changed')
        expected_retention['data'][split] = {key: bound[key] for key in old}
    require(admitted['retention'] == expected_retention,
            'the retained teacher, loss weight, or paired input changed')

    cfg = copy.deepcopy(parent)
    cfg.update(data=copy.deepcopy(admitted['data']), retention=expected_retention,
               audit=copy.deepcopy(admitted['audit']), output_dir=str(output_dir),
               initialize_weights=copy.deepcopy(initializer), initialize_released_backbone=None,
               resume=None, resume_loss_recipe_sha256=None, verify_resume_only=False,
               max_steps=training_steps, save_every=training_steps, eval_every=training_steps)
    # These describe the prior optimizer continuation/comparison, not this stage.
    for key in ('continuation', 'axis_experiment', 'axis_arm', 'memory_data_admission_sha256'):
        cfg['audit'].pop(key, None)
    cfg['audit'].update(parent_recipe_sha256=parent_sha256,
        correction_data_admission_sha256=admission_sha256,
        purpose='Current-policy corrections with unchanged old weighted data and frozen retained teacher.',
        stage_preparation_only=isolated_probe, capability_acceptance=False,
        full_candidate_requires_reviewed_model_preflight=True,
        correction_stage=dict(initializer_sha256=initializer['sha256'],
            retained_teacher_sha256=expected_retention['teacher']['sha256'],
            optimizer_state='fresh', input_data='append_after_complete_previous_weighted_sequence',
            preparation_scope='isolated_loader_probe' if isolated_probe else 'complete_current_cohort'))
    if cohort != 'clear_and_obstacle':
        cfg['audit']['correction_stage']['cohort'] = cohort
    return cfg


def preflight_exposure(indices, *, total_samples, old_samples, retained_samples):
    """Report actual sampler coverage without changing its order to pass a test."""
    indices = list(indices)
    require(len(indices) == 16 and 0 < retained_samples < old_samples < total_samples
            and all(type(i) is int and 0 <= i < total_samples for i in indices),
            'two updates must inspect sixteen actual in-range sample indices')
    counts = dict(retained=sum(i < retained_samples for i in indices),
                  old_other=sum(retained_samples <= i < old_samples for i in indices),
                  current_correction=sum(i >= old_samples for i in indices))
    return dict(indices=indices, counts=counts,
                exercises_retention_and_current_correction=counts['retained'] > 0
                    and counts['current_correction'] > 0,
                sampler_order_changed=False, model_executed=False)
