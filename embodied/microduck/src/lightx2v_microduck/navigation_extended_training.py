"""Bind the changed execution range, data and source exposure to checkpoint IO."""
from pathlib import Path

from .navigation_action_contracts import FORWARD04_NAVIGATION_CONTRACT, LEGACY_NAVIGATION_CONTRACT
from .navigation_extended_release import normalization_binding
from .navigation_extended_qualification import require


TARGET = 'lightx2v_microduck.wam_extended_history_data.FastWAMNavigationExtendedHistoryDataset'
REHEARSAL_TARGET = 'lightx2v_microduck.wam_rehearsal_history_data.FastWAMNavigationRehearsalHistoryDataset'
MEMORY_REHEARSAL_TARGET = 'lightx2v_microduck.wam_memory_rehearsal_data.FastWAMNavigationMemoryRehearsalDataset'


def training_extended_contract(train, validation, audit):
    from .navigation_correction_tail import TAIL_KEYS, correction_tail_contract

    configs = (train, validation)
    has_tail = any(c.get(k) is not None for c in configs for k in TAIL_KEYS)
    require(not has_tail or all(c.get('_target_') == MEMORY_REHEARSAL_TARGET for c in configs),
            'correction tails require the explicit memory rehearsal dataset')
    selected = [c.get('_target_') in (TARGET, REHEARSAL_TARGET, MEMORY_REHEARSAL_TARGET) for c in configs]
    if not any(selected):
        require(not any(c.get('extended_release_root') or c.get('extended_release_sha256')
            or c.get('guarded_recovery_roots') or c.get('guarded_recovery_sha256')
            or c.get('guarded_recovery_checkpoint_sha256')
            or c.get('guarded_recovery_checkpoint_sha256s')
            or c.get('navigation_contract', LEGACY_NAVIGATION_CONTRACT) != LEGACY_NAVIGATION_CONTRACT
            for c in configs) and audit.get('extended_teacher_index_sha256') is None,
            'extended data and actions require the explicit extended dataset implementation')
        return None
    require(all(selected) and train.get('_target_') == validation.get('_target_'),
            'training and validation must use the same extended dataset implementation')
    for key in ('root', 'extended_release_root', 'extended_release_sha256', 'navigation_contract'):
        require(train.get(key) is not None and train[key] == validation.get(key), 'extended train/validation identities differ: ' + key)
    require(train['navigation_contract'] == FORWARD04_NAVIGATION_CONTRACT
            and train.get('split', 'train') == 'train' and validation.get('split') == 'validation',
            'extended execution or independent validation split changed')
    digest = train['extended_release_sha256']
    require(isinstance(digest, str) and len(digest) == 64
            and audit.get('extended_teacher_index_sha256') == digest, 'extended release differs from frozen training audit')
    rehearsal = train['_target_'] in (REHEARSAL_TARGET, MEMORY_REHEARSAL_TARGET)
    memory_input = None
    if train['_target_'] == MEMORY_REHEARSAL_TARGET:
        from .navigation_history import training_history_contract

        memory_input = training_history_contract(train, validation)
    else:
        require(not any(c.get('memory_input_contract') is not None for c in configs),
                'paired memory inputs require the memory rehearsal dataset')
    if rehearsal:
        from .wam_rehearsal_history_data import appended_recovery_config, retained_dataset_config

        for key in ('retained_recipe_path', 'retained_recipe_sha256'):
            require(train.get(key) is not None and train[key] == validation.get(key),
                    'retained train/validation recipes differ')
        extra = {'_target_', 'retained_recipe_path', 'retained_recipe_sha256',
                 'extended_release_root', 'extended_release_sha256', 'navigation_contract', 'extended_repeat',
                  'appended_recovery_sha256', 'appended_recovery_checkpoint_sha256',
                  'guarded_recovery_roots', 'guarded_recovery_sha256', 'guarded_recovery_checkpoint_sha256',
                  'guarded_recovery_checkpoint_sha256s'}
        if memory_input is not None:
            extra.add('memory_input_contract')
            extra.update(TAIL_KEYS)
        for config in configs:
            retained = retained_dataset_config(config['retained_recipe_path'], config['retained_recipe_sha256'],
                                               config.get('split', 'train'))
            try:
                appended_recovery_config({k: v for k, v in config.items() if k not in extra}, retained,
                    config.get('appended_recovery_sha256', ()), config.get('appended_recovery_checkpoint_sha256'))
            except ValueError as exc:
                raise ValueError('rehearsal data differs from retained recipe: ' + str(exc)) from exc
    else:
        require(not any(c.get(k) for c in configs for k in ('qualified_teacher_roots', 'augmentation_roots',
            'retarget_roots', 'recovery_roots', 'sample_source_repeats', 'guarded_recovery_roots',
            'guarded_recovery_sha256', 'guarded_recovery_checkpoint_sha256',
            'guarded_recovery_checkpoint_sha256s')), 'unregistered data source in extended recipe')
    from .navigation_guarded_recovery import guarded_recovery_config
    guarded = guarded_recovery_config(train.get('guarded_recovery_roots', ()),
        train.get('guarded_recovery_sha256', ()), train.get('guarded_recovery_checkpoint_sha256'), 'train',
        checkpoints=train.get('guarded_recovery_checkpoint_sha256s'))
    guarded_recovery_config(validation.get('guarded_recovery_roots', ()),
        validation.get('guarded_recovery_sha256', ()), validation.get('guarded_recovery_checkpoint_sha256'), 'validation',
        checkpoints=validation.get('guarded_recovery_checkpoint_sha256s'))
    require(list(audit.get('guarded_recovery_index_sha256', ())) ==
            (guarded['index_sha256'] if guarded else []), 'guarded recovery audit hashes differ')
    repeats = {}
    for key in (('extended_repeat',) if rehearsal else ('clear_repeat', 'extended_repeat')):
        value = train.get(key, 1)
        require(type(value) is int and value >= 1 and validation.get(key, 1) == 1,
                'explicit training-only source repetitions required')
        repeats[key] = value
    contract = dict(schema='microduck_extended_navigation_training_v1',
        execution_contract=FORWARD04_NAVIGATION_CONTRACT,
        normalization_binding=normalization_binding(audit['stats_sha256']),
        base_index_sha256=audit['index_sha256'], extended_teacher_index_sha256=digest,
        base_root=str(Path(train['root']).resolve()),
        extended_root=str(Path(train['extended_release_root']).resolve()), source_repeats=repeats)
    if rehearsal:
        contract.update(schema='microduck_rehearsal_navigation_training_v1',
                        retained_recipe_sha256=train['retained_recipe_sha256'],
                        retained_recipe_path=str(Path(train['retained_recipe_path']).resolve()),
                        retained_sampling='complete_recipe_once_in_original_order')
        if train.get('appended_recovery_sha256'):
            count = len(train['appended_recovery_sha256'])
            contract['appended_recoveries'] = dict(index_sha256=list(train['appended_recovery_sha256']),
                student_checkpoint_sha256=train['appended_recovery_checkpoint_sha256'],
                history_prefix_sha256=list(train['history_prefix_sha256'][-count:]),
                sampling='after_complete_retained_and_extended_samples')
        if guarded:
            contract['guarded_recoveries'] = guarded
        if memory_input is not None:
            contract.update(schema='microduck_memory_rehearsal_training_v1',
                            memory_input_contract=memory_input)
    if has_tail:
        for key in ('correction_parent_recipe_path', 'correction_parent_recipe_sha256'):
            require(train.get(key) is not None and train[key] == validation.get(key),
                    'correction train/validation parent recipes differ')
        tail = correction_tail_contract(train)
        correction_tail_contract(validation)
        require(list(audit.get('correction_source_index_sha256', [])) ==
                [s['index_sha256'] for s in tail['sources']]
                and audit.get('correction_parent_recipe_sha256') == tail['parent_recipe_sha256'],
                'correction tail differs from its frozen training audit')
        contract['correction_tail'] = tail
    else:
        require(not audit.get('correction_source_index_sha256') and not audit.get('correction_parent_recipe_sha256'),
                'correction audit requires an explicitly declared data tail')
    return contract


def verify_extended_resume(receipt, contract):
    require(receipt.get('extended_navigation_training') == contract,
            'resume changed extended data/actions/normalization/source exposure; initialize an explicit new stage')
    if contract is not None:
        require(receipt.get('navigation_contract') == contract['execution_contract']
                and receipt.get('normalization_binding') == contract['normalization_binding'],
                'resume action contract is not bound to its frozen normalizer')
    verify_extended_input_history(receipt)


def verify_extended_input_history(receipt):
    """A new camera contract must be explicit in both training and checkpoint IO."""
    training = receipt.get('extended_navigation_training')
    if not isinstance(training, dict):
        return
    memory = training.get('memory_input_contract')
    if training.get('schema') != 'microduck_memory_rehearsal_training_v1':
        require(memory is None, 'memory training cannot be relabelled as an earlier training schema')
        return
    from .navigation_history import history_spec_from_contract

    require(isinstance(memory, dict) and memory.get('schema') == 'microduck_rgb_target_keyframe_v2',
            'memory training requires its explicit v2 input contract')
    spec = history_spec_from_contract(memory)
    require(receipt.get('input_history') == spec.as_dict(),
            'checkpoint input history differs from its memory training contract')
