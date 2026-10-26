"""Append audited current-policy corrections after a complete prior recipe.

Each source keeps its student identity and original camera prefix. The parent
recipe binds every earlier source, repetition and input option. Changing this
tail is a new training stage, never an implicit optimizer-state resume.
"""
from collections.abc import Mapping, Sequence
import json
from pathlib import Path

import yaml

from .navigation_augmentation import sha256


TARGET = 'lightx2v_microduck.wam_memory_rehearsal_data.FastWAMNavigationMemoryRehearsalDataset'
TAIL_KEYS = {'correction_parent_recipe_path', 'correction_parent_recipe_sha256', 'correction_sources'}


def require(value, message):
    if not value:
        raise ValueError(message)


def plain(value):
    if isinstance(value, Mapping):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [plain(v) for v in value]
    return str(value) if isinstance(value, Path) else value


def digest(value):
    return isinstance(value, str) and len(value) == 64 and not set(value) - set('0123456789abcdef')


def correction_tail_contract(config):
    """Check parent equality and source identity without loading training tensors."""
    config = plain(config)
    if not any(config.get(k) is not None for k in TAIL_KEYS):
        return None
    require(config.get('_target_') == TARGET, 'correction tails require the explicit memory rehearsal dataset')
    parent_path = config.get('correction_parent_recipe_path')
    parent_sha = config.get('correction_parent_recipe_sha256')
    require(isinstance(parent_path, str) and digest(parent_sha)
            and sha256(parent_path) == parent_sha, 'correction parent recipe changed or is not pinned')
    split = config.get('split', 'train')
    require(split in ('train', 'validation'), 'correction recipe cannot load test data')
    recipe = yaml.safe_load(Path(parent_path).read_text())
    parent = recipe['data']['train' if split == 'train' else 'val']
    binding = recipe.get('retention', {}).get('data', {}).get(split, {})
    require(type(binding.get('total_samples')) is int and binding['total_samples'] > 0
            and digest(binding.get('complete_sampling_sha256')),
            'correction parent lacks its audited complete weighted sample sequence')
    base = {k: v for k, v in config.items() if k not in TAIL_KEYS}
    prior = {k: v for k, v in parent.items() if k not in TAIL_KEYS}
    require(base == prior, 'correction stage changed a prior source, weight, input, split or normalizer')
    sources, earlier = config.get('correction_sources'), parent.get('correction_sources', [])
    require(isinstance(sources, list) and isinstance(earlier, list), 'correction sources require an explicit list')
    if split == 'validation':
        require(not sources and not earlier, 'correction sources cannot enter validation')
    else:
        require(len(sources) > len(earlier) and sources[:len(earlier)] == earlier,
                'new corrections must follow the complete previous source tail in its original order')
    roots, indexes = set(), set()
    original_roots = {str(Path(p).resolve()) for p in
                      [config['root'], config.get('extended_release_root', config['root']),
                       *config.get('recovery_roots', []), *config.get('guarded_recovery_roots', [])]}
    for source in sources:
        required = {'kind', 'root', 'index_sha256', 'student_checkpoint_sha256'}
        require(isinstance(source, dict) and source.get('kind') in ('clear_recovery', 'guarded_recovery', 'realtime_recovery'),
                'unknown corrective source kind')
        if source['kind'] == 'clear_recovery':
            required |= {'history_prefix_root', 'history_prefix_sha256'}
        allowed = required | {'sample_repeat'}
        require(required <= set(source) <= allowed and isinstance(source['root'], str)
                and digest(source['index_sha256']) and digest(source['student_checkpoint_sha256']),
                'corrective source identity is incomplete or ambiguous')
        repeat = source.get('sample_repeat', 1)
        require(type(repeat) is int and 1 <= repeat <= 32,
                'source sample_repeat must be an integer between 1 and 32')
        root = str(Path(source['root']).resolve())
        require(root not in roots | original_roots and source['index_sha256'] not in indexes,
                'corrective source duplicates a prior or appended release')
        roots.add(root); indexes.add(source['index_sha256'])
        require(sha256(Path(root)/'index.json') == source['index_sha256'], 'corrective source index changed')
        index = json.loads((Path(root)/'index.json').read_text())
        require(index.get('episodes') and all(e.get('split') == 'train'
                and e.get('student_checkpoint_sha256') == source['student_checkpoint_sha256']
                for e in index['episodes']), 'corrective source checkpoint or training split differs')
        if source['kind'] == 'clear_recovery':
            require(isinstance(source['history_prefix_root'], str) and digest(source['history_prefix_sha256'])
                    and sha256(Path(source['history_prefix_root'])/'index.json') == source['history_prefix_sha256'],
                    'corrective camera prefix index changed')
    return dict(schema='microduck_memory_correction_tail_v1',
        parent_recipe_path=str(Path(parent_path).resolve()), parent_recipe_sha256=parent_sha,
        parent_sample_binding=dict(total_samples=binding['total_samples'],
                                   complete_sampling_sha256=binding['complete_sampling_sha256']),
        preserved_tail_sources=len(earlier), sources=sources,
        sampling='append_after_complete_previous_weighted_sample_sequence')


def load_correction_source(source, config):
    """Reuse the production admission loaders, including full tensor checks."""
    root = str(Path(source['root']).resolve())
    require(sha256(Path(root)/'index.json') == source['index_sha256'], 'corrective source index changed')
    if source['kind'] == 'clear_recovery':
        from .navigation_history_recovery import load_prefixes
        from .navigation_recovery_data import load_recoveries

        entries = load_recoveries(root, config['root'],
            qualified_teacher_roots=config.get('qualified_teacher_roots'),
            qualified_teacher_sha256=config.get('qualified_teacher_sha256'))
        prefixes = load_prefixes(source['history_prefix_root'], source['history_prefix_sha256'], root, entries)
    elif source['kind'] == 'guarded_recovery':
        from .navigation_guarded_recovery import load_guarded_recoveries

        entries = load_guarded_recoveries(root, config['root'], config['extended_release_root'],
            config['extended_release_sha256'], source['index_sha256'], source['student_checkpoint_sha256'])
        prefixes = {e['path']: dict(path=str(Path(root)/e['path']/'camera_prefix.npz'),
                                   sha256=e['prefix_sha256']) for e in entries}
    elif source['kind'] == 'realtime_recovery':
        from .navigation_realtime_recovery import load_realtime_recoveries

        entries = load_realtime_recoveries(root, config['root'], config['extended_release_root'],
            config['extended_release_sha256'], source['index_sha256'], source['student_checkpoint_sha256'],
            horizon=config.get('horizon', 32), video_stride=config.get('video_stride', 4))
        prefixes = {e['path']: dict(path=str(Path(root)/e['realtime_prefix']['path']),
                                   files=e['realtime_prefix']['files']) for e in entries}
    else:
        raise ValueError('unknown corrective source kind')
    require(entries and all(e['split'] == 'train' and
        e['student_checkpoint_sha256'] == source['student_checkpoint_sha256'] for e in entries),
        'admitted correction source identity differs')
    return entries, prefixes


def append_correction_tail(dataset, contract, config):
    """Keep all prior sample objects in order and append only admitted sources."""
    from .navigation_recovery_data import identity

    if contract is None:
        return
    dataset.correction_parent_samples = len(dataset.samples)
    dataset.correction_tail_samples = 0
    dataset.correction_tail_contract = contract
    def verify_parent():
        from .navigation_retention_data import _digest, _identity

        binding = contract['parent_sample_binding']
        require(len(dataset.samples) == binding['total_samples'] and
                _digest([_identity(dataset, i) for i in range(len(dataset.samples))]) ==
                    binding['complete_sampling_sha256'],
                'correction stage changed the complete prior weighted sample sequence')
    if not contract['sources']:
        verify_parent()
        dataset.new_correction_samples = 0
        return
    weights = dataset.recovery_sampling
    seen = {identity(e) for e in dataset.entries if e.get('_root') and not e.get('_qualified_teacher_episode')}
    for i, source in enumerate(contract['sources']):
        if i == contract['preserved_tail_sources']:
            verify_parent()
            dataset.correction_parent_samples = len(dataset.samples)
        root = str(Path(source['root']).resolve())
        entries, prefixes = load_correction_source(source, config)
        repeat = source.get('sample_repeat', 1)
        count, unique = 0, set()
        for item in entries:
            key = identity(item)
            require(key not in seen, 'corrective source duplicates a previously sampled student state')
            seen.add(key)
            entry = dict(item, _root=root, _correction_tail_episode=True)
            if source['kind'] in ('guarded_recovery', 'realtime_recovery'):
                entry['_guarded_recovery_episode'] = True
            prefixes_by_episode = dataset.history_prefixes
            if source['kind'] == 'realtime_recovery':
                entry['_realtime_recovery_episode'] = True
                prefixes_by_episode = dataset.realtime_history_prefixes
            require((root, entry['path']) not in prefixes_by_episode, 'corrective prefix duplicates a prior episode')
            dataset.entries.append(entry)
            prefixes_by_episode[(root, entry['path'])] = prefixes[entry['path']]
            anchor_count = len(entry['actual_student_anchor_starts']) * weights['student_anchor_repeats']
            continuation_count = len(entry['teacher_continuation_starts']) * weights['teacher_continuation_repeats']
            if weights['student_anchor_repeats'] == weights['teacher_continuation_repeats']:
                samples = [(entry, start) for start in entry['sample_starts']] * weights['student_anchor_repeats']
            else:
                samples = ([(entry, start) for start in entry['actual_student_anchor_starts']] * weights['student_anchor_repeats']
                    + [(entry, start) for start in entry['teacher_continuation_starts']] * weights['teacher_continuation_repeats'])
            samples *= repeat
            dataset.samples.extend(samples)
            dataset.recovery_samples += (anchor_count + continuation_count) * repeat
            dataset.recovery_student_anchor_samples += anchor_count * repeat
            dataset.recovery_teacher_continuation_samples += continuation_count * repeat
            if source['kind'] in ('guarded_recovery', 'realtime_recovery'):
                dataset.guarded_recovery_samples += len(samples)
            if source['kind'] == 'realtime_recovery':
                dataset.realtime_recovery_samples = getattr(dataset, 'realtime_recovery_samples', 0) + len(samples)
            count += len(samples)
            unique.update((e['path'], start) for e, start in samples)
        require(count > 0, 'correction source has no eligible training samples')
        dataset.correction_tail_samples += count
        if dataset.source_sample_exposure is None:
            dataset.source_sample_exposure = {}
        dataset.source_sample_exposure[root] = dict(unique_windows=len(unique), weighted_samples=count,
                                                   weighting='verified_current_policy_correction_tail')
        if 'sample_repeat' in source:
            dataset.source_sample_exposure[root]['sample_repeat'] = repeat
    dataset.new_correction_samples = len(dataset.samples) - dataset.correction_parent_samples
