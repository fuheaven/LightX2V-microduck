"""Audit an appended memory-policy correction stage before model allocation.

This produces data configuration and evidence, never an optimizer recipe.
The full prior weighted sequence and validation remain frozen. Each distinct
new training window is materialized through the production paired-view loader.
"""
import copy
import gc
from pathlib import Path

import numpy as np
import yaml

from .navigation_augmentation import sha256
from .navigation_correction_tail import require
from .navigation_extended_training import training_extended_contract
from .navigation_retention_data import RetainedTrainingView, _digest, _identity

SCHEMA = 'microduck_memory_correction_data_audit_v1'


def prepare_data_recipe(parent_path, expected_parent_sha256, sources):
    parent_path = Path(parent_path)
    require(sha256(parent_path) == expected_parent_sha256, 'parent recipe changed')
    parent = yaml.safe_load(parent_path.read_text())
    require(isinstance(sources, list) and sources, 'completed correction releases required')
    recipe = dict(schema=SCHEMA, data=copy.deepcopy(parent['data']),
                  retention=copy.deepcopy(parent['retention']), audit=copy.deepcopy(parent['audit']))
    prior = recipe['data']['train'].get('correction_sources', [])
    for key in ('train', 'val'):
        config = recipe['data'][key]
        config.update(correction_parent_recipe_path=str(parent_path),
                      correction_parent_recipe_sha256=expected_parent_sha256,
                      correction_sources=copy.deepcopy(prior + sources if key == 'train' else []))
    recipe['audit'].update(stage_preparation_only=True,
        purpose='CPU data audit only; a new optimizer stage needs a separately verified model recipe.',
        correction_source_index_sha256=[s['index_sha256'] for s in prior + sources],
        correction_parent_recipe_sha256=expected_parent_sha256)
    # Fail before allocating a dataset if any old source or execution option differs.
    training_extended_contract(recipe['data']['train'], recipe['data']['val'], recipe['audit'])
    return recipe


def verify_window(sample, *, current_tick, frame_ticks, horizon, video_stride, retained):
    """Check model-visible boundaries independently of episode or teacher metadata."""
    import torch

    keys = {'video', 'action', 'proprio', 'prompt', 'action_is_pad', 'image_is_pad', 'proprio_is_pad',
            'history_source_ticks', 'history_is_pad', 'retention_video', 'retention_history_source_ticks',
            'retention_history_is_pad', 'retention_is_retained', 'context', 'context_mask'}
    require(set(sample) <= keys, 'private or undeclared field entered model sample')
    require(sample['retention_is_retained'].dtype == torch.bool
            and sample['retention_is_retained'].ndim == 0
            and sample['retention_is_retained'].item() is retained, 'retention supervision mask changed')
    frames = len(range(0, horizon + 1, video_stride))
    require(len(frame_ticks) == frames and int(frame_ticks[0]) == current_tick,
            'sample timestamps differ from recorded branch')
    for prefix in ('', 'retention_'):
        video, ticks, padding = (sample[prefix + key] for key in ('video', 'history_source_ticks', 'history_is_pad'))
        require(tuple(video.shape) == (3, frames, 512, 512) and video.dtype == torch.float32,
                'model video dimensions or precision changed')
        require(torch.isfinite(video).all().item() and video.min().item() >= -1 and video.max().item() <= 1,
                'nonfinite or out-of-range RGB input')
        require(tuple(ticks.shape) == tuple(padding.shape) == (frames, 4)
                and ticks.dtype == torch.int64 and padding.dtype == torch.bool,
                'history source or padding shape changed')
        require(bool((ticks >= 0).all()) and bool((ticks <= torch.as_tensor(frame_ticks)[:, None]).all()),
                'future observation entered a causal history grid')
    require(tuple(sample['action'].shape) == (horizon, 5)
            and tuple(sample['proprio'].shape) == (horizon, 48), 'model action/state contract changed')
    for key in ('action', 'proprio'):
        require(sample[key].dtype == torch.float32 and torch.isfinite(sample[key]).all().item(),
                'nonfinite model labels or state')
    for key, count in [('action_is_pad', horizon), ('proprio_is_pad', horizon), ('image_is_pad', frames)]:
        require(tuple(sample[key].shape) == (count,) and sample[key].dtype == torch.bool, 'padding mask differs')
    require(isinstance(sample['prompt'], str) and sample['prompt'], 'missing language instruction')


def audit_data_recipe(recipe, progress=None):
    """Verify every old identity and every distinct new public window on CPU."""
    import torch
    from .wam_memory_rehearsal_data import FastWAMNavigationMemoryRehearsalDataset

    recipe = copy.deepcopy(recipe)
    train, validation = recipe['data']['train'], recipe['data']['val']
    require(recipe['schema'] == SCHEMA and recipe['audit']['stage_preparation_only'] is True,
            'an explicit data-only preparation recipe is required')
    contract = training_extended_contract(train, validation, recipe['audit'])
    parent_path = Path(train['correction_parent_recipe_path'])
    parent_sha = train['correction_parent_recipe_sha256']
    require(sha256(parent_path) == parent_sha, 'parent recipe changed before audit')
    parent = yaml.safe_load(parent_path.read_text())
    old_retention = copy.deepcopy(parent['retention']['data'])
    stats_path = Path(train['root']) / 'dataset_stats.json'
    require(sha256(stats_path) == recipe['audit']['stats_sha256'], 'normalization changed')
    report = dict(schema=SCHEMA, status='running', parent_recipe_sha256=parent_sha,
                  splits={}, gpu_allocated=False, model_loaded=False, optimizer_steps=0,
                  model_promoted=False, formal_acceptance=False)
    for split, key in [('train', 'train'), ('validation', 'val')]:
        config = copy.deepcopy(recipe['data'][key]); config.pop('_target_')
        dataset = FastWAMNavigationMemoryRehearsalDataset(**config)
        prior = old_retention[split]
        old_count = prior['total_samples']
        require(dataset.correction_parent_samples == old_count
                and _digest([_identity(dataset, i) for i in range(old_count)]) == prior['complete_sampling_sha256'],
                'complete previous weighted sample sequence changed')
        view = RetainedTrainingView(dataset, split=split,
            retained_recipe_sha256=old_retention['retained_recipe_sha256'],
            expected_samples=len(dataset), expected_retained_samples=prior['retained_prefix_samples'])
        binding = view.retention_binding
        require(binding['retained_sampling_sha256'] == prior['retained_sampling_sha256'],
                'old retention subset changed')
        if split == 'validation':
            require(len(dataset) == old_count and dataset.new_correction_samples == 0
                    and binding['complete_sampling_sha256'] == prior['complete_sampling_sha256'],
                    'validation changed or received teacher corrections')
        else:
            require(dataset.new_correction_samples == len(dataset) - old_count > 0,
                    'no new corrective samples were admitted')
        result = dict(samples=len(dataset), old_samples=old_count, new_samples=len(dataset) - old_count,
            binding=binding, old_full_sequence_exact=True, source_exposure=dataset.source_sample_exposure,
            distinct_new_windows=0, distinct_new_branches=0, sampled_windows_verified=0)
        seen, branches = set(), set()
        # New sources are audited exhaustively without counting repeated anchor
        # weighting as additional independent supervision. Fixed old endpoints
        # also check that training and validation retention masks remain correct.
        indices = sorted({0, prior['retained_prefix_samples'] - 1, prior['retained_prefix_samples'], old_count - 1})
        indices += list(range(old_count, len(dataset)))
        for index in indices:
            identity = _identity(dataset, index)
            if index >= old_count and identity in seen:
                continue
            if index >= old_count:
                seen.add(identity); branches.add(identity[:3])
            sample = view[index]
            entry, start = dataset.samples[index]
            raw = dataset.cache
            n = len(raw['action'])
            frame_indices = np.minimum(start + np.arange(0, dataset.horizon + 1, dataset.video_stride), n)
            verify_window(sample, current_tick=int(raw['ticks'][start]), frame_ticks=raw['ticks'][frame_indices],
                horizon=dataset.horizon, video_stride=dataset.video_stride,
                retained=split == 'train' and index < prior['retained_prefix_samples'])
            action_indices = start + np.arange(dataset.horizon)
            require(torch.equal(sample['action'], torch.from_numpy(dataset._normalize(
                raw['action'][np.minimum(action_indices, n - 1)], 'action'))), 'public action label changed')
            require(torch.equal(sample['proprio'], torch.from_numpy(dataset._normalize(
                raw['proprio'][np.minimum(action_indices, n)], 'state'))), 'public robot state changed')
            require(torch.equal(sample['action_is_pad'], torch.from_numpy(action_indices >= n))
                    and torch.equal(sample['proprio_is_pad'], torch.from_numpy(action_indices > n))
                    and torch.equal(sample['image_is_pad'], torch.from_numpy(
                        start + np.arange(0, dataset.horizon + 1, dataset.video_stride) > n)),
                    'target padding differs from original trajectory length')
            result['sampled_windows_verified'] += 1
            if progress and result['sampled_windows_verified'] % 100 == 0:
                progress(dict(stage='materializing_' + split, sampled_windows_verified=result['sampled_windows_verified'],
                              new_distinct_windows=len(seen)))
        result.update(distinct_new_windows=len(seen), distinct_new_branches=len(branches))
        if split == 'train':
            tail = contract['correction_tail']
            new_roots = {str(Path(source['root']).resolve())
                         for source in tail['sources'][tail['preserved_tail_sources']:]}
            require(branches == {(str(Path(e['_root']).resolve()), e['path'], e['sha256'])
                                  for e in dataset.entries if e.get('_correction_tail_episode')
                                  and str(Path(e['_root']).resolve()) in new_roots},
                    'a corrective branch lacks verified model-visible supervision')
        report['splits'][split] = result
        recipe['retention']['data'][split] = {k: binding[k] for k in
            ('total_samples', 'retained_prefix_samples', 'retained_sampling_sha256', 'complete_sampling_sha256')}
        del sample, view, dataset; gc.collect()
    require(sha256(parent_path) == parent_sha and sha256(stats_path) == recipe['audit']['stats_sha256'],
            'frozen parent or normalization changed during audit')
    report.update(status='complete', extended_training_contract=contract,
                  validation_unchanged=True, normalization_unchanged=True,
                  next_gate='separate real-model forward/backward/checkpoint preflight before optimization')
    return recipe, report
