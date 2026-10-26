"""Explicit history-grid training data; existing single-frame recipes unchanged."""
from pathlib import Path
import json

import numpy as np

from .navigation_history import history_spec, compose_history_grid
from .wam_data import FastWAMNavigationDataset


class FastWAMNavigationHistoryDataset(FastWAMNavigationDataset):
    def __init__(self, root, split='train', *, history_offsets_ticks,
                 history_prefix_roots=None, history_prefix_sha256=None,
                 history_target_keyframe=False, history_target_keyframe_fallback_ticks=None, **kwargs):
        recovery_roots = kwargs.get('recovery_roots') or []
        prefix_roots, prefix_hashes = history_prefix_roots or [], history_prefix_sha256 or []
        if (isinstance(prefix_roots, (str, Path)) or isinstance(prefix_hashes, str)
                or any(kwargs.get(k) for k in ('augmentation_roots', 'retarget_roots'))
                or len(recovery_roots) != len(prefix_roots) or len(prefix_roots) != len(prefix_hashes)):
            raise ValueError('history training requires verified complete original episode prefixes')
        if prefix_roots and split != 'train':
            raise ValueError('recovery prefixes may only augment training')
        self.history_spec = history_spec(tuple(history_offsets_ticks), target_keyframe=history_target_keyframe,
                                        target_keyframe_fallback_ticks=history_target_keyframe_fallback_ticks)
        self.history_target_keyframe = history_target_keyframe
        self._target_memory_indices = {}
        self.history_contract = self.history_spec.as_dict()
        super().__init__(root, split, **kwargs)
        self.history_prefixes = {}
        if prefix_roots:
            from .navigation_history_recovery import load_prefixes
            for recovery, prefix, digest in zip(recovery_roots, prefix_roots, prefix_hashes):
                entries = [e for e in self.entries if e.get('_root') == str(Path(recovery).resolve())]
                for path, item in load_prefixes(prefix, digest, recovery, entries).items():
                    self.history_prefixes[(str(Path(recovery).resolve()), path)] = item
        self._history_cache_path = None
        # Populated only by a separate, qualified realtime release loader.
        # Ordinary dense/offline recoveries retain their existing validation.
        self.realtime_history_prefixes = {}
        self._realtime_history_cache = {}
        self._realtime_history_cache_path = None

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        return self.compose_sample_history(index, sample, self.history_spec,
                                           target_keyframe=self.history_target_keyframe)

    def compose_sample_history(self, index, sample, spec, *, target_keyframe):
        """Compose another declared view of the same loaded sample and labels."""
        import torch

        entry, start = self.samples[index]
        data = self.cache
        if entry.get('_realtime_recovery_episode', False):
            return self.compose_realtime_sample(entry, start, sample, spec, target_keyframe=target_keyframe)
        ticks = np.asarray(data['ticks'])
        images, camera_ticks = data['rgb'], ticks
        if entry.get('_root') is not None and not entry.get('_qualified_teacher_episode', False):
            from .navigation_augmentation import sha256
            from .navigation_history_recovery import validate_prefix
            item = self.history_prefixes[(entry['_root'], entry['path'])]
            if self._history_cache_path != item['path']:
                if sha256(item['path']) != item['sha256']:
                    raise ValueError('prefix pixels changed after loading')
                with np.load(item['path'], allow_pickle=False) as z:
                    self._prefix_rgb, self._prefix_ticks = z['rgb'].copy(), z['ticks'].copy()
                self._history_cache_path = item['path']
            validate_prefix(self._prefix_rgb, self._prefix_ticks, entry['source_tick'], images[0])
            images = np.concatenate((self._prefix_rgb[:-1], images))
            camera_ticks = np.concatenate((self._prefix_ticks[:-1], ticks))
        if not np.array_equal(camera_ticks, np.arange(len(images)) * 5):
            raise ValueError('history dataset requires an intact 10Hz episode from tick zero')
        if target_keyframe:
            from .navigation_visual_memory import target_memory_ticks, compose_target_memory_grid
            key = (str(self.cache_path), sample['prompt'])
            if key not in self._target_memory_indices:
                self._target_memory_indices[key] = target_memory_ticks(images, camera_ticks, sample['prompt'])
            memory_ticks = self._target_memory_indices[key]
        # Each target-video frame contains only its own current/past samples.
        # The action-conditioning first frame never includes target-video future.
        n = len(data['action'])
        indices = np.minimum(start + np.arange(0, self.horizon + 1, self.video_stride), n)
        grids, chosen, padding = [], [], []
        for i in indices:
            if target_keyframe:
                grid, source_ticks, is_pad = compose_target_memory_grid(
                    images, camera_ticks, int(ticks[i]), spec, memory_ticks)
            else:
                grid, source_ticks, is_pad = compose_history_grid(images, camera_ticks, int(ticks[i]), spec)
            grids.append(grid)
            chosen.append(source_ticks)
            padding.append(is_pad)
        sample['video'] = torch.from_numpy(np.stack(grids)).permute(3, 0, 1, 2).float().div(127.5).sub(1)
        sample['history_source_ticks'] = torch.from_numpy(np.stack(chosen))
        sample['history_is_pad'] = torch.from_numpy(np.stack(padding))
        return sample

    def compose_realtime_sample(self, entry, start, sample, spec, *, target_keyframe):
        from .navigation_outcomes import require
        from .navigation_realtime_history import read_measured_prefix, index_branch_history
        from .navigation_visual_memory import TargetMemoryGridSpec
        from .wam_data import PROMPT

        require(self.split == entry.get('split') == 'train', 'realtime corrections may only enter TRAIN')
        require(bool(target_keyframe) == isinstance(spec, TargetMemoryGridSpec),
                'realtime history selector differs from its declared input')
        binding = self.realtime_history_prefixes.get((entry.get('_root'), entry['path']))
        require(binding is not None, 'realtime correction has no registered source prefix')
        # One episode at a time, with both student and retention views cached.
        # Keep the complete measured prefix, but do not cache all dataset RGB.
        if self._realtime_history_cache_path != self.cache_path:
            self._realtime_history_cache = {}
            self._realtime_history_cache_path = self.cache_path
        key = json.dumps(spec.as_dict(), sort_keys=True)
        if key not in self._realtime_history_cache:
            prefix = read_measured_prefix(binding['path'], entry['source_tick'],
                                          expected_files=binding['files'])
            self._realtime_history_cache[key] = (index_branch_history(prefix, self.cache, spec),
                                                 prefix.instruction)
        history, instruction = self._realtime_history_cache[key]
        require(sample['prompt'] == PROMPT.format(task=instruction),
                'branch instruction differs from its original observed instruction')
        return history.compose_sample(
            sample, start=start, horizon=self.horizon, video_stride=self.video_stride)
