"""Causal, fixed-timestamp camera grids for explicit navigation experiments.

This is a spatial grid of measured history, not predicted video or a recurrent
world model. The same composition function serves training and inference. It
never receives detections, target positions, task outcomes or teacher state.
"""
from dataclasses import dataclass
from math import isqrt

import numpy as np


@dataclass(frozen=True)
class HistoryGridSpec:
    offsets_ticks: tuple[int, ...]
    camera_size: int = 256
    max_sample_age_ticks: int = 5

    def __post_init__(self):
        offsets = tuple(self.offsets_ticks)
        if (len(offsets) not in (4, 9) or any(type(t) is not int or t < 0 or t % 5 for t in offsets)
                or tuple(sorted(set(offsets), reverse=True)) != offsets or offsets[-1] != 0):
            raise ValueError('history requires four or nine distinct, descending decision-aligned offsets ending at zero')
        if self.camera_size != 256 or type(self.max_sample_age_ticks) is not int or not 0 <= self.max_sample_age_ticks <= 5:
            raise ValueError('history v1 requires 256px measured frames and at most one decision of staleness')
        object.__setattr__(self, 'offsets_ticks', offsets)

    @property
    def output_size(self):
        return self.camera_size * isqrt(len(self.offsets_ticks))

    def as_dict(self):
        return {'schema': 'microduck_causal_rgb_grid_v1', 'offsets_ticks': list(self.offsets_ticks),
                'camera_size': self.camera_size, 'output_size': self.output_size,
                'max_sample_age_ticks': self.max_sample_age_ticks,
                'layout': 'row_major_oldest_to_current', 'startup_padding': 'repeat_episode_first_frame',
                'selector': 'fixed_timestamps_only', 'control_hz': 50}


def training_history_contract(train, validation):
    """Bind input semantics to recipes and checkpoint receipts, not tensor size."""
    targets = {'lightx2v_microduck.wam_history_data.FastWAMNavigationHistoryDataset',
               'lightx2v_microduck.wam_extended_history_data.FastWAMNavigationExtendedHistoryDataset',
               'lightx2v_microduck.wam_rehearsal_history_data.FastWAMNavigationRehearsalHistoryDataset'}
    contracts = []
    for config in (train, validation):
        if config.get('_target_') == 'lightx2v_microduck.wam_memory_rehearsal_data.FastWAMNavigationMemoryRehearsalDataset':
            from .wam_memory_rehearsal_data import memory_rehearsal_spec

            contracts.append(memory_rehearsal_spec(config.get('memory_input_contract')).as_dict())
        elif config.get('memory_input_contract') is not None:
            raise ValueError('paired memory inputs require the memory rehearsal dataset implementation')
        elif config.get('_target_') in targets:
            spec = history_spec(tuple(config.get('history_offsets_ticks', ())),
                                target_keyframe=config.get('history_target_keyframe', False),
                                target_keyframe_fallback_ticks=config.get('history_target_keyframe_fallback_ticks'))
            contracts.append(spec.as_dict())
        else:
            if (config.get('history_offsets_ticks') is not None or config.get('history_target_keyframe')
                    or config.get('history_target_keyframe_fallback_ticks') is not None):
                raise ValueError('history offsets require the history dataset implementation')
            contracts.append(None)
    if contracts[0] != contracts[1]:
        raise ValueError('training and validation observation histories differ')
    return contracts[0]


def history_spec(offsets, *, target_keyframe=False, max_sample_age_ticks=5,
                 target_keyframe_fallback_ticks=None):
    if type(target_keyframe) is not bool:
        raise ValueError('target keyframe must be explicitly enabled or disabled')
    if target_keyframe:
        from .navigation_visual_memory import TargetMemoryGridSpec
        return TargetMemoryGridSpec(tuple(offsets), max_sample_age_ticks=max_sample_age_ticks,
                                    fallback_history_ticks=target_keyframe_fallback_ticks)
    if target_keyframe_fallback_ticks is not None:
        raise ValueError('keyframe fallback requires the target memory input contract')
    return HistoryGridSpec(tuple(offsets), max_sample_age_ticks=max_sample_age_ticks)


def history_spec_from_contract(contract):
    spec = history_spec(contract.get('offsets_ticks', ()),
                        target_keyframe=contract.get('schema') in (
                            'microduck_rgb_target_keyframe_v1', 'microduck_rgb_target_keyframe_v2'),
                        max_sample_age_ticks=contract.get('max_sample_age_ticks', -1),
                        target_keyframe_fallback_ticks=contract.get('fallback_history_ticks'))
    if spec.as_dict() != contract:
        raise ValueError('history input contract differs from its implementation')
    return spec


def verify_resume_history(receipt, contract):
    if receipt.get('input_history') != contract:
        raise ValueError('resume observation history changed; initialize an explicitly new training stage')


def verify_evaluation_history(receipt, spec, *, weights_sha256, stats_sha256):
    """A compatible image shape does not prove weights were trained for history."""
    verify_resume_history(receipt, spec.as_dict())
    if (receipt.get('weights_sha256') != weights_sha256
            or receipt.get('stats_sha256') != stats_sha256
            or type(receipt.get('step')) is not int or receipt['step'] < 1):
        raise ValueError('history evaluation requires the matching trained checkpoint and statistics')


def compose_history_grid(rgb, ticks, source_tick, spec):
    """Select only samples at/before each requested timestamp; never interpolate."""
    rgb, ticks = np.asarray(rgb), np.asarray(ticks)
    if (rgb.dtype != np.uint8 or rgb.shape != (len(ticks), 256, 256, 3)
            or ticks.ndim != 1 or not np.issubdtype(ticks.dtype, np.integer) or not len(ticks)
            or ticks[0] != 0 or np.any(np.diff(ticks) <= 0)):
        raise ValueError('history must contain ordered measured RGB starting at episode tick zero')
    if type(source_tick) is not int or source_tick < 0 or source_tick % 5:
        raise ValueError('source tick must be a navigation decision')
    requested = np.asarray([source_tick - lag for lag in spec.offsets_ticks], dtype=np.int64)
    indices = np.searchsorted(ticks, np.maximum(requested, 0), side='right') - 1
    if np.any(indices < 0):
        raise ValueError('missing episode prefix')
    selected_ticks = ticks[indices]
    ages = np.maximum(requested, 0) - selected_ticks
    if selected_ticks[-1] != source_tick or np.any(ages > spec.max_sample_age_ticks):
        raise ValueError('missing or stale history sample')
    side = isqrt(len(indices))
    grid = np.concatenate([np.concatenate([rgb[i] for i in indices[row * side:(row + 1) * side]], axis=1)
                           for row in range(side)], axis=0)
    return grid, selected_ticks.copy(), requested < 0


class CausalCameraHistory:
    """Per-episode measured frames, retained for a bounded simulation horizon."""
    def __init__(self, spec):
        self.spec = spec
        self.reset('')

    def reset(self, episode_id):
        self.episode_id = str(episode_id)
        self._frames, self._ticks = [], []

    def append(self, episode_id, tick, rgb):
        if not self.episode_id or episode_id != self.episode_id:
            raise ValueError('reset history for this episode first')
        frame = np.asarray(rgb)
        if (type(tick) is not int or tick < 0 or tick % 5 or frame.dtype != np.uint8
                or frame.shape != (256, 256, 3)):
            raise ValueError('history expects decision-aligned 256px RGB')
        if not self._ticks and tick != 0:
            raise ValueError('missing episode prefix')
        if self._ticks and tick <= self._ticks[-1]:
            if tick == self._ticks[-1] and np.array_equal(frame, self._frames[-1]):
                return
            raise ValueError('history cannot rewrite or reorder observations')
        self._ticks.append(tick)
        self._frames.append(frame.copy())
        # Keep the episode's first frame for explicit startup padding and all
        # samples within the oldest requested time plus allowed sample age.
        cutoff = tick - max(self.spec.offsets_ticks) - self.spec.max_sample_age_ticks
        while len(self._ticks) > 2 and self._ticks[1] < cutoff:
            del self._ticks[1]
            del self._frames[1]

    def compose(self, tick):
        return compose_history_grid(np.asarray(self._frames), np.asarray(self._ticks, dtype=np.int64), tick, self.spec)
