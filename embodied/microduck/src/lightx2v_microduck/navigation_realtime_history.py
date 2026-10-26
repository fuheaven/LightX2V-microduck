"""Measured realtime camera prefixes for teacher-branch training inputs.

This is tensor construction, not dataset admission. It preserves camera-loss
receipts, uses the deployed history selectors, and excludes windows whose
required observations are unavailable. A separate release must verify TRAIN
membership, the actual student replay and a successful qualified teacher branch.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from math import isqrt
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .navigation_camera_recovery import DeclaredLossTargetCameraHistory
from .navigation_history import CausalCameraHistory, HistoryGridSpec
from .navigation_outcomes import read, require, sha
from .navigation_visual_memory import TargetMemoryGridSpec


@dataclass(frozen=True)
class MeasuredCameraPrefix:
    rgb: np.ndarray
    ticks: np.ndarray
    capture_ns: np.ndarray
    receipts: tuple[dict, ...]
    episode_id: str
    instruction: str
    source_files: dict[str, str]

    @property
    def source_tick(self):
        return int(self.ticks[-1])


def read_measured_prefix(folder, source_tick, *, expected_files):
    """Read only frames delivered to the policy before the requested anchor.

    The startup frame is bound to its original prediction receipt. Every later
    captured or missing frame is bound to the snapshot-camera receipt. Images
    rendered only after the measured episode cannot become training anchors.
    """
    folder = Path(folder)
    required = {'report.json', 'recording.npz', 'predictions.json', 'camera_delivery.json'}
    require(required <= expected_files.keys(), 'camera prefix needs complete source hashes')
    require(type(source_tick) is int and 0 <= source_tick < 1500 and source_tick % 5 == 0,
            'camera prefix anchor must be an original navigation tick')
    for name, digest in expected_files.items():
        require(Path(name).name == name and not (folder/name).is_symlink()
                and sha(folder/name) == digest, 'camera prefix source changed')
    report, predictions = read(folder/'report.json'), read(folder/'predictions.json')
    require(report['mode'] == 'realtime' and report['initialization'] == 'reset_and_warmup',
            'camera prefix requires the original realtime initialization')
    starts = [p for p in predictions if p['phase'] == 'startup' and p['source_tick'] == 0]
    require(len(starts) == 1 and starts[0]['camera_tick'] == 0,
            'one original startup camera receipt is required')
    start = starts[0]
    require(isinstance(start['source_episode_id'], str) and start['source_episode_id']
            and isinstance(start['observed_instruction'], str) and start['observed_instruction']
            and type(start['source_monotonic_ns']) is int and start['source_monotonic_ns'] > 0,
            'startup camera identity or capture time is invalid')
    if report.get('evaluator', {}).get('episode_id') is not None:
        require(report['evaluator']['episode_id'] == start['source_episode_id'],
                'startup camera belongs to another episode')
    with np.load(folder/'recording.npz', allow_pickle=False) as packed:
        ticks, images = packed['camera_ticks'], packed['rgb']
        require(ticks.ndim == 1 and np.issubdtype(ticks.dtype, np.integer) and len(ticks)
                and ticks[0] == 0 and np.all(np.diff(ticks) > 0) and np.all(ticks % 5 == 0)
                and images.dtype == np.uint8 and images.shape == (len(ticks), 256, 256, 3),
                'recorded camera samples are missing, reordered or malformed')
        at = int(np.searchsorted(ticks, source_tick))
        require(at < len(ticks) and ticks[at] == source_tick, 'anchor image was never captured')
        ticks, images = ticks[:at+1].copy(), images[:at+1].copy()
    rows = read(folder/'camera_delivery.json')
    require(isinstance(rows, list), 'camera receipts must be a list')
    prefix_rows = [copy.deepcopy(r) for r in rows
                   if type(r.get('source_tick')) is int and 0 < r['source_tick'] <= source_tick]
    mapping = {r['source_tick']: r for r in prefix_rows}
    require(len(mapping) == len(prefix_rows)
            and set(mapping) == set(range(5, source_tick+1, 5)),
            'every prefix capture needs exactly one outcome receipt')
    captured = set(ticks.tolist())
    times = [start['source_monotonic_ns']]
    previous_capture, previous_delivery = times[0], start['admission_monotonic_ns']
    require(type(previous_delivery) is int and previous_delivery >= previous_capture,
            'startup frame admission predates its capture')
    for tick in range(5, source_tick+1, 5):
        row = mapping[tick]
        timestamp = row.get('capture_ns')
        require(type(timestamp) is int and timestamp > previous_capture,
                'prefix capture timestamps changed order')
        previous_capture = timestamp
        if tick in captured:
            require(row.get('status') == 'delivered' and row.get('delivery_phase') == 'measured'
                    and type(row.get('ready_ns')) is int and row['ready_ns'] >= timestamp
                    and type(row.get('delivered_ns')) is int
                    and row['delivered_ns'] >= max(row['ready_ns'], previous_delivery),
                    'prefix image was not delivered during the measured episode')
            previous_delivery = row['delivered_ns']
            times.append(timestamp)
        else:
            require(row.get('status') in ('dropped_busy', 'error')
                    and (row['status'] != 'error' or
                         isinstance(row.get('error'), str) and bool(row['error'])),
                    'a missing prefix image lacks an actual loss receipt')
    require(all(sha(folder/name) == digest for name, digest in expected_files.items()),
            'prefix source changed while reading')
    return MeasuredCameraPrefix(images, ticks, np.asarray(times, np.int64),
        tuple(prefix_rows), start['source_episode_id'], start['observed_instruction'], dict(expected_files))


@dataclass
class BranchHistoryIndex:
    """Grid selections, with failed selections retained instead of fabricated."""
    rgb: np.ndarray
    measured_ticks: np.ndarray
    branch_ticks: np.ndarray
    source_ticks: np.ndarray
    padding: np.ndarray
    errors: dict[int, str]
    history_contract: dict

    def compose(self, frame_index):
        require(type(frame_index) is int and 0 <= frame_index < len(self.branch_ticks),
                'history frame index is out of bounds')
        require(frame_index not in self.errors, 'unavailable history: '+self.errors.get(frame_index, ''))
        indices = np.searchsorted(self.measured_ticks, self.source_ticks[frame_index])
        require(np.all(indices < len(self.rgb))
                and np.array_equal(self.measured_ticks[indices], self.source_ticks[frame_index]),
                'selected history image is not measured')
        side = isqrt(len(indices))
        return np.concatenate([np.concatenate([self.rgb[i] for i in indices[row*side:(row+1)*side]],
                               axis=1) for row in range(side)], axis=0)

    def compose_sample(self, sample, *, start, horizon, video_stride):
        """Replace only the camera view of an already normalized action sample."""
        import torch

        require(all(type(x) is int and x > 0 for x in (horizon, video_stride))
                and type(start) is int and 0 <= start < len(self.branch_ticks)-1,
                'invalid realtime history sample window')
        indices = np.minimum(start + np.arange(0, horizon+1, video_stride), len(self.branch_ticks)-1)
        # compose rejects every unavailable historical frame, including video
        # prediction targets. It never substitutes the next successful image.
        grids = [self.compose(int(i)) for i in indices]
        sample['video'] = torch.from_numpy(np.stack(grids)).permute(3, 0, 1, 2).float().div(127.5).sub(1)
        sample['history_source_ticks'] = torch.from_numpy(self.source_ticks[indices].copy())
        sample['history_is_pad'] = torch.from_numpy(self.padding[indices].copy())
        return sample

    def window_starts(self, *, horizon, video_stride, sample_stride):
        require(all(type(x) is int and x > 0 for x in (horizon, video_stride, sample_stride)),
                'positive integer sampling parameters are required')
        count = len(self.branch_ticks)-1
        accepted, rejected = [], []
        for start in range(0, count, sample_stride):
            indices = np.minimum(start + np.arange(0, horizon+1, video_stride), count)
            unavailable = sorted({int(i) for i in indices if int(i) in self.errors})
            if unavailable:
                rejected.append(dict(start=start, unavailable_frames=unavailable))
            else:
                accepted.append(start)
        return dict(accepted=accepted, rejected=rejected)


def index_branch_history(prefix, public, spec):
    """Use the same streaming selectors as deployment on real prefix frames.

    Teacher continuation frames are measured simulated futures. They advance
    the physical tick only; no wall-clock capture timestamps are invented for
    them. The existing stale-sample limit applies to both current and future
    training frames. This does not mark an incomplete realtime episode passed.
    """
    from .navigation_extended_release import validate_public

    require(isinstance(prefix, MeasuredCameraPrefix), 'a measured source prefix is required')
    require(set(public) == {'rgb', 'proprio', 'action', 'ticks'}, 'only public teacher tensors are accepted')
    start = prefix.source_tick
    validate_public(dict(public, ticks=public['ticks']-start), len(public['action']))
    require(public['ticks'][0] == start and public['ticks'][-1] <= 1500,
            'branch timestamps changed the original task clock')
    np.testing.assert_array_equal(public['rgb'][0], prefix.rgb[-1])
    require(isinstance(spec, (HistoryGridSpec, TargetMemoryGridSpec)), 'unsupported history layout')
    # The target-memory ingestion path validates receipt semantics for both
    # views; a fixed-history retention view still uses its original selector.
    measured = DeclaredLossTargetCameraHistory(
        spec if isinstance(spec, TargetMemoryGridSpec) else TargetMemoryGridSpec(fallback_history_ticks=300))
    measured.reset(prefix.episode_id)
    fixed = CausalCameraHistory(spec) if isinstance(spec, HistoryGridSpec) else None
    if fixed is not None:
        fixed.reset(prefix.episode_id)
    require(prefix.capture_ns.shape == prefix.ticks.shape and len(prefix.rgb) == len(prefix.ticks),
            'prefix camera timestamps do not match its images')
    for tick, image, timestamp in zip(prefix.ticks, prefix.rgb, prefix.capture_ns):
        observed = SimpleNamespace(episode_id=prefix.episode_id, tick=int(tick), camera_tick=int(tick),
            rgb=image, monotonic_ns=int(timestamp), instruction=prefix.instruction)
        measured.ingest(observed, prefix.receipts)
        if fixed is not None:
            fixed.append(prefix.episode_id, int(tick), image)
    history = measured if fixed is None else fixed
    slots = len(spec.offsets_ticks) + int(isinstance(spec, TargetMemoryGridSpec))
    chosen = np.full((len(public['ticks']), slots), -1, np.int64)
    padding = np.zeros(chosen.shape, bool)
    errors = {}
    for i, (tick, image) in enumerate(zip(public['ticks'], public['rgb'])):
        tick = int(tick)
        if i:
            if fixed is None:
                history.append(prefix.episode_id, tick, image, instruction=prefix.instruction)
            else:
                history.append(prefix.episode_id, tick, image)
        try:
            _, chosen[i], padding[i] = history.compose(tick)
        except ValueError as exc:
            # The episode can become usable again after a gap moves out of the
            # requested history. Preserve the rejection for window selection.
            if str(exc) not in ('missing or stale recent camera sample',
                                'missing or stale fallback camera sample', 'missing or stale history sample'):
                raise
            errors[i] = str(exc)
    return BranchHistoryIndex(np.concatenate((prefix.rgb[:-1], public['rgb'])),
        np.concatenate((prefix.ticks[:-1], public['ticks'])), public['ticks'].copy(),
        chosen, padding, errors, spec.as_dict())
