"""Explicit mixed clear-room/guarded data; legacy dataset behavior is unchanged."""
from pathlib import Path

from .navigation_action_contracts import FORWARD04_NAVIGATION_CONTRACT
from .navigation_extended_release import load_extended_release
from .wam_history_data import FastWAMNavigationHistoryDataset


class FastWAMNavigationExtendedHistoryDataset(FastWAMNavigationHistoryDataset):
    def __init__(self, root, split='train', *, extended_release_root, extended_release_sha256,
                 navigation_contract, clear_repeat=1, extended_repeat=1, **kwargs):
        if navigation_contract != FORWARD04_NAVIGATION_CONTRACT:
            raise ValueError('guarded demonstrations require their explicit extended action contract')
        for value in (clear_repeat, extended_repeat):
            if type(value) is not int or value < 1:
                raise ValueError('dataset source repetitions must be positive integers')
        if split != 'train' and (clear_repeat != 1 or extended_repeat != 1):
            raise ValueError('validation and test samples cannot be repeated')
        if any(kwargs.get(k) for k in ('qualified_teacher_roots', 'augmentation_roots', 'retarget_roots',
                'recovery_roots', 'sample_source_repeats')):
            raise ValueError('the first extended recipe mixes only immutable clear-room and separate guarded releases')
        released = load_extended_release(extended_release_root, root, extended_release_sha256)
        super().__init__(root, split, **kwargs)
        source = str(Path(extended_release_root).resolve())
        starts = []
        for entry in released:
            if entry['split'] != split:
                continue
            # A complete qualified teacher episode has its own full camera
            # history from tick0, unlike a student-state correction branch.
            entry = dict(entry, _root=source, _qualified_teacher_episode=True, _extended_teacher_episode=True)
            self.entries.append(entry)
            starts.extend((entry, i) for i in range(0, entry['frames'] - 1, int(kwargs.get('sample_stride', 4))))
        if not starts:
            raise ValueError('no eligible guarded examples in the requested independent split')
        original_base = self.samples
        self.samples = [row for row in original_base for _ in range(clear_repeat)]
        self.samples.extend(row for row in starts for _ in range(extended_repeat))
        self.base_samples = len(original_base) * clear_repeat
        self.qualified_teacher_samples = self.extended_teacher_samples = len(starts) * extended_repeat
        self.extended_release_sha256 = extended_release_sha256
        self.navigation_contract = navigation_contract
        self.extended_source_exposure = dict(
            clear=dict(original_samples=len(original_base), repeats=clear_repeat, samples=self.base_samples),
            guarded=dict(original_samples=len(starts), repeats=extended_repeat, samples=self.extended_teacher_samples))
        # The trainer records this shared attribute. The inherited value was
        # None even though the extended loader had changed source exposure.
        self.source_sample_exposure = {
            str(Path(root).resolve()): dict(original_samples=len(original_base),
                repeats=clear_repeat, weighted_samples=self.base_samples),
            source: dict(original_samples=len(starts), repeats=extended_repeat,
                weighted_samples=self.extended_teacher_samples),
        }
