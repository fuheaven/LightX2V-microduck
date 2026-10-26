"""Append qualified obstacle data without dropping a retained training recipe.

The retained recipe is immutable, including recovery weights and camera prefixes.
This is a new experimental data route; existing extended runs are unchanged.
"""
from collections import Counter
from collections.abc import Sequence
import hashlib
import json
from pathlib import Path

import yaml

from .navigation_action_contracts import FORWARD04_NAVIGATION_CONTRACT
from .navigation_extended_release import load_extended_release
from .navigation_guarded_recovery import guarded_recovery_config, load_guarded_recoveries
from .wam_history_data import FastWAMNavigationHistoryDataset


def retained_dataset_config(path, digest, split):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError('retained training recipe checksum changed')
    recipe = yaml.safe_load(raw)
    config = dict(recipe['data']['train' if split == 'train' else 'val'])
    if config.pop('_target_') != 'lightx2v_microduck.wam_history_data.FastWAMNavigationHistoryDataset':
        raise ValueError('retained recipe must use the verified history dataset')
    if split not in ('train', 'validation') or config.get('split', 'train') != split:
        raise ValueError('retained recipe split differs; test data cannot enter training')
    if any(config.get(k) for k in ('augmentation_roots', 'retarget_roots', 'qualified_teacher_roots')):
        raise ValueError('rehearsal route supports the retained base and recovery sources only')
    if split == 'train' and not config.get('recovery_roots'):
        raise ValueError('retained training recipe contains no recovery data')
    return config


def appended_recovery_config(actual, retained, hashes=(), checkpoint=None):
    """Allow only a pinned training-only tail; retained fields remain exact."""
    if isinstance(hashes, (str, bytes)) or not isinstance(hashes, Sequence):
        raise ValueError('appended recovery hashes must be an explicit list')
    if not hashes:
        if checkpoint is not None or actual != retained:
            raise ValueError('rehearsal configuration differs from the retained recipe')
        return None
    if (actual.get('split') != 'train' or retained.get('sample_source_repeats')
            or not isinstance(checkpoint, str) or len(checkpoint) != 64
            or set(checkpoint) - set('0123456789abcdef')):
        raise ValueError('appended recoveries need a pinned student checkpoint and training-only sampling')
    original = dict(actual)
    added = {}
    for key in ('recovery_roots', 'history_prefix_roots', 'history_prefix_sha256'):
        old, combined = retained.get(key, []), actual.get(key, [])
        if (isinstance(combined, (str, bytes)) or not isinstance(combined, Sequence)
                or len(combined) != len(old) + len(hashes)
                or list(combined[:len(old)]) != list(old)):
            raise ValueError('appended corrections changed a retained source or history prefix')
        original[key] = old
        added[key] = list(combined[len(old):])
    if original != retained:
        raise ValueError('appended corrections changed the retained recipe or weighting')
    roots = [str(Path(p).resolve()) for p in actual['recovery_roots']]
    if len(roots) != len(set(roots)) or len(set(hashes)) != len(hashes):
        raise ValueError('duplicate appended recovery source')
    for root, digest in zip(added['recovery_roots'], hashes):
        path = Path(root) / 'index.json'
        if (not isinstance(digest, str) or len(digest) != 64
                or hashlib.sha256(path.read_bytes()).hexdigest() != digest):
            raise ValueError('appended recovery index changed')
        index = json.loads(path.read_text())
        if not index.get('episodes') or any(e.get('student_checkpoint_sha256') != checkpoint
                                             or e.get('split') != 'train' for e in index['episodes']):
            raise ValueError('appended recovery source checkpoint or split differs')
    for root, digest in zip(added['history_prefix_roots'], added['history_prefix_sha256']):
        if hashlib.sha256((Path(root)/'index.json').read_bytes()).hexdigest() != digest:
            raise ValueError('appended camera prefix index changed')
    return added


class FastWAMNavigationRehearsalHistoryDataset(FastWAMNavigationHistoryDataset):
    def __init__(self, root, split='train', *, retained_recipe_path, retained_recipe_sha256,
                 extended_release_root, extended_release_sha256, navigation_contract,
                  extended_repeat=1, appended_recovery_sha256=(),
                  appended_recovery_checkpoint_sha256=None, guarded_recovery_roots=(),
                  guarded_recovery_sha256=(), guarded_recovery_checkpoint_sha256=None,
                  guarded_recovery_checkpoint_sha256s=None, **kwargs):
        if navigation_contract != FORWARD04_NAVIGATION_CONTRACT:
            raise ValueError('guarded demonstrations require the explicit extended action contract')
        if type(extended_repeat) is not int or extended_repeat < 1 or (split != 'train' and extended_repeat != 1):
            raise ValueError('obstacle repetition must be a positive training-only integer')
        retained = retained_dataset_config(retained_recipe_path, retained_recipe_sha256, split)
        actual = dict(root=str(root), split=split, **kwargs)
        added = appended_recovery_config(actual, retained, appended_recovery_sha256,
                                         appended_recovery_checkpoint_sha256)
        guarded = guarded_recovery_config(guarded_recovery_roots, guarded_recovery_sha256,
                                          guarded_recovery_checkpoint_sha256, split,
                                          checkpoints=guarded_recovery_checkpoint_sha256s)
        released = load_extended_release(extended_release_root, root, extended_release_sha256)
        super().__init__(**retained)
        self.retained_samples = len(self.samples)
        self.retained_recipe_sha256 = retained_recipe_sha256
        source = str(Path(extended_release_root).resolve())
        if source in {str(Path(e.get('_root', self.root)).resolve()) for e in self.entries}:
            raise ValueError('obstacle source duplicates retained data')
        starts = []
        for entry in released:
            if entry['split'] == split:
                entry = dict(entry, _root=source, _qualified_teacher_episode=True, _extended_teacher_episode=True)
                self.entries.append(entry)
                starts.extend((entry, start) for start in range(0, entry['frames'] - 1,
                                                               int(retained.get('sample_stride', 4))))
        if not starts:
            raise ValueError('no guarded examples in the independent requested split')
        # Preserve the complete retained prefix of the sample list, including
        # all weighted student anchors and teacher continuation samples.
        self.samples.extend(row for row in starts for _ in range(extended_repeat))
        self.extended_teacher_samples = len(starts) * extended_repeat
        self.qualified_teacher_samples += self.extended_teacher_samples
        self.extended_release_sha256 = extended_release_sha256
        self.navigation_contract = navigation_contract
        self.pre_correction_samples = len(self.samples)
        self.appended_recovery_samples = 0
        added_sources = set()
        if added is not None:
            # Reuse the existing qualified release/prefix loader and sampling
            # semantics, but append only its new recoveries after the complete
            # retained+obstacle sample list. Never insert them into that prefix.
            extra = FastWAMNavigationHistoryDataset(**dict(retained, **added))
            added_sources = {str(Path(p).resolve()) for p in added['recovery_roots']}
            entries = [e for e in extra.entries if e.get('_root') in added_sources]
            from .navigation_recovery_data import identity
            old_keys = {identity(e) for e in self.entries
                        if e.get('_root') and not e.get('_qualified_teacher_episode')}
            if old_keys.intersection(identity(e) for e in entries):
                raise ValueError('appended recovery duplicates an existing student state')
            self.entries.extend(entries)
            samples = [(e, start) for e, start in extra.samples if e.get('_root') in added_sources]
            if not samples:
                raise ValueError('no eligible appended recovery samples')
            self.samples.extend(samples)
            self.history_prefixes.update(extra.history_prefixes)
            self.appended_recovery_samples = len(samples)
            for key in ('recovery_samples', 'recovery_student_anchor_samples', 'recovery_teacher_continuation_samples'):
                setattr(self, key, getattr(self, key) + getattr(extra, key))
        self.guarded_recovery_samples = 0
        guarded_sources = set()
        if guarded is not None:
            from .navigation_recovery_data import identity
            existing_sources = {str(Path(e.get('_root', self.root)).resolve()) for e in self.entries}
            seen = {identity(e) for e in self.entries if e.get('_root') and not e.get('_qualified_teacher_episode')}
            weights = retained.get('recovery_sampling') or {
                'student_anchor_repeats': retained.get('recovery_repeats', 1),
                'teacher_continuation_repeats': retained.get('recovery_repeats', 1)}
            checkpoints = guarded.get('student_checkpoint_sha256s')
            if checkpoints is None:
                checkpoints = [guarded['student_checkpoint_sha256']] * len(guarded['roots'])
            for recovery_root, digest, checkpoint in zip(guarded['roots'], guarded['index_sha256'], checkpoints):
                if recovery_root in existing_sources:
                    raise ValueError('guarded recovery source duplicates existing data')
                guarded_sources.add(recovery_root)
                entries = load_guarded_recoveries(recovery_root, root, extended_release_root,
                    extended_release_sha256, digest, checkpoint)
                for item in entries:
                    key = identity(item)
                    if key in seen:
                        raise ValueError('guarded recovery duplicates an existing student state')
                    seen.add(key)
                    entry = dict(item, _root=recovery_root, _guarded_recovery_episode=True)
                    self.entries.append(entry)
                    self.history_prefixes[(recovery_root, entry['path'])] = {
                        'path': str(Path(recovery_root) / entry['path'] / 'camera_prefix.npz'),
                        'sha256': entry['prefix_sha256']}
                    anchors = [(entry, start) for start in entry['actual_student_anchor_starts']] * weights['student_anchor_repeats']
                    continuation = [(entry, start) for start in entry['teacher_continuation_starts']] * weights['teacher_continuation_repeats']
                    self.samples.extend(anchors + continuation)
                    self.guarded_recovery_samples += len(anchors) + len(continuation)
                    self.recovery_samples += len(anchors) + len(continuation)
                    self.recovery_student_anchor_samples += len(anchors)
                    self.recovery_teacher_continuation_samples += len(continuation)
        self.source_sample_exposure = {}
        sources = Counter(str(Path(e.get('_root', self.root)).resolve()) for e, _ in self.samples)
        for name, count in sources.items():
            unique = {(e['path'], start) for e, start in self.samples
                      if str(Path(e.get('_root', self.root)).resolve()) == name}
            self.source_sample_exposure[name] = dict(unique_windows=len(unique), weighted_samples=count,
                weighting=('extended_repeat' if name == source else 'guarded_verified_recoveries'
                           if name in guarded_sources else 'appended_verified_recoveries'
                           if name in added_sources else 'unchanged_retained_recipe'))
