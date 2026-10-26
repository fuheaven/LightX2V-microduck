"""Portable measured camera prefixes for already qualified recovery labels."""
import json
from pathlib import Path

import numpy as np

from .navigation_augmentation import contained, sha256
from .navigation_recovery_data import load_recoveries
from .wam_data import write_json

SCHEMA = 'microduck_recovery_camera_prefix_v1'


def validate_prefix(rgb, ticks, source_tick, anchor):
    if (type(source_tick) is not int or source_tick < 0 or source_tick % 5
            or rgb.dtype != np.uint8 or rgb.shape != (source_tick // 5 + 1, 256, 256, 3)
            or not np.issubdtype(ticks.dtype, np.integer)
            or not np.array_equal(ticks, np.arange(source_tick // 5 + 1) * 5)
            or not np.array_equal(rgb[-1], anchor)):
        raise ValueError('camera prefix must run from tick zero through the exact recorded anchor')


def export_prefixes(recovery_root, base_dataset, output, *, current_checkpoint_sha256=None,
                    qualified_teacher_roots=None, qualified_teacher_sha256=None):
    recovery_root, output = Path(recovery_root), Path(output)
    if output.exists():
        raise FileExistsError('history prefix releases are immutable')
    entries = load_recoveries(recovery_root, base_dataset,
                              qualified_teacher_roots=qualified_teacher_roots,
                              qualified_teacher_sha256=qualified_teacher_sha256)
    if current_checkpoint_sha256 is not None and (
            not isinstance(current_checkpoint_sha256, str)
            or len(current_checkpoint_sha256) != 64
            or set(current_checkpoint_sha256) - set('0123456789abcdef')
            or not entries
            or any(e['student_checkpoint_sha256'] != current_checkpoint_sha256 for e in entries)):
        raise ValueError('current-policy prefixes require the exact source checkpoint for every recovery')
    protocol = json.loads((recovery_root / 'source_protocol.json').read_text())
    sources = {(s['task'], s['seed']): s for s in protocol['sources']}
    if len(sources) != len(protocol['sources']):
        raise ValueError('duplicate source identities')
    payloads = []
    for entry in entries:
        source = sources[entry['task'], entry['seed']]
        folder = Path(source['path'])
        if source['branch_tick'] != entry['source_tick']:
            raise ValueError('source branch differs')
        for name, digest in source['files'].items():
            if sha256(contained(folder, name)) != digest:
                raise ValueError('original student recording changed')
        if sha256(folder / 'recording.npz') != entry['source_recording_sha256']:
            raise ValueError('prefix belongs to another recording')
        report = json.loads((folder / 'report.json').read_text())
        if ((report['task'], report['seed'], report['split']) != (entry['task'], entry['seed'], 'train')
                or report['policy']['artifact_sha256'] != entry['student_checkpoint_sha256']
                or report['policy']['uses_privileged_state'] is not False):
            raise ValueError('prefix source policy or training split differs')
        with np.load(folder / 'recording.npz', allow_pickle=False) as z:
            ticks, rgb = z['camera_ticks'], z['rgb']
            end = entry['source_tick'] // 5 + 1
            ticks, rgb = ticks[:end].copy(), rgb[:end].copy()
        with np.load(contained(recovery_root, entry['path']) / 'navigation.npz', allow_pickle=False) as z:
            validate_prefix(rgb, ticks, entry['source_tick'], z['rgb'][0])
        payloads.append((entry, rgb, ticks))
    output.mkdir(parents=True)
    rows = []
    for entry, rgb, ticks in payloads:
        relative = entry['path'] + '/camera_prefix.npz'
        path = contained(output, relative);path.parent.mkdir(parents=True)
        np.savez_compressed(path, rgb=rgb, ticks=ticks)
        rows.append({'recovery_path': entry['path'], 'path': relative, 'sha256': sha256(path),
                     'source_tick': entry['source_tick'], 'source_recording_sha256': entry['source_recording_sha256'],
                     'recovery_navigation_sha256': entry['sha256'],
                     'student_checkpoint_sha256': entry['student_checkpoint_sha256']})
    index = {'schema': SCHEMA, 'recovery_index_sha256': sha256(recovery_root / 'index.json'),
             'source_protocol_sha256': sha256(recovery_root / 'source_protocol.json'),
             'public_tensor_keys': ['rgb', 'ticks'], 'episodes': rows,
             'current_model_rollouts': current_checkpoint_sha256 is not None, 'capability_accepted': False}
    if current_checkpoint_sha256 is not None:
        index['current_checkpoint_sha256'] = current_checkpoint_sha256
    write_json(output / 'index.json', index)
    return {'root': str(output), 'index_sha256': sha256(output / 'index.json'), 'episodes': len(rows)}


def load_prefixes(root, expected_hash, recovery_root, entries):
    root, recovery_root = Path(root), Path(recovery_root)
    if sha256(root / 'index.json') != expected_hash:
        raise ValueError('history prefix index changed')
    index = json.loads((root / 'index.json').read_text())
    if (index['schema'] != SCHEMA or index['recovery_index_sha256'] != sha256(recovery_root / 'index.json')
            or index['source_protocol_sha256'] != sha256(recovery_root / 'source_protocol.json')
            or index['public_tensor_keys'] != ['rgb', 'ticks']):
        raise ValueError('prefixes do not match the exact recovery release')
    rows = {e['recovery_path']: e for e in index['episodes']}
    if len(rows) != len(index['episodes']) or set(rows) != {e['path'] for e in entries}:
        raise ValueError('prefix cohort is missing or duplicated')
    result = {}
    for entry in entries:
        row = rows[entry['path']]
        for key in ('source_tick', 'source_recording_sha256', 'student_checkpoint_sha256'):
            if row[key] != entry[key]:
                raise ValueError('prefix identity changed')
        if row['recovery_navigation_sha256'] != entry['sha256']:
            raise ValueError('prefix teacher continuation changed')
        path = contained(root, row['path'])
        if sha256(path) != row['sha256']:
            raise ValueError('prefix pixels changed')
        with np.load(path, allow_pickle=False) as z, np.load(contained(recovery_root, entry['path']) / 'navigation.npz', allow_pickle=False) as branch:
            if set(z.files) != {'rgb', 'ticks'}:
                raise ValueError('only RGB and timestamps may enter history prefixes')
            validate_prefix(z['rgb'], z['ticks'], row['source_tick'], branch['rgb'][0])
        result[entry['path']] = {'path': path, 'sha256': row['sha256']}
    return result


def training_prefix_hashes(train, validation):
    if validation.get('history_prefix_roots') or validation.get('history_prefix_sha256'):
        raise ValueError('recovery camera prefixes cannot enter validation')
    roots, hashes = train.get('history_prefix_roots', []), train.get('history_prefix_sha256', [])
    targets = {'lightx2v_microduck.wam_history_data.FastWAMNavigationHistoryDataset',
               'lightx2v_microduck.wam_rehearsal_history_data.FastWAMNavigationRehearsalHistoryDataset',
               'lightx2v_microduck.wam_memory_rehearsal_data.FastWAMNavigationMemoryRehearsalDataset'}
    if (isinstance(roots, (str, Path)) or isinstance(hashes, str) or len(roots) != len(hashes)
            or (roots and (train.get('_target_') not in targets
                           or len(roots) != len(train.get('recovery_roots', []))))):
        raise ValueError('every history recovery prefix requires a pinned matching release')
    for root, digest in zip(roots, hashes):
        if sha256(Path(root) / 'index.json') != digest:
            raise ValueError('history prefix index changed since recipe was frozen')
    return list(hashes)


def verify_resume_prefixes(receipt, hashes):
    if receipt.get('history_prefix_index_sha256', []) != hashes:
        raise ValueError('resume history prefixes changed; initialize an explicit new training stage')
