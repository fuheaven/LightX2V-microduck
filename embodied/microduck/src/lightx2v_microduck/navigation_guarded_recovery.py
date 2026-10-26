"""Audited guarded-teacher recoveries with the student's complete visual history.

This is a distinct forward-0.4 release. It cannot enter the legacy clear-room
loader or use its seed hash to override the guarded release's fixed split.
Raw physics/teacher evidence stays outside the four model-visible tensors.
"""
from __future__ import annotations

from collections.abc import Sequence
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np

from .navigation_action_contracts import FORWARD04_NAVIGATION_CONTRACT
from .navigation_augmentation import contained, sha256
from .navigation_dataset_metadata import read_indexed_metadata
from .navigation_extended_collection import independent_success, verify_guard_diagnostics, write_json
from .navigation_extended_data import branch_supervision
from .navigation_extended_qualification import require
from .navigation_extended_release import (
    _base_identity, _read_pinned, load_extended_release, normalizer_coverage,
    normalization_binding, validate_public,
)
from .navigation_history_recovery import validate_prefix
from .navigation_rollin_admission import GuardedTrainingCase


KIND = 'guarded_student_recovery_sequences_v1'
PROTOCOL = 'guarded_student_recovery_protocol_v1'
TEACHER_KEYS = ('source_manifest_sha256', 'qualification_source_manifest_sha256',
                'calibration_sha256', 'teacher_contract')
STUDENT_FILES = {'report.json', 'recording.npz', 'control.json', 'predictions.json'}


def _digest(value):
    return (isinstance(value, str) and len(value) == 64
            and not set(value) - set('0123456789abcdef') and value != '0' * 64)


def _arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def _key(row):
    return row['task'], row['seed'], row['source_tick']


def validate_sources(protocol, base_dataset, extended_root, extended_sha256):
    """Bind every requested anchor to an already qualified training layout."""
    require(protocol.get('schema') == PROTOCOL
            and protocol.get('action_contract') == FORWARD04_NAVIGATION_CONTRACT
            and protocol.get('extended_release_sha256') == extended_sha256
            and _digest(protocol.get('student_checkpoint_sha256'))
            and protocol.get('original_deadline_tick') == 1500
            and protocol.get('training_release_allowed') is False,
            'guarded recovery protocol, real checkpoint or task deadline differs')
    released = load_extended_release(extended_root, base_dataset, extended_sha256)
    plan = json.loads((Path(extended_root) / 'plan.json').read_text())
    require(protocol.get('teacher') == {k: plan[k] for k in TEACHER_KEYS},
            'recovery teacher differs from the qualified collection runtime')
    training = {(e['task'], e['seed']): e for e in released if e['split'] == 'train'}
    sources = protocol.get('sources', [])
    require(bool(sources) and len(sources) == len({(r['task'], r['seed']) for r in sources}),
            'recorded recovery cohort is empty or duplicated')
    cases = {}
    for row in sources:
        key = row['task'], row['seed']
        require(key in training, 'recovery source is not in the qualified training split')
        e = training[key]
        case = GuardedTrainingCase(str(Path(extended_root).resolve()), extended_sha256,
            *key, e['path'], e['metadata_sha256'], e['sha256'], e['layout_sha256'])
        require(row.get('admission') == case.binding(), 'recorded layout admission changed')
        anchors = row.get('anchors', [])
        require(isinstance(anchors, list) and len(anchors) == len(set(anchors))
                and all(type(t) is int and 0 <= t < 1500 and t % 5 == 0 for t in anchors),
                'invalid or duplicated recovery anchor')
        require(set(row.get('files', {})) == STUDENT_FILES
                and all(_digest(h) for h in row['files'].values()), 'student source files are not pinned')
        cases[key] = case
    require(any(row['anchors'] for row in sources), 'no student recovery anchors were selected')
    return cases


def verify_student_replay(protocol, replay):
    sources = {(r['task'], r['seed']): r for r in protocol['sources']}
    rows = replay.get('cases', [])
    require(replay.get('status') == 'complete'
            and replay.get('checkpoint_sha256') == protocol['student_checkpoint_sha256']
            and len(rows) == len(sources) and {(r['task'], r['seed']) for r in rows} == set(sources),
            'student physical replay does not cover the complete selected recording cohort')
    for row in rows:
        source = sources[row['task'], row['seed']]
        n = row.get('control_ticks_verified')
        require(row.get('files') == source['files'] and type(n) is int and 0 < n <= 1500
                and row.get('decisions_verified') == (n + 4) // 5
                and row.get('action_contract') == FORWARD04_NAVIGATION_CONTRACT,
                'recorded student observations/actions lack exact replay')
    require(replay.get('exact_control_ticks') == sum(r['control_ticks_verified'] for r in rows)
            and replay.get('exact_rgb_frames') == sum(r['decisions_verified'] for r in rows),
            'student replay totals differ from its cohort')


def verify_branch_report(report, *, source_tick):
    """A completed process is insufficient: require physical replay and outcome."""
    n = report.get('control_ticks')
    require(type(n) is int and n > 0 and source_tick + n <= 1500
            and report.get('source_tick') == source_tick
            and report.get('original_deadline_tick') == 1500
            and report.get('original_student_unchanged') is True
            and report.get('training_release_allowed') is False,
            'teacher continuation changed the source state or original deadline')
    replay = report.get('replay', {})
    require(replay.get('exact_control_ticks') == n
            and replay.get('exact_rgb_frames') == n // 5 + 1
            and replay.get('complete_branch_evaluator_exact') is True
            and replay.get('original_student_outcome_exact') is True,
            'teacher continuation lacks its complete independent physical replay')
    require(report['result']['tick'] == source_tick + n, 'teacher result reset the task clock')
    return independent_success(report['result'], report['teacher_diagnostics'])


def validate_branch_tensors(data, prefix, source_tick):
    require(type(source_tick) is int and 0 <= source_tick < 1500 and source_tick % 5 == 0,
            'recovery anchor is not an original decision tick')
    require(set(prefix) == {'rgb', 'ticks'}, 'camera prefix contains non-public fields')
    validate_public(dict(data, ticks=data['ticks'] - source_tick), len(data['action']))
    require(data['ticks'][-1] <= 1500, 'recovery supervision extends the original task deadline')
    validate_prefix(prefix['rgb'], prefix['ticks'], source_tick, data['rgb'][0])


def export_guarded_recoveries(source, protocol_path, assessment_path, base_dataset,
                             extended_root, output, *, expected_protocol_sha256,
                             expected_assessment_sha256, extended_sha256, sample_stride=4):
    """Publish only strictly successful, replayed branches; preserve all outcomes."""
    source, output, base = map(Path, (source, output, base_dataset))
    require(not output.exists(), 'guarded recovery releases are immutable')
    require(type(sample_stride) is int and sample_stride > 0, 'positive sample stride required')
    protocol = _read_pinned(protocol_path, expected_protocol_sha256, 'recovery protocol')
    cases = validate_sources(protocol, base, extended_root, extended_sha256)
    replay_ref = protocol['student_replay']
    replay = _read_pinned(replay_ref['path'], replay_ref['sha256'], 'student physical replay')
    verify_student_replay(protocol, replay)
    assessment = _read_pinned(assessment_path, expected_assessment_sha256, 'recovery assessment')
    expected = {(r['task'], r['seed'], t) for r in protocol['sources'] for t in r['anchors']}
    rows = assessment.get('rows', [])
    require(assessment.get('status') == 'complete'
            and assessment.get('protocol_sha256') == expected_protocol_sha256
            and len(rows) == len(expected) and {_key(r) for r in rows} == expected,
            'recovery assessment omitted or duplicated a selected branch')
    _, stats, identity = _base_identity(base)
    sources = {}
    for item in protocol['sources']:
        original = Path(item['path'])
        for name, digest in item['files'].items():
            require(sha256(original / name) == digest, 'student source changed after replay')
        meta = json.loads((original / 'report.json').read_text())
        recording = _arrays(original / 'recording.npz')
        require(meta.get('status') == 'complete' and meta.get('collection_mode') == 'student_record_only'
                and meta['policy']['artifact_sha256'] == protocol['student_checkpoint_sha256']
                and meta['policy']['uses_privileged_state'] is False
                and meta.get('fallback_control_ticks') == 0,
                'recording is not an independent real student rollout')
        case = cases[item['task'], item['seed']]
        case.verify_recorded_start(meta, recording)
        sources[item['task'], item['seed']] = (item, recording, case)
    accepted, rejected, payloads = [], [], []
    for row in rows:
        task, seed, tick = _key(row)
        folder = contained(source, row['path'])
        report = _read_pinned(folder / 'report.json', row['report_sha256'], 'branch report')
        require((report.get('task'), report.get('seed'), report.get('split')) == (task, seed, 'train')
                and report.get('protocol_sha256') == expected_protocol_sha256
                and report.get('student_checkpoint_sha256') == protocol['student_checkpoint_sha256'],
                'teacher branch identity differs from its frozen student protocol')
        success = verify_branch_report(report, source_tick=tick)
        require(row.get('strict_success') is success, 'teacher success was relabeled')
        files = report['files']
        full = report['control_ticks'] // 5
        required = {'trajectory.npz'}
        if full:
            required.update(('navigation.npz', 'camera_prefix.npz'))
        if report['terminal_frame_measured']:
            required.add('terminal_rgb.npy')
        require(set(files) == required, 'branch evidence is incomplete or has unexpected files')
        for name, digest in files.items():
            require(sha256(folder / name) == digest, 'audited teacher evidence changed')
        trace = _arrays(folder / 'trajectory.npz')
        verify_guard_diagnostics(trace['action'], report['teacher_diagnostics'], source_tick=tick)
        require(len(trace['state61']) == report['control_ticks'], 'raw trace length differs from replay')
        terminal = np.load(folder / 'terminal_rgb.npy', allow_pickle=False) if report['terminal_frame_measured'] else None
        original, recording, case = sources[task, seed]
        item = dict(row, student_checkpoint_sha256=protocol['student_checkpoint_sha256'],
            admission=case.binding(), source_recording_sha256=original['files']['recording.npz'],
            source_report_sha256=original['files']['report.json'], decisions=full,
            candidate_for_release=bool(success and full))
        if not full:
            require(report.get('alignment') is None, 'partial control action cannot be a training label')
            rejected.append(item)
            continue
        public, prefix, alignment = branch_supervision(trace, source_tick=tick,
            student_rgb=recording['rgb'][:tick // 5 + 1],
            student_camera_ticks=recording['camera_ticks'][:tick // 5 + 1],
            student_proprio=recording['proprio48'][tick // 5], terminal_rgb=terminal)
        require(report.get('alignment') == alignment, 'branch public alignment differs')
        for filename, expected_arrays in [('navigation.npz', public), ('camera_prefix.npz', prefix)]:
            actual = _arrays(folder / filename)
            require(set(actual) == set(expected_arrays)
                    and all(np.array_equal(actual[k], v) for k, v in expected_arrays.items()),
                    'public tensors differ from the real student/teacher trace')
        validate_branch_tensors(public, prefix, tick)
        if not success:
            rejected.append(item)
            continue
        item['normalization_coverage'] = normalizer_coverage(public, stats)
        item['first_student_rgb_sha256'] = hashlib.sha256(public['rgb'][0].tobytes()).hexdigest()
        accepted.append(item)
        instruction = json.loads((Path(case.root) / case.path / 'episode.json').read_text())['instruction']
        require(report.get('instruction') == instruction, 'teacher recovery changed the instruction')
        payloads.append((item, instruction, folder))
    require(bool(accepted), 'no successful guarded recoveries to release')
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=output.name + '.staging-', dir=output.parent))
    try:
        for src, name in [(base / 'dataset_stats.json', 'dataset_stats.json'),
                          (protocol_path, 'source_protocol.json'), (assessment_path, 'source_assessment.json'),
                          (replay_ref['path'], 'student_replay.json')]:
            shutil.copyfile(src, staging / name)
        audit = dict(status='complete', accepted=accepted, rejected=rejected,
            protocol_sha256=expected_protocol_sha256, assessment_sha256=expected_assessment_sha256,
            student_checkpoint_sha256=protocol['student_checkpoint_sha256'], sample_stride=sample_stride,
            normalization_changed=False, qualification_episodes_released=False,
            formal_acceptance=False)
        write_json(staging / 'source_audit.json', audit)
        entries = []
        for row, instruction, src in payloads:
            relative = f"{row['task']}-seed{row['seed']}/recovery-{row['source_tick']:04d}"
            dest = staging / relative; dest.mkdir(parents=True)
            for name in ('navigation.npz', 'camera_prefix.npz'):
                shutil.copyfile(src / name, dest / name)
            starts = list(range(0, row['decisions'], sample_stride))
            entry = dict(task=row['task'], seed=row['seed'], split='train', path=relative,
                source_tick=row['source_tick'], student_checkpoint_sha256=protocol['student_checkpoint_sha256'],
                frames=row['decisions'] + 1, eligible=True, search_visibility_gated=True,
                sample_starts=starts, actual_student_anchor_starts=[0], teacher_continuation_starts=starts[1:],
                sha256=sha256(dest / 'navigation.npz'), prefix_sha256=sha256(dest / 'camera_prefix.npz'))
            meta = dict(entry, schema=FORWARD04_NAVIGATION_CONTRACT, kind=KIND,
                instruction=instruction, scene={'name': 'room_curriculum_one'}, profile='official_bam',
                first_student_rgb_sha256=row['first_student_rgb_sha256'],
                source_branch_report_sha256=row['report_sha256'], hardware_validated=False)
            write_json(dest / 'episode.json', meta)
            entry['metadata_sha256'] = sha256(dest / 'episode.json')
            entries.append(entry)
        index = dict(kind=KIND, schema=FORWARD04_NAVIGATION_CONTRACT, **identity,
            episodes=entries, training_release_allowed=True, metadata_pinned=True,
            extended_release_sha256=extended_sha256,
            student_checkpoint_sha256=protocol['student_checkpoint_sha256'],
            normalization_binding=normalization_binding(identity['stats_sha256']),
            evidence={name: sha256(staging / name) for name in
                      ('source_protocol.json', 'source_assessment.json', 'source_audit.json', 'student_replay.json')})
        write_json(staging / 'index.json', index)
        load_guarded_recoveries(staging, base, extended_root, extended_sha256,
                               sha256(staging / 'index.json'), protocol['student_checkpoint_sha256'])
        staging.rename(output)
        return index
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def load_guarded_recoveries(root, base_dataset, extended_root, extended_sha256,
                           expected_index_sha256, checkpoint_sha256):
    root = Path(root).resolve()
    index = _read_pinned(root / 'index.json', expected_index_sha256, 'guarded recovery index')
    require(index.get('kind') == KIND and index.get('schema') == FORWARD04_NAVIGATION_CONTRACT
            and index.get('training_release_allowed') is True and index.get('metadata_pinned') is True
            and _digest(checkpoint_sha256) and index.get('student_checkpoint_sha256') == checkpoint_sha256
            and index.get('extended_release_sha256') == extended_sha256,
            'guarded recovery identity, checkpoint or contract differs')
    _, stats, identity = _base_identity(base_dataset)
    require(all(index.get(k) == v for k, v in identity.items())
            and sha256(root / 'dataset_stats.json') == identity['stats_sha256']
            and index.get('normalization_binding') == normalization_binding(identity['stats_sha256']),
            'guarded recovery changed frozen normalization or base data')
    require(set(index['evidence']) == {'source_protocol.json', 'source_assessment.json', 'source_audit.json', 'student_replay.json'},
            'guarded recovery evidence is incomplete')
    protocol, assessment, audit = [_read_pinned(root / name, index['evidence'][name], name)
        for name in ('source_protocol.json', 'source_assessment.json', 'source_audit.json')]
    cases = validate_sources(protocol, base_dataset, extended_root, extended_sha256)
    require(protocol['student_replay']['sha256'] == index['evidence']['student_replay.json'],
            'guarded release changed its student physical replay')
    verify_student_replay(protocol, _read_pinned(root / 'student_replay.json',
                          index['evidence']['student_replay.json'], 'student replay evidence'))
    expected = {(r['task'], r['seed'], t) for r in protocol['sources'] for t in r['anchors']}
    rows = audit['accepted'] + audit['rejected']
    require(audit.get('status') == assessment.get('status') == 'complete'
            and audit.get('protocol_sha256') == assessment.get('protocol_sha256') == index['evidence']['source_protocol.json']
            and audit.get('assessment_sha256') == index['evidence']['source_assessment.json']
            and audit.get('normalization_changed') is False
            and audit.get('qualification_episodes_released') is False
            and protocol['student_checkpoint_sha256'] == audit['student_checkpoint_sha256'] == checkpoint_sha256
            and len(rows) == len(expected) and {_key(r) for r in rows} == expected
            and len(assessment['rows']) == len(expected) and {_key(r) for r in assessment['rows']} == expected,
            'guarded recovery omitted outcomes or changed its assessed protocol')
    assessed = {_key(r): r for r in assessment['rows']}
    for row in rows:
        require(all(row.get(k) == v for k, v in assessed[_key(row)].items()),
                'released recovery differs from its independent assessment')
    accepted = {_key(r): r for r in audit['accepted']}
    entries = index['episodes']
    require(bool(entries) and len(entries) == len(accepted) and {_key(e) for e in entries} == set(accepted)
            and all(r['candidate_for_release'] is True and r['strict_success'] is True for r in audit['accepted'])
            and all(r['candidate_for_release'] is False and not (r['strict_success'] and r['decisions'])
                    for r in audit['rejected']),
            'guarded recovery success denominator or released branches changed')
    for entry in entries:
        row = accepted[_key(entry)]
        case = cases[entry['task'], entry['seed']]
        require(row['admission'] == case.binding() and entry['split'] == 'train'
                and entry['eligible'] is True and entry['search_visibility_gated'] is True
                and entry['student_checkpoint_sha256'] == checkpoint_sha256,
                'guarded recovery became a held-out or unrelated student episode')
        folder = contained(root, entry['path'])
        meta = read_indexed_metadata(root, entry, require_digest=True)
        require(all(meta.get(k) == v for k, v in entry.items() if k != 'metadata_sha256')
                and meta.get('kind') == KIND and meta.get('schema') == FORWARD04_NAVIGATION_CONTRACT
                and meta.get('source_branch_report_sha256') == row['report_sha256']
                and meta['instruction'] == json.loads((Path(case.root) / case.path / 'episode.json').read_text())['instruction'],
                'guarded recovery metadata or instruction changed')
        require(sha256(folder / 'navigation.npz') == entry['sha256']
                and sha256(folder / 'camera_prefix.npz') == entry['prefix_sha256'], 'guarded recovery tensors changed')
        data, prefix = _arrays(folder / 'navigation.npz'), _arrays(folder / 'camera_prefix.npz')
        validate_branch_tensors(data, prefix, entry['source_tick'])
        n = len(data['action'])
        starts = list(range(0, n, audit['sample_stride']))
        require(n == row['decisions'] and entry['frames'] == n + 1 and entry['sample_starts'] == starts
                and entry['actual_student_anchor_starts'] == [0] and entry['teacher_continuation_starts'] == starts[1:]
                and hashlib.sha256(data['rgb'][0].tobytes()).hexdigest() == row['first_student_rgb_sha256'] == meta['first_student_rgb_sha256']
                and normalizer_coverage(data, stats) == row['normalization_coverage'],
                'guarded recovery sampling, original RGB or normalization changed')
    return entries


def guarded_recovery_config(roots, hashes, checkpoint, split, *, checkpoints=None):
    """Pin each DAgger source, preserving the legacy single-student contract."""
    require(isinstance(roots, Sequence) and not isinstance(roots, (str, bytes))
            and isinstance(hashes, Sequence) and not isinstance(hashes, (str, bytes))
            and len(roots) == len(hashes), 'guarded recoveries need matching root/hash lists')
    if checkpoints is not None:
        require(checkpoint is None and isinstance(checkpoints, Sequence)
                and not isinstance(checkpoints, (str, bytes)) and len(checkpoints) == len(roots),
                'guarded recoveries need one unambiguous checkpoint pin per source')
    if not roots:
        require(checkpoint is None and not checkpoints, 'unused guarded recovery checkpoint')
        return None
    pins = [checkpoint] * len(roots) if checkpoints is None else list(checkpoints)
    require(split == 'train' and all(_digest(p) for p in pins) and all(_digest(h) for h in hashes),
            'guarded recoveries require training-only sources and a real checkpoint')
    resolved = [str(Path(p).resolve()) for p in roots]
    require(len(set(resolved)) == len(roots) and len(set(hashes)) == len(hashes),
            'duplicate guarded recovery source')
    binding = (dict(student_checkpoint_sha256=checkpoint) if checkpoints is None else
               dict(student_checkpoint_sha256s=pins))
    return dict(roots=resolved, index_sha256=list(hashes), **binding,
                sampling='after_complete_retained_extended_and_clear_recovery_samples')
