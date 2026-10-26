"""Admission for separately collected guarded navigation demonstrations.

The qualification certificate and the collection are distinct artifacts. All
collection outcomes remain in the audit denominator; only independently replayed
successes expose public tensors to a model. Statistics are copied, never fitted.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile

import numpy as np

from .navigation_action_contracts import FORWARD04_NAVIGATION_CONTRACT, LEGACY_NAVIGATION_CONTRACT, action_contract
from .navigation_augmentation import contained, sha256
from .navigation_extended_collection import (
    COLLECTION, SCHEMA as COLLECTION_SCHEMA, independent_success, verify_collection_plan,
    verify_guard_diagnostics, verify_layout, write_json,
)
from .navigation_extended_data import supervised_transitions
from .navigation_extended_qualification import physical_layout, layout_sha256, require


KIND = 'separate_guarded_teacher_episodes_v1'
PUBLIC_KEYS = {'rgb', 'proprio', 'action', 'ticks'}
PUBLIC_META_KEYS = {'schema', 'task', 'seed', 'split', 'instruction', 'scene', 'profile',
    'student_state', 'student_camera', 'control_hz', 'camera_hz', 'warmup_ticks',
    'decision_count', 'eligible_training_demonstration', 'search_visibility_gated',
    'teacher_qualification_sha256', 'source_metadata_sha256', 'source_audit_sha256',
    'layout_sha256', 'files', 'hardware_validated'}


def _read_pinned(path, digest, label):
    require(isinstance(digest, str) and len(digest) == 64 and sha256(path) == digest,
            label + ' is not explicitly pinned or changed')
    return json.loads(Path(path).read_text())


def _base_identity(base):
    base = Path(base)
    index, stats, q = [json.loads((base / name).read_text())
        for name in ('index.json', 'dataset_stats.json', 'qualification_report.json')]
    ih, sh, qh = [sha256(base / name) for name in ('index.json', 'dataset_stats.json', 'qualification_report.json')]
    require(index.get('schema') == LEGACY_NAVIGATION_CONTRACT
        and stats.get('schema') == LEGACY_NAVIGATION_CONTRACT and stats.get('normalization_split') == 'train'
        and q.get('teacher_qualified') is True and q.get('scene') == 'room_clear'
        and q.get('index_sha256') == ih and q.get('stats_sha256') == sh,
        'the immutable clear-room base or its train-only statistics changed')
    return index, stats, dict(base_index_sha256=ih, base_qualification_sha256=qh, stats_sha256=sh)


def normalization_binding(stats_sha256):
    return dict(execution_contract=FORWARD04_NAVIGATION_CONTRACT,
                coordinate_contract=LEGACY_NAVIGATION_CONTRACT, stats_sha256=stats_sha256)


def validate_public(data, decisions):
    require(set(data) == PUBLIC_KEYS and decisions > 0, 'only aligned public tensors may enter a release')
    require(data['rgb'].dtype == np.uint8 and data['rgb'].shape == (decisions + 1, 256, 256, 3)
        and data['proprio'].shape == (decisions + 1, 48) and data['action'].shape == (decisions, 5)
        and np.issubdtype(data['ticks'].dtype, np.integer)
        and np.array_equal(data['ticks'], np.arange(decisions + 1) * 5), 'public tensor shape or clock changed')
    for key in ('proprio', 'action'):
        require(data[key].dtype == np.float32 and np.isfinite(data[key]).all(), 'public physical values must remain finite FP32')
    action_contract(FORWARD04_NAVIGATION_CONTRACT).validate(data['action'])


def normalizer_coverage(data, stats):
    """Compare actual dataset normalization with the original physical values."""
    from .wam_data import FastWAMNavigationDataset
    holder = object.__new__(FastWAMNavigationDataset); holder.stats = stats
    fields = {}
    for key, array in (('state', data['proprio']), ('action', data['action'])):
        ref = stats[key]['default']
        low, high = np.asarray(ref['global_min'], np.float32), np.asarray(ref['global_max'], np.float32)
        require(low.shape == high.shape == (array.shape[1],) and np.isfinite(low).all()
                and np.isfinite(high).all() and (high >= low).all(), 'invalid frozen normalizer dimensions')
        width = high - low; constant = width < 1e-4
        scale = 2 / np.where(constant, 2, width)
        offset = np.where(constant, -low, -1 - scale * low)
        unbounded = array * scale + offset
        normalized = holder._normalize(array, key)
        recovered = (normalized - offset) / scale
        fields[key] = dict(values=int(array.size), clipped_values=int((np.abs(unbounded) > 5).sum()),
            maximum_normalized_magnitude=float(np.abs(unbounded).max()),
            roundtrip_max_abs=float(np.abs(recovered - array).max()),
            original_minmax_exceedances=int(((array < low) | (array > high)).sum()))
        require(fields[key]['clipped_values'] == 0 and fields[key]['roundtrip_max_abs'] < 1e-5,
                'frozen normalization clips or changes the released physical labels/state')
    return fields


def verify_collection_summary(rows, plan, certificate):
    verify_collection_plan(plan, certificate)
    require(plan['purpose'] == COLLECTION, 'diagnostic cases cannot become training demonstrations')
    expected = {(c['seed'], c['task']): c for c in plan['cases']}
    require(len(rows) == 300 and {(r['seed'], r['task']) for r in rows} == set(expected),
            'collection audit must include all 300 outcomes, including failures')
    by_seed = {}
    for row in rows:
        case = expected[row['seed'], row['task']]
        require(row['split'] == case['split'] and type(row['strict_success']) is bool
                and row['final_evaluator_exact'] is True and row['purpose'] == COLLECTION,
                'collected outcome differs from its fixed split or independent replay')
        ticks = row['exact_control_ticks']; full = ticks // 5
        require(type(ticks) is int and 0 < ticks <= 1500
                and row['complete_decisions'] == full and row['tail_control_ticks'] == ticks % 5
                and row['exact_rgb_frames'] == full + 1,
                'independent replay omitted terminal RGB or partial physical tail')
        require(row['candidate_for_release'] == bool(row['strict_success'] and full)
                and row['training_release_allowed'] is False, 'invalid data release eligibility')
        if row['strict_success']:
            require(row['public_tensors_exact'] is True and full >= 20, 'success lacks a full stop interval or public alignment')
        world = row['initial_layout']
        require(physical_layout(world) == world and verify_layout(plan, certificate, world) == row['layout_sha256'],
                'actual collection geometry changed')
        by_seed.setdefault(row['seed'], set()).add(row['layout_sha256'])
    require(len(by_seed) == 100 and all(len(v) == 1 for v in by_seed.values()),
            'color instructions must share the same physical layout and split')
    require(len({next(iter(v)) for v in by_seed.values()}) == 100,
            'distinct collection seeds duplicate physical geometry across layouts or splits')
    return [r for r in rows if r['candidate_for_release']]


def export_extended_release(collection_root, plan_path, certificate_path, base_dataset, output, *,
                            expected_plan_sha256, expected_certificate_sha256):
    """Publish an immutable public release only after every collection replay."""
    source, base, output = map(Path, (collection_root, base_dataset, output))
    require(not output.exists(), 'extended dataset releases are immutable')
    plan = _read_pinned(plan_path, expected_plan_sha256, 'collection plan')
    require(plan.get('purpose') == COLLECTION and plan.get('ready_for_execution') is True,
            'diagnostic or inactive collection cannot become a training release')
    certificate = _read_pinned(certificate_path, expected_certificate_sha256, 'teacher certificate')
    verify_collection_plan(plan, certificate)
    require(plan.get('purpose') == COLLECTION and plan.get('ready_for_execution') is True
            and plan.get('qualification_certificate', {}).get('sha256') == expected_certificate_sha256,
            'collection template is not an activated certified execution plan')
    base_index, stats, base_identity = _base_identity(base)
    require(plan.get('normalization_binding') == normalization_binding(base_identity['stats_sha256']),
            'collection normalization binding differs from the frozen base')
    require(not {c['seed'] for c in plan['cases']}.intersection(e['seed'] for e in base_index['episodes']),
            'new collection overlaps immutable base layouts')
    rows, coverage, candidates = [], [], []
    for case in plan['cases']:
        name = '%s-seed%d' % (case['task'], case['seed'])
        folder = contained(source, name)
        audit = json.loads((folder / 'audit.json').read_text())
        require(audit['plan_sha256'] == expected_plan_sha256
            and all(audit[k] == case[k] for k in ('task', 'seed', 'split')), 'audit belongs to another collection')
        meta = _read_pinned(folder / 'episode.json', audit['metadata_sha256'], 'collection metadata')
        require(meta['status'] == 'complete' and meta['schema'] == COLLECTION_SCHEMA
            and meta['purpose'] == COLLECTION and meta['plan_sha256'] == expected_plan_sha256
            and meta['teacher_qualified'] is True and meta['qualification_only'] is False
            and meta['eligible_training_demonstration'] is False and meta['training_release_allowed'] is False
            and all(meta[k] == case[k] for k in ('task', 'seed', 'split'))
            and meta['files'] == audit['files'], 'unqualified or repurposed collection metadata')
        for filename, digest in meta['files'].items():
            require(Path(filename).name == filename and sha256(folder / filename) == digest, 'original collection file changed')
        require(meta['profile'] == plan['profile'] and meta['scene'] == plan['scene']
            and meta['teacher_contract'] == plan['teacher_contract'] and meta['student_contract'] == plan['student_contract'],
            'collection observation or action semantics changed')
        success = independent_success(meta['result'], meta['diagnostics'])
        require(success == audit['strict_success'] == meta['independent_success']
            and meta['result']['tick'] == audit['exact_control_ticks'] == meta['control_ticks'], 'collection success was relabeled')
        with np.load(folder / 'trajectory.npz', allow_pickle=False) as z:
            trace = {k: z[k].copy() for k in z.files}
        verify_guard_diagnostics(trace['action'], meta['diagnostics'])
        terminal = np.load(folder / 'terminal-rgb.npy', allow_pickle=False) if meta['terminal_frame_measured'] else None
        full = audit['complete_decisions']
        if full:
            expected, alignment = supervised_transitions(trace, terminal_rgb=terminal)
            require(alignment == meta['alignment'] and full == len(expected['action']), 'recorded alignment differs')
            with np.load(folder / 'navigation.npz', allow_pickle=False) as z:
                data = {k: z[k].copy() for k in z.files}
            validate_public(data, full)
            for key in PUBLIC_KEYS:
                require(np.array_equal(data[key], expected[key]), 'public tensors differ from the actually executed physical trace')
        else:
            require(not (folder / 'navigation.npz').exists(), 'partial action became a full training example')
        row = dict(audit, audit_sha256=sha256(folder / 'audit.json'),
            initial_layout=physical_layout(meta['initial']), source_path=str(folder.resolve()),
            instruction=meta['instruction'])
        require(layout_sha256(meta['initial']) == meta['layout_sha256'] == row['layout_sha256'], 'initial geometry differs')
        rows.append(row)
        if audit['candidate_for_release']:
            fields = normalizer_coverage(data, stats)
            coverage.append(dict(seed=case['seed'], task=case['task'], split=case['split'],
                navigation_sha256=meta['files']['navigation.npz'], fields=fields))
            candidates.append((case, folder, meta, row))
    accepted = verify_collection_summary(rows, plan, certificate)
    require(len(candidates) == len(accepted) and all(any(r['split'] == s for r in accepted)
            for s in ('train', 'validation', 'test')), 'release lacks eligible independent layout splits')
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=output.name + '.staging-', dir=output.parent))
    try:
        shutil.copyfile(base / 'dataset_stats.json', staging / 'dataset_stats.json')
        evidence = {}
        audit_report = dict(status='complete', rows=rows, total=len(rows),
            successes=sum(r['strict_success'] for r in rows), distinct_layouts=100,
            plan_sha256=expected_plan_sha256, certificate_sha256=expected_certificate_sha256,
            training_release_allowed=False, qualification_episodes_released=False)
        normal = dict(status='complete', stats_sha256=base_identity['stats_sha256'], normalization_changed=False,
            fitted_statistics=False, rows=coverage, no_clipping=True,
            scope='All released successes checked; failed collection outcomes are retained in the audit only.')
        for key, obj in [('plan', plan), ('certificate', certificate), ('collection_audit', audit_report), ('normalization', normal)]:
            filename = key + '.json'
            if key == 'plan':
                shutil.copyfile(plan_path, staging / filename)
            elif key == 'certificate':
                shutil.copyfile(certificate_path, staging / filename)
            else:
                write_json(staging / filename, obj)
            evidence[key] = dict(path=filename, sha256=sha256(staging / filename))
        entries = []
        for case, folder, meta, row in candidates:
            name = folder.name; dest = staging / name; dest.mkdir()
            shutil.copyfile(folder / 'navigation.npz', dest / 'navigation.npz')
            public = dict(schema=FORWARD04_NAVIGATION_CONTRACT, **case, instruction=meta['instruction'],
                scene={'name': plan['scene']}, profile=plan['profile'], student_state='obs61[:48]; command13 excluded',
                student_camera='head_rgb_256', control_hz=50, camera_hz=10, warmup_ticks=50,
                decision_count=row['complete_decisions'], eligible_training_demonstration=True,
                search_visibility_gated=True, teacher_qualification_sha256=expected_certificate_sha256,
                source_metadata_sha256=row['metadata_sha256'], source_audit_sha256=row['audit_sha256'],
                layout_sha256=row['layout_sha256'], files={'navigation.npz': row['files']['navigation.npz']},
                hardware_validated=False)
            write_json(dest / 'episode.json', public)
            entries.append(dict(case, path=name, eligible=True, search_visibility_gated=True,
                frames=row['complete_decisions'] + 1, sha256=row['files']['navigation.npz'],
                metadata_sha256=sha256(dest / 'episode.json'), layout_sha256=row['layout_sha256']))
        index = dict(kind=KIND, schema=FORWARD04_NAVIGATION_CONTRACT, **base_identity,
            normalization_binding=normalization_binding(base_identity['stats_sha256']),
            evidence=evidence, episodes=entries, metadata_pinned=True,
            collected_cases=300, qualification_episodes_released=False, training_release_allowed=True)
        write_json(staging / 'index.json', index)
        load_extended_release(staging, base, sha256(staging / 'index.json'))
        staging.rename(output)
        return index
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def load_extended_release(root, base_dataset, expected_index_sha256):
    """Verify a pinned public release before any dataset exposes its tensors."""
    root, base = Path(root).resolve(), Path(base_dataset).resolve()
    index = _read_pinned(root / 'index.json', expected_index_sha256, 'extended release index')
    require(index.get('kind') == KIND and index.get('schema') == FORWARD04_NAVIGATION_CONTRACT
        and index.get('training_release_allowed') is True and index.get('qualification_episodes_released') is False
        and index.get('metadata_pinned') is True, 'unsupported or incomplete extended release')
    base_index, stats, identity = _base_identity(base)
    require(all(index.get(k) == v for k, v in identity.items())
        and sha256(root / 'dataset_stats.json') == identity['stats_sha256']
        and index.get('normalization_binding') == normalization_binding(identity['stats_sha256']),
        'extended release changed the base or frozen normalization')
    evidence = {key: _read_pinned(contained(root, ref['path']), ref['sha256'], key + ' evidence')
                for key, ref in index['evidence'].items()}
    require(set(evidence) == {'plan', 'certificate', 'collection_audit', 'normalization'}, 'extended release evidence is incomplete')
    plan, cert, audit, normal = [evidence[k] for k in ('plan', 'certificate', 'collection_audit', 'normalization')]
    require(plan.get('ready_for_execution') is True and plan.get('normalization_binding') == index['normalization_binding']
        and plan.get('qualification_certificate', {}).get('sha256') == index['evidence']['certificate']['sha256'],
        'collection was not activated for its certificate and coordinates')
    accepted = verify_collection_summary(audit['rows'], plan, cert)
    require(audit.get('status') == 'complete' and audit.get('total') == index['collected_cases'] == 300
        and audit.get('successes') == len(accepted) and audit.get('distinct_layouts') == 100
        and audit.get('plan_sha256') == index['evidence']['plan']['sha256']
        and audit.get('certificate_sha256') == index['evidence']['certificate']['sha256']
        and audit.get('qualification_episodes_released') is False,
        'collection audit does not cover the activated plan')
    require(not {c['seed'] for c in plan['cases']}.intersection(e['seed'] for e in base_index['episodes']),
        'extended collection overlaps base layouts')
    entries = index['episodes']; expected = {(r['seed'], r['task']): r for r in accepted}
    require(len(entries) == len(expected) and {(e['seed'], e['task']) for e in entries} == set(expected),
        'release must preserve exactly the audited successful cases')
    normal_rows = {(r['seed'], r['task']): r for r in normal['rows']}
    require(normal.get('status') == 'complete' and normal.get('stats_sha256') == identity['stats_sha256']
        and normal.get('normalization_changed') is False and normal.get('fitted_statistics') is False
        and normal.get('no_clipping') is True and len(normal['rows']) == len(expected)
        and set(normal_rows) == set(expected), 'frozen-normalizer coverage is incomplete')
    from .navigation_dataset_metadata import read_indexed_metadata
    for entry in entries:
        row = expected[entry['seed'], entry['task']]
        meta = read_indexed_metadata(root, entry, require_digest=True)
        require(set(meta) == PUBLIC_META_KEYS and meta['schema'] == FORWARD04_NAVIGATION_CONTRACT
            and meta['instruction'] == row['instruction'] and entry['split'] == row['split']
            and meta['source_metadata_sha256'] == row['metadata_sha256']
            and meta['source_audit_sha256'] == row['audit_sha256']
            and meta['teacher_qualification_sha256'] == index['evidence']['certificate']['sha256']
            and meta['layout_sha256'] == entry['layout_sha256'] == row['layout_sha256']
            and meta['scene'] == {'name': 'room_curriculum_one'} and meta['profile'] == 'official_bam'
            and meta['student_state'] == 'obs61[:48]; command13 excluded' and meta['student_camera'] == 'head_rgb_256'
            and meta['control_hz'] == 50 and meta['camera_hz'] == 10 and meta['warmup_ticks'] == 50
            and meta['eligible_training_demonstration'] is True and meta['search_visibility_gated'] is True
            and entry['eligible'] is True and entry['search_visibility_gated'] is True,
            'public metadata changed its instruction, provenance or observation contract')
        path = contained(root, entry['path']) / 'navigation.npz'
        require(sha256(path) == entry['sha256'] == row['files']['navigation.npz'] == meta['files']['navigation.npz'],
                'extended tensors changed')
        with np.load(path, allow_pickle=False) as z:
            data = {k: z[k].copy() for k in z.files}
        decisions = row['complete_decisions']; validate_public(data, decisions)
        require(entry['frames'] == decisions + 1 and meta['decision_count'] == decisions, 'public episode length changed')
        coverage = normal_rows[entry['seed'], entry['task']]
        require(coverage['navigation_sha256'] == entry['sha256'] and coverage['split'] == entry['split']
            and coverage['fields'] == normalizer_coverage(data, stats), 'normalization evidence differs from public tensors')
    return entries
