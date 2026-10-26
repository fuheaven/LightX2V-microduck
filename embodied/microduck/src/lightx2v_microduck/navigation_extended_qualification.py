"""Verify a guarded teacher without turning its qualification cases into data.

This gate consumes independently replayed qualification evidence. Its returned
certificate only permits a later, separately seeded demonstration collection.
It never admits any of these 300 qualification episodes to model training.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .navigation_augmentation import contained, sha256


RAW_CONTRACT = 'raw_official_guarded_occlusion_teacher_probe_v1'
TASKS = {'find_red_ball', 'find_blue_ball', 'find_yellow_ball'}


def require(value, message):
    if not value:
        raise ValueError(message)


def physical_layout(world):
    """Ignore target instruction/seed IDs; compare actual geometry at 1 mm."""
    rounded = lambda values: [round(float(value), 3) for value in values]
    return dict(balls={name: rounded(position) for name, position in sorted(world['balls'].items())},
        obstacles=sorted([dict(xy=rounded(o['xy']), half_size=rounded(o['half_size']),
                               height=round(float(o['height']), 3)) for o in world['obstacles']],
                         key=lambda o: json.dumps(o, sort_keys=True)),
        room_bounds=rounded(world['room_bounds']))


def layout_sha256(world):
    encoded = json.dumps(physical_layout(world), sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def verify_qualification_summary(assessment, protocol, layouts):
    """Check the full denominator and actual layout identities before file IO."""
    require(protocol.get('purpose') == 'fresh_teacher_qualification'
            and protocol.get('profile') == 'official_bam'
            and protocol.get('scene') == 'room_curriculum_one'
            and protocol.get('action_contract') == RAW_CONTRACT,
            'teacher qualification purpose, dynamics, scene or action semantics changed')
    require(protocol.get('max_ticks') == 1500 and protocol.get('strict_stop_ticks') == 100
            and protocol.get('strict_stop_surface_m') == [.25, .40]
            and protocol.get('minimum_eligible_success_rate_per_task') == .95,
            'the teacher qualification thresholds must remain unchanged')
    seeds = protocol.get('seeds', [])
    require(len(seeds) == len(set(seeds)) == 100 and all(type(s) is int for s in seeds)
            and len(protocol.get('tasks', [])) == 3 and set(protocol['tasks']) == TASKS,
            'qualification requires 100 distinct seeds and all three colors')
    require(not set(seeds).intersection(range(10000, 10200)), 'reserved final layouts entered qualification')
    for start, stop in protocol.get('excluded_seed_ranges', []):
        require(not set(seeds).intersection(range(start, stop)), 'excluded development/training layouts entered qualification')
    expected = {(s, t) for s in seeds for t in TASKS}
    require(len(protocol.get('cases', [])) == 300
            and {(r['seed'], r['task']) for r in protocol['cases']} == expected,
            'qualification protocol omitted or duplicated a case')
    require(assessment.get('status') == 'complete' and assessment.get('teacher_qualified') is True
            and assessment.get('qualification_passed') is True
            and assessment.get('action_contract') == RAW_CONTRACT
            and assessment.get('training_release_allowed') is False,
            'a complete qualified teacher is required; qualification is not a training release')
    rows = assessment.get('rows', [])
    require(len(rows) == 300 and {(r['seed'], r['task']) for r in rows} == expected,
            'qualification must retain all 300 cases, including failures')
    for row in rows:
        ticks = row.get('exact_control_ticks')
        require(row.get('purpose') == 'fresh_teacher_qualification'
                and row.get('final_evaluator_exact') is True and type(row.get('strict_success')) is bool
                and type(ticks) is int and 0 < ticks <= 1500
                and row.get('exact_rgb_frames') == (ticks + 4) // 5
                and row.get('training_release_allowed') is False and row.get('teacher_qualified') is False,
                'qualification row lacks a complete independent replay')
        if row['strict_success']:
            require(not (row['collision'] or row['fallen'] or row['recovery_events'] or row['fault']),
                    'intervention, collision or fall cannot be an independent teacher success')
    colors = {}
    for task in sorted(TASKS):
        successes = sum(r['strict_success'] for r in rows if r['task'] == task)
        require(successes >= 95, 'a color failed the unchanged 95/100 teacher gate')
        colors[task] = dict(successes=successes, total=100)
    require(assessment.get('per_color') == colors and assessment.get('total') == 300
            and assessment.get('minimum_successes_per_color') == 95
            and assessment.get('strict_successes') == sum(c['successes'] for c in colors.values()),
            'saved qualification totals differ from individual cases')
    require(layouts.get('status') == 'complete' and layouts.get('geometry_resolution_m') == .001
            and layouts.get('teacher_queried') is False and layouts.get('model_queried') is False,
            'an independent actual-geometry initialization audit is required')
    measured = layouts.get('cases', [])
    require(len(measured) == 300 and {(r['seed'], r['task']) for r in measured} == expected,
            'initialization audit does not cover the qualification cases')
    by_seed = {}
    for row in measured:
        encoded = json.dumps(row['layout'], sort_keys=True, separators=(',', ':'), allow_nan=False)
        require(hashlib.sha256(encoded.encode()).hexdigest() == row['layout_sha256'],
                'measured geometry hash differs from actual geometry')
        by_seed.setdefault(row['seed'], set()).add(row['layout_sha256'])
    require(all(len(values) == 1 for values in by_seed.values()), 'target instruction changes scene geometry')
    geometry = {next(iter(values)) for values in by_seed.values()}
    require(len(geometry) == 100 and layouts.get('distinct_physical_layouts') == 100,
            '100 seed IDs do not establish 100 different physical layouts')
    return dict(per_color=colors, qualification_seeds=sorted(seeds),
                qualification_geometry_sha256=sorted(geometry))


def load_extended_teacher_qualification(root, *, evidence):
    """Verify pinned source, every independent audit and all original traces.

    ``evidence`` contains assessment/protocol/layouts path+SHA256 pairs relative
    to ``root``. Original traces stay under the protocol's pinned output folder.
    The result is never a list of training episodes.
    """
    root = Path(root).resolve()
    require(set(evidence) == {'assessment', 'protocol', 'layouts'}, 'all three pinned qualification reports are required')
    reports, paths = {}, {}
    for key, ref in evidence.items():
        path = contained(root, ref['path'])
        require(isinstance(ref.get('sha256'), str) and len(ref['sha256']) == 64
                and sha256(path) == ref['sha256'], key + ' evidence changed')
        paths[key] = path
        reports[key] = json.loads(path.read_text())
    a, p, layouts = (reports[k] for k in ('assessment', 'protocol', 'layouts'))
    summary = verify_qualification_summary(a, p, layouts)
    require(a['protocol_sha256'] == layouts['protocol_sha256'] == evidence['protocol']['sha256'],
            'qualification or layout audit belongs to another protocol')
    require(a['source_manifest_sha256'] == p['source_manifest_sha256'] == layouts['source_manifest_sha256'],
            'qualification source manifests differ')
    require(p.get('training_release_allowed') is False and p.get('automatic_training') is False,
            'qualification protocol must not schedule training')
    # Freeze the same controller and environment before new demonstrations.
    for name, digest in p['files'].items():
        require(sha256(contained(root, name)) == digest, 'teacher dependency changed: ' + name)
    source = contained(root, p['source'])
    for name, digest in p['source_files'].items():
        require(sha256(contained(source, name)) == digest, 'teacher runtime changed: ' + name)
    layout_by_case = {(r['seed'], r['task']): r for r in layouts['cases']}
    out = contained(root, p['output'])
    total_ticks = total_frames = 0
    for row in a['rows']:
        folder = out / ('%s-seed%d' % (row['task'], row['seed']))
        for name, field in [('episode.json', 'metadata_sha256'), ('trajectory.npz', 'trajectory_sha256'), ('audit.json', 'audit_sha256')]:
            require(sha256(contained(folder, name)) == row[field], 'qualification case bytes changed')
        audit = json.loads((folder / 'audit.json').read_text())
        require(audit == {k: v for k, v in row.items() if k != 'audit_sha256'}, 'case audit differs from the qualification row')
        meta = json.loads((folder / 'episode.json').read_text())
        require(meta['status'] == 'complete' and (meta['seed'], meta['task']) == (row['seed'], row['task'])
                and meta['protocol_sha256'] == evidence['protocol']['sha256']
                and meta['purpose'] == p['purpose'] and meta['action_contract'] == RAW_CONTRACT
                and meta['profile'] == p['profile'] and meta['scene'] == p['scene']
                and meta['qualification_only'] is True and meta['eligible_training_demonstration'] is False
                and meta['training_release_allowed'] is False,
                'qualification data identity or non-training purpose changed')
        require(meta['files'] == {'trajectory.npz': row['trajectory_sha256']}, 'qualification trace identity changed')
        world = meta['result']
        require(world['tick'] == row['exact_control_ticks']
                and meta['experimental_strict_success'] == row['strict_success'], 'case outcome changed')
        independent = bool(world['success'] and world['strict_success'] and not world['collision']
            and not world['ever_fallen'] and not world['fault'] and world['recovery_events'] == 0
            and not any(d.get('planner_phase', d.get('phase')) in ('recover', 'no_path') for d in meta['diagnostics']))
        require(independent == row['strict_success'], 'qualification used an intervention as success')
        if independent:
            require(world['goal_hold_ticks'] >= 100, 'teacher did not maintain the unchanged stop interval')
        actual_layout = layout_sha256(meta['initial'])
        initial_check = layout_by_case[row['seed'], row['task']]
        require(actual_layout == initial_check['layout_sha256'],
                'teacher actually ran a different physical layout')
        require(meta['model_signature'] == meta['initial']['model_signature']
                == world['model_signature'] == initial_check['model_signature'],
                'qualification dynamics differ from the pinned initialization audit')
        total_ticks += row['exact_control_ticks']; total_frames += row['exact_rgb_frames']
    require(a['exact_control_ticks'] == total_ticks and a['exact_rgb_frames'] == total_frames,
            'qualification replay totals changed')
    return dict(schema='guarded_teacher_qualification_certificate_v1', **summary,
        evidence=evidence, source_manifest_sha256=p['source_manifest_sha256'],
        teacher_contract=RAW_CONTRACT, eligible_for_separate_collection=True,
        qualification_episodes_may_train=False, training_release_allowed=False,
        exact_control_ticks=total_ticks, exact_rgb_frames=total_frames)
