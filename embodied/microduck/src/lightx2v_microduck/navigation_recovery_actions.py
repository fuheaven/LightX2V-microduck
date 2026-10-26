"""Inspect the corrective commands at authenticated student recovery states.

This measures teacher labels, not robot displacement, learned capability or
dataset admission. Existing publication, physics replay and dataset audits
remain required. In particular a walk skill with zero velocity is not counted
as a request to move, and the stand flag suppresses all three body velocities.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def pinned_json(path, expected):
    raw = Path(path).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected, 'changed input: ' + str(path))
    return json.loads(raw)


def command_evidence(actions, ticks, source_tick, student_control):
    """Compare commands at the branch point without inferring physical motion."""
    from lightx2v_microduck.navigation_action_contracts import action_contract
    actions, ticks = np.asarray(actions), np.asarray(ticks)
    require(actions.dtype == np.float32 and actions.ndim == 2 and actions.shape[1] == 5
            and len(actions) > 0 and np.isfinite(actions).all(), 'invalid physical action labels')
    action_contract('microduck_navigation_forward04_v1').validate(actions)
    require(type(source_tick) is int and source_tick >= 0 and source_tick % 5 == 0
            and np.array_equal(ticks, source_tick + np.arange(len(actions) + 1) * 5)
            and ticks[-1] <= 1500, 'teacher labels changed the original decision clock')
    require(student_control['tick'] == source_tick
            and student_control['execution_source'] == 'navigation_model'
            and student_control['skill'] in ('walk', 'stand'), 'wrong student control state')
    command = np.asarray(student_control['command'], np.float32)
    require(command.shape == (13,) and np.isfinite(command).all(), 'invalid student command')
    stand = actions[:, 4] >= .5
    effective = actions[:, :3].copy()
    effective[stand] = 0
    # This tolerance describes labels; it is not a new controller deadband.
    motion = np.any(np.abs(effective) > 1e-6, axis=1)
    first = np.flatnonzero(motion)
    return dict(student_skill=student_control['skill'], student_command3=command[:3].tolist(),
        teacher_first_action=actions[0].tolist(), teacher_first_skill='stand' if stand[0] else 'walk',
        teacher_first_effective_command3=effective[0].tolist(),
        teacher_first_requests_motion=bool(motion[0]),
        first_motion_delay_ticks=int(first[0]) * 5 if len(first) else None,
        first_motion_delay_s=int(first[0]) * .1 if len(first) else None,
        total_decisions=len(actions),
        prefixes={str(n): dict(decisions=min(n, len(actions)),
            stand_decisions=int(stand[:n].sum()), motion_decisions=int(motion[:n].sum()))
            for n in (1, 5, 10, 32)})


def summarize(rows):
    def group(selected):
        stopped = [r for r in selected if r['student_skill'] == 'stand']
        return dict(branches=len(selected), student_stand_branches=len(stopped),
            student_stand_to_immediate_teacher_motion=sum(r['teacher_first_requests_motion'] for r in stopped),
            first_teacher_stand=sum(r['teacher_first_skill'] == 'stand' for r in selected),
            first_teacher_motion=sum(r['teacher_first_requests_motion'] for r in selected),
            first_teacher_walk_without_body_motion=sum(r['teacher_first_skill'] == 'walk'
                and not r['teacher_first_requests_motion'] for r in selected),
            teacher_motion_within_one_second=sum(r['first_motion_delay_ticks'] is not None
                and r['first_motion_delay_ticks'] < 50 for r in selected))
    return dict(overall=group(rows),
        by_reason={reason: group([r for r in rows if reason in r['reasons']])
                   for reason in sorted({why for r in rows for why in r['reasons']})},
        by_task={task: group([r for r in rows if r['task'] == task])
                 for task in sorted({r['task'] for r in rows})})


def inspect_release(root, expected_index):
    root = Path(root).resolve()
    index = pinned_json(root / 'index.json', expected_index)
    checkpoint = index['student_checkpoint_sha256']
    require(index['schema'] == 'microduck_navigation_forward04_v1'
            and index['kind'] == 'guarded_student_recovery_sequences_v1'
            and index['training_release_allowed'] is True, 'not a guarded recovery publication')
    proofs = {name: pinned_json(root / name, digest) for name, digest in index['evidence'].items()}
    require(set(proofs) == {'source_protocol.json', 'source_assessment.json',
                           'source_audit.json', 'student_replay.json'}, 'incomplete publication evidence')
    protocol, audit = proofs['source_protocol.json'], proofs['source_audit.json']
    require(audit['status'] == 'complete'
            and audit['student_checkpoint_sha256'] == protocol['student_checkpoint_sha256'] == checkpoint,
            'different student or incomplete publication audit')
    key = lambda r: (r['task'], r['seed'], r['source_tick'])
    entries, accepted = {key(r): r for r in index['episodes']}, {key(r): r for r in audit['accepted']}
    require(entries and len(entries) == len(index['episodes']) == len(audit['accepted'])
            and set(entries) == set(accepted)
            and all(r['strict_success'] is True for r in accepted.values()), 'released branch outcomes differ')
    sources = {(r['task'], r['seed']): r for r in protocol['sources']}
    require(len(sources) == len(protocol['sources']), 'duplicate source recording')
    anchors = None
    if 'anchor_selection' in protocol:
        ref = protocol['anchor_selection']
        anchor_report = pinned_json(ref['path'], ref['sha256'])
        require(anchor_report['status'] == 'complete'
                and anchor_report['checkpoint_sha256'] == checkpoint, 'wrong anchor diagnostic')
        anchors = {(r['task'], r['seed']): r for r in anchor_report['cases']}
    controls, rows = {}, []
    for identity, entry in sorted(entries.items()):
        task, seed, tick = identity
        require(entry['split'] == 'train' and entry['student_checkpoint_sha256'] == checkpoint
                and entry['actual_student_anchor_starts'] == [0], 'different training anchor')
        folder = (root / entry['path']).resolve()
        require(root in folder.parents, 'branch leaves release root')
        require(sha(folder / 'navigation.npz') == entry['sha256'], 'changed teacher label file')
        original = sources[task, seed]
        require(tick in original['anchors'], 'branch missing from source protocol')
        if (task, seed) not in controls:
            controls[task, seed] = pinned_json(Path(original['path']) / 'control.json', original['files']['control.json'])
        source_controls = controls[task, seed]
        require(tick < len(source_controls), 'anchor exceeds source controls')
        with np.load(folder / 'navigation.npz', allow_pickle=False) as z:
            result = command_evidence(z['action'], z['ticks'], tick, source_controls[tick])
        row = dict(task=task, seed=seed, source_tick=tick, student_checkpoint_sha256=checkpoint,
            label_sha256=entry['sha256'], student_controls_sha256=original['files']['control.json'],
            reasons=['unclassified_legacy_anchor'], **result)
        if anchors is not None:
            selected = anchors[task, seed]
            require(tick in selected['anchors'], 'branch missing from anchor diagnostic')
            row['reasons'] = selected['reasons'][str(tick)]
            state = next(r for r in selected['selected_anchor_states'] if r['tick'] == tick)
            row['target_surface_distance_m'] = state['target_surface_distance_m']
        rows.append(row)
    return dict(root=str(root), index_sha256=expected_index, student_checkpoint_sha256=checkpoint,
                accepted_branches=len(accepted), rejected_branches=len(audit['rejected']),
                rows=rows, **summarize(rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    args = parser.parse_args()
    plan = pinned_json(args.plan, args.plan_sha256)
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'run this diagnostic without CUDA devices')
    require(sha(__file__) == plan['script_sha256'], 'diagnostic source changed')
    for path, digest in plan['bound_files'].items():
        require(sha(path) == digest, 'bound diagnostic input changed: ' + path)
    from lightx2v_microduck import navigation_action_contracts
    require(Path(navigation_action_contracts.__file__).resolve() ==
            Path(plan['source']) / 'embodied/microduck/src/lightx2v_microduck/navigation_action_contracts.py',
            'diagnostic imported another action contract')
    output = args.plan.parent / 'report.json'
    require(not output.exists(), 'diagnostic report already exists')
    sources = [inspect_release(row['root'], row['index_sha256']) for row in plan['releases']]
    rows = [r for source in sources for r in source['rows']]
    report = dict(schema='microduck_recovery_command_evidence_v1', status='complete',
        finished_unix=time.time(), plan_sha256=args.plan_sha256,
        script_sha256=plan['script_sha256'], sources=sources, **summarize(rows),
        gpu_allocated=False, optimizer_steps=0, labels_changed=False,
        physics_replayed_in_this_diagnostic=False, model_capability_measured=False,
        training_admission=False, formal_acceptance=False,
        limitation='Authenticated teacher command labels only. Motion requests do not prove robot displacement. '
                   'Reason groups may overlap; legacy anchors lack the current failure-state classification.')
    with output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps(report['overall']))


if __name__ == '__main__':
    main()
