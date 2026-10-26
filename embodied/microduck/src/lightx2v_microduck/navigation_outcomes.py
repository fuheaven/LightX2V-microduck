"""Summarize complete native FastWAM recordings and their existing offline replay.

This reads and binds replay evidence; it does not run physics or certify unseen
layouts, model lineage, realtime execution, or final acceptance. In particular,
scene/seed counts are never presented as distinct physical layout counts.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


FILES = frozenset(('control.json', 'decisions.json', 'predictions.json', 'recording.npz',
                   'report.json', 'unbounded-model-inputs.json', 'unbounded-model-predictions.npz'))


def recording_inventory_valid(files):
    """A retained cancellation request is the only optional episode artifact."""
    return isinstance(files, dict) and FILES <= set(files) <= FILES | {'cancel.request'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    def reject(value):
        raise ValueError('nonfinite JSON: ' + value)
    return json.loads(Path(path).read_text(), parse_constant=reject)


def key(row):
    scene, seed, task = (row.get(k) for k in ('scene', 'seed', 'task'))
    require(isinstance(scene, str) and scene and type(seed) is int and seed >= 0
            and task in ('find_red_ball', 'find_blue_ball', 'find_yellow_ball'), 'invalid navigation case')
    return scene, seed, task


def rows_by_case(rows):
    require(isinstance(rows, list) and rows, 'empty or missing navigation cohort')
    result = {key(row): row for row in rows}
    require(len(result) == len(rows), 'duplicate navigation case')
    return result


def summarize_navigation_outcomes(protocol_path, collector_path, replay_path, *, episode_root=None):
    protocol, collector, replay = (read(p) for p in (protocol_path, collector_path, replay_path))
    require(protocol.get('mode') == collector.get('mode') == 'offline'
            and protocol.get('training') is False and protocol.get('max_ticks') == 1500,
            'requires the fixed offline navigation task clock')
    require(collector.get('status') == 'complete' and collector.get('stage') == 'awaiting_independent_replay'
            and collector.get('protocol_sha256') == sha(protocol_path)
            and collector.get('weights_sha256') == protocol.get('weights_sha256')
            and collector.get('stats_sha256') == protocol.get('stats_sha256')
            and collector.get('source_manifest_sha256') == protocol.get('source_manifest_sha256')
            and collector.get('teacher_score') is False
            and collector.get('new_training') is False and collector.get('weights_updated') is False,
            'collector is incomplete or differs from the pinned model/runtime')
    require(replay.get('status') == 'complete' and replay.get('model_report_sha256') == sha(collector_path)
            and replay.get('checkpoint_sha256') == protocol.get('weights_sha256'),
            'replay is incomplete or bound to another collector/model')
    expected = rows_by_case(protocol.get('cases'))
    recorded, verified = rows_by_case(collector.get('episodes')), rows_by_case(replay.get('rows'))
    require(set(expected) == set(recorded) == set(verified)
            and collector.get('planned_episodes') == collector.get('completed_episodes')
                == replay.get('completed') == len(expected), 'partial or selected navigation cohort')
    contracts = protocol.get('contracts')
    require(isinstance(contracts, list) and len(contracts) == 1 and isinstance(contracts[0], str),
            'one explicit action contract required')
    root = Path(episode_root if episode_root is not None else protocol['output']).resolve()
    outcomes = []
    for case in expected:
        row, proof = recorded[case], verified[case]
        ticks = row.get('ticks')
        require(type(ticks) is int and 1 <= ticks <= 1500, 'invalid episode duration')
        require(row.get('contract') == proof.get('contract') == contracts[0]
                and proof.get('final_evaluator_exact') is True and proof.get('submitted_actions_exact') is True
                and proof.get('buffer_execution_exact') is True and proof.get('exact_control_ticks') == ticks
                and proof.get('exact_rgb_frames') == (ticks+4)//5,
                'incomplete physical, camera, action or clock replay')
        files = row.get('files')
        require(recording_inventory_valid(files) and proof.get('input_files_sha256') == files,
                'recording/replay file identity differs')
        relative = row.get('path')
        require(isinstance(relative, str) and relative and not Path(relative).is_absolute(),
                'episode folder must be relative')
        folder = (root/relative).resolve()
        require(root in folder.parents, 'episode folder escapes recording root')
        for name, digest in files.items():
            path = (folder/name).resolve()
            require(folder in path.parents and path.is_file() and sha(path) == digest,
                    'recording changed after replay: ' + name)
        episode = read(folder/'report.json')
        final, policy = episode['evaluator'], episode['policy']
        require('cancel.request' not in files or episode.get('cancelled') is True,
                'retained cancellation request lacks a cancelled outcome')
        if proof.get('schema') == 'microduck_navigation_cohort_case_replay_v2':
            predictions = read(folder/'predictions.json')
            require(isinstance(predictions,list) and len(predictions)==(ticks+4)//5,
                    'offline prediction inventory is incomplete')
            require([p['source_tick'] for p in predictions] == list(range(0,ticks,5)),
                    'offline predictions omit or duplicate decision observations')
            submitted = sum(p['submitted_actions'] is not None for p in predictions)
            require(proof.get('model_input_trace_complete') is True
                    and proof.get('model_input_trace_records') == submitted+1
                    and proof.get('submitted_prediction_records') == proof.get('exact_history_grids') == submitted
                    and proof.get('prediction_records') == len(predictions)
                    and proof.get('prediction_error_records') == len(predictions)-submitted
                    and proof.get('cancelled') == episode.get('cancelled'),
                    'replayed model input or error counts differ')
        else:
            require(proof.get('schema') is None
                    and proof.get('exact_history_grids') == (ticks+4)//5,
                    'incomplete or unknown history replay')
        require((episode.get('task'), episode.get('seed')) == (case[2], case[1])
                and episode.get('mode') == 'offline' and episode.get('action_contract') == contracts[0]
                and policy.get('artifact_sha256') == protocol['weights_sha256']
                and policy.get('uses_privileged_state') is False
                and final.get('tick') == ticks and final.get('scene', {}).get('name') == case[0]
                and final.get('target_color') == case[2][5:-5], 'episode policy, scene or task identity differs')
        for field in ('success', 'collision', 'ever_fallen'):
            require(type(final.get(field)) is bool, 'invalid evaluator outcome: ' + field)
        require(type(episode.get('cancelled')) is bool and isinstance(final.get('fault'), str),
                'missing cancellation or fault evidence')
        require(type(final.get('terminated')) is bool
                and (final['terminated'] or episode['cancelled'] or bool(final['fault']))
                and (ticks == 1500 or final['success'] or final['ever_fallen']
                     or episode['cancelled'] or bool(final['fault'])),
                'unfinished episode cannot count as a complete navigation trial')
        for seconds in (episode.get('sim_s'), final.get('evaluation_elapsed_s')):
            require(type(seconds) in (int, float) and math.isfinite(seconds)
                    and math.isclose(seconds, ticks / 50, rel_tol=0, abs_tol=1e-6),
                    'recorded task duration differs from the 50 Hz control clock')
        require(final.get('success_protocol') == 'visual_ball_stop_v2'
                and episode.get('inference_worker_stopped') is True,
                'missing fixed task predicate or terminal inference-worker evidence')
        for value in (final.get('recovery_events'), final.get('goal_hold_ticks'), episode.get('fallback_control_ticks')):
            require(type(value) is int and value >= 0, 'invalid intervention or goal-hold count')
        hazards = dict(collision=final['collision'], fallen=final['ever_fallen'], fault=bool(final['fault']),
            recovery=final['recovery_events'] > 0, fallback=episode['fallback_control_ticks'] > 0,
            cancelled=episode['cancelled'])
        strict = final['success'] and not any(hazards.values())
        require(type(row.get('strict_success')) is bool and type(proof.get('strict_success')) is bool
                and row['strict_success'] == proof['strict_success'] == episode.get('strict_model_success') == strict
                and episode.get('task_success') == final['success'], 'strict outcome differs from recorded evaluator')
        require(row.get('collision') == proof.get('collision') == final['collision']
                and row.get('fallen') == proof.get('fallen') == final['ever_fallen']
                and row.get('fallback_ticks') == proof.get('fallback_ticks') == episode['fallback_control_ticks']
                and proof.get('recovery_events') == final['recovery_events'], 'replayed intervention counts differ')
        if strict:
            robot, ball = final['robot_position'], final['balls'][final['target_color']]
            distance = math.hypot(robot[0]-ball[0], robot[1]-ball[1])-.035
            require(final['goal_hold_ticks'] >= 100 and math.isfinite(distance)
                    and .25-1e-8 <= distance <= .40+1e-8, 'successful episode lacks the fixed stop/hold predicate')
        reasons = [name for name, value in hazards.items() if value]
        if not strict and not reasons:
            reasons.append('goal_not_completed')
        outcomes.append(dict(scene=case[0], seed=case[1], task=case[2], strict_success=strict,
            failure_reasons=reasons, control_ticks=ticks, recording_files_sha256=files))

    def group(rows):
        wins = sum(row['strict_success'] for row in rows)
        return dict(trials=len(rows), strict_successes=wins, success_rate=wins/len(rows),
            scene_seed_pairs=len({(r['scene'], r['seed']) for r in rows}),
            failure_counts={name:sum(name in r['failure_reasons'] for r in rows) for name in
                ('collision','fallen','fault','recovery','fallback','cancelled','goal_not_completed')})

    return dict(schema='microduck_native_navigation_outcome_summary_v1', status='complete', mode='offline',
        protocol_sha256=sha(protocol_path), collector_sha256=sha(collector_path), replay_sha256=sha(replay_path),
        weights_sha256=protocol['weights_sha256'], stats_sha256=protocol['stats_sha256'],
        source_manifest_sha256=protocol['source_manifest_sha256'], contract=contracts[0],
        overall=group(outcomes),
        by_scene={scene:group([r for r in outcomes if r['scene']==scene]) for scene in sorted({r['scene'] for r in outcomes})},
        by_task={task:group([r for r in outcomes if r['task']==task]) for task in sorted({r['task'] for r in outcomes})},
        rows=outcomes, failure_counts_are_nonexclusive=True, recordings_rehashed=True,
        terminal_trials_verified=True, task_clock_verified=True,
        physics_rerun_in_this_summary=False, unique_physical_layouts=None, unseen_layouts_verified=False,
        complete_training_and_selection_lineage_verified=False, formal_acceptance=False, hardware_validated=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('protocol', 'collector', 'replay', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--episode-root', type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), 'summary output already exists')
    result = summarize_navigation_outcomes(args.protocol, args.collector, args.replay, episode_root=args.episode_root)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result['overall']))


if __name__ == '__main__':
    main()
