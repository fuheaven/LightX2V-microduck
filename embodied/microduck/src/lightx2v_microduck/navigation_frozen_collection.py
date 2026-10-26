"""Collect native FastWAM episodes from an already sealed prospective protocol.

This is a recording entrypoint, not a final acceptance certificate. It checks
the existing development gates and excludes cases recorded in the supplied
exposure census. Complete historical exposure, actual layout independence,
scenario coverage, ablations and realtime performance remain separate audits.
No cohort is generated, model trained, or retained model changed here.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from .navigation_final_protocol import CONTRACT, verify_frozen_protocol
from .navigation_outcomes import FILES, read, require, rows_by_case, sha, summarize_navigation_outcomes, recording_inventory_valid


DEVELOPMENT_INPUTS = frozenset(('development_protocol', 'development_collector', 'development_replay',
                                'development_runtime_config'))


def verify_development_gate(decision, summary):
    """Use recomputed, fully recorded/replayed outcomes, not a declared pass flag."""
    require(decision.get('status') == summary.get('status') == 'complete'
            and decision.get('formal_acceptance') is False
            and decision.get('model_promoted') is False,
            'a complete development decision and outcome summary are required')
    expected = {(scene, seed, 'find_' + color + '_ball')
        for scene, seeds in (('room_clear', (1011, 1032, 1040, 1086, 1091, 1095, 1098)),
                             ('room_curriculum_one', range(2300, 2320)))
        for seed in seeds for color in ('red', 'blue', 'yellow')}
    rows = rows_by_case(summary.get('rows'))
    require(set(rows) == expected, 'development cases differ from the fixed 81-case cohort')
    declared = decision.get('groups')
    require(isinstance(declared, list) and len(declared) == 2
            and {g.get('scene') for g in declared} == {'room_clear', 'room_curriculum_one'},
            'both development groups are required')
    groups = {g['scene']: g for g in declared}
    counts = {}
    for scene, total, baseline in (('room_clear', 21, 19), ('room_curriculum_one', 60, 25)):
        chosen = [r for (s, _, _), r in rows.items() if s == scene]
        require(all(type(r.get('strict_success')) is bool for r in chosen), 'invalid strict outcome')
        wins = sum(r['strict_success'] for r in chosen)
        group = groups[scene]
        require(all(type(group.get(k)) is int for k in ('total', 'baseline_successes', 'candidate_successes'))
                and group['total'] == total and group['baseline_successes'] == baseline
                and group['candidate_successes'] == wins,
                'declared development counts differ from the recorded outcomes')
        counts[scene] = wins
    kept, improved = counts['room_clear'] >= 19, counts['room_curriculum_one'] > 25
    require(decision.get('retained_clear_room') is kept
            and decision.get('obstacle_development_improved') is improved,
            'declared development decision differs from outcomes')
    require(kept and improved, 'development retention/improvement gates have not passed')
    return counts


def verify_known_exclusions(cases, census):
    """Conservative seed exclusion does not establish geometric independence."""
    require(census.get('status') == 'complete' and census.get('read_errors') == [],
            'exposure census is incomplete or contains read errors')
    known = rows_by_case(census.get('cases'))
    excluded = census.get('conservative_global_seed_exclusions')
    require(isinstance(excluded, list) and all(type(n) is int and n >= 0 for n in excluded),
            'conservative exposure seed exclusions are missing')
    excluded = set(excluded)
    unresolved = census.get('unresolved_cases')
    require(isinstance(unresolved, list), 'unresolved exposure records are missing')
    for row in unresolved:
        seed = row.get('seed')
        require(type(seed) is int and seed >= 0 and seed in excluded,
                'unresolved exposure cannot be excluded conservatively')
    planned = rows_by_case(cases)
    require(not set(planned).intersection(known)
            and not {seed for _, seed, _ in planned}.intersection(excluded),
            'prospective cohort overlaps known or unresolved exposure')
    return dict(known_case_count=len(known), excluded_seed_count=len(excluded),
                known_exposure_exclusion_verified=True, exposure_completeness_verified=False)


def check_collection_inputs(protocol_path, seal_path, *, collector_path=None):
    binding = verify_frozen_protocol(protocol_path, seal_path, collector_path=collector_path)
    protocol = read(protocol_path)
    inputs = protocol['frozen_inputs']
    require(DEVELOPMENT_INPUTS <= set(inputs), 'frozen development recording/replay inputs are missing')
    collector = Path(inputs['collector_script']['path']).resolve()
    source = Path(protocol['source']).resolve()
    require(collector == Path(__file__).resolve()
            and collector == source/'embodied/microduck/src/lightx2v_microduck/navigation_frozen_collection.py',
            'run the collector from the frozen source tree')
    manifest = read(inputs['source_manifest']['path'])['files']
    require(manifest.get(str(collector.relative_to(source))) == sha(collector),
            'collector is not covered by the frozen source manifest')
    if 'episode_runner' in inputs:
        runner=Path(inputs['episode_runner']['path']).resolve()
        require(runner == source/'embodied/microduck/src/lightx2v_microduck/navigation_offline_episode.py'
                and manifest.get(str(runner.relative_to(source))) == inputs['episode_runner']['sha256'],
                'offline episode runner is not covered by the frozen source tree')
    decision = read(inputs['development_decision']['path'])
    require(decision.get('assessment_sha256') == inputs['development_replay']['sha256'],
            'development decision identifies another replay')
    summary = summarize_navigation_outcomes(*(inputs[name]['path'] for name in
        ('development_protocol', 'development_collector', 'development_replay')))
    require(summary['weights_sha256'] == protocol['weights_sha256']
            and summary['stats_sha256'] == protocol['stats_sha256']
            and summary['contract'] == CONTRACT,
            'development evidence identifies another model, normalizer or contract')
    development = read(inputs['development_protocol']['path'])
    old_manifest_path = Path(development['source_manifest'])
    require(sha(old_manifest_path) == development['source_manifest_sha256'],
            'development source manifest changed')
    old_files = read(old_manifest_path)['files']
    require(old_files and all(manifest.get(name) == digest for name, digest in old_files.items()),
            'frozen collection source changes an existing development runtime file')
    old_config = Path(inputs['development_runtime_config']['path']).resolve()
    require(old_config == Path(development['output']).resolve()/(CONTRACT+'-config.json')
            and read(old_config) == read(inputs['runtime_config']['path']),
            'collection runtime configuration differs from the development run')
    counts = verify_development_gate(decision, summary)
    exclusion = verify_known_exclusions(protocol['cases'], read(inputs['exposure_census']['path']))
    require(all(Path(case['scene']).name == case['scene'] for case in protocol['cases']),
            'scene names cannot contain directory components')
    from .navigation_history import history_spec_from_contract, verify_evaluation_history
    receipt = read(inputs['checkpoint_receipt']['path'])
    verify_evaluation_history(receipt, history_spec_from_contract(receipt['input_history']),
        weights_sha256=protocol['weights_sha256'], stats_sha256=protocol['stats_sha256'])
    return protocol, dict(status='collection_inputs_checked', binding=binding,
        development_counts=counts, development_gate_verified=True, **exclusion,
        physical_layout_independence_verified=False, formal_acceptance=False,
        hardware_validated=False, model_promoted=False)


def verify_action_trace(traces, predictions):
    """Match raw model outputs to the actual submitted actions, state and language."""
    import numpy as np
    from .navigation_action_contracts import action_contract
    submitted = [p for p in predictions if p.get('submitted_actions') is not None]
    for prediction in predictions:
        if prediction.get('submitted_actions') is None:
            require(prediction.get('accepted') is False and prediction.get('reason') == 'prediction_error'
                    and isinstance(prediction.get('error'),str) and prediction['error'],
                    'an omitted raw prediction must have a recorded inference error')
    require(len(traces) == len(submitted) + 1 and predictions,
            'one warmup prediction and every measured prediction must be recorded')
    for trace, prediction in zip(traces[1:], submitted):
        action, state = np.asarray(trace['raw_action']), np.asarray(trace['state'])
        require(action.shape == (32, 5) and state.shape == (48,)
                and np.isfinite(action).all() and np.isfinite(state).all(), 'invalid model trace')
        require(prediction['action_contract'] == CONTRACT or (prediction.get('accepted') is False
                and prediction.get('reason') == 'wrong_contract'), 'recorded action contract differs')
        low, high = action_contract(prediction['action_contract']).bounds()
        np.testing.assert_array_equal(np.clip(action, low, high),
            np.asarray(prediction['submitted_actions'], np.float32))
        np.testing.assert_array_equal(state, np.asarray(prediction['observed_proprio48'], np.float32))
        require(trace['instruction'] == prediction['observed_instruction'], 'recorded instruction differs')


def recorded_forward_count(traces):
    """Keep the existing replay schema's diagnostic count, excluding warmup."""
    import numpy as np
    return int(sum((row['raw_action'][:, 0] > np.float32(.3)).sum() for row in traces[1:]))


def write(path, value):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


def episode_runner(protocol):
    """Existing protocols retain their original runner; a new one must be frozen."""
    reference=protocol['frozen_inputs'].get('episode_runner')
    if reference is None:
        from .navigation_eval import run_navigation_episode
        return run_navigation_episode
    from . import navigation_offline_episode
    file=Path(navigation_offline_episode.__file__).resolve()
    require(file == Path(reference['path']).resolve()
            and file == Path(protocol['source']).resolve()/'embodied/microduck/src/lightx2v_microduck/navigation_offline_episode.py'
            and sha(file)==reference['sha256'], 'loaded offline episode runner differs from the frozen input')
    return navigation_offline_episode.run_offline_navigation_episode


def collect(protocol_path, seal_path):
    protocol, checked = check_collection_inputs(protocol_path, seal_path)
    output = Path(protocol['output'])
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status='running', stage='loading', started_unix=time.time(),
        protocol_sha256=sha(protocol_path), protocol_seal_sha256=sha(seal_path),
        weights_sha256=protocol['weights_sha256'], stats_sha256=protocol['stats_sha256'],
        source_manifest_sha256=protocol['source_manifest_sha256'],
        scope='Prospectively frozen native navigation recordings; final acceptance requires separate audits.',
        mode='offline', planned_episodes=len(protocol['cases']), completed_episodes=0, episodes=[],
        input_checks=checked, new_training=False, weights_updated=False, teacher_score=False,
        model_promoted=False, formal_acceptance=False, hardware_validated=False)
    write(output/'report.json', report)
    core = original = None
    started = time.monotonic()
    try:
        import numpy as np
        import torch
        require(torch.cuda.is_available() and torch.cuda.device_count() == 1,
                'select exactly one available CUDA device before collection')
        import lightx2v_platform.set_ai_device  # noqa: F401
        from lightx2v.models.runners.wan.fastwam_runner import FastWAMPolicy
        from .env import MicroDuckEnv
        from .wam_history_policy import FastWAMNavigationHistoryPolicy
        run_navigation_episode=episode_runner(protocol)
        config = read(protocol['frozen_inputs']['runtime_config']['path'])
        write(output/(CONTRACT+'-config.json'), config)
        core = FastWAMPolicy.from_config(config)
        policy = FastWAMNavigationHistoryPolicy(core,
            model_id='frozen-' + protocol['weights_sha256'][:12], artifact_sha256=protocol['weights_sha256'])
        require(policy.contract == CONTRACT, 'loaded model action contract differs')
        original = core.predict_action_chunk
        traces = []

        def predict(*args, **kwargs):
            value = np.asarray(original(*args, **kwargs), dtype=np.float32)
            traces.append(dict(raw_action=value.copy(), state=np.asarray(kwargs['state']).copy(),
                rgb_sha256=hashlib.sha256(np.asarray(kwargs['images']['head_camera']).tobytes()).hexdigest(),
                instruction=kwargs['task_description']))
            return value

        core.predict_action_chunk = predict
        report['gpu_uuid'] = str(getattr(torch.cuda.get_device_properties(0), 'uuid', ''))
        for case in protocol['cases']:
            traces.clear()
            name = '%s-%s-seed%d' % (case['scene'], case['task'], case['seed'])
            folder = output/CONTRACT/name
            report.update(stage=CONTRACT+'/'+name); write(output/'report.json', report)
            with MicroDuckEnv(profile='official_bam', scene=case['scene'], image_size=256) as env:
                episode = run_navigation_episode(env, policy, folder, task=case['task'], seed=case['seed'],
                    realtime=False, max_ticks=1500, warmup_ticks=50, warmup_predictions=1,
                    record_frames=True, record_predictions=True)
            verify_action_trace(traces, read(folder/'predictions.json'))
            np.savez_compressed(folder/'unbounded-model-predictions.npz',
                action=np.stack([row['raw_action'] for row in traces]),
                state=np.stack([row['state'] for row in traces]))
            write(folder/'unbounded-model-inputs.json',
                [dict(rgb_sha256=row['rgb_sha256'], instruction=row['instruction']) for row in traces])
            files = {file.name: sha(file) for file in folder.iterdir() if file.is_file()}
            require(recording_inventory_valid(files)
                    and ('cancel.request' not in files or episode['cancelled'] is True),
                    'episode recording file set differs')
            report['episodes'].append(dict(case, contract=CONTRACT, path=str(folder.relative_to(output)),
                strict_success=episode['strict_model_success'], fallback_ticks=episode['fallback_control_ticks'],
                collision=episode['evaluator']['collision'], fallen=episode['evaluator']['ever_fallen'],
                cancelled=episode['cancelled'],
                ticks=episode['evaluator']['tick'],
                raw_forward_predictions_above_legacy=recorded_forward_count(traces), files=files))
            report['completed_episodes'] = len(report['episodes'])
            write(output/'report.json', report)
        report.update(status='complete', stage='awaiting_independent_replay')
    except BaseException as error:
        report.update(status='failed', error=type(error).__name__ + ': ' + str(error))
        raise
    finally:
        if core is not None and original is not None:
            core.predict_action_chunk = original
        report.update(finished_unix=time.time(), process_seconds=time.monotonic()-started)
        write(output/'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--seal', type=Path, required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if args.check_only:
        _, report = check_collection_inputs(args.protocol, args.seal)
    else:
        report = collect(args.protocol, args.seal)
    print(json.dumps({key: report[key] for key in ('status', 'formal_acceptance')}, allow_nan=False))


if __name__ == '__main__':
    main()
