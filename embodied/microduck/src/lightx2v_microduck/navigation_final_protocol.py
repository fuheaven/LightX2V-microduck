"""Seal a prospective navigation protocol before its first model rollout.

This binds files and the recorded ordering of freeze and collection. It does
not prove that an exposure inventory is complete, that declared scenarios are
present, or that a model passed. A seal is not evaluation admission or an
acceptance certificate. Existing development results cannot acquire one
retroactively. No simulator, model, GPU or training process is started here.
"""

import argparse
import json
import math
import os
from pathlib import Path
import time

from .navigation_outcomes import read, require, rows_by_case, sha


SCHEMA = 'microduck_navigation_final_protocol_v1'
SEAL_SCHEMA = 'microduck_navigation_protocol_seal_v1'
CONTRACT = 'microduck_navigation_forward04_v1'
TASK_RULES = {
    'success_protocol': 'visual_ball_stop_v2',
    'control_hz': 50, 'max_ticks': 1500, 'ball_radius_m': .035,
    'stop_surface_distance_m': [.25, .40], 'goal_hold_ticks': 100,
    'minimum_unseen_physical_layouts': 200, 'minimum_strict_success_rate': .8,
    'failure_events': ['collision', 'fallen', 'fault', 'recovery', 'fallback', 'cancelled'],
}
REQUIRED_INPUTS = frozenset((
    'checkpoint', 'checkpoint_receipt', 'stats', 'runtime_config',
    'source_manifest', 'backbone_manifest', 'collector_script', 'replay_script',
    'assessment_script', 'lineage_inventory', 'exposure_inventory',
    'exposure_census', 'scenario_spec', 'ablation_spec', 'realtime_spec',
    'development_decision',
))


def digest(value):
    require(isinstance(value, str) and len(value) == 64
            and not set(value) - set('0123456789abcdef'), 'invalid content digest')
    return value


def timestamp(value):
    require(type(value) in (int, float) and math.isfinite(value) and value > 0,
            'invalid recorded timestamp')
    return value


def bound_file(reference):
    require(isinstance(reference, dict) and set(reference) == {'path', 'sha256'},
            'each frozen input requires a path and content digest')
    require(isinstance(reference['path'], str) and Path(reference['path']).is_absolute(),
            'frozen inputs require absolute paths')
    path = Path(reference['path']).resolve()
    require(path.is_file() and sha(path) == digest(reference['sha256']),
            'frozen input missing or changed: ' + str(path))
    return path


def verify_tree(root, manifest):
    require(isinstance(root, str) and Path(root).is_absolute(), 'artifact tree requires an absolute root')
    root = Path(root).resolve()
    files = read(manifest).get('files')
    require(isinstance(files, dict) and files, 'empty artifact tree manifest')
    for relative, expected in files.items():
        require(isinstance(relative, str) and relative and not Path(relative).is_absolute(),
                'artifact tree entries must be relative')
        path = (root / relative).resolve()
        require(root in path.parents and path.is_file() and sha(path) == digest(expected),
                'artifact tree file missing, changed or outside its root: ' + relative)
    return len(files)


def check_protocol(protocol_path):
    """Check a prospective protocol and rehash its declared inputs, including weights.

    Scenario/exposure/lineage/decision files are bound, not certified by their
    filenames or self-reported booleans. Their substantive gates remain separate.
    """
    protocol = read(protocol_path)
    require(protocol.get('schema') == SCHEMA
            and protocol.get('purpose') == 'final_heldout_evaluation'
            and protocol.get('engine') == 'lightx2v-native-fastwam'
            and protocol.get('profile') == 'official_bam'
            and protocol.get('mode') == 'offline'
            and protocol.get('contracts') == [CONTRACT]
            and protocol.get('training') is False and protocol.get('model_selection') is False
            and protocol.get('formal_acceptance') is False and protocol.get('hardware_validated') is False,
            'requires an explicit native FastWAM final evaluation protocol')
    # JSON booleans compare equal to some numbers in Python; compare canonical
    # JSON so a boolean cannot silently stand in for a quantitative task rule.
    require(json.dumps(protocol.get('task_rules'), sort_keys=True, allow_nan=False)
            == json.dumps(TASK_RULES, sort_keys=True, allow_nan=False),
            'fixed navigation success, failure or layout rules changed')
    require(type(protocol.get('max_ticks')) is int and protocol['max_ticks'] == 1500
            and type(protocol.get('warmup_ticks')) is int and protocol['warmup_ticks'] == 50
            and type(protocol.get('warmup_predictions')) is int and protocol['warmup_predictions'] == 1,
            'evaluation or initialization clock changed')
    timestamp(protocol.get('created_unix'))
    cases = rows_by_case(protocol.get('cases'))
    require(len({(scene, seed) for scene, seed, _ in cases}) >= 200,
            'fewer than 200 candidate scene/seed pairs; physical independence is checked separately')
    require({task for _, _, task in cases} == {'find_red_ball', 'find_blue_ball', 'find_yellow_ball'},
            'the fixed cohort must include all three target instructions')
    inputs = protocol.get('frozen_inputs')
    require(isinstance(inputs, dict) and REQUIRED_INPUTS <= set(inputs),
            'model, runtime, provenance and evaluation specifications must all be frozen')
    paths = {name: bound_file(reference) for name, reference in inputs.items()}
    for role, field in (('checkpoint', 'weights_sha256'), ('stats', 'stats_sha256'),
                        ('checkpoint_receipt', 'checkpoint_receipt_sha256'),
                        ('source_manifest', 'source_manifest_sha256')):
        require(inputs[role]['sha256'] == digest(protocol.get(field)),
                'protocol and frozen input disagree: ' + role)
    receipt = read(paths['checkpoint_receipt'])
    require(receipt.get('weights_sha256') == protocol['weights_sha256']
            and receipt.get('stats_sha256') == protocol['stats_sha256']
            and Path(receipt.get('weights_path', '')).resolve() == paths['checkpoint']
            and type(protocol.get('checkpoint_step')) is int and protocol['checkpoint_step'] > 0
            and receipt.get('step') == protocol['checkpoint_step'],
            'checkpoint receipt, weights, normalizer or clock differ')
    config = read(paths['runtime_config'])
    for field, expected in (('adapter_model_path', paths['checkpoint']),
                            ('dataset_stats_path', paths['stats'])):
        require(isinstance(config.get(field), str) and Path(config[field]).is_absolute()
                and Path(config[field]).resolve() == expected,
                'runtime configuration points at another model input: ' + field)
    require(config.get('navigation_contract') == CONTRACT
            and config.get('policy_profile') == 'microduck_navigation'
            and config.get('normalization_binding') == {
                'execution_contract': CONTRACT, 'coordinate_contract': 'microduck_navigation_v1',
                'stats_sha256': protocol['stats_sha256']}
            and isinstance(receipt.get('input_history'), dict)
            and config.get('navigation_history') == receipt['input_history']
            and config.get('gripper_postprocess') is False,
            'runtime action, normalization or observation contract differs')
    trees = {role: verify_tree(protocol.get(root), paths[role]) for role, root in
             (('source_manifest', 'source'), ('backbone_manifest', 'backbone'))}
    require(isinstance(config.get('model_path'), str) and Path(config['model_path']).is_absolute()
            and Path(config['model_path']).resolve() == Path(protocol['backbone']).resolve(),
            'runtime configuration points at another backbone')
    require(isinstance(protocol.get('output'), str) and Path(protocol['output']).is_absolute(),
            'recording output must be absolute')
    return protocol, paths, trees


def freeze_protocol(protocol_path, seal_path):
    """Exclusively write a seal before an output directory or any collector exists."""
    protocol_path, seal_path = Path(protocol_path).resolve(), Path(seal_path).resolve()
    before = sha(protocol_path)
    protocol, paths, trees = check_protocol(protocol_path)
    output = Path(protocol['output']).resolve()
    require(not output.exists(), 'cannot freeze a protocol after collection output exists')
    require(seal_path != protocol_path and seal_path not in paths.values()
            and seal_path != output and output not in seal_path.parents,
            'seal must be separate from inputs and recording output')
    require(sha(protocol_path) == before, 'protocol changed while verifying its inputs')
    frozen = time.time()
    require(protocol['created_unix'] <= frozen, 'protocol creation is after its freeze')
    result = dict(schema=SEAL_SCHEMA, status='sealed', protocol_sha256=before,
        frozen_unix=frozen, frozen_inputs=protocol['frozen_inputs'], tree_files_rehashed=trees,
        recording_output=str(output), output_absent_at_freeze=True,
        evaluation_admitted=False, formal_acceptance=False, hardware_validated=False)
    # Exclusive creation prevents a later candidate from overwriting this seal.
    with seal_path.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    return result


def verify_frozen_protocol(protocol_path, seal_path, *, collector_path=None):
    """Verify bindings, and optionally freeze-before-collection recorded chronology.

    Files are not signed and local timestamps are not a tamper-proof external
    clock. The collector must call this before loading the model, carry this
    seal's digest, and later provide the independent rollout/replay evidence.
    """
    protocol, _, trees = check_protocol(protocol_path)
    seal = read(seal_path)
    require(seal.get('schema') == SEAL_SCHEMA and seal.get('status') == 'sealed'
            and seal.get('protocol_sha256') == sha(protocol_path)
            and seal.get('frozen_inputs') == protocol['frozen_inputs']
            and seal.get('recording_output') == str(Path(protocol['output']).resolve())
            and seal.get('tree_files_rehashed') == trees and seal.get('output_absent_at_freeze') is True
            and seal.get('formal_acceptance') is False and seal.get('evaluation_admitted') is False,
            'seal is missing or refers to another protocol or inputs')
    require(protocol['created_unix'] <= timestamp(seal.get('frozen_unix')),
            'protocol was not recorded before the freeze')
    if collector_path is not None:
        collector = read(collector_path)
        require(Path(collector_path).resolve() == Path(protocol['output']).resolve()/'report.json'
                and collector.get('protocol_seal_sha256') == sha(seal_path)
                and collector.get('protocol_sha256') == sha(protocol_path)
                and collector.get('weights_sha256') == protocol['weights_sha256']
                and collector.get('stats_sha256') == protocol['stats_sha256']
                and collector.get('source_manifest_sha256') == protocol['source_manifest_sha256']
                and collector.get('mode') == 'offline'
                and collector.get('teacher_score') is False
                and collector.get('new_training') is False and collector.get('weights_updated') is False
                and timestamp(collector.get('started_unix')) >= seal['frozen_unix'],
                'collector differs from the seal or started before it was frozen')
    return dict(schema='microduck_navigation_protocol_binding_v1', status='verified',
        protocol_sha256=sha(protocol_path), protocol_seal_sha256=sha(seal_path),
        inputs_rehashed=True, recorded_freeze_precedes_collection=collector_path is not None,
        candidate_scene_seed_pairs=len({(s, n) for s, n, _ in rows_by_case(protocol['cases'])}),
        physical_layout_independence_verified=False, exposure_completeness_verified=False,
        development_gate_verified=False, scenario_coverage_verified=False,
        ablations_verified=False, realtime_verified=False,
        evaluation_admitted=False, formal_acceptance=False, hardware_validated=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('freeze', 'verify'))
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--seal', type=Path, required=True)
    parser.add_argument('--collector', type=Path)
    args = parser.parse_args()
    if args.operation == 'freeze':
        require(args.collector is None, 'freeze does not accept already collected results')
        result = freeze_protocol(args.protocol, args.seal)
    else:
        result = verify_frozen_protocol(args.protocol, args.seal, collector_path=args.collector)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
