"""Package an evaluated native navigation checkpoint without changing weights.

The plan pins the original evaluation, resolved configuration and training
receipt. Large weights remain an explicitly declared, hash-verified reference.
Packaging does not promote the model or establish task acceptance.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def package(plan, output):
    from lightx2v_microduck.navigation_action_contracts import verify_artifact_action_contract
    from lightx2v_microduck.navigation_evaluation_contract import checkpoint_execution
    from lightx2v_microduck.navigation_history import history_spec_from_contract, verify_evaluation_history

    output = Path(output).resolve()
    require(not output.exists(), 'artifact destinations are immutable')
    require(plan.get('schema') == 'microduck_evaluated_artifact_package_v1', 'unsupported package plan')
    expected = {'checkpoint', 'stats', 'evaluation', 'resolved_config', 'checkpoint_receipt', 'backbone_lock'}
    require(set(plan['sources']) == expected, 'package needs all pinned model and evaluation sources')
    paths = {}
    for name, entry in plan['sources'].items():
        path = Path(entry['path'])
        require(path.is_absolute() and path.is_file(), 'missing absolute source: ' + name)
        require(sha(path) == entry['sha256'], 'source content changed: ' + name)
        paths[name] = path.resolve()
    report = json.loads(paths['evaluation'].read_text())
    config = json.loads(paths['resolved_config'].read_text())
    receipt = json.loads(paths['checkpoint_receipt'].read_text())
    stats = json.loads(paths['stats'].read_text())
    weights_sha = plan['sources']['checkpoint']['sha256']
    stats_sha = plan['sources']['stats']['sha256']
    require(report.get('status') == 'complete' and report.get('backend') == 'native',
            'a completed actual native evaluation is required')
    require(report.get('checkpoint_sha256') == weights_sha and report.get('stats_sha256') == stats_sha,
            'evaluation used different model bytes')
    require(report.get('checkpoint_receipt_sha256') == plan['sources']['checkpoint_receipt']['sha256'],
            'evaluation used another training receipt')
    require(report.get('input_intervention') == {'mask_rgb': False, 'instruction_color': None}
            and report.get('action_intervention') is None, 'cannot package an intervention as the ordinary policy')
    require(Path(config['adapter_model_path']).resolve() == paths['checkpoint']
            and Path(config['dataset_stats_path']).resolve() == paths['stats'],
            'resolved runtime points to different model files')
    contract, binding = checkpoint_execution(receipt, weights_sha256=weights_sha,
        stats_sha256=stats_sha, requested=report['navigation_contract'])
    require(config.get('navigation_contract', 'microduck_navigation_v1') == contract
            and config.get('normalization_binding') == binding == report.get('normalization_binding'),
            'execution and normalization binding changed')
    history = report.get('input_history')
    require(history == config.get('navigation_history'), 'runtime history differs from evaluated inputs')
    if history is not None:
        verify_evaluation_history(receipt, history_spec_from_contract(history),
                                  weights_sha256=weights_sha, stats_sha256=stats_sha)
    for report_key, config_key in [('inference_steps', 'action_infer_steps'),
                                   ('cache_prompt', 'fastwam_cache_prompt'),
                                   ('action_context_cache', 'fastwam_cache_action_context'),
                                   ('action_cuda_graph', 'fastwam_action_cuda_graph'),
                                   ('action_rms_one_pass', 'fastwam_action_rms_one_pass')]:
        require(report[report_key] == config[config_key], 'runtime option changed: ' + config_key)
    require(report.get('compiled') is False, 'this native package does not support compiled models')
    require((config['policy_profile'], config['action_dim'], config['robot_state_dim'])
            == ('microduck_navigation', 5, 48), 'not a MicroDuck navigation model')
    require(config['gripper_postprocess'] is False and config['binarize_gripper'] is False,
            'manipulator postprocessing is not supported')
    manifest = dict(schema='microduck_navigation_artifact_v1', model_id=plan['model_id'],
        embodiment='legs', contract=contract, normalization_binding=binding, history_contract=history,
        files={'model.pt': weights_sha, 'dataset_stats.json': stats_sha,
               'lightx2v_config.json': plan['sources']['resolved_config']['sha256']},
        storage={'model.pt': 'verified_external_reference'}, sources=plan['sources'],
        experimental=True, model_promoted=False, independent_acceptance=False, hardware_validated=False,
        evidence='Packaged evaluated weights and runtime; no new model run or acceptance claim.')
    verify_artifact_action_contract(manifest, config, stats)
    output.mkdir(parents=True)
    (output / 'model.pt').symlink_to(paths['checkpoint'])
    for source, target in [('stats', 'dataset_stats.json'), ('resolved_config', 'lightx2v_config.json'),
                           ('checkpoint_receipt', 'navigation_checkpoint.json'), ('backbone_lock', 'backbone-assets.lock.json')]:
        shutil.copyfile(paths[source], output / target)
        require(sha(output / target) == plan['sources'][source]['sha256'], 'copy changed: ' + target)
    (output / 'package-plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    (output / 'navigation_model.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return dict(status='complete', artifact=str(output), manifest_sha256=sha(output / 'navigation_model.json'),
                checkpoint_sha256=weights_sha, stats_sha256=stats_sha, contract=contract,
                optimizer_steps=0, model_promoted=False, independent_acceptance=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(sha(args.plan) == args.plan_sha256, 'package plan changed')
    result = package(json.loads(args.plan.read_text()), args.output)
    (args.output / 'package-result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
