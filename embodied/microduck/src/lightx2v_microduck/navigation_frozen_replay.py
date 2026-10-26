"""CPU replay of every episode in a prospectively frozen native cohort.

Reuse the independently verified control/RGB replay kernel without its fixed
81-case, two-contract command-line wrapper. Task failures remain in the cohort;
an unsupported recording or replay mismatch makes the whole audit incomplete.
This does not certify unseen layouts, complete exposure history, scenario
coverage, input ablations, realtime behavior, or final acceptance.
"""
import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import time

from .navigation_final_protocol import CONTRACT
from .navigation_frozen_collection import check_collection_inputs, write
from .navigation_holdout import initial_layout_identity
from .navigation_outcomes import FILES, key, read, require, rows_by_case, sha, recording_inventory_valid


def check_recordings(protocol, collector):
    expected = rows_by_case(protocol['cases'])
    recorded = rows_by_case(collector.get('episodes'))
    require(collector.get('status') == 'complete'
            and collector.get('stage') == 'awaiting_independent_replay'
            and type(collector.get('planned_episodes')) is int
            and type(collector.get('completed_episodes')) is int
            and collector['planned_episodes'] == collector['completed_episodes'] == len(expected)
            and set(expected) == set(recorded), 'recording cohort is incomplete or differs from the protocol')
    require(protocol['contracts'] == [CONTRACT], 'requires the frozen navigation action contract')
    root = Path(protocol['output']).resolve()
    folders = set()
    for row in recorded.values():
        require(row.get('contract') == CONTRACT and type(row.get('strict_success')) is bool,
                'invalid recording contract or outcome')
        require(isinstance(row.get('path'), str) and row['path']
                and not Path(row['path']).is_absolute(), 'episode path must be relative')
        folder = (root/row['path']).resolve()
        require(root in folder.parents and folder not in folders, 'episode path escapes root or is reused')
        folders.add(folder)
        require(recording_inventory_valid(row.get('files')),
                'episode recording files differ')
        require(all(folder in (folder/name).resolve().parents for name in row['files']),
                'recording file escapes its episode folder')
    return expected


def _replay_recordings(protocol, protocol_path, collector_path, output, initial_output, workers,
                      replay_rows, *, binding=None):
    """Common dispatcher; injection of a replay callable is for CPU verification."""
    require(type(workers) is int and 1 <= workers <= 16, 'CPU replay requires 1 to 16 workers')
    collector = read(collector_path)
    expected = check_recordings(protocol, collector)
    output, initial_output = Path(output), Path(initial_output)
    require(output.resolve() != initial_output.resolve()
            and not output.exists() and not initial_output.exists(), 'replay outputs must be new and separate')
    source_digest = sha(collector_path)
    report = dict(schema='microduck_frozen_navigation_replay_v1', status='running',
        started_unix=time.time(), protocol_sha256=sha(protocol_path),
        model_report_sha256=source_digest, checkpoint_sha256=protocol['weights_sha256'],
        stats_sha256=protocol['stats_sha256'], source_manifest_sha256=protocol['source_manifest_sha256'],
        planned=len(expected), completed=0, rows=[], cpu_workers=workers,
        replay_start_method='spawn' if workers > 1 else 'serial',
        input_binding=binding, gpu_allocated=False, new_training=False,
        model_promoted=False, formal_acceptance=False, hardware_validated=False)
    # Reserve both output names before any physical work, without overwriting
    # prior results. A partial inventory cannot be mistaken for complete output.
    with output.open('x') as stream:
        json.dump(report, stream, allow_nan=False)
    inventory = dict(schema='microduck_navigation_development_exposure_inventory_v1', status='running',
        role='frozen_cohort_initial_layout_evidence', protocol_sha256=report['protocol_sha256'],
        collected_sha256=source_digest, checkpoint_sha256=protocol['weights_sha256'],
        planned_cases=len(expected), completed_cases=0, rows=[],
        formal_acceptance=False, hardware_validated=False)
    with initial_output.open('x') as stream:
        json.dump(inventory, stream, allow_nan=False)
    seen = set()
    try:
        for row, verified, initial in replay_rows(Path(protocol['output']), collector['episodes'], protocol, workers):
            case = key(row)
            require(case in expected and case not in seen and key(verified) == case,
                    'replay omitted, duplicated or substituted a case')
            require(verified.get('contract') == CONTRACT
                    and verified.get('input_files_sha256') == row['files']
                    and type(verified.get('strict_success')) is bool
                    and verified['strict_success'] == row['strict_success']
                    and verified.get('final_evaluator_exact') is True
                    and verified.get('submitted_actions_exact') is True
                    and verified.get('buffer_execution_exact') is True,
                    'incomplete replay evidence or changed outcome')
            ticks = row.get('ticks')
            require(type(ticks) is int and 1 <= ticks <= 1500
                    and verified.get('exact_control_ticks') == ticks
                    and verified.get('exact_rgb_frames') == (ticks+4)//5,
                    'replay does not cover every recorded control and camera tick')
            if verified.get('schema') == 'microduck_navigation_cohort_case_replay_v2':
                submitted = verified.get('submitted_prediction_records')
                errors = verified.get('prediction_error_records')
                require(type(submitted) is int and submitted >= 0
                        and type(errors) is int and errors >= 0
                        and submitted+errors == verified.get('prediction_records') == (ticks+4)//5
                        and verified.get('model_input_trace_complete') is True
                        and verified.get('model_input_trace_records') == submitted+1
                        and verified.get('exact_history_grids') == submitted
                        and type(verified.get('cancelled')) is bool
                        and verified['cancelled'] == row.get('cancelled', False),
                        'model inputs, errors or cancellation were not fully replayed')
            else:
                require(verified.get('schema') is None
                        and verified.get('exact_history_grids') == (ticks+4)//5,
                        'incomplete or unknown model history replay')
            seen.add(case)
            report['rows'].append(verified)
            report['completed'] = len(seen)
            inventory['rows'].append(dict(scene=case[0], seed=case[1], task=case[2],
                initial_world=initial['world'], layout_identity=initial_layout_identity(initial['world']),
                recorded_files_sha256=row['files'], sensor_binding=dict(
                    qpos_exact=True, head_rgb_exact=True, proprio48_exact=True,
                    model_camera_grid_exact=True, instruction_exact=True)))
            inventory['completed_cases'] = len(seen)
            write(output, report); write(initial_output, inventory)
        require(seen == set(expected), 'replay stopped before all planned cases were verified')
        require(sha(collector_path) == source_digest, 'collector changed during independent replay')
        report.update(status='complete', finished_unix=time.time(),
            strict_successes=sum(row['strict_success'] for row in report['rows']),
            exact_control_ticks=sum(row['exact_control_ticks'] for row in report['rows']),
            exact_rgb_frames=sum(row['exact_rgb_frames'] for row in report['rows']))
        write(output, report)
        inventory.update(status='complete', assessment_sha256=sha(output), finished_unix=time.time())
        write(initial_output, inventory)
    except BaseException as error:
        report.update(status='failed', error=type(error).__name__+': '+str(error), finished_unix=time.time())
        inventory.update(status='failed', error=report['error'])
        write(output, report); write(initial_output, inventory)
        raise
    return report


def replay_frozen(protocol_path, seal_path, output, initial_output, workers):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'independent replay requires CUDA hidden')
    require(not sys.flags.optimize, 'the pinned replay kernel requires Python assertions enabled')
    protocol = read(protocol_path)
    collector_path = Path(protocol['output'])/'report.json'
    protocol, checked = check_collection_inputs(protocol_path, seal_path, collector_path=collector_path)
    inputs = protocol['frozen_inputs']
    source = Path(protocol['source']).resolve()
    this_file = Path(__file__).resolve()
    require(Path(inputs['replay_script']['path']).resolve() == this_file
            and this_file == source/'embodied/microduck/src/lightx2v_microduck/navigation_frozen_replay.py'
            and read(inputs['source_manifest']['path'])['files'].get(str(this_file.relative_to(source))) == sha(this_file),
            'run the replay dispatcher from its frozen source tree')
    require('replay_kernel' in inputs, 'the independent replay kernel must be frozen before collection')
    kernel_path = Path(inputs['replay_kernel']['path']).resolve()
    kernel_modules = {'assess_navigation_contract_cohort.py':'assess_navigation_contract_cohort',
                      'navigation_cohort_replay.py':'lightx2v_microduck.navigation_cohort_replay'}
    require(kernel_path.name in kernel_modules, 'unexpected replay kernel')
    # Keep this path in sys.path while multiprocessing starts: spawned workers
    # must import the same module when unpickling the pinned replay function.
    sys.path.insert(0, str(kernel_path.parent))
    try:
        kernel = importlib.import_module(kernel_modules[kernel_path.name])
        require(Path(kernel.__file__).resolve() == kernel_path
                and sha(kernel_path) == inputs['replay_kernel']['sha256'], 'loaded replay kernel differs')
        from . import env as env_module
        require(Path(env_module.__file__).resolve() == source/'embodied/microduck/src/lightx2v_microduck/env.py',
                'loaded simulator differs from the frozen source')
        return _replay_recordings(protocol, protocol_path, collector_path, output, initial_output, workers,
                                  kernel.replay_rows, binding=checked)
    finally:
        sys.path.pop(0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('protocol', 'seal', 'output', 'initial-output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    report = replay_frozen(args.protocol, args.seal, args.output, args.initial_output, args.workers)
    print(json.dumps({key:report[key] for key in ('status', 'completed', 'strict_successes', 'formal_acceptance')}))


if __name__ == '__main__':
    main()
