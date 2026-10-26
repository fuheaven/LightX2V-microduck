"""Join native navigation scores to independently recorded initial layouts.

This is an offline acceptance-evidence component, not a final acceptance gate.
It rehashes the original recordings and checks their existing replay receipts.
It counts physical layouts against the supplied exclusion inventory; it cannot
prove that inventory contains every training or model-selection exposure.
"""

import argparse
import json
from pathlib import Path

from .navigation_holdout import audit_layout_independence, initial_layout_identity
from .navigation_outcomes import key, read, require, rows_by_case, sha, summarize_navigation_outcomes


def summarize_layout_outcomes(protocol_path, collector_path, replay_path,
                              initial_inventory_path, exposure_inventory_path,
                              *, exposure_census_path=None, episode_root=None):
    outcomes = summarize_navigation_outcomes(protocol_path, collector_path, replay_path,
                                            episode_root=episode_root)
    initial = read(initial_inventory_path)
    exposures = read(exposure_inventory_path)
    require(initial.get('schema') == 'microduck_navigation_development_exposure_inventory_v1'
            and initial.get('status') == 'complete'
            and initial.get('protocol_sha256') == outcomes['protocol_sha256']
            and initial.get('collected_sha256') == outcomes['collector_sha256']
            and initial.get('assessment_sha256') == outcomes['replay_sha256']
            and initial.get('checkpoint_sha256') == outcomes['weights_sha256'],
            'initial-layout replay belongs to another or incomplete evaluated cohort')
    require(exposures.get('schema') == 'microduck_known_layout_exclusion_inventory_v2'
            and exposures.get('status') == 'complete'
            and exposures.get('role') == 'exclude_known_training_and_development_layouts',
            'requires an explicit completed physical exposure inventory')
    recorded, scored = rows_by_case(initial.get('rows')), rows_by_case(outcomes['rows'])
    require(set(recorded) == set(scored)
            and initial.get('planned_cases') == initial.get('completed_cases') == len(scored),
            'initial-layout replay omits, duplicates or selects evaluated cases')
    sensor_fields = ('qpos_exact', 'head_rgb_exact', 'proprio48_exact',
                     'model_camera_grid_exact', 'instruction_exact')
    for case, row in recorded.items():
        require(row.get('recorded_files_sha256') == scored[case]['recording_files_sha256']
                and all(row.get('sensor_binding', {}).get(field) is True for field in sensor_fields)
                and row.get('layout_identity') == initial_layout_identity(row.get('initial_world')),
                'initial-layout sensor or physical identity differs from evaluated recordings')
    declarations, unresolved = [], []
    if exposure_census_path is not None:
        census = read(exposure_census_path)
        require(census.get('schema') == 'microduck_available_report_exposure_census_v1'
                and census.get('status') == 'complete' and census.get('read_errors') == []
                and census.get('previous_inventory', {}).get('sha256') == sha(exposure_inventory_path),
                'exposure census is incomplete or refers to another physical inventory')
        declarations = census.get('cases')
        unresolved = census.get('conservative_global_seed_exclusions')
        require(isinstance(declarations, list) and isinstance(unresolved, list),
                'missing explicit exposure declarations or conservative seed exclusions')
    independence = audit_layout_independence(list(recorded.values()), exposures.get('rows'),
        declared_exposures=declarations, unresolved_scene_seeds=unresolved)
    identities = rows_by_case(independence['rows'])
    rows = [dict(row, geometry_sha256=identities[key(row)]['geometry_sha256'],
                 exposure_issues=identities[key(row)]['issues']) for row in outcomes['rows']]
    by_geometry = {}
    for row in rows:
        by_geometry.setdefault(row['geometry_sha256'], []).append(row)
    layouts = []
    for geometry, trials in sorted(by_geometry.items()):
        successes = sum(row['strict_success'] for row in trials)
        layouts.append(dict(geometry_sha256=geometry, trials=len(trials),
            strict_successes=successes, success_rate=successes/len(trials),
            exposure_issues=sorted({issue for row in trials for issue in row['exposure_issues']}),
            cases=[dict(scene=row['scene'], seed=row['seed'], task=row['task']) for row in trials]))
    return dict(schema='microduck_navigation_layout_outcomes_v1', status='complete',
        role='development_evidence_component', mode='offline',
        evidence=dict(protocol_sha256=outcomes['protocol_sha256'], collector_sha256=outcomes['collector_sha256'],
            replay_sha256=outcomes['replay_sha256'], initial_layout_replay_sha256=sha(initial_inventory_path),
            exposure_inventory_sha256=sha(exposure_inventory_path),
            exposure_census_sha256=sha(exposure_census_path) if exposure_census_path is not None else None),
        weights_sha256=outcomes['weights_sha256'], stats_sha256=outcomes['stats_sha256'],
        source_manifest_sha256=outcomes['source_manifest_sha256'], contract=outcomes['contract'],
        overall=outcomes['overall'], by_scene=outcomes['by_scene'], by_task=outcomes['by_task'],
        physical_layouts=dict(unique=len(layouts),
            unique_unseen_in_supplied_inventory=independence['unique_unseen_physical_layouts'],
            mean_success_rate=sum(row['success_rate'] for row in layouts)/len(layouts),
            all_evaluated_instructions_succeeded=sum(row['strict_successes'] == row['trials'] for row in layouts),
            mean_success_rate_is_informational=True),
        layouts=layouts, rows=rows, independence=independence,
        recorded_trial_score_target_met=outcomes['overall']['success_rate'] >= .8,
        supplied_unseen_layout_count_target_met=independence['unique_unseen_physical_layouts'] >= 200,
        terminal_trials_verified=True, task_clock_verified=True, recordings_rehashed=True,
        physics_rerun_in_this_summary=False, complete_training_and_selection_lineage_verified=False,
        final_protocol_admission_verified=False, coverage_and_ablations_verified=False,
        realtime_acceptance=False, formal_acceptance=False, hardware_validated=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('protocol', 'collector', 'replay', 'initial-inventory', 'exposure-inventory', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--exposure-census', type=Path)
    parser.add_argument('--episode-root', type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), 'layout outcome summary already exists')
    result = summarize_layout_outcomes(args.protocol, args.collector, args.replay,
        args.initial_inventory, args.exposure_inventory,
        exposure_census_path=args.exposure_census, episode_root=args.episode_root)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(overall=result['overall'], physical_layouts=result['physical_layouts'],
                         formal_acceptance=False)))


if __name__ == '__main__':
    main()
