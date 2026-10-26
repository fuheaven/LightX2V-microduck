"""Audit a predeclared offline hybrid cohort without dropping failed cases.

The caller supplies independently bound policy metadata and execution settings.
This does not call a model, allocate a GPU, establish historical runtime identity,
or promote development results to capability acceptance.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from .navigation_hybrid_audit import audit_offline_hybrid_episode
from .navigation_outcomes import read, require, sha
from .vlm_service import canonical


def case_key(case):
    return case['scene'], case['task'], case['seed']


def audit_hybrid_cohort(evaluation, requests, expectations, output):
    evaluation, output = Path(evaluation), Path(output)
    require(not output.exists(), 'audit output already exists')
    require(expectations['mode'] == 'offline' and expectations['independent_acceptance'] is False,
            'hybrid cohort audit supports offline development only')
    planned = expectations['cohort']
    keys = [case_key(c) for c in planned['cases']]
    require(keys and len(set(keys)) == len(keys), 'declared cohort is empty or duplicated')
    report_path = evaluation / 'report.json'
    report_sha = sha(report_path)
    report = read(report_path)
    require(report['schema'] == 'microduck_hybrid_navigation_evaluation_v1'
            and report['status'] in ('complete', 'failed', 'cancelled_between_episodes'),
            'evaluation has no supported terminal report')
    require(report['cohort'] == planned and report['mode'] == 'offline'
            and report['cohort_sha256'] == expectations['cohort_sha256']
            and report['policy'] == expectations['policy']
            and report['context'] == expectations['policy']['context_mode']
            and report['expected_skill_registry'] == expectations['policy']['skill_registry_sha256']
            and report['wam_manifest_sha256'] == expectations['wam_manifest_sha256']
            and report['max_ticks'] == expectations['max_ticks']
            and report['warmup_predictions'] == expectations['warmup_predictions']
            and report['source_sha256'] == expectations['evaluation_source_sha256']
            and report['independent_acceptance'] is False,
            'evaluation differs from the independently declared comparison')
    rows = report['episodes']
    observed = [case_key(c) for c in rows]
    require(len(set(observed)) == len(observed) and observed == keys[:len(observed)],
            'recorded cohort is reordered, duplicated or contains undeclared cases')
    require(report['expected_episodes'] == len(keys)
            and report['completed_episodes'] == len(rows)
            and all(type(c['strict_model_success']) is bool for c in rows)
            and report['strict_successes'] == sum(c['strict_model_success'] for c in rows),
            'reported case count or strict-success total differs')
    require(report['status'] != 'complete' or observed == keys,
            'complete evaluation omitted planned cases')
    output.mkdir(parents=True)
    results = []
    totals = Counter()
    tools = Counter()
    policy_sha = hashlib.sha256(canonical(expectations['policy'])).hexdigest()
    for index, planned_case in enumerate(planned['cases']):
        item = dict(planned_case, strict_success=False, audit_status='not_executed')
        results.append(item)
        if index >= len(rows):
            continue
        row = rows[index]
        relative = Path(planned_case['scene']) / f"{planned_case['task']}-seed{planned_case['seed']}"
        require(row['path'] == str(relative), 'episode path differs from the declared case')
        folder = evaluation / relative
        require(not folder.is_symlink(), 'episode cannot be a symlink')
        item['recorded_status'] = row['status']
        if not all((folder / n).is_file() for n in ('report.json', 'control.json', 'recording.npz', 'predictions.json')):
            require(row['status'] == 'failed' and not row['strict_model_success'],
                    'successful case is missing its physical recording')
            item.update(audit_status='missing_physical_recording', error=row.get('error'))
            continue
        controls = read(folder / 'control.json')
        if not controls:
            require(row['status'] == 'failed' and not row['strict_model_success'],
                    'successful case has no executed control ticks')
            item.update(audit_status='initialization_evidence_required', error=row.get('error'))
            continue
        case = dict(planned_case, profile=planned['profile'], mode='offline',
                    image_size=planned['image_size'], warmup_ticks=planned['warmup_ticks'],
                    contract=planned['comparison_contract'], hybrid_policy_sha256=policy_sha,
                    service_identity_sha256=expectations['policy']['expected_service_identity'],
                    execution_clock=expectations['execution_clock'])
        try:
            audit = audit_offline_hybrid_episode(folder, case, requests)
        except Exception as error:
            item.update(audit_status='failed', error=f'{type(error).__name__}: {error}')
            continue
        physical, sources = audit['physics'], audit['model_sources']
        require(physical['strict_success'] == row['strict_model_success'],
                'recomputed case success differs from evaluation summary')
        dest = output / f'case-{index:03d}.json'
        dest.write_text(json.dumps(audit, indent=2, allow_nan=False) + '\n')
        item.update(audit_status='complete', strict_success=physical['strict_success'],
                    audit_file=dest.name, audit_sha256=sha(dest))
        for field in ('exact_control_ticks', 'exact_prediction_source_observations'):
            totals[field] += physical[field]
        for field in ('exact_supervisor_requests', 'learned_core_inputs_verified',
                      'delegate_goal_mismatch_predictions'):
            totals[field] += sources[field]
        for field in ('exact_http_attempts', 'validated_supervisor_decisions',
                      'corrected_supervisor_decisions', 'rejected_tool_calls'):
            if field in sources:
                totals[field] += sources[field]
        totals['failed_unexecuted_requests'] += len(sources['failed_unexecuted_requests'])
        tools.update(sources['selected_tools'])
    require(sha(report_path) == report_sha, 'evaluation changed during audit')
    full = all(c['audit_status'] == 'complete' for c in results)
    result = dict(schema='microduck_offline_hybrid_cohort_audit_v1',
                  status='complete' if full else 'incomplete', expected_cases=len(keys),
                  recorded_cases=len(rows), audited_cases=sum(c['audit_status'] == 'complete' for c in results),
                  verified_strict_successes=sum(c['strict_success'] for c in results),
                  reported_strict_successes=report['strict_successes'], cases=results,
                  totals=dict(totals), selected_tools=dict(tools), all_planned_cases_accounted=True,
                  original_report_sha256=report_sha, formal_acceptance=False,
                  historical_runtime_identity_verified=False, neural_outputs_recomputed=False,
                  latency_remeasured=False, hardware_validated=False)
    result['verified_strict_success_rate'] = result['verified_strict_successes'] / len(keys) if full else None
    (output / 'report.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--requests', type=Path, required=True)
    parser.add_argument('--expectations', type=Path, required=True)
    parser.add_argument('--expectations-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(sha(args.expectations) == args.expectations_sha256, 'expected cohort or model changed')
    result = audit_hybrid_cohort(args.evaluation, args.requests, read(args.expectations), args.output)
    require(sha(args.expectations) == args.expectations_sha256, 'expectations changed during audit')
    print(json.dumps({k: result[k] for k in ('status', 'expected_cases', 'audited_cases', 'verified_strict_successes')}))
    if result['status'] != 'complete':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
