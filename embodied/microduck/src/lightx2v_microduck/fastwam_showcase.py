"""Reviewed capability recordings, separate from live training snapshots."""
import hashlib
import json
import os
from pathlib import Path


def directory():
    return Path(os.environ.get('MICRODUCK_FASTWAM_SHOWCASE',
                Path(__file__).resolve().parents[4]/'artifacts/navigation/capability-showcase-v1'))


def manifest():
    root = directory()
    value = json.loads((root/'manifest.json').read_text())
    if value.get('schema') != 'fastwam_capability_showcase_v1':
        raise ValueError('unsupported capability manifest')
    assessment = checked_file(root, value['assessment'])
    report = json.loads(assessment)
    if (report.get('status') != 'complete' or report.get('successes', {}).get('optimized') != value['passed']
            or report.get('cases_per_variant') != value['total']
            or report.get('distinct_development_layouts') != value['layouts']
            or report.get('checkpoint_sha256') != value['checkpoint_sha256']
            or report.get('formal_acceptance') is not False
            or value.get('formal_acceptance') is not False or value.get('hardware_validated') is not False):
        raise ValueError('capability summary differs from its assessment')
    identifiers = {x['id'] for x in value['clips']}
    for group in value.get('obstacle_groups', []):
        _check_obstacle_group(root, group, value['checkpoint_sha256'])
        if group['id'] in identifiers or any(c['id'] in identifiers for c in group['clips']):
            raise ValueError('duplicate capability identifier')
        identifiers.add(group['id'])
        identifiers.update(c['id'] for c in group['clips'])
    for group in value.get('retest_groups', []):
        _check_retest_group(root, group, value['checkpoint_sha256'])
        if group['id'] in identifiers or any(c['id'] in identifiers for c in group['clips']):
            raise ValueError('duplicate retest identifier')
        identifiers.add(group['id'])
        identifiers.update(c['id'] for c in group['clips'])
    return value


def _check_retest_group(root, group, checkpoint):
    if group.get('id') == 'no-age-split':
        return _check_no_age_retest_group(root, group, checkpoint)
    report = json.loads(checked_file(root, group['evaluation']))
    analysis = json.loads(checked_file(root, group['assessment']))
    audit = json.loads(checked_file(root, group['clock_audit']))
    if (group.get('id') != 'ttl1000-split' or group.get('mode') != 'realtime'
            or group.get('scene_id') != 'room_clear' or group.get('ttl_ms') != 1000
            or group.get('deployment') != 'local_render_remote_model'
            or group.get('formal_acceptance') is not False or group.get('hardware_validated') is not False
            or group.get('checkpoint_sha256') != checkpoint
            or report.get('status') != 'complete' or report.get('clock_expiry_verified') is not True
            or report.get('checkpoint_sha256') != checkpoint or report.get('deployment') != group['deployment']
            or report['runtime_provenance']['max_wall_age_s'] != 1.0
            or analysis.get('status') != 'complete' or analysis.get('formal_acceptance') is not False
            or analysis.get('report_sha256') != group['evaluation']['sha256']
            or report['execution_clock_audit']['sha256'] != group['clock_audit']['sha256']
            or audit.get('clock_expiry_verified') is not True or audit.get('episodes') != 60):
        raise ValueError('retest scope or audit differs from evidence')
    cases = {(e['task'], e['seed']): e for e in report['episodes']}
    expected = {(f'find_{c}_ball', s) for c in ('red','blue','yellow') for s in range(2000,2020)}
    scores = [dict(color=c,label=cn,
        passed=sum(e['realtime_success'] is True for e in cases.values() if e['task']==f'find_{c}_ball'),total=20)
        for c,cn in [('red','红'),('blue','蓝'),('yellow','黄')]]
    if (set(cases) != expected or len(report['episodes']) != 60 or group.get('total') != 60
            or group.get('layouts') != 20 or group.get('color_scores') != scores
            or group.get('passed') != sum(e['realtime_success'] is True for e in cases.values())
            or group['passed'] != analysis.get('realtime_successes')
            or len({c['id'] for c in group['clips']}) != len(group['clips'])):
        raise ValueError('retest counts differ from complete development cohort')
    for clip in group['clips']:
        episode = json.loads(checked_file(root, clip['report']))
        if (episode != cases.get((clip['task'],clip['seed']))
                or clip.get('passed') != episode['realtime_success']
                or clip.get('fallback_ticks') != episode['fallback_control_ticks']
                or clip.get('checkpoint_sha256') != checkpoint or clip.get('ttl_ms') != 1000
                or clip.get('deployment') != group['deployment'] or clip.get('formal_acceptance') is not False):
            raise ValueError('retest clip differs from recorded episode')


def _check_no_age_retest_group(root, group, checkpoint):
    """No-age simulation evidence must never satisfy the normal TTL verifier."""
    report = json.loads(checked_file(root, group['evaluation']))
    analysis = json.loads(checked_file(root, group['assessment']))
    if (group.get('mode') != 'realtime' or group.get('scene_id') != 'room_clear'
            or group.get('ttl_ms', 'missing') is not None
            or group.get('deployment') != 'local_render_remote_model'
            or group.get('formal_acceptance') is not False or group.get('hardware_validated') is not False
            or group.get('checkpoint_sha256') != checkpoint or report.get('checkpoint_sha256') != checkpoint
            or report.get('status') != 'complete' or report.get('stage') != 'complete'
            or report.get('no_age_execution_verified') is not True
            or report.get('deployment') != group['deployment']
            or report.get('training_allowed') is not False
            or any(report.get(k) is not False for k in ('cache_prompt', 'compiled', 'action_context_cache', 'action_cuda_graph'))
            or report.get('inference_steps') != 20
            or analysis.get('status') != 'complete' or analysis.get('formal_acceptance') is not False
            or analysis.get('report_sha256') != group['evaluation']['sha256']):
        raise ValueError('no-age retest scope differs from verified evidence')
    expected = {(f'find_{c}_ball', s) for c in ('red', 'blue', 'yellow') for s in range(2000, 2020)}
    cases = {(e['task'], e['seed']): e for e in report['episodes']}
    rows = {(r['task'], r['seed']): r for r in analysis['rows']}
    if (set(cases) != expected or set(rows) != expected
            or len(report['episodes']) != 60 or len(analysis['rows']) != 60
            or group.get('total') != 60 or group.get('layouts') != 20):
        raise ValueError('incomplete no-age cohort')
    for key, episode in cases.items():
        clock, audit = episode['execution_clock'], rows[key]['audit']
        if (clock.get('max_age_ticks', 'missing') is not None
                or clock.get('max_wall_age_ns', 'missing') is not None
                or clock.get('observation_age_rejection_enabled') is not False
                or clock.get('source_capture_timestamp_preserved') is not True
                or audit.get('no_age_limit_execution_verified') is not True
                or audit.get('fallback_ticks') != episode['fallback_control_ticks']
                or audit.get('verified_ticks') != episode['evaluator']['tick']
                or rows[key]['realtime_success'] != episode['realtime_success']):
            raise ValueError('no-age execution audit differs from episode')
    scores = [dict(color=c, label=cn,
        passed=sum(e['realtime_success'] is True for e in cases.values() if e['task'] == f'find_{c}_ball'), total=20)
        for c, cn in [('red', '红'), ('blue', '蓝'), ('yellow', '黄')]]
    if (group.get('color_scores') != scores
            or group.get('passed') != sum(e['realtime_success'] is True for e in cases.values())
            or group['passed'] != analysis.get('realtime_successes')
            or len({c['id'] for c in group['clips']}) != len(group['clips'])):
        raise ValueError('no-age summary differs from cohort')
    for clip in group['clips']:
        episode = json.loads(checked_file(root, clip['report']))
        if (episode != cases.get((clip['task'], clip['seed']))
                or clip.get('passed') != episode['realtime_success']
                or clip.get('fallback_ticks') != episode['fallback_control_ticks']
                or clip.get('checkpoint_sha256') != checkpoint
                or clip.get('ttl_ms', 'missing') is not None
                or clip.get('formal_acceptance') is not False):
            raise ValueError('no-age recording differs from its episode')


def _check_obstacle_group(root, group, checkpoint):
    """Keep obstacle development evidence separate from accepted capabilities."""
    expected = {'single-obstacle': ('offline', 'room_curriculum_one'),
                'random-obstacles': ('realtime', 'room_randomized')}
    if (group.get('id') not in expected
            or (group.get('mode'), group.get('scene_id')) != expected[group['id']]
            or group.get('formal_acceptance') is not False
            or group.get('hardware_validated') is not False
            or group.get('checkpoint_sha256') != checkpoint):
        raise ValueError('invalid obstacle evidence scope')
    evaluation = json.loads(checked_file(root, group['evaluation']))
    assessment = json.loads(checked_file(root, group['assessment']))
    if (evaluation.get('status') != 'complete' or assessment.get('status') != 'complete'
            or evaluation.get('checkpoint_sha256') != checkpoint
            or assessment.get('checkpoint_sha256') != checkpoint
            or assessment.get('formal_acceptance') is not False
            or evaluation.get('mode') != group['mode'] or evaluation.get('scene') != group['scene_id']):
        raise ValueError('obstacle summary differs from verified model evidence')
    episodes = evaluation['episodes']
    key = 'realtime_success' if group['mode'] == 'realtime' else 'strict_model_success'
    cases = {(e['task'], e['seed']): e for e in episodes}
    color_scores = [dict(color=c, label=cn,
        passed=sum(e[key] is True for e in episodes if e['task'] == f'find_{c}_ball'),
        total=sum(e['task'] == f'find_{c}_ball' for e in episodes))
        for c, cn in [('red', '红'), ('blue', '蓝'), ('yellow', '黄')]]
    if (len(cases) != len(episodes) or len(episodes) != group['total']
            or sum(e[key] is True for e in episodes) != group['passed']
            or len({e['seed'] for e in episodes}) != group['layouts']
            or group['color_scores'] != color_scores):
        raise ValueError('obstacle counts differ from complete cohort')
    if group['id'] == 'single-obstacle':
        rows = [r for r in assessment['rows'] if r['scene'] == group['scene_id']
                and r['contract'] == 'microduck_navigation_v1']
        if (len(rows) != len(cases) or {(r['task'], r['seed']) for r in rows} != set(cases)
                or any(r['strict_success'] != cases[(r['task'], r['seed'])][key] for r in rows)):
            raise ValueError('single-obstacle scores differ from independent replay audit')
    elif (assessment['episodes'] != group['total'] or assessment['realtime_successes'] != group['passed']):
        raise ValueError('random-obstacle scores differ from independent audit')
    if len({c['id'] for c in group['clips']}) != len(group['clips']):
        raise ValueError('duplicate obstacle clip')
    for clip in group['clips']:
        episode = json.loads(checked_file(root, clip['report']))
        if (episode != cases.get((clip['task'], clip['seed']))
                or clip.get('passed') != episode[key]
                or clip.get('mode') != group['mode']
                or clip.get('scene_id') != group['scene_id']
                or clip.get('checkpoint_sha256') != checkpoint
                or clip.get('formal_acceptance') is not False
                or clip.get('fallback_ticks') != episode['fallback_control_ticks']):
            raise ValueError('obstacle clip differs from its recorded episode')


def checked_file(root, ref):
    name = ref['file']
    if Path(name).name != name or name in ('.', '..'):
        raise ValueError('invalid capability filename')
    p = (root/name).resolve()
    if not p.is_relative_to(root.resolve()):
        raise ValueError('capability file escapes root')
    data = p.read_bytes()
    if hashlib.sha256(data).hexdigest() != ref['sha256']:
        raise ValueError('capability artifact checksum mismatch')
    return data


def artifact(identifier, kind):
    data = manifest()
    if identifier == 'assessment' and kind == 'report':
        ref = data['assessment']
    elif kind == 'report' and any(g['id'] == identifier for g in data.get('obstacle_groups', []) + data.get('retest_groups', [])):
        ref = next(g['evaluation'] for g in data.get('obstacle_groups', []) + data.get('retest_groups', []) if g['id'] == identifier)
    else:
        clips = data['clips'] + [c for g in data.get('obstacle_groups', []) + data.get('retest_groups', []) for c in g['clips']]
        clip = next((x for x in clips if x['id'] == identifier), None)
        if clip is None or kind not in ('video','poster','report'):
            raise ValueError('unknown capability artifact')
        ref = clip[kind]
    content = checked_file(directory(), ref)
    return content, {'video':'video/mp4','poster':'image/jpeg','report':'application/json'}[kind]
