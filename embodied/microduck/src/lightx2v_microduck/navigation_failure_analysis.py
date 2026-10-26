"""Describe recorded navigation failures; do not infer perception or causal labels."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_bound(path, digest):
    require(sha(path) == digest, 'evidence changed: ' + str(path))
    return json.loads(Path(path).read_text())


def episode_patterns(report, controls):
    """Use the pinned visual_ball_stop_v2 task's distance and 50 Hz clock."""
    ev = report['evaluator']
    require(ev['success_protocol'] == 'visual_ball_stop_v2', 'unsupported success protocol')
    require(len(controls) == ev['tick'] and len(controls) > 0, 'incomplete control history')
    require([r['tick'] for r in controls] == list(range(len(controls))), 'control ticks are not contiguous')
    require(type(report['strict_model_success']) is bool, 'invalid success flag')
    fallback = sum(r['execution_source'] != 'navigation_model' for r in controls)
    require(fallback == report['fallback_control_ticks'], 'fallback accounting differs')
    position = ev['robot_position']
    distances = {color: math.hypot(ball[0]-position[0], ball[1]-position[1])-.035
                 for color, ball in ev['balls'].items()}
    require(all(math.isfinite(x) for x in distances.values()), 'nonfinite terminal position')
    distance = distances[ev['target_color']]
    tail = 0
    for control in reversed(controls):
        if control['skill'] != 'stand' or control['execution_source'] != 'navigation_model':
            break
        tail += 1
    in_band = .25 <= distance <= .40
    patterns = []
    if not report['strict_model_success']:
        if fallback:
            patterns.append('official_fallback')
        if ev['collision']:
            patterns.append('collision')
        if ev['ever_fallen']:
            patterns.append('fall')
        if tail >= 100 and not in_band:
            patterns.append('terminal_model_stand_outside_goal_band')
        if in_band and ev['goal_hold_ticks'] < 100:
            patterns.append('terminal_in_band_without_two_second_hold')
        if distance < .25:
            patterns.append('terminal_too_close')
        if distance > .40:
            patterns.append('terminal_too_far')
        if not patterns:
            patterns.append('other_recorded_task_failure')
    return dict(strict_success=report['strict_model_success'], failure_patterns=patterns,
                terminal_surface_distance_m=distance, terminal_nearest_ball=min(distances, key=distances.get),
                target_color=ev['target_color'], terminal_model_stand_seconds=tail/50,
                goal_hold_ticks=ev['goal_hold_ticks'], collision=ev['collision'], fallen=ev['ever_fallen'],
                fallback_ticks=fallback, path_m=ev['planar_path_length_m'], total_ticks=len(controls),
                model_stand_ticks=sum(r['skill']=='stand' and r['execution_source']=='navigation_model' for r in controls))


def analyze_candidate(protocol_path, protocol_sha256, assessment_path, assessment_sha256):
    """Summarize each scene from a completed independently replayed cohort."""
    protocol = read_bound(protocol_path, protocol_sha256)
    assessment = read_bound(assessment_path, assessment_sha256)
    require(assessment['status'] == 'complete', 'independent replay is unfinished')
    root = Path(protocol['output']).resolve()
    model = read_bound(root/'report.json', assessment['model_report_sha256'])
    require(model['status'] == 'complete' and model['protocol_sha256'] == protocol_sha256,
            'model cohort or protocol is not complete and bound')
    require(model['weights_sha256'] == assessment['checkpoint_sha256'] == protocol['weights_sha256'],
            'checkpoint identities differ')
    key = lambda row: (row['scene'], row['seed'], row['task'])
    expected = {key(row) for row in protocol['cases']}
    require(len(expected) == len(protocol['cases']), 'duplicate protocol case')
    indexed = {key(row): row for row in assessment['rows']}
    require(len(indexed) == len(assessment['rows']) == len(model['episodes']) == len(expected)
            and set(indexed) == expected == {key(row) for row in model['episodes']}, 'cohort differs')
    rows = []
    for episode in model['episodes']:
        replay = indexed[key(episode)]
        require(replay['input_files_sha256'] == episode['files'] and all(replay.get(flag) is True
                for flag in ('final_evaluator_exact', 'submitted_actions_exact', 'buffer_execution_exact')),
                'episode lacks independent replay evidence')
        folder = (root / episode['path']).resolve()
        folder.relative_to(root)
        report = read_bound(folder/'report.json', episode['files']['report.json'])
        controls = read_bound(folder/'control.json', episode['files']['control.json'])
        require((report['evaluator']['scene']['name'], report['seed'], report['task']) == key(episode)
                and report['policy']['artifact_sha256'] == model['weights_sha256']
                and report['mode'] == protocol['mode']
                and episode['contract'] in protocol['contracts']
                and report['action_contract'] == episode['contract']
                and report['strict_model_success'] == replay['strict_success'] == episode['strict_success'],
                'recorded episode identity or outcome differs')
        patterns = episode_patterns(report, controls)
        rows.append(dict(scene=episode['scene'], seed=episode['seed'], task=episode['task'],
                         report_sha256=episode['files']['report.json'], control_sha256=episode['files']['control.json'],
                         **patterns))
    groups = []
    for scene in sorted({row['scene'] for row in rows}):
        group = [row for row in rows if row['scene'] == scene]
        failed = [row for row in group if not row['strict_success']]
        groups.append(dict(scene=scene, cases=len(group), successes=len(group)-len(failed),
            failure_pattern_counts=dict(sorted(Counter(tag for row in failed for tag in row['failure_patterns']).items())),
            median_terminal_surface_distance_m=statistics.median(row['terminal_surface_distance_m'] for row in group),
            median_terminal_model_stand_seconds=statistics.median(row['terminal_model_stand_seconds'] for row in group),
            total_control_ticks=sum(row['total_ticks'] for row in group),
            model_stand_ticks=sum(row['model_stand_ticks'] for row in group),
            fallback_ticks=sum(row['fallback_ticks'] for row in group)))
    return dict(schema='microduck_navigation_failure_patterns_v1', checkpoint_sha256=model['weights_sha256'],
                mode=protocol['mode'], protocol_sha256=protocol_sha256, assessment_sha256=assessment_sha256,
                model_report_sha256=assessment['model_report_sha256'], groups=groups, rows=rows,
                formal_acceptance=False, model_promoted=False, causal_conclusion=False,
                interpretation_limits=[
                    'Patterns overlap; their counts must not be summed as exclusive failure classes.',
                    'Terminal distance describes the final state, not whether the target was ever seen or approached.',
                    'Nearest ball is geometric evidence, not a recognition or instruction-understanding label.',
                    'Long terminal standing outside the goal band does not establish its training or perception cause.',
                    'This summarizes recorded behavior; the independent replay remains a separate prerequisite.'])
