"""Reconcile camera timing with recorded physics ticks and policy sources.

This is independent of model inference and task scoring. Dropped, failed and
cancelled frames remain outcomes; a valid trace does not imply task success.
"""
from collections import Counter
import math
from pathlib import Path

from .navigation_outcomes import read, require, sha


def integer(value, label, minimum=0):
    require(type(value) is int and value >= minimum, 'invalid '+label)
    return value


def percentile(values, quantile=.95):
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered)-1)*quantile
    low, high = math.floor(rank), math.ceil(rank)
    return ordered[low] + (ordered[high]-ordered[low])*(rank-low)


def audit_camera_timing(episode, controls, predictions, receipts):
    camera = episode.get('camera_delivery') or {}
    require(episode['mode'] == 'realtime' and camera.get('schema') == 'microduck_snapshot_camera_v1'
            and camera.get('backend') == 'osmesa' and camera.get('capture_timestamp_preserved') is True,
            'snapshot camera timing contract missing')
    require(all(isinstance(rows, list) for rows in (controls, predictions, receipts)), 'invalid recording lists')
    count = len(controls)
    require(count > 0 and episode['evaluator']['tick'] == count, 'control trace is empty or incomplete')
    previous_execution = 0
    for tick, control in enumerate(controls):
        capture = integer(control['observation_capture_monotonic_ns'], 'control capture', 1)
        execution = integer(control['execution_monotonic_ns'], 'control execution', 1)
        require(control['tick'] == tick and previous_execution <= capture <= execution,
                'control order or clock differs')
        previous_execution = execution
    require(type(camera.get('initial_synchronous_frame')) is bool, 'startup camera mode missing')
    initial = camera['initial_synchronous_frame']
    expected = set(range(5 if initial else 0, count, 5))
    cancelled = camera['cancelled_ticks']
    require(isinstance(cancelled, list) and all(type(t) is int for t in cancelled)
            and len(set(cancelled)) == len(cancelled), 'cancelled frame inventory differs')
    by_tick, delivery_order, delivery_times = {}, [], []
    latency, render_latency = [], []
    for row in receipts:
        tick = integer(row['source_tick'], 'camera source tick')
        require(tick in expected and tick not in by_tick, 'duplicate or unexpected camera capture')
        capture = integer(row['capture_ns'], 'camera capture', 1)
        require(capture == controls[tick]['observation_capture_monotonic_ns'],
                'camera timestamp differs from its physics capture')
        status = row['status']
        require(status in ('delivered', 'error', 'dropped_busy'), 'unknown camera outcome')
        if status == 'dropped_busy':
            require(not any(k in row for k in ('delivered_ns','ready_ns','delivery_phase')),
                    'dropped camera frame was delivered')
        else:
            times = [integer(row[k], k, 1) for k in ('capture_ns','copy_started_ns','copy_finished_ns',
                      'render_started_ns','ready_ns','delivered_ns')]
            require(times == sorted(times), 'camera timing order differs')
            require(type(row.get('worker')) is int and 0 <= row['worker'] < camera['workers'], 'camera worker differs')
            require(row.get('delivery_phase') in ('measured','after_measurement'), 'camera delivery phase missing')
            if row['delivery_phase'] == 'after_measurement':
                require(row['delivered_ns'] >= controls[-1]['execution_monotonic_ns'],
                        'post-measurement frame predates final control')
            else:
                require(row['delivered_ns'] <= controls[-1]['execution_monotonic_ns'],
                        'measured frame arrived after final control')
            if status == 'error':
                require(isinstance(row.get('error'),str) and row['error'], 'camera failure lacks a reason')
            delivery_order.append(tick); delivery_times.append(row['delivered_ns'])
            if status == 'delivered' and row['delivery_phase'] == 'measured':
                latency.append((row['delivered_ns']-capture)/1e6)
                render_latency.append((row['ready_ns']-row['render_started_ns'])/1e6)
        by_tick[tick] = row
    require(set(by_tick).isdisjoint(cancelled) and set(by_tick) | set(cancelled) == expected,
            'camera capture inventory omits or repeats outcomes')
    require(delivery_order == sorted(delivery_order) and delivery_times == sorted(delivery_times),
            'camera delivery was not ordered by capture')
    outcomes = Counter(row['status'] for row in receipts)
    require(camera['delivered_frames'] == outcomes['delivered'] and camera['failed_frames'] == outcomes['error']
            and camera['dropped_frames'] == outcomes['dropped_busy']
            and camera['expected_frames'] == (count+4)//5, 'camera summary disagrees with receipts')
    cadence = (outcomes['delivered'] + int(initial) == camera['expected_frames']
               and not outcomes['error'] and not outcomes['dropped_busy'] and not cancelled
               and not camera['worker_errors'] and camera['workers_stopped'])
    require(camera['complete_camera_cadence'] is bool(cadence), 'camera cadence claim differs from actual outcomes')
    source_times, executable_sources, ages, model_durations = {}, set(), [], []
    model_replies = preparation_errors = camera_errors = inference_errors = 0
    for prediction in predictions:
        tick = integer(prediction['source_tick'], 'prediction source tick')
        capture = integer(prediction['source_monotonic_ns'], 'prediction capture', 1)
        phase = prediction['phase']
        image_identity = prediction['camera_tick'] is None if phase == 'camera' else prediction['camera_tick'] == tick
        require(image_identity and prediction['source_episode_id'] == episode['evaluator']['episode_id'],
                'prediction camera or episode identity differs')
        if initial and tick == 0 and phase == 'startup':
            require(all(row['observation_eligible'] is False and row['finished_ns'] <= capture
                        for row in camera['initialization'])
                    and capture <= controls[0]['observation_capture_monotonic_ns'],
                    'startup uses an initialization frame or future capture')
        else:
            require(tick in by_tick, 'prediction used an absent or cancelled frame')
            row = by_tick[tick]
            require(row['capture_ns'] == capture, 'prediction retimestamped the camera')
            if phase == 'camera':
                require(row['status'] == 'error' and prediction['submitted_actions'] is None
                        and prediction['accepted'] is False, 'camera failure submitted model actions')
                camera_errors += 1
                continue
            require(row['status'] == 'delivered' and row['delivery_phase'] == 'measured',
                    'model used an unavailable or post-measurement image')
            receipt_tick = integer(prediction['tick'], 'prediction receipt tick')
            require(receipt_tick < count and row['delivered_ns'] <= controls[receipt_tick]['execution_monotonic_ns'],
                    'model result precedes camera delivery')
            if prediction['admission_monotonic_ns'] is not None:
                require(row['delivered_ns'] <= prediction['admission_monotonic_ns'], 'action admission precedes image delivery')
        require(phase in ('startup','realtime','preparation'), 'unexpected model receipt phase')
        require(tick not in source_times or source_times[tick] == capture, 'one source tick has conflicting times')
        source_times[tick] = capture
        if prediction['accepted'] is True:
            require(phase in ('startup', 'realtime') and prediction['error'] is None
                    and isinstance(prediction['submitted_actions'], list) and prediction['submitted_actions'],
                    'failed observation produced accepted actions')
            admission = integer(prediction['admission_monotonic_ns'], 'action admission', 1)
            require(capture <= admission <= controls[integer(prediction['tick'], 'receipt tick')]['execution_monotonic_ns'],
                    'accepted action admission is outside its receipt tick')
            executable_sources.add(tick)
        if phase == 'preparation':
            preparation_errors += 1
        elif phase == 'realtime':
            model_replies += 1
            if prediction['error'] is not None:
                inference_errors += 1
            else:
                duration = prediction['duration_ms']
                require(type(duration) in (int,float) and math.isfinite(duration) and duration >= 0,
                        'invalid inference duration')
                model_durations.append(duration)
                admission = integer(prediction['admission_monotonic_ns'], 'action admission', 1)
                require(admission >= capture, 'action source is from the future')
                ages.append((admission-capture)/1e6)
    model_ticks = 0
    for control in controls:
        if control['execution_source'] != 'navigation_model':
            continue
        tick = control['source_tick']
        require(tick in executable_sources and control['checked_plan_source_monotonic_ns'] == source_times[tick],
                'executed action source does not match recorded camera')
        age = (control['execution_monotonic_ns']-source_times[tick])/1e6
        require(math.isclose(control['observation_age_ms'],age,rel_tol=1e-12,abs_tol=1e-6),
                'executed action age differs from camera capture')
        model_ticks += 1
    from .navigation_request_timing import audit_request_timing
    request_timing = audit_request_timing(episode, controls, predictions, receipts)
    return dict(schema='microduck_camera_timing_audit_v1',status='complete',control_ticks=count,
        expected_camera_frames=camera['expected_frames'],delivered_frames=outcomes['delivered'],
        dropped_frames=outcomes['dropped_busy'],failed_frames=outcomes['error'],cancelled_frames=len(cancelled),
        after_measurement_frames=sum(r.get('delivery_phase')=='after_measurement' for r in receipts),
        complete_camera_cadence=bool(cadence),model_replies=model_replies,inference_errors=inference_errors,
        preparation_errors=preparation_errors,camera_error_events=camera_errors,model_control_ticks=model_ticks,
        measured_camera_delivery_p95_ms=percentile(latency),measured_camera_render_p95_ms=percentile(render_latency),
        successful_inference_p95_ms=percentile(model_durations),admission_observation_age_p95_ms=percentile(ages),
        capture_timestamps_verified=True,post_measurement_images_excluded=True,
        model_inference_replayed=False,task_success_verified=False,formal_acceptance=False,
        request_timing=request_timing,
        limitation=('Legacy model-call timestamps are absent; no additive latency decomposition.'
                    if request_timing['status'] == 'unavailable' else
                    'No realtime replies were consumed; request latency is unmeasured.'
                    if request_timing['status'] == 'no_samples' else
                    'Per-request stages reconcile; do not add stage percentiles or infer unconsumed requests.'))


def audit_recording(folder, *, sealed_files=None):
    folder=Path(folder)
    paths=[folder/name for name in ('report.json','control.json','predictions.json','camera_delivery.json')]
    hashes={str(path):sha(path) for path in paths}
    if sealed_files is not None:
        require(all(sealed_files.get(name)==digest for name,digest in hashes.items()), 'unsealed or modified camera timing input')
    result=audit_camera_timing(*(read(path) for path in paths))
    require(all(sha(path)==hashes[str(path)] for path in paths), 'camera timing inputs changed during audit')
    result['input_files_sha256']=hashes
    return result


def audit_job(spec):
    """Audit every specified case, including task failures and invalid traces.

    Structural cohort errors fail the job. Per-case errors remain in the result
    and denominator, rather than silently reducing the set being verified.
    """
    root = Path(spec['report']).parent
    require(root.is_absolute(), 'absolute recording location required')
    expected = {(task, seed) for task in spec['tasks'] for seed in spec['seeds']}
    require(len(expected) == len(spec['tasks'])*len(spec['seeds']) == spec['expected_cases'] > 0,
            'duplicate or empty camera audit cohort')
    digest = sha(spec['report'])
    require(spec['files'].get(str(root/'report.json')) == digest, 'unsealed camera audit job')
    report = read(spec['report'])
    require(report['mode'] == 'realtime' and report['checkpoint_sha256'] == spec['weights_sha256']
            and report['completed_episodes'] == len(report['episodes']) == len(expected),
            'camera audit job incomplete or checkpoint differs')
    cases = [(row['task'], row['seed']) for row in report['episodes']]
    require(len(set(cases)) == len(cases) and set(cases) == expected, 'camera audit cohort differs')
    rows = []
    for episode in report['episodes']:
        task, seed = episode['task'], episode['seed']
        row = dict(task=task, seed=seed, verified=False,
                   recorded_strict_success=episode['strict_model_success'])
        folder = root/(task+'-seed'+str(seed))
        try:
            require(folder.resolve().is_relative_to(root.resolve()) and not folder.is_symlink(),
                    'camera recording escapes job directory')
            require(read(folder/'report.json') == episode, 'aggregate and per-case camera reports disagree')
            require(episode['policy']['artifact_sha256'] == spec['weights_sha256']
                    and episode['policy']['contract'] == spec['contract']
                    and episode['evaluator']['scene']['name'] == spec['scene']
                    and episode['execution_clock'] == spec['execution_clock'], 'camera episode contract differs')
            row['audit'] = audit_recording(folder, sealed_files=spec['files'])
            row['verified'] = True
        except Exception as exc:
            row['error'] = type(exc).__name__+': '+str(exc)
        rows.append(row)
    require(sha(spec['report']) == digest, 'camera audit job changed during analysis')
    verified = sum(row['verified'] for row in rows)
    return dict(schema='microduck_camera_timing_job_audit_v1',
        status='complete' if verified == len(expected) else 'incomplete',
        job=spec['job'], expected_cases=len(expected), attempted_cases=len(rows),
        verified_cases=verified, failed_verification_cases=len(rows)-verified, cases=rows,
        recorded_task_failures=sum(not row['recorded_strict_success'] for row in rows),
        control_ticks=sum(row['audit']['control_ticks'] for row in rows if row['verified']),
        input_job_sha256=digest, model_inference_replayed=False, task_success_verified=False,
        formal_acceptance=False, fresh_realtime_measurement=False)
