"""Validate main-thread realtime traces and decompose long execution gaps.

This is timing evidence, never a capability or performance-improvement gate.
Worker inference overlaps these stages and must not be added to their sum.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

BOUNDARIES = ('capture_started_ns', 'capture_finished_ns', 'recording_finished_ns',
              'request_submission_finished_ns', 'admission_finished_ns',
              'consume_finished_ns', 'physics_finished_ns', 'pacing_finished_ns')
STAGES = ('capture', 'recording', 'request_submission', 'admission', 'consume', 'physics', 'pacing')


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def audit(rows, controls, report):
    require(report.get('mode') == 'realtime', 'requires realtime execution clocks')
    metadata = report.get('stage_timing_recording', {})
    require(metadata.get('schema') == 'microduck_navigation_main_thread_stages_v1', 'missing stage schema')
    require(metadata.get('clock') == 'monotonic_ns', 'unexpected clock')
    require(metadata.get('serialization_outside_measurement') is True, 'serialization scope missing')
    require(len(rows) == len(controls) == metadata.get('records') and len(rows) > 0, 'incomplete stage/control trace')
    durations = {name: [] for name in STAGES}
    gaps = []
    for index, (row, control) in enumerate(zip(rows, controls)):
        require(type(row.get('tick')) is int and row['tick'] == control.get('tick') == index,
                'missing, duplicate or out-of-order control tick')
        values = [row.get(key) for key in BOUNDARIES]
        require(all(type(v) is int and v >= 0 for v in values), 'invalid monotonic boundary')
        require(all(a <= b for a, b in zip(values, values[1:])), 'stage clocks reversed')
        capture, execution = control.get('observation_capture_monotonic_ns'), control.get('execution_monotonic_ns')
        require(type(capture) is int and type(execution) is int, 'missing execution or observation clock')
        require(values[0] <= capture <= values[1], 'observation timestamp outside capture interval')
        require(values[4] <= execution <= values[5], 'execution timestamp outside consume interval')
        for name, start, end in zip(STAGES, values, values[1:]):
            durations[name].append(end-start)
        if index:
            previous = rows[index-1]
            previous_execution = controls[index-1]['execution_monotonic_ns']
            require(previous['pacing_finished_ns'] <= values[0], 'main-thread intervals overlap')
            components = dict(
                previous_consume_tail=previous['consume_finished_ns']-previous_execution,
                previous_physics=previous['physics_finished_ns']-previous['consume_finished_ns'],
                previous_pacing=previous['pacing_finished_ns']-previous['physics_finished_ns'],
                between_ticks=values[0]-previous['pacing_finished_ns'],
                capture=values[1]-values[0], recording=values[2]-values[1],
                request_submission=values[3]-values[2], admission=values[4]-values[3],
                current_consume_prefix=execution-values[4])
            require(all(v >= 0 for v in components.values()), 'negative execution gap component')
            gap = execution-previous_execution
            require(sum(components.values()) == gap, 'execution gap does not reconcile')
            gaps.append(dict(tick=index, execution_gap_ms=gap/1e6,
                             components_ms={k:v/1e6 for k,v in components.items()},
                             largest_component=max(components, key=components.get)))
    summary = {name: dict(p50_ms=float(np.percentile(v,50))/1e6,
                          p95_ms=float(np.percentile(v,95))/1e6,
                          maximum_ms=max(v)/1e6) for name,v in durations.items()}
    return dict(verified_control_ticks=len(rows), stage_durations=summary,
                longest_execution_gaps=sorted(gaps,key=lambda r:r['execution_gap_ms'],reverse=True)[:10],
                all_gap_components_reconciled=True, capability_claim=False,
                limitations=['Intervals include instrumentation and OS scheduling delays.',
                             'Capture includes all env.observe work; physics includes the local locomotion policy.',
                             'Request submission includes preparation and enqueue; inference runs concurrently.',
                             'A long stage localizes wall time but does not prove its CPU/GPU root cause.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    require(not args.output.exists(),'output already exists')
    data={name:json.loads((args.episode/name).read_text()) for name in ('stage_timings.json','control.json','report.json')}
    result=audit(data['stage_timings.json'],data['control.json'],data['report.json'])
    result.update(status='complete',files_sha256={name:hashlib.sha256((args.episode/name).read_bytes()).hexdigest() for name in data},
                  assessor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with args.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(verified_control_ticks=result['verified_control_ticks'],capability_claim=False)))

if __name__=='__main__':main()
