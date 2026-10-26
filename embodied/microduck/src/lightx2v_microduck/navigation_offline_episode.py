"""Versioned offline recorder with explicit prediction errors and local fallback.

Normal executions retain the existing 50 Hz / 10 Hz navigation contract.
Prediction exceptions do not pause a realtime world here: this entrypoint is
strictly offline. They leave the previous bounded plan intact; when it expires,
the same official-controller fallback is recorded. Initialization/physics or
serialization failures remain incomplete and cannot become accepted trials.
"""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .navigation import DECISION_TICKS, NAVIGATION_CONTRACT, NavigationBuffer, NavigationChunk, intent_from_action
from .wam_data import write_json


def run_offline_navigation_episode(env, policy, output, *, task='find_red_ball', seed=0,
        realtime=False, max_ticks=1500, warmup_ticks=50, warmup_predictions=1,
        record_frames=True, record_predictions=True):
    if (realtime is not False or record_frames is not True or record_predictions is not True
            or type(max_ticks) is not int or not 1 <= max_ticks <= 1500
            or type(warmup_ticks) is not int or warmup_ticks < 0
            or type(warmup_predictions) is not int or warmup_predictions != 1):
        raise ValueError('offline recorder needs explicit sensors, one warmup and a bounded control clock')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    buffer=NavigationBuffer(max_age_ticks=25,max_wall_age_s=1e9,
                            contract=getattr(policy,'metadata',{}).get('contract',NAVIGATION_CONTRACT))
    decisions,predictions,controls,frames,frame_ticks,qpos,physics_ticks=[],[],[],[],[],[],[]
    latencies,ages,camera_ms=[],[],[]
    warmup_ms=[]
    phase='initialization'
    started=time.monotonic()

    def capture():
        before=time.monotonic_ns();observation=env.observe()
        if observation.rgb is not None:camera_ms.append((time.monotonic_ns()-before)/1e6)
        return replace(observation,monotonic_ns=before)

    def predict(observation, phase):
        before=time.monotonic();result=None;error=None;admission=None;accepted=False
        try:
            result=policy.predict(observation)
            if not isinstance(result,NavigationChunk):
                raise TypeError('navigation policy must return NavigationChunk')
            if (result.episode_id,result.source_tick,result.source_monotonic_ns) != (
                    observation.episode_id,observation.tick,observation.monotonic_ns):
                raise ValueError('offline response does not match its observation')
        except Exception as exception:
            result=None;error=type(exception).__name__+': '+str(exception)
        duration_ms=(time.monotonic()-before)*1000
        if result is not None:
            admission=env.observe() if phase=='startup' else observation
            accepted=buffer.submit(result,admission)
        receipt=dict(tick=env.tick,accepted=accepted,error=error,
            reason='prediction_error' if result is None else buffer.last_reason,
            source_tick=observation.tick,phase=phase,
            admission_monotonic_ns=None if admission is None else admission.monotonic_ns)
        decisions.append(receipt)
        predictions.append(dict(receipt,duration_ms=duration_ms,
            source_episode_id=observation.episode_id,source_monotonic_ns=observation.monotonic_ns,
            camera_tick=observation.camera_tick,observed_instruction=observation.instruction,
            observed_proprio48=observation.state[:48].copy(),
            valid_until_tick=None if result is None else result.valid_until_tick,
            action_interval_ticks=None if result is None else result.action_interval_ticks,
            action_contract=None if result is None else result.contract,
            submitted_actions=None if result is None else result.actions,
            diagnostics=None if result is None else result.diagnostics))
        return duration_ms

    try:
        env.render_enabled=False;env.reset(seed,task)
        for _ in range(warmup_ticks):
            env.step(intent_from_action([0,0,0,0,1],episode_id=env.episode_id,tick=env.tick))
        env.begin_task_at_current_pose();env.render_enabled=True
        observation=capture();policy.reset(observation.episode_id);buffer.reset(observation.episode_id)
        phase='warmup'
        before=time.monotonic();policy.predict(observation)
        warmup_ms.append((time.monotonic()-before)*1000)
        policy.reset(observation.episode_id);observation=capture()
        phase='startup';startup_ms=predict(observation,'startup')
        started=time.monotonic();phase='control'
        while not env.terminated and env.tick < max_ticks and not (output/'cancel.request').exists():
            env.render_enabled=env.tick % DECISION_TICKS == 0
            observation=capture()
            if observation.rgb is not None and observation.camera_tick==env.tick:
                frames.append(observation.rgb.copy());frame_ticks.append(env.tick)
            qpos.append(env.physics.data.qpos.copy());physics_ticks.append(env.tick)
            if env.tick % DECISION_TICKS==0 and env.tick != 0:
                latencies.append(predict(observation,'offline'))
            checked=buffer.chunk
            intent,evidence=buffer.consume(observation)
            if 'observation_age_ms' in evidence:ages.append(evidence['observation_age_ms'])
            controls.append(dict(tick=env.tick,skill=intent.skill,command=intent.command.tolist(),
                execution_monotonic_ns=observation.monotonic_ns,
                observation_capture_monotonic_ns=observation.monotonic_ns,
                checked_plan_source_monotonic_ns=None if checked is None else checked.source_monotonic_ns,
                **evidence))
            env.render_enabled=False;env.step(intent)
        elapsed=time.monotonic()-started;final=env.evaluator_state()
        cancelled=(output/'cancel.request').exists()
        hazards=(buffer.fallback_ticks or cancelled or final.get('collision') or final.get('ever_fallen')
                 or final.get('fault') or final.get('recovery_events',0))
        report=dict(mode='offline',task=task,seed=seed,action_contract=buffer.contract,
            initialization='reset_and_warmup',policy=getattr(policy,'metadata',{}),
            task_success=bool(final['success']),strict_model_success=bool(final['success'] and not hazards),
            cancelled=cancelled,realtime_timing_qualified=None,realtime_success=None,timing_gate=None,
            fallback_control_ticks=buffer.fallback_ticks,rejected_plans=buffer.rejected,
            execution_clock=dict(schema='microduck_navigation_execution_clock_v2',
                expiry_basis='offline_observation_clock',max_age_ticks=25,max_wall_age_ns=10**18,
                source_capture_timestamp_preserved=True),
            warmup_inference_ms=warmup_ms,worker_warmup_inference_ms=[],startup_inference_ms=startup_ms,
            injected_delay_ms=0,model_latency_p50_ms=float(np.percentile(latencies,50)) if latencies else None,
            model_latency_p95_ms=float(np.percentile(latencies,95)) if latencies else None,
            observation_age_p95_ms=float(np.percentile(ages,95)) if ages else None,
            camera_capture_p95_ms=float(np.percentile(camera_ms,95)) if camera_ms else None,
            request_preparation_p95_ms=None,request_preparation_errors=0,
            wall_s=elapsed,sim_s=env.tick*.02,real_time_factor=env.tick*.02/max(elapsed,1e-12),
            maximum_clock_lag_s=0.,control_overruns=0,inference_worker_stopped=True,
            evaluator=final,hardware_validated=False,
            prediction_errors=sum(p['submitted_actions'] is None for p in predictions),
            offline_recorder=dict(schema='microduck_offline_prediction_failure_recorder_v1',
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                prediction_error_behavior='retain_existing_plan_then_official_fallback',
                initialization_failure_is_incomplete=True))
        if hasattr(env, 'camera_configuration'):
            report['camera_rendering'] = env.camera_configuration()
        phase='serialization'
        if frames:
            np.savez_compressed(output/'recording.npz',rgb=np.stack(frames),camera_ticks=np.asarray(frame_ticks),
                                qpos=np.stack(qpos),physics_ticks=np.asarray(physics_ticks))
            report['recording']='recording.npz'
        report['prediction_recording']=dict(path='predictions.json',records=len(predictions),
            actions='bounded submitted chunk before stand decoding and age-based indexing',
            observations='original sensor observation',warmup_predictions_excluded=True,
            serialization_outside_measurement=True)
        values=dict(report=report,decisions=decisions,predictions=predictions,control=controls)
        for name,value in values.items():
            write_json(output/(name+'.json'),json.loads(json.dumps(value,default=lambda x:x.tolist())))
        return json.loads(json.dumps(report,default=lambda x:x.tolist()))
    except BaseException as error:
        # Preserve the actual failure; this is not a complete episode or an
        # invented replacement for missing model/physics observations.
        write_json(output/'incomplete-recording.json',dict(status='incomplete',phase=phase,
            tick=getattr(env,'tick',None),error=type(error).__name__+': '+str(error),
            recorded_control_entries=len(controls),recorded_prediction_entries=len(predictions),
            formal_acceptance=False))
        raise
