"""Prospectively admit realtime TRAIN roll-ins, without authorizing labels.

The ordinary evaluation controller remains the recording engine. A pinned plan
selects qualified training layouts before model loading; episode admission is
checked after physical warmup and before learned inference. Failures, camera loss and official fallback
remain recorded failures. Independent replay and teacher qualification are later
gates, never inferred from collection process success.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

from .navigation_action_contracts import FORWARD04_NAVIGATION_CONTRACT
from .navigation_outcomes import read, require, sha
from .navigation_rollin_admission import GuardedTrainingCase, admit_guarded_cases


SCHEMA = 'microduck_realtime_train_collection_v1'
PLAN_ARGUMENTS = {'training_collection_plan', 'expected_training_collection_sha256'}
RECORDING_FILES = frozenset(('report.json', 'recording.npz', 'control.json', 'decisions.json',
    'predictions.json', 'camera_delivery.json', 'stage_timings.json', 'model-input-audit.json',
    'collection-admission.json'))


def valid_digest(value):
    return isinstance(value,str) and len(value)==64 and not set(value)-set('0123456789abcdef') and value!='0'*64


def evaluation_arguments(args):
    def plain(value):
        if isinstance(value,Path): return str(value.resolve())
        if isinstance(value,(list,tuple)): return [plain(v) for v in value]
        return value
    return {k:plain(v) for k,v in vars(args).items() if k not in PLAN_ARGUMENTS}


def runtime_files(entrypoint):
    entrypoint=Path(entrypoint).resolve()
    package=Path(__file__).resolve().parent
    repo=entrypoint.parents[2]
    require((repo/'lightx2v/models/runners/wan/fastwam_runner.py').is_file(),
            'native collection runtime is incomplete')
    files={entrypoint,*package.rglob('*.py'),*(repo/'lightx2v').rglob('*.py')}
    return {str(p):sha(p) for p in sorted(files)}


def validate_options(args):
    require(args.backend=='native' and args.mode=='realtime' and args.scene=='room_curriculum_one'
        and args.navigation_contract==FORWARD04_NAVIGATION_CONTRACT and args.max_ticks==1500
        and args.snapshot_camera_workers>0 and args.camera_offsamples is None
        and args.record_frames and args.record_predictions and args.record_model_inputs and args.record_stage_timings
        and args.history_target_keyframe and args.history_target_keyframe_fallback_ticks==300
        and args.history_offsets_ticks==[150,50,0] and args.checkpoint_receipt is not None
        and not args.mask_rgb and args.instruction_color is None and not args.project_constant_actions
        and args.input_ablation_stage=='observation',
        'TRAIN roll-ins require the native realtime guarded task, complete recordings and unchanged target-memory inputs')
    require(args.seeds and len(args.seeds)==len(set(args.seeds))
            and args.tasks and len(args.tasks)==len(set(args.tasks)), 'duplicate or empty training cohort')


@dataclass(frozen=True)
class RealtimeTrainingCollection:
    plan_sha256: str
    checkpoint_sha256: str
    cases: dict


def prepare_plan(args, *, extended_root, extended_sha256, evaluation_protocol,
                 evaluation_protocol_sha256, entrypoint):
    """CPU-only plan construction; does not launch a model or allocate a GPU."""
    from .navigation_evaluation_contract import checkpoint_execution
    from .navigation_history import history_spec, verify_evaluation_history
    from .navigation import navigation_expiry_limits

    validate_options(args)
    require(not any(getattr(args,k,None) for k in PLAN_ARGUMENTS),'prepare the plan before attaching it to a run')
    require(valid_digest(extended_sha256) and valid_digest(evaluation_protocol_sha256)
            and sha(evaluation_protocol)==evaluation_protocol_sha256, 'qualified training/evaluation source is not pinned')
    navigation_expiry_limits(args.max_action_age_ms,realtime=True)
    weights,stats=sha(args.checkpoint),sha(Path(args.dataset)/'dataset_stats.json')
    receipt=read(args.checkpoint_receipt)
    execution,_=checkpoint_execution(receipt,weights_sha256=weights,stats_sha256=stats,
                                     requested=args.navigation_contract)
    spec=history_spec((150,50,0),target_keyframe=True,target_keyframe_fallback_ticks=300)
    verify_evaluation_history(receipt,spec,weights_sha256=weights,stats_sha256=stats)
    require(execution==FORWARD04_NAVIGATION_CONTRACT and valid_digest(weights),'untrained or incompatible student identity')
    cases=admit_guarded_cases(extended_root,args.dataset,extended_sha256,read(evaluation_protocol),args.seeds,args.tasks)
    return dict(schema=SCHEMA,purpose='prospective_training_rollin',created_unix=time.time(),
        evaluation_arguments=evaluation_arguments(args),runtime_files=runtime_files(entrypoint),
        model=dict(checkpoint_sha256=weights,stats_sha256=stats,receipt_sha256=sha(args.checkpoint_receipt),
                   config_sha256=sha(args.model_config),input_history=spec.as_dict(),action_contract=execution),
        extended_release_root=str(Path(extended_root).resolve()),extended_release_sha256=extended_sha256,
        evaluation_protocol=dict(path=str(Path(evaluation_protocol).resolve()),sha256=evaluation_protocol_sha256),
        admissions=[cases[k].binding() for k in sorted(cases)],original_deadline_tick=1500,
        gpu_started=False,training_release_allowed=False,formal_acceptance=False)


def load_plan(args, *, weights_sha256, stats_sha256, entrypoint):
    path=getattr(args,'training_collection_plan',None)
    digest=getattr(args,'expected_training_collection_sha256',None)
    if path is None and digest is None:return None
    validate_options(args)
    require(path is not None and valid_digest(digest) and sha(path)==digest,'training collection plan changed or is not pinned')
    plan=read(path)
    require(plan.get('schema')==SCHEMA and plan.get('purpose')=='prospective_training_rollin'
        and plan.get('original_deadline_tick')==1500 and plan.get('training_release_allowed') is False
        and plan.get('evaluation_arguments')==evaluation_arguments(args),
        'collection invocation differs from its prospective TRAIN plan')
    require(plan.get('runtime_files')==runtime_files(entrypoint),'collection runtime changed after planning')
    model=plan['model']
    from .navigation_visual_memory import TargetMemoryGridSpec
    require(valid_digest(weights_sha256) and model['checkpoint_sha256']==weights_sha256
        and model['stats_sha256']==stats_sha256 and model['receipt_sha256']==sha(args.checkpoint_receipt)
        and model['config_sha256']==sha(args.model_config)
        and model['action_contract']==FORWARD04_NAVIGATION_CONTRACT
        and model['input_history']==TargetMemoryGridSpec(fallback_history_ticks=300).as_dict(),
        'student model, config or frozen normalization changed')
    protocol=plan['evaluation_protocol']
    require(sha(protocol['path'])==protocol['sha256'],'reserved evaluation layout protocol changed')
    cases=admit_guarded_cases(plan['extended_release_root'],args.dataset,plan['extended_release_sha256'],
                             read(protocol['path']),args.seeds,args.tasks)
    require(plan['admissions']==[cases[k].binding() for k in sorted(cases)],'qualified TRAIN membership changed')
    return RealtimeTrainingCollection(digest,weights_sha256,cases)


def episode_metadata(admission, *, plan_sha256, policy, env, task, seed, realtime,
                     warmup_ticks, reset_environment, max_ticks, record_frames,
                     record_predictions, record_stage_timings, snapshot_camera_workers):
    from .navigation_model_inputs import RECORDING_CONTRACT

    require(isinstance(admission,GuardedTrainingCase) and valid_digest(plan_sha256),
            'a qualified TRAIN case and prospective plan are required')
    metadata=getattr(policy,'metadata',{})
    require(realtime and reset_environment and warmup_ticks==50 and max_ticks==1500
        and record_frames and record_predictions and record_stage_timings and snapshot_camera_workers>0
        and valid_digest(metadata.get('artifact_sha256')) and metadata.get('uses_privileged_state') is False
        and metadata.get('model_input_recording')==RECORDING_CONTRACT
        and not metadata.get('input_intervention') and not metadata.get('action_intervention'),
        'TRAIN recording requires a real student identity and complete original realtime observations')
    admission.validate_request(task=task,seed=seed,scene=env.scene,profile=env.profile,
        contract=metadata.get('contract'),record_only=True)
    return dict(schema=SCHEMA,purpose='training_rollin',split='train',
        collection_mode='realtime_student_record_only',training_admission=admission.binding(),
        training_collection_plan_sha256=plan_sha256,student_checkpoint_sha256=metadata['artifact_sha256'],
        training_release_allowed=False,formal_acceptance=False)


def seal_recording(folder, collection, admission):
    """Bind completed artifacts; leave physics replay and label admission pending."""
    from .wam_data import write_json

    folder=Path(folder)
    require(isinstance(collection,RealtimeTrainingCollection) and isinstance(admission,GuardedTrainingCase)
        and collection.cases.get((admission.task,admission.seed))==admission,'unregistered realtime collection case')
    report,start,audit=(read(folder/n) for n in ('report.json','collection-admission.json','model-input-audit.json'))
    require(report.get('training_admission')==start.get('training_admission')==admission.binding()
        and report.get('training_collection_plan_sha256')==start.get('training_collection_plan_sha256')==collection.plan_sha256
        and report.get('mode')=='realtime' and report.get('initialization')=='reset_and_warmup'
        and report.get('collection_mode')=='realtime_student_record_only' and report.get('split')=='train'
        and report['policy']['artifact_sha256']==start['student_checkpoint_sha256']==collection.checkpoint_sha256
        and report['policy']['uses_privileged_state'] is False
        and report.get('training_recording_admission_sha256')==sha(folder/'collection-admission.json')
        and start.get('initial_training_observation_verified') is True
        and start.get('admitted_before_inference') is True
        and start.get('initialization')=='reset_and_warmup'
        and start.get('physical_warmup_ticks')==50 and start.get('admission_tick')==0
        and report.get('inference_worker_stopped') is True
        and audit.get('status') in ('complete','complete_with_recorded_camera_loss')
        and audit.get('submitted_core_call_inputs_verified') is True,
        'recorded student identity, prospective admission or model inputs are incomplete')
    files={name:sha(folder/name) for name in sorted(RECORDING_FILES)}
    receipt=dict(schema='microduck_realtime_train_recording_v1',status='awaiting_independent_physical_replay',
        task=admission.task,seed=admission.seed,training_admission=admission.binding(),
        training_collection_plan_sha256=collection.plan_sha256,student_checkpoint_sha256=collection.checkpoint_sha256,
        files=files,original_strict_model_success=report['strict_model_success'],
        original_realtime_success=report['realtime_success'],fallback_control_ticks=report['fallback_control_ticks'],
        source_timing_gate_unchanged=True,training_release_allowed=False,formal_acceptance=False)
    require(not (folder/'training-recording.json').exists(),'training recording receipt already exists')
    write_json(folder/'training-recording.json',receipt)
    return receipt
