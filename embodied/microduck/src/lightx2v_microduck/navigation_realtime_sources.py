"""Admit and replay prospective realtime TRAIN recordings for teacher labeling.

This does not publish optimizer input. In particular, a failed student episode
may supply a teacher's starting state, but keeps its original failure outcome.
Synthetic controller fixtures and evaluation recordings are never admitted.
"""
from dataclasses import dataclass
import hashlib
from pathlib import Path
from types import SimpleNamespace

from .navigation_action_contracts import FORWARD04_NAVIGATION_CONTRACT
from .navigation_outcomes import read, require, sha
from .navigation_realtime_collection import SCHEMA, RECORDING_FILES, validate_options, valid_digest
from .navigation_rollin_admission import admit_guarded_cases


@dataclass(frozen=True)
class RealtimeCollectionSources:
    plan_path: Path
    plan_sha256: str
    plan: dict
    cases: dict


def completed_source_cohort(collection, expected_report_sha256):
    """Select the entire planned cohort; do not cherry-pick successful episodes."""
    args, model = collection.plan['evaluation_arguments'], collection.plan['model']
    root = Path(args['output'])
    require(valid_digest(expected_report_sha256) and sha(root/'report.json') == expected_report_sha256,
            'completed realtime collection report is not pinned')
    report = read(root/'report.json')
    require(report.get('status') == 'complete' and report.get('backend') == 'native'
            and report.get('purpose') == 'training_rollin' and report.get('mode') == 'realtime'
            and report.get('training_collection_plan_sha256') == collection.plan_sha256
            and report.get('checkpoint_sha256') == model['checkpoint_sha256']
            and report.get('stats_sha256') == model['stats_sha256']
            and report.get('input_history') == model['input_history']
            and report.get('training_admissions') == collection.plan['admissions'],
            'collection completion belongs to a different model or planned training cohort')
    provenance = report['runtime_provenance']
    frozen = collection.plan['runtime_files']
    def frozen_digest(suffix):
        matches = [h for p,h in frozen.items() if p.endswith('/'+suffix)]
        require(len(matches) == 1, 'recorded runtime source is absent or ambiguous: '+suffix)
        return matches[0]
    require(provenance['evaluation_script_sha256'] == frozen_digest('scripts/microduck/fastwam_navigation_eval.py')
            and provenance['model_config_sha256'] == model['config_sha256'],
            'completed collection did not use its frozen entrypoint and model config')
    require(bool(provenance['source_sha256']), 'completed collection lacks runtime source provenance')
    for name,digest in provenance['source_sha256'].items():
        suffix = name if name.endswith('.py') else name.replace('.', '/')+'.py'
        require(digest == frozen_digest(suffix), 'completed runtime source changed: '+name)
    for field,arg in (('cache_prompt','cache_prompt'), ('compiled','compile'),
                      ('action_context_cache','cache_action_context'), ('action_cuda_graph','action_cuda_graph'),
                      ('inference_steps','inference_steps'), ('snapshot_camera_workers','snapshot_camera_workers'),
                      ('scene','scene'), ('camera_offsamples','camera_offsamples')):
        require(report.get(field) == args[arg], 'completed collection changed planned option: '+field)
    episodes = report.get('episodes', [])
    keys = [(r['task'],r['seed']) for r in episodes]
    require(len(keys) == len(set(keys)) == report.get('completed_episodes') == len(collection.cases)
            and set(keys) == set(collection.cases), 'completed collection omitted or duplicated a planned case')
    result = []
    for row in episodes:
        folder = root/f'{row["task"]}-seed{row["seed"]}'
        require(read(folder/'report.json') == row, 'cohort outcome differs from original episode')
        digest = sha(folder/'training-recording.json')
        verify_source_recording(collection, folder, digest)
        result.append(dict(path=str(folder), task=row['task'], seed=row['seed'], receipt_sha256=digest))
    return sorted(result, key=lambda row:(row['task'],row['seed']))


def open_collection_sources(plan_path, expected_sha256):
    """Verify the frozen collection runtime and real checkpoint once per cohort.

    The reader may live in a newer runtime than collection; it checks the
    original source files on disk rather than silently rebinding their hashes
    to its own implementation.
    """
    from .navigation_evaluation_contract import checkpoint_execution
    from .navigation_history import verify_evaluation_history
    from .navigation_visual_memory import TargetMemoryGridSpec

    path = Path(plan_path).resolve()
    require(valid_digest(expected_sha256) and sha(path) == expected_sha256,
            'prospective realtime collection plan changed')
    plan = read(path)
    require(plan.get('schema') == SCHEMA and plan.get('purpose') == 'prospective_training_rollin'
            and plan.get('original_deadline_tick') == 1500
            and plan.get('training_release_allowed') is False,
            'only a prospective TRAIN plan can admit a real student recording')
    args = SimpleNamespace(**plan['evaluation_arguments'])
    validate_options(args)
    runtime = plan['runtime_files']
    require(bool(runtime) and all(Path(p).is_absolute() and valid_digest(h)
            for p, h in runtime.items()), 'collection source runtime is not pinned')
    for name, digest in runtime.items():
        require(sha(name) == digest, 'frozen collection runtime changed: '+name)
    model = plan['model']
    weights = sha(args.checkpoint)
    stats = sha(Path(args.dataset)/'dataset_stats.json')
    require(valid_digest(weights) and model['checkpoint_sha256'] == weights
            and model['stats_sha256'] == stats and model['receipt_sha256'] == sha(args.checkpoint_receipt)
            and model['config_sha256'] == sha(args.model_config),
            'recorded student checkpoint or frozen normalization changed')
    receipt = read(args.checkpoint_receipt)
    execution, _ = checkpoint_execution(receipt, weights_sha256=weights, stats_sha256=stats,
                                        requested=args.navigation_contract)
    history = TargetMemoryGridSpec(fallback_history_ticks=300)
    verify_evaluation_history(receipt, history, weights_sha256=weights, stats_sha256=stats)
    require(execution == model['action_contract'] == FORWARD04_NAVIGATION_CONTRACT
            and model['input_history'] == history.as_dict(), 'student model contract changed')
    protocol = plan['evaluation_protocol']
    require(sha(protocol['path']) == protocol['sha256'], 'reserved evaluation conditions changed')
    cases = admit_guarded_cases(plan['extended_release_root'], args.dataset,
        plan['extended_release_sha256'], read(protocol['path']), args.seeds, args.tasks)
    require(plan['admissions'] == [cases[k].binding() for k in sorted(cases)],
            'qualified TRAIN membership changed after collection')
    return RealtimeCollectionSources(path, expected_sha256, plan, cases)


def verify_source_recording(collection, folder, expected_receipt_sha256):
    """Bind a completed source to the prior plan, including honest failures."""
    from .navigation import navigation_expiry_limits
    from .navigation_model_inputs import RECORDING_CONTRACT

    require(isinstance(collection, RealtimeCollectionSources), 'verified collection context required')
    folder = Path(folder).resolve()
    args, model = collection.plan['evaluation_arguments'], collection.plan['model']
    require(folder.parent == Path(args['output']).resolve(),
            'student recording is outside its prospective collection output')
    receipt_path = folder/'training-recording.json'
    require(valid_digest(expected_receipt_sha256) and not receipt_path.is_symlink()
            and sha(receipt_path) == expected_receipt_sha256, 'student recording receipt changed')
    receipt = read(receipt_path)
    key = receipt['task'], receipt['seed']
    require(key in collection.cases and folder.name == f'{key[0]}-seed{key[1]}',
            'recording is not a prospectively selected TRAIN case')
    admission = collection.cases[key]
    require(receipt.get('schema') == 'microduck_realtime_train_recording_v1'
            and receipt.get('status') == 'awaiting_independent_physical_replay'
            and receipt.get('training_collection_plan_sha256') == collection.plan_sha256
            and receipt.get('student_checkpoint_sha256') == model['checkpoint_sha256']
            and receipt.get('training_admission') == admission.binding()
            and receipt.get('training_release_allowed') is False
            and set(receipt.get('files', {})) == RECORDING_FILES,
            'recording identity or prospective admission differs')
    for name, digest in receipt['files'].items():
        require(valid_digest(digest) and not (folder/name).is_symlink()
                and sha(folder/name) == digest, 'recorded student evidence changed: '+name)
    report, start, audit = (read(folder/n) for n in
        ('report.json', 'collection-admission.json', 'model-input-audit.json'))
    policy = report['policy']
    require(policy.get('synthetic_policy') is not True
            and policy.get('artifact_sha256') == model['checkpoint_sha256']
            and policy.get('uses_privileged_state') is False
            and policy.get('history_contract') == model['input_history']
            and policy.get('model_input_recording') == RECORDING_CONTRACT
            and not policy.get('input_intervention') and not policy.get('action_intervention'),
            'synthetic, privileged or different model cannot supply training roll-ins')
    require((report['task'], report['seed']) == key and report.get('mode') == 'realtime'
            and report.get('initialization') == start.get('initialization') == 'reset_and_warmup'
            and report.get('split') == start.get('split') == 'train'
            and report.get('collection_mode') == 'realtime_student_record_only'
            and report.get('training_admission') == start.get('training_admission') == admission.binding()
            and report.get('training_collection_plan_sha256') == start.get('training_collection_plan_sha256')
                == collection.plan_sha256
            and start.get('student_checkpoint_sha256') == model['checkpoint_sha256']
            and report.get('training_recording_admission_sha256') == receipt['files']['collection-admission.json']
            and start.get('initial_training_observation_verified') is True
            and start.get('admitted_before_inference') is True
            and start.get('physical_warmup_ticks') == 50 and start.get('admission_tick') == 0
            and start['admitted_unix'] >= collection.plan['created_unix']
            and report.get('inference_worker_stopped') is True,
            'student recording was not admitted at the qualified original task origin')
    require(audit.get('status') in ('complete', 'complete_with_recorded_camera_loss')
            and audit.get('submitted_core_call_inputs_verified') is True
            and receipt['original_strict_model_success'] == report['strict_model_success']
            and receipt['original_realtime_success'] == report['realtime_success']
            and receipt['fallback_control_ticks'] == report['fallback_control_ticks'],
            'student outcome or recorded model-input evidence changed')
    limits = navigation_expiry_limits(args['max_action_age_ms'], realtime=True)
    clock = report['execution_clock']
    expected_ns = None if limits['max_wall_age_s'] is None else round(limits['max_wall_age_s']*1e9)
    require(clock['max_age_ticks'] == limits['max_age_ticks']
            and clock['max_wall_age_ns'] == expected_ns
            and clock['source_capture_timestamp_preserved'] is True
            and report['action_contract'] == model['action_contract']
            and report['evaluator']['scene']['name'] == args['scene'],
            'recording changed the planned execution clock, action contract or scene')
    case = dict(mode='realtime', profile='official_bam', scene=args['scene'], seed=key[1], task=key[0],
        weights_sha256=model['checkpoint_sha256'], contract=model['action_contract'],
        warmup_ticks=50, image_size=256, execution_clock=clock,
        snapshot_camera_workers=args['snapshot_camera_workers'], camera_offsamples=args['camera_offsamples'])
    return admission, case, receipt


def replay_training_source(collection, folder, expected_receipt_sha256, *, snapshot_observer=None):
    """Recheck model inputs and every executed physical/camera state before labeling."""
    from .navigation_camera_recovery import CAMERA_LOSS_RECOVERY
    from .navigation_model_inputs import audit_recorded_inputs
    from .navigation_recording_replay import replay_recording
    import numpy as np

    folder = Path(folder)
    admission, case, receipt = verify_source_recording(collection, folder, expected_receipt_sha256)
    recovery = CAMERA_LOSS_RECOVERY if collection.plan['evaluation_arguments']['recover_declared_camera_loss'] else None
    inputs = audit_recorded_inputs(folder, expected_history_contract=collection.plan['model']['input_history'],
                                   expected_camera_loss_recovery=recovery)
    original = read(folder/'model-input-audit.json')
    require({k:v for k,v in inputs.items() if k != 'audit_module_sha256'}
            == {k:v for k,v in original.items() if k != 'audit_module_sha256'},
            'independent model input reconstruction differs from the recorded audit')
    captured_states = {}
    def capture(snapshot, observation):
        state = observation.state
        require(state.shape == (61,) and state.dtype == np.float32 and np.isfinite(state).all(),
                'replayed source state must retain all 61 physical FP32 coordinates')
        key = str(observation.tick)
        require(key not in captured_states, 'source camera state was replayed twice')
        captured_states[key] = hashlib.sha256(state.tobytes()).hexdigest()
        if snapshot_observer is not None:
            snapshot_observer(snapshot, observation)
    physical = replay_recording(folder, case, snapshot_observer=capture)
    require(physical['status'] == 'complete' and physical['physical_replay']
            and physical['camera_replay'] and physical['final_evaluator_exact'],
            'incomplete physical/camera replay cannot supply teacher branches')
    # The admission boolean alone is insufficient: independently reconstructed
    # task-origin sensors must also equal the qualified training demonstration.
    expected_rgb, expected_state, instruction = admission._initial_observation()
    with np.load(folder/'recording.npz', allow_pickle=False) as packed:
        require(int(packed['camera_ticks'][0]) == 0, 'recording omitted its original task-origin camera')
        require(set(captured_states) == {str(int(t)) for t in packed['camera_ticks']},
                'independent replay omitted a captured camera state')
        np.testing.assert_array_equal(packed['rgb'][0], expected_rgb)
    np.testing.assert_array_equal(physical['initial_proprio48'], expected_state)
    require(physical['initial_instruction'] == instruction, 'replayed instruction changed from its TRAIN source')
    verify_source_recording(collection, folder, expected_receipt_sha256)
    return dict(schema='microduck_realtime_training_source_replay_v1', status='complete',
        training_admission=admission.binding(), collection_plan_sha256=collection.plan_sha256,
        recording_receipt_sha256=expected_receipt_sha256, source_files=receipt['files'],
        student_checkpoint_sha256=collection.plan['model']['checkpoint_sha256'],
        original_strict_model_success=receipt['original_strict_model_success'],
        original_realtime_success=receipt['original_realtime_success'],
        fallback_control_ticks=receipt['fallback_control_ticks'], model_input_audit=inputs,
        physical_replay=physical, eligible_for_teacher_branching=True,
        captured_state61_sha256=captured_states,
        qualified_initial_sensors_independently_verified=True,
        training_release_allowed=False, formal_acceptance=False, model_promoted=False)
