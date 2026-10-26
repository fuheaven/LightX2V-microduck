"""Release qualified teacher continuations from actual realtime TRAIN states.

Dense/offline releases retain their original loader. This release carries the
measured camera receipts, rejects unusable windows in either training view,
and preserves every selected teacher outcome, including rejected branches.
"""
from pathlib import Path
import hashlib
import shutil
import tempfile

import numpy as np

from .navigation_action_contracts import FORWARD04_NAVIGATION_CONTRACT as CONTRACT
from .navigation_augmentation import contained
from .navigation_extended_collection import independent_success, verify_guard_diagnostics
from .navigation_extended_data import supervised_transitions
from .navigation_extended_release import _base_identity, _read_pinned, normalizer_coverage, normalization_binding
from .navigation_guarded_recovery import TEACHER_KEYS
from .navigation_history import HistoryGridSpec
from .navigation_outcomes import read, require, sha
from .navigation_realtime_collection import valid_digest
from .navigation_realtime_history import read_measured_prefix, index_branch_history
from .navigation_realtime_sources import open_collection_sources, completed_source_cohort, verify_source_recording
from .navigation_visual_memory import TargetMemoryGridSpec
from .wam_data import write_json


KIND = 'microduck_realtime_guarded_recovery_v1'
PROTOCOL = 'microduck_realtime_guarded_recovery_protocol_v1'
BRANCH = 'microduck_realtime_guarded_branch_v1'
PREFIX_FILES = {'report.json', 'recording.npz', 'predictions.json', 'camera_delivery.json'}


def arrays(path):
    with np.load(path, allow_pickle=False) as packed:
        return {k:packed[k].copy() for k in packed.files}


def key(row):
    return row['task'], row['seed'], row['source_tick']


def sampling_contract(value):
    require(isinstance(value, dict) and set(value) == {'horizon','video_stride','sample_stride'}
            and all(type(v) is int and v > 0 for v in value.values())
            and value['horizon'] % value['video_stride'] == 0
            and (value['horizon']//value['video_stride']) % 4 == 0,
            'realtime corrections require an explicit compatible video/action sampling contract')
    return dict(value)


def validate_protocol(protocol):
    require(protocol.get('schema') == PROTOCOL and protocol.get('action_contract') == CONTRACT
            and protocol.get('original_deadline_tick') == 1500
            and protocol.get('training_release_allowed') is False
            and protocol.get('synthetic_fixture') is not True,
            'realtime recovery protocol or original task deadline differs')
    sampling_contract(protocol['sampling'])
    ref = protocol['collection_plan']
    context = open_collection_sources(ref['path'], ref['sha256'])
    cohort = completed_source_cohort(context, protocol['collection_report_sha256'])
    environments = [digest for name,digest in context.plan['runtime_files'].items()
                    if name.endswith('/lightx2v_microduck/env.py')]
    require(len(environments) == 1
            and protocol.get('collection_environment_sha256') == environments[0],
            'teacher replay must use the frozen student collection environment')
    root = Path(context.plan['extended_release_root'])
    require(sha(root/'plan.json') == protocol['teacher_plan_sha256'], 'qualified teacher plan changed')
    teacher = read(root/'plan.json')
    require(protocol['teacher'] == {k:teacher[k] for k in TEACHER_KEYS},
            'realtime labels changed the qualified teacher')
    runtime = protocol['teacher_runtime']
    manifest = _read_pinned(runtime['manifest'], protocol['teacher']['source_manifest_sha256'], 'teacher source manifest')
    for name,digest in manifest['files'].items():
        require(sha(contained(runtime['source'],name)) == digest, 'qualified teacher runtime changed: '+name)
    require(runtime['environment_sha256'] == manifest['files']['lightx2v_microduck/env.py']
            and sha(Path(runtime['calibration'])/'report.json') == protocol['teacher']['calibration_sha256'],
            'qualified teacher environment or calibration changed')
    requested = protocol.get('sources', [])
    wanted = {(r['task'],r['seed']):r for r in cohort}
    require(len(requested) == len(wanted) and {(r['task'],r['seed']) for r in requested} == set(wanted),
            'recovery protocol omitted or duplicated part of the collected cohort')
    sources = {}
    for row in requested:
        case_key = row['task'],row['seed']
        source = wanted[case_key]
        require(all(row.get(k) == v for k,v in source.items()), 'source receipt differs from original collection')
        admission,case,receipt = verify_source_recording(context,row['path'],row['receipt_sha256'])
        proof_ref = row['source_replay']
        proof = _read_pinned(proof_ref['path'],proof_ref['sha256'],'realtime student replay')
        physical,inputs = proof.get('physical_replay',{}),proof.get('model_input_audit',{})
        require(proof.get('schema') == 'microduck_realtime_training_source_replay_v1'
            and proof.get('status') == 'complete'
            and proof.get('collection_plan_sha256') == context.plan_sha256
            and proof.get('recording_receipt_sha256') == row['receipt_sha256']
            and proof.get('source_files') == receipt['files']
            and proof.get('training_admission') == admission.binding()
            and proof.get('student_checkpoint_sha256') == context.plan['model']['checkpoint_sha256']
            and proof.get('qualified_initial_sensors_independently_verified') is True
            and proof.get('eligible_for_teacher_branching') is True
            and proof.get('training_release_allowed') is False
            and physical.get('case') == case and physical.get('status') == 'complete'
            and physical.get('physical_replay') is True and physical.get('camera_replay') is True
            and physical.get('final_evaluator_exact') is True
            and inputs.get('submitted_core_call_inputs_verified') is True,
            'teacher source lacks complete independent model-input/physics/camera replay')
        for part in (physical,inputs):
            require(bool(part.get('input_files_sha256')) and all(receipt['files'].get(n) == d
                for n,d in part['input_files_sha256'].items()), 'student replay bound different original files')
        require(all(proof[k] == receipt[k] for k in
            ('original_strict_model_success','original_realtime_success','fallback_control_ticks')),
            'student failure or fallback was reinterpreted as success')
        with np.load(Path(row['path'])/'recording.npz',allow_pickle=False) as packed:
            captured_ticks = {str(int(t)) for t in packed['camera_ticks']}
        states = proof.get('captured_state61_sha256',{})
        require(set(states) == captured_ticks and all(valid_digest(v) for v in states.values()),
                'independent replay omitted the physical state at a captured frame')
        anchors = row.get('anchors',[])
        require(isinstance(anchors,list) and len(anchors) == len(set(anchors))
            and all(type(t) is int and 0 <= t < min(1500,physical['exact_control_ticks']) and t % 5 == 0
                    for t in anchors), 'teacher anchor is not a unique live student decision')
        for tick in anchors:
            read_measured_prefix(row['path'],tick,expected_files=receipt['files'])
        sources[case_key] = dict(row=row,receipt=receipt,admission=admission,proof=proof)
    require(any(item['row']['anchors'] for item in sources.values()), 'no actual teacher branches selected')
    return context,sources


def branch_sample(folder, report, source, protocol, *, stats):
    """Reconstruct public labels and both causal camera views from raw evidence."""
    folder = Path(folder)
    tick,n = report['source_tick'],report.get('control_ticks')
    require(report.get('schema') == BRANCH and report.get('split') == 'train'
        and type(tick) is int and 0 <= tick < 1500 and tick % 5 == 0
        and tick in source['row']['anchors']
        and type(n) is int and 0 < n <= 1500-tick
        and report.get('original_deadline_tick') == 1500
        and report.get('original_student_unchanged') is True
        and report.get('student_snapshot_exact') is True
        and report.get('training_release_allowed') is False
        and report['result']['tick'] == tick+n
        and report.get('teacher') == protocol['teacher']
        and report.get('source_replay_sha256') == source['row']['source_replay']['sha256'],
        'teacher continuation changed its qualified identity, student state or task clock')
    files = report['files'];full=n//5
    expected = {'trajectory.npz'} | ({'navigation.npz'} if full else set())
    if n % 5 == 0:expected.add('terminal_rgb.npy')
    require(set(files) == expected and report['terminal_frame_measured'] == (n % 5 == 0),
            'raw branch omits its measured terminal frame or partial tail')
    for name,digest in files.items():
        require(valid_digest(digest) and sha(folder/name) == digest, 'teacher trace changed: '+name)
    for name,expected_env in (
        ('teacher',protocol['teacher_runtime']['environment_sha256']),
        ('collection',protocol['collection_environment_sha256']),
    ):
        replay=report['replay'][name]
        require(replay.get('status') == 'complete' and replay.get('exact_control_ticks') == n
            and replay.get('exact_rgb_frames') == full+1 and replay.get('environment_sha256') == expected_env
            and replay.get('teacher_environment_sha256') == protocol['teacher_runtime']['environment_sha256']
            and replay.get('final_evaluator_exact_except_bound_source_digest') is True
            and replay.get('teacher_queried_during_replay') is False
            and replay.get('trajectory_sha256') == files['trajectory.npz'],
            'teacher branch lacks complete command-only replay in both runtimes')
    trace=arrays(folder/'trajectory.npz')
    require(len(trace['state61']) == n, 'raw branch length differs from its physical proof')
    require(trace['state61'].shape == (n,61) and trace['state61'].dtype == np.float32
            and hashlib.sha256(trace['state61'][0].tobytes()).hexdigest()
                == source['proof']['captured_state61_sha256'][str(tick)],
            'teacher labels do not start at the independently replayed student state')
    verify_guard_diagnostics(trace['action'],report['teacher_diagnostics'],source_tick=tick)
    success=independent_success(report['result'],report['teacher_diagnostics'])
    if not full:return dict(strict_success=success,decisions=0,eligible=False,reason='no_complete_decision'),None
    terminal=np.load(folder/'terminal_rgb.npy',allow_pickle=False) if n % 5 == 0 else None
    public,alignment=supervised_transitions(trace,terminal_rgb=terminal)
    public['ticks']+=tick
    recorded=arrays(folder/'navigation.npz')
    require(set(recorded) == set(public) and all(np.array_equal(recorded[k],v) for k,v in public.items()),
            'released action/state/image tensors differ from the complete raw teacher trace')
    prefix=read_measured_prefix(source['row']['path'],tick,expected_files=source['receipt']['files'])
    np.testing.assert_array_equal(public['rgb'][0],prefix.rgb[-1])
    require(prefix.instruction == report['instruction'], 'teacher changed the observed instruction')
    choices={}
    for name,spec in [('student',TargetMemoryGridSpec(fallback_history_ticks=300)),
                      ('retention',HistoryGridSpec((300,150,50,0)))]:
        choices[name]=index_branch_history(prefix,public,spec).window_starts(**protocol['sampling'])
    starts=sorted(set(choices['student']['accepted']) & set(choices['retention']['accepted']))
    eligible=bool(success and starts)
    coverage=normalizer_coverage(public,stats) if eligible else None
    return dict(strict_success=success,decisions=full,eligible=eligible,
        reason='accepted' if eligible else ('teacher_failed' if not success else 'no_causal_paired_windows'),
        alignment=alignment,paired_window_selection=choices,sample_starts=starts,
        actual_student_anchor_starts=[0] if 0 in starts else [],teacher_continuation_starts=[s for s in starts if s],
        normalization_coverage=coverage),public


def assessed_branches(source_root, protocol, protocol_sha256, assessment, context, sources, stats):
    expected={(r['task'],r['seed'],t) for r in protocol['sources'] for t in r['anchors']}
    rows=assessment.get('rows',[])
    require(assessment.get('status') == 'complete' and assessment.get('protocol_sha256') == protocol_sha256
            and len(rows) == len(expected) and {key(r) for r in rows} == expected,
            'teacher assessment omitted or duplicated selected branch outcomes')
    for row in rows:
        folder=contained(source_root,row['path'])
        report=_read_pinned(folder/'report.json',row['report_sha256'],'teacher branch report')
        require(key(report) == key(row) and report.get('protocol_sha256') == protocol_sha256
                and report.get('student_checkpoint_sha256') == context.plan['model']['checkpoint_sha256'],
                'branch identity differs from the actual student and selected protocol')
        source=sources[row['task'],row['seed']]
        result,public=branch_sample(folder,report,source,protocol,stats=stats)
        require(row.get('strict_success') is result['strict_success'], 'teacher branch success was relabeled')
        yield dict(row,**result),public,report,source


def export_realtime_recoveries(source_root, protocol_path, assessment_path, base_dataset, output, *,
                               expected_protocol_sha256, expected_assessment_sha256):
    """Publish exact public tensors; leave all unsuccessful sources in the audit."""
    output=Path(output);require(not output.exists(),'realtime recovery releases are immutable')
    protocol=_read_pinned(protocol_path,expected_protocol_sha256,'realtime recovery protocol')
    assessment=_read_pinned(assessment_path,expected_assessment_sha256,'teacher assessment')
    context,sources=validate_protocol(protocol)
    _,stats,base=_base_identity(base_dataset)
    require(base['stats_sha256'] == context.plan['model']['stats_sha256']
            and Path(base_dataset).resolve() == Path(context.plan['evaluation_arguments']['dataset']).resolve(),
            'realtime release changed the original base dataset or normalization')
    output.parent.mkdir(parents=True,exist_ok=True)
    staging=Path(tempfile.mkdtemp(prefix=output.name+'.staging-',dir=output.parent))
    try:
        for src,name in [(protocol_path,'source_protocol.json'),(assessment_path,'source_assessment.json'),
                         (Path(base_dataset)/'dataset_stats.json','dataset_stats.json')]:shutil.copyfile(src,staging/name)
        entries,outcomes,prefixes=[],[],{}
        for row,public,report,source in assessed_branches(source_root,protocol,expected_protocol_sha256,
                                                        assessment,context,sources,stats):
            outcomes.append(row)
            if not row['eligible']:continue
            task,seed,tick=key(row);source_name=f'{task}-seed{seed}'
            relative=f'episodes/{source_name}/anchor-{tick:04d}'
            folder=staging/relative;folder.mkdir(parents=True)
            np.savez_compressed(folder/'navigation.npz',**public)
            prefix_dir='sources/'+source_name
            if prefix_dir not in prefixes:
                (staging/prefix_dir).mkdir(parents=True)
                files={n:source['receipt']['files'][n] for n in sorted(PREFIX_FILES)}
                for name in files:shutil.copyfile(Path(source['row']['path'])/name,staging/prefix_dir/name)
                prefixes[prefix_dir]=files
            entry=dict(task=task,seed=seed,source_tick=tick,path=relative,split='train',eligible=True,
                search_visibility_gated=True,student_checkpoint_sha256=context.plan['model']['checkpoint_sha256'],
                frames=row['decisions']+1,sha256=sha(folder/'navigation.npz'),
                sample_starts=row['sample_starts'],actual_student_anchor_starts=row['actual_student_anchor_starts'],
                teacher_continuation_starts=row['teacher_continuation_starts'],
                realtime_prefix=dict(path=prefix_dir,files=prefixes[prefix_dir]))
            write_json(folder/'episode.json',dict(entry,schema=CONTRACT,kind=KIND,
                instruction=report['instruction'],scene={'name':'room_curriculum_one'},profile='official_bam',
                source_branch_report_sha256=row['report_sha256'],hardware_validated=False))
            entry['metadata_sha256']=sha(folder/'episode.json');entries.append(entry)
        require(bool(entries),'no successful causal teacher branches to release')
        audit=dict(status='complete',protocol_sha256=expected_protocol_sha256,
            assessment_sha256=expected_assessment_sha256,source_root=str(Path(source_root).resolve()),
            outcomes=outcomes,normalization_changed=False,all_selected_outcomes_preserved=True,
            student_success_reinterpreted=False,formal_acceptance=False)
        write_json(staging/'source_audit.json',audit)
        index=dict(kind=KIND,schema=CONTRACT,**base,episodes=entries,
            student_checkpoint_sha256=context.plan['model']['checkpoint_sha256'],
            extended_release_sha256=context.plan['extended_release_sha256'],sampling=protocol['sampling'],
            normalization_binding=normalization_binding(base['stats_sha256']),
            training_release_allowed=True,metadata_pinned=True,
            evidence={n:sha(staging/n) for n in ('source_protocol.json','source_assessment.json','source_audit.json')})
        write_json(staging/'index.json',index)
        load_realtime_recoveries(staging,base_dataset,context.plan['extended_release_root'],
            context.plan['extended_release_sha256'],sha(staging/'index.json'),index['student_checkpoint_sha256'],
            horizon=protocol['sampling']['horizon'],video_stride=protocol['sampling']['video_stride'])
        staging.rename(output)
        return index
    finally:
        if staging.exists():shutil.rmtree(staging)


def load_realtime_recoveries(root, base_dataset, extended_root, extended_sha256,
                             expected_index_sha256, checkpoint_sha256, *, horizon, video_stride):
    """Revalidate release provenance, raw labels, copied prefixes and sampling."""
    root=Path(root).resolve();index=_read_pinned(root/'index.json',expected_index_sha256,'realtime correction index')
    require(index.get('kind') == KIND and index.get('schema') == CONTRACT
        and index.get('training_release_allowed') is True and index.get('metadata_pinned') is True
        and valid_digest(checkpoint_sha256) and index.get('student_checkpoint_sha256') == checkpoint_sha256
        and index.get('extended_release_sha256') == extended_sha256
        and index['sampling']['horizon'] == horizon and index['sampling']['video_stride'] == video_stride,
        'realtime correction identity, split or paired-window contract changed')
    require(set(index['evidence']) == {'source_protocol.json','source_assessment.json','source_audit.json'},
            'realtime correction evidence is incomplete')
    protocol,assessment,audit=[_read_pinned(root/n,index['evidence'][n],n) for n in
        ('source_protocol.json','source_assessment.json','source_audit.json')]
    context,sources=validate_protocol(protocol)
    _,stats,base=_base_identity(base_dataset)
    require(all(index.get(k) == v for k,v in base.items()) and sha(root/'dataset_stats.json') == base['stats_sha256']
        and index['normalization_binding'] == normalization_binding(base['stats_sha256'])
        and index['sampling'] == protocol['sampling']
        and context.plan['model']['checkpoint_sha256'] == checkpoint_sha256
        and Path(context.plan['extended_release_root']).resolve() == Path(extended_root).resolve()
        and context.plan['extended_release_sha256'] == extended_sha256,
        'realtime correction changed frozen sources or normalization')
    require(audit['status'] == 'complete' and audit['normalization_changed'] is False
        and audit.get('all_selected_outcomes_preserved') is True
        and audit.get('student_success_reinterpreted') is False
        and audit.get('formal_acceptance') is False
        and audit['protocol_sha256'] == index['evidence']['source_protocol.json']
        and audit['assessment_sha256'] == index['evidence']['source_assessment.json'], 'realtime source audit differs')
    entries=index['episodes'];mapping={key(e):e for e in entries};outcomes=[];accepted=set()
    require(bool(entries) and len(entries) == len(mapping),'duplicate or empty realtime correction entries')
    for row,public,report,source in assessed_branches(audit['source_root'],protocol,
            index['evidence']['source_protocol.json'],assessment,context,sources,stats):
        outcomes.append(row)
        if not row['eligible']:continue
        accepted.add(key(row));require(key(row) in mapping,'successful teacher branch omitted from release')
        entry=mapping[key(row)];folder=contained(root,entry['path'])
        require(entry['split'] == 'train' and entry['eligible'] is True and entry['search_visibility_gated'] is True
                and entry['student_checkpoint_sha256'] == checkpoint_sha256
                and sha(folder/'navigation.npz') == entry['sha256'], 'realtime public tensors or training identity changed')
        meta=_read_pinned(folder/'episode.json',entry['metadata_sha256'],'realtime episode metadata')
        require(all(meta.get(k) == v for k,v in entry.items() if k != 'metadata_sha256')
            and meta.get('kind') == KIND and meta.get('schema') == CONTRACT
            and meta['instruction'] == report['instruction']
            and meta['source_branch_report_sha256'] == row['report_sha256'], 'realtime episode metadata changed')
        data=arrays(folder/'navigation.npz')
        require(set(data) == set(public) and all(np.array_equal(data[k],v) for k,v in public.items())
            and entry['frames'] == row['decisions']+1
            and all(entry[k] == row[k] for k in ('sample_starts','actual_student_anchor_starts','teacher_continuation_starts')),
            'released tensors or eligible paired windows differ from raw teacher evidence')
        prefix=entry['realtime_prefix']
        require(prefix['files'] == {n:source['receipt']['files'][n] for n in PREFIX_FILES},'source camera receipt changed')
        read_measured_prefix(contained(root,prefix['path']),entry['source_tick'],expected_files=prefix['files'])
    require(accepted == set(mapping) and outcomes == audit['outcomes'],
            'realtime release omitted rejected outcomes or introduced unassessed branches')
    return entries
