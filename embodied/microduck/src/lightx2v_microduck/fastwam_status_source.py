"""Read-only H200 collector. Executed over SSH; only emits bounded public metrics."""
import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path('/mtc/fuhaiwen1/microduck-tinyvla/world-action')
READ_DIGESTS = {}
CONTINUATION_STEPS = {
    'action-scope-continuation-pilot-v2': (1000, 750),
    'action-scope-epoch-pilot-v2': (3000, 2000),
    'retention-action-pilot-v2': (250, 250),
    'retention-strong-pilot-v1': (250, 250),
    'retained-corrective-pilot-v1': (250, 250),
    'memory-continuation-pilot-v1': (1000, 750),
    'memory-continuation-2000-pilot-v1': (2000, 1000),
}
MEMORY_CONTINUATIONS = {
    'memory-continuation-v1': dict(run_id='memory-continuation-pilot-v1',
        postchecks='memory-continuation-postchecks-v1', source_step=250, target_step=1000),
    'memory-continuation-2000-v1': dict(run_id='memory-continuation-2000-pilot-v1',
        postchecks='memory-continuation-2000-postchecks-v1', source_step=1000, target_step=2000),
}
MEMORY_RUN_BINDINGS = {b['run_id']: dict(b, worker=w) for w, b in MEMORY_CONTINUATIONS.items()}
MEMORY_POSTCHECK_BINDINGS = {b['postchecks']: dict(b, worker=w) for w, b in MEMORY_CONTINUATIONS.items()}
MEMORY_CHECKPOINT_REVIEWS = {
    'memory-checkpoint-1500-postchecks-v1': dict(
        MEMORY_RUN_BINDINGS['memory-continuation-2000-pilot-v1'], checkpoint_step=1500,
        evaluation_id='memory-checkpoint-1500-pilot-v1-development'),
}


def current_worker_folder(stem, versions=(2, 1)):
    """Select an attempted revision; callers still verify registration and liveness.

    A broken new registration must be shown as invalid rather than borrowing a
    previous attempt's results. Merely preparing a directory does not select it.
    """
    base = ROOT / 'config/guarded-rehearsal-pilot-v1'
    for version in versions:
        folder = base / (stem + '-v' + str(version))
        if (folder / 'launch.json').exists():
            return folder
    return base / (stem + '-v' + str(versions[-1]))


def registered_expanded_plan(worker):
    """Resolve only a launched, registered selected-recipe job; this is not liveness."""
    limits = {'expanded-corrective-training-job-v1': (None, 1),
              'expanded-corrective-preflight-job-v1': (2, 2),
              'expanded-corrective-postchecks-v1': (0, 1),
              'expanded-corrective-training-job-v2': (None, 1),
              'expanded-corrective-preflight-job-v2': (2, 2),
              'expanded-corrective-postchecks-v2': (0, 1),
              'expanded-corrective-training-job-v3': (None, 1),
              'expanded-corrective-preflight-job-v3': (2, 2),
              'expanded-corrective-postchecks-v3': (0, 1),
              'expanded-corrective-training-job-v4': (None, 1),
              'expanded-corrective-preflight-job-v4': (2, 2),
              'expanded-corrective-postchecks-v4': (0, 1),
              'expanded-candidate-training-v1': (None, 1),
              'expanded-candidate-preflight-v1': (2, 2),
              'expanded-candidate-postchecks-v1': (0, 1)}
    if worker not in limits:
        return None
    folder = ROOT / 'config/guarded-rehearsal-pilot-v1' / worker
    try:
        plan, launch = read(folder / 'plan.json'), read(folder / 'launch.json')
        control = read(ROOT / 'config/gpu-execution-control.json') or {}
        permit = control.get('conditional_pipelines', {}).get(worker, {})
        if not plan or not launch:
            return None
        steps = plan.get('training_steps', plan.get('final_global_step'))
        maximum, allocations = limits[worker]
        maximum = steps if maximum is None else maximum
        if (type(steps) is not int or steps < 250
                or permit.get('plan_sha256') != identity(folder / 'plan.json')['sha256']
                or launch.get('plan_sha256') != permit['plan_sha256']
                or plan.get('maximum_optimizer_steps') != maximum
                or permit.get('maximum_optimizer_steps') != maximum
                or plan.get('maximum_gpu_allocations') != allocations
                or permit.get('maximum_gpu_allocations') != allocations):
            return None
        output = Path(plan['train_output']).resolve()
        run_version = 2 if worker.endswith(('-v3', '-v4')) else 1
        run_prefix = ('expanded-candidate-corrective-pilot-v1-'
                      if worker.startswith('expanded-candidate-')
                      else f'expanded-corrective-pilot-v{run_version}-')
        if (output.parent != (ROOT / 'runs').resolve()
                or output.name not in {run_prefix + suffix
                                       for suffix in ('250', '1000', 'epoch')}):
            return None
        suffix = output.name.rsplit('-', 1)[-1]
        if suffix != 'epoch' and int(suffix) != steps:
            return None
        return dict(plan=plan, folder=folder, run_id=output.name, steps=steps)
    except (OSError, ValueError, KeyError, TypeError):
        return None


def registered_axis_run_name(name):
    """Resolve a verified axis job to its recorded run; this does not establish liveness."""
    match = re.fullmatch(
        r'microduck-fastwam-memory-axis-(control|weighted)-(preflight-v2-(preflight|resume)|training-v1-train)',
        name)
    if not match:
        return None
    arm = match.group(1)
    role = match.group(3) or 'train'
    worker = f'memory-axis-{arm}-' + ('training-v1' if role == 'train' else 'preflight-v2')
    folder = ROOT / 'config/guarded-rehearsal-pilot-v1' / worker
    try:
        plan, launch = read(folder / 'plan.json'), read(folder / 'launch.json')
        control = read(ROOT / 'config/gpu-execution-control.json') or {}
        permit = control.get('conditional_pipelines', {}).get(worker, {})
        if not plan or not launch:
            return None
        source = folder / ('run.py' if role == 'train' else 'run-verified.py')
        maximum, allocations = (250, 1) if role == 'train' else (2, 2)
        expected_schema = ('microduck_verified_axis_training_job_v1' if role == 'train'
                           else 'microduck_axis_two_step_restore_v1')
        if (plan.get('schema') != expected_schema or plan.get('training_steps') != 250
                or plan.get('arm') != arm
                or plan.get('axis_config') != dict(axis_order=['vx', 'vy', 'yaw', 'head_yaw', 'stand'],
                                                  weights=[1, 8 if arm == 'weighted' else 1, 1, 1, 1])
                or launch.get('plan_sha256') != identity(folder / 'plan.json')['sha256']
                or permit.get('plan_sha256') != launch.get('plan_sha256')
                or launch.get('worker_sha256') != hashlib.sha256(source.read_bytes()).hexdigest()
                or plan.get('files', {}).get(str(source)) != launch.get('worker_sha256')
                or any(v.get('maximum_optimizer_steps') != maximum
                       or v.get('maximum_gpu_allocations') != allocations
                       or v.get('model_promotion_allowed') is not False for v in (plan, permit))):
            return None
        suffix = {'train': '', 'preflight': '-preflight', 'resume': '-restore'}[role]
        output = ROOT / 'runs' / f'memory-axis-{arm}-pilot-v1{suffix}'
        if plan.get(role + '_output') != str(output):
            return None
        return 'microduck-fastwam-' + output.name
    except (OSError, ValueError, TypeError, KeyError):
        return None


def registered_axis_evaluation_name(name):
    """Resolve only the registered arm's evaluation, without implying a result."""
    match = re.fullmatch(r'microduck-fastwam-memory-axis-(control|weighted)-postchecks-v1-evaluate', name)
    if not match:
        return None
    arm = match.group(1)
    folder = ROOT/'config/guarded-rehearsal-pilot-v1'/f'memory-axis-{arm}-postchecks-v1'
    output = ROOT/'reports'/f'memory-axis-{arm}-pilot-v1-development'
    try:
        plan, launch = read(folder/'plan.json'), read(folder/'launch.json')
        permit = (read(ROOT/'config/gpu-execution-control.json') or {}).get(
            'conditional_pipelines', {}).get(folder.name, {})
        if (not plan or not launch or plan.get('schema') != 'microduck_axis_memory_development_postchecks_v1'
                or plan.get('axis_config') != dict(axis_order=['vx', 'vy', 'yaw', 'head_yaw', 'stand'],
                                                  weights=[1, 8 if arm == 'weighted' else 1, 1, 1, 1])
                or plan.get('evaluation_output') != str(output)
                or plan.get('train_output') != str(ROOT/'runs'/f'memory-axis-{arm}-pilot-v1')
                or plan.get('original_worker') != str(folder.parent/f'memory-axis-{arm}-training-v1')
                or plan.get('paired_development_cases') != 81
                or plan.get('final_global_step') != 250 or plan.get('new_optimizer_steps') != 250
                or launch.get('plan_sha256') != identity(folder/'plan.json')['sha256']
                or permit.get('plan_sha256') != launch['plan_sha256']
                or launch.get('worker_sha256') != hashlib.sha256((folder/'run.py').read_bytes()).hexdigest()
                or plan.get('bound_files', {}).get(str(folder/'run.py')) != launch['worker_sha256']
                or any(x.get('maximum_optimizer_steps') != 0 or x.get('maximum_gpu_allocations') != 1
                       for x in (plan, permit))
                or permit.get('model_promotion_allowed') is not False):
            return None
        return 'microduck-fastwam-'+output.name
    except (OSError, ValueError, TypeError, KeyError):
        return None


def registered_memory_continuation_name(name):
    """Resolve the registered full-state continuation, without implying completion."""
    worker_name = next((w for w in MEMORY_CONTINUATIONS
                        if name == 'microduck-fastwam-'+w+'-train'), None)
    if worker_name is None:
        return None
    binding = MEMORY_CONTINUATIONS[worker_name]
    source_step, target_step = binding['source_step'], binding['target_step']
    folder = ROOT / 'config/guarded-rehearsal-pilot-v1' / worker_name
    output = ROOT / 'runs' / binding['run_id']
    try:
        plan, launch = read(folder / 'plan.json'), read(folder / 'launch.json')
        permit = (read(ROOT / 'config/gpu-execution-control.json') or {}).get(
            'conditional_pipelines', {}).get(folder.name, {})
        worker = folder / 'run.py'
        if (not plan or not launch
                or plan.get('schema') != 'microduck_verified_full_state_continuation_v1'
                or plan.get('source_step') != source_step or plan.get('source_batch_in_epoch') != source_step*8
                or plan.get('target_step') != target_step or plan.get('training_steps') != target_step
                or plan.get('train_output') != str(output)
                or launch.get('plan_sha256') != identity(folder / 'plan.json')['sha256']
                or permit.get('plan_sha256') != launch.get('plan_sha256')
                or launch.get('worker_sha256') != hashlib.sha256(worker.read_bytes()).hexdigest()
                or plan.get('files', {}).get(str(worker)) != launch.get('worker_sha256')
                or any(x.get('maximum_optimizer_steps') != target_step-source_step
                       or x.get('maximum_gpu_allocations') != 1
                       or x.get('model_promotion_allowed') is not False for x in (plan, permit))):
            return None
        return 'microduck-fastwam-' + output.name
    except (OSError, ValueError, TypeError, KeyError):
        return None


def registered_memory_continuation_evaluation(name):
    post_name = next((w for w in MEMORY_POSTCHECK_BINDINGS
                      if name == 'microduck-fastwam-'+w+'-evaluate'), None)
    if post_name is None:
        return None
    binding = MEMORY_POSTCHECK_BINDINGS[post_name]
    source_step, target_step = binding['source_step'], binding['target_step']
    folder = ROOT / 'config/guarded-rehearsal-pilot-v1' / post_name
    try:
        plan, launch = read(folder/'plan.json'), read(folder/'launch.json')
        permit = (read(ROOT/'config/gpu-execution-control.json') or {}).get(
            'conditional_pipelines', {}).get(folder.name, {})
        if (not plan or not launch
                or plan.get('schema') != 'microduck_full_state_memory_development_postchecks_v1'
                or plan.get('source_step') != source_step or plan.get('final_global_step') != target_step
                or plan.get('new_optimizer_steps') != target_step-source_step or plan.get('paired_development_cases') != 81
                or plan.get('axis_config') != dict(axis_order=['vx','vy','yaw','head_yaw','stand'],weights=[1]*5)
                or plan.get('train_output') != str(ROOT/'runs'/binding['run_id'])
                or plan.get('original_worker') != str(folder.parent/binding['worker'])
                or plan.get('evaluation_output') != str(ROOT/'reports'/(binding['run_id']+'-development'))
                or launch.get('plan_sha256') != identity(folder/'plan.json')['sha256']
                or permit.get('plan_sha256') != launch.get('plan_sha256')
                or launch.get('worker_sha256') != hashlib.sha256((folder/'run.py').read_bytes()).hexdigest()
                or plan.get('bound_files', {}).get(str(folder/'run.py')) != launch.get('worker_sha256')
                or any(x.get('maximum_optimizer_steps') != 0 or x.get('maximum_gpu_allocations') != 1
                       for x in (plan, permit)) or permit.get('model_promotion_allowed') is not False):
            return None
        return 'microduck-fastwam-'+binding['run_id']+'-development'
    except (OSError, ValueError, TypeError, KeyError):
        return None


def realtime_correction_allocation(container):
    """Resolve a registered allocation to its actual output, without scoring it.

    These jobs use realtime-collected training data; their candidate validation
    still runs offline. A worker name alone establishes neither mode nor run ID.
    """
    name = container.get('Name', '').lstrip('/')
    match = re.fullmatch(r'microduck-fastwam-(realtime-correction-'
                         r'(model-preflight|candidate-training|candidate-postchecks)-v[1-9][0-9]*)-'
                         r'(preflight|resume|train|evaluate)', name)
    if not match:
        return None
    worker, kind, role = match.groups()
    definitions = {
        'model-preflight': ('microduck_realtime_model_two_update_restore_queue_v1', 2, 2, ('preflight', 'resume')),
        'candidate-training': ('microduck_realtime_correction_candidate_queue_v1', 250, 1, ('train',)),
        'candidate-postchecks': ('microduck_current_correction_development_postchecks_v1', 0, 1, ('evaluate',)),
    }
    schema, steps, allocations, roles = definitions[kind]
    try:
        if role not in roles:
            return None
        folder = ROOT / 'config/guarded-rehearsal-pilot-v1' / worker
        queue_path = folder / ('plan.json' if role == 'evaluate' else 'queue-plan.json')
        queue, handle = read(queue_path), read(folder / 'launch.json')
        permit = (read(ROOT / 'config/gpu-execution-control.json') or {}).get('conditional_pipelines', {}).get(worker, {})
        digest = identity(queue_path)['sha256']
        if (queue['schema'] != schema or handle['plan_sha256'] != digest
                or permit.get('plan_sha256') != digest
                or handle['worker_sha256'] != hashlib.sha256((folder / 'run.py').read_bytes()).hexdigest()
                or any(row.get('maximum_optimizer_steps') != steps
                       or row.get('maximum_gpu_allocations') != allocations
                       or row.get('model_promotion_allowed') is not False for row in (queue, permit))):
            return None
        launch = read(folder / (role + '-launch.json'))
        if (launch['container_id'] != container['Id'] or container['Image'] != queue['image']
                or container['Config']['User'] != '1031:1031'):
            return None
        argv = launch['argv']
        image_index = argv.index(queue['image'])
        if argv.count(queue['image']) != 1 or argv[image_index + 1:] != container['Config']['Cmd']:
            return None
        devices = container['HostConfig'].get('DeviceRequests') or []
        if (len(devices) != 1 or devices[0].get('DeviceIDs') != [launch['gpu_uuid']]
                or not any('gpu' in group for group in devices[0].get('Capabilities', []))):
            return None
        if role == 'evaluate':
            if (launch['plan_sha256'] != digest or queue['paired_development_cases'] != 81
                    or queue['physical_development_layouts'] != 27 or queue['formal_acceptance'] is not False):
                return None
            output = Path(queue['evaluation_output']).resolve()
            parent = ROOT / 'reports'
        else:
            execution_path = folder / 'execution-plan.json'
            execution = read(execution_path)
            execution_digest = identity(execution_path)['sha256']
            if (launch['execution_plan_sha256'] != execution_digest
                    or execution['files'].get(str(queue_path)) != digest
                    or execution['image'] != queue['image']
                    or '--plan-sha256' not in container['Config']['Cmd']
                    or container['Config']['Cmd'][container['Config']['Cmd'].index('--plan-sha256') + 1] != execution_digest):
                return None
            output = Path(execution[role + '_output']).resolve()
            parent = ROOT / 'runs'
            if role == 'train' and output != Path(queue['train_output']).resolve():
                return None
        if output.parent != parent.resolve() or not output.name.startswith('realtime-current-memory-'):
            return None
        return dict(name='microduck-fastwam-' + output.name, run_id=output.name,
                    activity='offline_evaluation' if role == 'evaluate' else 'training' if role == 'train' else 'preflight',
                    identity_verified=True, formal_acceptance=False, model_promoted=False)
    except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError):
        return None


def dashboard_container_name(name, *, container=None):
    """Map stage container names to existing public run IDs, preserving raw names separately."""
    name = name.lstrip('/')
    if name.startswith('microduck-fastwam-realtime-correction-'):
        binding = realtime_correction_allocation(container) if container else None
        return binding['name'] if binding else name
    if re.fullmatch(r'microduck-fastwam-memory-current(?:-three-color)?-candidate-postchecks-v[12]-evaluate', name):
        try:
            current = current_correction_postcheck_registration(three_color='-three-color-' in name)
            if current and name == 'microduck-fastwam-' + current['folder'].name + '-evaluate':
                return 'microduck-fastwam-' + Path(current['plan']['evaluation_output']).name
        except (OSError, ValueError, TypeError, KeyError, IndexError):
            return name
    axis_name = (registered_axis_run_name(name) or registered_axis_evaluation_name(name)
                 or registered_memory_continuation_name(name)
                 or registered_memory_continuation_evaluation(name))
    if axis_name:
        return axis_name
    if name in {'microduck-fastwam-'+w+'-evaluate' for w in MEMORY_CHECKPOINT_REVIEWS}:
        try:
            selected = registered_checkpoint_review()
            if selected and name == 'microduck-fastwam-'+selected['folder'].name+'-evaluate':
                return 'microduck-fastwam-'+Path(selected['plan']['evaluation_output']).name
        except (OSError, ValueError, TypeError, KeyError, IndexError):
            pass
    expanded_stages = {
        f'microduck-fastwam-expanded-corrective-{job}-v{version}-{stage}':
            (f'expanded-corrective-{job}-v{version}', key, parent,
             ending + '-v4' if version == 4 and job == 'preflight-job' else ending)
        for version in (1, 2, 3, 4)
        for job, stage, key, parent, ending in [
            ('training-job', 'train', 'train_output', 'runs', ''),
            ('preflight-job', 'preflight', 'preflight_output', 'runs', '-preflight'),
            ('preflight-job', 'resume', 'resume_output', 'runs', '-resume'),
            ('postchecks', 'evaluate', 'evaluation_output', 'reports', '-development'),
        ]
    }
    expanded_stages.update({
        'microduck-fastwam-expanded-candidate-training-v1-train':
            ('expanded-candidate-training-v1', 'train_output', 'runs', ''),
        'microduck-fastwam-expanded-candidate-preflight-v1-preflight':
            ('expanded-candidate-preflight-v1', 'preflight_output', 'runs', '-preflight-v1'),
        'microduck-fastwam-expanded-candidate-preflight-v1-resume':
            ('expanded-candidate-preflight-v1', 'resume_output', 'runs', '-resume-v1'),
        'microduck-fastwam-expanded-candidate-postchecks-v1-evaluate':
            ('expanded-candidate-postchecks-v1', 'evaluation_output', 'reports', '-development'),
    })
    if name in expanded_stages:
        worker, key, parent, ending = expanded_stages[name]
        binding = registered_expanded_plan(worker)
        if binding and key in binding['plan']:
            output = Path(binding['plan'][key]).resolve()
            if output.parent == (ROOT / parent).resolve() and output.name == binding['run_id'] + ending:
                return 'microduck-fastwam-' + output.name
    if name == 'microduck-fastwam-memory-candidate-postchecks-v1-evaluate':
        folder = ROOT/'config/guarded-rehearsal-pilot-v1/memory-candidate-postchecks-v1'
        try:
            plan, handle = read(folder/'plan.json'), read(folder/'launch.json')
            permit = (read(ROOT/'config/gpu-execution-control.json') or {}).get(
                'conditional_pipelines', {}).get(folder.name, {})
            output = ROOT/'reports/target-memory-paired-retention-pilot-v1-development'
            if (plan and handle and plan.get('schema') == 'microduck_paired_memory_development_postchecks_v1'
                    and plan.get('evaluation_output') == str(output) and plan.get('paired_development_cases') == 81
                    and plan.get('final_global_step') == 250
                    and permit.get('plan_sha256') == handle.get('plan_sha256') == identity(folder/'plan.json')['sha256']
                    and handle.get('worker_sha256') == hashlib.sha256((folder/'run.py').read_bytes()).hexdigest()
                    and all(v.get('maximum_optimizer_steps') == 0 and v.get('maximum_gpu_allocations') == 1
                            for v in (plan, permit))):
                return 'microduck-fastwam-' + output.name
        except (OSError, ValueError, KeyError, TypeError):
            pass
    return {
        'microduck-fastwam-action-scope-pilot-v1-train': 'microduck-fastwam-action-scope-pilot-v1',
        'microduck-fastwam-action-scope-pilot-v1-evaluate': 'microduck-fastwam-action-scope-pilot-v1-development',
        'microduck-fastwam-action-scope-corrective-pilot-v1-train': 'microduck-fastwam-action-scope-corrective-pilot-v1',
        'microduck-fastwam-action-scope-corrective-pilot-v1-evaluate': 'microduck-fastwam-action-scope-corrective-pilot-v1-development',
        'microduck-fastwam-action-scope-corrective-postchecks-v2-evaluate': 'microduck-fastwam-action-scope-corrective-pilot-v1-development',
        'microduck-fastwam-action-scope-continuation-pilot-v2-train': 'microduck-fastwam-action-scope-continuation-pilot-v2',
        'microduck-fastwam-action-scope-continuation-postchecks-v2-evaluate': 'microduck-fastwam-action-scope-continuation-pilot-v2-development',
        'microduck-fastwam-action-scope-epoch-pilot-v2-train': 'microduck-fastwam-action-scope-epoch-pilot-v2',
        'microduck-fastwam-action-scope-epoch-postchecks-v2-evaluate': 'microduck-fastwam-action-scope-epoch-pilot-v2-development',
        'microduck-fastwam-action-scope-epoch-postchecks-v3-evaluate': 'microduck-fastwam-action-scope-epoch-pilot-v2-development',
        'microduck-fastwam-retention-training-preflight-job-v1-train': 'microduck-fastwam-retention-training-preflight-v1',
        'microduck-fastwam-retention-training-preflight-job-v1-resume': 'microduck-fastwam-retention-resume-preflight-v1',
        'microduck-fastwam-retention-training-preflight-job-v2-train': 'microduck-fastwam-retention-training-preflight-v2',
        'microduck-fastwam-retention-training-preflight-job-v2-resume': 'microduck-fastwam-retention-resume-preflight-v2',
        'microduck-fastwam-retention-action-pilot-v2-train': 'microduck-fastwam-retention-action-pilot-v2',
        'microduck-fastwam-retention-action-postchecks-v1-evaluate': 'microduck-fastwam-retention-action-pilot-v2-development',
        'microduck-fastwam-retention-strong-preflight-job-v1-train': 'microduck-fastwam-retention-strong-preflight-v1',
        'microduck-fastwam-retention-strong-preflight-job-v1-resume': 'microduck-fastwam-retention-strong-resume-v1',
        'microduck-fastwam-retention-strong-pilot-v1-train': 'microduck-fastwam-retention-strong-pilot-v1',
        'microduck-fastwam-retention-strong-postchecks-v1-evaluate': 'microduck-fastwam-retention-strong-pilot-v1-development',
        'microduck-fastwam-retained-corrective-preflight-job-v1-train': 'microduck-fastwam-retained-corrective-pilot-v1-preflight',
        'microduck-fastwam-retained-corrective-preflight-job-v1-resume': 'microduck-fastwam-retained-corrective-pilot-v1-resume',
        'microduck-fastwam-retained-corrective-pilot-v1-train': 'microduck-fastwam-retained-corrective-pilot-v1',
        'microduck-fastwam-retained-corrective-postchecks-v1-evaluate': 'microduck-fastwam-retained-corrective-pilot-v1-development',
    }.get(name, name)


def container_summary(container):
    state = container['State']
    binding = realtime_correction_allocation(container)
    correction = container['Name'].lstrip('/').startswith('microduck-fastwam-realtime-correction-')
    return {'name': dashboard_container_name(container['Name'], container=container),
            'docker_name': container['Name'].lstrip('/'), 'id': container['Id'],
            'model_family': container_model_family(container),
            'activity': binding['activity'] if binding else 'unverified' if correction else None,
            'running': state['Running'], 'state': state['Status'],
            'pid_live': bool(state['Pid'] and Path('/proc', str(state['Pid'])).exists()),
            'exit_code': None if state['Running'] else state['ExitCode'], 'started_at': state['StartedAt']}


def container_model_family(container):
    """The common allocation prefix is a scheduler namespace, not a model type."""
    name = container.get('Name', '').lstrip('/')
    if not name.startswith('microduck-fastwam-'):
        return None
    declared = (container.get('Config', {}).get('Labels') or {}).get('microduck.family')
    for family in ('vlm', 'hybrid'):
        if name.startswith('microduck-fastwam-' + family + '-'):
            return family if declared == family else 'unclassified'
    return 'fastwam' if declared in (None, 'fastwam') else 'unclassified'


def model_gpu_accounting(containers, ledger, now):
    """Conserve all project allocation time while separating model families."""
    seconds = dict(fastwam=0., vlm=0., hybrid=0., unclassified=0.)
    for container in containers:
        family = container_model_family(container)
        if family is None:
            continue
        count = sum(len(d.get('DeviceIDs') or []) for d in container.get('HostConfig', {}).get('DeviceRequests') or []
                    if any('gpu' in group for group in d.get('Capabilities', [])))
        state = container['State']
        if count and not state['StartedAt'].startswith('0001'):
            end = now if state['Running'] else docker_timestamp(state['FinishedAt'])
            seconds[family] += max(0, end - docker_timestamp(state['StartedAt'])) * count
    legacy_removed = 0.
    for item in ledger.get('reported_removed_container_checks', []):
        # The existing ledger predates the VLM experiments and belongs to
        # FastWAM. New removed-container receipts can carry an explicit family.
        family = item.get('model_family', 'fastwam')
        family = family if family in seconds else 'unclassified'
        value = item.get('allocated_gpu_seconds', 0)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError('invalid removed-container GPU accounting')
        seconds[family] += value
        if 'model_family' not in item:
            legacy_removed += value
    return dict(schema='microduck_model_gpu_accounting_v1',
                by_family={k: v / 3600 for k, v in seconds.items()}, total_gpu_hours=sum(seconds.values()) / 3600,
                legacy_removed_fastwam_gpu_hours=legacy_removed / 3600, observed_unix=now,
                basis='actual_GPU_device_allocation_time_including_load_and_inference_not_task_success')


def teacher_current_experiment():
    path = ROOT / 'reports/exports/teacher-cpu-status-v1.json'
    worker = read(path)
    if not worker or not worker.get('current_experiment'):
        return None
    try:
        e = worker['current_experiment']
        observed = e['observed_unix']
        registered = {
            'dense-search-grid-v1': ('b5bb3361e18488c2f836a43dfe26aa9247bd11ee05efbe1109907f723376edd7', 120, ['room_curriculum_one', 'room_curriculum_two']),
            'spatial-utility-expanded-v1': ('fdebed30d7ee789869dc117ac8f466e5f79eab498471b370260da13761a71cdf', 60, ['room_curriculum_one']),
            'spatial-utility-qualification-v1': ('9fd3cd64d3a00411ec26de036d161e316bbdf3f1d8a51fa0ced8c423024d296c', 300, ['room_curriculum_one']),
            'physics-shortlist-pilot-v1': ('57b663c250750c58657f3ce818dac79f88be634506c18ae77ca1da146c2a0796', 9, ['room_curriculum_one']),
            'occlusion-view-pilot-v1': ('555fc44a7070bda867bc628c651d6df95e18b3f6db37cc94d7fd5b2f8e042707', 9, ['room_curriculum_one']),
            'occlusion-view-expanded-v1': ('fb58317961974dabb34aa8da6cb16d33068dd0fda42066f132b9080e14321eb9', 60, ['room_curriculum_one']),
            'occlusion-view-qualification-v1': ('bdedcc03e5951bfa80a85358d39b27a9943ebec72f105765973bb146c49d96eb', 300, ['room_curriculum_one']),
            'guarded-occlusion-qualification-v1': ('337ade5aba8691a50d69ed6afbdd8211dbdf09ec8c6195ff17316683ecd245fd', 300, ['room_curriculum_one']),
        }
        digest, total, scenes = registered[e['id']]
        if (worker.get('schema') != 'microduck_cpu_teacher_progress_v1'
                or e['protocol_sha256'] != digest
                or type(observed) not in (int, float) or not math.isfinite(observed)
                or observed > time.time() + 5 or observed != worker['observed_unix']
                or e['model_acceptance'] is not False or e['training_release_allowed'] is not False
                or e['total'] != total or len(e['rows']) != len(scenes)
                or [r['scene'] for r in e['rows']] != scenes):
            raise ValueError('invalid current CPU experiment')
        rows = []
        per_scene = total // len(scenes)
        for r in e['rows']:
            if (type(r['completed']) is not int or not 0 <= r['completed'] <= per_scene or r['total'] != per_scene
                    or r['state'] not in ('running', 'auditing', 'queued', 'complete', 'failed', 'unavailable')
                    or (r['state'] in ('running', 'auditing') and r['verified_unix'] != observed)
                    or (r['state'] == 'queued' and r['completed'] != 0)
                    or (r['state'] in ('complete', 'auditing') and r['completed'] != per_scene)):
                raise ValueError('invalid current CPU counts')
            rows.append({k: r[k] for k in ('scene', 'state', 'completed', 'total')})
        if e['completed'] != sum(r['completed'] for r in rows):
            raise ValueError('invalid aggregate count')
        return dict(state='snapshot', id=e['id'], observed_unix=observed, rows=rows,
                    completed=e['completed'], total=total, source=identity(path),
                    model_acceptance=False, training_release_allowed=False)
    except (KeyError, TypeError, ValueError):
        return dict(state='invalid', model_acceptance=False, training_release_allowed=False)


def occlusion_teacher_comparison(progress):
    if (progress.get('guarded_occlusion_teacher_expansion') or {}).get('cohort_complete'):
        return guarded_teacher_comparison(progress)
    descriptor = (progress.get('teacher_occlusion_search_development') or {}).get('full_comparison')
    if not descriptor:
        return None
    path = ROOT / 'reports/exports/occlusion-view-expanded-assessment-v2.json'
    try:
        report = read(path)
        if (report is None or identity(path)['sha256'] != descriptor['assessment_sha256']
                or report['status'] != 'complete' or report['expansion_gate_passed'] is not True
                or report['protocol_sha256'] != 'fb58317961974dabb34aa8da6cb16d33068dd0fda42066f132b9080e14321eb9'
                or report['source_manifest_sha256'] != '5af90f87772489a6e67bd707d7fa7774e9b3a53e77f4cbcc511a060c68887cb8'
                or report['execution_plan_sha256'] != '53556ba491ba36de3b8f1148cce39ee65f9cb45d079ecac7079bc096077077e4'
                or report['teacher_qualified'] is not False or report['formal_acceptance'] is not False):
            raise ValueError('unverified teacher development comparison')
        tasks = ('find_red_ball', 'find_blue_ball', 'find_yellow_ball')
        expected = {(s, t) for s in range(2300, 2320) for t in tasks}
        rows = report['rows']
        if (len(rows) != 60 or {(r['seed'], r['task']) for r in rows} != expected
                or report['paired_episodes'] != 60 or report['distinct_layouts'] != 20):
            raise ValueError('incomplete teacher development comparison')
        for r in rows:
            if (type(r['baseline_success']) is not bool or type(r['candidate_success']) is not bool
                    or r['candidate_collision'] or r['candidate_fall'] or r['candidate_recovery_events']
                    or r['candidate_recovery_active'] or r['candidate_fault']
                    or not 0 < r['verified_controls'] == r['candidate_final_tick'] <= 1500
                    or r['verified_rgb_frames'] != r['verified_controls'] // 5 + 1):
                raise ValueError('unsafe or incompletely replayed teacher case')
        baseline = sum(r['baseline_success'] for r in rows)
        candidate = sum(r['candidate_success'] for r in rows)
        gains = sum(not r['baseline_success'] and r['candidate_success'] for r in rows)
        regressions = sum(r['baseline_success'] and not r['candidate_success'] for r in rows)
        if (baseline != report['baseline_successes'] or candidate != report['candidate_successes']
                or gains != report['gains'] or regressions != report['regressions']
                or baseline != 56 or gains <= 0 or regressions != 0 or report['unsafe_episodes'] != 0
                or sum(r['verified_controls'] for r in rows) != report['verified_control_ticks']
                or sum(r['verified_rgb_frames'] for r in rows) != report['verified_rgb_frames']):
            raise ValueError('teacher comparison summary disagrees with complete records')
        return dict(state='verified', baseline=baseline, candidate=candidate, total=60, layouts=20,
            gains=gains, regressions=0, unsafe_episodes=0, observed_unix=report['checked_unix'],
            colors=[dict(task=t, baseline=sum(r['baseline_success'] for r in rows if r['task'] == t),
                         candidate=sum(r['candidate_success'] for r in rows if r['task'] == t), total=20)
                    for t in tasks], source=identity(path),
            teacher_qualification=False, learned_model_acceptance=False, hardware_validated=False)
    except (KeyError, TypeError, ValueError, OSError):
        return dict(state='invalid', teacher_qualification=False, learned_model_acceptance=False)


def guarded_teacher_comparison(progress):
    descriptor = progress['guarded_occlusion_teacher_expansion']
    path = ROOT / 'reports/exports/extended-navigation-training-readiness-v1/guarded-occlusion-teacher-expanded-closure-v1.json'
    state_path = path.with_name('guarded-occlusion-teacher-expanded-state-v1.json')
    try:
        report, state = read(path), read(state_path)
        if (report is None or state is None or identity(path)['sha256'] != descriptor['closure_sha256']
                or identity(state_path)['sha256'] != report['source_state_sha256']
                or report['status'] != 'complete' or state['status'] != 'complete'
                or state['protocol_sha256'] != 'ad5db10219aa0ffd640eb6b2c221996bae0fd60302464d2ce2bbe242c8addcd7'
                or report['teacher_qualified'] is not False or report['formal_acceptance'] is not False
                or report['training_release_allowed'] is not False):
            raise ValueError('unverified guarded development evidence')
        tasks = ('find_red_ball', 'find_blue_ball', 'find_yellow_ball')
        rows = state['rows']
        if len(rows) != 60 or {(r['seed'], r['task']) for r in rows} != {(s, t) for s in range(2300, 2320) for t in tasks}:
            raise ValueError('incomplete guarded development cohort')
        for row in rows:
            if (type(row['baseline_success']) is not bool or type(row['candidate_success']) is not bool
                    or row['final_evaluator_exact'] is not True or row['collision'] or row['fallen']
                    or row['recovery_events'] or row['fault']):
                raise ValueError('unsafe guarded development case')
        baseline = sum(r['baseline_success'] for r in rows)
        candidate = sum(r['candidate_success'] for r in rows)
        if (baseline != report['baseline_successes'] or baseline != 59
                or candidate != report['candidate_successes'] or candidate != 60
                or sum(r['exact_control_ticks'] for r in rows) != report['exact_control_ticks']
                or sum(r['exact_rgb_frames'] for r in rows) != report['exact_rgb_frames']
                or report['regressions'] != 0 or report['unsafe_cases'] != 0):
            raise ValueError('guarded summary differs from complete replay records')
        return dict(state='verified', baseline=baseline, candidate=candidate, total=60, layouts=20,
            gains=1, regressions=0, unsafe_episodes=0, observed_unix=report['observed_unix'],
            candidate_label='受约束的接近加速教师',
            colors=[dict(task=t, baseline=sum(r['baseline_success'] for r in rows if r['task'] == t),
                         candidate=sum(r['candidate_success'] for r in rows if r['task'] == t), total=20) for t in tasks],
            source=identity(path), teacher_qualification=False, learned_model_acceptance=False, hardware_validated=False)
    except (KeyError, TypeError, ValueError, OSError):
        return dict(state='invalid', teacher_qualification=False, learned_model_acceptance=False)


def pinned_teacher_evidence(descriptor):
    """Read one declared campaign artifact, retaining the digest of the bytes read."""
    path = ROOT / descriptor['path']
    path.resolve().relative_to(ROOT.resolve())
    value = read(path)
    evidence = identity(path)
    if value is None or evidence['sha256'] != descriptor['sha256']:
        raise ValueError('teacher evidence digest mismatch')
    return value, evidence


def guarded_teacher_qualification(progress):
    declaration = progress.get('guarded_occlusion_teacher_qualification') or {}
    if declaration.get('status') != 'complete':
        return None
    try:
        certificate, cert_source = pinned_teacher_evidence(declaration['certificate'])
        report, source = pinned_teacher_evidence(declaration['assessment'])
        tasks = ('find_red_ball', 'find_blue_ball', 'find_yellow_ball')
        if (certificate['schema'] != 'guarded_teacher_qualification_certificate_v1'
                or certificate['evidence']['assessment']['sha256'] != source['sha256']
                or certificate['evidence']['protocol']['sha256'] != report['protocol_sha256']
                or report['protocol_sha256'] != '337ade5aba8691a50d69ed6afbdd8211dbdf09ec8c6195ff17316683ecd245fd'
                or certificate['source_manifest_sha256'] != report['source_manifest_sha256']
                or certificate['qualification_plan_sha256'] != report['plan_sha256']
                or certificate['teacher_contract'] != report['action_contract']
                or certificate['eligible_for_separate_collection'] is not True
                or certificate['qualification_episodes_may_train'] is not False
                or certificate['training_release_allowed'] is not False
                or report['status'] != 'complete' or report['teacher_qualified'] is not True
                or report['qualification_passed'] is not True
                or report['training_release_allowed'] is not False
                or report['formal_acceptance'] is not False or report['hardware_validated'] is not False
                or report['minimum_successes_per_color'] != 95
                or set(certificate['qualification_seeds']) != set(range(420000, 420100))
                or len(set(certificate['qualification_geometry_sha256'])) != 100):
            raise ValueError('unverified teacher qualification')
        rows = report['rows']
        if (len(rows) != 300 or {(r['seed'], r['task']) for r in rows}
                != {(s, t) for s in range(420000, 420100) for t in tasks}):
            raise ValueError('incomplete teacher qualification')
        for row in rows:
            if (type(row['strict_success']) is not bool or row['final_evaluator_exact'] is not True
                    or row['collision'] or row['fallen'] or row['recovery_events'] or row['fault']
                    or not 0 < row['exact_control_ticks'] <= 1500
                    or row['exact_rgb_frames'] != (row['exact_control_ticks'] + 4) // 5):
                raise ValueError('unsafe or unreplayed qualification case')
        colors = {t: dict(successes=sum(r['strict_success'] for r in rows if r['task'] == t), total=100)
                  for t in tasks}
        if (colors != report['per_color'] or colors != certificate['per_color']
                or any(r['successes'] < 95 for r in colors.values())
                or sum(r['successes'] for r in colors.values()) != report['strict_successes']
                or report['total'] != 300 or report['distinct_layouts'] != 100 or report['unsafe_cases'] != 0
                or any(sum(r[k] for r in rows) != report[k] or report[k] != certificate[k]
                       for k in ('exact_control_ticks', 'exact_rgb_frames'))):
            raise ValueError('qualification summary differs from its complete cohort')
        return dict(state='verified', qualified=True, successes=report['strict_successes'], total=300,
                    layouts=100, minimum_per_color=95, observed_unix=report['checked_unix'],
                    colors=[dict(task=t, **colors[t]) for t in tasks], source=source,
                    certificate=cert_source, learned_model_acceptance=False, training_release_allowed=False)
    except (KeyError, TypeError, ValueError, OSError):
        return dict(state='invalid', learned_model_acceptance=False, training_release_allowed=False)


def guarded_demonstration_collection(progress, qualification, containers):
    declaration = progress.get('guarded_separate_collection')
    if not declaration:
        return None
    try:
        if not qualification or qualification['state'] != 'verified':
            raise ValueError('collection requires verified teacher qualification')
        plan, plan_source = pinned_teacher_evidence(declaration['plan'])
        batch, batch_source = pinned_teacher_evidence(declaration['batch'])
        path = ROOT / 'reports/guarded-separate-collection-v1/report.json'
        report = read(path)
        state = read(ROOT / 'reports/guarded-collection-handoff-v1/state.json')
        if (report['collection_plan_sha256'] != plan_source['sha256']
                or batch['collection_plan']['sha256'] != plan_source['sha256']
                or report['batch_sha256'] != batch_source['sha256']
                or report['qualification_certificate_sha256'] != qualification['certificate']['sha256']
                or plan['qualification_certificate']['sha256'] != qualification['certificate']['sha256']
                or plan['purpose'] != 'separate_teacher_demonstrations'
                or plan['ready_for_execution'] is not True or report['planned_cases'] != 300
                or any(report[k] is not False for k in ('gpu_allocated', 'new_training', 'model_promoted', 'formal_acceptance'))):
            raise ValueError('collection evidence belongs to another protocol')
        cases = {(r['seed'], r['task']): r['split'] for r in plan['cases']}
        tasks = ('find_red_ball', 'find_blue_ball', 'find_yellow_ball')
        expected = {(s, t) for s in range(430000, 430100) for t in tasks}
        if len(plan['cases']) != 300 or set(cases) != expected:
            raise ValueError('unexpected collection cohort')
        rows = report['rows']
        if len({(r['seed'], r['task']) for r in rows}) != len(rows):
            raise ValueError('duplicate collection outcomes')
        for row in rows:
            if ((row['seed'], row['task']) not in cases or cases.get((row['seed'], row['task'])) != row['split']
                    or row['plan_sha256'] != plan_source['sha256']
                    or row['purpose'] != 'separate_teacher_demonstrations'
                    or type(row['strict_success']) is not bool or type(row['candidate_for_release']) is not bool
                    or (row['candidate_for_release'] and not row['strict_success'])
                    or row['final_evaluator_exact'] is not True or row['public_tensors_exact'] is not True
                    or row['training_release_allowed'] is not False or row['formal_acceptance'] is not False):
                raise ValueError('invalid collection outcome')
        matched = [c for c in containers if c['id'] == declaration['container_id'] == state['container_id']
                   and c['started_at'] == state['container_state']['StartedAt']]
        live = len(matched) == 1 and matched[0]['running'] and matched[0]['pid_live']
        released, release_source = False, None
        if report['training_data_released'] is True:
            if report['status'] != 'complete' or len(rows) != 300:
                raise ValueError('partial collection cannot be released')
            index, release_source = pinned_teacher_evidence(dict(
                path=str(Path(report['release_output']) / 'index.json'), sha256=report['release_index_sha256']))
            eligible = {(r['seed'], r['task']) for r in rows if r['candidate_for_release']}
            if (index['kind'] != 'separate_guarded_teacher_episodes_v1'
                    or index['schema'] != 'microduck_navigation_forward04_v1'
                    or index['training_release_allowed'] is not True
                    or index['qualification_episodes_released'] is not False
                    or index['metadata_pinned'] is not True or index['collected_cases'] != 300
                    or index['evidence']['plan']['sha256'] != plan_source['sha256']
                    or index['evidence']['certificate']['sha256'] != qualification['certificate']['sha256']
                    or len(index['episodes']) != len(eligible)
                    or {(r['seed'], r['task']) for r in index['episodes']} != eligible):
                raise ValueError('release index differs from audited collection outcomes')
            released = True
        phase = 'complete' if released else 'failed' if report['status'] == 'failed' else (
            'auditing' if live and len(rows) == 300 else 'running' if live else 'unavailable')
        return dict(state='snapshot', phase=phase, live=bool(live), completed=len(rows), total=300,
                    layouts=100, successes=sum(r['strict_success'] for r in rows),
                    eligible=sum(r['candidate_for_release'] for r in rows),
                    failures=sum(not r['strict_success'] for r in rows),
                    observed_unix=time.time(), source=identity(path), plan=plan_source,
                    training_data_released=released, release_source=release_source,
                    learned_model_acceptance=False, new_training=False, gpu_allocated=False)
    except (KeyError, TypeError, ValueError, OSError):
        return dict(state='invalid', learned_model_acceptance=False, training_data_released=False)


def teacher_development(progress, containers=()):
    """Bounded CPU-teacher evidence; timestamps are snapshots, never live GPU claims."""
    qualification = progress.get('obstacle_curriculum_qualification')
    if not qualification:
        return None
    result = {'state': 'snapshot', 'observed_unix': progress.get('updated_unix'),
              'activity': str(progress.get('current_activity', ''))[:1000],
              'curricula': [], 'pilot': None, 'corrected_case': None,
              'learned_model_acceptance': False, 'hardware_validated': False,
              'current_experiment': teacher_current_experiment(),
              'completed_teacher_comparison': occlusion_teacher_comparison(progress)}
    result['qualified_teacher'] = guarded_teacher_qualification(progress)
    result['data_collection'] = guarded_demonstration_collection(progress, result['qualified_teacher'], containers)
    if result['qualified_teacher'] is not None:
        # A completed, audited cohort supersedes an older hourly progress mirror.
        result['current_experiment'] = None
    completed = result['completed_teacher_comparison']
    current = result['current_experiment']
    guarded = progress.get('guarded_occlusion_teacher_qualification') or {}
    if (completed and completed.get('candidate_label') and completed['state'] == 'verified'
            and guarded.get('status') == 'running'
            and guarded.get('protocol_sha256') == '337ade5aba8691a50d69ed6afbdd8211dbdf09ec8c6195ff17316683ecd245fd'
            and (current is None or (current.get('state') == 'snapshot'
                 and current['id'] != 'guarded-occlusion-qualification-v1'
                 and current['observed_unix'] < guarded.get('observed_unix', 0)))):
        result['current_experiment'] = dict(state='awaiting_snapshot', id='guarded-occlusion-qualification-v1',
            previous_source=current.get('source') if current else None,
            model_acceptance=False, training_release_allowed=False)
    successor = (progress.get('teacher_occlusion_search_development') or {}).get('independent_qualification') or {}
    if (result['qualified_teacher'] is None and completed and completed['state'] == 'verified'
            and successor.get('status') == 'collection_started'
            and successor.get('protocol_sha256') == 'bdedcc03e5951bfa80a85358d39b27a9943ebec72f105765973bb146c49d96eb'
            and successor.get('development_assessment_sha256') == completed['source']['sha256']
            and (current is None or (current.get('state') == 'snapshot'
                 and current['id'] != 'occlusion-view-qualification-v1'
                 and current['observed_unix'] < completed['observed_unix']))):
        # An hourly worker snapshot from the predecessor cannot override its
        # newer audited completion. Wait for successor counts instead of mixing cohorts.
        result['current_experiment'] = dict(state='awaiting_snapshot', id='occlusion-view-qualification-v1',
            previous_source=current.get('source') if current else None,
            model_acceptance=False, training_release_allowed=False)

    def pinned(folder, name, descriptor):
        path = ROOT / 'reports/exports' / folder / name
        if not isinstance(descriptor, dict) or not descriptor.get('sha256'):
            raise ValueError('missing pinned teacher evidence')
        data = read(path)
        if data is None or identity(path)['sha256'] != descriptor['sha256']:
            raise ValueError('teacher evidence changed or unavailable')
        return data, identity(path)

    try:
        protocol, evidence = pinned('curriculum-qualification-and-physical-guard-v1',
            'curriculum-qualification-protocol-v1.json', qualification.get('protocol'))
        if (protocol['scenes'] != ['room_curriculum_one', 'room_curriculum_two']
                or len(set(protocol['seeds'])) != 100 or len(set(protocol['tasks'])) != 3
                or protocol['episodes_per_scene'] != 300 or protocol['total_episodes'] != 600
                or protocol['minimum_eligible_success_rate_per_task'] != .95):
            raise ValueError('unexpected teacher qualification cohort')
        jobs = qualification.get('jobs', {})
        # An unrelated campaign report can be newer than the actual CPU
        # counts. Compare observations of these jobs, not the campaign clock.
        job_times = [job.get(key) for job in jobs.values()
                     for key in ('observed_unix', 'live_verified_unix', 'finished_unix')
                     if job.get(key) is not None]
        job_times.extend(qualification[key] for key in ('observed_unix', 'updated_unix')
                         if qualification.get(key) is not None)
        if any(type(t) not in (int, float) or not math.isfinite(t) for t in job_times):
            raise ValueError('invalid teacher observation timestamp')
        job_observed = max(job_times) if job_times else (result['observed_unix'] or 0)
        worker_path = ROOT / 'reports/exports/teacher-cpu-status-v1.json'
        worker = read(worker_path)
        if worker is not None:
            observed = worker.get('observed_unix')
            if (worker.get('schema') != 'microduck_cpu_teacher_progress_v1'
                    or worker.get('protocol_sha256') != evidence['sha256']
                    or type(observed) not in (int, float) or not math.isfinite(observed)
                    or observed > time.time() + 5
                    or worker.get('teacher_qualification') is not False
                    or worker.get('model_acceptance') is not False
                    or set(worker.get('jobs', {})) != set(protocol['scenes'])):
                raise ValueError('invalid local teacher status mirror')
            if observed >= job_observed:
                jobs = worker['jobs']
                result['observed_unix'] = observed
                result['worker_status_source'] = identity(worker_path)
        for scene, label in zip(protocol['scenes'], ('单障碍课程', '双障碍课程')):
            job = jobs.get(scene)
            completed = job.get('completed_episodes') if job else 0
            if type(completed) is not int or not 0 <= completed <= 300:
                raise ValueError('invalid completed teacher count')
            status = job.get('status') if job else 'queued'
            if status not in ('running', 'complete', 'failed', 'queued'):
                raise ValueError('invalid teacher worker state')
            result['curricula'].append({'scene': scene, 'label': label, 'state': status,
                'completed': completed, 'total': 300, 'layouts': 100, 'instructions': 3,
                'verified_unix': job.get('live_verified_unix') if job else None,
                'qualification_claimed': False})
        result['protocol_source'] = evidence
        result['completed'] = sum(row['completed'] for row in result['curricula'])
        result['total'] = 600
    except (ValueError, KeyError, TypeError, OSError):
        result.update(state='invalid', curricula=[], completed=None, total=None)

    gate_descriptor = progress.get('curriculum_qualification_upper_bound')
    if gate_descriptor:
        try:
            gate, gate_source = pinned('curriculum-gate-replay-v1',
                'curriculum-gate-replay-v1.json', gate_descriptor.get('assessment'))
            gate_protocol, _ = pinned('curriculum-gate-replay-v1',
                'curriculum-gate-replay-protocol-v1.json', gate_descriptor.get('protocol'))
            if result['state'] == 'invalid':
                raise ValueError('qualification protocol unavailable')
            cases = gate['cases']
            total = len(protocol['seeds'])
            required = math.ceil(total * protocol['minimum_eligible_success_rate_per_task'] - 1e-12)
            if (gate['status'] != 'complete' or gate['qualification_possible'] is not False
                    or gate['full_cohort_complete'] is not False
                    or gate['training_release_allowed'] is not False
                    or gate_protocol['qualification_protocol_sha256'] != result['protocol_source']['sha256']
                    or gate['protocol_sha256'] != gate_descriptor['protocol']['sha256']
                    or gate_protocol['scene'] not in protocol['scenes']
                    or gate_protocol['task'] not in protocol['tasks']
                    or gate['scope'] != gate_protocol['scene'] + '/' + gate_protocol['task']
                    or {(c['seed'], c['metadata_sha256']) for c in cases} != {
                        (c['seed'], c['metadata_sha256']) for c in gate_protocol['cases']}
                    or len(cases) != len({c['seed'] for c in cases})
                    or not cases or any(c['seed'] not in protocol['seeds']
                        or c['task'] != gate_protocol['task'] or c['strict_success'] is not False
                        or c['all_state61_and_raw_actions_exact'] is not True
                        or c['control_ticks_replayed'] != 1500 for c in cases)
                    or gate['verified_failures'] != len(cases)
                    or gate['planned_episodes_per_task'] != total
                    or gate['minimum_required_successes'] != required
                    or gate['maximum_possible_successes'] != total - len(cases)
                    or total - len(cases) >= required):
                raise ValueError('invalid negative qualification bound')
            result['qualification_upper_bound'] = {'state': 'verified',
                'scene': gate_protocol['scene'], 'task': gate_protocol['task'],
                'verified_failures': len(cases), 'total': total,
                'maximum_successes': total - len(cases), 'required_successes': required,
                'source': gate_source, 'full_cohort_complete': False}
        except (ValueError, KeyError, TypeError, OSError):
            result['qualification_upper_bound'] = {'state': 'invalid'}

    try:
        pilot, evidence = pinned('curriculum-qualification-and-physical-guard-v1',
            'curriculum-pilot-assessment-v1.json', (progress.get('obstacle_curriculum_pilot') or {}).get('assessment'))
        cohorts = pilot['cohorts']
        if pilot['status'] != 'complete' or set(cohorts) != {'room_curriculum_one', 'room_curriculum_two'}:
            raise ValueError('incomplete pilot')
        counts = {scene: row['successes'] for scene, row in cohorts.items()}
        if any(row['episodes'] != 9 or type(row['successes']) is not int or not 0 <= row['successes'] <= 9
               for row in cohorts.values()):
            raise ValueError('invalid pilot count')
        result['pilot'] = {'successes': counts, 'episodes_per_scene': 9, 'layouts_per_scene': 3,
                           'source': evidence, 'teacher_qualification': False}
    except (ValueError, KeyError, TypeError, OSError):
        pass  # Missing evidence is not a measured failure or a qualification.

    try:
        case, evidence = pinned('physical-guard-full-task-v1', 'physical-guard-blue-assessment-v1.json',
            (progress.get('physical_guard_teacher_probe') or {}).get('assessment'))
        if (case['status'] != 'complete' or case['baseline_strict_success'] is not False
                or case['candidate_strict_success'] is not True
                or case['candidate_first_observation_tick'] != 845 or case['candidate_final_tick'] != 1298
                or case['goal_hold_ticks'] < 100 or case['collision'] or case['fallen']
                or case['intervention_disqualified']):
            raise ValueError('corrected case does not match the audited full task')
        result['corrected_case'] = {'seed': 20, 'task': 'find_blue_ball', 'before': 'failed', 'after': 'passed',
            'first_observation_s': 16.9, 'completion_s': 25.96, 'source': evidence,
            'scope': 'one_development_teacher_case'}
    except (ValueError, KeyError, TypeError, OSError):
        pass
    result['comparison_state'] = (progress.get('physical_guard_teacher_pilot') or {}).get('status')
    return result


def docker_timestamp(value):
    """Normalize RFC3339Nano to microseconds for the H200 Python 3.8 runtime."""
    match = re.fullmatch(r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d{1,9}))?(Z|[+-]\d{2}:\d{2})', value)
    if not match:
        raise ValueError('invalid Docker timestamp')
    base, fraction, zone = match.groups()
    fraction = (fraction or '').ljust(6, '0')[:6]
    zone = '+00:00' if zone == 'Z' else zone
    return datetime.datetime.fromisoformat(base + '.' + fraction + zone).timestamp()


def evaluation_case_count(containers, run_id):
    """Read the actual argv cohort, not the historical seven-layout protocol."""
    matches = [c for c in containers if dashboard_container_name(c.get('Name', ''), container=c) == 'microduck-fastwam-' + run_id]
    if len(matches) != 1:
        return None
    argv = matches[0].get('Config', {}).get('Cmd') or []
    if '--protocol' in argv or '--protocol-sha256' in argv:
        # Explicit scene/case protocols supersede the old Cartesian CLI cohort.
        # Counts are progress only; this path never validates a task score.
        try:
            if (argv.count('--protocol') != 1 or argv.count('--protocol-sha256') != 1
                    or '--seeds' in argv or '--tasks' in argv):
                return None
            path = Path(argv[argv.index('--protocol') + 1]).resolve()
            path.relative_to(ROOT.resolve())
            protocol = read(path)
            if identity(path)['sha256'] != argv[argv.index('--protocol-sha256') + 1]:
                return None
            cases, contracts = protocol['cases'], protocol['contracts']
            if (not isinstance(cases, list) or not cases or not isinstance(contracts, list)
                    or not contracts or any(not isinstance(c, str) or not c for c in contracts)
                    or len(set(contracts)) != len(contracts)):
                return None
            keys = []
            for case in cases:
                if (type(case['seed']) is not int or not isinstance(case['scene'], str)
                        or not case['scene'] or not isinstance(case['task'], str) or not case['task']):
                    return None
                keys.append((case['scene'], case['seed'], case['task']))
            return len(keys) * len(contracts) if len(set(keys)) == len(keys) else None
        except (IndexError, KeyError, TypeError, ValueError, OSError):
            return None
    groups = []
    for flag in ('--seeds', '--tasks'):
        if argv.count(flag) != 1:
            return None
        group = []
        for item in argv[argv.index(flag) + 1:]:
            if item.startswith('--'):
                break
            group.append(item)
        if not group or len(group) != len(set(group)):
            return None
        groups.append(group)
    try:
        seeds = [int(s) for s in groups[0]]
    except ValueError:
        return None
    if len(set(seeds)) != len(seeds):
        return None
    return len(seeds) * len(groups[1])


def read(path):
    if not path.exists():
        return None
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError('report too large')
    content = path.read_bytes()
    READ_DIGESTS[str(path)] = hashlib.sha256(content).hexdigest()
    return json.loads(content)


def identity(path):
    return {'path': str(path.relative_to(ROOT)), 'sha256': READ_DIGESTS.get(str(path)) or hashlib.sha256(path.read_bytes()).hexdigest()}


def campaign_closeout(progress):
    """Keep the user's pause separate from completed-job or capability status."""
    path = ROOT / 'config/gpu-execution-control.json'
    control = read(path)
    public_control = None
    if control:
        public_control = {'status': control.get('status'),
                          'gpu_execution_allowed': control.get('gpu_execution_allowed'),
                          'requested_at_unix': control.get('requested_at_unix'),
                          'source': identity(path)}
    descriptor = progress.get('pause_closeout')
    if not descriptor:
        return public_control, None
    path = ROOT / 'reports/exports/fastwam-pause-closeout-20261013.json'
    try:
        report = read(path)
        if (descriptor.get('path') != str(path.relative_to(ROOT)) or not report
                or identity(path)['sha256'] != descriptor.get('sha256')
                or report.get('schema') != 'microduck_fastwam_pause_closeout_v1'
                or report.get('status') != 'complete'
                or report.get('formal_acceptance') is not False
                or report.get('model_promoted') is not False):
            raise ValueError('invalid closeout evidence')
        return public_control, dict(report, state='complete', source=identity(path))
    except (TypeError, ValueError, KeyError):
        return public_control, {'state': 'invalid'}


def split_retest_result(folder, plan, ready):
    """Publish a completed, audited cohort separately from the single-host attempt."""
    base = folder / 'split-completed'
    try:
        report = read(base/'report.json')
        analysis = read(base/'analysis.json')
        audit = read(base/'execution-clock-audit.json')
        if not report or not analysis or not audit:
            return None
        if (report.get('status') != 'complete' or report.get('stage') != 'complete'
                or report.get('deployment') != 'local_render_remote_model'
                or report.get('checkpoint_sha256') != plan['checkpoint_sha256']
                or report.get('protocol_sha256') != plan['protocol_sha256']
                or report.get('model_service') != ready
                or report.get('training_allowed') is not False
                or report.get('formal_acceptance') is not False
                or report.get('clock_expiry_verified') is not True
                or report.get('mode') != 'realtime' or report.get('scene') != 'room_clear'
                or report.get('inference_steps') != 20
                or any(report.get(k) is not False for k in
                       ('cache_prompt', 'compiled', 'action_context_cache', 'action_cuda_graph'))
                or report['runtime_provenance']['max_age_ticks'] != 50
                or report['runtime_provenance']['max_wall_age_s'] != 1.0
                or analysis.get('status') != 'complete'
                or analysis.get('formal_acceptance') is not False
                or analysis.get('report_sha256') != identity(base/'report.json')['sha256']
                or analysis.get('checkpoint_sha256') != report['checkpoint_sha256']
                or analysis.get('deployment') != report['deployment']
                or audit.get('status') != 'complete'
                or audit.get('clock_expiry_verified') is not True
                or audit.get('capability_acceptance') is not False
                or audit.get('assessor_sha256') != '1c58c5974e4b3d6df8589c2f339af2b4d2f174ca20b7fc75c1b9ef24139ed378'
                or report['execution_clock_audit']['sha256'] != identity(base/'execution-clock-audit.json')['sha256']
                or audit['raw_report_sha256'] != identity(base/'evaluation-report.json')['sha256']):
            return None
        expected = {(s, t) for s in range(2000, 2020)
                    for t in ('find_red_ball', 'find_blue_ball', 'find_yellow_ball')}
        episodes = report['episodes']
        mapped = {(e['seed'], e['task']): e for e in episodes}
        rows = {(e['seed'], e['task']): e for e in analysis['rows']}
        audits = {(e['seed'], e['task']): e for e in audit['rows']}
        if (len(episodes) != len(analysis['rows']) or len(episodes) != len(audit['rows'])
                or len(episodes) != 60 or set(mapped) != expected
                or set(rows) != expected or set(audits) != expected
                or report.get('completed_episodes') != 60 or analysis.get('completed') != 60
                or analysis.get('planned') != 60 or audit.get('episodes') != 60):
            return None
        for key, episode in mapped.items():
            row, clock = rows[key], audits[key]
            if (episode['evaluator']['terminated'] is not True
                    or episode['inference_worker_stopped'] is not True
                    or row['task_success'] != episode['task_success']
                    or row['strict_success'] != episode['strict_model_success']
                    or row['realtime_success'] != episode['realtime_success']
                    or row['fallback_ticks'] != episode['fallback_control_ticks']
                    or clock['verified_ticks'] != row['ticks']
                    or clock['model_ticks'] != row['model_ticks']
                    or clock['fallback_ticks'] != row['fallback_ticks']
                    or clock['wall_clock_expiry_verified'] is not True
                    or row['clock_audit'] != {k:v for k,v in clock.items() if k not in ('seed','task')}):
                return None
        tasks = sum(e['task_success'] is True for e in episodes)
        strict = sum(e['strict_model_success'] is True for e in episodes)
        realtime = sum(e['realtime_success'] is True for e in episodes)
        ticks = sum(e['ticks'] for e in rows.values())
        model_ticks = sum(e['model_ticks'] for e in rows.values())
        fallback = sum(e['fallback_ticks'] for e in rows.values())
        if (analysis.get('task_successes') != tasks or report.get('task_successes') != tasks
                or analysis.get('strict_successes') != strict or report.get('strict_successes') != strict
                or analysis.get('realtime_successes') != realtime or report.get('realtime_successes') != realtime
                or analysis.get('control_ticks') != ticks or audit.get('verified_control_ticks') != ticks
                or analysis.get('model_ticks') != model_ticks or analysis.get('fallback_ticks') != fallback
                or ticks != model_ticks + fallback):
            return None
        return dict(status='complete', total=60, task_successes=tasks, strict_successes=strict,
                    realtime_successes=realtime, model_ticks=model_ticks, fallback_ticks=fallback,
                    control_ticks=ticks, formal_acceptance=False, deployment=report['deployment'],
                    source=identity(base/'report.json'), analysis_source=identity(base/'analysis.json'),
                    clock_audit_source=identity(base/'execution-clock-audit.json'))
    except (OSError, ValueError, KeyError, TypeError):
        return None


def split_renderer_diagnostic(containers, folder, plan):
    """Identify the bounded inference service; it never counts as model training."""
    version = 'v3' if (folder/'split-service-v3-launch.json').is_file() else 'v2'
    path = folder/f'split-service-{version}-launch.json'
    launch = read(path)
    if not launch:
        return None
    digest = {'v2': '65f2f0278ddac282c6bacc3b38bd45891bca1432712becf84ba4ab576df9b681',
              'v3': 'fa532635f6523a901db74c9d625fad8d230755e90dfa4806535ee92d884fb2d7'}[version]
    try:
        if (launch.get('server_sha256') != digest
                or identity(folder/f'split-model-service-{version}.py')['sha256'] != digest
                or launch.get('training_allowed') is not False
                or launch.get('capability_acceptance') is not False
                or launch.get('protocol') != 'split-renderer-runtime-diagnostic'):
            return None
        matched = [c for c in containers if c['Id'] == launch.get('container_id')
                   and c['Name'].lstrip('/') == f'microduck-fastwam-ttl1000-split-service-{version}']
        if len(matched) != 1:
            return None
        container = matched[0]
        image = next(x for x in plan['argv'] if x.startswith('sha256:'))
        if (container['Image'] != image or container['Config']['Cmd'] !=
                ['python', '-u', str(folder/f'split-model-service-{version}.py')]):
            return None
        state = container['State']
        live = bool(state['Running'] and state['Pid'] and Path('/proc', str(state['Pid'])).exists())
        result = dict(waiting=False, live=live, container_id=container['Id'],
                    deployment='local_render_remote_model', status='running' if live else 'ended',
                    capability_acceptance=False, source=identity(path))
        progress = read(folder/'split-cohort-progress.json')
        ready = read(folder/f'split-service-{version}-ready.json')
        if ready:
            completed_result = split_retest_result(folder, plan, ready)
            if completed_result is not None:
                result['evaluation_result'] = completed_result
        if progress and ready:
            cases = progress.get('cases', [])
            allowed = {(s, t) for s in range(2000, 2020)
                       for t in ('find_red_ball', 'find_blue_ball', 'find_yellow_ball')}
            observed = {(e['seed'], e['task']) for e in cases}
            if (progress.get('schema') == 'microduck_split_retest_progress_v1'
                    and progress.get('service_id') == ready.get('service_id')
                    and progress.get('server_sha256') == ready.get('server_sha256') == digest
                    and progress.get('checkpoint_sha256') == plan['checkpoint_sha256']
                    and progress.get('protocol_sha256') == plan['protocol_sha256']
                    and progress.get('training_allowed') is False
                    and progress.get('formal_acceptance') is False
                    and type(progress.get('observed_unix')) in (int, float)
                    and math.isfinite(progress['observed_unix'])
                    and 0 < progress['observed_unix'] <= time.time() + 5
                    and progress.get('status') in ('running', 'complete', 'failed')
                    and type(progress.get('completed')) is int
                    and progress['completed'] == len(cases) == len(observed)
                    and observed <= allowed and progress.get('total') == 60):
                result['evaluation_progress'] = dict(completed=len(cases), total=60,
                    observed_unix=progress['observed_unix'], status=progress['status'],
                    source=identity(folder/'split-cohort-progress.json'))
        return result
    except (OSError, ValueError, KeyError, TypeError, StopIteration):
        return None


def retest_diagnostic(containers, folder, plan):
    """A bounded diagnostic is distinct from both training and scored cases."""
    split = split_renderer_diagnostic(containers, folder, plan)
    if split is not None:
        return split
    queue = read(folder/'diagnostic-queue.json')
    if not queue:
        return None
    script_sha = '615ddb7485866d7dc1fbbc2c36825dcefe58f69dde450a6560a664d11410d338'
    try:
        if (queue['script_sha256'] != script_sha
                or identity(folder/'wait-for-diagnostic.py')['sha256'] != script_sha
                or queue['training_allowed'] is not False
                or queue['capability_acceptance'] is not False
                or queue['plan_sha256'] != identity(folder/'plan.json')['sha256']):
            return None
        try:
            fields = Path('/proc', str(queue['pid']), 'stat').read_text().rsplit(')', 1)[1].split()
            worker_live = fields[19] == queue['process_start_ticks'] and fields[0] not in ('Z', 'X')
        except OSError:
            worker_live = False
        launch = read(folder/'model-diagnostic-launch.json')
        result = dict(waiting=worker_live and queue['status'] == 'waiting_for_gpu', live=False,
                      status=queue['status'], source=identity(folder/'diagnostic-queue.json'))
        if not launch:
            return result
        matched = [c for c in containers if c['Id'] == launch.get('container_id')
                   and c['Name'].lstrip('/') == 'microduck-fastwam-ttl1000-model-diagnostic-v1']
        if len(matched) != 1:
            return None
        container = matched[0]
        image = next(x for x in plan['argv'] if x.startswith('sha256:'))
        if (container['Image'] != image
                or container['Config']['Cmd'] != ['python', '-X', 'faulthandler', '-u', str(folder/'diagnose-model.py')]
                or launch['diagnostic_sha256'] != '65bd9183f8623d6787ea1871b50b3e87d9d216dd83d06faf647ce58abdc44fe8'
                or identity(folder/'diagnose-model.py')['sha256'] != launch['diagnostic_sha256']):
            return None
        state = container['State']
        result.update(container_id=container['Id'],
                      live=bool(state['Running'] and state['Pid'] and Path('/proc', str(state['Pid'])).exists()),
                      status='running' if state['Running'] else 'complete' if state['ExitCode'] == 0 else 'failed')
        return result
    except (OSError, ValueError, KeyError, TypeError, StopIteration):
        return None


def authorized_retest(containers):
    """Display the one user-authorized evaluation without lifting training pause."""
    folder = ROOT / 'config/plain-ttl1000-retest-v1'
    try:
        authorization = read(folder/'authorization.json')
        if authorization is None:
            return None
        if (identity(folder/'authorization.json')['sha256'] !=
                '63a30ceec4672ce0f2b2079e700587306188dc5a136da68f9dbf4e92c1431ff2'
                or authorization.get('training_allowed') is not False
                or authorization.get('evaluation_authorized') is not True):
            return None
        plan, launch = read(folder/'plan.json'), read(folder/'launch.json')
        if (not plan or not launch or identity(folder/'plan.json')['sha256'] != authorization['plan_sha256']
                or launch['plan_sha256'] != authorization['plan_sha256']):
            return None
        matched = [c for c in containers if c['Id'] == launch.get('container_id')
                   and c['Name'].lstrip('/') == plan['name']]
        if len(matched) != 1:
            return None
        container = matched[0]
        argv = plan['argv']
        image = next(x for x in argv if x.startswith('sha256:'))
        if container['Image'] != image or container['Config']['Cmd'] != argv[argv.index(image)+1:]:
            return None
        report_path = ROOT/'reports/plain-ttl1000-realtime-v1/report.json'
        report = read(report_path)
        if (not report or report.get('checkpoint_sha256') != plan['checkpoint_sha256']
                or report.get('runtime_provenance', {}).get('max_age_ticks') != 50
                or report.get('runtime_provenance', {}).get('max_wall_age_s') != 1.0):
            return None
        completed = len(report.get('episodes', []))
        if completed > plan['episodes'] or evaluation_case_count(containers, authorization['evaluation']) != 60:
            return None
        state = container['State']
        live = bool(state['Running'] and state['Pid'] and Path('/proc', str(state['Pid'])).exists())
        return dict(id=authorization['evaluation'], name=plan['name'], container_id=container['Id'],
                    live=live, completed=completed, total=60, stage=report.get('stage'),
                    status=report.get('status'), ttl_ms=1000, training_allowed=False,
                    report_source=identity(report_path), authorization_source=identity(folder/'authorization.json'),
                    diagnostic=retest_diagnostic(containers, folder, plan),
                    formal_acceptance=False)
    except (OSError, ValueError, KeyError, TypeError, StopIteration):
        return None


def current_queue(progress):
    name = progress.get('active_queue', 'origin-sampling-training-queue-v1.json')
    # Campaign writers also record the full H200 report path. Accept only
    # direct children of this campaign's reports directory, never arbitrary
    # absolute paths or traversal. Keep the returned record format unchanged.
    if isinstance(name, str) and Path(name).is_absolute():
        path = Path(name)
        if path.parent != ROOT / 'reports' or path.resolve().parent != (ROOT / 'reports').resolve():
            raise ValueError('invalid campaign queue filename')
        name = path.name
    if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_.-]+-queue-v[0-9]+\.json', name):
        raise ValueError('invalid campaign queue filename')
    path = ROOT / 'reports' / name
    if path.resolve().parent != (ROOT / 'reports').resolve():
        raise ValueError('invalid campaign queue filename')
    result = read(path) or {}
    declaration = progress.get('partial_rms_realtime_comparison', {})
    if declaration.get('queue') == 'reports/' + name:
        launch = declaration.get('launch', {})
        completed = result.get('completed', [])
        total = declaration.get('planned_jobs')
        if (result.get('plan_sha256') != declaration.get('plan_sha256')
                or result.get('pid') != launch.get('pid')
                or type(total) is not int or not 0 <= len(completed) <= total
                or len({r['name'] for r in completed}) != len(completed)):
            raise ValueError('current evaluation queue differs from its declaration')
        observed = time.time()
        try:
            fields = Path('/proc', str(launch['pid']), 'stat').read_text().rsplit(')', 1)[1].split()
            process_state = ('live' if fields[19] == str(launch['process_start_ticks'])
                             and fields[0] not in ('Z', 'X') else 'identity_ended')
        except FileNotFoundError:
            process_state = 'missing'
        except OSError:
            process_state = 'unknown'
        result = {**result, 'completed_jobs': len(completed), 'total_jobs': total,
                  'process_state': process_state, 'observed_unix': observed}
    return result


def history_receipt(report):
    """Read a bounded checkpoint receipt; never load model weights in the UI."""
    try:
        checkpoint = Path(report['checkpoint']).resolve()
        parts = checkpoint.relative_to(ROOT.resolve()).parts
    except (KeyError, ValueError, TypeError):
        return None, None
    if (len(parts) != 5 or parts[0] != 'runs' or parts[2:4] != ('checkpoints', 'weights')
            or not re.fullmatch(r'step_[0-9]{6}\.pt', parts[-1])):
        return None, None
    receipt = checkpoint.parent.parent / 'state' / checkpoint.stem / 'navigation_checkpoint.json'
    try:
        receipt.resolve().relative_to(ROOT.resolve())
    except ValueError:
        return None, None
    data = read(receipt)
    return (data, READ_DIGESTS[str(receipt)]) if data is not None else (None, None)


def valid_history_contract(history):
    if not isinstance(history, dict):
        return False
    offsets = history.get('offsets_ticks')
    if (not isinstance(offsets, list) or len(offsets) not in (4, 9)
            or any(type(t) is not int or t < 0 or t % 5 for t in offsets)
            or sorted(set(offsets), reverse=True) != offsets or offsets[-1] != 0):
        return False
    age = history.get('max_sample_age_ticks')
    return type(age) is int and 0 <= age <= 5 and history == {
        'schema': 'microduck_causal_rgb_grid_v1', 'offsets_ticks': offsets,
        'camera_size': 256, 'output_size': 512 if len(offsets) == 4 else 768,
        'max_sample_age_ticks': age, 'layout': 'row_major_oldest_to_current',
        'startup_padding': 'repeat_episode_first_frame', 'selector': 'fixed_timestamps_only', 'control_hz': 50}


def validated_score(report, protocol, *, receipt=None, receipt_sha256=None, execution_mode='offline'):
    """Only count a complete, sensor-only, unmodified offline development cohort."""
    if execution_mode not in ('offline', 'realtime'):
        raise ValueError('unknown evaluation mode')
    if not protocol or report.get('status') != 'complete' or report.get('mode') != execution_mode:
        return None
    history = report.get('input_history')
    if history is not None:
        if (not valid_history_contract(history) or not isinstance(receipt, dict)
                or not isinstance(receipt_sha256, str) or not re.fullmatch('[0-9a-f]{64}', receipt_sha256)
                or report.get('checkpoint_receipt_sha256') != receipt_sha256
                or receipt.get('input_history') != history
                or receipt.get('weights_sha256') != report.get('checkpoint_sha256')
                or receipt.get('stats_sha256') != report.get('stats_sha256')
                or type(receipt.get('step')) is not int or receipt['step'] < 1):
            return None
    rows = report.get('episodes', [])
    expected = {(s, t) for s in protocol['development_seeds'] for t in protocol['tasks']}
    if (len(rows) != len(expected) or report.get('completed_episodes') != len(expected)
            or {(r.get('seed'), r.get('task')) for r in rows} != expected
            or report.get('scene') != protocol['scene']
            or report.get('input_intervention') not in (None, 'none', {'mask_rgb': False, 'instruction_color': None})
            or report.get('action_intervention') not in (None, 'none')):
        return None
    colors = {}
    for row in rows:
        policy, truth = row.get('policy', {}), row.get('evaluator', {})
        if (policy.get('uses_privileged_state') is not False
                or policy.get('input') != ('head_rgb_history_grid+instruction+proprio48' if history is not None else 'head_rgb+instruction+proprio48')
                or policy.get('history_contract') != history
                or policy.get('artifact_sha256') != report.get('checkpoint_sha256')
                or truth.get('evaluator_source_sha256') != protocol['evaluator_sha256']
                or truth.get('terminated') is not True or row.get('injected_delay_ms') != 0
                or type(truth.get('strict_success')) is not bool
                or type(row.get('strict_model_success')) is not bool
                or row['strict_model_success'] != bool(truth.get('strict_success')
                     and not row.get('fallback_control_ticks') and not row.get('cancelled'))):
            return None
        c = colors.setdefault(row['task'], {'passed': 0, 'total': 0})
        c['total'] += 1
        c['passed'] += int(row['strict_model_success'])
    passed = sum(c['passed'] for c in colors.values())
    if report.get('strict_successes') != passed:
        return None
    return {'passed': passed, 'total': len(rows), 'layouts': len(protocol['development_seeds']), 'colors': colors}


def validated_realtime_score(report, protocol, *, receipt=None, receipt_sha256=None):
    score = validated_score(report, protocol, execution_mode='realtime',
                            receipt=receipt, receipt_sha256=receipt_sha256)
    if score is None:
        return None
    outcomes, factors, latencies = [], [], []
    fallback_episodes = rejected = 0
    for row in report['episodes']:
        values = [row.get(k) for k in ('wall_s', 'sim_s', 'real_time_factor', 'maximum_clock_lag_s')]
        if any(type(v) not in (float, int) or not math.isfinite(v) for v in values):
            return None
        wall, sim, factor, lag = values
        if (row.get('mode') != 'realtime' or wall <= 0 or sim <= 0 or lag < 0
                or not math.isclose(factor, sim/wall, rel_tol=1e-8)
                or row.get('timing_gate') != {'min_real_time_factor': .95, 'max_clock_lag_s': .1}):
            return None
        qualified = factor >= .95 and lag <= .1
        success = bool(row['strict_model_success'] and qualified)
        if row.get('realtime_timing_qualified') is not qualified or row.get('realtime_success') is not success:
            return None
        for key in ('fallback_control_ticks', 'rejected_plans'):
            if type(row.get(key)) is not int or row[key] < 0:
                return None
        latency = row.get('model_latency_p95_ms')
        if latency is not None:
            if type(latency) not in (float, int) or not math.isfinite(latency) or latency < 0:
                return None
            latencies.append(latency)
        factors.append(factor)
        fallback_episodes += int(row['fallback_control_ticks'] > 0)
        rejected += row['rejected_plans']
        outcomes.append({'seed': row['seed'], 'task': row['task'], 'success': success, 'timing_qualified': qualified})
    passed = sum(r['success'] for r in outcomes)
    if report.get('realtime_successes') != passed:
        return None
    return {'passed': passed, 'total': score['total'], 'timing_qualified': sum(r['timing_qualified'] for r in outcomes),
            'fallback_episodes': fallback_episodes, 'rejected_plans': rejected,
            'minimum_real_time_factor': min(factors),
            'episode_p95_range_ms': [min(latencies), max(latencies)] if latencies else None,
            'outcomes': outcomes}


def expanded_realtime_comparison():
    """Keep newly held-out development layouts separate from repeated layouts."""
    path = ROOT / 'reports/exports/expanded-clear-development-assessment-v1.json'
    try:
        assessment = read(path)
        if assessment is None:
            return None
        if identity(path)['sha256'] != 'b32fc7ad46e79b28f987d5ebdec82fe4d758af4a18303682959d00b28b4157d7':
            raise ValueError('expanded assessment changed')
        protocol_path = ROOT / 'config/expanded-clear-development-protocol-v1.json'
        protocol = read(protocol_path)
        manifest_path = ROOT / 'reports/exports/expanded-clear-development-evidence-v1.manifest.json'
        manifest = read(manifest_path)
        if (protocol is None or manifest is None
                or identity(protocol_path)['sha256'] != '9c6bc89a15f41169cee85be73d225f738027b7ec9b13443171074af72b37f2f6'
                or identity(manifest_path)['sha256'] != assessment['manifest_sha256']
                or assessment['status'] != 'complete' or not assessment['initial_inputs_equal']
                or assessment['distinct_development_layouts'] != 20 or assessment['cases_per_variant'] != 60
                or protocol['development_seeds'] != list(range(2000, 2020))):
            raise ValueError('expanded cohort or provenance differs')
        variants = {}
        for variant in ('plain', 'optimized'):
            name = f'reports/expanded-clear-{variant}-realtime-v1/report.json'
            report_path = ROOT / name
            report = read(report_path)
            if (report is None or identity(report_path)['sha256'] != manifest['files'][name]['sha256']
                    or report.get('clock_expiry_verified') is not True
                    or report.get('checkpoint_sha256') != assessment['checkpoint_sha256']):
                raise ValueError('expanded trial missing or changed')
            receipt, digest = history_receipt(report)
            score = validated_realtime_score(report, protocol, receipt=receipt, receipt_sha256=digest)
            audited = assessment['raw_evidence_audits'][variant]
            if (score is None or score['total'] != 60 or score['passed'] != assessment['successes'][variant]
                    or audited['execution']['episodes'] != 60
                    or audited['execution']['verified_ticks'] != sum(e['evaluator']['tick'] for e in report['episodes'])):
                raise ValueError('expanded score or execution audit differs')
            variants[variant] = {
                'passed': score['passed'], 'total': 60, 'fallback_episodes': score['fallback_episodes'],
                'steady_inference_p95_ms': audited['pooled_steady_inference_p95_ms'],
                'source': identity(report_path)}
        return {'state': 'complete', 'variants': variants, 'distinct_layouts': 20, 'cases_per_variant': 60,
                'layout_results': assessment['layout_results'], 'checkpoint_sha256': assessment['checkpoint_sha256'],
                'paired_gains': assessment['paired_gains'], 'paired_regressions': assessment['paired_regressions'],
                'source': identity(path), 'protocol_source': identity(protocol_path),
                'formal_acceptance': False, 'obstacle_acceptance': False, 'hardware_validated': False}
    except (KeyError, TypeError, ValueError):
        return {'state': 'invalid', 'variants': None, 'source': identity(path), 'formal_acceptance': False}


def repeated_realtime_comparison(protocol):
    """Expose only the immutable six-trial audit bound to all actual reports."""
    path = ROOT / 'reports/exports/clock-v82-repeat-assessment-v1.json'
    assessment = read(path)
    if assessment is None:
        return None
    try:
        if identity(path)['sha256'] != 'efef66e32b6674a0021f4994f12392bef9ebefbf327f3747ec977c7ab6ac5292':
            raise ValueError('repeated assessment changed')
        if (assessment['status'] != 'complete' or not assessment['all_repeats_reported']
                or assessment['independent_development_layouts'] != len(protocol['development_seeds'])
                or len(protocol['development_seeds']) != 7 or len(protocol['tasks']) != 3):
            raise ValueError('repeated cohort differs')
        repeats, totals = [], {'plain': 0, 'optimized': 0}
        for repeat in range(3):
            variants = {}
            for variant in ('plain', 'optimized'):
                key = f'r{repeat}-{variant}'
                report_path = ROOT/f'reports/clock-v82-{key}-realtime-v1/report.json'
                report = read(report_path)
                if (report is None or identity(report_path)['sha256'] != assessment['report_sha256'][key]
                        or report.get('clock_expiry_verified') is not True):
                    raise ValueError('repeated trial is missing or changed')
                receipt, digest = history_receipt(report)
                score = validated_realtime_score(report, protocol, receipt=receipt, receipt_sha256=digest)
                if score is None or score['total'] != 21:
                    raise ValueError('repeated trial score is invalid')
                audited = assessment['raw_evidence_audits'][key]
                if (audited['execution']['episodes'] != 21 or
                        audited['execution']['verified_ticks'] != sum(e['evaluator']['tick'] for e in report['episodes'])):
                    raise ValueError('repeated execution evidence is incomplete')
                variants[variant] = {'passed': score['passed'], 'total': 21,
                    'fallback_episodes': score['fallback_episodes'],
                    'steady_inference_p95_ms': audited['pooled_steady_inference_p95_ms'],
                    'source': identity(report_path)}
                totals[variant] += score['passed']
            expected = assessment['per_repeat_successes'][repeat]
            if variants['plain']['passed'] != expected['control'] or variants['optimized']['passed'] != expected['candidate']:
                raise ValueError('repeated paired score differs')
            repeats.append({'repeat': repeat, 'variants': variants})
        if totals != assessment['realtime_successes_all_repeats']:
            raise ValueError('repeated total differs')
        return {'state': 'complete', 'passed': totals, 'total_per_variant': 63, 'distinct_layouts': 7,
                'repeats': repeats, 'checkpoint_sha256': assessment['checkpoint_sha256'],
                'source': identity(path), 'formal_acceptance': False, 'hardware_validated': False}
    except (KeyError, TypeError, ValueError):
        return {'state': 'invalid', 'passed': None, 'source': identity(path), 'formal_acceptance': False}


REALTIME_TRIALS = {
    'retained': ('retained-sameweight', '历史：单帧保留模型', None),
    'low_lr_history': ('repeat16-low-lr-sameweight', '历史画面模型 · 基础加速对照',
                       'a802ed761ae53a333ab32a9b86275c55be83c8b6d36f16ea0ab375d77a346f03'),
    'low_lr_promptcache': ('repeat16-low-lr-promptcache', '最新：历史画面模型 · 增加语言缓存',
                          '1911b6602e682f13296121a5888a0ffaeffba3523d682371a238da68b8ff337a'),
}


def trial_flags(trial, variant):
    enabled = trial == 'low_lr_promptcache' or variant == 'optimized'
    return {'cache_prompt': trial == 'low_lr_promptcache' and variant == 'optimized',
            'action_context_cache': enabled, 'action_cuda_graph': enabled}


def trial_report_prefix(trial, variant):
    if trial == 'low_lr_promptcache':
        return 'repeat16-low-lr-sameweight-optimized' if variant == 'plain' else 'repeat16-low-lr-promptcache'
    return REALTIME_TRIALS[trial][0] + '-' + variant


def history_pair_assessment(protocol, declared, reports, variants, trial='low_lr_history'):
    """Bind the independent full-cohort audit to these exact displayed reports.

    The H200 host collector has neither NumPy nor the model environment. The
    immutable assessor runs separately in the analysis environment and audits
    every saved control tick. Missing/stale assessment means no paired claim.
    """
    prefix = REALTIME_TRIALS[trial][0]
    expected_assessor = {
        'low_lr_history': 'cfe9498f6a076f64414e2abf3fbda18b4c40e466f7088d7a3609363d3dc2064f',
        'low_lr_promptcache': '4d98e4d60bb53e6722b0053a04e7bf716118ccd1b639e23e0284c517fcf275b6',
    }[trial]
    path = ROOT/f'reports/exports/{prefix}-realtime-assessment-v1.json'
    result = read(path)
    if result is None:
        return None, None
    if (result.get('status') != 'complete'
            or result.get('script_sha256') != expected_assessor
            or result.get('checkpoint_sha256') != declared['checkpoint_sha256']
            or result.get('protocol_sha256') != REALTIME_TRIALS[trial][2]
            or result.get('evaluation_protocol_sha256') != declared['evaluation_protocol_sha256']
            or result.get('episodes_per_variant') != len(protocol['development_seeds'])*len(protocol['tasks'])):
        raise ValueError('independent assessment identity differs')
    rows = {}
    for name, variant in (('control', 'plain'), ('candidate', 'optimized')):
        record = reports[variant]
        if (result.get('report_sha256', {}).get(name) != variants[variant]['source']['sha256']
                or result.get('runtime_provenance') != record.get('runtime_provenance')
                or result.get('realtime_successes', {}).get(name) != variants[variant]['score']['passed']
                or result.get('optimization_flags', {}).get(name) != declared['variants'][name]):
            raise ValueError('independent assessment reports differ')
        audit = result.get('execution_audit', {}).get(name, {})
        if (audit.get('episodes') != len(record['episodes']) or audit.get('verified_ticks') !=
                sum(e['evaluator']['tick'] for e in record['episodes'])):
            raise ValueError('independent execution audit is incomplete')
        rows[name] = {(e['task'], e['seed']): e['success'] for e in variants[variant]['score']['outcomes']}
    improvements = sum(rows['candidate'][k] and not v for k, v in rows['control'].items())
    regressions = sum(v and not rows['candidate'][k] for k, v in rows['control'].items())
    if result.get('paired_improvements') != improvements or result.get('paired_regressions') != regressions:
        raise ValueError('independent paired counts differ')
    return {'improvements': improvements, 'regressions': regressions}, identity(path)


def realtime_comparison(protocol, trial='retained'):
    """Bounded read-only view of the declared same-weight development trial."""
    prefix, label, digest = REALTIME_TRIALS[trial]
    suffix = 'protocol-v1' if trial == 'low_lr_promptcache' else 'realtime-protocol-v1'
    declared_path = ROOT/f'reports/exports/{prefix}-{suffix}.json'
    declared = read(declared_path)
    if not declared:
        return None
    if digest is not None and identity(declared_path)['sha256'] != digest:
        raise ValueError('declared realtime trial changed')
    checkpoint = declared['checkpoint_sha256']
    history = declared.get('input_history')
    if history is not None:
        protocol_path = ROOT/'config/clear-room-evaluation-protocol-v1.json'
        if (not valid_history_contract(history) or not protocol_path.is_file()
                or identity(protocol_path)['sha256'] != declared['evaluation_protocol_sha256']):
            raise ValueError('history trial evaluation protocol changed')
    variants, reports, benchmarks = {}, {}, {}
    for variant in ('plain', 'optimized'):
        flags = trial_flags(trial, variant)
        report_prefix = trial_report_prefix(trial, variant)
        path = ROOT/f'reports/{report_prefix}-realtime-v1/report.json'
        report = read(path)
        row = {'state': 'pending', 'completed': None, 'total': len(protocol['development_seeds'])*len(protocol['tasks']), 'score': None}
        if report:
            row.update(state=report.get('status', 'unknown'), completed=report.get('completed_episodes'), source=identity(path))
            matching = (report.get('checkpoint_sha256') == checkpoint and report.get('mode') == 'realtime'
                and all(report.get(k) is v for k, v in flags.items()) and report.get('compiled') is False
                and report.get('backend') == 'native'
                and report.get('inference_steps') == 20 and report.get('input_history') == history)
            if history is not None:
                matching = matching and all(report.get(k) == declared.get(k) for k in (
                    'checkpoint_receipt_sha256', 'stats_sha256'))
            if not matching:
                row['state'] = 'invalid'
            elif report.get('status') == 'complete':
                receipt, receipt_digest = history_receipt(report) if history is not None else (None, None)
                row['score'] = validated_realtime_score(report, protocol, receipt=receipt, receipt_sha256=receipt_digest)
                if row['score'] is None:
                    row['state'] = 'invalid'
            reports[variant] = report
        variants[variant] = row
        alignment_path = ROOT/f'reports/{report_prefix}-align-v1/report.json'
        alignment = read(alignment_path)
        if (alignment and alignment.get('status') == 'complete' and alignment.get('checkpoint_sha256') == checkpoint
                and alignment.get('input_history') == history
                and (history is None or alignment.get('stats_sha256') == declared.get('stats_sha256'))):
            benchmarks[variant] = (alignment, identity(alignment_path))
    benchmark = None
    if len(benchmarks) == 2:
        plain, optimized = (benchmarks[n][0] for n in ('plain', 'optimized'))
        match = all(plain.get(k) == optimized.get(k) for k in (
            'checkpoint_sha256', 'stats_sha256', 'input_sha256', 'seed', 'inference_steps', 'dtype', 'frame_indices', 'input_history'))
        match = (match and all(isinstance(plain.get(k), str) and re.fullmatch('[0-9a-f]{64}', plain[k])
                               for k in ('stats_sha256', 'input_sha256'))
                 and type(plain.get('seed')) is int and bool(plain.get('frame_indices')))
        flags_match = all(all(record.get('prompt_cache' if k == 'cache_prompt' else k) is v
                              for k, v in trial_flags(trial, variant).items()) and record.get('backend') == 'native'
                          and record.get('inference_steps') == 20 and record.get('dtype') == 'bfloat16'
                          for record, variant in ((plain, 'plain'), (optimized, 'optimized')))
        if match and flags_match and optimized.get('all_tensors_within_diagnostic_tolerance') is True:
            values = {}
            for name, (record, origin) in benchmarks.items():
                samples = record.get('latency_diagnostic', {}).get('milliseconds', [])
                if not samples or any(type(x) not in (float, int) or not math.isfinite(x) or x < 0 for x in samples):
                    break
                samples = sorted(samples); position = (len(samples)-1)*.95
                lower = int(position); upper = min(lower+1, len(samples)-1)
                value = samples[lower] + (samples[upper]-samples[lower])*(position-lower)
                values[name] = {'calls': len(samples), 'p95_ms': value, 'source': origin}
            if len(values) == 2:
                benchmark = values
    complete = all(v['score'] is not None for v in variants.values())
    paired = None
    comparison_error = None
    assessment_source = None
    if complete and history is not None:
        try:
            paired, assessment_source = history_pair_assessment(protocol, declared, reports, variants, trial)
        except (ValueError, KeyError, TypeError, OSError) as error:
            comparison_error = type(error).__name__
    elif complete:
        try:
            # Reuse the pinned Python3.8-compatible full-cohort assessor.
            # Training snapshots remain immutable. UI metrics alone do not
            # establish identical worlds, expiry or GPU runtime.
            import importlib.util
            scripts = ROOT/'reports/exports/dashboard-assessors-py38-v1'
            expected = {'assess_fastwam_navigation_development.py': 'e63f7dc744bf8c8c4a52e52c78efe531c92733bbe3afb976c069e03b745cd311'}
            expected['assess_fastwam_navigation_realtime.py'] = '0a158fd04f5ae7ceaafaf76144c6e7a11c2200335f3bfa756a06be3a7211de3a'
            for name, digest in expected.items():
                if hashlib.sha256((scripts/name).read_bytes()).hexdigest() != digest:
                    raise ValueError('frozen assessor source changed')
            previous_path = sys.path[:]
            previous_bytecode = sys.dont_write_bytecode
            try:
                sys.dont_write_bytecode = True
                sys.path.insert(0, str(scripts))
                spec = importlib.util.spec_from_file_location('dashboard_realtime_assessor', scripts/'assess_fastwam_navigation_realtime.py')
                module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
                result = module.compare(reports['plain'], reports['optimized'], protocol, checkpoint_sha256=checkpoint)
            finally:
                sys.path[:] = previous_path
                sys.dont_write_bytecode = previous_bytecode
            paired = {'improvements': result['paired_improvements'], 'regressions': result['paired_regressions']}
        except (ValueError, KeyError, TypeError, OSError, ImportError, AttributeError) as error:
            comparison_error = type(error).__name__
    return {'trial_id': trial, 'label': label, 'input_history': history,
            'variant_labels': ({'plain': '上下文缓存 + CUDA Graph', 'optimized': '再增加语言缓存'}
                               if trial == 'low_lr_promptcache' else
                               {'plain': '普通推理', 'optimized': '上下文缓存 + CUDA Graph'}),
            'checkpoint_sha256': checkpoint, 'variants': variants, 'benchmark': benchmark,
            'paired': paired, 'comparison_error': comparison_error, 'complete': complete,
            'assessment_source': assessment_source,
            'protocol_source': identity(declared_path), 'formal_acceptance': False}


def pinned_json(descriptor):
    """Read a declared campaign artifact without following paths outside ROOT."""
    path = ROOT / descriptor['path']
    path.resolve().relative_to(ROOT.resolve())
    result = read(path)
    if result is None or identity(path)['sha256'] != descriptor['sha256']:
        raise ValueError('model selection evidence changed or unavailable')
    return result


def selected_model_score(entry, runs):
    """Bind a model role to its exact weight, protocol, report and assessment."""
    run = next(r for r in runs if r['id'] == entry['run_id'])
    weight = entry['checkpoint_sha256']
    if run['checkpoint_sha256'] != weight:
        raise ValueError('registered model differs from training receipt')
    report = pinned_json(entry['evaluation'])
    protocol = pinned_json(entry['protocol'])
    assessment = pinned_json(entry['assessment'])
    if report.get('checkpoint_sha256') != weight:
        raise ValueError('score belongs to another checkpoint')
    if 'cohorts' in protocol:
        cohort = next(c for c in protocol['cohorts']
                      if c['scene'] == report['scene'] and c['mode'] == 'offline')
        runtime = report['runtime_provenance']
        if (runtime['source_sha256'] != cohort['runtime_sources']
                or any(runtime[k] != cohort[k] for k in ('evaluation_script_sha256', 'model_config_sha256'))
                or any(r['evaluator']['tasks_source_sha256'] != cohort['tasks_source_sha256']
                       for r in report['episodes'])):
            raise ValueError('evaluation differs from its registered protocol')
        protocol = {**cohort, 'development_seeds': cohort['seeds']}
    receipt, digest = history_receipt(report) if report.get('input_history') is not None else (None, None)
    score = validated_score(report, protocol, receipt=receipt, receipt_sha256=digest)
    if score is None:
        raise ValueError('registered offline development score is invalid')
    if entry['assessment_kind'] == 'paired':
        arm = entry['assessment_arm']
        assessed = assessment['assessments']['offline']
        if (assessment['status'] != 'complete'
                or assessment['evidence'][arm]['reports']['offline'] != entry['evaluation']['sha256']
                or assessed['checkpoint_sha256'][arm] != weight
                or assessed['strict_successes'][arm] != score['passed']
                or assessed['episodes'] != score['total']):
            raise ValueError('paired assessment does not bind this score')
    elif entry['assessment_kind'] == 'raw_cohort':
        audit = pinned_json(entry['raw_audit'])
        if (assessment['report_sha256'] != entry['evaluation']['sha256']
                or assessment['protocol_sha256'] != entry['protocol']['sha256']
                or audit['status'] != 'complete' or audit['checkpoint_sha256'] != weight):
            raise ValueError('raw assessment does not bind this score')
        assessed = next(c for c in audit['cohorts'] if c['scene'] == report['scene'] and c['mode'] == 'offline')
        outcomes = {(r['seed'], r['task']): r['strict_model_success'] for r in report['episodes']}
        audited = {(r['seed'], r['task']): r['strict_success'] for r in assessed['rows']}
        if (outcomes != audited or len(assessed['rows']) != score['total']
                or assessed['strict_successes'] != score['passed']
                or assessment['strict_successes'] != score['passed']):
            raise ValueError('raw assessment outcomes differ')
    else:
        raise ValueError('unknown model assessment kind')
    return {'state': 'verified', 'id': Path(entry['evaluation']['path']).parent.name,
            'run_id': entry['run_id'], 'label': entry['label'], 'checkpoint_sha256': weight,
            'mode': 'offline', 'scene': report['scene'], 'scope_label': entry['scope_label'],
            'decision': entry['decision'], 'formal_acceptance': False,
            'protocol_source': entry['protocol'], 'assessment_source': entry['assessment'],
            'raw_audit_source': entry.get('raw_audit'), 'source': entry['evaluation'],
            'evaluator_sha256': protocol['evaluator_sha256'], **score}


def model_selection(runs, guarded=None):
    """Explicit retained choice; newest training run, never newest report mtime."""
    path = ROOT / 'config/dashboard-model-selection-v1.json'
    declaration = read(path)
    result = {'state': 'unavailable', 'retained': None, 'latest': None,
              'formal_acceptance': False, 'comparison_note': '模型版本登记尚未取得。'}
    if declaration is None:
        return result
    try:
        if declaration['schema'] != 'microduck_fastwam_model_selection_v1':
            raise ValueError('invalid model selection schema')
        result.update(state='verified', source=identity(path),
                      comparison_note=declaration['comparison_note'])
        result['retained'] = selected_model_score(declaration['retained'], runs)
    except (KeyError, ValueError, TypeError, StopIteration, OSError):
        result.update(state='invalid', comparison_note='保留模型的版本或评测来源需核查。')
    # Missing results on a new run must not silently resurrect an older candidate.
    # Forward/backward and checkpoint checks are operational runs, not candidate policies.
    candidates = [r for r in runs if 'preflight' not in r['id']
                  and '-resume-verify-' not in r['id']
                  and r.get('verify_resume_only') is not True
                  and r.get('stage') != 'resume_verified']
    latest = max(candidates, key=lambda r: r.get('started_unix') or 0) if candidates else None
    if latest:
        entry = next((e for e in declaration.get('candidates', [])
                      if e['run_id'] == latest['id'] and e['checkpoint_sha256'] == latest['checkpoint_sha256']), None)
        result['latest'] = {'state': 'pending', 'run_id': latest['id'], 'label': latest['id'],
                            'checkpoint_sha256': latest['checkpoint_sha256']}
        if entry:
            try:
                result['latest'] = selected_model_score(entry, runs)
            except (KeyError, ValueError, TypeError, StopIteration, OSError):
                result['latest']['state'] = 'invalid'
        if result['latest']['state'] != 'verified':
            result['comparison_note'] = '最新训练版本的评测尚未核验，不以旧候选的成绩代替；历史成绩保留在下方。'
        if (guarded and guarded.get('state') == 'verified'
                and guarded.get('run_id') == latest['id']
                and guarded.get('checkpoint_sha256') == latest['checkpoint_sha256']):
            groups = guarded['groups']
            by_scene = {g['scene']: g for g in groups}
            clear, obstacle = by_scene['room_clear'], by_scene['room_curriculum_one']
            qualified = clear['candidate_successes'] >= 19 and obstacle['candidate_successes'] > 25
            reason = ('开发门槛已达到，仍需独立布局与同权重实时验收；尚未替换保留模型。' if qualified else
                      '本轮未同时保持空房间能力并改善单障碍成绩，保留模型未替换。')
            result['latest'] = dict(state='verified', run_id=latest['id'],
                label=result['latest']['label'], checkpoint_sha256=latest['checkpoint_sha256'],
                groups=groups, mode='offline', scope_label='空房间 / 单障碍 · 离线开发',
                sources=guarded['sources'],
                decision={'status': 'awaiting_review' if qualified else 'not_adopted', 'reason': reason},
                formal_acceptance=False, model_promoted=False)
            result['comparison_note'] = '最新候选已完成 81 案例独立重放；两组同协议对照见下方。保留模型卡片展示其历史开发范围。'
    return result


def no_age_retest(containers):
    folder = ROOT / 'config/plain-no-age-retest-v1'
    status = read(folder / 'progress.json')
    launch = read(folder / 'split-service-v3-launch.json')
    if not status or not launch:
        return None
    if (status.get('schema') != 'microduck_no_age_progress_v1'
            or status.get('total') != 60 or type(status.get('completed')) is not int
            or not 0 <= status['completed'] <= 60
            or status.get('checkpoint_sha256') != '8ea7793290f0a22bff56ec4e44168232564ec5d57a9e21515a8a8c4a989114ea'
            or status.get('max_action_age_ms', 'missing') is not None
            or status.get('formal_acceptance') is not False):
        return {'state': 'invalid'}
    matched = next((c for c in containers if c['Id'] == launch.get('container_id')), None)
    value = dict(state=status['state'], completed=status['completed'], total=60,
                 live=bool(matched and matched['State']['Running']),
                 observed_unix=status['observed_unix'], source=identity(folder / 'progress.json'))
    closed = read(folder / 'verified-closeout.json')
    if closed and closed.get('container_id') == launch.get('container_id'):
        report, analysis = read(folder / 'completed-report.json'), read(folder / 'analysis.json')
        if (report and analysis and closed.get('completed') == 60
                and closed.get('no_age_execution_verified') is True
                and closed.get('report_sha256') == identity(folder / 'completed-report.json')['sha256']
                and closed.get('analysis_sha256') == identity(folder / 'analysis.json')['sha256']
                and analysis.get('report_sha256') == closed['report_sha256']
                and report.get('status') == analysis.get('status') == 'complete'
                and len(report.get('episodes', [])) == len(analysis.get('rows', [])) == 60
                and report.get('no_age_execution_verified') is True):
            value['result'] = {k: analysis.get(k) for k in ('task_successes', 'strict_successes',
                'realtime_successes', 'timing_qualified', 'model_ticks', 'fallback_ticks',
                'executed_older_than_1000ms', 'observation_age_ms')}
            value['state'] = 'verified_complete'
    return value


def candidate_postcheck_folder(run_id, folder, training, expanded=None):
    """Follow a registered evaluation continuation for the same saved weights."""
    postchecks = {
        'action-scope-corrective-pilot-v1': 'action-scope-corrective-postchecks-v2',
        'action-scope-continuation-pilot-v2': 'action-scope-continuation-postchecks-v2',
        'action-scope-epoch-pilot-v2': 'action-scope-epoch-postchecks-v3',
        'retention-action-pilot-v2': 'retention-action-postchecks-v1',
        'retention-strong-pilot-v1': 'retention-strong-postchecks-v1',
        'retained-corrective-pilot-v1': 'retained-corrective-postchecks-v1',
        'target-memory-paired-retention-pilot-v1': 'memory-candidate-postchecks-v1',
        **{name: b['postchecks'] for name, b in MEMORY_RUN_BINDINGS.items()},
    }
    steps = CONTINUATION_STEPS.get(run_id)
    if run_id == 'target-memory-paired-retention-pilot-v1':
        steps = (250, 250)
    axis = re.fullmatch(r'memory-axis-(control|weighted)-pilot-v1', run_id)
    if axis:
        postchecks[run_id] = f'memory-axis-{axis.group(1)}-postchecks-v1'
        steps = (250, 250)
    if expanded and expanded['run_id'] == run_id:
        version = expanded['folder'].name.rsplit('-v', 1)[-1]
        postchecks[run_id] = ('expanded-candidate-postchecks-v1'
                             if expanded['folder'].name == 'expanded-candidate-training-v1'
                             else 'expanded-corrective-postchecks-v' + version)
        steps = (expanded['steps'], expanded['steps'])
    if run_id not in postchecks:
        return folder
    continued = folder.parent / postchecks[run_id]
    path = continued / 'plan.json'
    plan = read(path)
    if plan is None:
        return folder
    control = read(ROOT / 'config/gpu-execution-control.json') or {}
    permit = control.get('conditional_pipelines', {}).get(continued.name, {})
    if (permit.get('plan_sha256') != identity(path)['sha256']
            or permit.get('maximum_optimizer_steps') != 0
            or permit.get('maximum_gpu_allocations') != 1
            or plan.get('maximum_optimizer_steps') != 0
            or plan.get('maximum_gpu_allocations') != 1
            or Path(plan['original_worker']).resolve() != folder.resolve()
            or Path(plan['train_output']).resolve() != (ROOT / 'runs' / run_id).resolve()):
        raise ValueError('candidate postcheck continuation does not match completed training')
    if steps is not None:
        target_step, new_steps = steps
        # This plan was registered before training produced its final weights.
        # The postcheck worker's admission binds the actual completed training
        # state; never fill this card with its parent or a previous candidate.
        admission = read(continued / 'training-admission.json')
        if admission is None:
            if (continued / 'protocol.json').exists():
                raise ValueError('continuation evaluation lacks training admission')
            return continued
        completed_path = folder / 'state.json'
        completed = read(completed_path) or {}
        completed_new_steps = completed.get('new_optimizer_steps')
        if run_id in MEMORY_RUN_BINDINGS:
            source_step = MEMORY_RUN_BINDINGS[run_id]['source_step']
            proof = read(folder/'completion.json') or {}
            if (proof.get('status') != 'passed' or proof.get('source_step') != source_step
                    or proof.get('final_step') != target_step
                    or completed.get('maximum_new_optimizer_steps') != new_steps
                    or completed.get('completion_sha256') != identity(folder/'completion.json')['sha256']):
                raise ValueError('full-state continuation completion is missing')
            completed_new_steps = proof.get('new_optimizer_steps')
        if (admission.get('status') != 'verified' or admission.get('step') != target_step
                or completed.get('status') != 'complete' or training.get('status') != 'complete'
                or training.get('optimizer_steps') != completed.get('optimizer_steps')
                or completed.get('optimizer_steps') != target_step or completed_new_steps != new_steps
                or admission.get('source_state_sha256') != identity(completed_path)['sha256']
                or admission.get('weights_sha256') != completed.get('weights_sha256')
                or completed.get('weights_sha256') != training.get('last_checkpoint', {}).get('weights_sha256')):
            raise ValueError('continuation admission does not bind the completed declared model')
    elif plan['trained_weights_sha256'] != training.get('last_checkpoint', {}).get('weights_sha256'):
        raise ValueError('candidate postcheck continuation does not match completed training')
    return continued


def memory_training_preflight(containers):
    """Observe the bounded paired-input check; it never supplies capability scores."""
    base = ROOT / 'config/guarded-rehearsal-pilot-v1'
    three_color = current_worker_folder('memory-current-three-color-model-preflight')
    current_three_color = (three_color / 'launch.json').exists()
    current = three_color if current_three_color else current_worker_folder('memory-current-model-preflight')
    correction = (current / 'launch.json').exists()
    folder = current if correction else base / 'memory-stage-preflight-v1'
    result = dict(state='invalid', live=False, gpu_allocated=False, optimizer_steps=None,
                  maximum_optimizer_steps=2, candidate_training_registered=False,
                  model_promoted=False, formal_acceptance=False)
    if correction:
        result['runtime_contract'] = 'current_policy_corrections'
    try:
        launch = read(folder / 'launch.json')
        if not launch:
            return None
        registration_path = folder / ('queue-plan.json' if correction else 'plan.json')
        registered_plan = read(registration_path)
        permit = (read(ROOT / 'config/gpu-execution-control.json') or {}).get(
            'conditional_pipelines', {}).get(folder.name, {})
        registration_digest = identity(registration_path)['sha256']
        if (registered_plan.get('schema') != ('microduck_current_model_two_update_restore_queue_v1' if correction
                                             else 'microduck_memory_paired_two_step_restore_v1')
                or permit.get('plan_sha256') != registration_digest
                or launch.get('plan_sha256') != registration_digest
                or launch.get('worker_sha256') != hashlib.sha256((folder/'run.py').read_bytes()).hexdigest()
                or any(x.get('maximum_optimizer_steps') != 2 or x.get('maximum_gpu_allocations') != 2
                       for x in (registered_plan, permit))
                or registered_plan.get('candidate_training_allowed' if correction else 'candidate_training_registered') is not False
                or registered_plan.get('model_promotion_allowed') is not False):
            return result
        try:
            fields = (Path('/proc') / str(launch['pid']) / 'stat').read_text().rsplit(')', 1)[1].split()
            process_live = fields[0] not in ('Z', 'X') and fields[19] == str(launch['proc_start_ticks'])
        except (OSError, ValueError, IndexError):
            process_live = False
        state = read(folder / 'state.json') or {}
        plan_path = folder / ('execution-plan.json' if correction else 'plan.json')
        if correction and not plan_path.exists():
            if any((folder / (role + '-launch.json')).exists() for role in ('preflight', 'resume')):
                return result
            return dict(result, state='failed' if state.get('status') == 'failed' else
                        'waiting_for_data' if process_live and state.get('status') == 'running' else 'unavailable',
                        live=process_live, optimizer_steps=0, stage=state.get('stage'),
                        sources=dict(plan=identity(registration_path), launch=identity(folder / 'launch.json')))
        plan = read(plan_path); digest = identity(plan_path)['sha256']
        if correction and (state.get('execution_plan_sha256') != digest
                or plan.get('schema') != 'microduck_current_correction_stage_model_preflight_v1'
                or plan.get('files', {}).get(str(registration_path)) != registration_digest
                or plan.get('image') != registered_plan.get('image')
                or plan.get('parent_weights_sha256') != registered_plan.get('initializer_sha256')
                or plan.get('retained_weights_sha256') != registered_plan.get('retained_sha256')
                or plan.get('maximum_optimizer_steps') != 2 or plan.get('maximum_gpu_allocations') != 2
                or plan.get('maximum_concurrent_gpus') != 1
                or plan.get('eligible_for_model_preflight') is not True
                or plan.get('preparation_scope') != 'complete_current_cohort'
                or plan.get('candidate_training_registered') is not False
                or plan.get('model_promotion_allowed') is not False):
            return result
        run = ROOT / ('runs/memory-current-three-color-corrective-pilot-v1' if current_three_color else
                      'runs/memory-current-corrective-pilot-v1' if correction
                      else 'runs/target-memory-paired-retention-pilot-v1')
        outputs = ({'preflight': run.with_name(run.name + '-preflight'),
                    'resume': run.with_name(run.name + '-restore')} if correction else
                   {role: run.with_name(run.name + '-' + role + '-v1') for role in ('preflight', 'resume')})
        if plan.get('train_output') != str(run) or any(
                plan.get(role + '_output') != str(path) for role, path in outputs.items()):
            return result
        preparation_stem = ('memory-current-three-color-stage-preparation' if current_three_color
                            else 'memory-current-stage-preparation')
        manifest_path = (current_worker_folder(preparation_stem) / 'verified/retention-manifest.json'
                         if correction else folder / 'retention-manifest.json')
        manifest = read(manifest_path)
        if (plan.get('retention_manifest') != str(manifest_path)
                or plan.get('files', {}).get(str(manifest_path)) != identity(manifest_path)['sha256']
                or manifest['input_pair']['schema'] != 'microduck_memory_retention_pair_v1'
                or manifest['input_pair']['student_history']['schema'] != 'microduck_rgb_target_keyframe_v2'
                or manifest['input_pair']['teacher_history']['offsets_ticks'] != [300, 150, 50, 0]):
            return result
        actual = {}
        for role in outputs:
            allocated = read(folder / (role + '-launch.json'))
            if not allocated:
                continue
            container = next((c for c in containers if c.get('Id') == allocated.get('container_id')), None)
            if container is None:
                # Missing observation is not evidence that an allocation stopped.
                return dict(result, state='unavailable', live=process_live)
            expected = ['accelerate', 'launch', '--config_file', plan['accelerate_config'],
                        '--main_process_port', '29642', plan['entrypoint'], '--plan', str(plan_path),
                        '--plan-sha256', digest, '--config', plan[role+'_recipe']]
            if (allocated.get('execution_plan_sha256' if correction else 'plan_sha256') != digest
                    or container.get('Image') != plan['image']
                    or container['Config'].get('Cmd') != expected
                    or container['Config'].get('User') != '1031:1031'
                    or container.get('Name', '').lstrip('/') != 'microduck-fastwam-'+folder.name+'-'+role):
                return result
            devices = [device for request in container.get('HostConfig', {}).get('DeviceRequests') or []
                       if any('gpu' in group for group in request.get('Capabilities', []))
                       for device in request.get('DeviceIDs') or []]
            if devices != [allocated.get('gpu_uuid')] or not str(devices[0]).startswith('GPU-'):
                return result
            actual[role] = container
        live_roles = [role for role, c in actual.items() if c['State']['Running']
                      and c['State'].get('Pid') and Path('/proc', str(c['State']['Pid'])).exists()]
        if len(live_roles) > 1:
            return result
        result.update(state='unavailable', live=process_live or bool(live_roles),
                      gpu_allocated=bool(live_roles), stage=state.get('stage'),
                      optimizer_steps=0, sources=dict(plan=identity(plan_path),
                      launch=identity(folder/'launch.json'), inputs=identity(manifest_path)))
        report = read(outputs['preflight']/'report.json') or {}
        if report:
            step = report.get('optimizer_steps')
            if type(step) is not int or not 0 <= step <= 2:
                return dict(result, state='invalid', optimizer_steps=None)
            result['optimizer_steps'] = step
        if state.get('status') == 'failed' or read(folder/'controller-error.json'):
            result['state'] = 'failed'
        elif state.get('status') == 'complete':
            if state.get('stage') == 'skipped_for_successful_saved_checkpoint' and not actual:
                result.update(state='skipped', optimizer_steps=0)
            else:
                completion = read(folder/'completion.json') or {}
                if (state.get('completion_sha256') != identity(folder/'completion.json')['sha256']
                        or completion.get('status') != 'passed' or completion.get('optimizer_steps_performed') != 2
                        or set(actual) != {'preflight', 'resume'}
                        or any(c['State']['Status'] != 'exited' or c['State']['Running']
                               or c['State']['ExitCode'] != 0 for c in actual.values())):
                    return dict(result, state='invalid')
                required = [outputs['preflight']/'report.json', outputs['resume']/'report.json',
                            outputs['resume']/'resume-verification.json',
                            outputs['resume']/'restore-clock-verification.json']
                bindings = completion.get('files', {})
                if not all(str(p) in bindings for p in required) or not 4 <= len(bindings) <= 16:
                    return dict(result, state='invalid')
                for name, sha in bindings.items():
                    path = Path(name).resolve()
                    if (ROOT.resolve() not in path.parents or path.stat().st_size > 16*1024*1024
                            or hashlib.sha256(path.read_bytes()).hexdigest() != sha):
                        return dict(result, state='invalid')
                restored = read(outputs['resume']/'report.json')
                exact = read(outputs['resume']/'resume-verification.json')
                clock = read(outputs['resume']/'restore-clock-verification.json')
                if (report.get('status') != 'complete' or report.get('optimizer_steps') != 2
                        or report.get('retention') != manifest
                        or report.get('input_history') != manifest['input_pair']['student_history']
                        or restored.get('status') != 'complete' or restored.get('stage') != 'resume_verified'
                        or restored.get('optimizer_steps') != 2 or restored.get('weights_updated') is not False
                        or restored.get('retention') != manifest
                        or exact.get('all_trainable_tensors_exact') is not True
                        or exact.get('optimizer_moments_and_fp32_masters_exact') is not True
                        or exact.get('optimizer_step') != 2 or exact.get('weights_updated') is not False
                        or clock.get('status') != 'verified' or clock.get('random_states_exact') is not True
                        or clock.get('scheduler_exact') is not True or clock.get('optimizer_steps_performed') != 0):
                    return dict(result, state='invalid')
                if correction and (report.get('initialization', {}).get('weights_sha256') != registered_plan['initializer_sha256']
                        or report.get('initialization', {}).get('optimizer_restored') is not False
                        or report.get('retention_teacher_initialization', {}).get('weights_sha256') != registered_plan['retained_sha256']
                        or manifest.get('teacher_weights_sha256') != registered_plan['retained_sha256']):
                    return dict(result, state='invalid')
                result.update(state='complete', optimizer_steps=2)
                result['sources']['completion'] = identity(folder/'completion.json')
        elif live_roles:
            result['state'] = 'restoring' if 'resume' in live_roles else 'training'
        elif process_live:
            result['state'] = ('waiting_for_gpu' if state.get('stage') == 'waiting_for_idle_gpu'
                               else 'waiting_for_review' if not state else 'checking')
        return result
    except (OSError, ValueError, TypeError, KeyError, IndexError):
        return dict(result, state='invalid')


def current_correction_postcheck_registration(*, three_color=False):
    """Resolve the current candidate's independent evaluation, without assuming it ran."""
    base = ROOT / 'config/guarded-rehearsal-pilot-v1'
    stem = 'memory-current-three-color' if three_color else 'memory-current'
    folder = current_worker_folder(stem + '-candidate-postchecks')
    if not (folder / 'launch.json').exists():
        return None
    plan_path = folder / 'plan.json'; plan = read(plan_path)
    launch, state = read(folder / 'launch.json'), read(folder / 'state.json') or {}
    digest = identity(plan_path)['sha256']
    permit = (read(ROOT / 'config/gpu-execution-control.json') or {}).get('conditional_pipelines', {}).get(folder.name, {})
    parent = current_worker_folder(stem + '-candidate-training')
    if (plan.get('schema') != 'microduck_current_correction_development_postchecks_v1'
            or launch.get('plan_sha256') != digest or permit.get('plan_sha256') != digest
            or launch.get('worker_sha256') != hashlib.sha256((folder / 'run.py').read_bytes()).hexdigest()
            or state.get('pid') != launch.get('pid') or plan.get('parent') != str(parent)
            or plan.get('parent_handle') != read(parent / 'launch.json')
            or plan.get('parent_plan_sha256') != identity(parent / 'queue-plan.json')['sha256']
            or plan.get('train_output') != str(ROOT / 'runs' / (stem + '-corrective-pilot-v1'))
            or plan.get('evaluation_output') != str(ROOT / 'reports' / (stem + '-corrective-pilot-v1-development'))
            or plan.get('paired_development_cases') != 81 or plan.get('physical_development_layouts') != 27
            or plan.get('clear_success_minimum') != 19 or plan.get('obstacle_success_exclusive_minimum') != 25
            or any(p.get('maximum_optimizer_steps') != 0 or p.get('maximum_gpu_allocations') != 1
                   or p.get('model_promotion_allowed') is not False for p in (plan, permit))):
        raise ValueError('current candidate development registration differs')
    for name in ('run.py', 'gates.py', 'runtime.py', 'evaluate.py', 'assess.py'):
        path = folder / name
        if plan.get('files', {}).get(str(path)) != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError('current development implementation changed')
    try:
        fields = (Path('/proc') / str(launch['pid']) / 'stat').read_text().rsplit(')', 1)[1].split()
        live = fields[0] not in ('Z', 'X') and fields[19] == str(launch['proc_start_ticks'])
    except (OSError, IndexError, ValueError):
        live = False
    stage = state.get('stage', '')
    observed = ('failed' if state.get('status') == 'failed' else 'ended' if state.get('status') == 'complete'
                else 'waiting' if live and stage.startswith('waiting_') else 'running' if live else 'unavailable')
    return dict(folder=folder, plan=plan, progress=dict(state=observed, live=live, stage=stage,
                source=identity(plan_path), optimizer_steps=0))


def current_correction_alignment_registration(*, three_color=False):
    """Show this candidate's numerical queue without borrowing prior alignment results."""
    base = ROOT / 'config/guarded-rehearsal-pilot-v1'
    stem = 'memory-current-three-color' if three_color else 'memory-current'
    folder = current_worker_folder(stem + '-candidate-alignment')
    if not (folder / 'launch.json').exists():
        return None
    plan_path = folder / 'plan.json'; plan = read(plan_path)
    launch, state = read(folder / 'launch.json'), read(folder / 'state.json') or {}
    digest = identity(plan_path)['sha256']
    permit = (read(ROOT / 'config/gpu-execution-control.json') or {}).get('conditional_pipelines', {}).get(folder.name, {})
    parent = current_worker_folder(stem + '-candidate-postchecks')
    development = current_correction_postcheck_registration(three_color=three_color)
    if (not development or plan.get('schema') != 'microduck_current_correction_alignment_v1'
            or launch.get('plan_sha256') != digest or permit.get('plan_sha256') != digest
            or state.get('pid') != launch.get('pid') or plan.get('predecessor') != str(parent)
            or plan.get('pipeline_launch_sha256') != identity(parent / 'launch.json')['sha256']
            or plan.get('bound_files', {}).get(str(parent / 'plan.json')) != identity(parent / 'plan.json')['sha256']
            or plan.get('source') != development['plan']['source']
            or plan.get('image') != development['plan']['image']
            or plan.get('variants') != ['reference', 'native_plain', 'native_optimized']
            or plan.get('clear_success_minimum') != 19 or plan.get('obstacle_success_exclusive_minimum') != 25
            or any(p.get('maximum_optimizer_steps') != 0 or p.get('maximum_gpu_allocations') != 3
                   or p.get('maximum_concurrent_gpus') != 1 or p.get('model_promotion_allowed') is not False
                   for p in (plan, permit))):
        raise ValueError('current candidate numerical registration differs')
    for name in ('run.py', 'predecessor.py', 'gates.py', 'prepare.py', 'assess.py'):
        if plan.get('helpers', {}).get(name) != hashlib.sha256((folder / name).read_bytes()).hexdigest():
            raise ValueError('current candidate numerical implementation changed')
    if launch.get('worker_sha256') != plan['helpers']['run.py']:
        raise ValueError('numerical worker identity differs')
    try:
        fields = (Path('/proc') / str(launch['pid']) / 'stat').read_text().rsplit(')', 1)[1].split()
        live = fields[0] not in ('Z', 'X') and fields[19] == str(launch['proc_start_ticks'])
    except (OSError, IndexError, ValueError):
        live = False
    stage = state.get('stage', '')
    observed = ('failed' if state.get('status') == 'failed' else 'ended' if state.get('status') == 'complete'
                else 'waiting' if live and stage.startswith('waiting_') else 'running' if live else 'unavailable')
    return dict(state=observed, live=live, stage=stage, skipped=state.get('skipped') is True,
                source=identity(plan_path), optimizer_steps=0, formal_acceptance=False, model_promoted=False)


def current_correction_realtime_registration(*, three_color=False):
    """Show the new candidate's three-arm queue, never infer performance from exit status."""
    base = ROOT / 'config/guarded-rehearsal-pilot-v1'
    stem = 'memory-current-three-color' if three_color else 'memory-current'
    folder = current_worker_folder(stem + '-candidate-realtime', (4, 3, 2, 1))
    if not (folder / 'launch.json').exists():
        return None
    plan_path = folder / 'plan.json'; plan = read(plan_path)
    launch, state = read(folder / 'launch.json'), read(folder / 'state.json') or {}
    digest = identity(plan_path)['sha256']
    permit = (read(ROOT / 'config/gpu-execution-control.json') or {}).get('conditional_pipelines', {}).get(folder.name, {})
    development = current_correction_postcheck_registration(three_color=three_color)
    alignment = current_correction_alignment_registration(three_color=three_color)
    prepared = Path(plan['source']).parent
    preparations = (('memory-current-three-color-realtime-preparation-v1', 'snapshot-camera-runtime-v5',
                     'snapshot-camera-runtime-v6') if three_color else
                    ('memory-current-realtime-preparation-v2', 'memory-current-realtime-preparation-v4'))
    if prepared.parent != base or prepared.name not in preparations:
        raise ValueError('unrecognized realtime preparation source')
    snapshot_camera = prepared.name in ('snapshot-camera-runtime-v5', 'snapshot-camera-runtime-v6')
    if snapshot_camera:
        proof_path = folder / 'camera-proof.json'
        proof = read(proof_path) or {}
        if (plan.get('camera_backend') != 'osmesa' or plan.get('snapshot_camera_workers') != 4
                or 'camera_offsamples' not in plan or plan['camera_offsamples'] is not None
                or proof.get('status') != 'verified' or proof.get('source') != plan['source']
                or proof.get('formal_acceptance') is not False or proof.get('performance_claim') is not False
                or plan.get('bound_files', {}).get(str(proof_path)) != identity(proof_path)['sha256']):
            raise ValueError('snapshot camera preparation differs')
        if prepared.name == 'snapshot-camera-runtime-v6' and (
                proof.get('post_measurement_only_fix') is not True
                or proof.get('original_failure_preserved') is not True
                or proof.get('unchanged_model_operator_files') != 851):
            raise ValueError('camera loss repair evidence differs')
    source_manifest = prepared / 'source-manifest.json'
    if (not development or not alignment or alignment.get('state') == 'invalid'
            or plan.get('schema') != 'microduck_current_correction_realtime_v1'
            or launch.get('plan_sha256') != digest or permit.get('plan_sha256') != digest
            or state.get('pid') != launch.get('pid')
            or plan.get('image') != development['plan']['image']
            or plan.get('evaluation_source') != development['plan']['source']
            or plan.get('source') != (read(source_manifest) or {}).get('source')
            or plan.get('bound_files', {}).get(str(source_manifest)) != identity(source_manifest)['sha256']
            or plan.get('backend_arms') != ['reference', 'plain', 'optimized']
            or plan.get('expiry_ms') != [500, 1000, None]
            or plan.get('planned_episodes') != 729 or plan.get('distinct_physical_layouts') != 27
            or plan.get('repetitions') != 1
            or any(p.get('maximum_optimizer_steps') != 0 or p.get('maximum_gpu_allocations') != 18
                   or p.get('maximum_concurrent_gpus') != 1 or p.get('model_promotion_allowed') is not False
                   for p in (plan, permit))):
        raise ValueError('current candidate realtime registration differs')
    for parent in (current_worker_folder(stem + '-candidate-postchecks'),
                   current_worker_folder(stem + '-candidate-alignment')):
        for name in ('plan.json', 'run.py', 'launch.json'):
            path = parent / name
            if plan.get('bound_files', {}).get(str(path)) != identity(path)['sha256']:
                raise ValueError('realtime predecessor changed')
    helpers = ('run.py', 'runtime.py', 'gates.py', 'bind.py', 'entry.py', 'assess.py', 'predecessor.py')
    if snapshot_camera:
        helpers += ('camera_proof.py', 'renderer_identity.py')
    for name in helpers:
        if plan.get('helpers', {}).get(name) != hashlib.sha256((folder / name).read_bytes()).hexdigest():
            raise ValueError('current realtime implementation changed')
    if launch.get('worker_sha256') != plan['helpers']['run.py']:
        raise ValueError('realtime worker identity differs')
    try:
        fields = (Path('/proc') / str(launch['pid']) / 'stat').read_text().rsplit(')', 1)[1].split()
        live = fields[0] not in ('Z', 'X') and fields[19] == str(launch['proc_start_ticks'])
    except (OSError, IndexError, ValueError):
        live = False
    stage = state.get('stage', '')
    observed = ('failed' if state.get('status') == 'failed' else 'ended' if state.get('status') == 'complete'
                else 'waiting' if live and stage.startswith('waiting_') else 'running' if live else 'unavailable')
    return dict(state=observed, live=live, stage=stage, skipped=state.get('skipped') is True,
                source=identity(plan_path), optimizer_steps=0, planned_episodes=729,
                physical_development_layouts=27, backend_arms=plan['backend_arms'],
                formal_acceptance=False, model_promoted=False, performance_claim=False)


def current_correction_physical_replay_registration(*, three_color=False):
    """Expose a registered full-cohort replay without promoting recorded scores."""
    stem = 'memory-current-three-color' if three_color else 'memory-current'
    folder = current_worker_folder(stem + '-candidate-physical-replay', (4, 3, 2, 1))
    if not (folder / 'launch.json').exists():
        return None
    parent = current_worker_folder(stem + '-candidate-realtime', (4, 3, 2, 1))
    realtime = current_correction_realtime_registration(three_color=three_color)
    parent_plan = read(parent / 'plan.json')
    plan_path = folder / 'plan.json'; plan = read(plan_path)
    launch, state = read(folder / 'launch.json'), read(folder / 'state.json') or {}
    digest = identity(plan_path)['sha256']
    permit = (read(ROOT / 'config/gpu-execution-control.json') or {}).get('conditional_pipelines', {}).get(folder.name, {})
    snapshot_camera = plan.get('schema') == 'microduck_current_candidate_physical_replay_plan_v2'
    expected_gpu_allocations = 0 if snapshot_camera else 1
    if snapshot_camera:
        parent_handle = read(parent / 'launch.json') or {}
        registration = read(folder / 'registration-verification.json') or {}
        observed_parent = registration.get('check', {}).get('live_parent') or {}
        if (plan.get('camera_backend') != 'osmesa' or plan.get('snapshot_camera_workers') != 4
                or 'camera_offsamples' not in plan or plan['camera_offsamples'] is not None
                or parent_plan.get('camera_backend') != 'osmesa' or parent_plan.get('snapshot_camera_workers') != 4
                or plan.get('parent') != str(parent)
                or observed_parent.get('pid') != parent_handle.get('pid')
                or observed_parent.get('proc_start_ticks') != parent_handle.get('proc_start_ticks')):
            raise ValueError('CPU replay camera or registered parent identity differs')
    if (not realtime or plan.get('schema') not in ('microduck_current_candidate_physical_replay_plan_v1',
                                                 'microduck_current_candidate_physical_replay_plan_v2')
            or launch.get('plan_sha256') != digest or permit.get('plan_sha256') != digest
            or state.get('pid') != launch.get('pid')
            or plan.get('parent_plan_sha256') != identity(parent / 'plan.json')['sha256']
            or plan.get('image') != parent_plan['image'] or plan.get('source') != parent_plan['source']
            or plan.get('planned_episodes') != 729 or plan.get('planned_jobs') != 18
            or plan.get('optimizer_steps') != 0 or permit.get('maximum_optimizer_steps') != 0
            or plan.get('camera_mismatches_remain_incomplete') is not True
            or any(p.get('maximum_gpu_allocations') != expected_gpu_allocations or p.get('maximum_concurrent_gpus') != 1
                   or p.get('model_promotion_allowed') is not False for p in (plan, permit))):
        raise ValueError('physical replay registration differs')
    # CPU replay is prepared before its new parent starts. The immutable parent
    # plan/code are bound here; its actual launched handle is checked above.
    paths = [parent / n for n in (('plan.json', 'run.py') if snapshot_camera else
                                 ('plan.json', 'run.py', 'launch.json'))]
    paths.append(Path(plan['source']).parent / 'source-manifest.json')
    for path in paths:
        if plan.get('bound_files', {}).get(str(path)) != identity(path)['sha256']:
            raise ValueError('physical replay predecessor or runtime changed')
    for name in ('run.py', 'replay.py', 'gates.py', 'replay_module.py', 'register.py'):
        if plan.get('helpers', {}).get(name) != hashlib.sha256((folder / name).read_bytes()).hexdigest():
            raise ValueError('physical replay implementation changed')
    if launch.get('worker_sha256') != plan['helpers']['run.py']:
        raise ValueError('physical replay worker identity differs')
    try:
        fields = (Path('/proc') / str(launch['pid']) / 'stat').read_text().rsplit(')', 1)[1].split()
        live = fields[0] not in ('Z', 'X') and fields[19] == str(launch['proc_start_ticks'])
    except (OSError, IndexError, ValueError):
        live = False
    stage = state.get('stage', '')
    observed = ('failed' if state.get('status') == 'failed' else 'ended' if state.get('status') == 'complete'
                else 'waiting' if live and stage.startswith('waiting_') else 'running' if live else 'unavailable')
    return dict(state=observed, live=live, stage=stage, skipped=state.get('skipped') is True,
                source=identity(plan_path), optimizer_steps=0, planned_episodes=729,
                physical_development_layouts=27, model_inference_replayed=False,
                formal_acceptance=False, model_promoted=False, performance_claim=False)


def current_correction_score_inputs(binding, protocol, training):
    """Bind new checkpoint metadata to the common independent-replay score reader."""
    folder, plan = binding['folder'], binding['plan']
    source_manifest = Path(plan['source_manifest']).resolve()
    source_manifest.relative_to(ROOT.resolve())
    manifest = read(source_manifest)
    receipt_path = Path(protocol['checkpoint_receipt']).resolve()
    receipt_path.relative_to(ROOT.resolve())
    receipt = read(receipt_path)
    if (plan['files'].get(str(source_manifest)) != identity(source_manifest)['sha256']
            or protocol.get('source_manifest_sha256') != identity(source_manifest)['sha256']
            or protocol.get('source_manifest') != str(source_manifest) or protocol.get('source') != plan['source']
            or protocol.get('output') != plan['evaluation_output']
            or protocol.get('checkpoint_receipt_sha256') != identity(receipt_path)['sha256']
            or receipt.get('step') != 250 or receipt.get('weights_sha256') != protocol['weights_sha256']
            or protocol.get('stats_sha256') != plan['stats_sha256']
            or protocol.get('initializer_sha256') != plan['initializer_sha256']
            or protocol.get('retained_teacher_sha256') != plan['retained_sha256']
            or receipt.get('initialization') != training.get('initialization')
            or receipt.get('initialization', {}).get('weights_sha256') != plan['initializer_sha256']
            or receipt.get('initialization', {}).get('optimizer_restored') is not False
            or receipt.get('retention') != training.get('retention')
            or receipt.get('retention', {}).get('teacher_weights_sha256') != plan['retained_sha256']
            or receipt.get('retention', {}).get('inference_teacher_required') is not False
            or receipt.get('input_history') != training.get('input_history')
            or protocol.get('input_history') != receipt.get('input_history')
            or protocol.get('training_axis_objective') != receipt.get('training_objective')
            or receipt.get('training_objective') != training.get('training_objective')
            or protocol.get('checkpoint_step') != 250):
        raise ValueError('current candidate model/replay binding differs')
    return dict(assessment_folder=folder, source=plan['source'], image=plan['image'],
                history=protocol['input_history'], assessor=str(folder / 'assess.py'),
                assessor_sha256=plan['files'][str(folder / 'assess.py')],
                replay_sha256=manifest['files']['scripts/microduck/assess_navigation_contract_cohort.py'])


def current_correction_candidate_progress(containers, *, three_color=None):
    """Read the current 250-step queue without publishing capability results."""
    base = ROOT / 'config/guarded-rehearsal-pilot-v1'
    if three_color is None:
        three_color = (current_worker_folder('memory-current-three-color-candidate-training') / 'launch.json').exists()
    stem = 'memory-current-three-color' if three_color else 'memory-current'
    folder = current_worker_folder(stem + '-candidate-training')
    result = dict(state='invalid', live=False, gpu_allocated=False, optimizer_steps=None,
                  maximum_optimizer_steps=250, model_promoted=False, formal_acceptance=False,
                  runtime_contract='current_policy_corrections', run_id=stem + '-corrective-pilot-v1',
                  continuations={name: dict(state='not_registered') for name in
                                 ('development', 'alignment', 'realtime', 'physical_replay', 'training_fit')})
    try:
        launch = read(folder / 'launch.json')
        if not launch:
            return None
        try:
            follow = current_correction_postcheck_registration(three_color=three_color)
            if follow:
                result['continuations']['development'] = follow['progress']
        except (OSError, ValueError, TypeError, KeyError, IndexError):
            result['continuations']['development'] = dict(state='invalid', live=False)
        try:
            alignment = current_correction_alignment_registration(three_color=three_color)
            if alignment:
                result['continuations']['alignment'] = alignment
        except (OSError, ValueError, TypeError, KeyError, IndexError):
            result['continuations']['alignment'] = dict(state='invalid', live=False)
        try:
            realtime = current_correction_realtime_registration(three_color=three_color)
            if realtime:
                result['continuations']['realtime'] = realtime
        except (OSError, ValueError, TypeError, KeyError, IndexError):
            result['continuations']['realtime'] = dict(state='invalid', live=False)
        try:
            replay = current_correction_physical_replay_registration(three_color=three_color)
            if replay:
                result['continuations']['physical_replay'] = replay
        except (OSError, ValueError, TypeError, KeyError, IndexError):
            result['continuations']['physical_replay'] = dict(state='invalid', live=False)
        queue_path = folder / 'queue-plan.json'; queue = read(queue_path)
        digest = identity(queue_path)['sha256']
        permit = (read(ROOT / 'config/gpu-execution-control.json') or {}).get(
            'conditional_pipelines', {}).get(folder.name, {})
        if (queue.get('schema') != 'microduck_current_correction_candidate_queue_v1'
                or launch.get('plan_sha256') != digest or permit.get('plan_sha256') != digest
                or launch.get('worker_sha256') != hashlib.sha256((folder / 'run.py').read_bytes()).hexdigest()
                or any(x.get('maximum_optimizer_steps') != 250 or x.get('maximum_gpu_allocations') != 1
                       or x.get('model_promotion_allowed') is not False for x in (queue, permit))
                or queue.get('maximum_concurrent_gpus') != 1):
            return result
        state = read(folder / 'state.json') or {}
        if state.get('pid') != launch.get('pid'):
            return result
        try:
            fields = (Path('/proc') / str(launch['pid']) / 'stat').read_text().rsplit(')', 1)[1].split()
            live = fields[0] not in ('Z', 'X') and fields[19] == str(launch['proc_start_ticks'])
        except (OSError, IndexError, ValueError):
            live = False
        result.update(live=live, stage=state.get('stage'), optimizer_steps=0,
                      sources=dict(queue=identity(queue_path), launch=identity(folder / 'launch.json')))
        plan_path = folder / 'execution-plan.json'
        if not plan_path.exists():
            if (folder / 'train-launch.json').exists():
                return dict(result, state='invalid')
            return dict(result, state='failed' if state.get('status') == 'failed' else
                        'waiting_for_preflight' if live and state.get('status') == 'running' else 'unavailable')
        plan = read(plan_path); plan_digest = identity(plan_path)['sha256']
        run = ROOT / 'runs' / (stem + '-corrective-pilot-v1')
        parent = current_worker_folder(stem + '-model-preflight')
        if (state.get('execution_plan_sha256') != plan_digest
                or plan.get('schema') != 'microduck_current_correction_candidate_training_v1'
                or plan.get('train_output') != str(run) or queue.get('train_output') != str(run)
                or plan.get('preflight_job') != str(parent) or queue.get('parent') != str(parent)
                or plan.get('files', {}).get(str(queue_path)) != digest
                or plan.get('parent_weights_sha256') != queue.get('initializer_sha256')
                or plan.get('retained_weights_sha256') != queue.get('retained_sha256')
                or plan.get('image') != queue.get('image') or plan.get('training_steps') != 250
                or plan.get('maximum_optimizer_steps') != 250 or plan.get('maximum_gpu_allocations') != 1
                or plan.get('maximum_concurrent_gpus') != 1 or plan.get('model_promotion_allowed') is not False
                or plan.get('preparation_scope') != 'complete_current_cohort'
                or plan.get('eligible_for_model_preflight') is not True
                or plan.get('candidate_training_registered') is not True):
            return dict(result, state='invalid')
        parent_completion = read(parent / 'completion.json') or {}
        if (plan['files'].get(str(parent / 'completion.json')) != identity(parent / 'completion.json')['sha256']
                or parent_completion.get('status') != 'passed'
                or parent_completion.get('optimizer_steps_performed') != 2):
            return dict(result, state='invalid')
        manifest_path = Path(plan['retention_manifest']); manifest = read(manifest_path)
        if (plan['files'].get(str(manifest_path)) != identity(manifest_path)['sha256']
                or manifest.get('teacher_weights_sha256') != queue['retained_sha256']
                or manifest.get('input_pair', {}).get('student_history', {}).get('schema') != 'microduck_rgb_target_keyframe_v2'):
            return dict(result, state='invalid')
        result['sources']['plan'] = identity(plan_path)
        report = read(run / 'report.json') or {}
        if report:
            step = report.get('optimizer_steps')
            if type(step) is not int or not 0 <= step <= 250:
                return dict(result, state='invalid', optimizer_steps=None)
            for key, expected in (('retention', manifest), ('input_history', manifest['input_pair']['student_history'])):
                if key in report and report[key] != expected:
                    return dict(result, state='invalid')
            result.update(optimizer_steps=step, stage=report.get('stage', result['stage']))
        allocated = read(folder / 'train-launch.json')
        actual = next((c for c in containers if allocated and c.get('Id') == allocated.get('container_id')), None)
        if allocated and actual is None:
            return dict(result, state='unavailable')
        if actual:
            devices = [d for request in actual.get('HostConfig', {}).get('DeviceRequests') or []
                       if any('gpu' in group for group in request.get('Capabilities', []))
                       for d in request.get('DeviceIDs') or []]
            expected = ['accelerate', 'launch', '--config_file', plan['accelerate_config'],
                '--main_process_port', '29642', plan['entrypoint'], '--plan', str(plan_path),
                '--plan-sha256', plan_digest, '--config', plan['train_recipe']]
            if (allocated.get('execution_plan_sha256') != plan_digest or actual.get('Image') != plan['image']
                    or actual['Config'].get('Cmd') != expected or actual['Config'].get('User') != '1031:1031'
                    or actual.get('Name', '').lstrip('/') != 'microduck-fastwam-' + folder.name + '-train'
                    or devices != [allocated.get('gpu_uuid')] or not str(devices[0]).startswith('GPU-')):
                return dict(result, state='invalid')
            cs = actual['State']; gpu_live = bool(cs['Running'] and cs.get('Pid') and Path('/proc', str(cs['Pid'])).exists())
            result.update(live=live or gpu_live, gpu_allocated=gpu_live, container_id=actual['Id'])
            if cs.get('OOMKilled') or cs.get('Error') or (cs['Status'] == 'exited' and cs['ExitCode'] != 0):
                return dict(result, state='failed')
        if state.get('status') == 'failed':
            return dict(result, state='failed')
        if state.get('status') == 'complete':
            proof = read(folder / 'completion.json') or {}; bindings = proof.get('files', {})
            required = [run / 'report.json', run / 'metrics.jsonl', run / 'scheduler-binding.json',
                        run / 'entry-completion.json', run / 'checkpoints/state/step_000250/navigation_checkpoint.json']
            if (not actual or actual['State']['Status'] != 'exited' or actual['State']['Running']
                    or actual['State']['ExitCode'] != 0 or state.get('completion_sha256') != identity(folder / 'completion.json')['sha256']
                    or proof.get('status') != 'passed' or proof.get('optimizer_steps_performed') != 250
                    or not 5 <= len(bindings) <= 10 or not all(str(p) in bindings for p in required)):
                return dict(result, state='invalid')
            for name, expected in bindings.items():
                path = Path(name).resolve()
                if (ROOT.resolve() not in path.parents or path.stat().st_size > 16 * 1024 * 1024
                        or hashlib.sha256(path.read_bytes()).hexdigest() != expected):
                    return dict(result, state='invalid')
            if (report.get('status') != 'complete' or report.get('optimizer_steps') != 250
                    or report.get('retention') != manifest or report.get('input_history') != manifest['input_pair']['student_history']
                    or report.get('initialization', {}).get('weights_sha256') != queue['initializer_sha256']
                    or report.get('initialization', {}).get('optimizer_restored') is not False
                    or report.get('retention_teacher_initialization', {}).get('weights_sha256') != queue['retained_sha256']):
                return dict(result, state='invalid')
            result['sources']['completion'] = identity(folder / 'completion.json')
            return dict(result, state='complete', optimizer_steps=250)
        return dict(result, state='training' if result['gpu_allocated'] else
                    'waiting_for_gpu' if live and state.get('stage') == 'waiting_for_idle_gpu' else
                    'checking' if live else 'unavailable')
    except (OSError, ValueError, TypeError, KeyError, IndexError):
        return dict(result, state='invalid')


def memory_candidate_progress(containers):
    """Operational progress for the real candidate, never a capability score."""
    base = ROOT / 'config/guarded-rehearsal-pilot-v1'
    if any((current_worker_folder(stem + '-candidate-training') / 'launch.json').exists()
           for stem in ('memory-current', 'memory-current-three-color')):
        return current_correction_candidate_progress(containers)
    folder = base / 'memory-candidate-training-v1'
    result = dict(state='invalid', live=False, gpu_allocated=False, optimizer_steps=None,
                  maximum_optimizer_steps=250, model_promoted=False, formal_acceptance=False)
    try:
        launch = read(folder/'launch.json')
        if not launch:
            return None
        control = read(ROOT/'config/gpu-execution-control.json') or {}

        def binding(path, schema, steps, allocations):
            plan, handle = read(path/'plan.json'), read(path/'launch.json')
            permit = control.get('conditional_pipelines', {}).get(path.name, {})
            digest = identity(path/'plan.json')['sha256']
            if (not handle or plan.get('schema') != schema
                    or handle.get('plan_sha256') != digest or permit.get('plan_sha256') != digest
                    or handle.get('worker_sha256') != hashlib.sha256((path/'run.py').read_bytes()).hexdigest()
                    or any(v.get('maximum_optimizer_steps') != steps
                           or v.get('maximum_gpu_allocations') != allocations for v in (plan, permit))
                    or permit.get('model_promotion_allowed') is not False):
                raise ValueError('memory pipeline registration changed')
            try:
                fields = (Path('/proc')/str(handle['pid'])/'stat').read_text().rsplit(')', 1)[1].split()
                live = fields[0] not in ('Z', 'X') and fields[19] == str(handle['proc_start_ticks'])
            except (OSError, ValueError, IndexError):
                live = False
            return plan, read(path/'state.json') or {}, live

        plan, state, controller_live = binding(folder, 'microduck_paired_memory_candidate_training_v1', 250, 1)
        run = ROOT/'runs/target-memory-paired-retention-pilot-v1'
        manifest_path = base/'memory-stage-preflight-v1/retention-manifest.json'
        manifest = read(manifest_path)
        if (plan.get('training_steps') != 250 or plan.get('train_output') != str(run)
                or plan.get('retention_manifest') != str(manifest_path)
                or plan.get('files', {}).get(str(manifest_path)) != identity(manifest_path)['sha256']
                or manifest['input_pair']['schema'] != 'microduck_memory_retention_pair_v1'
                or manifest['input_pair']['student_history']['schema'] != 'microduck_rgb_target_keyframe_v2'):
            raise ValueError('memory input or run binding differs')
        result.update(state='unavailable', run_id=run.name, live=controller_live, stage=state.get('stage'),
                      sources=dict(plan=identity(folder/'plan.json'), launch=identity(folder/'launch.json'),
                                   inputs=identity(manifest_path)))
        report = read(run/'report.json') or {}
        if report:
            step = report.get('optimizer_steps')
            if (type(step) is not int or not 0 <= step <= 250 or report.get('retention') != manifest
                    or report.get('input_history') != manifest['input_pair']['student_history']):
                raise ValueError('training report belongs to another input contract')
            result.update(optimizer_steps=step, stage=report.get('stage'))
            result['sources']['training'] = identity(run/'report.json')
        allocated = read(folder/'train-launch.json')
        actual = next((c for c in containers if allocated and c.get('Id') == allocated.get('container_id')), None)
        if allocated and actual is None:
            return result  # An unavailable observation is not a stopped allocation.
        if actual:
            expected = ['accelerate', 'launch', '--config_file', plan['accelerate_config'],
                        '--main_process_port', '29642', plan['entrypoint'], '--plan', str(folder/'plan.json'),
                        '--plan-sha256', identity(folder/'plan.json')['sha256'], '--config', plan['train_recipe']]
            devices = [d for request in actual.get('HostConfig', {}).get('DeviceRequests') or []
                       if any('gpu' in group for group in request.get('Capabilities', []))
                       for d in request.get('DeviceIDs') or []]
            if (allocated.get('plan_sha256') != identity(folder/'plan.json')['sha256']
                    or actual.get('Image') != plan['image'] or actual['Config'].get('Cmd') != expected
                    or actual['Config'].get('User') != '1031:1031'
                    or actual.get('Name', '').lstrip('/') != 'microduck-fastwam-'+folder.name+'-train'
                    or devices != [allocated.get('gpu_uuid')] or not str(devices[0]).startswith('GPU-')):
                raise ValueError('training allocation differs')
            status = actual['State']
            gpu_live = bool(status.get('Running') and status.get('Pid') and Path('/proc', str(status['Pid'])).exists())
            result.update(gpu_allocated=gpu_live, live=controller_live or gpu_live,
                          container_id=actual['Id'])
            failed = status.get('OOMKilled') or status.get('Error') or (
                status.get('Status') == 'exited' and status.get('ExitCode') != 0)
            if failed:
                result['state'] = 'failed'
            elif gpu_live:
                result['state'] = 'training'
            elif (status.get('Status') == 'exited' and status.get('ExitCode') == 0
                    and not status.get('Running') and state.get('status') == 'complete'
                    and report.get('status') == 'complete' and result['optimizer_steps'] == 250):
                result['state'] = 'complete'
            elif controller_live:
                result['state'] = 'checking'
        elif controller_live:
            result['state'] = 'waiting_for_gpu' if state.get('stage') == 'waiting_for_idle_gpu' else 'checking'
        if state.get('status') == 'failed':
            result['state'] = 'failed'
        stages = {}
        alignment_version = 2 if (base/'memory-candidate-success-alignment-v2/launch.json').exists() else 1
        fit_version = 2 if (base/'memory-training-fit-execution-v2/launch.json').exists() else 1
        for role, name, schema, allocations in (
                ('development', 'memory-candidate-postchecks-v1', 'microduck_paired_memory_development_postchecks_v1', 1),
                ('alignment', f'memory-candidate-success-alignment-v{alignment_version}',
                 f'microduck_memory_alignment_v{alignment_version}', 3),
                ('training_fit', f'memory-training-fit-execution-v{fit_version}',
                 'microduck_memory_training_fit_execution_v1', 2)):
            path = base/name
            if not (path/'launch.json').exists():
                stages[role] = dict(state='not_registered', live=False)
                continue
            try:
                follow, progress, live = binding(path, schema, 0, allocations)
                if role == 'development' and (follow.get('train_output') != str(run)
                        or follow.get('original_worker') != str(folder)
                        or follow.get('final_global_step') != 250 or follow.get('paired_development_cases') != 81):
                    raise ValueError('development queue belongs to another training run')
                if role == 'alignment':
                    predecessor = base/('memory-candidate-replay-completion-v1' if alignment_version == 2
                                        else 'memory-candidate-postchecks-v1')
                    if follow.get('pipeline_launch_sha256') != identity(predecessor/'launch.json')['sha256']:
                        raise ValueError('alignment queue belongs to another development run')
                    if alignment_version == 2:
                        original, previous = base/'memory-candidate-postchecks-v1', base/'memory-candidate-success-alignment-v1'
                        if (follow.get('predecessor') != str(predecessor)
                                or follow.get('original_evaluation') != str(original)
                                or follow.get('supersedes') != str(previous)
                                or follow.get('original_pipeline_launch_sha256') != identity(original/'launch.json')['sha256']
                                or follow.get('superseded_alignment_launch_sha256') != identity(previous/'launch.json')['sha256']):
                            raise ValueError('repaired alignment predecessor differs')
                        required = {'run.py', 'prepare.py', 'assess.py', 'predecessor.py', 'gates.py'}
                        if not required.issubset(follow.get('helpers', {})):
                            raise ValueError('repaired alignment helpers are incomplete')
                        for helper in required:
                            if hashlib.sha256((path/helper).read_bytes()).hexdigest() != follow['helpers'][helper]:
                                raise ValueError('repaired alignment helper changed')
                if role == 'training_fit':
                    expected = {
                        'predecessor': base/'memory-candidate-replay-completion-v1',
                        'original_evaluation': base/'memory-candidate-postchecks-v1',
                        'success_alignment': base/'memory-candidate-success-alignment-v2',
                        'superseded_alignment': base/'memory-candidate-success-alignment-v1',
                    }
                    probe_path = base/'memory-training-fit-v2/probe-plan.json'
                    bound = follow.get('bound_files', {})
                    if (any(follow.get(key) != str(value) for key, value in expected.items())
                            or follow.get('probe_plan') != str(probe_path)
                            or follow.get('probe_plan_sha256') != identity(probe_path)['sha256']
                            or follow.get('model_promotion') is not False
                            or follow.get('formal_acceptance') is not False):
                        raise ValueError('training-fit scope or predecessor differs')
                    for dependency in [probe_path, *(p/'launch.json' for p in expected.values())]:
                        if bound.get(str(dependency)) != identity(dependency)['sha256']:
                            raise ValueError('training-fit dependency changed')
                    if fit_version == 2:
                        previous = base/'memory-training-fit-execution-v1'
                        if (follow.get('supersedes') != str(previous)
                                or (read(previous/'state.json') or {}).get('status') != 'failed'):
                            raise ValueError('training-fit repair does not follow a failed allocation')
                        for name in ('plan.json', 'launch.json', 'state.json'):
                            dependency = previous/name
                            if bound.get(str(dependency)) != identity(dependency)['sha256']:
                                raise ValueError('training-fit failed predecessor changed')
                    required = {'run.py', 'checks.py', 'assess.py', 'fit_metrics.py'}
                    if not required.issubset(follow.get('helpers', {})):
                        raise ValueError('training-fit helpers are incomplete')
                    for helper in required:
                        if hashlib.sha256((path/helper).read_bytes()).hexdigest() != follow['helpers'][helper]:
                            raise ValueError('training-fit helper changed')
                stage = progress.get('stage', '')
                observed = ('failed' if progress.get('status') == 'failed' else
                            'ended' if progress.get('status') == 'complete' else
                            'waiting' if live and stage.startswith('waiting_') else
                            'running' if live else 'unavailable')
                stages[role] = dict(state=observed, live=live, stage=stage,
                                    source=identity(path/'plan.json'))
                if role == 'training_fit':
                    stages[role].update(new_optimizer_steps=0, formal_acceptance=False,
                                        evidence_kind='training_input_diagnostic',
                                        skipped=progress.get('skipped') is True)
            except (OSError, ValueError, TypeError, KeyError, IndexError):
                stages[role] = dict(state='invalid', live=False)
        repair_path = base/'memory-candidate-replay-completion-v1'
        if (repair_path/'launch.json').exists():
            try:
                original = base/'memory-candidate-postchecks-v1'
                memory_replay_completion(original, read(original/'protocol.json'))
                _, progress, live = binding(repair_path, 'microduck_memory_replay_completion_v1', 0, 0)
                stage = progress.get('stage', '')
                observed = ('failed' if progress.get('status') == 'failed' else
                            'ended' if progress.get('status') == 'complete' else
                            'waiting' if live and stage.startswith('waiting_') else
                            'running' if live else 'unavailable')
                stages['replay_completion'] = dict(state=observed, live=live, stage=stage,
                    source=identity(repair_path/'plan.json'), gpu_allocated=False)
                if stages.get('development', {}).get('state') in ('failed', 'ended'):
                    stages['development'] = stages['replay_completion']
            except (OSError, ValueError, TypeError, KeyError, IndexError):
                stages['replay_completion'] = dict(state='invalid', live=False)
        result['continuations'] = stages
        return result
    except (OSError, ValueError, TypeError, KeyError, IndexError):
        return dict(result, state='invalid', optimizer_steps=None, live=False, gpu_allocated=False)


def registered_checkpoint_review():
    """Locate an explicitly registered saved model without changing training identity."""
    base = ROOT / 'config/guarded-rehearsal-pilot-v1'
    control = read(ROOT / 'config/gpu-execution-control.json') or {}
    candidates = []
    for step in (250, 500, 750):
        folder = base / f'expanded-checkpoint-{step}-postchecks-v1'
        launch = read(folder / 'launch.json')
        if launch:
            candidates.append((launch.get('started_unix', 0), step, folder, launch, None))
    for worker, binding in MEMORY_CHECKPOINT_REVIEWS.items():
        folder = base / worker
        launch = read(folder / 'launch.json')
        if launch:
            candidates.append((launch.get('started_unix', 0), binding['checkpoint_step'], folder, launch, binding))
    if not candidates:
        return None
    _, step, folder, launch, memory_binding = max(candidates, key=lambda row: row[0])
    final_step = memory_binding['target_step'] if memory_binding else 1000
    new_steps = final_step-memory_binding['source_step'] if memory_binding else 1000
    run_id = memory_binding['run_id'] if memory_binding else 'expanded-candidate-corrective-pilot-v1-1000'
    original_worker = memory_binding['worker'] if memory_binding else 'expanded-candidate-training-v1'
    evaluation_id = (memory_binding['evaluation_id'] if memory_binding else
                     f'expanded-candidate-checkpoint-{step:06d}-development-v1')
    plan = read(folder / 'plan.json')
    permit = control.get('conditional_pipelines', {}).get(folder.name, {})
    digest = identity(folder / 'plan.json')['sha256']
    if (not plan or permit.get('plan_sha256') != digest or launch.get('plan_sha256') != digest
            or plan.get('maximum_optimizer_steps') != 0 or permit.get('maximum_optimizer_steps') != 0
            or plan.get('maximum_gpu_allocations') != 1 or permit.get('maximum_gpu_allocations') != 1
            or plan.get('checkpoint_step') != step or plan.get('final_global_step') != final_step
            or plan.get('new_optimizer_steps') != new_steps
            or plan.get('paired_development_cases') != 81
            or identity(folder/'run.py')['sha256'] != launch.get('worker_sha256')):
        raise ValueError('saved checkpoint review registration differs')
    if memory_binding and (
            plan.get('schema') != 'microduck_selected_memory_continuation_development_v1'
            or plan.get('source_step') != memory_binding['source_step']
            or permit.get('model_promotion_allowed') is not False
            or plan.get('bound_files', {}).get(str(folder/'run.py')) != launch['worker_sha256']
            or registered_memory_continuation_name('microduck-fastwam-'+original_worker+'-train') is None):
        raise ValueError('saved memory checkpoint registration differs')
    run = Path(plan['train_output']).resolve()
    if (run.parent != (ROOT/'runs').resolve() or run.name != run_id
            or Path(plan['original_worker']).resolve() != (base/original_worker).resolve()
            or Path(plan['evaluation_output']).resolve()
            != (ROOT/'reports'/evaluation_id).resolve()):
        raise ValueError('saved checkpoint review uses another run or output')
    receipt_path = run / f'checkpoints/state/step_{step:06d}/navigation_checkpoint.json'
    receipt, training = read(receipt_path), read(run/'report.json')
    completed_path = base/original_worker/'state.json'
    completed = read(completed_path)
    if (not receipt or not training or not completed or training.get('status') != 'complete'
            or training.get('optimizer_steps') != final_step or completed.get('status') != 'complete'):
        raise ValueError('saved checkpoint review requires completed original training')
    completed_steps = completed.get('maximum_new_optimizer_steps') if memory_binding else completed.get('new_optimizer_steps')
    if (completed.get('optimizer_steps') != final_step or completed_steps != new_steps
            or completed.get('weights_sha256') != plan['final_weights_sha256']
            or training.get('last_checkpoint', {}).get('weights_sha256') != plan['final_weights_sha256']
            or receipt.get('weights_sha256') != plan['selected_weights_sha256']
            or receipt.get('step') != step
            or Path(receipt['weights_path']).resolve() != run/f'checkpoints/weights/step_{step:06d}.pt'
            or plan['selected_weights_sha256'] == plan['final_weights_sha256']):
        raise ValueError('saved checkpoint identity differs from the completed run')
    if memory_binding:
        final_receipt_path = run/f'checkpoints/state/step_{final_step:06d}/navigation_checkpoint.json'
        final_receipt = read(final_receipt_path) or {}
        changing = {'step', 'weights_path', 'state_path', 'weights_sha256', 'training_state_files'}
        if (plan.get('final_receipt') != str(final_receipt_path)
                or final_receipt.get('step') != final_step
                or final_receipt.get('weights_sha256') != plan['final_weights_sha256']
                or {k:v for k,v in final_receipt.items() if k not in changing}
                != {k:v for k,v in receipt.items() if k not in changing}):
            raise ValueError('saved checkpoint changed the trained input or action contract')
    admission = read(folder/'training-admission.json')
    if admission is not None and (admission.get('status') != 'verified' or admission.get('step') != step
            or admission.get('completed_training_step') != final_step
            or admission.get('weights_sha256') != receipt['weights_sha256']
            or admission.get('source_state_sha256') != identity(completed_path)['sha256']):
        raise ValueError('saved checkpoint admission differs')
    if (folder/'protocol.json').exists() and admission is None:
        raise ValueError('saved checkpoint evaluation lacks admission')
    return dict(folder=folder, launch=launch, plan=plan, step=step, completed_step=final_step,
                memory_binding=memory_binding,
                training_run_id=run.name, weights_sha256=receipt['weights_sha256'],
                receipt_source=identity(receipt_path))


def saved_checkpoint_review(containers):
    try:
        selected = registered_checkpoint_review()
        if selected is None:
            return None
        value = guarded_candidate_development(containers, selected['training_run_id'], _selected=selected)
        folder, launch = selected['folder'], selected['launch']
        state = read(folder/'state.json') or {}
        try:
            fields = (Path('/proc')/str(launch['pid'])/'stat').read_text().rsplit(')', 1)[1].split()
            process_live = fields[0] not in ('Z', 'X') and fields[19] == str(launch['proc_start_ticks'])
        except (OSError, ValueError, IndexError):
            process_live = False
        active = next((c for c in containers if c.get('Id') == state.get('waiting_container')), None)
        allocation_live = bool(active and active['State']['Running'] and active['State'].get('Pid'))
        value.update(live=process_live or allocation_live, process_live=process_live,
                     stage=state.get('stage'), container_id=state.get('waiting_container'),
                     new_optimizer_steps=0, selected_checkpoint_step=selected['step'],
                     training_completed_step=selected['completed_step'], checkpoint_sha256=selected['weights_sha256'],
                     model_promoted=False, formal_acceptance=False,
                     selection_sources=dict(plan=identity(folder/'plan.json'), receipt=selected['receipt_source']))
        if state.get('status') == 'failed':
            value.update(state='failed', groups=[])
        return value
    except (OSError, ValueError, TypeError, KeyError, IndexError):
        return dict(state='invalid', live=False, groups=[], formal_acceptance=False, model_promoted=False)


def memory_score_registration(folder):
    """A named run alone cannot authorize publishing memory-candidate scores."""
    if folder.name in MEMORY_CONTINUATIONS:
        if registered_memory_continuation_name('microduck-fastwam-'+folder.name+'-train') is None:
            raise ValueError('full-state continuation registration differs')
        return
    axis = re.fullmatch(r'memory-axis-(control|weighted)-training-v1', folder.name)
    if axis:
        name = 'microduck-fastwam-'+folder.name+'-train'
        if registered_axis_run_name(name) != 'microduck-fastwam-memory-axis-'+axis.group(1)+'-pilot-v1':
            raise ValueError('axis candidate training registration differs')
        return
    plan, launch = read(folder/'plan.json'), read(folder/'launch.json')
    permit = (read(ROOT/'config/gpu-execution-control.json') or {}).get(
        'conditional_pipelines', {}).get(folder.name, {})
    if (not plan or not launch or plan.get('schema') != 'microduck_paired_memory_candidate_training_v1'
            or launch.get('plan_sha256') != identity(folder/'plan.json')['sha256']
            or permit.get('plan_sha256') != launch['plan_sha256']
            or launch.get('worker_sha256') != hashlib.sha256((folder/'run.py').read_bytes()).hexdigest()
            or plan.get('train_output') != str(ROOT/'runs/target-memory-paired-retention-pilot-v1')
            or plan.get('training_steps') != 250
            or any(x.get('maximum_optimizer_steps') != 250 or x.get('maximum_gpu_allocations') != 1
                   for x in (plan, permit))):
        raise ValueError('memory candidate training registration differs')


def memory_replay_completion(folder, protocol):
    """Resolve the registered CPU repair without changing the evaluated source."""
    repair = folder.parent/'memory-candidate-replay-completion-v1'
    plan = read(repair/'plan.json')
    if plan is None:
        return None
    handle = read(repair/'launch.json') or {}
    permit = (read(ROOT/'config/gpu-execution-control.json') or {}).get(
        'conditional_pipelines', {}).get(repair.name, {})
    if (plan.get('schema') != 'microduck_memory_replay_completion_v1'
            or plan.get('original_postchecks') != str(folder)
            or plan.get('protocol') != str(folder/'protocol.json')
            or plan.get('protocol_sha256') != identity(folder/'protocol.json')['sha256']
            or handle.get('plan_sha256') != identity(repair/'plan.json')['sha256']
            or permit.get('plan_sha256') != handle['plan_sha256']
            or handle.get('worker_sha256') != hashlib.sha256((repair/'run.py').read_bytes()).hexdigest()
            or any(x.get('maximum_optimizer_steps') != 0 or x.get('maximum_gpu_allocations') != 0
                   for x in (plan, permit))
            or plan.get('fixed_cases') != 81 or plan.get('retained_clear_minimum') != 19
            or plan.get('obstacle_minimum_exclusive') != 25):
        raise ValueError('CPU replay repair registration differs')
    for name, digest in plan['bound_files'].items():
        file = Path(name).resolve(); file.relative_to(ROOT.resolve())
        if identity(file)['sha256'] != digest:
            raise ValueError('CPU replay repair input changed')
    manifest_path = Path(plan['source_manifest']).resolve()
    manifest_path.relative_to(repair.resolve())
    source = Path(plan['source']).resolve(); source.relative_to(repair.resolve())
    manifest, parent = read(manifest_path), read(Path(protocol['source_manifest']))
    wrapper = 'scripts/microduck/assess_navigation_contract_cohort.py'
    digest = 'a504c0a0436c6b89ba35b5216681c25dfcf0ba7545413c1345197a4cd599aa65'
    if (plan.get('wrapper_sha256') != digest or wrapper in parent['files']
            or manifest.get('source') != str(source)
            or manifest.get('parent_manifest_sha256') != identity(Path(protocol['source_manifest']))['sha256']
            or manifest.get('files') != dict(parent['files'], **{wrapper:digest})):
        raise ValueError('CPU replay source changed more than the missing scheduler')
    for name, expected in manifest['files'].items():
        path = (source/name).resolve(); path.relative_to(source)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('CPU replay source changed')
    return dict(folder=repair, source=source, manifest=manifest,
                source_manifest=str(manifest_path), plan=plan)


def memory_score_inputs(folder, training, protocol, selected=None):
    """Bind the new memory inputs to their trained receipt and frozen replayer."""
    plan, launch = read(folder/'plan.json'), read(folder/'launch.json')
    axis = re.fullmatch(r'memory-axis-(control|weighted)-postchecks-v1', folder.name)
    continuation = selected['memory_binding'] if selected else MEMORY_POSTCHECK_BINDINGS.get(folder.name)
    schema = ('microduck_selected_memory_continuation_development_v1' if selected else
              'microduck_full_state_memory_development_postchecks_v1' if continuation else
              'microduck_axis_memory_development_postchecks_v1' if axis
              else 'microduck_paired_memory_development_postchecks_v1')
    final_step = continuation['target_step'] if continuation else 250
    step = selected['step'] if selected else final_step
    source_step = continuation['source_step'] if continuation else 0
    new_steps = final_step-source_step
    if axis and registered_axis_evaluation_name('microduck-fastwam-'+folder.name+'-evaluate') is None:
        raise ValueError('axis evaluation registration differs')
    if continuation and not selected and registered_memory_continuation_evaluation('microduck-fastwam-'+folder.name+'-evaluate') is None:
        raise ValueError('continuation evaluation registration differs')
    permit = (read(ROOT/'config/gpu-execution-control.json') or {}).get(
        'conditional_pipelines', {}).get(folder.name, {})
    if (plan.get('schema') != schema or not launch
            or launch.get('plan_sha256') != identity(folder/'plan.json')['sha256']
            or permit.get('plan_sha256') != launch['plan_sha256']
            or launch.get('worker_sha256') != hashlib.sha256((folder/'run.py').read_bytes()).hexdigest()
            or plan.get('final_global_step') != final_step or plan.get('new_optimizer_steps') != new_steps
            or plan.get('paired_development_cases') != 81):
        raise ValueError('memory evaluation registration differs')

    def pinned(path):
        path = Path(path).resolve()
        path.relative_to(ROOT.resolve())
        if (plan.get('bound_files', {}).get(str(path)) != identity(path)['sha256']
                or path.stat().st_size > 16*1024*1024):
            raise ValueError('memory evaluation source is not pinned')
        return read(path)

    run = ROOT/'runs'/(continuation['run_id'] if continuation else
                       f'memory-axis-{axis.group(1)}-pilot-v1' if axis
                       else 'target-memory-paired-retention-pilot-v1')
    receipt_path = run/f'checkpoints/state/step_{step:06d}/navigation_checkpoint.json'
    receipt = read(receipt_path)
    admitted = pinned(plan['data_admission'])
    retention = pinned(plan['retention_manifest'])
    manifest = pinned(plan['evaluation_source_manifest'])
    parent = pinned(plan['parent_evaluation_protocol'])
    if continuation:
        restored = read(run/'resume-verification.json') or {}
        clock = read(run/'restore-clock-verification.json') or {}
        previous = pinned(plan['resume_source_receipt']) or {}
        objective = retention.get('supervised_axis_objective') or {}
        if (objective.get('kind') != 'action_axis' or objective.get('weights') != [1]*5
                or objective.get('axis_order') != plan['axis_config']['axis_order']
                or retention.get('supervised_objective') != 'action_axis'
                or any(x.get('training_objective') != objective or x.get('initialization') is not None
                       for x in (receipt or {}, training))
                or protocol.get('training_axis_objective') != objective
                or protocol.get('resume_source_receipt') != plan['resume_source_receipt']
                or protocol.get('resume_source_receipt_sha256') != identity(Path(plan['resume_source_receipt']))['sha256']
                or protocol.get('resume_source_weights_sha256') != plan['resume_source_weights_sha256']
                or previous.get('step') != source_step or previous.get('weights_sha256') != plan['resume_source_weights_sha256']
                or protocol.get('source_step') != source_step or protocol.get('new_optimizer_steps') != new_steps
                or protocol.get('resume_verification') != str(run/'resume-verification.json')
                or protocol.get('restore_clock') != str(run/'restore-clock-verification.json')
                or restored != training.get('resume_verification') or restored.get('optimizer_step') != source_step
                or restored.get('all_trainable_tensors_exact') is not True
                or restored.get('optimizer_moments_and_fp32_masters_exact') is not True
                or restored.get('weights_updated') is not False
                or clock.get('status') != 'verified'
                or clock.get('clock') != dict(global_step=source_step,epoch=0,batch_in_epoch=source_step*8)
                or clock.get('sampler_resume_batch_offset') != source_step*8
                or clock.get('remaining_sampling_items') != 30927-source_step*8
                or clock.get('scheduler_exact') is not True or clock.get('random_states_exact') is not True
                or clock.get('optimizer_steps_performed') != 0
                or clock.get('source_receipt_sha256') != protocol.get('resume_source_receipt_sha256')):
            raise ValueError('full-state continuation lineage or sampling progress differs')
        for path in (run/'resume-verification.json', run/'restore-clock-verification.json', Path(plan['resume_source_receipt'])):
            if protocol.get('cohort_sources', {}).get(str(path)) != identity(path)['sha256']:
                raise ValueError('continuation protocol omits restoration evidence')
    if axis:
        objective = retention.get('supervised_axis_objective')
        initialization = plan.get('initialization_weights_sha256')
        if (not isinstance(objective, dict) or objective.get('kind') != 'action_axis'
                or retention.get('supervised_objective') != 'action_axis'
                or any(objective.get(k) != plan['axis_config'][k] for k in ('axis_order', 'weights'))
                or any(x.get('training_objective') != objective for x in (receipt or {}, training))
                or protocol.get('training_axis_objective') != objective
                or not initialization or initialization == plan.get('parent_weights_sha256')
                or any(x.get('initialization', {}).get('weights_sha256') != initialization
                       for x in (receipt or {}, training))
                or protocol.get('initialization_weights_sha256') != initialization):
            raise ValueError('axis candidate objective or initialization differs')
    history = admitted.get('input_history')
    if (not receipt or admitted.get('status') != 'complete' or not isinstance(history, dict)
            or history.get('schema') != 'microduck_rgb_target_keyframe_v2'
            or history.get('current_target_position_available') is not False
            or any(x.get('input_history') != history for x in (receipt, training, protocol))
            or receipt.get('step') != step or protocol.get('checkpoint_step') != step
            or protocol.get('checkpoint_receipt') != str(receipt_path)
            or protocol.get('checkpoint_receipt_sha256') != identity(receipt_path)['sha256']
            or receipt.get('weights_sha256') != protocol.get('weights_sha256')
            or receipt.get('weights_sha256') != (selected['weights_sha256'] if selected else
                                                training.get('last_checkpoint', {}).get('weights_sha256'))
            or receipt.get('weights_path') != str(run/f'checkpoints/weights/step_{step:06d}.pt')
            or protocol.get('checkpoint') != receipt['weights_path']
            or any(x.get('retention') != retention for x in (receipt, training))
            or retention['input_pair']['student_history'] != history
            or receipt.get('extended_navigation_training') != admitted.get('extended_contract')
            or training.get('extended_navigation_training') != admitted.get('extended_contract')
            or protocol.get('cases') != parent.get('cases')
            or protocol.get('output') != plan['evaluation_output']
            or protocol.get('source') != plan['evaluation_source']
            or protocol.get('source_manifest') != plan['evaluation_source_manifest']
            or protocol.get('source_manifest_sha256') != identity(Path(plan['evaluation_source_manifest']))['sha256']):
        raise ValueError('trained memory receipt or fixed cohort differs')
    for path in (receipt_path, run/'report.json', folder/'plan.json', Path(plan['data_admission']),
                 Path(plan['retention_manifest']), Path(plan['evaluation_source_manifest'])):
        if protocol.get('cohort_sources', {}).get(str(path)) != identity(path)['sha256']:
            raise ValueError('memory protocol omits a bound training input')
    # Axis runs already contain the complete replay package. A repair registered
    # for the historical memory candidate must never redirect a new arm.
    repair = None if axis or continuation else memory_replay_completion(folder, protocol)
    replay = Path(plan['replay_script']).resolve()
    replay.relative_to(Path(plan['evaluation_source']).resolve())
    # Pin the implementation of the history selector and exact replay, not only
    # the unchanged CLI wrapper. These files are small and read-only.
    names = ['scripts/microduck/assess_navigation_contract_cohort.py',
             'embodied/microduck/src/lightx2v_microduck/navigation_cohort_replay.py',
             'embodied/microduck/src/lightx2v_microduck/navigation_history.py',
             'embodied/microduck/src/lightx2v_microduck/navigation_visual_memory.py']
    if replay != Path(plan['evaluation_source'])/names[0]:
        raise ValueError('memory replay entry point differs')
    replay_source = repair['source'] if repair else Path(plan['evaluation_source'])
    replay_manifest = repair['manifest'] if repair else manifest
    for name in names:
        path = replay_source/name
        if hashlib.sha256(path.read_bytes()).hexdigest() != replay_manifest['files'].get(name):
            raise ValueError('memory replay implementation changed')
    assessment_folder = repair['folder'] if repair else folder
    assessor = assessment_folder/'assess.py'
    bound = repair['plan']['bound_files'] if repair else plan['bound_files']
    if bound.get(str(assessor)) != hashlib.sha256(assessor.read_bytes()).hexdigest():
        raise ValueError('memory assessment implementation changed')
    return dict(history=history, replay_sha256=replay_manifest['files'][names[0]],
                source=str(replay_source), image=plan['image'], assessment_folder=assessment_folder,
                assessor=str(assessor), assessor_sha256=bound[str(assessor)],
                replay_source_manifest=repair['source_manifest'] if repair else None)


def guarded_candidate_development(containers, run_id=None, *, _selected=None):
    """Publish the fixed mixed-scene candidate only after independent replay."""
    candidates = {
        'guarded-navigation-pilot-v1': 'guarded-model-preflight-v1',
        'guarded-rehearsal-pilot-v1': 'guarded-rehearsal-pilot-v1',
        'action-scope-pilot-v1': 'guarded-rehearsal-pilot-v1/action-scope-pilot-v1',
        'action-scope-corrective-pilot-v1': 'guarded-rehearsal-pilot-v1/action-scope-corrective-pilot-v1',
        'action-scope-continuation-pilot-v2': 'guarded-rehearsal-pilot-v1/action-scope-continuation-pilot-v2',
        'action-scope-epoch-pilot-v2': 'guarded-rehearsal-pilot-v1/action-scope-epoch-pilot-v2',
        'retention-action-pilot-v2': 'guarded-rehearsal-pilot-v1/retention-action-pilot-v2',
        'retention-strong-pilot-v1': 'guarded-rehearsal-pilot-v1/retention-strong-pilot-v1',
        'retained-corrective-pilot-v1': 'guarded-rehearsal-pilot-v1/retained-corrective-pilot-v1',
    }
    expanded_workers = ['expanded-candidate-training-v1'] + [
        f'expanded-corrective-training-job-v{version}' for version in (4, 3, 2, 1)]
    expanded = next((binding for worker in expanded_workers
                     if (binding := registered_expanded_plan(worker))), None)
    if expanded:
        candidates[expanded['run_id']] = 'guarded-rehearsal-pilot-v1/' + expanded['folder'].name
    candidates['target-memory-paired-retention-pilot-v1'] = 'guarded-rehearsal-pilot-v1/memory-candidate-training-v1'
    for arm in ('weighted', 'control'):
        name = f'memory-axis-{arm}-pilot-v1'
        worker = f'memory-axis-{arm}-training-v1'
        if registered_axis_run_name('microduck-fastwam-'+worker+'-train') == 'microduck-fastwam-'+name:
            candidates[name] = 'guarded-rehearsal-pilot-v1/'+worker
    for worker, binding in MEMORY_CONTINUATIONS.items():
        if registered_memory_continuation_name('microduck-fastwam-'+worker+'-train'):
            candidates[binding['run_id']] = 'guarded-rehearsal-pilot-v1/'+worker
    current_correction_folder = current_worker_folder('memory-current-candidate-training')
    if (current_correction_folder / 'launch.json').exists():
        candidates['memory-current-corrective-pilot-v1'] = str(current_correction_folder.relative_to(ROOT / 'config'))
    three_color_folder = current_worker_folder('memory-current-three-color-candidate-training')
    if (three_color_folder / 'launch.json').exists():
        candidates['memory-current-three-color-corrective-pilot-v1'] = str(three_color_folder.relative_to(ROOT / 'config'))
    if run_id is None:
        # A new training run starts with pending evidence. Never fill its card
        # with the previous candidate's completed scores.
        run_id = next((name for name in reversed(candidates)
                       if (ROOT / 'runs' / name / 'report.json').exists()), None)
    if _selected:
        candidates[run_id] = str(_selected['folder'].relative_to(ROOT/'config'))
    if run_id not in candidates:
        return None
    folder = ROOT / 'config' / candidates[run_id]
    scoped = run_id in ('action-scope-pilot-v1', 'action-scope-corrective-pilot-v1',
                       'action-scope-continuation-pilot-v2', 'action-scope-epoch-pilot-v2',
                       'retention-action-pilot-v2', 'retention-strong-pilot-v1', 'retained-corrective-pilot-v1')
    is_expanded = bool(expanded and expanded['run_id'] == run_id)
    is_memory = (run_id == 'target-memory-paired-retention-pilot-v1' or run_id in MEMORY_RUN_BINDINGS
                 or re.fullmatch(r'memory-axis-(control|weighted)-pilot-v1', run_id) is not None)
    is_three_color = run_id == 'memory-current-three-color-corrective-pilot-v1'
    is_current = run_id == 'memory-current-corrective-pilot-v1' or is_three_color
    scoped = scoped or is_expanded or is_memory or is_current or bool(_selected)
    target_step = expanded['steps'] if is_expanded else CONTINUATION_STEPS.get(run_id, (250, 250))[0]
    run = ROOT / 'runs' / run_id / 'report.json'
    if not run.exists():
        return None
    value = dict(state='pending', mode='offline', run_id=run_id, total=81, completed=0, groups=[],
                 training_target_step=target_step, formal_acceptance=False, model_promoted=False)
    try:
        if is_current:
            current = current_correction_candidate_progress(containers, three_color=is_three_color)
            if not current or current['state'] in ('invalid', 'unavailable'):
                raise ValueError('current correction candidate identity is unverified')
            current_binding = current_correction_postcheck_registration(three_color=is_three_color)
            value.update(training_step=current['optimizer_steps'], training_status=read(run).get('status'),
                         development_registered=current_binding is not None, sources=current.get('sources', {}))
            if current['state'] == 'failed':
                value['state'] = 'failed'
                return value
            if not current_binding:
                return value
            folder = current_binding['folder']
            value['stage'] = current_binding['progress']['stage']
            if current_binding['progress']['state'] == 'failed':
                value['state'] = 'failed'
                return value
        if is_memory:
            memory_score_registration(Path(_selected['plan']['original_worker']) if _selected else folder)
        training = read(run)
        value.update(training_step=training['optimizer_steps'], training_status=training['status'])
        expected_weights = (training.get('last_checkpoint') or {}).get('weights_sha256')
        if _selected:
            target_step = _selected['completed_step']
            expected_weights = _selected['weights_sha256']
            value.update(run_id=f"{run_id}-step{_selected['step']:06d}", training_run_id=run_id,
                         selected_checkpoint_step=_selected['step'], training_target_step=target_step)
        elif not is_current:
            folder = candidate_postcheck_folder(run_id, folder, training, expanded)
        protocol_path = folder / ('protocol.json' if scoped else 'candidate-evaluation-protocol.json')
        protocol = read(protocol_path)
        if not protocol:
            return value
        if is_current and current['state'] != 'complete':
            raise ValueError('current candidate evaluation started before verified training completion')
        if _selected and (protocol.get('checkpoint_selection') != 'saved_intermediate_from_completed_run'
                or protocol.get('checkpoint_step') != _selected['step']
                or protocol.get('training_completed_step') != target_step
                or protocol.get('training_final_weights_sha256') != training['last_checkpoint']['weights_sha256']
                or protocol.get('checkpoint_receipt_sha256') != _selected['receipt_source']['sha256']):
            raise ValueError('saved checkpoint protocol identity differs')
        if scoped and protocol.get('training_scope') != {
                'schema': 'microduck_navigation_training_scope_v1', 'kind': 'action_and_proprio'}:
            raise ValueError('action-scope candidate contract differs')
        if run_id == 'action-scope-corrective-pilot-v1' or run_id in CONTINUATION_STEPS or is_expanded or is_memory or is_current or _selected:
            contract_path = Path(protocol['expected_training_contract']).resolve()
            contract_path.relative_to(ROOT.resolve())
            if (identity(contract_path)['sha256'] != protocol['expected_training_contract_sha256']
                    or read(contract_path) != training.get('extended_navigation_training')):
                raise ValueError('corrective candidate data contract differs')
        memory = (current_correction_score_inputs(current_binding, protocol, training) if is_current else
                  memory_score_inputs(folder, training, protocol, _selected) if is_memory else None)
        output = Path(protocol['output']).resolve()
        output.relative_to(ROOT.resolve())
        model_path = output / 'report.json'
        model = read(model_path)
        if model is None:
            return value
        expected_cases = {(r['scene'], r['seed'], r['task']) for r in protocol['cases']}
        partial_cases = [(r['scene'], r['seed'], r['task']) for r in model.get('episodes', [])]
        if (model.get('weights_sha256') != protocol['weights_sha256']
                or protocol['weights_sha256'] != expected_weights
                or model.get('protocol_sha256') != identity(protocol_path)['sha256']
                or model.get('mode') != 'offline' or len(expected_cases) != 81
                or len(partial_cases) != len(set(partial_cases))
                or not set(partial_cases).issubset(expected_cases)):
            raise ValueError('partial progress belongs to a different or invalid candidate cohort')
        value.update(state='evaluating', completed=len(model.get('episodes', [])))
        if model['status'] == 'failed':
            value['state'] = 'failed'
            return value
        assessment_folder = memory['assessment_folder'] if memory else folder
        assessment_path = (assessment_folder / 'assessment.json' if scoped else
                           ROOT / 'reports/exports' / (run_id + '-development-assessment.json'))
        assessment = read(assessment_path)
        if not assessment or assessment['status'] != 'complete':
            value['state'] = 'replaying' if assessment and assessment['status'] == 'running' else 'awaiting_replay' if model['status'] == 'complete' else 'evaluating'
            if assessment:
                value['replayed'] = assessment.get('completed', 0)
                if assessment['status'] == 'failed':
                    value['state'] = 'failed'
            return value
        config_path = assessment_folder / ('assessment-config.json' if scoped else 'candidate-assessment-config.json')
        config = read(config_path)
        if config is None or identity(config_path)['sha256'] != assessment['config_sha256']:
            raise ValueError('assessment config mismatch')
        execution = read(assessment_folder / ('replay-launch.json' if scoped else 'candidate-evaluation-replay-launch.json'))
        matches = [c for c in containers if execution and c.get('Id') == execution['container_id']]
        if len(matches) != 1:
            raise ValueError('assessment execution missing')
        executed = matches[0]
        if executed['State']['Running']:
            value['state'] = 'replaying'
            return value
        if (executed['State']['Status'] != 'exited' or executed['State']['ExitCode'] != 0
                or executed.get('HostConfig', {}).get('DeviceRequests')):
            raise ValueError('independent CPU assessment failed')
        argv = executed.get('Config', {}).get('Cmd') or []
        if (argv.count('--config') != 1 or argv.count('--config-sha256') != 1
                or argv[argv.index('--config')+1] != str(config_path)
                or argv[argv.index('--config-sha256')+1] != assessment['config_sha256']):
            raise ValueError('assessment command differs')
        if memory and (executed.get('Image') != memory['image']
                or executed['State'].get('OOMKilled') or executed['State'].get('Error')
                or executed['Config'].get('User') != '1031:1031'
                or argv != ['python', '-u', memory['assessor'], '--config', str(config_path),
                            '--config-sha256', assessment['config_sha256']]
                or config.get('script_sha256') != memory['assessor_sha256']
                or config.get('replay_source') != memory['source']
                or (memory.get('replay_source_manifest') is not None
                    and config.get('replay_source_manifest') != memory['replay_source_manifest'])):
            raise ValueError('memory replay allocation or implementation differs')
        for name, digest in config['files'].items():
            path = Path(name).resolve()
            path.relative_to(ROOT.resolve())
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError('assessment input changed')
        if (str(protocol_path) != config['protocol'] or str(assessment_path) != config['output']
                or config['replay_script_sha256'] != (memory['replay_sha256'] if memory else
                    'a504c0a0436c6b89ba35b5216681c25dfcf0ba7545413c1345197a4cd599aa65')
                or model['protocol_sha256'] != identity(protocol_path)['sha256']
                or model['status'] != 'complete' or model['mode'] != 'offline'
                or assessment['model_report_sha256'] != identity(model_path)['sha256']
                or assessment['checkpoint_sha256'] != model['weights_sha256']
                or model['weights_sha256'] != protocol['weights_sha256']
                or model['weights_sha256'] != expected_weights
                or training['status'] != 'complete' or training['optimizer_steps'] != target_step
                or protocol['contracts'] != ['microduck_navigation_forward04_v1']):
            raise ValueError('candidate evidence mismatch')
        baseline = pinned_json(dict(path=config['baseline_assessment'],sha256=assessment['baseline_assessment_sha256']))
        key = lambda r: (r['scene'], r['seed'], r['task'])
        expected = {key(r) for r in protocol['cases']}
        old = {key(r):r for r in baseline['rows'] if r['contract']==protocol['contracts'][0]}
        rows = {key(r):r for r in assessment['rows']}
        episodes = {key(r):r for r in model['episodes']}
        comparisons = {key(r):r for r in model['comparison']}
        if (baseline['status'] != 'complete' or len(expected) != 81 or len(assessment['rows']) != 81
                or len(model['episodes']) != 81 or len(model['comparison']) != 81
                or expected != set(rows) or expected != set(old) or expected != set(episodes)
                or expected != set(comparisons)):
            raise ValueError('candidate cohort mismatch')
        for k,r in rows.items():
            if memory and (r.get('input_history') != memory['history']
                    or r.get('model_input_trace_complete') is not True
                    or any(type(r.get(field)) is not int or r[field] < 0
                           for field in ('exact_rgb_frames', 'exact_history_grids', 'model_input_trace_records'))):
                raise ValueError('memory observations were not independently reconstructed')
            if (type(r['strict_success']) is not bool or r['strict_success'] != episodes[k]['strict_success']
                    or r['input_files_sha256'] != episodes[k]['files']
                    or comparisons[k]['initial_sensor_state_exact'] is not True
                    or comparisons[k]['initial_camera_grid_exact'] is not True
                    or comparisons[k]['baseline_success'] != old[k]['strict_success']
                    or comparisons[k]['candidate_success'] != r['strict_success']
                    or not all(r.get(flag) is True for flag in ('final_evaluator_exact','submitted_actions_exact','buffer_execution_exact'))):
                raise ValueError('candidate replay mismatch')
        groups = []
        for scene, total, before in [('room_clear',21,19),('room_curriculum_one',60,25)]:
            keys = [k for k in expected if k[0] == scene]
            if len(keys)!=total or sum(old[k]['strict_success'] for k in keys)!=before:
                raise ValueError('comparison baseline mismatch')
            groups.append(dict(scene=scene,total=total,baseline_successes=before,
                candidate_successes=sum(rows[k]['strict_success'] for k in keys),
                gained=sum(rows[k]['strict_success'] and not old[k]['strict_success'] for k in keys),
                regressed=sum(old[k]['strict_success'] and not rows[k]['strict_success'] for k in keys)))
        if groups != assessment['groups']:
            raise ValueError('assessment totals mismatch')
        value.update(state='verified',groups=groups,checkpoint_sha256=model['weights_sha256'],
                     sources=dict(protocol=identity(protocol_path),model=identity(model_path),assessment=identity(assessment_path)))
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        value.update(state='invalid',groups=[])
    return value


def teacher_correction_counts(folder, plan, plan_digest):
    """Count reported teacher work; these counters never establish model ability."""
    report_path = folder / 'report.json'
    report = read(report_path)
    if report is None:
        return None
    protocol_path = folder / 'label-protocol.json'
    protocol = read(protocol_path)
    output = Path(plan['labels_output']).resolve()
    output.relative_to(ROOT.resolve())
    assessment_path = output / 'assessment.json'
    assessment = read(assessment_path)
    if assessment is None:
        return None
    if not all(isinstance(value, dict) for value in (report, protocol, assessment)):
        raise ValueError('teacher progress report or protocol is missing')
    expected = {(task, seed) for task in plan['tasks'] for seed in plan['seeds']}
    if (len(expected) != 60 or report.get('plan_sha256') != plan_digest
            or report.get('actual_student_cases') != len(expected)
            or report.get('protocol_sha256') != identity(protocol_path)['sha256']
            or assessment.get('protocol_sha256') != report['protocol_sha256']
            or protocol.get('student_checkpoint_sha256') != plan['checkpoint_sha256']
            or assessment.get('status') not in ('running', 'complete')):
        raise ValueError('teacher progress belongs to another plan or failed run')
    planned_sources = [(r['task'], r['seed']) for r in protocol['sources']]
    anchors = {(r['task'], r['seed'], tick) for r in protocol['sources'] for tick in r['anchors']}
    sources = [(r['task'], r['seed']) for r in assessment['sources']]
    branches = [(r['task'], r['seed'], r['source_tick']) for r in assessment['rows']]
    expected_branches = {key for key in anchors if key[:2] in set(sources)}
    if (len(planned_sources) != len(expected) or set(planned_sources) != expected
            or report.get('selected_anchors') != len(anchors)
            or len(sources) != len(set(sources)) or not set(sources).issubset(expected)
            or not all(r.get('source_outcome_exact') is True for r in assessment['sources'])
            or len(branches) != len(set(branches)) or set(branches) != expected_branches
            or not all(type(r.get('strict_success')) is bool for r in assessment['rows'])
            or (assessment['status'] == 'complete' and set(sources) != expected)):
        raise ValueError('teacher case or branch coverage differs')
    return dict(completed=len(sources), total=len(expected),
        branches_completed=len(branches), branches_total=len(anchors),
        teacher_successful_branches=sum(r['strict_success'] for r in assessment['rows']),
        progress_sources=dict(publication=identity(report_path), protocol=identity(protocol_path),
                              assessment=identity(assessment_path)))


def memory_clear_teacher_counts(plan, plan_digest):
    """Count current-policy teacher branches without claiming learned ability."""
    output = Path(plan['output']).resolve()
    output.relative_to(ROOT.resolve())
    report_path = output/'report.json'
    report = read(report_path)
    if report is None:
        return None
    config_path = Path(plan['collection_config']).resolve()
    config_path.relative_to(ROOT.resolve())
    if plan['bound_files'].get(str(config_path)) != identity(config_path)['sha256']:
        raise ValueError('clear teacher collection configuration changed')
    config = read(config_path)
    parent = Path(plan['anchor_pipeline']).resolve()
    parent.relative_to(ROOT.resolve())
    state, anchors = read(parent/'state.json'), read(parent/'report.json')
    if not all(isinstance(v, dict) for v in (config, state, anchors)):
        raise ValueError('clear teacher configuration or completed anchor evidence missing')
    anchor_digest = identity(parent/'report.json')['sha256']
    if (state.get('status') != 'complete' or state.get('report_sha256') != anchor_digest
            or report.get('anchor_report_sha256') != anchor_digest
            or report.get('plan_sha256') != plan_digest
            or report.get('checkpoint_sha256') != plan['checkpoint_sha256']
            or anchors.get('checkpoint_sha256') != plan['checkpoint_sha256']
            or config.get('checkpoint_sha256') != plan['checkpoint_sha256']
            or report.get('status') not in ('running','complete')
            or report.get('training_data_released') is not False or report.get('optimizer_steps') != 0):
        raise ValueError('clear teacher progress has a different model, scope or anchor report')
    expected = {(task, seed) for seed in config['seeds'] for task in config['tasks']}
    clear_cases = [r for r in anchors['cases'] if r['scene']=='room_clear']
    planned = {(r['task'],r['seed']):r for r in clear_cases}
    if (len(expected) != 60 or len(clear_cases) != 60 or set(planned) != expected
            or len(anchors['cases']) != 120
            or len({(r['scene'],r['task'],r['seed']) for r in anchors['cases']}) != 120):
        raise ValueError('clear teacher cohort is incomplete')
    branch_keys = {(r['task'],r['seed'],tick) for r in planned.values() for tick in r['anchors']}
    sources = [(r['task'],r['seed']) for r in report['sources']]
    rows = report['rows'];actual = [(r['task'],r['seed'],r['source_tick']) for r in rows]
    if (len(sources) != len(set(sources)) or not set(sources).issubset(expected)
            or report.get('selected_cases') != 60 or report.get('selected_anchors') != len(branch_keys)
            or len(actual) != len(set(actual))
            or set(actual) != {k for k in branch_keys if k[:2] in set(sources)}
            or (report['status']=='complete' and set(sources)!=expected)):
        raise ValueError('clear teacher processed cases or branches differ')
    for row in report['sources']:
        original = planned[row['task'],row['seed']]
        if (row.get('source_files') != original['files'] or row.get('anchors') != len(original['anchors'])
                or (original['anchors'] and row.get('source_outcome_exact') is not True)):
            raise ValueError('clear teacher source replay differs')
    if any(type(r.get('teacher_success')) is not bool or type(r.get('eligible_training_supervision')) is not bool
           or (r['eligible_training_supervision'] and (not r['teacher_success'] or r.get('public_frames',0)<2)) for r in rows):
        raise ValueError('clear teacher label eligibility differs')
    return dict(completed=len(sources),total=60,branches_completed=len(actual),branches_total=len(branch_keys),
        teacher_successful_branches=sum(r['teacher_success'] for r in rows),
        progress_sources=dict(assessment=identity(report_path),anchors=identity(parent/'report.json')))


def multi_scene_recording_counts(plan):
    """Count admitted recordings, never teacher or learned-policy successes."""
    cohorts = plan.get('cohorts', [])
    if (len(cohorts) != 2 or {c['name'] for c in cohorts} != {'clear', 'obstacle'}
            or plan.get('cases') != 120):
        raise ValueError('two-scene collection cohort differs')
    completed, total, sources, rows = 0, 0, {}, []
    for cohort in cohorts:
        path = Path(cohort['path']).resolve()
        path.relative_to(ROOT.resolve())
        digest = identity(path)['sha256']
        if cohort['sha256'] != digest or plan['bound_files'].get(str(path)) != digest:
            raise ValueError('collection configuration changed')
        config = read(path)
        expected = {(seed, task) for seed in config['seeds'] for task in config['tasks']}
        scene = 'room_clear' if cohort['name'] == 'clear' else 'room_curriculum_one'
        if (len(expected) != cohort['cases'] or len(expected) != 60
                or config['scene'] != scene or cohort['scene'] != scene
                or config['checkpoint_sha256'] != plan['checkpoint_sha256']
                or config['input_history'] != plan['input_history']):
            raise ValueError('collection model or scene differs')
        output = Path(config['output']).resolve()
        output.relative_to(ROOT.resolve())
        report_path = output/'report.json'
        report = read(report_path)
        count = 0
        if report:
            keys = [(r['seed'], r['task']) for r in report.get('rollouts', [])]
            if (report.get('checkpoint_sha256') != plan['checkpoint_sha256']
                    or report.get('input_history') != config['input_history']
                    or report.get('scene') != scene
                    or report.get('collection_mode') != 'student_record_only'
                    or len(keys) != len(set(keys)) or not set(keys).issubset(expected)
                    or (report.get('status') == 'complete' and set(keys) != expected)):
                raise ValueError('collection report belongs to another model or cohort')
            count = len(keys)
            sources[cohort['name']] = identity(report_path)
        rows.append(dict(scene=scene, completed=count, total=60,
                         complete=bool(report and report.get('status') == 'complete')))
        completed += count
        total += 60
    return dict(completed=completed, total=total, cohorts=rows,
                checkpoint_sha256=plan['checkpoint_sha256'], collection_sources=sources)


def realtime_correction_data_progress(containers):
    """Read the registered realtime repair, including rejected teacher branches."""
    base = ROOT / 'config/guarded-rehearsal-pilot-v1'
    control = read(ROOT / 'config/gpu-execution-control.json') or {}
    registered = control.get('conditional_pipelines', {})
    names = [name for name in registered
             if re.fullmatch(r'realtime-correction-label-repair-v[1-9][0-9]*', name)]
    if not names:
        return None
    try:
        name = max(names, key=lambda n: int(n.rsplit('-v', 1)[1]))
        folder = base / name
        plan_path = folder / 'plan.json'
        plan, state, handle = read(plan_path), read(folder / 'state.json'), read(folder / 'launch.json')
        digest = identity(plan_path)['sha256']
        if (registered[name].get('plan_sha256') != digest or handle.get('plan_sha256') != digest
                or registered[name].get('maximum_optimizer_steps') != 0
                or plan.get('maximum_optimizer_steps') != 0
                or plan.get('maximum_gpu_allocations') != 0
                or state.get('status') not in ('running', 'failed', 'cancelled', 'complete')):
            raise ValueError('realtime data worker registration differs')
        protocol_path = Path(plan['protocol']).resolve()
        protocol_path.relative_to(ROOT.resolve())
        protocol_digest = identity(protocol_path)['sha256']
        if not any(ref['path'] == str(protocol_path) and ref['sha256'] == protocol_digest
                   for ref in plan['repair_bound_files']):
            raise ValueError('realtime teacher protocol is not bound')
        protocol = read(protocol_path)
        sources = protocol['sources']
        cases = {(r['task'], r['seed']) for r in sources}
        anchors = {(r['task'], r['seed'], tick) for r in sources for tick in r['anchors']}
        if (len(cases) != len(sources) or len(anchors) != sum(len(r['anchors']) for r in sources)
                or not anchors or protocol.get('training_release_allowed') is not False):
            raise ValueError('realtime correction cohort differs')
        try:
            fields = Path('/proc', str(handle['pid']), 'stat').read_text().rsplit(')', 1)[1].split()
            worker_live = fields[0] not in ('Z', 'X') and fields[19] == str(handle['proc_start_ticks'])
        except (OSError, ValueError, IndexError):
            worker_live = False
        cid = state.get('container_id') or state.get('waiting_container')
        actual = next((c for c in containers if c.get('Id') == cid), None)
        if actual and (not actual['Name'].lstrip('/').startswith('microduck-cpu-' + name + '-')
                       or actual['Image'] != plan['image'] or actual['HostConfig'].get('DeviceRequests')):
            raise ValueError('realtime correction CPU allocation differs')
        container_live = bool(actual and actual['State']['Running'] and actual['State']['Pid']
                              and Path('/proc', str(actual['State']['Pid'])).exists())
        failed = state['status'] in ('failed', 'cancelled') or bool(actual
            and actual['State'].get('Status') == 'exited' and
            (actual['State'].get('ExitCode') != 0 or actual['State'].get('OOMKilled') or actual['State'].get('Error')))
        live = worker_live or container_live
        phase = ('repair_check' if state['stage'] == 'waiting_for_render_diagnostic' else
                 'preparation' if state['stage'] in ('prepare', 'ready_for_model_preflight') else 'teacher')
        value = dict(state='failed' if failed else 'complete' if state['status'] == 'complete'
                     else 'snapshot' if live else 'unavailable', phase=phase, stage=state['stage'],
            live=live, worker_live=worker_live, container_live=container_live, container_id=cid,
            waiting=str(state['stage']).startswith('waiting_'), optimizer_steps=0,
            formal_acceptance=False, model_promoted=False, completed=None, total=None,
            layout_count=len({r['seed'] for r in sources}), task_count=len({r['task'] for r in sources}),
            sources=dict(plan=identity(plan_path), worker=identity(folder/'state.json'),
                         protocol=identity(protocol_path)))
        root = Path(plan['repair_output']).resolve() / 'labels'
        root.relative_to(ROOT.resolve())
        assessment = read(root/'assessment.json')
        rows, done = (assessment.get('rows', []), assessment.get('sources', [])) if assessment else ([], [])
        if assessment and assessment.get('protocol_sha256') != protocol_digest:
            raise ValueError('teacher assessment belongs to another protocol')
        keys = [(r['task'], r['seed'], r['source_tick']) for r in rows]
        finished = [(r['task'], r['seed']) for r in done]
        expected_by_case = {(r['task'], r['seed']): {(r['task'], r['seed'], tick) for tick in r['anchors']}
                            for r in sources}
        if (len(keys) != len(set(keys)) or not set(keys).issubset(anchors)
                or len(finished) != len(set(finished)) or not set(finished).issubset(cases)
                or any(not expected_by_case[key].issubset(set(keys)) for key in finished)
                or any(type(r.get('strict_success')) is not bool for r in rows)
                or (assessment and assessment['status'] == 'complete' and
                    (set(keys) != anchors or set(finished) != cases))):
            raise ValueError('teacher progress omitted or duplicated selected outcomes')
        for row in rows:
            report_path = (root / row['path'] / 'report.json').resolve()
            report_path.relative_to(root)
            report = read(report_path)
            if (identity(report_path)['sha256'] != row['report_sha256']
                    or report.get('protocol_sha256') != protocol_digest
                    or (report['task'], report['seed'], report['source_tick']) !=
                       (row['task'], row['seed'], row['source_tick'])
                    or any(report['replay'][kind]['status'] != 'complete' for kind in ('teacher', 'collection'))):
                raise ValueError('teacher progress lacks its exact branch replay')
        if phase == 'teacher':
            # The labeler flushes each verified branch before its completed-source
            # list. Count a fully replayed case immediately; never infer progress
            # for a zero-anchor case without its explicit completion record.
            completed_cases = set(finished) | {key for key, selected in expected_by_case.items()
                                               if selected and selected.issubset(set(keys))}
            value.update(completed=len(completed_cases), total=len(cases), branches_completed=len(rows),
                         branches_total=len(anchors), teacher_successful_branches=sum(r['strict_success'] for r in rows))
        if assessment:
            value['sources']['assessment'] = identity(root/'assessment.json')
        return value
    except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError):
        return dict(state='invalid', live=False, optimizer_steps=0, formal_acceptance=False)


def correction_data_progress(containers):
    """Describe the registered data pipeline without treating it as model training."""
    realtime = realtime_correction_data_progress(containers)
    if realtime is not None:
        return realtime
    base = ROOT / 'config/guarded-rehearsal-pilot-v1'
    stages = [
        ('corrective-candidate-expanded-rollin-v1', 'plan.json', 'collection'),
        ('corrective-candidate-expanded-labels-v2', 'plan.json', 'teacher'),
        ('corrective-candidate-expanded-data-audit-v2', 'queue-plan.json', 'audit'),
        ('expanded-corrective-preparation-v1', 'queue-plan.json', 'preparation'),
    ]
    first = base / stages[0][0]
    if not (first / 'plan.json').exists():
        return None
    try:
        control = read(ROOT / 'config/gpu-execution-control.json') or {}
        registered = control.get('conditional_pipelines', {})
        # Follow the registered repair chain; a draft directory alone does not
        # replace the failed attempt or establish that a worker is running.
        if 'corrective-candidate-expanded-labels-v3' in registered:
            stages[1:] = [
                ('corrective-candidate-expanded-labels-v3', 'plan.json', 'teacher'),
                ('corrective-candidate-expanded-data-audit-v3', 'queue-plan.json', 'audit'),
                ('expanded-corrective-preparation-v2', 'queue-plan.json', 'preparation'),
            ]
            if 'corrective-candidate-expanded-data-audit-v4' in registered:
                stages[3:] = [
                    ('corrective-candidate-expanded-data-audit-v4', 'queue-plan.json', 'audit'),
                    ('expanded-corrective-preparation-v4', 'queue-plan.json', 'preparation'),
                ]
        if 'expanded-candidate-rollin-v1' in registered:
            stages.append(('expanded-candidate-rollin-v1', 'plan.json', 'collection'))
            if 'expanded-candidate-anchor-diagnostic-v2' in registered:
                stages.append(('expanded-candidate-anchor-diagnostic-v2', 'plan.json', 'anchors'))
                if 'expanded-candidate-labels-v1' in registered:
                    stages.append(('expanded-candidate-labels-v1', 'plan.json', 'teacher'))
                    if 'expanded-candidate-data-audit-v1' in registered:
                        stages.append(('expanded-candidate-data-audit-v1', 'queue-plan.json', 'audit'))
                        if 'expanded-candidate-recovery-commands-v1' in registered:
                            stages.append(('expanded-candidate-recovery-commands-v1', 'queue-plan.json', 'audit'))
        if 'memory-step1500-rollin-v1' in registered:
            stages.append(('memory-step1500-rollin-v1', 'plan.json', 'collection'))
            if 'memory-current-train-anchors-v1' in registered:
                stages.append(('memory-current-train-anchors-v1', 'plan.json', 'anchors'))
                if 'memory-current-clear-labels-v1' in registered:
                    stages.append(('memory-current-clear-labels-v1', 'plan.json', 'teacher'))
                if 'memory-current-clear-release-v1' in registered:
                    stages.append(('memory-current-clear-release-v1', 'plan.json', 'audit'))
                if 'memory-current-obstacle-labels-v1' in registered:
                    stages.append(('memory-current-obstacle-labels-v1', 'plan.json', 'teacher'))
                if 'memory-current-correction-data-audit-v1' in registered:
                    stages.append(('memory-current-correction-data-audit-v1', 'plan.json', 'audit'))
                preparation = current_worker_folder('memory-current-stage-preparation').name
                if preparation in registered:
                    stages.append((preparation, 'queue-plan.json', 'preparation'))
        if 'memory-current-three-color-rollin-v1' in registered:
            stages.append(('memory-current-three-color-rollin-v1', 'plan.json', 'collection'))
            if 'memory-current-three-color-anchors-v1' in registered:
                stages.append(('memory-current-three-color-anchors-v1', 'plan.json', 'anchors'))
                if 'memory-current-three-color-clear-labels-v1' in registered:
                    stages.append(('memory-current-three-color-clear-labels-v1', 'plan.json', 'teacher'))
                if 'memory-current-three-color-clear-release-v1' in registered:
                    stages.append(('memory-current-three-color-clear-release-v1', 'plan.json', 'audit'))
                if 'memory-current-three-color-obstacle-labels-v1' in registered:
                    stages.append(('memory-current-three-color-obstacle-labels-v1', 'plan.json', 'teacher'))
                if 'memory-current-three-color-data-audit-v1' in registered:
                    stages.append(('memory-current-three-color-data-audit-v1', 'plan.json', 'audit'))
                    preparation = current_worker_folder('memory-current-three-color-stage-preparation').name
                    if preparation in registered:
                        stages.append((preparation, 'queue-plan.json', 'preparation'))
        selected = None
        for name, plan_name, phase in stages:
            folder = base / name
            state, handle = read(folder / 'state.json'), read(folder / 'launch.json')
            if state is None or handle is None:
                break
            plan_path = folder / plan_name
            plan = read(plan_path)
            digest = identity(plan_path)['sha256']
            if (registered.get(name, {}).get('plan_sha256') != digest
                    or handle.get('plan_sha256') != digest
                    or registered[name].get('maximum_optimizer_steps') != 0):
                raise ValueError('data worker registration differs')
            if state.get('status') == 'complete':
                continue
            if state.get('status') not in ('running', 'failed', 'cancelled'):
                raise ValueError('unexpected data worker status')
            selected = (name, phase, folder, plan, state, handle)
            break
        if selected is None:
            return None
        name, phase, folder, plan, state, handle = selected
        try:
            fields = Path('/proc', str(handle['pid']), 'stat').read_text().rsplit(')', 1)[1].split()
            worker_live = fields[0] not in ('Z', 'X') and fields[19] == str(handle['proc_start_ticks'])
        except (OSError, ValueError, IndexError):
            worker_live = False
        cid = state.get('waiting_container') or state.get('container_id')
        actual = next((c for c in containers if c.get('Id') == cid), None)
        if actual:
            prefixes = ('microduck-fastwam-' + name, 'microduck-cpu-' + name)
            actual_name = actual['Name'].lstrip('/')
            if not any(actual_name == p or actual_name.startswith(p + '-') for p in prefixes):
                raise ValueError('data worker container differs')
        container_live = bool(actual and actual['State']['Running'] and actual['State']['Pid']
                              and Path('/proc', str(actual['State']['Pid'])).exists())
        live = worker_live or container_live
        terminal = state['status'] in ('failed', 'cancelled')
        value = dict(state='failed' if terminal else 'snapshot' if live else 'unavailable',
            phase=phase, stage=state.get('stage'), live=live, worker_live=worker_live,
            waiting=str(state.get('stage', '')).startswith('waiting_'),
            container_id=cid, container_live=container_live, completed=None, total=None,
            optimizer_steps=0, formal_acceptance=False, model_promoted=False,
            sources=dict(plan=identity(plan_path),
                         worker=identity(folder / 'state.json')))
        if phase == 'collection' and 'cohorts' in plan:
            counts = multi_scene_recording_counts(plan)
            value['sources']['collections'] = counts.pop('collection_sources')
            value.update(counts)
            if state.get('stage') == 'independent_training_recording_replay':
                if value['completed'] != value['total'] or not all(c['complete'] for c in value['cohorts']):
                    raise ValueError('replay began without both complete collections')
                value.update(phase='replay', recorded=value['completed'], completed=None, total=None)
        elif phase == 'collection':
            config_path = Path(plan['collection_config']).resolve()
            config_path.relative_to(ROOT.resolve())
            config = read(config_path)
            if plan['bound_files'].get(str(config_path)) != identity(config_path)['sha256']:
                raise ValueError('collection cohort changed')
            expected = {(seed, task) for seed in config['seeds'] for task in config['tasks']}
            if len(expected) != plan['cases'] or len(expected) != 60:
                raise ValueError('collection cohort differs')
            output = Path(plan['collection_output']).resolve()
            output.relative_to(ROOT.resolve())
            report_path = output / 'report.json'
            report = read(report_path)
            value.update(total=60, completed=0)
            if report:
                rows = report.get('rollouts', [])
                keys = [(r['seed'], r['task']) for r in rows]
                if (report.get('checkpoint_sha256') != plan['checkpoint_sha256']
                        or report.get('collection_mode') != 'student_record_only'
                        or len(keys) != len(set(keys)) or not set(keys).issubset(expected)):
                    raise ValueError('collection report belongs to another model or cohort')
                value.update(completed=len(keys), checkpoint_sha256=plan['checkpoint_sha256'])
                value['sources']['collection'] = identity(report_path)
            if state.get('stage') == 'independent_training_rollout_replay':
                if not report or report.get('status') != 'complete' or value['completed'] != len(expected):
                    raise ValueError('replay began without complete collection')
                value.update(phase='replay', completed=None, total=None, recorded=len(expected))
        elif phase == 'teacher' and value['state'] == 'snapshot' and not value['waiting']:
            counts = (memory_clear_teacher_counts(plan, digest)
                      if plan.get('schema') == 'memory_current_model_clear_teacher_continuations_v1'
                      else teacher_correction_counts(folder, plan, digest))
            if counts is not None:
                value['sources'].update(counts.pop('progress_sources'))
                value.update(counts)
        return value
    except (OSError, ValueError, TypeError, KeyError, IndexError):
        return dict(state='invalid', live=False, optimizer_steps=0, formal_acceptance=False)


def campaign_container_ids():
    """Include the registered CPU replay without labelling it as GPU inference."""
    ids = []
    for prefix in ('microduck-fastwam-', 'microduck-cpu-realtime-correction-',
                   'microduck-cpu-retention-action-postchecks-v1-',
                   'microduck-cpu-retention-strong-postchecks-v1-',
                   'microduck-cpu-retained-corrective-postchecks-v1-',
                   'microduck-cpu-corrective-candidate-expanded-rollin-v1-',
                   'microduck-cpu-expanded-candidate-rollin-v1-',
                   'microduck-cpu-expanded-candidate-anchor-diagnostic-v',
                   'microduck-cpu-expanded-candidate-labels-v',
                   'microduck-cpu-expanded-candidate-data-audit-v',
                   'microduck-cpu-expanded-candidate-recovery-commands-v',
                   'microduck-cpu-corrective-candidate-expanded-labels-v',
                   'microduck-cpu-corrective-candidate-expanded-data-audit-v',
                   'microduck-cpu-expanded-corrective-preparation-v',
                   'microduck-cpu-expanded-corrective-postchecks-v',
                   'microduck-cpu-expanded-candidate-postchecks-v',
                   'microduck-cpu-memory-candidate-postchecks-v1-',
                   'microduck-cpu-memory-candidate-replay-completion-v1-',
                   'microduck-cpu-memory-candidate-success-alignment-v2-',
                   'microduck-cpu-memory-axis-weighted-postchecks-v1-',
                   'microduck-cpu-memory-axis-control-postchecks-v1-',
                   'microduck-cpu-memory-continuation-postchecks-v1-',
                   'microduck-cpu-memory-continuation-2000-postchecks-v1-',
                   'microduck-cpu-memory-checkpoint-1500-postchecks-v1-',
                   'microduck-cpu-memory-step1500-correction-alignment-v1-',
                   'microduck-cpu-memory-step1500-rollin-v1-',
                   'microduck-cpu-memory-current-correction-alignment-v1-',
                   'microduck-cpu-memory-current-three-color-rollin-v1-',
                   'microduck-cpu-memory-current-three-color-anchors-v1',
                   'microduck-cpu-memory-current-three-color-clear-labels-v1',
                   'microduck-cpu-memory-current-three-color-clear-release-v1',
                   'microduck-cpu-memory-current-three-color-obstacle-labels-v1',
                   'microduck-cpu-memory-current-three-color-data-audit-v1',
                   'microduck-cpu-memory-current-three-color-stage-preparation-v',
                   'microduck-cpu-memory-current-three-color-model-preflight-v',
                   'microduck-cpu-memory-current-three-color-candidate-training-v',
                   'microduck-cpu-memory-current-three-color-candidate-postchecks-v',
                   'microduck-cpu-memory-current-three-color-candidate-alignment-v',
                   'microduck-cpu-memory-current-three-color-realtime-preparation-v',
                   'microduck-cpu-memory-current-three-color-candidate-realtime-v',
                   'microduck-cpu-memory-current-three-color-candidate-physical-replay-v',
                   'microduck-cpu-memory-current-train-anchors-v1',
                   'microduck-cpu-memory-current-clear-labels-v1',
                   'microduck-cpu-memory-current-clear-release-v1',
                   'microduck-cpu-memory-current-obstacle-labels-v1',
                   'microduck-cpu-memory-current-correction-data-audit-v1',
                   'microduck-cpu-memory-current-stage-preparation-v',
                   'microduck-cpu-memory-current-candidate-postchecks-v',
                   'microduck-cpu-memory-current-candidate-alignment-v',
                   'microduck-cpu-memory-current-candidate-realtime-v',
                   'microduck-cpu-memory-training-fit-execution-v',
                   'microduck-cpu-expanded-checkpoint-'):
        ids.extend(subprocess.check_output(
            ['docker', 'ps', '-aq', '--filter', 'name=' + prefix], text=True, timeout=10).split())
    return list(dict.fromkeys(ids))


def collect():
    now = time.time()
    # Backups/receipts also start with progress-. Their mtime is not authority.
    progress_path = ROOT / 'reports/progress-20261010.json'
    progress = read(progress_path) if progress_path else {}
    protocol_path = ROOT / 'config/clear-room-evaluation-protocol-v1.json'
    protocol = read(protocol_path)
    ids = campaign_container_ids()
    containers = json.loads(subprocess.check_output(['docker', 'inspect', *ids], text=True, timeout=15)) if ids else []
    active, completed = [], []
    other_models = {family: dict(active=[], completed_containers=[]) for family in ('vlm', 'hybrid', 'unclassified')}
    for c in containers:
        name = c['Name'].lstrip('/')
        if not name.startswith('microduck-fastwam-'):
            continue
        item = container_summary(c)
        family = item['model_family']
        if family == 'fastwam':
            (active if item['running'] else completed).append(item)
        else:
            other_models[family]['active' if item['running'] else 'completed_containers'].append(item)
    runs = []
    for path in (ROOT / 'runs').glob('*/report.json'):
        r = read(path)
        if not r or not isinstance(r.get('optimizer_steps'), int):
            continue
        config = path.parent / 'config.yaml'
        config_text = config.read_text() if config.exists() else ''
        match = re.search(r'^max_steps:\s*(\d+)\s*$', config_text, re.M)
        verify_resume_only = bool(re.search(r'^verify_resume_only:\s*true\s*$', config_text, re.M | re.I))
        points = []
        metrics = path.parent / 'metrics.jsonl'
        if metrics.exists():
            with metrics.open('rb') as stream:
                size = metrics.stat().st_size
                stream.seek(max(0, size - 2 * 1024 * 1024))
                if stream.tell():
                    stream.readline()
                for line in stream:
                    try:
                        point = json.loads(line)
                        if isinstance(point.get('step'), int) and 'train/loss' in point:
                            points.append({k: point[k] for k in ('step', 'time_unix', 'train/loss', 'train/loss_action', 'train/loss_video') if k in point})
                    except (ValueError, UnicodeError):
                        pass  # last line can be in flight
        runs.append({'id': path.parent.name, 'status': r.get('status'), 'stage': r.get('stage'),
                     'verify_resume_only': verify_resume_only,
                     'step': r['optimizer_steps'], 'max_steps': int(match[1]) if match else None,
                     'updated_unix': r.get('updated_unix'), 'started_unix': r.get('started_unix'),
                     'metrics': points, 'last_metrics': r.get('last_metrics', {}),
                     'checkpoint_sha256': (r.get('last_checkpoint') or {}).get('weights_sha256'),
                     'train_samples': r.get('train_samples'), 'source': identity(path)})
    evaluations, pending = [], []
    # Realtime jobs have their own progress; their scores never enter offline
    # checkpoint selection. Completed realtime comparisons require a separate
    # paired timing/expiry assessment, not this development-score validator.
    evaluation_paths = set((ROOT / 'reports').glob('*dev*/report.json'))
    # New explicit offline run names still use the same score validator;
    # discovery alone never promotes a different scene or cohort to acceptance.
    evaluation_paths.update((ROOT / 'reports').glob('*offline*/report.json'))
    evaluation_paths.update((ROOT / 'reports').glob('*realtime*/report.json'))
    for path in sorted(evaluation_paths):
        r = read(path)
        if not r:
            continue
        receipt, receipt_digest = history_receipt(r) if r.get('input_history') is not None else (None, None)
        score = validated_score(r, protocol, receipt=receipt, receipt_sha256=receipt_digest)
        if score:
            evaluations.append({'id': path.parent.name, 'checkpoint_sha256': r.get('checkpoint_sha256'),
                                'updated_unix': path.stat().st_mtime, 'input_history': r.get('input_history'), **score, 'source': identity(path)})
        elif r.get('status') == 'running' and isinstance(r.get('completed_episodes'), int):
            pending.append({'id': path.parent.name, 'mode': r.get('mode'), 'completed': r['completed_episodes'],
                            'total': evaluation_case_count(containers, path.parent.name)})
    runs.sort(key=lambda r: r.get('started_unix') or 0)
    evaluations.sort(key=lambda r: r['updated_unix'])
    ledger = read(ROOT / 'reports/gpu-ledger.json') or {}
    accounting = model_gpu_accounting(containers, ledger, now)
    hours = accounting['by_family']['fastwam']
    for family, rows in other_models.items():
        rows['gpu_hours'] = accounting['by_family'][family]
    teacher_path = ROOT / 'data/teacher-clear-qualification-v1/qualification_report.json'
    teacher = read(teacher_path) or {}
    guarded = guarded_candidate_development(containers)
    selection = model_selection(runs, guarded)
    memory_candidate = memory_candidate_progress(containers)
    if memory_candidate and memory_candidate.get('state') not in ('invalid', 'unavailable'):
        for item in active + completed:
            if item['id'] == memory_candidate.get('container_id'):
                item['name'] = 'microduck-fastwam-' + memory_candidate['run_id']
    active_trial = 'low_lr_history' if progress.get('low_lr_same_weight_realtime_execution') else 'retained'
    if progress.get('low_lr_prompt_cache_execution'):
        active_trial = 'low_lr_promptcache'
    comparisons = {trial: realtime_comparison(protocol, trial) for trial in REALTIME_TRIALS} if protocol else {}
    execution_control, closeout = campaign_closeout(progress)
    return {'schema': 'microduck_fastwam_dashboard_v1', 'observed_unix': now,
            'execution_control': execution_control, 'closeout': closeout,
            'authorized_retest': authorized_retest(containers),
            'no_age_retest': no_age_retest(containers),
            'guarded_candidate_development': guarded,
            'checkpoint_review': saved_checkpoint_review(containers),
            'memory_preflight': memory_training_preflight(containers),
            'memory_candidate': memory_candidate,
            'correction_data_progress': correction_data_progress(containers),
            'active': active, 'completed_containers': completed, 'runs': runs, 'evaluations': evaluations,
            'pending_evaluations': pending,
            'realtime_comparison': comparisons.get(active_trial),
            'realtime_comparisons': {k: v for k, v in comparisons.items() if v is not None},
            'repeated_realtime': repeated_realtime_comparison(protocol) if protocol else None,
            'expanded_realtime': expanded_realtime_comparison(),
            'active_realtime_trial': active_trial,
            'protocol': protocol, 'protocol_source': identity(protocol_path) if protocol else None,
            'retained_checkpoint': (selection.get('retained') or {}).get('checkpoint_sha256'),
            'model_selection': selection,
            'teacher': {'qualified': teacher.get('teacher_qualified'), 'tasks': teacher.get('tasks', {}),
                        'obstacles_qualified': teacher.get('obstacle_avoidance_qualified')},
            'teacher_development': teacher_development(progress, active + completed),
            'gpu_hours': hours, 'gpu_accounting': accounting, 'other_models': other_models,
            'gpu_accounting_unix': ledger.get('updated_unix'),
            'progress_source': identity(progress_path) if progress_path else None,
            'queue': current_queue(progress)}


if __name__ == '__main__':
    print(json.dumps(collect(), allow_nan=False))
