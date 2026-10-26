"""Checkpoint/data bindings for the optional frozen-parent training objective."""
import hashlib
import json
from pathlib import Path

from .navigation_retention_data import RetainedTrainingView
from .navigation_retention_loss import retention_manifest


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _digest(value):
    if not isinstance(value, str) or len(value)!=64 or set(value)-set('0123456789abcdef'):
        raise ValueError('invalid retention data digest')
    return value


def training_retention_manifest(config, *, objective, scope):
    if config is None:
        return {'kind':'disabled'}
    if objective.get('kind') == 'action_axis':
        from .navigation_axis_loss import axis_objective_manifest

        expected = axis_objective_manifest({k: objective.get(k) for k in ('axis_order', 'weights')})
        if objective != expected:
            raise ValueError('unverified axis supervision manifest')
    elif objective != {'kind':'author_uniform'}:
        raise ValueError('retention currently requires uniform tokens and the action/state scope')
    if scope.get('kind') != 'action_and_proprio':
        raise ValueError('retention currently requires the action/state scope')
    if (not isinstance(config,dict) or set(config) not in (
            {'teacher','weight','data'}, {'teacher','weight','data','input_pair'})):
        raise ValueError('retention requires explicit teacher, weight and data bindings')
    teacher=config['teacher'];data=config['data']
    if (not isinstance(teacher,dict) or set(teacher)!={'path','sha256','parent_receipt','receipt_sha256'}
            or not all(isinstance(teacher[k],str) and teacher[k] for k in ('path','parent_receipt'))):
        raise ValueError('retention teacher needs checkpoint and receipt paths')
    manifest=retention_manifest(dict(teacher_weights_sha256=teacher['sha256'],
        teacher_receipt_sha256=teacher['receipt_sha256'],weight=config['weight'],
        apply_to='retained_training_items'))
    if objective['kind'] == 'action_axis':
        manifest.update(supervised_objective='action_axis',
                        supervised_axis_objective=json.loads(json.dumps(objective)))
    if not isinstance(data,dict) or set(data)!={'retained_recipe_sha256','train','validation'}:
        raise ValueError('retention requires explicit train and validation membership')
    _digest(data['retained_recipe_sha256'])
    for split in ('train','validation'):
        item=data[split]
        if (not isinstance(item,dict) or set(item)!={'total_samples','retained_prefix_samples',
                'retained_sampling_sha256','complete_sampling_sha256'}
                or type(item['total_samples']) is not int or type(item['retained_prefix_samples']) is not int
                or not 0<item['retained_prefix_samples']<item['total_samples']):
            raise ValueError('invalid retention sampling membership')
        _digest(item['retained_sampling_sha256']);_digest(item['complete_sampling_sha256'])
    manifest.update(data=json.loads(json.dumps(data)),
        data_source_sha256=_sha(Path(__file__).with_name('navigation_retention_data.py')))
    if 'input_pair' in config:
        from .navigation_history import HistoryGridSpec
        from .wam_memory_rehearsal_data import memory_rehearsal_spec

        pair = config['input_pair']
        if (not isinstance(pair, dict) or set(pair) != {'teacher_history', 'student_history'}
                or pair['teacher_history'] != HistoryGridSpec((300, 150, 50, 0)).as_dict()):
            raise ValueError('paired retention requires the original fixed teacher history')
        memory_rehearsal_spec(pair['student_history'])
        manifest['input_pair'] = dict(schema='microduck_memory_retention_pair_v1',
            **json.loads(json.dumps(pair)),
            dataset_source_sha256=_sha(Path(__file__).with_name('wam_memory_rehearsal_data.py')))
    return manifest


def verify_resume_retention(receipt, manifest):
    if receipt.get('retention',{'kind':'disabled'})!=manifest:
        raise ValueError('resume retention objective or membership changed; initialize a new stage')


def verify_teacher_receipt(config, manifest):
    if manifest['kind']=='disabled':
        return
    teacher=config['teacher']
    if _sha(teacher['parent_receipt'])!=manifest['teacher_receipt_sha256']:
        raise ValueError('retention teacher receipt changed')
    receipt=json.loads(Path(teacher['parent_receipt']).read_text())
    if receipt.get('weights_sha256')!=manifest['teacher_weights_sha256']:
        raise ValueError('retention teacher receipt is for different weights')
    if manifest.get('input_pair') is not None:
        if receipt.get('input_history') != manifest['input_pair']['teacher_history']:
            raise ValueError('retention teacher was not trained for the paired original input')
    # Full checkpoint and normalization hashing occurs in initialize_from_weights.


def verify_retention_history(config, manifest, student_history):
    """Prevent input adaptation from silently changing the frozen parent's view."""
    if manifest['kind'] == 'disabled':
        return
    verify_teacher_receipt(config, manifest)
    teacher = json.loads(Path(config['teacher']['parent_receipt']).read_text())
    pair = manifest.get('input_pair')
    expected = teacher.get('input_history') if pair is None else pair['student_history']
    if student_history != expected:
        raise ValueError('retention input history differs; declare a matching paired-input stage')


def wrap_retention_datasets(train, validation, manifest):
    if manifest['kind']=='disabled':
        return train,validation
    wrapped=[]
    for split,dataset in (('train',train),('validation',validation)):
        pair = manifest.get('input_pair')
        if pair is not None:
            if (getattr(dataset, 'history_contract', None) != pair['student_history']
                    or getattr(dataset, 'retention_history_contract', None) != pair['teacher_history']):
                raise ValueError('paired retention dataset views differ from their bound contracts')
        elif getattr(dataset, 'retention_history_contract', None) is not None:
            raise ValueError('paired visual data requires an explicitly paired retention objective')
        expected=manifest['data'][split]
        view=RetainedTrainingView(dataset,split=split,
            retained_recipe_sha256=manifest['data']['retained_recipe_sha256'],
            expected_samples=expected['total_samples'],
            expected_retained_samples=expected['retained_prefix_samples'])
        if any(view.retention_binding[key]!=value for key,value in expected.items()):
            raise ValueError('retention sample identities or weighting changed')
        wrapped.append(view)
    return tuple(wrapped)
