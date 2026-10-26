"""Explicit execution bounds for recordings from an unchanged older model.

This sidecar does not rewrite its training receipt or statistics. It permits a
separately declared executor only for student-state recordings; supervision and
training publication still require their existing replay and admission checks.
"""


def recording_execution_binding(document, receipt, stats, *, receipt_sha256,
                                checkpoint_sha256, stats_sha256, contract, record_only):
    fields = {'schema','checkpoint_sha256','checkpoint_receipt_sha256','stats_sha256',
              'execution_contract','coordinate_contract','purpose','training_contract_changed'}
    if not isinstance(document, dict) or set(document) != fields:
        raise ValueError('recording execution binding has unexpected fields')
    if (document['schema'] != 'microduck_recording_execution_binding_v1'
            or document['purpose'] != 'student_record_only'
            or document['training_contract_changed'] is not False or record_only is not True):
        raise ValueError('execution binding is only for separately admitted student recordings')
    expected = {'checkpoint_sha256':checkpoint_sha256,
                'checkpoint_receipt_sha256':receipt_sha256,'stats_sha256':stats_sha256}
    for key, digest in expected.items():
        if (not isinstance(digest,str) or len(digest) != 64
                or set(digest)-set('0123456789abcdef') or document[key] != digest):
            raise ValueError('recording binding differs from its immutable artifact: '+key)
    if (not isinstance(receipt,dict) or receipt.get('weights_sha256') != checkpoint_sha256
            or receipt.get('stats_sha256') != stats_sha256):
        raise ValueError('original checkpoint receipt differs from the recording')
    if (document['execution_contract'] != contract
            or document['coordinate_contract'] != stats.get('schema')):
        raise ValueError('recording execution or normalization coordinates differ')
    binding = dict(execution_contract=contract,coordinate_contract=stats['schema'],stats_sha256=stats_sha256)
    existing = receipt.get('normalization_binding')
    if existing is not None and existing != binding:
        raise ValueError('recording sidecar cannot override an existing checkpoint binding')
    from .navigation_action_contracts import verify_artifact_action_contract
    verify_artifact_action_contract(
        dict(contract=contract,embodiment='legs',normalization_binding=binding,
             files={'dataset_stats.json':stats_sha256}),
        dict(navigation_contract=contract,normalization_binding=binding),stats)
    return binding
