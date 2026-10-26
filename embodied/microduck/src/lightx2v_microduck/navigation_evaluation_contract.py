"""Resolve execution semantics from the trained receipt before loading a GPU."""
from .navigation_action_contracts import FORWARD04_NAVIGATION_CONTRACT, LEGACY_NAVIGATION_CONTRACT, action_contract
from .navigation_extended_qualification import require
from .navigation_extended_release import normalization_binding
from .navigation_extended_training import verify_extended_resume


def checkpoint_execution(receipt, *, weights_sha256, stats_sha256, requested=None):
    if receipt is None:
        require(requested in (None, LEGACY_NAVIGATION_CONTRACT),
                'extended evaluation requires its trained checkpoint receipt')
        return LEGACY_NAVIGATION_CONTRACT, None
    require(receipt.get('weights_sha256') == weights_sha256 and receipt.get('stats_sha256') == stats_sha256,
            'evaluation checkpoint or statistics differ from the trained receipt')
    contract = action_contract(receipt.get('navigation_contract', LEGACY_NAVIGATION_CONTRACT)).name
    require(requested is None or requested == contract,
            'requested execution contract differs from the trained checkpoint')
    if contract == LEGACY_NAVIGATION_CONTRACT:
        require(receipt.get('extended_navigation_training') is None and receipt.get('normalization_binding') is None,
                'legacy receipt cannot hide extended training or normalization')
        return contract, None
    training = receipt.get('extended_navigation_training')
    require(contract == FORWARD04_NAVIGATION_CONTRACT and isinstance(training, dict)
            and training.get('schema') in ('microduck_extended_navigation_training_v1',
                                           'microduck_rehearsal_navigation_training_v1',
                                           'microduck_memory_rehearsal_training_v1')
            and training.get('execution_contract') == contract
            and training.get('normalization_binding') == normalization_binding(stats_sha256)
            and type(receipt.get('step')) is int and receipt['step'] >= 1,
            'extended evaluation requires a completed trained execution/normalization binding')
    verify_extended_resume(receipt, training)
    return contract, dict(training['normalization_binding'])
