"""Separate sensor-cadence input preparation from potentially slow inference."""


def request_methods(policy):
    prepare = getattr(policy, 'prepare_request', None)
    predict = getattr(policy, 'predict_prepared', None)
    if prepare is None and predict is None:
        return (lambda observation: observation), policy.predict
    if not callable(prepare) or not callable(predict):
        raise ValueError('navigation request preparation and prediction must be implemented together')
    return prepare, predict
