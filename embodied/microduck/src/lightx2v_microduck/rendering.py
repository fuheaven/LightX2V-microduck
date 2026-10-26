"""Explicit camera sampling configuration, separate from robot dynamics."""


def validate_camera_offsamples(value):
    if value is not None and (type(value) is not int or value not in (0, 4)):
        raise ValueError("camera_offsamples must be None (MJCF default), 0, or 4")
    return value


def validate_snapshot_camera_workers(value):
    if type(value) is not int or not 0 <= value <= 8:
        raise ValueError("snapshot_camera_workers must be an integer in 0..8")
    return value


def paired_camera_delivery(control, candidate, protocol):
    """Keep camera scheduling changes out of a model-optimization comparison.

    Historical reports without delivery metadata describe the synchronous
    camera. Failure/latency counters are outcomes, not configuration: differing
    counters must remain in the comparison rather than removing failed cases.
    """
    workers = validate_snapshot_camera_workers(protocol.get('snapshot_camera_workers', 0))
    configuration = dict(mode='snapshot_workers' if workers else 'synchronous', workers=workers)
    for report in (control, candidate):
        sources = protocol.get('camera_runtime_source_sha256')
        if sources is not None:
            actual = report.get('runtime_provenance', {}).get('source_sha256', {})
            if (not isinstance(sources, dict) or not sources
                    or any(actual.get(name) != digest for name, digest in sources.items())):
                raise ValueError('camera runtime source differs from the comparison protocol')
        if validate_snapshot_camera_workers(report.get('snapshot_camera_workers', 0)) != workers:
            raise ValueError("camera delivery differs from the comparison protocol")
        for episode in report['episodes']:
            delivery = episode.get('camera_delivery')
            if not workers:
                if delivery is not None:
                    raise ValueError("undeclared snapshot camera delivery in synchronous comparison")
                continue
            expected = dict(schema='microduck_snapshot_camera_v1', workers=workers,
                backend='osmesa', source='mj_copyData_without_forward', ordering='capture_order',
                drop_policy='drop_new_when_all_slots_busy', capture_timestamp_preserved=True)
            if (not isinstance(delivery, dict)
                    or any(delivery.get(k) != v for k, v in expected.items())
                    or type(delivery.get('workers')) is not int
                    or delivery.get('capture_timestamp_preserved') is not True):
                raise ValueError("snapshot camera delivery contract differs across comparison arms")
            camera = episode.get('camera_rendering', {})
            initialization = delivery.get('initialization')
            if (not isinstance(camera, dict) or camera.get('backend') != 'osmesa'
                    or type(camera.get('context_offsamples')) is not int
                    or not isinstance(initialization, list)
                    or len(initialization) != workers
                    or any(not isinstance(row, dict) or type(row.get('worker')) is not int
                           or type(row.get('context_offsamples')) is not int for row in initialization)
                    or {row.get('worker') for row in initialization} != set(range(workers))
                    or any(row.get('observation_eligible') is not False
                           or row.get('context_offsamples') != camera.get('context_offsamples')
                           for row in initialization)):
                raise ValueError("snapshot worker contexts differ from declared camera configuration")
            configuration.update({k: v for k, v in expected.items() if k != 'workers'})
    return configuration


def replay_camera_options(report, case):
    """Reject an undeclared camera change before reconstructing an episode.

    Old recordings without camera metadata retain their original MJCF defaults.
    An explicit override must be present in both the case and the recording.
    """
    expected = validate_camera_offsamples(case.get("camera_offsamples"))
    camera = report.get("camera_rendering")
    if camera is None:
        if expected is not None:
            raise ValueError("recording has no evidence for the requested camera override")
        return {}
    if not isinstance(camera, dict) or camera.get("schema") != "microduck_camera_rendering_v1":
        raise ValueError("unknown recorded camera configuration")
    requested = validate_camera_offsamples(camera.get("requested_offsamples"))
    if requested != expected:
        raise ValueError("recorded camera override differs from replay case")
    if camera.get("image_size") != case["image_size"]:
        raise ValueError("recorded camera resolution differs from replay case")
    for key in ("model_offsamples", "context_offsamples"):
        actual = camera.get(key)
        if type(actual) is not int or actual < 0 or (requested is not None and actual != requested):
            raise ValueError("recorded camera sampling does not match its declared configuration")
    return {} if expected is None else {"camera_offsamples": expected}


def replay_camera_delivery(report, case):
    """A replay must use the declared camera path, including snapshot contexts."""
    workers = validate_snapshot_camera_workers(case.get('snapshot_camera_workers', 0))
    envelope = dict(snapshot_camera_workers=workers, episodes=[report])
    return paired_camera_delivery(envelope, envelope, dict(snapshot_camera_workers=workers))


def paired_camera_configuration(control, candidate, protocol):
    """A speed comparison cannot quietly change the model's camera inputs."""
    paired_camera_delivery(control, candidate, protocol)
    expected = validate_camera_offsamples(protocol.get("camera_offsamples"))
    configurations = []
    for report in (control, candidate):
        if validate_camera_offsamples(report.get("camera_offsamples")) != expected:
            raise ValueError("comparison camera override differs from its protocol")
        for episode in report['episodes']:
            replay_camera_options(episode, dict(camera_offsamples=expected,
                                               image_size=protocol.get('image_size', 256)))
            configurations.append(episode.get('camera_rendering'))
    if not configurations or any(value != configurations[0] for value in configurations):
        raise ValueError("camera configuration differs across comparison episodes or backends")
    return configurations[0]
