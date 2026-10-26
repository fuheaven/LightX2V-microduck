"""Validate recorded RGB inventory without inventing images for camera failures."""
from pathlib import Path

import numpy as np

from .navigation_outcomes import read, require, sha


def verify_camera_clock(folder, episode, ticks):
    """Allow missing RGB only when original camera/control receipts prove its loss.

    This does not waive the realtime camera-cadence gate. It lets auditors
    reconstruct a failed episode while preserving its missing observations.
    """
    folder = Path(folder)
    ticks = np.asarray(ticks)
    count = episode['evaluator']['tick']
    expected = np.arange(0, count, 5)
    # Offline inference can fail after capturing the next source observation,
    # before that tick executes. Keep and verify this terminal frame; it must
    # not be counted as an executed tick or silently dropped from the audit.
    terminal = bool(len(ticks) and ticks[-1] == count)
    terminal_evidence = None
    if terminal:
        error = episode.get('execution_error') or {}
        require(count > 0 and count % 5 == 0 and episode.get('mode') == 'offline'
                and error.get('phase') == 'execution' and error.get('tick') == count
                and isinstance(error.get('error_type'), str) and error['error_type']
                and episode.get('strict_model_success') is False,
                'terminal camera frame has no matching offline execution failure')
        failure = folder / 'execution_failure.json'
        terminal_evidence = sha(failure)
        receipt = read(failure)
        require(all(receipt.get(k) == v for k, v in error.items())
                and receipt.get('mode') == 'offline' and receipt.get('strict_model_success') is False
                and receipt.get('task') == episode['task'] and receipt.get('seed') == episode['seed']
                and receipt.get('prefix_report') == 'report.json',
                'terminal camera failure receipt differs')
        expected = np.append(expected, count)
    require(ticks.ndim == 1 and len(ticks) > 0 and ticks.dtype.kind in 'iu'
            and ticks[0] == 0 and np.all(np.diff(ticks) > 0)
            and np.all(ticks % 5 == 0) and np.all(ticks <= count if terminal else ticks < count),
            'recording has an invalid measured camera clock')
    if np.array_equal(ticks, expected):
        result = dict(complete=True, missing_ticks=[], declared_camera_loss_verified=False)
        if terminal:
            result.update(terminal_unexecuted_observation_tick=count,
                          terminal_failure_sha256=terminal_evidence)
        return result

    camera = episode.get('camera_delivery') or {}
    require(episode.get('mode') == 'realtime'
            and camera.get('schema') == 'microduck_snapshot_camera_v1'
            and camera.get('initial_synchronous_frame') is True
            and camera.get('complete_camera_cadence') is False
            and episode.get('realtime_timing_qualified') is False
            and episode.get('realtime_success') is False,
            'recording lacks the complete measured camera clock')
    names = ('control.json', 'predictions.json', 'camera_delivery.json')
    hashes = {name: sha(folder/name) for name in names}
    controls, predictions, receipts = [read(folder/name) for name in names]
    from .navigation_camera_audit import audit_camera_timing

    timing = audit_camera_timing(episode, controls, predictions, receipts)
    delivered = sorted([0] + [row['source_tick'] for row in receipts
                              if row['status'] == 'delivered'])
    require(np.array_equal(ticks, delivered),
            'recorded RGB does not match the delivered camera inventory')
    missing = sorted(set(expected.tolist()) - set(delivered))
    require(missing and not timing['complete_camera_cadence'],
            'missing RGB has no declared camera failure')
    require(all(sha(folder/name) == digest for name, digest in hashes.items()),
            'camera evidence changed during verification')
    return dict(complete=False, missing_ticks=missing,
                declared_camera_loss_verified=True, evidence_sha256=hashes,
                camera_timing=timing, verifier_sha256=sha(__file__))
