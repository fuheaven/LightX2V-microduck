"""Own a navigation model in a spawned process, separately from GPU rendering.

Only RGB, proprioception, language and an optional sampling seed cross this
local pipe. Physical state, task truth and evaluator objects never enter it.
The existing policy and executor retain episode/tick/expiry ownership. This
module imports neither torch nor a renderer in the parent process.
"""
from __future__ import annotations

import hashlib
import multiprocessing
import threading
import traceback

import numpy as np


def _input_digest(images, state, task_description, seed):
    digest = hashlib.sha256()
    for name, value in [('head_camera', images['head_camera']), ('state', state)]:
        array = np.asarray(value)
        digest.update(repr((name, array.shape, array.dtype.str)).encode())
        digest.update(array.tobytes(order='C'))
    digest.update(repr((task_description, seed)).encode())
    return digest.hexdigest()


def _worker(connection, factory, kwargs):
    core = None
    try:
        core = factory(**kwargs)
        from .navigation_action_contracts import core_action_contract

        metadata = {key: getattr(core, key) for key in (
            'policy_profile', 'action_dim', 'robot_state_dim', 'camera_size', 'gripper_postprocess')}
        metadata['navigation_contract'] = core_action_contract(core)
        metadata['history_contract'] = (getattr(core, 'history_contract', None)
                                        or getattr(core, 'config', {}).get('navigation_history'))
        metadata['runtime_provenance'] = getattr(core, 'runtime_provenance', {})
        metadata['reference_provenance'] = getattr(core, 'reference_provenance', None)
        metadata['resolved_config'] = getattr(core, 'resolved_config', None)
        connection.send(('ready', 0, metadata))
        while True:
            operation, sequence, payload = connection.recv()
            try:
                if operation == 'close':
                    if hasattr(core, 'close'):
                        core.close()
                    core = None
                    connection.send(('result', sequence, None))
                    break
                if operation == 'reset':
                    core.reset()
                    result = None
                elif operation == 'predict':
                    before = _input_digest(**payload)
                    actions = core.predict_action_chunk(**payload)
                    if _input_digest(**payload) != before:
                        raise ValueError('model core mutated its received RGB/state/language inputs')
                    # Always return CPU arrays, never CUDA IPC tensors or a
                    # device handle whose lifecycle belongs to another process.
                    actions = np.asarray(actions, dtype=np.float32)
                    if (actions.ndim != 2 or actions.shape[1] != 5
                            or not 1 <= len(actions) <= 256 or not np.isfinite(actions).all()):
                        raise ValueError('model core returned invalid navigation actions')
                    result = dict(actions=actions, input_sha256=before)
                elif operation == 'runtime_metrics':
                    result = core.runtime_metrics() if hasattr(core, 'runtime_metrics') else {}
                else:
                    raise ValueError('unknown navigation model operation')
                connection.send(('result', sequence, result))
            except Exception:
                connection.send(('error', sequence, traceback.format_exc()))
    except (EOFError, BrokenPipeError):
        pass
    except BaseException:
        try:
            connection.send(('error', 0, traceback.format_exc()))
        except (EOFError, BrokenPipeError, OSError):
            pass
    finally:
        if core is not None and hasattr(core, 'close'):
            core.close()
        connection.close()


class ProcessNavigationCore:
    """Synchronous model API; use it inside the existing inference worker.

    Loading/inference may wait while this exact child is alive. Child failure
    raises instead of restarting or manufacturing an action. Intentional close
    can terminate this owned child after a bounded graceful shutdown.
    """

    def __init__(self, factory, **kwargs):
        context = multiprocessing.get_context('spawn')
        self._connection, child = context.Pipe()
        self._lock = threading.Lock()
        self._closed = False
        self._sequence = 0
        self._process = context.Process(target=_worker, args=(child, factory, kwargs),
                                        name='microduck-navigation-model')
        self.close_receipt = None
        try:
            self._process.start()
        except BaseException:
            self._closed = True
            self._connection.close()
            raise
        finally:
            child.close()
        try:
            kind, sequence, metadata = self._receive()
            if kind != 'ready' or sequence != 0:
                raise RuntimeError('navigation model initialization failed: ' + str(metadata))
            for key, value in metadata.items():
                setattr(self, key, value)
            if (self.policy_profile, self.action_dim, self.robot_state_dim) != ('microduck_navigation', 5, 48):
                raise ValueError('spawned model has incompatible navigation dimensions')
            if self.gripper_postprocess or type(self.camera_size) is not int or self.camera_size < 1:
                raise ValueError('spawned model camera/action semantics are invalid')
        except BaseException:
            self.close()
            raise

    @property
    def worker_pid(self):
        return self._process.pid

    def _receive(self):
        while not self._closed:
            # Drain a buffered error/result before checking process liveness.
            if self._connection.poll(.05):
                try:
                    return self._connection.recv()
                except EOFError as exc:
                    self._process.join(timeout=0)
                    raise RuntimeError(f'navigation model pipe closed (exit {self._process.exitcode})') from exc
            if not self._process.is_alive():
                self._process.join(timeout=0)
                raise RuntimeError(f'navigation model process exited ({self._process.exitcode})')
        raise RuntimeError('navigation model session is closed')

    def _exchange(self, operation, payload=None, *, wait_for_idle=True):
        if not self._lock.acquire(blocking=wait_for_idle):
            raise RuntimeError('navigation model still has an in-flight request')
        try:
            if self._closed:
                raise RuntimeError('navigation model session is closed')
            self._sequence += 1
            sequence = self._sequence
            try:
                self._connection.send((operation, sequence, payload))
                kind, response_sequence, value = self._receive()
            except (BrokenPipeError, EOFError, OSError) as exc:
                raise RuntimeError('navigation model transport failed') from exc
            if response_sequence != sequence:
                raise RuntimeError('navigation model response does not match the request')
            if kind == 'error':
                raise RuntimeError('navigation model failed:\n' + value)
            if kind != 'result':
                raise RuntimeError('unexpected navigation model response')
            return value
        finally:
            self._lock.release()

    def reset(self):
        self._exchange('reset')

    def predict_action_chunk(self, images, state, task_description, seed=None):
        if set(images) != {'head_camera'} or not isinstance(task_description, str) or not task_description:
            raise ValueError('navigation model requires only head_camera and a nonempty instruction')
        rgb, state = np.asarray(images['head_camera']), np.asarray(state)
        if rgb.dtype != np.uint8 or rgb.shape != (self.camera_size, self.camera_size, 3):
            raise ValueError('navigation model RGB does not match its camera contract')
        if state.dtype != np.float32 or state.shape != (48,) or not np.isfinite(state).all():
            raise ValueError('navigation model requires finite float32 proprio48')
        if seed is not None and type(seed) is not int:
            raise ValueError('sampling seed must be an integer or None')
        payload = dict(images={'head_camera': rgb.copy()}, state=state.copy(),
                       task_description=task_description, seed=seed)
        expected = _input_digest(**payload)
        result = self._exchange('predict', payload)
        if result['input_sha256'] != expected:
            raise RuntimeError('model process received different sensor inputs')
        return result['actions']

    def runtime_metrics(self):
        # Shutdown/reporting must not queue behind a hung inference. The
        # caller can record the missing metric and deliberately close instead.
        return self._exchange('runtime_metrics', wait_for_idle=False)

    def close(self, grace_seconds=5.):
        if self._closed:
            return self.close_receipt
        if grace_seconds < 0 or not np.isfinite(grace_seconds):
            raise ValueError('shutdown grace period must be finite and nonnegative')
        acquired = self._lock.acquire(timeout=grace_seconds)
        self._closed = True  # An in-flight receive must not survive cancellation.
        forced = False
        try:
            if acquired and self._process.is_alive():
                try:
                    self._sequence += 1
                    self._connection.send(('close', self._sequence, None))
                except (BrokenPipeError, EOFError, OSError):
                    pass
            self._process.join(timeout=grace_seconds)
            if self._process.is_alive():
                forced = True
                self._process.terminate()
                self._process.join(timeout=5.)
            if self._process.is_alive():
                raise RuntimeError('owned navigation model process did not stop after termination')
            self.close_receipt = dict(pid=self.worker_pid, exitcode=self._process.exitcode,
                                      forced_termination=forced, stopped=True)
            return self.close_receipt
        finally:
            self._connection.close()
            if acquired:
                self._lock.release()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
