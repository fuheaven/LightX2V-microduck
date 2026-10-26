"""Optional file cancellation polling outside the measured control thread."""
from __future__ import annotations

import math
from pathlib import Path
import threading
import time


def validate_cancel_polling(interval_ms, *, realtime):
    if interval_ms is None:
        return None
    if (isinstance(interval_ms, bool) or not isinstance(interval_ms, (int, float))
            or not math.isfinite(interval_ms) or not 0 < interval_ms <= 1000 or not realtime):
        raise ValueError('async cancellation polling requires realtime mode and 0 < interval_ms <= 1000')
    return float(interval_ms)


class FileCancellationMonitor:
    """The control thread reads only an Event; a slow stat cannot pause physics.

    This is not an emergency-stop transport. Detecting a request still depends
    on the filesystem call completing. An initial check precedes measurement,
    and an unfinished worker or an I/O error fails this runtime's health gate.
    """
    def __init__(self, path, *, interval_ms):
        self.path = Path(path)
        self.interval_ms = validate_cancel_polling(interval_ms, realtime=True)
        if self.interval_ms is None:
            raise ValueError('an explicit polling interval is required')
        self.requested = threading.Event()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread = None
        self._started = False
        self._checks = 0
        self._maximum_check_ms = 0.
        self._error = None
        self._detected_ns = None

    def _check(self):
        started = time.monotonic_ns()
        found, error = False, None
        try:
            found = self.path.exists()
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
        finished = time.monotonic_ns()
        with self._lock:
            self._checks += 1
            self._maximum_check_ms = max(self._maximum_check_ms, (finished-started)/1e6)
            if not self._stop.is_set():
                if error is not None:
                    self._error = error
                if found or error is not None:
                    self._detected_ns = finished
                    self.requested.set()

    def _run(self):
        while not self._stop.wait(self.interval_ms/1000):
            self._check()
            if self.requested.is_set():
                break

    def start(self):
        if self._started or self._stop.is_set():
            raise RuntimeError('cancellation monitor cannot be restarted')
        self._started = True
        self._check()  # Honor an existing request before the measured task starts.
        if not self.requested.is_set():
            self._thread = threading.Thread(target=self._run, name='microduck-file-cancel', daemon=True)
            self._thread.start()

    def close(self, timeout_s=.25):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(0., timeout_s))
        return self._thread is None or not self._thread.is_alive()

    def metrics(self):
        with self._lock:
            return dict(schema='microduck_async_file_cancellation_v1', interval_ms=self.interval_ms,
                initial_check_outside_measurement=True, control_loop_filesystem_calls=False,
                file_checks=self._checks, maximum_check_ms=self._maximum_check_ms,
                cancellation_requested=self.requested.is_set(), detected_monotonic_ns=self._detected_ns,
                error=self._error, worker_stopped=self._thread is None or not self._thread.is_alive(),
                detection_latency_includes_filesystem=True)
