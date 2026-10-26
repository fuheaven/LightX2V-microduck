"""Bounded CPU camera workers rendering exact, immutable MuJoCo snapshots.

Only the control thread calls submit/poll/close. Each renderer and its OpenGL
context live on one worker thread. No worker reads the live physics data, runs
forward dynamics, or replaces a capture timestamp with its completion time.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, replace
import os
import queue
import threading
import time

import numpy as np

from .contracts import Observation


@dataclass(frozen=True)
class CameraFrame:
    observation: Observation
    receipt: dict
    error: str | None = None


class SnapshotCamera:
    def __init__(self, physics, episode_id, *, workers=4, startup_timeout_s=30.,
                 _renderer_factory=None):
        if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 8:
            raise ValueError("camera workers must be an integer in 1..8")
        if not episode_id or not np.isfinite(startup_timeout_s) or startup_timeout_s <= 0:
            raise ValueError("camera needs an episode and a positive startup timeout")
        if os.environ.get("MUJOCO_GL") != "osmesa":
            raise ValueError("snapshot camera currently requires explicit MUJOCO_GL=osmesa")
        self.physics, self.episode_id = physics, episode_id
        self._owner = threading.get_ident()
        self._closed = False
        self._stop = threading.Event()
        self._slots, self._threads = [], []
        self._free = queue.Queue(maxsize=workers)
        self._completed = queue.Queue(maxsize=workers)
        self._pending, self._ready = [], {}
        self._last_tick = -1
        self._last_capture_ns = -1
        self.receipts = []
        self.initialization = []
        self.cancelled_ticks = []
        factory = _renderer_factory or physics.mj.Renderer
        try:
            # Serial initialization also keeps cold OpenGL context creation
            # outside the measured control loop. Its frame is never delivered.
            for index in range(workers):
                model = copy.copy(physics.model)
                data = physics.mj.MjData(model)
                physics.mj.mj_copyData(data, model, physics.data)
                slot = dict(index=index, model=model, data=data,
                            requests=queue.Queue(maxsize=1), ready=threading.Event())
                self._slots.append(slot)
                thread = threading.Thread(target=self._run, args=(slot, factory),
                                          name=f"microduck-camera-{index}", daemon=True)
                self._threads.append(thread)
                thread.start()
                if not slot["ready"].wait(startup_timeout_s):
                    raise TimeoutError("snapshot renderer initialization did not finish")
                if "init_error" in slot:
                    raise RuntimeError(slot["init_error"])
                self.initialization.append(slot["initialization"])
                self._free.put_nowait(index)
        except BaseException:
            self.close()
            raise

    def _run(self, slot, factory):
        renderer = None
        try:
            start = time.monotonic_ns()
            renderer = factory(slot["model"], height=self.physics.image_size,
                               width=self.physics.image_size)
            renderer.update_scene(slot["data"], camera=self.physics.camera)
            renderer.render()  # Discarded initialization frame, never an observation.
            actual_samples = int(renderer._mjr_context.offSamples)
            expected_samples = int(slot["model"].vis.quality.offsamples)
            if actual_samples != expected_samples:
                raise ValueError("snapshot context changed the camera sample count")
            slot["initialization"] = dict(worker=slot["index"], started_ns=start,
                finished_ns=time.monotonic_ns(), observation_eligible=False,
                context_offsamples=actual_samples)
            slot["ready"].set()
            while not self._stop.is_set():
                try:
                    observed, receipt = slot["requests"].get(timeout=.05)
                except queue.Empty:
                    continue
                if self._stop.is_set():
                    break
                receipt["render_started_ns"] = time.monotonic_ns()
                error, completed = None, observed
                try:
                    # mj_copyData in submit preserved all derived camera/geom
                    # arrays too. mj_forward here would change the source image.
                    renderer.update_scene(slot["data"], camera=self.physics.camera)
                    rgb = renderer.render()
                    if self.physics.profile.name == "official_bam":
                        rgb = np.rot90(rgb, k=-1)
                    completed = replace(observed, rgb=rgb, camera_tick=observed.tick)
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                receipt["ready_ns"] = time.monotonic_ns()
                self._completed.put_nowait(CameraFrame(completed, receipt, error))
        except Exception as exc:
            slot["init_error"] = f"{type(exc).__name__}: {exc}"
        finally:
            slot["ready"].set()
            if renderer is not None:
                try:
                    renderer.close()
                except Exception as exc:
                    slot["close_error"] = f"{type(exc).__name__}: {exc}"

    def _check_owner(self):
        if threading.get_ident() != self._owner:
            raise RuntimeError("camera submission and delivery belong to the control thread")
        if self._closed:
            raise RuntimeError("snapshot camera is closed")

    def submit(self, observed):
        """Copy at the current control tick; return False instead of waiting."""
        self._check_owner()
        if (observed.episode_id != self.episode_id or observed.rgb is not None
                or observed.camera_tick is not None or observed.tick <= self._last_tick
                or observed.monotonic_ns <= self._last_capture_ns):
            raise ValueError("camera requires an ordered fresh sensor observation from its episode")
        self._last_tick, self._last_capture_ns = observed.tick, observed.monotonic_ns
        try:
            index = self._free.get_nowait()
        except queue.Empty:
            self.receipts.append(dict(source_tick=observed.tick,
                capture_ns=observed.monotonic_ns, status="dropped_busy"))
            return False
        slot = self._slots[index]
        before = time.monotonic_ns()
        self.physics.mj.mj_copyData(slot["data"], slot["model"], self.physics.data)
        receipt = dict(worker=index, source_tick=observed.tick,
            capture_ns=observed.monotonic_ns, copy_started_ns=before,
            copy_finished_ns=time.monotonic_ns())
        self._pending.append(observed.tick)
        slot["requests"].put_nowait((observed, receipt))
        return True

    def poll(self):
        """Deliver in capture order, including explicit failures; never wait."""
        self._check_owner()
        while True:
            try:
                frame = self._completed.get_nowait()
            except queue.Empty:
                break
            self._ready[frame.observation.tick] = frame
        result = []
        while self._pending and self._pending[0] in self._ready:
            frame = self._ready.pop(self._pending.pop(0))
            frame.receipt.update(delivered_ns=time.monotonic_ns(),
                                 status="error" if frame.error else "delivered")
            if frame.error:
                frame.receipt["error"] = frame.error
            self.receipts.append(frame.receipt)
            result.append(frame)
            self._free.put_nowait(frame.receipt["worker"])
        return result

    def close(self, timeout_s=1.):
        if threading.get_ident() != self._owner:
            raise RuntimeError("camera shutdown belongs to the control thread")
        self._closed = True
        self._stop.set()
        self.cancelled_ticks = list(self._pending)
        deadline = time.monotonic() + timeout_s
        for thread in self._threads:
            thread.join(max(0., deadline - time.monotonic()))
        return all(not thread.is_alive() for thread in self._threads)

    def metrics(self):
        return dict(schema="microduck_snapshot_camera_v1", backend="osmesa",
            workers=len(self._slots), source="mj_copyData_without_forward",
            ordering="capture_order", drop_policy="drop_new_when_all_slots_busy",
            capture_timestamp_preserved=True, initialization=self.initialization,
            delivered_frames=sum(r["status"] == "delivered" for r in self.receipts),
            failed_frames=sum(r["status"] == "error" for r in self.receipts),
            dropped_frames=sum(r["status"] == "dropped_busy" for r in self.receipts),
            cancelled_ticks=self.cancelled_ticks,
            workers_stopped=all(not thread.is_alive() for thread in self._threads),
            worker_errors=[s[k] for s in self._slots for k in ("init_error", "close_error") if k in s])
