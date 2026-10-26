"""Opt-in BAM write/integration evidence, not a handoff controller.

The existing body target selection and fixed-rate physics remain unchanged.
Neither these receipts nor a row-four snapshot prove a complete five-row prefix,
deadline compliance, TinyVLA task success, or hardware execution.
"""
from __future__ import annotations

from collections import deque
import copy
import hashlib
import threading
import time

import numpy as np

from .contracts import pack_observation
from .daemon import BamBodyServer


RENDER_PREPARATION_PROTOCOL = "snapshot_renderer_preparation_v1"


def _uint(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _execution(value):
    """Copy optional transport attribution; never infer it from target values."""
    if value is None:
        return None
    if not isinstance(value, dict) or type(value.get("protocol")) is not int or value["protocol"] != 1:
        raise ValueError("execution requires protocol 1")
    result = copy.deepcopy(value)
    for key in ("daemon_tick", "selected_monotonic_ns"):
        _uint(result.get(key), key)
    for key in ("episode_id", "execution_mode", "execution_source"):
        if not isinstance(result.get(key), str) or not result[key]:
            raise ValueError(f"execution requires {key}")
    for key in ("source_tick", "action_index"):
        if result.get(key) is not None:
            _uint(result[key], key)
    raw = result.get("raw_action14")
    if raw is not None:
        raw = np.asarray(raw, np.float32)
        if raw.shape != (14,) or not np.isfinite(raw).all():
            raise ValueError("execution raw_action14 must be finite 14D")
        result["raw_action14"] = raw.tolist()
    if result["execution_source"] == "external" and any(result.get(key) is None for key in ("source_tick", "action_index", "raw_action14")):
        raise ValueError("external execution requires source_tick, action_index and raw_action14")
    if result.get("source_capture") is not None and not isinstance(result["source_capture"], dict):
        raise ValueError("execution source_capture must be an object or null")
    return result


class IntegratedBamBodyServer(BamBodyServer):
    """An isolated, instrumented body using the parent's RemoteIo operations.

    ``cameras`` retains the parent's latest-frame stream. ``render_snapshots``
    independently enables a renderer for immutable post-integration copies.
    Read evidence with ``execution_evidence`` and drain snapshots with
    ``post_action_snapshot(after_sequence=..., require_rgb=...)``. Both return
    detached copies and never advance physics or change an action/history.
    """

    def __init__(self, env, host="127.0.0.1", port=0, cameras=True, *, render_snapshots=False,
                 evidence_capacity=8192, snapshot_capacity=32):
        if type(evidence_capacity) is not int or evidence_capacity < 1 or type(snapshot_capacity) is not int or snapshot_capacity < 1:
            raise ValueError("evidence capacities must be positive integers")
        super().__init__(env, host=host, port=port, cameras=cameras)
        self.render_snapshots = bool(render_snapshots)
        self._events = deque(maxlen=evidence_capacity)
        self._event_sequence = self._write_sequence = self._snapshot_sequence = 0
        self._current_write = None
        self._highest_io_write_sequence = None
        self._snapshots = deque(maxlen=snapshot_capacity)
        self._render_queue = deque(maxlen=snapshot_capacity)
        self._render_event = threading.Event()
        self._render_dropped = 0
        self._render_error = None
        self._snapshot_capture_error = None
        self._render_prepare_data = None
        self._render_preparation = {"protocol": RENDER_PREPARATION_PROTOCOL,
                                    "status": "not_requested" if self.render_snapshots else "disabled",
                                    "observation_eligible": False}
        if self.render_snapshots:
            self.threads.append(threading.Thread(target=self._snapshot_render_loop, name="bam-integrated-snapshot-render", daemon=True))

    def _record(self, kind, **fields):
        self._event_sequence += 1
        row = {"protocol": 1, "sequence": self._event_sequence, "kind": kind, **copy.deepcopy(fields)}
        self._events.append(row)
        return row

    def dispatch(self, request):
        with self.lock:
            op = request.get("op")
            if op == "prepare_snapshot_renderer":
                if request.get("protocol") != RENDER_PREPARATION_PROTOCOL:
                    raise ValueError("explicit snapshot renderer preparation protocol required")
                return self.prepare_snapshot_renderer()
            if op == "write":
                # Validate attribution before the original target write. Missing
                # attribution replaces it with None, including Safety holds.
                execution = _execution(request.get("execution"))
                io_sequence = request.get("io_write_sequence")
                if io_sequence is not None:
                    _uint(io_sequence, "io_write_sequence")
                targets = np.asarray(request["targets"], np.float64).copy()
                reply = super().dispatch(request)
                received = time.monotonic_ns()
                self._write_sequence += 1
                replayed_sequence = io_sequence is not None and self._highest_io_write_sequence is not None and io_sequence <= self._highest_io_write_sequence
                if io_sequence is not None:
                    self._highest_io_write_sequence = max(io_sequence, self._highest_io_write_sequence or 0)
                self._current_write = {"write_sequence": self._write_sequence, "io_write_sequence": io_sequence,
                    "execution": execution, "targets15": targets.tolist(), "received_monotonic_ns": received,
                    "io_sequence_replayed": replayed_sequence, "write_application_count": 0}
                self._record("write", **self._current_write, body_tick=self.steps, task_tick=self.env.tick,
                             task_epoch=self.task_epoch, episode_id=self.env.episode_id)
                return reply
            reply = super().dispatch(request)
            if op == "torque":
                # Enabling resets targets to current qpos in the parent;
                # disabling does not physically apply the buffered targets.
                self._current_write = None
            return reply

    def prepare_snapshot_renderer(self):
        """Queue one warmup on the actual render thread, without a policy frame.

        This nonblocking, idempotent preparation neither starts an owner nor
        advances/resets physics. The frozen warmup image is discarded. Context
        construction and rendering stay on the thread used by later snapshots.
        """
        with self.lock:
            if self.stop_event.is_set():
                raise ValueError("cannot prepare a closed body")
            if self._render_preparation["status"] != "not_requested":
                return copy.deepcopy(self._render_preparation)
            current_source = ((self._current_write or {}).get("execution") or {}).get("execution_source")
            if self._snapshot_sequence or current_source == "external":
                raise ValueError("prepare renderer before external execution")
            physics = self.env.physics
            data = physics.mj.MjData(physics.model)
            physics.mj.mj_copyData(data, physics.model, physics.data)
            self._render_prepare_data = data
            self._render_preparation.update(status="pending", requested_monotonic_ns=time.monotonic_ns(),
                                            body_tick=self.steps, task_tick=self.env.tick,
                                            episode_id=self.env.episode_id, task_epoch=self.task_epoch)
            self._render_event.set()
            return copy.deepcopy(self._render_preparation)

    def _integrate_once(self):
        """One original body-loop iteration plus atomic, passive evidence."""
        with self.lock:
            physics = self.env.physics
            started = time.monotonic_ns()
            write = self._current_write if self.released and self.torque_on else None
            targets = self.targets.copy()
            if not self.released:
                physics.restore_state(self.initial)
                mode = "held_before_enable"
            elif self.torque_on:
                physics.step(targets)
                mode = "targets"
            else:
                physics.data.ctrl[physics.actuator_indices] = 0.0
                for _ in range(4):
                    physics.mj.mj_step(physics.model, physics.data)
                mode = "torque_off"
            ended = time.monotonic_ns()
            self.steps += 1
            self.step_times_ns.append(ended)
            if self.released:
                self.env.tick += 1
                self.env._rgb_tick = -1
                self._update_command()
                self.env._record_metrics()
                self.env._update_task()
            if write is not None:
                write["write_application_count"] += 1
            attribution = copy.deepcopy(write) if write is not None else {
                "write_sequence": None, "io_write_sequence": None, "execution": None,
                "targets15": None, "received_monotonic_ns": None, "io_sequence_replayed": False,
                "write_application_count": 0}
            row = self._record("integrate", **attribution, body_tick=self.steps, task_tick=self.env.tick,
                task_epoch=self.task_epoch, episode_id=self.env.episode_id, physics_mode=mode,
                integration_start_monotonic_ns=started, integration_end_monotonic_ns=ended,
                sim_time_s=float(physics.data.time), applied_policy_targets14=targets.tolist() if mode == "targets" else None,
                repeated_write=bool(write is not None and write["write_application_count"] > 1),
                external_attribution=bool(write is not None and (write["execution"] or {}).get("execution_source") == "external"))
            execution = attribution["execution"]
            if (execution is not None and execution["execution_source"] == "external" and execution["action_index"] == 4
                    and attribution["write_application_count"] == 1 and not attribution["io_sequence_replayed"]):
                try:
                    self._capture_row_four(row)
                except Exception as exc:
                    # An incomplete evidence copy is a diagnostic failure,
                    # never a reason to stop or roll back real body integration.
                    self._snapshot_capture_error = repr(exc)
            return copy.deepcopy(row)

    def _physics_loop(self):
        # Deliberately mirrors BamBodyServer._physics_loop's physical cadence,
        # late-tick accounting and severe-overrun handling. No model rendezvous.
        deadline = time.monotonic()
        try:
            while not self.stop_event.is_set():
                self._integrate_once()
                deadline += 0.02
                remaining = deadline - time.monotonic()
                if remaining < -0.02:
                    self.late_ticks += 1
                if remaining < -0.25:
                    deadline = time.monotonic()
                self.stop_event.wait(max(0.0, remaining))
        except Exception as exc:
            self.fault = repr(exc)
            self.stop_event.set()

    def _capture_row_four(self, row):
        physics = self.env.physics
        captured = time.monotonic_ns()
        data = physics.mj.MjData(physics.model)
        physics.mj.mj_copyData(data, physics.model, physics.data)
        gyro, gravity, positions, velocities = physics.sensors()
        command = self._command.copy()
        command_source = "body_command"
        # This curriculum overrides observe(), but the parent body does not use
        # that method. Resolve its pure clock program without sensor side effects.
        if self.env.task.suite == "foundation_walk":
            from .foundation_walk import command_context
            command = command_context(self.env.tick, warming_up=self.env._walk_warming_up)[1]
            command_source = "foundation_walk.command_context"
        execution = row["execution"]
        state = pack_observation(gyro, gravity, positions, velocities, execution["raw_action14"], command)
        self._snapshot_sequence += 1
        snapshot = {"protocol": 1, "snapshot_sequence": self._snapshot_sequence, "capture_phase": "post_integration",
            "capture_monotonic_ns": captured, "render_ready_monotonic_ns": None,
            "render_status": "failed" if self._render_error else "pending" if self.render_snapshots else "disabled", "rgb": None,
            "body_tick": self.steps, "task_tick": self.env.tick, "task_epoch": self.task_epoch,
            "episode_id": self.env.episode_id, "sim_time_s": float(data.time),
            "integration_sequence": row["sequence"], "integration_end_monotonic_ns": row["integration_end_monotonic_ns"],
            "write_sequence": row["write_sequence"], "io_write_sequence": row["io_write_sequence"],
            "execution": copy.deepcopy(execution), "execution_episode_matches_body": execution["episode_id"] == self.env.episode_id,
            "state61": state.tolist(), "command13": command.tolist(), "command13_source": command_source,
            "sensor_sample_phase": "official_live_physics_sensors_after_mj_step",
            "sensor_phase_note": "qpos/qvel are post-integration; gyro/gravity derived fields may precede the final substep. Sensor phase equivalence is not verified.",
            "sensors": {"gyro": gyro.tolist(), "gravity": gravity.tolist(), "positions": positions.tolist(),
                        "velocities": velocities.tolist(), "quat": data.xquat[physics.trunk].tolist()},
            "qpos": data.qpos.tolist(), "qvel": data.qvel.tolist(),
            "row4_integrated": True, "complete_prefix_verified": False, "hardware_validated": False}
        self._snapshots.append(snapshot)
        if self._render_error:
            snapshot["render_error"] = self._render_error
        if self.render_snapshots and not self._render_error:
            if len(self._render_queue) == self._render_queue.maxlen:
                dropped, _ = self._render_queue.popleft()
                dropped["render_status"] = "dropped"
                self._render_dropped += 1
            self._render_queue.append((snapshot, data))
            self._render_event.set()

    def _snapshot_render_loop(self):
        renderer = None
        try:
            physics = self.env.physics
            while not self.stop_event.is_set():
                self._render_event.wait(.05)
                with self.lock:
                    if self._render_prepare_data is not None:
                        data = self._render_prepare_data
                        self._render_prepare_data = None
                        snapshot = None
                        self._render_preparation.update(status="warming", started_monotonic_ns=time.monotonic_ns())
                    elif self._render_queue:
                        snapshot, data = self._render_queue.popleft()
                    else:
                        self._render_event.clear()
                        continue
                # Context creation and rendering occur outside the body lock.
                if renderer is None:
                    renderer = physics.mj.Renderer(physics.model, height=physics.image_size, width=physics.image_size)
                # mj_step can leave derived camera poses from before its final
                # integration. Forward ONLY the frozen copy to render its final
                # qpos; keep official live sensors and control dynamics intact.
                physics.mj.mj_forward(physics.model, data)
                renderer.update_scene(data, camera=physics.camera)
                frame = np.rot90(renderer.render(), k=-1).copy()
                ready = time.monotonic_ns()
                with self.lock:
                    if snapshot is None:
                        self._render_preparation.update(
                            status="cancelled" if self.stop_event.is_set() else "ready",
                            finished_monotonic_ns=ready, frame_shape=list(frame.shape),
                            discarded_frame_sha256=hashlib.sha256(frame.tobytes()).hexdigest())
                    else:
                        snapshot.update(rgb=frame, render_ready_monotonic_ns=ready, render_status="ready",
                                        render_data_phase="frozen_post_integration_qpos_after_mj_forward")
        except Exception as exc:
            # An evidence renderer failure cannot stop or alter body physics.
            with self.lock:
                self._render_error = repr(exc)
                if self._render_preparation["status"] in ("pending", "warming"):
                    self._render_preparation.update(status="failed", error=self._render_error,
                                                    finished_monotonic_ns=time.monotonic_ns())
                self._render_prepare_data = None
                for snapshot in self._snapshots:
                    if snapshot["render_status"] == "pending":
                        snapshot.update(render_status="failed", render_error=self._render_error)
        finally:
            if renderer is not None:
                renderer.close()

    def execution_evidence(self, after_sequence=0):
        _uint(after_sequence, "after_sequence")
        with self.lock:
            if after_sequence > self._event_sequence:
                raise ValueError("evidence cursor is in the future")
            oldest = self._events[0]["sequence"] if self._events else self._event_sequence + 1
            return {"protocol": 1, "events": copy.deepcopy([row for row in self._events if row["sequence"] > after_sequence]),
                "latest_sequence": self._event_sequence, "oldest_sequence": oldest,
                "missing_events": max(0, oldest - after_sequence - 1), "snapshot_sequence": self._snapshot_sequence,
                "oldest_snapshot_sequence": self._snapshots[0]["snapshot_sequence"] if self._snapshots else self._snapshot_sequence + 1,
                "render_dropped": self._render_dropped, "render_error": self._render_error,
                "snapshot_capture_error": self._snapshot_capture_error,
                "renderer_preparation": copy.deepcopy(self._render_preparation),
                "prototype": True, "handoff_verified": False, "hardware_validated": False}

    def post_action_snapshot(self, after_sequence=0, *, require_rgb=False):
        """Return the earliest retained newer matching snapshot, or None.

        A disabled/pending image is never reported as a fresh frame. Cursor
        loss is visible in execution_evidence.oldest_snapshot_sequence.
        """
        _uint(after_sequence, "after_sequence")
        with self.lock:
            for snapshot in self._snapshots:
                if snapshot["snapshot_sequence"] > after_sequence and (not require_rgb or snapshot["render_status"] == "ready"):
                    return copy.deepcopy(snapshot)
        return None

    def close(self):
        # The parent blocks in shutdown() if never started; direct CPU tests and
        # preflight users may construct this prototype without starting it.
        self.stop_event.set()
        self._render_event.set()
        with self.lock:
            if self._render_preparation["status"] in ("pending", "warming"):
                self._render_preparation["status"] = "cancelled"
            self._render_prepare_data = None
        if self.threads[0].is_alive():
            self.server.shutdown()
        self.server.server_close()
        for thread in self.threads:
            if thread.ident is not None:
                thread.join(timeout=3.0)
