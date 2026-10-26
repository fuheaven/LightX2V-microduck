"""Opt-in fixed-slot BAM transport foundation, not a real-time VLA backend.

No default service uses this class. A future robotd integration must send the
Safety-processed write with an explicit slot and implement owner leases and
post-integration observation handoff. Direct probe writes are not Safety proof.
"""
from __future__ import annotations

import copy
import time

import numpy as np

from .integrated_body import IntegratedBamBodyServer, _execution, _uint


MODE = "fixed_body_slot_probe_v1"
PERIOD_NS = 20_000_000
SOURCE_TTL_NS = 200_000_000


def _same_source(known, received):
    """RemoteCapture serializes command13 as f32; compare that exact contract.

    Rust may serialize f32(0.35) as 0.35 while Python published its expanded
    double representation. This is not permission to change any f32 value or
    the integer/string identity fields.
    """
    if known is None or set(known) != set(received):
        return False
    if any(type(known[k]) is not type(received[k]) or known[k] != received[k]
           for k in known if k != "command13"):
        return False
    command = received.get("command13")
    if not isinstance(command, list) or len(command) != 13 or any(type(v) not in (int, float) for v in command):
        return False
    with np.errstate(over="ignore", invalid="ignore"):
        values = np.asarray(command, dtype=np.float32)
    return bool(np.isfinite(values).all() and np.array_equal(values, np.asarray(known["command13"], dtype=np.float32)))


class SlottedBamBodyServer(IntegratedBamBodyServer):
    """One scheduled, generation-bound write for exactly the next body tick.

    Legacy writes remain available to local standby/Safety and preempt queued
    external writes. On a missed slot physics continues with an unattributed
    target hold, recorded as a fault in the active prefix. This is not an
    implementation of official-policy fallback or a deployment safety claim.
    """

    def __init__(self, *args, monotonic_ns=None, **kwargs):
        self._slot_now = monotonic_ns or time.monotonic_ns
        self._injected_clock = monotonic_ns is not None
        self._slot_origin_ns = None
        self._generation = 0
        self._pending_slot = None
        self._prefix = None
        self._last_slot_io_sequence = -1
        self._known_sources = {}
        super().__init__(*args, **kwargs)

    def _deadline(self, tick):
        if self._slot_origin_ns is None:
            self._slot_origin_ns = self._slot_now() + PERIOD_NS
        return self._slot_origin_ns + (tick - 1) * PERIOD_NS

    def _invalidate(self, reason):
        prior = self._generation
        queued = self._pending_slot
        self._generation += 1
        self._pending_slot = None
        self._prefix = None
        self._current_write = None
        self._record("slots_invalidated", old_generation=prior, generation=self._generation,
                     reason=reason, body_tick=self.steps,
                     pending_body_tick=queued["tick"] if queued else None,
                     monotonic_ns=self._slot_now())

    def _source(self, capture):
        key = capture["capture_monotonic_ns"]
        self._known_sources[key] = copy.deepcopy(capture)
        # Sources older than the original ten-body-tick TTL can no longer be
        # used. Also bound memory when a client polls repeatedly at one tick.
        self._known_sources = {k: v for k, v in self._known_sources.items()
                               if v["body_tick"] + 10 > self.steps}
        while len(self._known_sources) > 128:
            del self._known_sources[next(iter(self._known_sources))]

    def dispatch(self, request):
        with self.lock:
            op = request.get("op")
            if op == "slot_status":
                tick = self.steps + 1
                return {"protocol": MODE, "generation": self._generation,
                        "body_tick": self.steps, "next_body_tick": tick,
                        "integration_deadline_ns": self._deadline(tick),
                        "period_ns": PERIOD_NS, "episode_id": self.env.episode_id,
                        "task_epoch": self.task_epoch,
                        "injected_clock": self._injected_clock,
                        "handoff_verified": False}
            if op == "invalidate_slots":
                generation = _uint(request.get("generation"), "generation")
                if generation != self._generation:
                    raise ValueError("retired generation cannot invalidate the current session")
                self._invalidate("explicit_cancel")
                return {"generation": self._generation}
            if op == "write" and "slot" in request:
                return self._admit_slot(request)
            if op == "write" and request.get("execution") is not None:
                raise ValueError("attributed external writes require an explicit slot")
            # Validate an ordinary write before letting it preempt the queue.
            reply = super().dispatch(request)
            if op in ("write", "torque"):
                self._invalidate("local_" + op)
            if op == "read":
                self._source(reply["source_capture"])
            return reply

    def _admit_slot(self, request):
        slot = request["slot"]
        if not isinstance(slot, dict) or slot.get("protocol") != MODE:
            raise ValueError("explicit fixed-body-slot protocol required")
        generation = _uint(slot.get("generation"), "generation")
        tick = _uint(slot.get("intended_body_tick"), "intended_body_tick")
        valid_until = _uint(slot.get("valid_until_body_tick"), "valid_until_body_tick")
        execution = _execution(request.get("execution"))
        sequence = _uint(request.get("io_write_sequence"), "io_write_sequence")
        targets = np.asarray(request.get("targets"), dtype=np.float64)
        if targets.shape != (15,) or not np.isfinite(targets).all():
            raise ValueError("15 finite Safety-processed targets required")
        if execution is None or execution["execution_mode"] != MODE or execution["execution_source"] != "external":
            raise ValueError("fixed slot requires explicit external probe execution")
        index = execution["action_index"]
        if index not in range(5):
            raise ValueError("only the original five-row prefix is supported")
        capture = execution.get("source_capture")
        if not isinstance(capture, dict):
            raise ValueError("published body source is required")
        captured = _uint(capture.get("capture_monotonic_ns"), "capture_monotonic_ns")
        source_tick = _uint(capture.get("body_tick"), "source body_tick")
        if generation != self._generation:
            raise ValueError("retired generation")
        if execution["episode_id"] != self.env.episode_id or capture.get("episode_id") != self.env.episode_id:
            raise ValueError("foreign episode")
        if capture.get("task_epoch") != self.task_epoch:
            raise ValueError("foreign task epoch")
        if not _same_source(self._known_sources.get(captured), capture):
            raise ValueError("unknown or changed source identity")
        now = self._slot_now()
        if not captured <= execution["selected_monotonic_ns"] <= now:
            raise ValueError("selection must follow the source capture and precede admission")
        deadline = self._deadline(tick)
        reason = None
        if not self.torque_on or not self.released:
            reason = "body_not_enabled"
        elif sequence <= max(self._last_slot_io_sequence, self._highest_io_write_sequence
                             if self._highest_io_write_sequence is not None else -1):
            reason = "duplicate_io_sequence"
        elif tick != self.steps + 1 or tick != source_tick + index + 1:
            reason = "nonconsecutive_body_slot"
        elif captured > now or now - captured >= SOURCE_TTL_NS or tick >= min(valid_until, source_tick + 10):
            reason = "source_expired"
        elif now >= deadline:
            reason = "admission_deadline_missed"
        elif self._pending_slot is not None:
            reason = "slot_already_occupied"
        identity = (execution["episode_id"], execution["source_tick"], captured)
        if reason is None and index == 0:
            if self._prefix is not None and self._prefix["next_index"] != 5:
                reason = "previous_prefix_incomplete"
        elif reason is None:
            if (self._prefix is None or self._prefix["identity"] != identity
                    or self._prefix["next_index"] != index):
                reason = "action_row_out_of_order"
        if reason:
            self._record("slot_rejected", reason=reason, generation=generation,
                         intended_body_tick=tick, monotonic_ns=now)
            # Current-session stale/duplicate writes cannot leave other pending
            # actions executable. An old generation cannot cancel a newer one.
            self._invalidate(reason)
            raise ValueError(reason)
        copied = copy.deepcopy(request)
        copied.pop("slot")
        self._pending_slot = {"tick": tick, "generation": generation, "request": copied,
                              "identity": identity, "index": index, "source_ns": captured,
                              "valid_until": min(valid_until, source_tick + 10),
                              "deadline_ns": deadline, "admitted_ns": now}
        self._last_slot_io_sequence = sequence
        self._record("slot_admitted", generation=generation, intended_body_tick=tick,
                     action_index=index, io_write_sequence=sequence,
                     deadline_ns=deadline, monotonic_ns=now)
        return {"generation": generation, "intended_body_tick": tick,
                "integration_deadline_ns": deadline}

    def _integrate_once(self):
        with self.lock:
            tick = self.steps + 1
            deadline = self._deadline(tick)
            now = self._slot_now()
            if now < deadline:
                raise RuntimeError("body cannot integrate before its fixed slot")
            queued = self._pending_slot
            self._pending_slot = None
            reason = "standby_hold"
            applied = False
            if queued is not None:
                if (queued["generation"] != self._generation or queued["tick"] != tick
                        or now - queued["source_ns"] >= SOURCE_TTL_NS or tick >= queued["valid_until"]):
                    reason = "queued_write_expired"
                    self._invalidate(reason)
                elif now >= deadline + PERIOD_NS:
                    reason = "integration_slot_missed"
                    self._invalidate(reason)
                elif not self.torque_on or not self.released:
                    reason = "body_not_enabled"
                    self._invalidate(reason)
                else:
                    # Execute the same 15D RemoteIo write and BAM mapping; the
                    # parent produces write -> integration -> snapshot evidence.
                    super().dispatch(queued["request"])
                    applied = True
                    reason = "scheduled_external_write"
            elif self._prefix is not None:
                reason = "scheduled_target_missing_hold"
                self._invalidate(reason)
            if not applied:
                self._current_write = None
            row = super()._integrate_once()
            if applied:
                self._prefix = {"identity": queued["identity"], "next_index": queued["index"] + 1}
            # Never reuse an external write identity on another physics tick.
            self._current_write = None
            self._record("slot_resolution", body_tick=tick, generation=self._generation,
                         integration_sequence=row["sequence"], deadline_ns=deadline,
                         resolved_monotonic_ns=now, reason=reason,
                         external_applied=applied, injected_clock=self._injected_clock)
            return row

    def _physics_loop(self):
        try:
            while not self.stop_event.is_set():
                with self.lock:
                    deadline = self._deadline(self.steps + 1)
                self.stop_event.wait(max(0., (deadline - self._slot_now()) / 1e9))
                if self.stop_event.is_set():
                    break
                if self._slot_now() >= deadline + PERIOD_NS:
                    self.late_ticks += 1
                self._integrate_once()
                # The origin never moves with model/RPC latency. Overrun slots
                # remain explicit; no missed physical transition is skipped.
        except Exception as exc:
            self.fault = repr(exc)
            self.stop_event.set()

    def begin_task(self):
        with self.lock:
            self._invalidate("task_epoch_changed")
            self._known_sources.clear()
            return super().begin_task()
