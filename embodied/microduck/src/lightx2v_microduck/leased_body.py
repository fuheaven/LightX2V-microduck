"""Opt-in ownership leases for the private fixed-slot BAM transport.

This is not the robotd policy session: its future proxy must forward the actual
model-owner liveness, not renew blindly. The ordinary local Safety write still
preempts external control. Missing control remains an unattributed target hold,
not a verified official-policy fallback or a complete real-time backend.
"""
from __future__ import annotations

import secrets

from .integrated_body import _uint
from .slotted_body import SlottedBamBodyServer


LEASE_PROTOCOL = "fixed_slot_owner_lease_probe_v1"
LEASE_NS = 150_000_000


class LeasedSlottedBamBodyServer(SlottedBamBodyServer):
    """One owner capability, bounded by body-clock expiry and generation.

    Reads and normal TCP EOF do not renew or close a lease. Only an explicitly
    increasing heartbeat does. The physics timer checks expiry even when no
    RPC arrives; heartbeat never changes a queued action's source TTL.
    """

    def __init__(self, *args, **kwargs):
        self._owner = None
        super().__init__(*args, **kwargs)

    def _invalidate(self, reason):
        owner = self._owner
        self._owner = None
        super()._invalidate(reason)
        self._known_sources.clear()
        if owner is not None:
            self._record("owner_retired", generation=owner["generation"],
                         reason=reason, lease_deadline_ns=owner["deadline_ns"],
                         detected_monotonic_ns=self._slot_now(), body_tick=self.steps)

    def _expire_owner(self):
        if self._owner is not None and self._slot_now() >= self._owner["deadline_ns"]:
            self._invalidate("owner_lease_expired")

    def _authenticate(self, request):
        generation = _uint(request.get("generation"), "generation")
        token = request.get("owner_token")
        if (self._owner is None or generation != self._owner["generation"]
                or type(token) is not str or len(token) != 64
                or any(c not in "0123456789abcdef" for c in token)
                or not secrets.compare_digest(token, self._owner["token"])):
            raise ValueError("owner capability absent, expired or retired")
        return self._owner

    def prepare_snapshot_renderer(self):
        with self.lock:
            if self._owner is not None and self._render_preparation["status"] == "not_requested":
                raise ValueError("prepare renderer before opening an owner")
            return super().prepare_snapshot_renderer()

    def dispatch(self, request):
        with self.lock:
            self._expire_owner()
            op = request.get("op")
            if op == "owner_open":
                if request.get("protocol") != LEASE_PROTOCOL:
                    raise ValueError("explicit owner lease protocol required")
                if request.get("episode_id") != self.env.episode_id:
                    raise ValueError("foreign owner episode")
                if _uint(request.get("task_epoch"), "task_epoch") != self.task_epoch:
                    raise ValueError("foreign owner task epoch")
                if self._owner is not None:
                    raise ValueError("an owner is already active")
                if self._render_preparation["status"] not in ("disabled", "not_requested", "ready"):
                    raise ValueError("requested renderer preparation must complete before owner open")
                if not self.torque_on or not self.released:
                    raise ValueError("body is not enabled")
                self._invalidate("owner_open")
                now = self._slot_now()
                self._owner = {"token": secrets.token_hex(32), "generation": self._generation,
                               "heartbeat_sequence": 0, "deadline_ns": now + LEASE_NS}
                self._record("owner_opened", generation=self._generation,
                             monotonic_ns=now, lease_deadline_ns=self._owner["deadline_ns"],
                             body_tick=self.steps)
                return {"protocol": LEASE_PROTOCOL, "owner_token": self._owner["token"],
                        "generation": self._generation, "lease_ns": LEASE_NS,
                        "lease_deadline_ns": self._owner["deadline_ns"]}
            if op == "owner_heartbeat":
                owner = self._authenticate(request)
                sequence = _uint(request.get("heartbeat_sequence"), "heartbeat_sequence")
                if sequence <= owner["heartbeat_sequence"]:
                    raise ValueError("heartbeat sequence must increase")
                owner["heartbeat_sequence"] = sequence
                owner["deadline_ns"] = self._slot_now() + LEASE_NS
                self._record("owner_heartbeat", generation=owner["generation"],
                             heartbeat_sequence=sequence, monotonic_ns=self._slot_now(),
                             lease_deadline_ns=owner["deadline_ns"], body_tick=self.steps)
                return {"generation": owner["generation"], "lease_deadline_ns": owner["deadline_ns"]}
            if op == "owner_close":
                self._authenticate(request)
                self._invalidate("owner_closed")
                return {"generation": self._generation}
            if op == "invalidate_slots":
                self._authenticate(request)
            if op == "write" and "slot" in request:
                if not isinstance(request["slot"], dict):
                    raise ValueError("slot must be an object")
                self._authenticate(request["slot"])
            reply = super().dispatch(request)
            if op == "slot_status":
                reply.update(ownership_protocol=LEASE_PROTOCOL, owner_active=self._owner is not None,
                             lease_ns=LEASE_NS,
                             lease_deadline_ns=self._owner["deadline_ns"] if self._owner else None)
            return reply

    def _integrate_once(self):
        with self.lock:
            self._expire_owner()
            return super()._integrate_once()
