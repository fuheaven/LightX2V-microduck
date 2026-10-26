"""Post-prefix observations for the opt-in leased BAM transport.

This completes the body observation boundary, not the robotd controller or
model handoff. The default simulator does not use this experimental class.
"""
from __future__ import annotations

import base64
import copy

import numpy as np

from .contracts import POLICY_INDICES, scatter_action
from .integrated_body import _uint
from .leased_body import LeasedSlottedBamBodyServer
from .slotted_body import SOURCE_TTL_NS, _same_source


SNAPSHOT_PROTOCOL = "leased_post_prefix_snapshot_v1"


class PrefixSnapshotBamBodyServer(LeasedSlottedBamBodyServer):
    """Publish a frozen fifth-step observation for exactly the next slot.

    Queries never render, advance physics, renew a lease, or refresh capture
    time. A late image remains useful as diagnostic evidence but cannot become
    an eligible action source. The inherited renderer works outside the body
    lock, from the same immutable MuJoCo copy as the observation.
    """

    def _capture_row_four(self, row):
        owner = self._owner
        rows = [e for e in self._events if e["kind"] == "integrate"][-5:]
        source = row["execution"]["source_capture"]
        if owner is None or self._slot_now() >= owner["deadline_ns"] or len(rows) != 5:
            raise ValueError("complete prefix and live owner required for snapshot")
        sequences = set()
        for index, event in enumerate(rows):
            execution = event.get("execution") or {}
            if (event["body_tick"] != source["body_tick"] + index + 1
                    or event["episode_id"] != self.env.episode_id
                    or event["task_epoch"] != self.task_epoch
                    or not event["external_attribution"]
                    or event["write_application_count"] != 1
                    or event["io_sequence_replayed"] or event["repeated_write"]
                    or execution.get("action_index") != index
                    or execution.get("source_tick") != row["execution"]["source_tick"]
                    or not _same_source(source, execution.get("source_capture") or {})):
                raise ValueError("snapshot predecessor is not one complete physical prefix")
            sequences.add(event["io_write_sequence"])
            admissions = [e for e in self._events if e["kind"] == "slot_admitted"
                          and e["io_write_sequence"] == event["io_write_sequence"]]
            if (len(admissions) != 1 or admissions[0]["generation"] != owner["generation"]
                    or admissions[0]["intended_body_tick"] != event["body_tick"]):
                raise ValueError("snapshot prefix ownership or admission evidence missing")
            expected = scatter_action(np.asarray(execution["raw_action14"], np.float32))[list(POLICY_INDICES)]
            if not np.allclose(event["applied_policy_targets14"], expected, atol=1e-6, rtol=0):
                raise ValueError("applied targets differ from the trained raw-action contract")
        if len(sequences) != 5 or None in sequences:
            raise ValueError("snapshot prefix requires five unique IO writes")
        super()._capture_row_four(row)
        snapshot = self._snapshots[-1]
        # The inherited capture runs under the body lock, so the render thread
        # cannot publish this snapshot before these identities are attached.
        capture = {"protocol": 1, "body_tick": snapshot["body_tick"],
                   "task_tick": snapshot["task_tick"], "task_epoch": snapshot["task_epoch"],
                   "episode_id": snapshot["episode_id"],
                   "capture_monotonic_ns": snapshot["capture_monotonic_ns"],
                   "command13": snapshot["command13"],
                   "command_program_sha256": self.command_program_sha256}
        snapshot.update(source_capture=capture, owner_generation=owner["generation"],
                        next_body_tick=snapshot["body_tick"] + 1,
                        next_integration_deadline_ns=self._deadline(snapshot["body_tick"] + 1),
                        complete_prefix_verified=True,
                        prefix_integrations=[{k: e[k] for k in
                            ("sequence", "body_tick", "io_write_sequence", "integration_end_monotonic_ns")}
                            for e in rows])

    def dispatch(self, request):
        if request.get("op") != "post_prefix_snapshot":
            return super().dispatch(request)
        if request.get("protocol") != SNAPSHOT_PROTOCOL:
            raise ValueError("explicit post-prefix snapshot protocol required")
        cursor = _uint(request.get("after_sequence", 0), "after_sequence")
        require_rgb = request.get("require_rgb", True)
        if type(require_rgb) is not bool:
            raise ValueError("require_rgb must be boolean")
        with self.lock:
            self._expire_owner()
            owner = self._authenticate(request)
            if cursor > self._snapshot_sequence:
                raise ValueError("snapshot cursor is in the future")
            oldest = self._snapshots[0]["snapshot_sequence"] if self._snapshots else self._snapshot_sequence + 1
            reply = {"protocol": SNAPSHOT_PROTOCOL, "status": "waiting", "eligible": False,
                     "latest_sequence": self._snapshot_sequence, "oldest_sequence": oldest,
                     "missing_snapshots": max(0, oldest - cursor - 1),
                     "require_rgb": require_rgb, "handoff_verified": False,
                     "counts_as_skill_success": False, "hardware_validated": False}
            if reply["missing_snapshots"]:
                reply["status"] = "cursor_lost"
                return reply
            snapshot = next((s for s in self._snapshots if s["snapshot_sequence"] > cursor), None)
            if snapshot is None:
                return reply
            reply.update(snapshot_sequence=snapshot["snapshot_sequence"],
                         next_body_tick=snapshot["next_body_tick"],
                         next_integration_deadline_ns=snapshot["next_integration_deadline_ns"],
                         render_status=snapshot["render_status"])
            now = self._slot_now()
            if (snapshot["owner_generation"] != owner["generation"]
                    or snapshot["task_epoch"] != self.task_epoch
                    or snapshot["episode_id"] != self.env.episode_id):
                reply["status"] = "retired"
                return reply
            if (self.steps != snapshot["body_tick"] or now >= snapshot["next_integration_deadline_ns"]
                    or now - snapshot["capture_monotonic_ns"] >= SOURCE_TTL_NS):
                reply["status"] = "expired"
                return reply
            if require_rgb and snapshot["render_status"] != "ready":
                reply["status"] = "image_" + snapshot["render_status"]
                return reply
            # Source admission retains the original capture identity. Returning
            # it repeatedly does not change either the action TTL or lease.
            self._source(snapshot["source_capture"])
            reply.update(status="ready", eligible=True,
                         source_capture=copy.deepcopy(snapshot["source_capture"]),
                         prefix_integrations=copy.deepcopy(snapshot["prefix_integrations"]),
                         capture_phase=snapshot["capture_phase"],
                         render_ready_monotonic_ns=snapshot["render_ready_monotonic_ns"],
                         sensor_sample_phase=snapshot["sensor_sample_phase"],
                         state61=list(snapshot["state61"]), rgb=None)
            frame = snapshot["rgb"].copy() if require_rgb else None
        # Encoding/JSON serialization are not part of the physics critical
        # section. A response that arrives late still fails normal admission.
        if frame is not None:
            reply["rgb"] = {"encoding": "base64_uint8_rgb", "shape": list(frame.shape),
                            "data": base64.b64encode(frame.tobytes()).decode("ascii")}
        return reply
