"""Bounded action consumption shared by realtime harnesses and protocol tests.

Actions are indexed by the source observation's control tick. Late arrivals do
not restart row zero, and a packet can never revive a previous episode.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .contracts import JointResidualChunk


@dataclass(frozen=True)
class ExecutionDecision:
    action: np.ndarray | None
    source: str
    reason: str
    source_tick: int | None = None
    action_index: int | None = None


class ActionBuffer:
    def __init__(self, *, max_observation_age_ticks=10, max_wall_age_s=0.2):
        if max_observation_age_ticks < 1 or max_wall_age_s <= 0:
            raise ValueError("positive freshness limits required")
        self.max_observation_age_ticks = max_observation_age_ticks
        self.max_wall_age_s = max_wall_age_s
        self.reset("")

    def reset(self, episode_id):
        self.episode_id = episode_id
        self.chunk = None
        self.received_at = 0.0
        self.last_source_tick = -1
        self.last_consumed_tick = -1
        self.accepted = self.rejected = self.fallback_ticks = self.external_ticks = 0
        self.rejection_reason = ""

    def submit(self, chunk: JointResidualChunk, *, current_tick: int, now: float | None = None):
        reason = ""
        if chunk.episode_id != self.episode_id:
            reason = "wrong_episode"
        elif chunk.action_profile != "raw_residual":
            reason = "unsupported_action_profile"
        elif chunk.source_tick <= self.last_source_tick:
            reason = "out_of_order"
        elif chunk.source_tick > current_tick:
            reason = "future_observation"
        elif current_tick - chunk.source_tick >= self.max_observation_age_ticks:
            reason = "stale_observation"
        elif current_tick >= chunk.source_tick + len(chunk.actions):
            reason = "chunk_exhausted"
        elif chunk.valid_until_tick is not None and current_tick >= chunk.valid_until_tick:
            reason = "expired"
        if reason:
            self.rejected += 1
            self.rejection_reason = reason
            return False
        self.chunk = chunk
        self.last_source_tick = chunk.source_tick
        self.received_at = time.monotonic() if now is None else now
        self.accepted += 1
        return True

    def consume(self, tick: int, *, now: float | None = None) -> ExecutionDecision:
        if tick <= self.last_consumed_tick:
            raise ValueError("control ticks must advance exactly once per consumption")
        self.last_consumed_tick = tick
        now = time.monotonic() if now is None else now
        chunk = self.chunk
        reason = "no_chunk"
        if chunk is not None:
            index = tick - chunk.source_tick
            expiry = min(chunk.source_tick + self.max_observation_age_ticks, chunk.valid_until_tick or (chunk.source_tick + len(chunk.actions)))
            if now - self.received_at >= self.max_wall_age_s or tick >= expiry:
                reason = "expired"
            elif index < 0 or index >= len(chunk.actions):
                reason = "chunk_exhausted"
            else:
                self.external_ticks += 1
                return ExecutionDecision(chunk.actions[index].copy(), "external", "valid", chunk.source_tick, index)
            self.chunk = None
        self.fallback_ticks += 1
        return ExecutionDecision(None, "official_fallback", reason)

    def status(self):
        return {
            "episode_id": self.episode_id,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "external_ticks": self.external_ticks,
            "fallback_ticks": self.fallback_ticks,
            "rejection_reason": self.rejection_reason,
        }
