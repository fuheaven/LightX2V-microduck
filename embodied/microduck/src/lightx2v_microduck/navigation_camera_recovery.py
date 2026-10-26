"""Opt-in recovery from receipt-proven camera loss, using only measured images.

This changes the runtime's handling of missing captures, not the trained image
layout. It does not change action expiry or make an incomplete camera cadence
pass the existing realtime acceptance gate.
"""
from dataclasses import dataclass, replace

import numpy as np

from .navigation_visual_memory import (
    CausalTargetCameraHistory, TargetMemoryGridSpec, instruction_target_color,
)
from .wam_history_policy import FastWAMNavigationHistoryPolicy, HistoryNavigationRequest


CAMERA_LOSS_RECOVERY = dict(
    schema='microduck_declared_camera_loss_recovery_v1',
    missing_capture_evidence='snapshot_camera_receipts',
    synthesized_images=False, capture_timestamps_preserved=True,
    history_sample_age_unchanged=True, action_expiry_unchanged=True,
    incomplete_cadence_remains_realtime_failure=True,
)


class DeclaredLossTargetCameraHistory(CausalTargetCameraHistory):
    def reset(self, episode_id):
        super().reset(episode_id)
        self._last_capture_ns = None
        self._declared_losses = []

    @property
    def declared_losses(self):
        return tuple(self._declared_losses)

    def ingest(self, observation, receipts=()):
        """Ingest one real frame; every intervening gap needs its own receipt."""
        tick, captured = observation.tick, observation.monotonic_ns
        image = np.asarray(observation.rgb)
        if (observation.episode_id != self.episode_id or not self.episode_id
                or type(tick) is not int or tick < 0 or tick % 5
                or observation.camera_tick != tick or type(captured) is not int or captured <= 0
                or image.dtype != np.uint8 or image.shape != (256, 256, 3)):
            raise ValueError('camera recovery requires an original timestamped episode frame')
        color = instruction_target_color(observation.instruction)
        if self._ticks and color != self._memory.target_color:
            raise ValueError('reset camera recovery before changing the instruction target')
        previous = self.last_processed_tick
        if previous is None:
            if tick != 0:
                raise ValueError('camera recovery cannot replace the initial measured frame')
        elif tick < previous or (tick == previous and captured != self._last_capture_ns):
            raise ValueError('camera recovery cannot rewrite source time or reorder captures')
        elif tick > previous and captured <= self._last_capture_ns:
            raise ValueError('camera capture timestamps must increase')
        missing = [] if previous is None else list(range(previous + 5, tick, 5))
        declarations = []
        last_capture = self._last_capture_ns
        for absent in missing:
            matches = [r for r in receipts if type(r.get('source_tick')) is int and r['source_tick'] == absent]
            if len(matches) != 1:
                raise ValueError('missing capture requires exactly one camera receipt')
            receipt = matches[0]
            status, timestamp = receipt.get('status'), receipt.get('capture_ns')
            if (status not in ('dropped_busy', 'error') or type(timestamp) is not int
                    or not last_capture < timestamp < captured
                    or (status == 'error' and not isinstance(receipt.get('error'), str))
                    or (status == 'error' and not receipt['error'])):
                raise ValueError('camera loss receipt cannot hide a captured frame or change its time')
            declarations.append((absent, timestamp, status))
            last_capture = timestamp
        # Validate the entire gap before changing memory. A receipt never
        # supplies RGB, a detection, a robot state or a refreshed timestamp.
        for absent, timestamp, status in declarations:
            self.declare_missing(self.episode_id, absent)
            self._declared_losses.append((absent, timestamp, status))
        self.append(self.episode_id, tick, image, instruction=observation.instruction)
        self._last_capture_ns = captured


@dataclass(frozen=True)
class DeclaredLossNavigationRequest(HistoryNavigationRequest):
    camera_losses: tuple[tuple[int, int, str], ...] = ()


def loss_diagnostics(losses):
    return dict(contract=dict(CAMERA_LOSS_RECOVERY), declared_losses=[
        dict(source_tick=tick, capture_ns=captured, status=status) for tick, captured, status in losses])


class FastWAMDeclaredCameraLossPolicy(FastWAMNavigationHistoryPolicy):
    """Experimental runtime adapter; ordinary history policies remain strict."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.history.spec, TargetMemoryGridSpec):
            raise ValueError('declared camera recovery requires target-keyframe history')
        self.history = DeclaredLossTargetCameraHistory(self.history.spec)
        self.metadata['camera_loss_recovery'] = dict(CAMERA_LOSS_RECOVERY)

    def observe(self, observation):
        self.history.ingest(observation)

    def prepare_camera_request(self, observation, receipts):
        self.history.ingest(observation, receipts)
        grid, ticks, padding = self.history.compose(observation.tick)
        return DeclaredLossNavigationRequest(
            replace(observation, rgb=grid), self.history.spec, tuple(ticks.tolist()),
            tuple(padding.tolist()), self._history_generation, self.history.declared_losses)

    def prepare_request(self, observation):
        return self.prepare_camera_request(observation, ())

    def predict_prepared(self, request):
        if not isinstance(request, DeclaredLossNavigationRequest):
            raise ValueError('camera recovery needs its explicitly prepared request')
        result = super().predict_prepared(request)
        result.diagnostics['camera_loss_recovery'] = loss_diagnostics(request.camera_losses)
        return result
