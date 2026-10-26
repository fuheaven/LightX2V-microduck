"""Optional RGB-only target keyframes for navigation research.

This memory does not estimate a current target location or issue actions.
An old image remains an old image; consumers must retain its capture tick.
Existing trained policies do not consume this experimental input contract.
"""
from dataclasses import dataclass
import re

import numpy as np

from .pixel_policy import PIXEL_SOURCE_SHA256, detect_colored_ball


@dataclass(frozen=True)
class TargetKeyframe:
    episode_id: str
    target_color: str
    capture_tick: int
    rgb: np.ndarray


class CausalTargetKeyframeMemory:
    """Retain the latest target detection using only captured RGB and instruction color."""

    def __init__(self):
        self.episode_id = None
        self.target_color = None
        self._last_tick = None
        self._last_rgb = None
        self._keyframe = None

    @staticmethod
    def contract():
        return {
            "schema": "microduck_rgb_target_keyframe_v1",
            "selector": "latest_color_component_detection",
            "detector_source_sha256": PIXEL_SOURCE_SHA256,
            "image_size": 256,
            "control_hz": 50,
            "decision_interval_ticks": 5,
            "expiry": "episode_reset",
            "missing_memory": None,
            "current_target_position_available": False,
            "requires_new_model_input_contract": True,
        }

    def reset(self, episode_id, target_color):
        if not isinstance(episode_id, str) or not episode_id or target_color not in ("red", "blue", "yellow"):
            raise ValueError("memory requires an episode and an explicit instruction target color")
        self.episode_id, self.target_color = episode_id, target_color
        self._last_tick = self._last_rgb = self._keyframe = None

    def append(self, episode_id, target_color, tick, rgb):
        if not self.episode_id or (episode_id, target_color) != (self.episode_id, self.target_color):
            raise ValueError("reset memory before changing episode or instruction target")
        image = np.asarray(rgb)
        if type(tick) is not int or tick < 0 or tick % 5 or image.dtype != np.uint8 or image.shape != (256, 256, 3):
            raise ValueError("memory requires decision-aligned captured RGB")
        if self._last_tick is None:
            if tick != 0:
                raise ValueError("memory requires the episode prefix")
        elif tick == self._last_tick and np.array_equal(image, self._last_rgb):
            return
        elif tick != self._last_tick + 5:
            raise ValueError("memory cannot rewrite, reorder or silently omit observations")
        self._last_rgb = image.copy()
        self._last_tick = tick
        if detect_colored_ball(image, self.target_color) is not None:
            self._keyframe = TargetKeyframe(self.episode_id, self.target_color, tick, image.copy())

    def snapshot(self, episode_id, source_tick):
        if (episode_id != self.episode_id or self._last_tick is None
                or self._last_rgb is None or source_tick != self._last_tick):
            raise ValueError("memory may only be read for the latest ingested observation")
        frame = self._keyframe
        return None if frame is None else TargetKeyframe(frame.episode_id, frame.target_color, frame.capture_tick, frame.rgb.copy())

    @property
    def last_processed_tick(self):
        return self._last_tick

    def declare_missing(self, episode_id, tick):
        """Advance an explicitly unavailable capture without creating an image.

        Only the opt-in receipt-verified camera adapter calls this method.
        Ordinary append still rejects every undeclared gap.
        """
        if (episode_id != self.episode_id or self._last_tick is None
                or type(tick) is not int or tick != self._last_tick + 5):
            raise ValueError('missing camera declaration must follow the known episode prefix')
        self._last_tick, self._last_rgb = tick, None


def instruction_target_color(instruction):
    colors = set(re.findall(r"\b(red|blue|yellow)\b", instruction.lower()))
    if len(colors) != 1:
        raise ValueError("target memory requires exactly one explicit instruction color")
    return colors.pop()


@dataclass(frozen=True)
class TargetMemoryGridSpec:
    offsets_ticks: tuple[int, ...] = (150, 50, 0)
    camera_size: int = 256
    max_sample_age_ticks: int = 5
    fallback_history_ticks: int | None = None

    def __post_init__(self):
        if (tuple(self.offsets_ticks) != (150, 50, 0) or self.camera_size != 256
                or type(self.max_sample_age_ticks) is not int or not 0 <= self.max_sample_age_ticks <= 5):
            raise ValueError("target memory v1 requires keyframe, 3s, 1s and current RGB")
        if self.fallback_history_ticks is not None and (
                type(self.fallback_history_ticks) is not int or self.fallback_history_ticks != 300):
            raise ValueError('target memory v2 requires a six-second fallback history')
        object.__setattr__(self, 'offsets_ticks', tuple(self.offsets_ticks))

    @property
    def output_size(self):
        return 512

    def as_dict(self):
        contract = dict(CausalTargetKeyframeMemory.contract(), offsets_ticks=list(self.offsets_ticks),
                    camera_size=self.camera_size, output_size=self.output_size,
                    max_sample_age_ticks=self.max_sample_age_ticks,
                    layout='row_major_target_keyframe_then_3s_1s_current',
                    startup_padding='repeat_episode_first_frame',
                    missing_keyframe='repeat_episode_first_frame_and_mark_padding',
                    capture_age='recorded_metadata_not_additional_model_state')
        if self.fallback_history_ticks is not None:
            contract.update(schema='microduck_rgb_target_keyframe_v2',
                missing_keyframe='use_fixed_history_frame_and_mark_missing_keyframe',
                fallback_history_ticks=self.fallback_history_ticks)
        return contract


def target_memory_ticks(rgb, ticks, instruction):
    """Precompute causal selections once per episode; future detections never rewrite history."""
    rgb, ticks = np.asarray(rgb), np.asarray(ticks)
    if (rgb.dtype != np.uint8 or rgb.shape != (len(ticks), 256, 256, 3)
            or not len(ticks) or not np.issubdtype(ticks.dtype, np.integer)
            or not np.array_equal(ticks, np.arange(len(ticks)) * 5)):
        raise ValueError('memory indexing requires the intact captured episode prefix')
    color = instruction_target_color(instruction)
    memory = CausalTargetKeyframeMemory()
    memory.reset('index', color)
    selected = []
    for tick, image in zip(ticks, rgb):
        memory.append('index', color, int(tick), image)
        frame = memory.snapshot('index', int(tick))
        selected.append(-1 if frame is None else frame.capture_tick)
    return np.asarray(selected, dtype=np.int64)


def _compose_memory(images, ticks, source_tick, spec, frame, frame_tick):
    if type(source_tick) is not int or source_tick < 0 or source_tick % 5:
        raise ValueError('memory grid source must be a decision tick')
    requested = np.maximum(0, source_tick - np.asarray(spec.offsets_ticks))
    indices = np.searchsorted(ticks, requested, side='right') - 1
    if np.any(indices < 0):
        raise ValueError('missing camera prefix')
    selected = np.asarray(ticks)[indices]
    if selected[-1] != source_tick or np.any(requested - selected > spec.max_sample_age_ticks):
        raise ValueError('missing or stale recent camera sample')
    if frame_tick is not None and not 0 <= frame_tick <= source_tick:
        raise ValueError('future keyframes are forbidden')
    if frame is None and spec.fallback_history_ticks is not None:
        fallback_requested = max(0, source_tick - spec.fallback_history_ticks)
        fallback_index = int(np.searchsorted(ticks, fallback_requested, side='right') - 1)
        if fallback_index < 0 or fallback_requested - ticks[fallback_index] > spec.max_sample_age_ticks:
            raise ValueError('missing or stale fallback camera sample')
        keyframe, selected_keyframe_tick = images[fallback_index], int(ticks[fallback_index])
    else:
        keyframe = images[0] if frame is None else frame
        selected_keyframe_tick = 0 if frame_tick is None else frame_tick
    grid = np.concatenate((np.concatenate((keyframe, images[indices[0]]), axis=1),
                           np.concatenate((images[indices[1]], images[indices[2]]), axis=1)), axis=0)
    return (grid, np.asarray([selected_keyframe_tick, *selected], dtype=np.int64),
            np.asarray([frame is None, *(source_tick < np.asarray(spec.offsets_ticks))], dtype=bool))


def compose_target_memory_grid(rgb, ticks, source_tick, spec, selected_memory_ticks):
    """Compose measured images from a previously computed, versioned detector index."""
    ticks, selected_memory_ticks = np.asarray(ticks), np.asarray(selected_memory_ticks)
    if (len(ticks) != len(rgb) or not np.array_equal(ticks, np.arange(len(ticks)) * 5)
            or selected_memory_ticks.shape != ticks.shape
            or not np.issubdtype(selected_memory_ticks.dtype, np.integer)
            or np.any(selected_memory_ticks < -1) or np.any(selected_memory_ticks > ticks)
            or np.any(np.diff(selected_memory_ticks) < 0)
            or np.any((selected_memory_ticks >= 0) & (selected_memory_ticks % 5 != 0))):
        raise ValueError('invalid causal target memory index')
    at = np.searchsorted(ticks, source_tick)
    if at >= len(ticks) or ticks[at] != source_tick:
        raise ValueError('missing current frame')
    memory_tick = int(selected_memory_ticks[at])
    frame = None if memory_tick == -1 else rgb[memory_tick // 5]
    return _compose_memory(rgb, ticks, source_tick, spec, frame, None if frame is None else memory_tick)


class CausalTargetCameraHistory:
    """Bounded streaming equivalent of the offline memory grid composer."""
    def __init__(self, spec):
        self.spec = spec
        self.reset('')

    def reset(self, episode_id):
        self.episode_id = episode_id
        self._memory = CausalTargetKeyframeMemory()
        self._frames, self._ticks = [], []

    @property
    def last_processed_tick(self):
        return self._memory.last_processed_tick

    def declare_missing(self, episode_id, tick):
        if not self.episode_id or episode_id != self.episode_id or not self._ticks:
            raise ValueError('missing camera declaration requires a measured episode prefix')
        self._memory.declare_missing(episode_id, tick)

    def append(self, episode_id, tick, rgb, *, instruction):
        if not self.episode_id or episode_id != self.episode_id:
            raise ValueError('reset camera history for the episode first')
        color = instruction_target_color(instruction)
        if not self._ticks:
            self._memory.reset(episode_id, color)
        self._memory.append(episode_id, color, tick, rgb)
        if self._ticks and self._ticks[-1] == tick:
            return
        self._ticks.append(tick)
        self._frames.append(np.asarray(rgb).copy())
        retained_horizon = max(*self.spec.offsets_ticks, self.spec.fallback_history_ticks or 0)
        cutoff = tick - retained_horizon - self.spec.max_sample_age_ticks
        while len(self._ticks) > 2 and self._ticks[1] < cutoff:
            del self._ticks[1]
            del self._frames[1]

    def compose(self, tick):
        frame = self._memory.snapshot(self.episode_id, tick)
        return _compose_memory(self._frames, self._ticks, tick, self.spec,
                               None if frame is None else frame.rgb,
                               None if frame is None else frame.capture_tick)
