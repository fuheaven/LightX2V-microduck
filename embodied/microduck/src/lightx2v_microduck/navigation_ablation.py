"""Explicit input/action interventions, never changes to task scoring."""

from dataclasses import replace

import numpy as np

from .contracts import Observation
from .navigation import ACTION_HIGH, ACTION_LOW, ACTION_NAMES, NavigationChunk
from .navigation_requests import request_methods
from .tasks import get_task


class NavigationInputAblation:
    """Alter model inputs while preserving the original sensor and task record.

    The evaluator still scores the requested task. An instruction override is
    a causal diagnostic, not a new user command or an accepted capability run.
    The wrapper has no simulator or privileged-state access.

    The default observation stage includes changes to language-conditioned
    frame selection. The prepared_request stage intervenes only after input
    preparation, preserving the original history and target-memory selection.
    """

    def __init__(self, policy, *, mask_rgb=False, instruction_color=None, stage="observation"):
        if instruction_color not in (None, "red", "blue", "yellow"):
            raise ValueError("instruction_color must be red, blue, yellow or None")
        if stage not in ("observation", "prepared_request"):
            raise ValueError("input ablation stage must be observation or prepared_request")
        self.policy = policy
        self._prepare, self._predict_prepared = request_methods(policy)
        self.mask_rgb = bool(mask_rgb)
        self.instruction_color = instruction_color
        self.stage = stage
        self.intervention = {"mask_rgb": self.mask_rgb, "instruction_color": instruction_color,
                             "evaluator_target_unchanged": True,
                             "sensor_recording_unchanged": True,
                             "application_stage": stage,
                             "input_preparation_unchanged": stage == "prepared_request",
                             "not_capability_acceptance": True}
        self.metadata = {**getattr(policy, "metadata", {}), "input_intervention": self.intervention}

    def reset(self, episode_id):
        self.policy.reset(episode_id)

    def _intervene(self, observation):
        changes = {}
        if self.mask_rgb:
            if observation.rgb is None:
                raise ValueError("cannot mask an absent camera observation")
            changes["rgb"] = np.zeros_like(observation.rgb)
        if self.instruction_color:
            if observation.task_id not in {f"find_{color}_ball" for color in ("red", "blue", "yellow")}:
                raise ValueError("instruction-color intervention requires a visual target task")
            changes["instruction"] = get_task(f"find_{self.instruction_color}_ball").instruction
        # episode/tick/camera_tick/capture time and proprioception are never
        # rewritten, so ablation cannot disguise stale observations or plans.
        return replace(observation, **changes)

    def prepare_request(self, observation):
        if self.stage == "observation":
            return self._prepare(self._intervene(observation))
        request = self._prepare(observation)
        if isinstance(request, Observation):
            return self._intervene(request)
        from .wam_history_policy import HistoryNavigationRequest

        if not isinstance(request, HistoryNavigationRequest):
            raise TypeError("prepared input ablation requires an Observation or HistoryNavigationRequest")
        # Keep source ticks, padding, contract and reset generation intact.
        # Only the immutable request snapshot changes, never live history.
        return replace(request, observation=self._intervene(request.observation))

    def predict_prepared(self, request):
        result = self._predict_prepared(request)
        if not isinstance(result, NavigationChunk):
            raise TypeError("input ablation requires a navigation policy")
        return replace(result, diagnostics={**result.diagnostics, "input_intervention": self.intervention})

    def predict(self, observation):
        return self.predict_prepared(self.prepare_request(observation))

    def close(self):
        close = getattr(self.policy, "close", None)
        if close:
            close()


class NavigationConstantActionAblation:
    """Project explicitly selected constant training dimensions for diagnosis.

    This does not silently change the native model or normalization. Reports
    retain the intervention, original model identity, and unchanged timing.
    """

    def __init__(self, policy, stats, *, dimensions, stats_sha256):
        if stats.get("normalization_split") != "train":
            raise ValueError("action projection requires training-only statistics")
        if not dimensions or len(dimensions) != len(set(dimensions)):
            raise ValueError("select nonempty unique constant action dimensions")
        action_stats = stats["action"]["default"]
        low = np.asarray(action_stats["global_min"], dtype=np.float32)
        high = np.asarray(action_stats["global_max"], dtype=np.float32)
        if low.shape != (5,) or high.shape != (5,) or not np.isfinite([low, high]).all():
            raise ValueError("invalid navigation action statistics")
        self.constants = {}
        for name in dimensions:
            if name not in ACTION_NAMES:
                raise ValueError(f"unknown navigation dimension: {name}")
            index = ACTION_NAMES.index(name)
            if low[index] != high[index]:
                raise ValueError(f"training action dimension is not constant: {name}")
            if not ACTION_LOW[index] <= low[index] <= ACTION_HIGH[index]:
                raise ValueError(f"constant is outside the navigation contract: {name}")
            self.constants[index] = float(low[index])
        self.policy = policy
        self._prepare, self._predict_prepared = request_methods(policy)
        self.intervention = {"kind": "project_training_constants",
                             "fixed_actions": {ACTION_NAMES[i]: value for i, value in self.constants.items()},
                             "stats_sha256": stats_sha256, "evaluator_target_unchanged": True,
                             "timing_and_expiry_unchanged": True, "not_capability_acceptance": True}
        self.metadata = {**getattr(policy, "metadata", {}), "action_intervention": self.intervention}

    def reset(self, episode_id):
        self.policy.reset(episode_id)

    def prepare_request(self, observation):
        return self._prepare(observation)

    def predict_prepared(self, request):
        result = self._predict_prepared(request)
        if not isinstance(result, NavigationChunk):
            raise TypeError("action ablation requires a navigation policy")
        actions = result.actions.copy()
        for index, value in self.constants.items():
            actions[:, index] = value
        return replace(result, actions=actions,
                       diagnostics={**result.diagnostics, "action_intervention": self.intervention})

    def predict(self, observation):
        return self.predict_prepared(self.prepare_request(observation))

    def close(self):
        close = getattr(self.policy, "close", None)
        if close:
            close()
