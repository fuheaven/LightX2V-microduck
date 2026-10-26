"""VLM decisions in the same action/execution contract as a navigation WAM.

The evaluator owns physics, ONNX, clocks and acceptance. This adapter receives
actor observations and execution receipts only. It cannot query scene truth.
"""

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import threading
from types import SimpleNamespace

import numpy as np

from .contracts import Observation
from .navigation import DECISION_TICKS, NavigationChunk
from .navigation_action_contracts import action_contract
from .planner_context import PlannerContext
from .planner_tools import MEASURED_DECISION_PROFILE
from .skill_planner import SkillPlannerPolicy, validate_plan


@dataclass(frozen=True)
class VLMRequest:
    generation: int
    sequence: int
    observation: Observation
    context: dict
    previous: Observation | None


class VLMNavigationPolicy:
    def __init__(self, *, endpoint, model, contract, context_mode="feedback", expected_service_identity=None,
                 prompt_profile="legacy_v1"):
        bounds = action_contract(contract)
        # The HTTP client has no simulator/bank reference. The common evaluator
        # alone turns the returned navigation chunk into official ONNX actions.
        self.client = SkillPlannerPolicy(SimpleNamespace(profile="official_bam"), endpoint=endpoint,
                                        model=model, context_mode=context_mode, navigation_contract=contract,
                                        expected_service_identity=expected_service_identity, prompt_profile=prompt_profile)
        self.contract, self.context_mode = contract, context_mode
        self.metadata = {
            "family": "VLM", "model": model, "backend": "HTTP_model_service",
            "expected_service_identity": expected_service_identity,
            "contract": contract, "context_mode": context_mode,
            "prompt_profile": prompt_profile,
            "policy_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "actions": ["vx_mps", "vy_mps", "yaw_rate_rps", "head_yaw_rad", "stand_probability"],
            "lower": list(bounds.lower), "upper": list(bounds.upper),
            "skills": ["walk", "stand"], "state_dim": 48,
            "task_memory": "model_hypothesis_applied_only_after_actual_chunk_execution",
            "truth_access": False, "hardware_validated": False,
            "http_request_timeout_s": 5.0,
            "expiry_owner": "shared_navigation_buffer", "action_chunk": "constant_intent_for_model_requested_duration",
        }
        self.context = PlannerContext(measured_heading=prompt_profile == MEASURED_DECISION_PROFILE)
        if prompt_profile == MEASURED_DECISION_PROFILE:
            from .planner_motion import MOTION_CONTRACT_SHA256, MOTION_VERSION

            self.metadata.update(motion_feedback_version=MOTION_VERSION,
                                 motion_feedback_contract_sha256=MOTION_CONTRACT_SHA256)
        self.lock = threading.Lock()
        self._generation = 0
        self.closed = False
        self.reset("")

    def reset(self, episode_id):
        with self.lock:
            self._generation += 1
            self._sequence = 0
            self.episode_id = episode_id
            self.context.reset(episode_id)
            self.sources = {}
            self.executing = None
            self.last_fallback = None
            self.client.reset(episode_id)

    def prepare_request(self, observation):
        with self.lock:
            if self.closed or observation.episode_id != self.episode_id:
                raise ValueError("planner session does not match observation")
            if observation.rgb is None:
                raise ValueError("navigation VLM requires an actual camera image")
            self.context.observe(observation)
            context, previous = self.context.snapshot(observation, self.context_mode)
            self._sequence += 1
            return VLMRequest(self._generation, self._sequence, observation, context, previous)

    def predict_prepared(self, request):
        with self.lock:
            if self.closed or request.generation != self._generation:
                raise ValueError("request was reset before inference")
        value = self.client.request_plan(request.observation, context=request.context, previous=request.previous)
        plan = validate_plan(value, navigation_contract=self.contract)
        action = np.asarray([*plan["command"], plan["head_yaw_rad"], float(plan["skill"] == "stand")], np.float32)
        duration_ticks = round(plan["duration_s"] * 50)
        actions = np.repeat(action[None], math.ceil(duration_ticks / DECISION_TICKS), axis=0)
        key = f"{request.generation}:{request.sequence}"
        with self.lock:
            if self.closed or request.generation != self._generation:
                raise ValueError("request was reset during inference")
            # Only completed predictions need source images; a bounded ring
            # covers the current plan and queued replies. Rejected plans never
            # update task memory or execution history.
            self.sources[key] = request.observation
            while len(self.sources) > 8:
                oldest = next(name for name in self.sources if name != self.executing)
                del self.sources[oldest]
        observation = request.observation
        return NavigationChunk(actions, observation.episode_id, observation.tick, observation.monotonic_ns,
                               observation.tick + duration_ticks, contract=self.contract,
                               diagnostics={"planner_generation": request.generation, "planner_request": key,
                                            "plan": plan, "serving_evidence": value.get("_serving_evidence"),
                                            "context_mode": self.context_mode})

    def predict(self, observation):
        return self.predict_prepared(self.prepare_request(observation))

    def observe_execution(self, before, after, intent, evidence, chunk):
        """Observe one completed control tick, never evaluator outcomes/truth.

        An executor receipt proves which high-level intent was consumed. The
        subsequent raw-action sensor is recorded as a measurement, not as an
        independent confirmation of a commanded displacement or task success.
        """
        with self.lock:
            if self.closed or before.episode_id != self.episode_id or after.episode_id != self.episode_id:
                return
            self.context.observe(before)
            if evidence.get("execution_source") != "navigation_model" or chunk is None:
                reason = evidence.get("reason", "no_plan")
                if reason != self.last_fallback:
                    self.context.failure("executor_" + reason, before.tick)
                self.context.finish("executor_fallback", before.tick)
                self.executing, self.last_fallback = None, reason
                self.context.observe(after)
                return
            diagnostics = chunk.diagnostics
            if diagnostics.get("planner_generation") != self._generation:
                raise ValueError("executor used a plan from another planner session")
            key = diagnostics["planner_request"]
            source = self.sources.get(key)
            if source is None:
                raise ValueError("executed plan lacks its recorded source observation")
            if key != self.executing:
                plan = validate_plan(diagnostics["plan"], navigation_contract=self.contract)
                self.context.begin(plan, source, before)
                # The source image is explicitly labelled by its camera tick;
                # it may precede the actual action start in realtime mode.
                self.context.reference = source
                self.context.active["reference_image_origin"] = "executed_plan_source"
                self.context.active["receipt_semantics"] = "executor_consumed_intent_not_verified_motion"
                self.executing = key
            self.last_fallback = None
            self.context.active["issued_ticks"] += 1
            self.context.active["executor_consumed_ticks"] = self.context.active.get("executor_consumed_ticks", 0) + 1
            self.context.active["last_executed_command13"] = intent.command.tolist()
            self.context.active["last_joint_action_l2"] = float(np.linalg.norm(after.state[34:48]))
            self.context.observe(after)

    def close(self):
        with self.lock:
            self.closed = True
            self._generation += 1
            self.sources.clear()
            self.client.close()
