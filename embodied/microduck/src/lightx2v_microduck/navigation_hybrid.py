"""A VLM selects bounded skills; a selected WAM uses fresh actor observations.

The common navigation evaluator owns physics, official ONNX, expiry and task
scoring. A supervisor decision is a finite skill request, not a motor command.
Every delegated action retains its own camera tick and capture timestamp.
"""

from copy import deepcopy
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import re
import threading
import time
from types import SimpleNamespace

import numpy as np

from .contracts import Observation
from .navigation import DECISION_TICKS, NavigationChunk
from .navigation_action_contracts import action_contract
from .navigation_requests import request_methods
from .planner_context import PlannerContext, camera_tick
from .planner_tools import TOOL_ACTIONS, TOOL_CONTRACT_SHA256, decode_tool_call
from .planner_response import JSON_OR_FENCE
from .planner_validation_feedback import VALIDATION_FEEDBACK_PROTOCOL, request_with_validation_feedback
from .skill_planner import SkillPlannerPolicy


HYBRID_PROTOCOL = "microduck_vlm_wam_skill_router_v2"
NAVIGATION_INSTRUCTIONS = {
    f"navigate_{color}": f"Find the {color} ball, approach it without hitting obstacles, then stop and wait."
    for color in ("red", "blue", "yellow")
}


def validate_skill_request(value, available):
    if not isinstance(value, dict) or set(value) - {"tool", "duration_s", "memory"}:
        raise ValueError("hybrid tool call contains unsupported arguments")
    tool = value.get("tool")
    if not isinstance(tool, str) or tool not in (*TOOL_ACTIONS, *available):
        raise ValueError("unknown or unavailable hybrid skill")
    # Use the existing strict duration/memory contract without decoding a
    # learned skill into a constant velocity or choosing a replacement action.
    decode_tool_call({**value, "tool": "stop"})
    return deepcopy(value)


def with_hybrid_skills(named_payload, skills, registry_sha256):
    """Reuse actor context/images verbatim while exposing the declared skills."""
    if named_payload.get("metadata", {}).get("prompt_profile") != "bounded_tools_named_sensors_v2":
        raise ValueError("hybrid decisions require the named actor input")
    if not skills or set(skills) - NAVIGATION_INSTRUCTIONS.keys():
        raise ValueError("unsupported hybrid skills")
    if not re.fullmatch(r"[0-9a-f]{64}", registry_sha256):
        raise ValueError("hybrid registry needs a SHA256 binding")
    payload = deepcopy(named_payload)
    names = ", ".join(skills)
    payload["messages"][0]["content"] = (
        "Choose one MicroDuck skill from the current task, head camera and sensor feedback. "
        'Return one compact JSON object only: {"tool":"...","duration_s":1.0,"memory":"..."}. '
        f"Available learned navigation skills: {names}. Each navigate_COLOR requests an experimental "
        "camera-based policy to find that color ball, approach without collision and stop. "
        "It receives fresh images during execution; it may fail and is not an oracle. "
        "Select the color requested by the user, not a different visible object. "
        "Other tools: advance walks forward at commanded 0.30m/s; turn_left/right turn the BODY "
        "along a forward arc (0.15m/s, +/-1.1rad/s); look_left/right turn only the HEAD while "
        "standing (+/-0.45rad); stop stands and centers the head. These are commands, not guaranteed motion. "
        "Duration is 0.1..2.0 seconds in multiples of 0.02. After this interval you observe again. "
        "Use execution feedback to decide whether to continue, search differently or stop. "
        "Image left/right is relative to the head; encoder readings are not world coordinates. "
        "Command consumption does not prove displacement, clearance or task completion. "
        "Optional memory is a brief uncertain hypothesis, at most 160 characters. "
        "Do not add velocities, explanations, Markdown or an assertion of evaluator success."
    )
    payload["metadata"].update(prompt_profile=HYBRID_PROTOCOL, tool_protocol=HYBRID_PROTOCOL,
                               tool_contract_sha256=registry_sha256,
                               primitive_contract_sha256=TOOL_CONTRACT_SHA256,
                               skill_registry_sha256=registry_sha256,
                               response_framing=JSON_OR_FENCE)
    payload["max_tokens"] = 96
    return payload


@dataclass(frozen=True)
class HybridRequest:
    generation: int
    observation: Observation
    context: dict
    previous: Observation | None
    delegates: dict


class HybridNavigationPolicy:
    def __init__(self, *, skills, endpoint, model, contract, context_mode="feedback",
                 expected_service_identity=None, validation_feedback=False, allow_empty_skills=False):
        action_contract(contract)
        if type(validation_feedback) is not bool:
            raise ValueError("validation_feedback must be a boolean")
        if validation_feedback and not re.fullmatch(r"[0-9a-f]{64}", str(expected_service_identity)):
            raise ValueError("hybrid tool correction requires a pinned service identity")
        self.validation_feedback = validation_feedback
        if type(allow_empty_skills) is not bool:
            raise ValueError("allow_empty_skills must be a boolean")
        if (not skills and not allow_empty_skills) or set(skills) - NAVIGATION_INSTRUCTIONS.keys():
            raise ValueError("hybrid v1 requires explicitly registered color-navigation skills")
        if len({id(p) for p in skills.values()}) != len(skills):
            raise ValueError("each skill needs its own task/history session; cores may be shared")
        self.skills = dict(skills)
        bindings = {}
        self.methods = {}
        for name, policy in self.skills.items():
            metadata = deepcopy(policy.metadata)
            if metadata.get("contract") != contract or not re.fullmatch(
                    r"[0-9a-f]{64}", str(metadata.get("artifact_sha256", ""))):
                raise ValueError("learned skill requires a matching contract and pinned artifact")
            bindings[name] = {"instruction": NAVIGATION_INSTRUCTIONS[name], "policy": metadata,
                              "availability": "experimental_not_independent_acceptance"}
            self.methods[name] = request_methods(policy)
        self.contract, self.context_mode = contract, context_mode
        self.client = SkillPlannerPolicy(
            SimpleNamespace(profile="official_bam"), endpoint=endpoint, model=model,
            context_mode=context_mode, navigation_contract=contract,
            expected_service_identity=expected_service_identity,
            prompt_profile="bounded_tools_named_sensors_v2", response_framing=JSON_OR_FENCE)
        self.metadata = {
            "family": "VLM+WAM", "protocol": HYBRID_PROTOCOL, "contract": contract,
            "response_framing": JSON_OR_FENCE,
            "supervisor_model": model, "context_mode": context_mode,
            "expected_service_identity": expected_service_identity,
            "learned_skills": bindings, "primitive_contract_sha256": TOOL_CONTRACT_SHA256,
            "skill_window_s": [.1, 2.], "supervisor_http_timeout_s": 5.,
            "action_freshness": "delegate_camera_source_preserved_primitive_source_never_refreshed",
            "decision_schedule": "new_supervisor_request_at_end_of_requested_skill_window",
            "task_memory": "model_hypothesis_committed_only_after_execution",
            "truth_access": False, "hardware_validated": False,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
        self.metadata["skill_registry_sha256"] = hashlib.sha256(json.dumps(
            bindings, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        if validation_feedback:
            self.metadata.update(validation_feedback_protocol=VALIDATION_FEEDBACK_PROTOCOL,
                maximum_requests_per_decision=2,
                correction_source_clock="unchanged_original_observation",
                correction_expiry="common_navigation_buffer_original_source",
                supervisor_latency_scope="complete_decision_including_correction")
        self.context = PlannerContext()
        self.lock = threading.RLock()
        self.inference_lock = threading.Lock()
        self.generation = 0
        self.closed = False
        self.reset("")

    def reset(self, episode_id):
        # Invalidate outstanding HTTP/model results before waiting for the
        # inference lane to drain. New episode admission waits for this reset.
        with self.lock:
            if self.closed:
                raise ValueError("hybrid policy is closed")
            self.generation += 1
            self.resetting = True
            self.episode_id = str(episode_id)
            self.decision = None
            self.decisions = {}
            self.sequence = 0
            self.executing = None
            self.last_fallback = None
            self.context.reset(self.episode_id)
            self.client.reset(self.episode_id)
        with self.inference_lock:
            for policy in self.skills.values():
                policy.reset(self.episode_id)
        with self.lock:
            self.resetting = False

    def _check(self, generation=None, episode_id=None):
        if (self.closed or self.resetting or (generation is not None and generation != self.generation)
                or (episode_id is not None and episode_id != self.episode_id)):
            raise ValueError("hybrid request belongs to a closed or reset session")

    def prepare_request(self, observation):
        with self.lock:
            self._check(episode_id=observation.episode_id)
            if observation.rgb is None or observation.camera_tick != observation.tick:
                raise ValueError("hybrid navigation requires a fresh measured camera frame")
            self.context.observe(observation)
            context, previous = self.context.snapshot(observation, self.context_mode)
            # All skill histories see the same sensor cadence, including when
            # a head/stop tool is running. Only the selected core is inferred.
            delegates = {
                name: self.methods[name][0](replace(observation, instruction=NAVIGATION_INSTRUCTIONS[name]))
                for name in self.skills
            }
            return HybridRequest(self.generation, observation, context, previous, delegates)

    def request_payload(self, request):
        payload = self.client.request_payload(request.observation, context=request.context, previous=request.previous)
        return with_hybrid_skills(payload, self.skills, self.metadata["skill_registry_sha256"])

    def validate_call(self, value):
        return validate_skill_request(value, self.skills)

    def primitive_action(self, call):
        return TOOL_ACTIONS[call["tool"]]

    def primitive_plan(self, call):
        return decode_tool_call(call)

    def prediction_required(self, observation):
        """Keep an already issued timed primitive without refreshing its age.

        The evaluator still advances physics, captures sensors and applies
        normal buffer expiry. A held plan can therefore expire and fall back.
        Learned navigation instead receives fresh predictions at sensor cadence.
        """
        with self.lock:
            self._check(episode_id=observation.episode_id)
            return (self.decision is None or observation.tick >= self.decision["until_tick"]
                    or self.decision["call"]["tool"] in self.skills)

    def predict_prepared(self, request):
        with self.inference_lock:
            with self.lock:
                self._check(request.generation, request.observation.episode_id)
                decision = self.decision
            observation = request.observation
            supervisor_ms = delegate_ms = None
            if decision is None or observation.tick >= decision["until_tick"]:
                started = time.monotonic()
                request_trace = None
                if self.validation_feedback:
                    def check_current():
                        with self.lock:
                            self._check(request.generation, observation.episode_id)
                    call, serving, request_trace = request_with_validation_feedback(
                        self.request_payload(request), self.client.request_raw,
                        self.validate_call,
                        check_current=check_current, response_framing=JSON_OR_FENCE)
                else:
                    raw, serving = self.client.request_json(self.request_payload(request))
                    call = self.validate_call(raw)
                supervisor_ms = (time.monotonic() - started) * 1000
                with self.lock:
                    self._check(request.generation, observation.episode_id)
                    self.sequence += 1
                    key = f"{self.generation}:{self.sequence}"
                    decision = {"key": key, "generation": self.generation, "call": call,
                                "source": observation, "serving_evidence": serving,
                                "until_tick": observation.tick + round(call["duration_s"] * 50)}
                    if self.validation_feedback:
                        decision["request_trace"] = request_trace
                    self.decision = self.decisions[key] = decision
                    while len(self.decisions) > 8:
                        oldest = next(k for k in self.decisions if k not in (key, self.executing))
                        del self.decisions[oldest]
            tool = decision["call"]["tool"]
            if tool in self.skills:
                started = time.monotonic()
                chunk = self.methods[tool][1](request.delegates[tool])
                delegate_ms = (time.monotonic() - started) * 1000
                if (not isinstance(chunk, NavigationChunk) or chunk.contract != self.contract
                        or chunk.episode_id != observation.episode_id
                        or chunk.source_tick != observation.tick
                        or chunk.source_monotonic_ns != observation.monotonic_ns):
                    raise ValueError("delegated action does not match its actual source observation")
                chunk = replace(chunk, valid_until_tick=min(chunk.valid_until_tick, decision["until_tick"]),
                                diagnostics=deepcopy(chunk.diagnostics))
            else:
                source = decision["source"]
                ticks = decision["until_tick"] - source.tick
                actions = np.repeat(np.asarray(self.primitive_action(decision["call"]), np.float32)[None],
                                    math.ceil(ticks / DECISION_TICKS), axis=0)
                chunk = NavigationChunk(actions, source.episode_id, source.tick, source.monotonic_ns,
                                        decision["until_tick"], contract=self.contract)
            with self.lock:
                self._check(request.generation, observation.episode_id)
            chunk.diagnostics["hybrid"] = {
                "protocol": self.metadata["protocol"], "generation": request.generation,
                "decision_key": decision["key"], "call": deepcopy(decision["call"]),
                "supervisor_source_tick": decision["source"].tick,
                "supervisor_camera_tick": camera_tick(decision["source"]),
                "supervisor_monotonic_ns": decision["source"].monotonic_ns,
                "supervisor_valid_until_tick": decision["until_tick"],
                "serving_evidence": deepcopy(decision["serving_evidence"]),
                "execution_kind": "learned_navigation" if tool in self.skills else "official_primitive",
                "skill_registry_sha256": self.metadata["skill_registry_sha256"],
                "delegate_instruction": NAVIGATION_INSTRUCTIONS.get(tool),
                "delegate_policy": deepcopy(self.skills[tool].metadata) if tool in self.skills else None,
                "supervisor_requested_this_prediction": supervisor_ms is not None,
                "supervisor_http_elapsed_ms": supervisor_ms,
                "delegate_inference_elapsed_ms": delegate_ms,
            }
            if self.validation_feedback:
                chunk.diagnostics["hybrid"]["request_trace"] = deepcopy(decision["request_trace"])
            return chunk

    def predict(self, observation):
        return self.predict_prepared(self.prepare_request(observation))

    def observe_execution(self, before, after, intent, evidence, chunk):
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
            record = chunk.diagnostics.get("hybrid", {})
            if record.get("generation") != self.generation:
                raise ValueError("executor used an old hybrid session")
            decision = self.decisions.get(record.get("decision_key"))
            if decision is None or record["call"] != decision["call"]:
                raise ValueError("executed hybrid action lacks its original supervisor decision")
            if decision["key"] != self.executing:
                call, source = decision["call"], decision["source"]
                if call["tool"] in self.skills:
                    self.context.finish("superseded", before.tick)
                    self.context.reference = source
                    self.context.active = {
                        "skill": call["tool"], "execution_kind": "learned_navigation",
                        "requested_duration_s": call["duration_s"],
                        "source_tick": source.tick, "source_camera_tick": camera_tick(source),
                        "started_tick": before.tick, "last_observed_tick": before.tick,
                        "status": "running", "issued_ticks": 0,
                        "outcome": "not_inferred_from_command_or_image_change",
                    }
                    if "memory" in call:
                        self.context.memory = {"text": call["memory"], "source_tick": source.tick,
                                               "provenance": "model_hypothesis"}
                else:
                    self.context.begin(self.primitive_plan(call), source, before)
                    self.context.reference = source
                self.context.active.update(supervisor_tool_call=deepcopy(call),
                    receipt_semantics="executor_consumed_intent_not_verified_motion",
                    skill_registry_sha256=self.metadata["skill_registry_sha256"])
                self.executing = decision["key"]
            self.last_fallback = None
            active = self.context.active
            active["issued_ticks"] += 1
            active["executor_consumed_ticks"] = active.get("executor_consumed_ticks", 0) + 1
            active["last_executed_command13"] = intent.command.tolist()
            active["last_delegate_source_tick"] = chunk.source_tick
            active["last_joint_action_l2"] = float(np.linalg.norm(after.state[34:48]))
            self.context.observe(after)

    def close(self):
        with self.lock:
            self.closed = True
            self.generation += 1
            self.decision = None
            self.decisions.clear()
            self.client.close()
        with self.inference_lock:
            for policy in self.skills.values():
                close = getattr(policy, "close", None)
                if close is not None:
                    close()
