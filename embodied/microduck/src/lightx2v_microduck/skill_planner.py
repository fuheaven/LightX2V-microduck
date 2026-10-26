"""quackd-inspired VLM skill planning over our physical locomotion executor.

Only RGB, instruction and proprioception reach the model. Skills cannot set
object poses or fake successful contact. No upstream simulator is imported.
"""

import base64
import hashlib
import io
import json
import math
import os
import queue
import re
import threading
import time
import urllib.request
import uuid
from dataclasses import dataclass
from urllib.parse import urlparse

import numpy as np
from PIL import Image

from .contracts import JointResidualChunk, Observation, PolicyManifest
from .planner_context import CONTEXT_MODES, CONTEXT_VERSION, PlannerContext, camera_tick, capture_monotonic_ns
from .navigation_action_contracts import action_contract
from .planner_prompts import PROMPT_PROFILES, present_task
from .planner_tools import ACTOR_TOOL_PROFILES, DECISION_TOOL_PROFILES, MEASURED_DECISION_PROFILE
from .planner_response import JSON_ONLY, JSON_OR_FENCE, decode_framed_object

ALLOWED_SKILLS = {"walk", "stand", "kick_left", "kick_right", "roulade", "ground_pick"}
HTTP_REQUEST_TIMEOUT_S = 5.0
HTTP_CANCEL_TIMEOUT_S = 2.0


def validate_plan(value, *, navigation_contract=None):
    if not isinstance(value, dict) or value.get("skill") not in ALLOWED_SKILLS:
        raise ValueError("planner must select a supported skill")
    command = np.asarray(value.get("command", [0.0, 0.0, 0.0]), dtype=np.float32)
    duration = float(value.get("duration_s", 0.5))
    if command.shape != (3,) or not np.isfinite(command).all() or not math.isfinite(duration) or not 0.1 <= duration <= 2.0:
        raise ValueError("invalid planner command/duration")
    if np.any(np.abs(command) > np.asarray([0.4, 0.2, 1.1], dtype=np.float32)):
        raise ValueError("planner command exceeds locomotion envelope")
    if value["skill"] != "walk" and np.any(command):
        raise ValueError("non-walk skills require zero velocity command")
    head_yaw = float(value.get("head_yaw_rad", 0.0))
    if not math.isfinite(head_yaw) or abs(head_yaw) > 0.45:
        raise ValueError("head yaw must be finite within +/-0.45 radians")
    result = {"skill": value["skill"], "command": command.tolist(), "duration_s": duration,
              "head_yaw_rad": head_yaw, "reason": str(value.get("reason", ""))[:1000]}
    if "memory" in value:
        if not isinstance(value["memory"], str) or len(value["memory"]) > 1000:
            raise ValueError("planner memory must be a string of at most 1000 characters")
        result["memory"] = value["memory"]
    if "tool_call" in value:
        from .planner_tools import decode_tool_call

        translated = decode_tool_call(value["tool_call"])
        if (any(value.get(key) != translated[key] for key in ("tool_protocol", "tool_contract_sha256"))
                or result["skill"] != translated["skill"]
                or not np.array_equal(command, np.asarray(translated["command"], np.float32))
                or result["head_yaw_rad"] != translated["head_yaw_rad"]
                or result["duration_s"] != translated["duration_s"]
                or result.get("memory") != translated.get("memory")):
            raise ValueError("translated navigation tool differs from its recorded call")
        result.update({key: translated[key] for key in
                       ("tool_call", "tool_protocol", "tool_contract_sha256")})
    if "decision_call" in value:
        from .planner_decisions import decode_decision_call

        decision = decode_decision_call(value["decision_call"])
        if any(value.get(key) != decision[key] for key in
               ("tool_call", "decision_protocol", "model_assessment")):
            raise ValueError("model assessment differs from the recorded visual decision")
        result.update({key: decision[key] for key in
                       ("decision_call", "decision_protocol", "model_assessment")})
    if navigation_contract is not None:
        if result["skill"] not in ("walk", "stand"):
            raise ValueError("shared navigation supports only walk and stand")
        action_contract(navigation_contract).validate(
            [*result["command"], result["head_yaw_rad"], float(result["skill"] == "stand")])
    return result


@dataclass(frozen=True)
class PlannerRequest:
    generation: int
    sequence: int
    observation: Observation
    context: dict
    previous: Observation | None


@dataclass(frozen=True)
class PlannerReply:
    request: PlannerRequest
    started: float
    latency_ms: float
    plan: dict | None
    error: str | None
    serving_evidence: dict | None = None
    request_trace: list | None = None


class SkillPlannerPolicy:
    def __init__(self, env, *, endpoint, model, context_mode="feedback", navigation_contract=None,
                 expected_service_identity=None, prompt_profile="legacy_v1", response_framing=JSON_ONLY,
                 validation_feedback=False):
        if not endpoint or urlparse(endpoint).scheme not in ("http", "https") or not model:
            raise ValueError("skill planner requires --endpoint and --planner-model; no scripted substitute")
        self.env, self.endpoint, self.model = env, endpoint.rstrip("/"), model
        if context_mode not in CONTEXT_MODES:
            raise ValueError("unknown planner context mode")
        self.context_mode = context_mode
        if prompt_profile not in PROMPT_PROFILES:
            raise ValueError("unknown planner prompt profile")
        self.prompt_profile = prompt_profile
        if prompt_profile == MEASURED_DECISION_PROFILE and context_mode != "feedback":
            raise ValueError("measured heading profile requires feedback context")
        if response_framing not in (JSON_ONLY, JSON_OR_FENCE):
            raise ValueError("unknown planner response framing")
        self.response_framing = response_framing
        if prompt_profile in ACTOR_TOOL_PROFILES and env.profile != "official_bam":
            raise ValueError("bounded navigation tools require official_bam")
        if prompt_profile == "calibrated_tools_v3":
            from .navigation_tool_response import load_tool_response

            load_tool_response(env.profile)
        self.navigation_contract = navigation_contract
        if expected_service_identity is not None and not re.fullmatch(r"[0-9a-f]{64}", expected_service_identity):
            raise ValueError("expected service identity must be a SHA256 digest")
        self.expected_service_identity = expected_service_identity
        if type(validation_feedback) is not bool or (validation_feedback and expected_service_identity is None):
            raise ValueError("validation feedback requires an explicit flag and pinned service identity")
        self.validation_feedback = validation_feedback
        if navigation_contract is not None:
            action_contract(navigation_contract)
        self.manifest = PolicyManifest(
            "vlm_skill_planner",
            "joint_residual",
            "rgb_proprio",
            env.profile,
            metadata={
                "model": model,
                "architecture": "RGB_VLM_to_skills_to_official_ONNX",
                "inspiration": "rokbenko/quackd",
                "required_sensors": ["rgb", "proprioception"],
                "max_source_age_s": 2.0,
                "max_skill_duration_s": 2.0,
                "expiry_applies_during_execution": True,
                "truth_access": False,
                "context_version": CONTEXT_VERSION,
                "context_mode": context_mode,
                "prompt_profile": prompt_profile,
                "response_framing": response_framing,
                "state_input": "proprio48_without_command13",
                "history_limit": 4 if context_mode == "feedback" else 0,
                "memory_provenance": "model_hypothesis_not_evaluator_truth",
                "head_yaw_limit_rad": 0.45,
                "velocity_abs_limits_mps_rps": [0.4, 0.2, 1.1],
                "allowed_skills": ["stand", "walk"] if navigation_contract else sorted(ALLOWED_SKILLS),
                "navigation_contract": navigation_contract,
                "navigation_bounds": ({"lower": list(action_contract(navigation_contract).lower),
                                       "upper": list(action_contract(navigation_contract).upper)}
                                      if navigation_contract else None),
                "service_evidence": "optional_request_bound_identity_digest",
                "expected_service_identity": expected_service_identity,
                "policy_source_version": 6,
                "pinned_service_lifecycle": "cancel_inflight_on_timeout_reset_close",
            },
        )
        self.requests, self.replies = queue.Queue(maxsize=1), queue.Queue(maxsize=1)
        if prompt_profile in ACTOR_TOOL_PROFILES:
            from .planner_tools import TOOL_CONTRACT, TOOL_CONTRACT_SHA256

            self.manifest.metadata.update(tool_contract=TOOL_CONTRACT,
                                          tool_contract_sha256=TOOL_CONTRACT_SHA256,
                                          allowed_skills=["stand", "walk"],
                                          output_protocol="typed_tool_call_to_official_commands")
        if prompt_profile in ("bounded_tools_named_sensors_v2", *DECISION_TOOL_PROFILES):
            from .planner_sensors import NAMED_SENSOR_CONTRACT_SHA256, NAMED_SENSOR_VERSION

            self.manifest.metadata.update(named_sensor_version=NAMED_SENSOR_VERSION,
                                          named_sensor_contract_sha256=NAMED_SENSOR_CONTRACT_SHA256)
        if prompt_profile in DECISION_TOOL_PROFILES:
            from .planner_decisions import DECISION_PROTOCOL

            self.manifest.metadata.update(decision_protocol=DECISION_PROTOCOL,
                                          assessment_provenance="model_hypothesis_not_verified")
        if prompt_profile in ("perception_action_compact_v4", MEASURED_DECISION_PROFILE):
            from .planner_decisions import COMPACT_LAYOUT_ID

            self.manifest.metadata.update(output_layout=COMPACT_LAYOUT_ID,
                                          max_output_tokens=96,
                                          memory_word_limit_enforcement="prompt_only",
                                          output_repair=False)
        if prompt_profile == MEASURED_DECISION_PROFILE:
            from .planner_motion import MOTION_CONTRACT_SHA256, MOTION_VERSION

            self.manifest.metadata.update(motion_feedback_version=MOTION_VERSION,
                                          motion_feedback_contract_sha256=MOTION_CONTRACT_SHA256)
        if validation_feedback:
            from .planner_validation_feedback import VALIDATION_FEEDBACK_PROTOCOL

            self.manifest.metadata.update(validation_feedback_protocol=VALIDATION_FEEDBACK_PROTOCOL,
                                          maximum_requests_per_decision=2,
                                          correction_source_clock="unchanged_original_observation",
                                          policy_source_version=7)
        self.stopped = threading.Event()
        self.realtime = False
        self.worker = None
        self.context = PlannerContext(measured_heading=prompt_profile == MEASURED_DECISION_PROFILE)
        self._http_lock = threading.Lock()
        self._http_inflight = set()
        self._generation = 0
        self._sequence = 0
        self.reset("")

    def reset(self, episode_id):
        self._generation += 1
        self._cancel_http_requests()
        self.episode_id = episode_id
        self.plan = None
        self.expires = -1
        self.last_request_tick = -1000
        self.error = None
        self.last_diagnostics = {}
        self._pending_request = None
        self._last_accepted_tick = -1
        self._last_request_latency_ms = None
        self._last_serving_evidence = None
        self._last_request_trace = None
        self._last_plan_source = None
        self._plan_wall_origin_ns = 0
        self.context.reset(episode_id)
        # Cancellation is cooperative; generation checks still reject late
        # replies even when the server cannot stop the current kernel promptly.
        for channel in (self.requests, self.replies):
            while True:
                try:
                    channel.get_nowait()
                except queue.Empty:
                    break

    def set_realtime(self, enabled):
        self.realtime = enabled
        if enabled and self.worker is None:
            self.worker = threading.Thread(target=self._worker, daemon=True, name="microduck-skill-planner")
            self.worker.start()

    def request_payload(self, observation, *, context=None, previous=None):
        if observation.rgb is None:
            raise ValueError("VLM skill planning requires RGB")
        if context is None:
            compiler = self.context
            if observation.episode_id != self.episode_id:
                compiler = PlannerContext(measured_heading=self.prompt_profile == MEASURED_DECISION_PROFILE)
                compiler.reset(observation.episode_id)
            context, previous = compiler.snapshot(observation, self.context_mode)
        def camera_content(obs, label):
            buffer = io.BytesIO()
            Image.fromarray(obs.rgb).save(buffer, format="JPEG")
            return [{"type": "text", "text": f"{label}; camera tick {camera_tick(obs)}"},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()}}]
        prompt = (
            "Choose the next MicroDuck skill from the camera and instruction. Return a JSON object only: "
            '{"skill":"walk|stand|kick_left|kick_right|roulade|ground_pick",'
            '"command":[vx,vy,yaw_rate],"head_yaw_rad":0.0,"duration_s":0.5,"reason":"...","memory":"..."}. '
            "Limits: abs(vx)<=0.4, abs(vy)<=0.2, abs(yaw_rate)<=1.1, duration 0.1..2 seconds. "
            "Velocities are body-frame commands in m/s and rad/s, not guaranteed displacement. "
            "Positive vx is forward, positive vy is left, positive yaw turns left. "
            "Head yaw is radians within [-0.45,0.45], positive left; it can be used while standing. "
            "Non-walk velocity commands must be zeros. You do not know world coordinates. Stand if uncertain. "
            "History confirms raw commands via the next sensor sample, not successful movement or task completion. "
            "Image change is not a collision or success detector. Compare images and feedback before repeating a failed attempt. "
            "Optional memory (at most 1000 characters) is your uncertain task hypothesis for the next decision, not verified truth. "
            "Only the independent evaluator scores success."
        )
        if self.navigation_contract is not None:
            contract = action_contract(self.navigation_contract)
            prompt = prompt.replace("walk|stand|kick_left|kick_right|roulade|ground_pick", "walk|stand")
            prompt = prompt.replace(
                "Limits: abs(vx)<=0.4, abs(vy)<=0.2, abs(yaw_rate)<=1.1, duration 0.1..2 seconds. ",
                f"Limits: {contract.lower[0]}<=vx<={contract.upper[0]}, "
                f"{contract.lower[1]}<=vy<={contract.upper[1]}, abs(yaw_rate)<=1.1, duration 0.1..2 seconds. ")
            prompt = prompt.replace(
                "History confirms raw commands via the next sensor sample, not successful movement or task completion. ",
                "History records high-level intents consumed by the shared executor. This receipt alone does not "
                "verify joint commands, movement or task completion; inspect the sensor samples and images. ")
            prompt += " This comparison permits only walk and stand; other skills are unavailable."
        return present_task({
            "model": self.model,
            "temperature": 0,
            "max_tokens": 512,
            "metadata": {"episode_id": str(observation.episode_id), "source_tick": str(observation.tick),
                         "camera_tick": str(camera_tick(observation)), "context_version": CONTEXT_VERSION},
            "messages": [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": [{"type": "text", "text": json.dumps(context, allow_nan=False)}]
                    + (camera_content(previous, "Before the most recent accepted action") if previous is not None else [])
                    + camera_content(observation, "Current observation"),
                },
            ],
        }, self.prompt_profile)
    def request_plan(self, observation, *, context=None, previous=None):
        payload = self.request_payload(observation, context=context, previous=previous)
        def validator(value):
            if self.prompt_profile in ACTOR_TOOL_PROFILES:
                from .planner_tools import decode_actor_reply

                value = decode_actor_reply(value, self.prompt_profile)
            return validate_plan(value, navigation_contract=self.navigation_contract)

        if self.validation_feedback:
            from .planner_validation_feedback import request_with_validation_feedback

            generation = self._generation
            def check_current():
                if (self.stopped.is_set() or generation != self._generation
                        or observation.episode_id != self.episode_id):
                    raise ValueError("planner correction belongs to a closed or reset session")
                if self.realtime:
                    captured_ns = capture_monotonic_ns(observation)
                    if not captured_ns or not 0 <= (time.monotonic_ns() - captured_ns) / 1e9 <= 2.0:
                        raise ValueError("planner correction source observation expired")

            plan, evidence, trace = request_with_validation_feedback(
                payload, self.request_raw, validator, check_current=check_current,
                response_framing=self.response_framing)
            plan["_request_trace"] = trace
        else:
            value, evidence = self.request_json(payload)
            plan = validator(value)
        if evidence is not None:
            plan["_serving_evidence"] = evidence
        return plan

    def request_json(self, payload):
        content, evidence = self.request_raw(payload)
        if self.response_framing == JSON_OR_FENCE:
            value, framing = decode_framed_object(content)
            if evidence is not None:
                evidence.update(framing)
        else:
            value = json.loads(content)
        return value, evidence

    def request_raw(self, payload):
        """Identity-checked transport shared by typed robot tool interpreters.

        The caller must validate the returned value against its own action
        protocol. This method neither selects tools nor repairs model output.
        """
        if payload.get("model") != self.model:
            raise ValueError("request model differs from the pinned planner model")
        identifier = uuid.uuid4().hex
        headers = {"Content-Type": "application/json", "X-Microduck-Request-Id": identifier}
        key = os.environ.get("MICRODUCK_PLANNER_API_KEY")
        if key:
            headers["Authorization"] = "Bearer " + key
        url = self.endpoint if self.endpoint.endswith("/chat/completions") else self.endpoint + "/chat/completions"
        request = urllib.request.Request(url, json.dumps(payload).encode(), headers, method="POST")
        with self._http_lock:
            if self.stopped.is_set():
                raise RuntimeError("skill planner is closed")
            self._http_inflight.add(identifier)
        try:
            with urllib.request.urlopen(request, timeout=HTTP_REQUEST_TIMEOUT_S) as response:
                result = json.loads(response.read(1024 * 1024))
        except OSError as error:
            from .planner_transport_evidence import capture_transport_error

            error.planner_transport_error = capture_transport_error(
                error, payload, identifier, self.expected_service_identity)
            # Preserve the original failure and never retry the action. Only
            # our identity-pinned server promises this cancellation endpoint.
            self._cancel_http_requests([identifier])
            raise
        finally:
            with self._http_lock:
                self._http_inflight.discard(identifier)
        evidence = result.get("microduck_evidence")
        if self.expected_service_identity is not None and evidence is None:
            raise ValueError("pinned model service omitted its identity evidence")
        if evidence is not None:
            digest = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                               separators=(",", ":"), allow_nan=False).encode()).hexdigest()
            if (not isinstance(evidence, dict) or evidence.get("request_id") != identifier
                    or evidence.get("request_sha256") != digest or evidence.get("model_id") != self.model
                    or not re.fullmatch(r"[0-9a-f]{64}", str(evidence.get("identity_sha256", "")))):
                raise ValueError("model service evidence does not match this request")
            if self.expected_service_identity is not None and evidence["identity_sha256"] != self.expected_service_identity:
                raise ValueError("model service identity changed during the episode")
            evidence = {key: evidence[key] for key in
                        ("request_id", "request_sha256", "identity_sha256", "model_id")}
        content = result["choices"][0]["message"]["content"]
        return content, evidence

    def _cancel_http_requests(self, identifiers=None):
        if self.expected_service_identity is None:
            return
        if identifiers is None:
            with self._http_lock:
                identifiers = list(self._http_inflight)
        base = self.endpoint.removesuffix("/chat/completions")
        for identifier in identifiers:
            headers = {"Content-Type": "application/json"}
            key = os.environ.get("MICRODUCK_PLANNER_API_KEY")
            if key:
                headers["Authorization"] = "Bearer " + key
            request = urllib.request.Request(base + f"/requests/{identifier}/cancel", b"{}", headers, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=HTTP_CANCEL_TIMEOUT_S) as response:
                    response.read(4096)
            except OSError:
                # A missing or disconnected server cannot prove cancellation.
                # Episode admission and shutdown independently check quiescence.
                pass

    def _worker(self):
        while not self.stopped.is_set():
            try:
                request = self.requests.get(timeout=0.1)
            except queue.Empty:
                continue
            if request.generation != self._generation:
                continue
            result = self._infer(request)
            if not self.stopped.is_set() and request.generation == self._generation:
                while not self.stopped.is_set():
                    try:
                        self.replies.put_nowait(result)
                        break
                    except queue.Full:
                        try:
                            self.replies.get_nowait()
                        except queue.Empty:
                            pass

    def _new_request(self, observation):
        self._sequence += 1
        context, previous = self.context.snapshot(observation, self.context_mode)
        request = PlannerRequest(self._generation, self._sequence, observation, context, previous)
        self._pending_request = (request.generation, request.sequence)
        self.last_request_tick = observation.tick
        return request

    def _infer(self, request):
        started = time.monotonic()
        evidence = trace = None
        try:
            plan = self.request_plan(request.observation, context=request.context, previous=request.previous)
            if isinstance(plan, dict):
                plan = dict(plan)
                evidence = plan.pop("_serving_evidence", None)
                trace = plan.pop("_request_trace", None)
            error = None
        except Exception as exc:
            trace = getattr(exc, "planner_request_trace", None)
            plan, error = None, f"{type(exc).__name__}: planner request failed"
        return PlannerReply(request, started, (time.monotonic() - started) * 1000, plan, error, evidence, trace)

    def _accept(self, reply, observation):
        request, source = reply.request, reply.request.observation
        if ((request.generation, request.sequence) != self._pending_request
                or request.generation != self._generation or source.episode_id != self.episode_id):
            return  # A late reply must not poison a new episode's memory/error.
        self._pending_request = None
        self._last_request_latency_ms = reply.latency_ms
        self._last_serving_evidence = reply.serving_evidence
        self._last_request_trace = reply.request_trace
        captured_ns = capture_monotonic_ns(source)
        wall_age = (time.monotonic_ns() - captured_ns) / 1e9 if captured_ns else time.monotonic() - reply.started
        if source.tick <= self._last_accepted_tick or source.tick > observation.tick:
            error = "out_of_order_planner_reply"
        elif self.realtime and (observation.tick - camera_tick(source) > 100 or wall_age > 2.0 or wall_age < 0):
            error = "expired_planner_reply"
        else:
            error = reply.error
        if error:
            self.error = error
            self.context.failure(error, observation.tick, source.tick)
            self.context.finish("interrupted_by_request_error", observation.tick)
            return
        try:
            plan = validate_plan(reply.plan, navigation_contract=self.navigation_contract)
        except (ValueError, TypeError, OverflowError):
            self.error = "invalid_planner_reply"
            self.context.failure(self.error, observation.tick, source.tick)
            self.context.finish("interrupted_by_request_error", observation.tick)
            return
        self.plan, self.expires, self.error = plan, observation.tick + round(plan["duration_s"] * 50), None
        self._last_accepted_tick = source.tick
        self._last_plan_source = {"tick": source.tick, "camera_tick": camera_tick(source), "capture_monotonic_ns": captured_ns}
        self._plan_wall_origin_ns = captured_ns or int(reply.started * 1e9)
        self.context.begin(plan, source, observation)

    def predict(self, observation):
        if self.stopped.is_set():
            raise RuntimeError("skill planner is closed")
        if observation.episode_id != self.episode_id:
            self.reset(observation.episode_id)
        self.context.observe(observation)
        if self.plan is not None and observation.tick >= self.expires:
            self.context.finish("command_window_elapsed", observation.tick)
        if self.realtime:
            while True:
                try:
                    self._accept(self.replies.get_nowait(), observation)
                except queue.Empty:
                    break
            if observation.tick - self.last_request_tick >= 25 and self._pending_request is None:
                self.requests.put_nowait(self._new_request(observation))
        elif observation.tick >= self.expires:
            self._accept(self._infer(self._new_request(observation)), observation)
            if self.error:
                self.expires = observation.tick + 25
        active = self.plan is not None and observation.tick < self.expires and self.error is None
        if active and self.realtime:
            wall_age = (time.monotonic_ns() - self._plan_wall_origin_ns) / 1e9
            if observation.tick - self._last_plan_source["camera_tick"] > 100 or not 0 <= wall_age <= 2.0:
                active, self.error = False, "expired_active_plan"
                self.context.failure(self.error, observation.tick, self._last_plan_source["tick"])
                self.context.finish("interrupted_by_source_expiry", observation.tick)
        role = self.plan["skill"] if active else "stand"
        command = np.zeros(13, np.float32)
        if active:
            command[:3] = self.plan["command"]
            command[5] = self.plan["head_yaw_rad"]
        if role == "ground_pick":
            elapsed = self.plan["duration_s"] - max(0, self.expires - observation.tick) * 0.02
            command[:2] = (math.cos(2 * math.pi * elapsed / 3.0), math.sin(2 * math.pi * elapsed / 3.0))
        state = observation.state.copy()
        state[48:] = command
        raw = self.env.bank.infer(role, state)
        if active:
            self.context.issued(observation, raw)
        self.last_diagnostics = {
            "planner_model": self.model,
            "plan": self.plan,
            "active_skill": role,
            "source": "VLM_plan" if active else "local_stand",
            "error": self.error,
            "intervention": bool(self.error or (self.plan is not None and not active)),
            "uses_privileged_state": False,
            "context_version": CONTEXT_VERSION,
            "context_mode": self.context_mode,
            "plan_source": self._last_plan_source,
            "planner_latency_ms": self._last_request_latency_ms,
            "request_sequence": self._sequence,
            "last_request_tick": self.last_request_tick,
            "serving_evidence": self._last_serving_evidence,
            "command13": command.tolist(),
            "execution_feedback": dict(self.context.active) if self.context.active is not None else None,
            "task_memory": dict(self.context.memory) if self.context_mode == "feedback" else None,
        }
        if self.validation_feedback:
            self.last_diagnostics["request_trace"] = self._last_request_trace
        return JointResidualChunk(raw[None], observation.episode_id, observation.tick, observation.sim_time_s, diagnostics=self.last_diagnostics)

    def close(self):
        self.stopped.set()
        self._generation += 1
        self._cancel_http_requests()
        self._pending_request = None
        self.plan = None
