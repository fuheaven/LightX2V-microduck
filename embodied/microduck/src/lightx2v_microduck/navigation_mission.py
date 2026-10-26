"""Model-declared task progress, isolated from physical scoring.

The policy receives the user's full instruction and its own executed claims.
The expected target sequence belongs only to the evaluator. Neither physical
success nor that sequence selects a skill or supplies the next goal to a model.
The first execution mode is offline; it does not establish realtime autonomy.
"""
from copy import deepcopy
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import time
import uuid

from .env import MicroDuckEnv
from .navigation import intent_from_action
from .navigation_eval import run_navigation_episode
from .navigation_execution_summary import ExecutedCommandSummary, EXECUTION_SUMMARY_PROTOCOL
from .navigation_hybrid import HybridNavigationPolicy, validate_skill_request, with_hybrid_skills
from .navigation_mission_timing import primitive_timing_contract, validate_primitive_timing, with_primitive_timing
from .navigation_outcomes import read, sha
from .navigation_realtime_labeling import _same_tree
from .navigation_task_switch import _hazards, _motion_state
from .planner_context import camera_tick
from .planner_tools import TOOL_ACTIONS, decode_tool_call
from .tasks import get_task
from .wam_data import write_json


MISSION_PROTOCOL = "microduck_model_declared_mission_v1"
CLAIM_TOOLS = ("complete_subgoal", "finish_mission")
COLORS = ("red", "blue", "yellow")
CLAIM_DURATION_S = .1
MISSION_ACTION_MAX_AGE_MS = 500


def validate_mission_request(value, available):
    if not isinstance(value, dict) or value.get("tool") not in CLAIM_TOOLS:
        return validate_skill_request(value, available)
    if (set(value) - {"tool", "target", "duration_s", "memory"}
            or value.get("target") not in COLORS
            or isinstance(value.get("duration_s"), bool)
            or value.get("duration_s") != CLAIM_DURATION_S):
        raise ValueError("completion requires target red/blue/yellow and duration_s=0.1")
    decode_tool_call({k: v for k, v in {**value, "tool": "stop"}.items() if k != "target"})
    return deepcopy(value)


@dataclass(frozen=True)
class MissionSpec:
    instruction: str
    expected_targets: tuple[str, ...]
    max_claims: int = 4
    max_ticks_per_claim: int = 1500

    def __post_init__(self):
        if (not isinstance(self.instruction, str) or not self.instruction.strip()
                or len(self.instruction) > 2048 or not isinstance(self.expected_targets, tuple)
                or not 1 <= len(self.expected_targets) <= 4
                or any(t not in COLORS for t in self.expected_targets)
                or type(self.max_claims) is not int or not len(self.expected_targets) <= self.max_claims <= 4
                or type(self.max_ticks_per_claim) is not int or not 5 <= self.max_ticks_per_claim <= 1500):
            raise ValueError("invalid bounded mission specification")


class MissionMemory:
    """Only actor claims and their execution provenance, never evaluator truth."""
    def __init__(self):
        self.reset("", "")

    def reset(self, mission_id, instruction):
        self.mission_id, self.instruction = mission_id, instruction
        self.claims = []

    def append(self, claim):
        if not self.mission_id or len(self.claims) >= 4:
            raise ValueError("no active mission or claim budget exhausted")
        if claim["mission_id"] != self.mission_id or claim["target"] not in COLORS:
            raise ValueError("claim belongs to another mission")
        if any(c["decision_key"] == claim["decision_key"] for c in self.claims):
            raise ValueError("completion claim was already committed")
        self.claims.append(deepcopy(claim))

    def snapshot(self):
        return dict(schema=MISSION_PROTOCOL, mission_id=self.mission_id,
                    instruction=self.instruction, claims=deepcopy(self.claims),
                    provenance="model_declarations_after_command_execution_not_verified_success")


class MissionNavigationPolicy(HybridNavigationPolicy):
    """VLM tool controller with optional learned skills and explicit completion."""
    def __init__(self, *, execution_summary=False, primitive_max_duration_ms=None, **kwargs):
        if type(execution_summary) is not bool:
            raise ValueError("execution_summary must be a boolean")
        self.execution_summary = ExecutedCommandSummary() if execution_summary else None
        self.primitive_timing = (None if primitive_max_duration_ms is None
                                 else primitive_timing_contract(primitive_max_duration_ms))
        self.mission_memory = MissionMemory()
        self._completion_claim = None
        super().__init__(**kwargs, allow_empty_skills=True)
        self.metadata.update(protocol=MISSION_PROTOCOL,
            family="VLM+WAM" if self.skills else "VLM",
            mission_memory="executed_model_claims_preserved_across_subtask_sessions",
            completion="model_declared_after_5_consumed_stand_ticks_independently_scored",
            expected_target_sequence_visible=False,
            mission_source_sha256=sha(__file__))
        self.metadata["skill_registry_sha256"] = hashlib.sha256(json.dumps(dict(
            learned=self.metadata["learned_skills"], primitive=self.metadata["primitive_contract_sha256"],
            protocol=MISSION_PROTOCOL, claim_tools=CLAIM_TOOLS, claim_duration_s=CLAIM_DURATION_S),
            sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if self.execution_summary is not None:
            self.metadata["execution_summary_protocol"] = EXECUTION_SUMMARY_PROTOCOL
        if self.primitive_timing is not None:
            self.metadata["primitive_timing_contract"] = deepcopy(self.primitive_timing)
            self.metadata["skill_registry_sha256"] = hashlib.sha256(json.dumps(dict(
                base_registry=self.metadata["skill_registry_sha256"], primitive_timing=self.primitive_timing),
                sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def begin_mission(self, mission_id, instruction):
        if not isinstance(mission_id, str) or not mission_id or not isinstance(instruction, str) or not instruction:
            raise ValueError("mission identity and original user instruction are required")
        # Reset also invalidates prepared/in-flight requests. Starting a new
        # mission explicitly discards all earlier task claims.
        self.reset(mission_id + "/initializing")
        with self.lock:
            self.mission_memory.reset(mission_id, instruction)

    def reset(self, episode_id):
        super().reset(episode_id)
        with self.lock:
            self._completion_claim = None
            if self.execution_summary is not None:
                self.execution_summary.reset(episode_id)

    def prepare_request(self, observation):
        with self.lock:
            if (not self.mission_memory.mission_id
                    or observation.instruction != self.mission_memory.instruction):
                raise ValueError("mission policy requires the original full instruction")
            request = super().prepare_request(observation)
            context = deepcopy(request.context)
            context["mission_memory"] = self.mission_memory.snapshot()
            if self.execution_summary is not None:
                context["execution_summary"] = self.execution_summary.snapshot()
            return replace(request, context=context)

    def request_payload(self, request):
        payload = self.client.request_payload(request.observation, context=request.context, previous=request.previous)
        if self.skills:
            payload = with_hybrid_skills(payload, self.skills, self.metadata["skill_registry_sha256"])
        payload["messages"][0]["content"] += (
            " This is a complete multi-step user mission. Choose the next action yourself from the full "
            "instruction and your prior declarations; no evaluator will tell you what to do next. "
            "Two additional tools: complete_subgoal declares that you have reached the named target "
            "and stopped 0.25..0.40m from its surface for at least two seconds, and that more requested "
            "steps remain; finish_mission declares that same target completion and that the whole user "
            "mission is finished. For either return exactly tool, target (red/blue/yellow), "
            "duration_s=0.1, and optional memory of at most 160 characters. Both consume a short stand "
            "command; they do not check your claim or move you to the target. A navigation skill stopping "
            "or a command being consumed is not proof of goal completion. Use images and sensor evidence; "
            "if unsure, observe again. Prior claims are your unverified declarations, not ground truth. "
            "For ordinary movement/navigation tools omit target and follow their existing duration rules."
        )
        # The regular named-sensor compiler intentionally accepts only its own
        # fields. Explicitly attach this immutable actor-only ledger as text.
        payload["messages"][-1]["content"].append(dict(type="text", text="Mission memory: " + json.dumps(
            request.context["mission_memory"], sort_keys=True, separators=(",", ":"))))
        payload["metadata"].update(prompt_profile=MISSION_PROTOCOL, tool_protocol=MISSION_PROTOCOL,
            tool_contract_sha256=self.metadata["skill_registry_sha256"],
            mission_family="hybrid" if self.skills else "vlm",
            mission_id=request.context["mission_memory"]["mission_id"])
        payload["max_tokens"] = 128
        if self.execution_summary is not None:
            payload["messages"][0]["content"] += (
                " execution_summary reports actual consumed commands across recent decisions. "
                "consecutive_model_zero_body_command_s counts continuous zero body-velocity commands "
                "from this policy; fallback or moving commands break that count. It does not measure "
                "physical speed, target distance or success. If you believe you are searching but the "
                "commands remain stationary, reassess the images and choose an observation or movement "
                "tool as appropriate. A long stationary command alone cannot justify completion."
            )
            payload["metadata"]["execution_summary_protocol"] = EXECUTION_SUMMARY_PROTOCOL
        return with_primitive_timing(payload, self.primitive_timing, learned_skills=bool(self.skills),
                                     registry_sha256=self.metadata["skill_registry_sha256"])

    def validate_call(self, value):
        return validate_primitive_timing(validate_mission_request(value, self.skills), self.primitive_timing)

    def primitive_action(self, call):
        return TOOL_ACTIONS["stop"] if call["tool"] in CLAIM_TOOLS else super().primitive_action(call)

    def primitive_plan(self, call):
        if call["tool"] not in CLAIM_TOOLS:
            return super().primitive_plan(call)
        result = decode_tool_call({k: v for k, v in {**call, "tool": "stop"}.items() if k != "target"})
        result.update(tool_call=deepcopy(call), tool_protocol=MISSION_PROTOCOL,
                      tool_contract_sha256=self.metadata["skill_registry_sha256"])
        return result

    def observe_execution(self, before, after, intent, evidence, chunk):
        super().observe_execution(before, after, intent, evidence, chunk)
        with self.lock:
            if self.execution_summary is not None:
                self.execution_summary.observe(before, after, intent, evidence)
            if (self._completion_claim is not None or chunk is None
                    or evidence.get("execution_source") != "navigation_model"):
                return
            record = chunk.diagnostics.get("hybrid", {})
            decision = self.decisions.get(record.get("decision_key"))
            if decision is None or decision["call"]["tool"] not in CLAIM_TOOLS:
                return
            active = self.context.active
            if (active is None or active["issued_ticks"] != 5
                    or after.tick != decision["until_tick"] or self.executing != decision["key"]):
                return
            call, source = decision["call"], decision["source"]
            claim = dict(mission_id=self.mission_memory.mission_id, tool=call["tool"], target=call["target"],
                memory=call.get("memory", ""), source_episode_id=source.episode_id,
                source_tick=source.tick, source_camera_tick=camera_tick(source),
                source_monotonic_ns=source.monotonic_ns, executed_through_tick=after.tick,
                decision_key=decision["key"], serving_evidence=deepcopy(decision["serving_evidence"]),
                provenance="model_claim_not_physical_success")
            self.mission_memory.append(claim)
            self._completion_claim = claim

    def completion_claim(self):
        with self.lock:
            return deepcopy(self._completion_claim)


class MissionEnv(MicroDuckEnv):
    """The evaluator measures readiness but never uses it to end a subtask."""
    def __init__(self, *, instruction, **kwargs):
        self.mission_instruction = instruction
        self.completion_requested = None
        super().__init__(**kwargs)

    def observe(self, *args, **kwargs):
        return replace(super().observe(*args, **kwargs), instruction=self.mission_instruction,
                       task_id="compound_visual_mission")

    def _update_task(self):
        super()._update_task()
        if self.task.suite != "visual_goal":
            raise ValueError("mission evaluator only supports declared visual-goal tasks")
        # Keep the exact distance/velocity/hold scoring in the original method.
        # Only success-triggered termination changes, under a distinct protocol.
        self._terminated = self._fallen_ticks >= 50 or self.tick >= round(self.task.duration_s * 50)

    @property
    def terminated(self):
        return super().terminated or bool(self.completion_requested and self.completion_requested())

    def evaluator_state(self):
        state = super().evaluator_state()
        state["mission_termination_protocol"] = MISSION_PROTOCOL
        return state


def run_mission(env, policy, output, *, spec, seed=0, warmup_ticks=50):
    if (not isinstance(env, MissionEnv) or env.profile != "official_bam" or env.daemon_backend is not None
            or not isinstance(policy, MissionNavigationPolicy) or not isinstance(spec, MissionSpec)
            or type(seed) is not int or seed < 0 or type(warmup_ticks) is not int or warmup_ticks < 0
            or env.mission_instruction != spec.instruction):
        raise ValueError("mission requires explicit actor/evaluator separation and official local physics")
    if (policy.primitive_timing is not None
            and policy.primitive_timing["maximum_duration_ms"] > MISSION_ACTION_MAX_AGE_MS):
        raise ValueError("mission primitive duration exceeds the executor's 500 ms source-age budget")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    mission_id = uuid.uuid4().hex
    policy.begin_mission(mission_id, spec.instruction)
    report = dict(schema=MISSION_PROTOCOL, status="running", mode="offline", mission_id=mission_id,
        instruction=spec.instruction, expected_targets=list(spec.expected_targets), scene=env.scene, seed=seed,
        max_claims=spec.max_claims, max_ticks_per_claim=spec.max_ticks_per_claim, warmup_ticks=warmup_ticks,
        segments=[], strict_success=False, started_unix=time.time(), physics_waits_for_inference=True,
        expected_sequence_visible_to_policy=False, truth_visible_to_policy=False,
        formal_acceptance=False, hardware_validated=False, source_sha256=sha(__file__))
    write_json(output / "mission.json", report)
    try:
        env.render_enabled = False
        env.completion_requested = None
        env.reset(seed, "find_" + spec.expected_targets[0] + "_ball", {"episode_id": mission_id + "/step-0"})
        for _ in range(warmup_ticks):
            env.step(intent_from_action([0, 0, 0, 0, 1], episode_id=env.episode_id, tick=env.tick))
        env.begin_task_at_current_pose()
        for index in range(spec.max_claims):
            if (output / "cancel.request").exists():
                report["reason"] = "cancelled_between_subtasks"
                break
            # This target is solely a scoring reference; actor observations
            # always contain the full original instruction, never this target.
            target = spec.expected_targets[min(index, len(spec.expected_targets) - 1)]
            folder = output / f"step-{index:02d}"
            # Discard the preceding segment's pending completion before the
            # shared executor tests env.terminated. The actor-only ledger is
            # deliberately preserved; the executor resets its own new buffer.
            policy.reset(env.episode_id)
            env.completion_requested = policy.completion_claim
            try:
                result = run_navigation_episode(env, policy, folder, task="find_" + target + "_ball", seed=seed,
                    realtime=False, max_ticks=spec.max_ticks_per_claim, warmup_ticks=0, warmup_predictions=0,
                    record_frames=True, record_predictions=True, reset_environment=False,
                    max_action_age_ms=MISSION_ACTION_MAX_AGE_MS)
            except Exception:
                # The shared executor records the actual motion prefix before
                # propagating a mid-episode exception. Index that evidence even
                # though this mission remains incomplete and unsuccessful.
                if (folder / "report.json").is_file():
                    result = read(folder / "report.json")
                    claim = policy.completion_claim()
                    report["segments"].append(dict(path=folder.name, report_sha256=sha(folder / "report.json"),
                        claim=claim, expected_target=target, physical_goal_satisfied=bool(result["task_success"]),
                        hazards=_hazards(result), claim_target_matches=bool(claim and claim["target"] == target)))
                report["reason"] = "execution_exception"
                raise
            claim = policy.completion_claim()
            row = dict(path=folder.name, report_sha256=sha(folder / "report.json"),
                claim=claim, expected_target=target, physical_goal_satisfied=bool(result["task_success"]),
                hazards=_hazards(result), claim_target_matches=bool(claim and claim["target"] == target))
            report["segments"].append(row)
            write_json(output / "mission.json", report)
            if row["hazards"] or claim is None:
                report["reason"] = "execution_hazard" if row["hazards"] else "no_model_completion_claim"
                break
            if claim["tool"] == "finish_mission":
                report["reason"] = "model_finished_mission"
                break
            if index + 1 == spec.max_claims:
                report["reason"] = "claim_limit_without_model_finish"
                break
            # Advance on the model's declaration even if its physical claim
            # was false. Never feed correctness back into the next decision.
            env.completion_requested = None
            before = env.clone_state()
            env.task = get_task("find_" + spec.expected_targets[min(index + 1, len(spec.expected_targets) - 1)] + "_ball")
            env.episode_id = mission_id + f"/step-{index + 1}"
            env.begin_task_at_current_pose()
            _same_tree(_motion_state(before), _motion_state(env.clone_state()))
            write_json(output / f"boundary-{index:02d}.json", dict(previous_claim=claim,
                next_episode_id=env.episode_id, full_motion_state_unchanged=True,
                truth_feedback_to_policy=False, mission_memory=policy.mission_memory.snapshot()))
        claims = [r["claim"] for r in report["segments"] if r["claim"] is not None]
        report.update(status="complete", actor_memory=policy.mission_memory.snapshot(),
            strict_success=bool(report.get("reason") == "model_finished_mission"
                and [c["target"] for c in claims] == list(spec.expected_targets)
                and all(r["physical_goal_satisfied"] and r["claim_target_matches"] and not r["hazards"]
                        for r in report["segments"])))
        return report
    except BaseException as error:
        report.update(status="incomplete", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        env.completion_requested = None
        report["actor_memory"] = policy.mission_memory.snapshot()
        report["finished_unix"] = time.time()
        report["files"] = {str(p.relative_to(output)): sha(p) for p in output.rglob("*")
                           if p.is_file() and p.name != "mission.json"}
        write_json(output / "mission.json", report)
