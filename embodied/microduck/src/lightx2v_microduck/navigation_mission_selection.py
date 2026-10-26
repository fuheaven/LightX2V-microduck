"""Optional actor-authored subgoal commitment, without evaluator guidance.

The model selects its current target explicitly. The tool registry checks
consistency with that choice, not whether the choice follows the user's goal.
A wrong selection or false completion remains an independently scored failure.
"""
from copy import deepcopy
from dataclasses import replace
from functools import partial
import hashlib
import json

from . import navigation_mission_visual_memory
from .navigation_mission import MissionNavigationPolicy, validate_mission_request, CLAIM_TOOLS
from .navigation_mission_visual_memory import (
    SelectedTargetVisualMemory, VisualMissionRequest, VISUAL_MEMORY_PROTOCOL, with_historical_target,
)
from .navigation_outcomes import sha
from .navigation_mission_timing import validate_primitive_timing
from .planner_context import camera_tick
from .planner_tools import TOOL_ACTIONS, decode_tool_call


SELECTED_GOAL_PROTOCOL = "microduck_model_selected_subgoal_v1"
SELECT_TOOL = "select_subgoal"


def validate_selected_goal_request(value, available, selected=None):
    if isinstance(value, dict) and value.get("tool") == SELECT_TOOL:
        valid = validate_mission_request({**value, "tool": "complete_subgoal"}, available)
        return {**valid, "tool": SELECT_TOOL}
    valid = validate_mission_request(value, available)
    if selected is None:
        raise ValueError("select_subgoal(target, duration_s=0.1) must execute before movement or completion")
    if valid["tool"] in available and valid["tool"] != "navigate_" + selected["target"]:
        raise ValueError("navigation differs from your selected subgoal; explicitly select a new goal to change it")
    if valid["tool"] in CLAIM_TOOLS and valid["target"] != selected["target"]:
        raise ValueError("completion target differs from your selected subgoal")
    return valid


class SelectedGoalMissionPolicy(MissionNavigationPolicy):
    def __init__(self, *, visual_memory=False, **kwargs):
        if type(visual_memory) is not bool:
            raise ValueError("visual_memory must be a boolean")
        self.visual_memory = SelectedTargetVisualMemory() if visual_memory else None
        self.selected_goal = None
        super().__init__(**kwargs)
        self.metadata.update(protocol=SELECTED_GOAL_PROTOCOL,
            goal_selection="model_choice_committed_after_5_executed_stand_ticks",
            goal_selection_source_sha256=sha(__file__),
            selection_correctness_checked_by_executor=False)
        if self.visual_memory is not None:
            from .navigation_visual_memory import CausalTargetKeyframeMemory
            self.metadata.update(visual_memory_protocol=VISUAL_MEMORY_PROTOCOL,
                visual_memory_contract=CausalTargetKeyframeMemory.contract(),
                visual_memory_source_sha256=sha(navigation_mission_visual_memory.__file__))
        self.metadata["skill_registry_sha256"] = hashlib.sha256(json.dumps(dict(
            base_registry=self.metadata["skill_registry_sha256"], protocol=SELECTED_GOAL_PROTOCOL,
            selection_tool=SELECT_TOOL, duration_s=.1), sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def reset(self, episode_id):
        super().reset(episode_id)
        with self.lock:
            # The next subtask starts with the same completed-claim ledger,
            # but selecting its next target is a fresh model decision.
            self.selected_goal = None
            if self.visual_memory is not None:
                self.visual_memory.reset(episode_id)

    def prepare_request(self, observation):
        with self.lock:
            request = super().prepare_request(observation)
            context = deepcopy(request.context)
            context["selected_subgoal"] = deepcopy(self.selected_goal)
            if self.visual_memory is not None:
                self.visual_memory.append(observation)
                frame, receipt = self.visual_memory.snapshot(observation.episode_id, observation.tick, self.selected_goal)
                context["visual_memory"] = receipt
                return VisualMissionRequest(request.generation, request.observation, context,
                                            request.previous, request.delegates, frame)
            return replace(request, context=context)

    def request_payload(self, request):
        payload = super().request_payload(request)
        payload["messages"][0]["content"] += (
            " Before movement or completion, explicitly choose your current subgoal using "
            "select_subgoal: return tool='select_subgoal', target='red'/'blue'/'yellow', duration_s=0.1, "
            "and optional memory. This executes a short stand command and records only YOUR target "
            "choice, not a successful task. Read the full user instruction and your prior claims to "
            "choose the next target. While a goal is selected, color-navigation tools and completion "
            "targets must match that selection. To change it, call select_subgoal explicitly. "
            "Directional movement and observation tools remain available. After a completion claim "
            "the next subtask requires a fresh selection. The selection can be wrong: no hidden "
            "planner checks its correctness or supplies a target."
        )
        payload["messages"][-1]["content"].append(dict(type="text", text="Selected subgoal: " + json.dumps(
            request.context["selected_subgoal"], sort_keys=True, separators=(",", ":"))))
        payload["metadata"].update(prompt_profile=SELECTED_GOAL_PROTOCOL, tool_protocol=SELECTED_GOAL_PROTOCOL)
        return with_historical_target(payload, request) if self.visual_memory is not None else payload

    def validate_call(self, value):
        with self.lock:
            return validate_primitive_timing(
                validate_selected_goal_request(value, self.skills, self.selected_goal), self.primitive_timing)

    def primitive_action(self, call):
        return TOOL_ACTIONS["stop"] if call["tool"] == SELECT_TOOL else super().primitive_action(call)

    def primitive_plan(self, call):
        if call["tool"] != SELECT_TOOL:
            return super().primitive_plan(call)
        plan = decode_tool_call({k: v for k, v in {**call, "tool": "stop"}.items() if k != "target"})
        plan.update(tool_call=deepcopy(call), tool_protocol=SELECTED_GOAL_PROTOCOL,
                    tool_contract_sha256=self.metadata["skill_registry_sha256"])
        return plan

    def observe_execution(self, before, after, intent, evidence, chunk):
        super().observe_execution(before, after, intent, evidence, chunk)
        with self.lock:
            if chunk is None or evidence.get("execution_source") != "navigation_model":
                return
            record = chunk.diagnostics.get("hybrid", {})
            decision = self.decisions.get(record.get("decision_key"))
            if decision is None or decision["call"]["tool"] != SELECT_TOOL:
                return
            active = self.context.active
            if (active is None or active["issued_ticks"] != 5 or after.tick != decision["until_tick"]
                    or self.executing != decision["key"]):
                return
            source, call = decision["source"], decision["call"]
            self.selected_goal = dict(target=call["target"], memory=call.get("memory", ""),
                source_episode_id=source.episode_id, source_tick=source.tick, source_camera_tick=camera_tick(source),
                executed_through_tick=after.tick, decision_key=decision["key"],
                provenance="executed_model_selection_not_evaluator_target_or_verified_success")


def mission_policy_class(mode, *, execution_summary=False, visual_memory=False, primitive_max_duration_ms=None):
    if type(execution_summary) is not bool or type(visual_memory) is not bool:
        raise ValueError("mission options must be booleans")
    if mode == "declarations":
        if visual_memory:
            raise ValueError("visual memory requires a model-selected goal")
        cls = MissionNavigationPolicy
    elif mode == "selected_goal":
        cls = SelectedGoalMissionPolicy
    else:
        raise ValueError("unknown mission memory mode")
    options = {}
    if execution_summary:
        options["execution_summary"] = True
    if visual_memory:
        options["visual_memory"] = True
    if primitive_max_duration_ms is not None:
        options["primitive_max_duration_ms"] = primitive_max_duration_ms
    return partial(cls, **options) if options else cls
