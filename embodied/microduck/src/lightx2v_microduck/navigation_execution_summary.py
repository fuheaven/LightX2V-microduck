"""Bounded summaries of consumed control commands, never physical success."""
from copy import deepcopy
import math


EXECUTION_SUMMARY_PROTOCOL = "microduck_consumed_command_summary_v1"


class ExecutedCommandSummary:
    def __init__(self):
        self.reset("")

    def reset(self, episode_id):
        self.episode_id = episode_id
        self.next_tick = None
        self.history_start_tick = None
        self.history_gaps = 0
        self.control_ticks = self.fallback_ticks = 0
        self.zero_ticks = self.model_zero_ticks = 0
        self.last_record = None

    def observe(self, before, after, intent, evidence):
        if (before.episode_id != self.episode_id or after.episode_id != self.episode_id
                or intent.episode_id != self.episode_id or type(before.tick) is not int
                or type(after.tick) is not int or before.tick < 0 or after.tick != before.tick + 1):
            raise ValueError("execution summary requires one actual control tick in its own episode")
        command = tuple(float(v) for v in intent.command)
        source = evidence.get("execution_source")
        if len(command) != 13 or not all(math.isfinite(v) for v in command) or not isinstance(source, str) or not source:
            raise ValueError("execution summary requires a finite consumed command and its source")
        record = dict(tick=before.tick, command13=command, skill=intent.skill, execution_source=source)
        if self.last_record is not None and before.tick == self.last_record["tick"]:
            if record != self.last_record:
                raise ValueError("conflicting execution at the same control tick")
            return
        if self.next_tick is not None and before.tick < self.next_tick:
            raise ValueError("old execution cannot change the current summary")
        if self.next_tick is None or before.tick != self.next_tick:
            self.history_gaps += int(self.next_tick is not None)
            self.history_start_tick = before.tick
            self.control_ticks = self.fallback_ticks = self.zero_ticks = self.model_zero_ticks = 0
        model = source == "navigation_model"
        zero = command[:3] == (0., 0., 0.)
        self.control_ticks += 1
        self.fallback_ticks += int(not model)
        self.zero_ticks = self.zero_ticks + 1 if zero else 0
        self.model_zero_ticks = self.model_zero_ticks + 1 if zero and model else 0
        self.last_record, self.next_tick = record, after.tick

    def snapshot(self):
        return dict(schema=EXECUTION_SUMMARY_PROTOCOL, source="actual_executor_consumed_commands",
            through_tick=self.next_tick, continuous_history_start_tick=self.history_start_tick,
            history_gap_count=self.history_gaps, observed_control_ticks=self.control_ticks,
            non_model_control_ticks=self.fallback_ticks,
            consecutive_zero_body_command_s=self.zero_ticks / 50,
            consecutive_model_zero_body_command_s=self.model_zero_ticks / 50,
            last_execution=deepcopy(self.last_record),
            interpretation="commanded_body_velocity_only_not_measured_motion_distance_or_goal_completion")
