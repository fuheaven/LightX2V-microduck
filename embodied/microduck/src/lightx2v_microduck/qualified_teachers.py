"""Opt-in versioned teachers for project skills; legacy author presets remain.

This module exposes the actual prescribed command13 to both student and
teacher. Teacher success is separate from student capability and all hardware
claims. Eligibility comes from a matching independent teacher gate, not from
the mere existence of a task/version in this registry.
"""
from __future__ import annotations
import copy
from dataclasses import asdict, replace
import hashlib
import json

import numpy as np

from .contracts import JointResidualChunk, PolicyManifest
from .env import MicroDuckEnv
from .preset_campaign import CONDITIONS as LEGACY_CONDITIONS
from .tasks import PROGRAMS, get_task, teacher_context


PROTOCOL = "project_qualified_teachers_v2"
ACCEPTANCE_SCOPE = "project_tasks_v2"
OBSERVATION_PROFILE = "microduck_rgb_proprio_qualified_command_v2"
PROGRAM_VERSIONS = {
    "walk_forward_project_v2": {"task": "walk_forward", "mode": "walk", "speed": .28, "stop_tick": 125},
    "circle_left_project_v2": {"task": "circle_left", "mode": "author", "stop_tick": 150},
    "figure_eight_original_v1": {"task": "figure_eight", "mode": "author", "stop_tick": 175},
    "zigzag_project_v2": {"task": "zigzag", "mode": "author", "stop_tick": 175, "yaw_multiplier": 1.2},
    "sway_circle_project_v2": {"task": "sway_circle", "mode": "author", "stop_tick": 150},
    "kick_left_bam_v2": {"task": "kick_left", "mode": "author", "stop_tick": 175, "kick_ball_forward_offset_m": .09},
    "kick_right_bam_v2": {"task": "kick_right", "mode": "author", "stop_tick": 175, "kick_ball_forward_offset_m": .09},
    "sit_walk_proprio_handoff_v2": {"task": "sit_walk", "mode": "sit_handoff", "sit_until_tick": 63, "minimum_rise_ticks": 10,
                                     "catch_knee_offset_rad": .8, "catch_gravity_z": -.95, "settle_ticks": 22, "speed": .28, "stop_tick": 160},
    "roll_circle_original_v1": {"task": "roll_circle", "mode": "author", "stop_tick": 175},
    "roller_glide_coast_v2": {"task": "roller_glide", "mode": "roller", "speed": .35, "stop_tick": 150},
    "roller_crouch_coast_v2": {"task": "roller_crouch", "mode": "author", "stop_tick": 175},
    "roller_slalom_coast_v2": {"task": "roller_slalom", "mode": "author", "stop_tick": 175},
}
TASK_TEACHER_VERSIONS = {config["task"]: version for version, config in PROGRAM_VERSIONS.items()}
TASK_TEACHER_VERSIONS["ground_lower"] = "ground_lower_v1"
ROLLER_CALIBRATION = {"version": "official_roller_inference_physics_v1", "reset_height_m": .1385, "passive_wheel_frictionloss": .003,
                      "source_revision": "2b581c641406a48346e696212930ea881c222c52", "source": "microduck_rl/scripts/infer_policy.py:1065-1097"}
PHYSICAL_CHECKS = {"forward_progress_after_rise_m": .04, "minimum_upright_height_m": .073, "maximum_upright_gravity_z": -.58,
                   "rise_complete_height_m": .105, "rise_complete_gravity_z": -.95, "maximum_reverse_excursion_m": .02,
                   "post_crouch_rise_s": .5}


def task_specs():
    from .ground_lower import TASK as GROUND_LOWER_TASK
    return {task: asdict(GROUND_LOWER_TASK if task == "ground_lower" else get_task(task)) for task in TASK_TEACHER_VERSIONS}


def initial_conditions(task):
    """Return the exact recipe without constructing physics or loading models."""
    if task == "ground_lower":
        from .ground_lower import CONDITIONS
        return copy.deepcopy(CONDITIONS)
    version = TASK_TEACHER_VERSIONS[task]
    recipe = {**LEGACY_CONDITIONS, "version": "qualified_initial_v2", "teacher_version": version,
              "program": copy.deepcopy(PROGRAM_VERSIONS[version]), "command": "prescribed_visible_command13"}
    if get_task(task).embodiment == "rollers":
        recipe["physical_calibration"] = dict(ROLLER_CALIBRATION)
    return recipe


def training_contracts():
    """Explicit task/version mapping; registration alone grants no capability."""
    from . import ground_lower
    records = {}
    for task, spec in task_specs().items():
        version = TASK_TEACHER_VERSIONS[task]
        program = ground_lower.COMMAND_PROGRAM if task == "ground_lower" else PROGRAM_VERSIONS[version]
        records[task] = {"task_spec": spec, "teacher_version": version, "embodiment": spec["embodiment"], "initial_conditions": initial_conditions(task),
                         "acceptance_scope": ground_lower.TASK.suite if task == "ground_lower" else ACCEPTANCE_SCOPE,
                         "protocol": ground_lower.PROTOCOL if task == "ground_lower" else PROTOCOL,
                         "observation_profile": ground_lower.OBSERVATION_PROFILE if task == "ground_lower" else OBSERVATION_PROFILE,
                         "command_program": copy.deepcopy(program), "command_program_sha256": hashlib.sha256(json.dumps(program,sort_keys=True).encode()).hexdigest(),
                         "requires_matching_teacher_gate": True, "student_capability_validated": False, "uses_prescribed_command13": True,
                         "author_program_reproduction": False, "hardware_unvalidated": True}
    return records


def program_context(version, tick, state, phase_state, *, warming_up=False, rollers=False):
    """Advance a single reproducible clock/latch using only visible encoders/IMU."""
    config = PROGRAM_VERSIONS[version]
    command = np.zeros(13, np.float32)
    if warming_up:
        return "roller" if rollers else "stand", command
    if tick >= config["stop_tick"]:
        return "roller" if rollers else "stand", command
    if config["mode"] in ("walk", "roller"):
        command[0] = config["speed"]
        return config["mode"], command
    if config["mode"] == "sit_handoff":
        if tick < config["sit_until_tick"]:
            command[0] = 1
            return "sitstand", command
        if phase_state.get("rise_captured_tick") is None and tick >= config["sit_until_tick"] + config["minimum_rise_ticks"] and max(abs(state[9]), abs(state[18])) <= config["catch_knee_offset_rad"] and state[5] < config["catch_gravity_z"]:
            phase_state["rise_captured_tick"] = tick
        captured = phase_state.get("rise_captured_tick")
        if captured is None:
            return "sitstand", command
        if tick < captured + config["settle_ticks"]:
            return "stand", command
        phase_state.setdefault("walk_start_tick", tick)
        command[0] = config["speed"]
        return "walk", command
    role, command = teacher_context(config["task"], tick)
    if role == "walk":
        command[2] *= config.get("yaw_multiplier", 1.)
    return role, command


class QualifiedTeacherEnv(MicroDuckEnv):
    def __init__(self, *, teacher_version, asset_root=None, render=True, image_size=256):
        if teacher_version not in PROGRAM_VERSIONS:
            raise ValueError("unknown qualified teacher version; ground_lower uses its independent environment")
        self.teacher_version = teacher_version
        self.program_config = copy.deepcopy(PROGRAM_VERSIONS[teacher_version])
        self.condition_recipe = {**LEGACY_CONDITIONS, "version": "qualified_initial_v2", "teacher_version": teacher_version,
                                 "program": copy.deepcopy(self.program_config), "command": "prescribed_visible_command13"}
        if get_task(self.program_config["task"]).embodiment == "rollers":
            self.condition_recipe["physical_calibration"] = dict(ROLLER_CALIBRATION)
        super().__init__(profile="official_bam", scene="flat", asset_root=asset_root, render=render, image_size=image_size, action_delay_ticks=0)
        self.preset_clock = "program"
        self.initial_condition = None
        self._qualified_warming_up = False
        self._qualified_initialization_failed = False
        self._phase_state = {}
        self._qualified_result = None

    def reset(self, seed=0, task=None, options=None):
        spec = get_task(task or self.program_config["task"])
        if spec.task_id != self.program_config["task"] or spec.duration_s != 5.:
            raise ValueError("task does not match the versioned five-second teacher program")
        self._qualified_warming_up, self._qualified_initialization_failed = True, False
        self._phase_state, self._qualified_result, self.initial_condition = {}, None, None
        super().reset(seed, spec, options)
        rng, recipe = np.random.default_rng(seed), self.condition_recipe
        xy = rng.uniform(-recipe["xy_half_range_m"], recipe["xy_half_range_m"], 2)
        yaw = float(rng.uniform(-recipe["yaw_half_range_rad"], recipe["yaw_half_range_rad"]))
        joint = rng.uniform(-recipe["joint_half_range_rad"], recipe["joint_half_range_rad"], 14)
        light = float(rng.uniform(1-recipe["light_half_range"], 1+recipe["light_half_range"]))
        rollers = spec.embodiment == "rollers"
        if rollers:
            passive = [int(self.physics.model.jnt_dofadr[j]) for j in range(self.physics.model.njnt)
                       if (self.physics.mj.mj_id2name(self.physics.model, self.physics.mj.mjtObj.mjOBJ_JOINT, j) or "").startswith("passive_")]
            if not passive:
                raise ValueError("roller calibration requires passive wheel joints")
            self.physics.model.dof_frictionloss[passive] = ROLLER_CALIBRATION["passive_wheel_frictionloss"]
        side = PROGRAMS[spec.task_id].ball_side
        self.physics.reset(robot_xy=xy, robot_yaw=yaw, robot_height=ROLLER_CALIBRATION["reset_height_m"] if rollers else .125, kick_side=side, seed=seed)
        if side:
            offset = np.array((self.program_config["kick_ball_forward_offset_m"], .042 if side == "left" else -.042))
            c, s = np.cos(yaw), np.sin(yaw)
            self.physics.set_free_body("ball_free", (*((np.array(((c,-s),(s,c))) @ offset) + xy), .035))
        self.physics.data.qpos[self.physics.qpos_indices] += joint
        controller = self.physics.bam_controller
        controller.reset(self.physics.data.qpos)
        controller.last_ts = self.physics.data.time
        controller.q_target[:] = self.physics.last_targets
        self.physics.model.light_diffuse[:] *= light
        self.physics.mj.mj_forward(self.physics.model, self.physics.data)
        self._signature = hashlib.sha256((self._model_signature()+json.dumps(recipe, sort_keys=True)).encode()+self.physics.model.dof_frictionloss.tobytes()).hexdigest()
        self._robot_signature = self.physics.robot_signature()
        rendering, self.render_enabled = self.render_enabled, False
        try:
            for _ in range(recipe["warmup_ticks"]):
                observation = self.observe()
                self.step(self.bank.infer("roller" if rollers else "stand", observation.state))
        finally:
            self.render_enabled = rendering
        self._qualified_initialization_failed = bool(self._ever_fallen or self._collision or self._fault or self._recovery_events)
        self._qualified_warming_up = False
        self._phase_state = {}
        self.begin_task_at_current_pose()
        self.initial_condition = {"recipe": copy.deepcopy(recipe), "task": spec.task_id, "xy": xy.tolist(), "yaw": yaw, "joint_offset": joint.tolist(), "light_scale": light}
        return self.observe()

    def observe(self):
        observation = super().observe()
        _, command = program_context(self.teacher_version, observation.tick, observation.state, self._phase_state,
                                     warming_up=self._qualified_warming_up, rollers=self.task.embodiment == "rollers")
        state = observation.state.copy()
        state[48:] = command
        return replace(observation, state=state)

    def _update_task(self):
        super()._update_task()
        if self._qualified_warming_up or not self._terminated:
            return
        trace = np.asarray(self._trajectory)
        checks = {**self._strict_result["checks"], "initialization_valid": not self._qualified_initialization_failed, "no_fault": not bool(self._fault)}
        start = 0.
        if self.task.task_id == "sit_walk":
            start = self._phase_state.get("walk_start_tick", 250) * .02
            checks["walk_phase_started"] = start < self.program_config["stop_tick"]*.02
        elif self.task.task_id == "roll_circle":
            start = 2.  # The planned forward roll is not an accidental fall.
        elif self.task.task_id in ("roller_crouch", "roller_slalom"):
            # Official roller_crouch has a 0.5 s rise segment. A legal crouch
            # can be below the standing fall-height threshold; it must never
            # tip, then must regain standing height after that rise window.
            start = (2. if self.task.task_id == "roller_crouch" else 2.3) + PHYSICAL_CHECKS["post_crouch_rise_s"]
            checks["no_unintended_tip"] = bool(np.all(trace[:,4] <= PHYSICAL_CHECKS["maximum_upright_gravity_z"]))
        active = trace[trace[:, 0] >= start - 1e-7]
        checks["no_unintended_fall"] = bool(np.all((active[:,3] >= PHYSICAL_CHECKS["minimum_upright_height_m"]) & (active[:,4] <= PHYSICAL_CHECKS["maximum_upright_gravity_z"])))
        if self.task.task_id == "sit_walk":
            yaw = trace[0,5]
            walking = active[active[:,0] <= self.program_config["stop_tick"]*.02+1e-7]
            if not len(walking):
                walking = active[-1:]
            forward = (walking[:,1]-walking[0,1])*np.cos(yaw)+(walking[:,2]-walking[0,2])*np.sin(yaw)
            checks.update(rise_complete_before_walk=bool(active[0,3] >= PHYSICAL_CHECKS["rise_complete_height_m"] and active[0,4] <= PHYSICAL_CHECKS["rise_complete_gravity_z"]),
                          forward_not_path_only=bool(forward[-1] >= PHYSICAL_CHECKS["forward_progress_after_rise_m"]),
                          no_backward_excursion=bool(forward.min() >= -PHYSICAL_CHECKS["maximum_reverse_excursion_m"]))
        if self.task.task_id == "roller_glide":
            phase = trace[trace[:,0] <= self.program_config["stop_tick"]*.02+1e-7]
            forward = (phase[:,1]-phase[0,1])*np.cos(trace[0,5])+(phase[:,2]-phase[0,2])*np.sin(trace[0,5])
            checks.update(forward_not_path_only=bool(forward[-1]>=.04), no_backward_excursion=bool(forward.min()>=-.02))
        self._qualified_result = {"protocol": PROTOCOL, "teacher_version": self.teacher_version, "success": all(checks.values()), "checks": checks,
                                  "acceptance_scope": ACCEPTANCE_SCOPE,
                                  "failed_checks": [key for key,value in checks.items() if not value], "reference_strict_result": copy.deepcopy(self._strict_result),
                                  "additional_physical_thresholds": dict(PHYSICAL_CHECKS), "counts_as_student_skill_success": False}
        self._success = self._qualified_result["success"]

    def evaluator_state(self):
        result = super().evaluator_state()
        initial = copy.deepcopy(self.initial_condition)
        result.update(qualified_teacher_result=copy.deepcopy(self._qualified_result), teacher_version=self.teacher_version,
                      strict_preset_result=copy.deepcopy(self._qualified_result), reference_strict_result=copy.deepcopy(self._strict_result),
                      author_success=None, reference_author_style_success=self._author_success,
                      acceptance_scope=ACCEPTANCE_SCOPE, success_protocol=PROTOCOL,
                      command_program_sha256=hashlib.sha256(json.dumps(self.program_config,sort_keys=True).encode()).hexdigest(),
                      command_program=copy.deepcopy(self.program_config), initial_condition=initial,
                      initial_condition_sha256=hashlib.sha256(json.dumps(initial,sort_keys=True).encode()).hexdigest() if initial else None,
                      initial_conditions=copy.deepcopy(self.condition_recipe), phase_state=copy.deepcopy(self._phase_state),
                      uses_prescribed_command13=True, claims_visual_language_autonomy=False, counts_as_student_skill_success=False,
                      author_command_timing_unchanged=self.program_config["mode"]=="author" and self.program_config["stop_tick"]==175 and "yaw_multiplier" not in self.program_config)
        return result

    def clone_state(self):
        snapshot=super().clone_state()
        snapshot["qualified_teacher"]={"version": self.teacher_version, "conditions": copy.deepcopy(self.condition_recipe), "initial_condition":copy.deepcopy(self.initial_condition),
                                       "phase_state":copy.deepcopy(self._phase_state), "warming_up":self._qualified_warming_up,
                                       "initialization_failed":self._qualified_initialization_failed, "result":copy.deepcopy(self._qualified_result)}
        return snapshot

    snapshot=clone_state

    def restore_state(self,snapshot):
        metadata=snapshot.get("qualified_teacher",{})
        if metadata.get("version")!=self.teacher_version or metadata.get("conditions")!=self.condition_recipe:
            raise ValueError("qualified teacher snapshot has a different execution or condition version")
        super().restore_state(snapshot)
        self.initial_condition=copy.deepcopy(metadata["initial_condition"])
        self._phase_state=copy.deepcopy(metadata["phase_state"])
        self._qualified_warming_up=metadata["warming_up"]
        self._qualified_initialization_failed=metadata["initialization_failed"]
        self._qualified_result=copy.deepcopy(metadata["result"])
        return self.observe()

    restore=restore_state


class QualifiedTeacher:
    def __init__(self,env):
        self.env=env
        self.manifest=PolicyManifest("official_"+env.teacher_version,"joint_residual","official_61d","official_bam",
                                     embodiment=get_task(env.program_config["task"]).embodiment,
                                     metadata={"teacher_version":env.teacher_version,"initial_conditions":copy.deepcopy(env.condition_recipe),
                                               "acceptance_scope":ACCEPTANCE_SCOPE,"protocol":PROTOCOL,
                                               "command_program":copy.deepcopy(env.program_config),
                                               "command_program_sha256":hashlib.sha256(json.dumps(env.program_config,sort_keys=True).encode()).hexdigest(),
                                               "uses_prescribed_command13":True,"claims_visual_language_autonomy":False,"counts_as_student_skill_success":False})

    def reset(self,episode_id=""):
        self.episode_id=str(episode_id)

    def predict(self,observation):
        if observation.episode_id!=self.env.episode_id or observation.tick!=self.env.tick:
            raise ValueError("qualified teacher requires the current environment observation")
        role,command=program_context(self.env.teacher_version,observation.tick,observation.state,self.env._phase_state,
                                     warming_up=self.env._qualified_warming_up,rollers=self.env.task.embodiment=="rollers")
        if not np.array_equal(observation.state[48:],command):
            raise ValueError("student and teacher command13 do not match the execution version")
        raw=self.env.bank.infer(role,observation.state)
        return JointResidualChunk(raw[None],observation.episode_id,observation.tick,observation.sim_time_s,
                                  diagnostics={"teacher_policy":role,"teacher_command":command.tolist(),"teacher_version":self.env.teacher_version,
                                               "phase":"project_program","uses_privileged_state":False})


def make_environment(task,**kwargs):
    if task=="ground_lower":
        from .ground_lower import GroundLowerEnv
        return GroundLowerEnv(**kwargs)
    return QualifiedTeacherEnv(teacher_version=TASK_TEACHER_VERSIONS[task],**kwargs)


def make_teacher(env):
    from .ground_lower import GroundLowerEnv, GroundLowerTeacher
    return GroundLowerTeacher(env) if isinstance(env,GroundLowerEnv) else QualifiedTeacher(env)
