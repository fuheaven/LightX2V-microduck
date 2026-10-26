"""Independent MicroDuck environment: direct residuals, real physics, no ROS.

``observe`` exposes only sensors. World positions are available exclusively
through ``evaluator_state`` for scoring and privileged teacher generation.
"""

from __future__ import annotations

import copy
import hashlib
import time
import uuid
from collections import deque
from dataclasses import replace
from pathlib import Path

import numpy as np

from .assets import load_lock, resolve_asset_root
from .contracts import HOME_POLICY, JointResidualChunk, Observation, SkillIntent, TaskSpec, pack_observation
from .physics import PROFILES, Physics
from .rendering import validate_camera_offsamples
from .policies import PolicyBank
from .scenes import CURRICULUM_ROOMS, build_scene, sample_ball_positions
from .tasks import PROGRAMS, TASKS_SOURCE_SHA256, get_task, strict_preset_result, task_command, teacher_context

EVALUATOR_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class MicroDuckEnv:
    def __init__(self, profile="official_bam", scene="room", asset_root=None, render=True, *, image_size=256, action_delay_ticks=None, daemon_backend=None, camera_offsamples=None):
        if profile not in PROFILES:
            raise ValueError(f"unknown profile {profile!r}")
        if profile == "official_daemon" and daemon_backend is None:
            raise ValueError("official_daemon requires an actual robotd/duck-body daemon_backend; start duck-sim and use the daemon client")
        self.profile, self.scene = profile, str(scene)
        self.camera_offsamples = validate_camera_offsamples(camera_offsamples)
        if daemon_backend is not None and camera_offsamples is not None:
            raise ValueError("camera_offsamples applies only to the local MuJoCo renderer")
        self.asset_root = resolve_asset_root(asset_root)
        self.render_enabled = bool(render)
        self.image_size = int(image_size)
        self.action_delay_ticks = (1 if profile == "tinyvla_author_compat" else 0) if action_delay_ticks is None else int(action_delay_ticks)
        self._configured_delay_ticks = action_delay_ticks
        if not 0 <= self.action_delay_ticks <= 50:
            raise ValueError("action_delay_ticks must be in 0..50")
        self.daemon_backend = daemon_backend
        self.physics = None
        self.task: TaskSpec | None = None
        policy_files = load_lock()["sources"]["microduck-policies"]["files"]
        hashes = {entry["path"]: entry["sha256"] for entry in policy_files}
        self.bank = PolicyBank(self.asset_root / "microduck-policies", hashes=hashes, profile=profile)
        self.tick = 0
        self.episode_id = ""
        self.preset_clock = "legacy_recovery"

    def reset(self, seed=0, task="walk_forward", options=None):
        if self.daemon_backend is not None:
            return self.daemon_backend.reset(seed=seed, task=task, options=options)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.task = get_task(task)
        if self.task.task_id == "run_roll_recover":
            from .tricks import roll_trigger_tick
            self.task = replace(self.task, duration_s=(roll_trigger_tick(self.seed) + 225) * .02)
        if "instruction" in (options or {}):
            self.task = replace(self.task, instruction=str(options["instruction"]))
        self.episode_id = str((options or {}).get("episode_id", uuid.uuid4().hex))
        if self.physics is not None:
            self.physics.close()
        seeded_room = self.scene == "room_randomized" or self.scene in CURRICULUM_ROOMS
        spec, self.description = build_scene(self.asset_root, self.profile, self.scene, self.task, obstacle_layout=self.seed if seeded_room else self.seed % 2)
        tof_source = self.asset_root / "microduck_rl/src/mjlab_microduck/sim/tof.py"
        self.physics = Physics(spec, profile=self.profile, image_size=self.image_size, robot_prefix=self.description.robot_prefix, tof_source=tof_source if tof_source.is_file() else None, camera_offsamples=self.camera_offsamples)
        ball_positions = sample_ball_positions(self.description, self.rng)
        program = PROGRAMS.get(self.task.task_id)
        basketball = self.task.suite == "community_basketball"
        self.physics.reset(
            ball_positions=ball_positions,
            kick_side=None if program is None else program.ball_side,
            seed=self.seed,
            robot_height=0.368 if basketball else 0.125,
            free_bodies={"ball_free": (0.0, 0.0, 0.121)} if basketball else None,
        )
        self._filtered_action = np.zeros(14, np.float32)
        if self.task.suite == 'community_climb':
            from .climb import load_scene
            from .contracts import POLICY_INDICES
            geometry = load_scene(self.asset_root)
            pose = np.asarray(geometry['spawn_qpos22'])
            self._climb_desk = geometry['desktop']
            self.physics.data.qpos[self.physics.trunk_qpos:self.physics.trunk_qpos + 7] = pose[:7]
            self.physics.data.qpos[self.physics.qpos_indices] = pose[7:][list(POLICY_INDICES)]
            self.physics.mj.mj_forward(self.physics.model, self.physics.data)
            self.physics.bam_controller.reset(self.physics.data.qpos)
            if self._configured_delay_ticks is None:
                self.action_delay_ticks = 1
        elif self._configured_delay_ticks is None:
            self.action_delay_ticks = 1 if self.profile == 'tinyvla_author_compat' else 0
        self.tick = 0
        self.last_action = np.zeros(14, dtype=np.float32)
        self.applied_action = np.zeros(14, dtype=np.float32)
        self._delay = deque([np.zeros(14, dtype=np.float32) for _ in range(self.action_delay_ticks)])
        mj, model = self.physics.mj, self.physics.model
        self._mouth_tip = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, self.description.robot_prefix + "mouth_tip")
        self._wheel_dofs = [int(model.jnt_dofadr[i]) for i in range(model.njnt) if "wheel" in (mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, i) or "")]
        self._signature = self._model_signature()
        self._robot_signature = self.physics.robot_signature()
        self.bank.reset(self.episode_id)
        self.begin_task_at_current_pose()
        return self.observe()

    def begin_task_at_current_pose(self):
        """Start the task clock after warmup, preserving all controller state.

        This resets scoring only: MjData, absolute simulation time, BAM memory,
        raw action history/delay, policy state and both RNGs remain untouched.
        No observation is taken here because sampling a sensor can advance RNG.
        """
        if self.daemon_backend is not None:
            return self.daemon_backend.begin_task_at_current_pose()
        if self.physics is None:
            raise RuntimeError("call reset before beginning a task window")
        self.tick = 0
        self._evaluation_origin_s = self.sim_time_s
        self._recovery_active = False
        self._recovery_upright_ticks = 0
        self._task_start_tick = 0
        self._recovery_events = 0
        self._fallen_ticks = 0
        self._goal_hold_ticks = 0
        self._follow_in_band_ticks = 0
        self._off_ball = False
        self._desk_hold_ticks = 0
        self._max_desk_hold_ticks = 0
        self._termination_reason = ""
        self._success = False
        self._author_success = None
        self._strict_result = None
        self._ever_fallen = False
        self._contact_events = set()
        self._wheel_rotation = 0.0
        self._terminated = False
        self._collision = False
        self._fault = ""
        self._rgb_cache = None
        self._rgb_tick = -1
        self._observer_rgb_cache = None
        self._observer_rgb_tick = -1
        self._trajectory = []
        self._initial_ball = {k: np.asarray(v).copy() for k, v in self._ball_positions().items()}
        self._initial_position = self.physics.data.xpos[self.physics.trunk].copy()
        self._record_metrics(include_contacts=False, integrate_wheels=False)

    def perception_truth(self):
        """Visible target mask for the evaluator only; never part of observe()."""
        physics = self.physics
        physics.render()  # Initialize the renderer in the simulation thread.
        renderer = physics.renderer
        renderer.enable_segmentation_rendering()
        try:
            renderer.update_scene(physics.data, camera=physics.camera)
            segmentation = renderer.render().copy()
        finally:
            renderer.disable_segmentation_rendering()
        if self.profile == "official_bam":
            segmentation = np.rot90(segmentation, k=-1)
        target_geoms = [i for i in range(physics.model.ngeom)
                        if (physics.mj.mj_id2name(physics.model, physics.mj.mjtObj.mjOBJ_BODY,
                                                int(physics.model.geom_bodyid[i])) or '').startswith('target_')]
        visible = np.isin(segmentation[:, :, 0], target_geoms) & (segmentation[:, :, 1] == int(physics.mj.mjtObj.mjOBJ_GEOM))
        y, x = np.nonzero(visible)
        return {"bbox": [int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1] if len(x) else None,
                "visible_pixels": len(x), "evaluator_only": True}

    def _model_signature(self):
        digest = hashlib.sha256()
        digest.update(f"{self.profile}/{self.description}/{self.action_delay_ticks}".encode())
        for key in ("body_mass", "body_inertia", "body_pos", "geom_pos", "geom_size", "jnt_range", "cam_quat"):
            digest.update(np.asarray(getattr(self.physics.model, key)).tobytes())
        return digest.hexdigest()

    @property
    def sim_time_s(self):
        return self.tick * 0.02 if self.physics is None else float(self.physics.data.time)

    def camera_configuration(self):
        if self.daemon_backend is not None:
            return None
        if self.physics is None:
            raise RuntimeError("call reset before reading the camera configuration")
        return self.physics.camera_configuration()

    @property
    def terminated(self):
        return self._terminated

    def observe(self):
        if self.daemon_backend is not None:
            return self.daemon_backend.observe()
        if self.physics is None:
            raise RuntimeError("call reset before observe")
        gyro, gravity, joints, velocities = self.physics.sensors()
        command = task_command(self.task, self.tick, recovery=self._recovery_active, task_start_tick=self._task_start_tick)
        state = pack_observation(gyro, gravity, joints, velocities, self.last_action, command)
        tof, tof_metadata = self.physics.read_tof(round(self.sim_time_s * 50))
        if tof_metadata.get("available"):
            tof_metadata["physics_capture_tick"] = tof_metadata["capture_tick"]
            tof_metadata["capture_tick"] -= round(self._evaluation_origin_s * 50)
        return Observation(
            self.episode_id,
            self.tick,
            self.sim_time_s,
            state,
            rgb=self.render() if self.render_enabled else None,
            instruction=self.task.instruction,
            task_id=self.task.task_id,
            camera_tick=self.tick if self.render_enabled else None,
            monotonic_ns=time.monotonic_ns(),
            camera_metadata=self.physics.camera_metadata(),
            tof=tof,
            tof_metadata=tof_metadata,
        )

    def render(self):
        if self.daemon_backend is not None:
            return self.daemon_backend.render()
        if self.physics is None:
            raise RuntimeError("call reset before render")
        if self._rgb_tick != self.tick:
            self._rgb_cache = self.physics.render()
            self._rgb_tick = self.tick
        else:
            # A restored observation can already have its measured RGB while
            # the old renderer was discarded. Visibility/segmentation clients
            # still need a context at this pose, without recapturing that RGB.
            self.physics.head_renderer()
        return self._rgb_cache.copy()

    def render_observer(self):
        """Optional offline review view; deliberately absent from Observation."""
        if self.daemon_backend is not None:
            raise RuntimeError("third-person observer rendering is available only in the offline simulator")
        if self.physics is None:
            raise RuntimeError("call reset before rendering an observer view")
        if not self.render_enabled:
            return None
        if self._observer_rgb_tick != self.tick:
            self._observer_rgb_cache = self.physics.render_observer()
            self._observer_rgb_tick = self.tick
        return self._observer_rgb_cache.copy()

    def step(self, action):
        if self.daemon_backend is not None:
            return self.daemon_backend.step(action)
        if self.physics is None:
            raise RuntimeError("call reset before step")
        if self._terminated:
            raise RuntimeError("episode terminated; reset before stepping")
        if isinstance(action, JointResidualChunk):
            if action.episode_id != self.episode_id or action.source_tick > self.tick or action.source_time_s > self.sim_time_s + 1e-8:
                raise ValueError("cross-episode or future action chunk")
            if action.valid_until_tick is not None and self.tick >= action.valid_until_tick:
                raise ValueError("expired action chunk")
            if action.action_profile != "raw_residual" and not (self.task.suite == 'community_climb' and action.action_profile in ('climb_raw', 'climb_getup_raw')):
                raise ValueError("decode/denormalize actions before MicroDuckEnv.step")
            offset = self.tick - action.source_tick
            if offset >= len(action.actions):
                raise ValueError("action chunk contains no unexecuted steps")
            stop = len(action.actions)
            if action.valid_until_tick is not None:
                stop = min(stop, action.valid_until_tick - action.source_tick)
            rows = action.actions[offset:stop]
        elif isinstance(action, SkillIntent):
            if action.episode_id and action.episode_id != self.episode_id:
                raise ValueError("cross-episode skill intent")
            if action.source_tick > self.tick:
                raise ValueError("future skill intent")
            render_enabled = self.render_enabled
            try:
                self.render_enabled = False
                for _ in range(action.duration_ticks):
                    observation = self.observe()
                    teacher_state = observation.state.copy()
                    teacher_state[48:61] = action.command
                    self._step_raw(self.bank.infer(action.skill, teacher_state))
                    if self._terminated:
                        break
            finally:
                self.render_enabled = render_enabled
            return self.observe()
        else:
            values = np.asarray(action, dtype=np.float32)
            if values.shape != (14,):
                raise ValueError("pass one raw14 action or a typed JointResidualChunk")
            rows = values[None]
        for row in rows:
            self._step_raw(row, action_profile=action.action_profile if isinstance(action, JointResidualChunk) else 'raw_residual')
            if self._terminated:
                break
        return self.observe()

    def _step_raw(self, action, *, action_profile='raw_residual'):
        raw = np.asarray(action, dtype=np.float32)
        if raw.shape != (14,) or not np.isfinite(raw).all():
            raise ValueError("raw action must be finite 14D")
        executed = raw.copy()
        if action_profile in ('climb_raw', 'climb_getup_raw'):
            from .contracts import POLICY_JOINT_NAMES
            recovering = action_profile == 'climb_getup_raw'
            if recovering:
                alpha = np.array([.5 if name.startswith(('neck', 'head')) else .7 for name in POLICY_JOINT_NAMES], np.float32)
                executed = alpha * raw + (1 - alpha) * self._filtered_action
            self.physics.bam_model.actuator.kp = 160. if recovering else 200.
        self._filtered_action = executed.copy()
        self._delay.append(executed)
        self.applied_action = self._delay.popleft()
        # Preserve policy raw history independently from motor target conversion
        # and transport delay; observations never use absolute targets here.
        targets = np.asarray(HOME_POLICY, dtype=np.float32) + self.applied_action
        try:
            self.physics.step(targets)
        except FloatingPointError as exc:
            self._fault, self._terminated = str(exc), True
            raise
        self.last_action = raw.copy()
        self.tick += 1
        # Sensor noise/time advances at the physical cadence even when a chunk
        # executes multiple controls between observations or rendering is off.
        self.physics.read_tof(round(self.sim_time_s * 50))
        self._rgb_tick = -1
        self._observer_rgb_tick = -1
        self._record_metrics()
        self._update_task()

    def _ball_positions(self):
        positions = {}
        for color in self.description.target_colors:
            idx = self.physics._id(self.physics.mj.mjtObj.mjOBJ_BODY, f"lx_ball_{color}")
            positions[color] = self.physics.data.xpos[idx].copy()
        ball = self.physics.mj.mj_name2id(self.physics.model, self.physics.mj.mjtObj.mjOBJ_BODY, "ball")
        if ball >= 0:
            positions["basketball" if self.task.suite == "community_basketball" else "kick_ball"] = self.physics.data.xpos[ball].copy()
        return positions

    def _record_metrics(self, *, include_contacts=True, integrate_wheels=True):
        data, physics = self.physics.data, self.physics
        gyro, gravity, _, _ = physics.sensors()
        quat = data.xquat[physics.trunk]
        w, x, y, z = quat
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
        mouth_z = float(data.site_xpos[self._mouth_tip, 2]) if self._mouth_tip >= 0 else 1.0
        speed = float(np.linalg.norm(data.qvel[physics.trunk_qvel : physics.trunk_qvel + 2]))
        if integrate_wheels:
            self._wheel_rotation += float(np.sum(np.abs(data.qvel[self._wheel_dofs]))) * 0.02
        self._trajectory.append(
            (
                float(data.time) - self._evaluation_origin_s,
                *data.xpos[physics.trunk].tolist(),
                float(gravity[2]),
                float(yaw),
                float(gyro[1]),
                float(roll),
                float(pitch),
                speed,
                mouth_z,
                self._wheel_rotation,
            )
        )
        if not include_contacts:
            return
        for a, b in physics.last_contacts:
            names = [physics.mj.mj_id2name(physics.model, physics.mj.mjtObj.mjOBJ_GEOM, i) or "" for i in (a, b)]
            bodies = [physics.mj.mj_id2name(physics.model, physics.mj.mjtObj.mjOBJ_BODY, int(physics.model.geom_bodyid[i])) or "" for i in (a, b)]
            if "ball" in bodies:
                other = bodies[1] if bodies[0] == "ball" else bodies[0]
                if other.endswith("ankle_left"):
                    self._contact_events.add("left_foot_ball")
                if other.endswith("ankle_right"):
                    self._contact_events.add("right_foot_ball")
            if any(n in ("floor", "terrain_floor") for n in names) and any(n.endswith("jaw_soft") for n in bodies):
                self._contact_events.add("mouth_ground")
            robot_geoms = []
            for geom in (a, b):
                body = int(physics.model.geom_bodyid[geom])
                belongs = False
                while body > 0:
                    if body == physics.trunk:
                        belongs = True
                        break
                    body = int(physics.model.body_parentid[body])
                robot_geoms.append(belongs)
            if sum(robot_geoms) == 1:
                other = b if robot_geoms[0] else a
                other_name = bodies[1] if robot_geoms[0] else bodies[0]
                floor = physics.model.geom_type[other] == physics.mj.mjtGeom.mjGEOM_PLANE
                intended_kick = other_name == "ball" and self.task.task_id in ("kick_left", "kick_right", "ball_apex")
                intended_climb = self.task.suite == 'community_climb' and (physics.mj.mj_id2name(physics.model, physics.mj.mjtObj.mjOBJ_GEOM, other) or '').startswith('lx_climb_')
                if not floor and not intended_kick and not intended_climb:
                    self._collision = True

    def _update_task(self):
        data = self.physics.data
        _, gravity, _, _ = self.physics.sensors()
        height = float(data.xpos[self.physics.trunk, 2])
        fallen = bool(gravity[2] > -0.58 or height < 0.073)
        upright = bool(gravity[2] < -0.82 and height > 0.09)
        if self.task.suite == 'community_climb':
            from .climb import desk_state
            status = desk_state(self)
            self._desk_hold_ticks = self._desk_hold_ticks + 1 if status['standing_safe'] else 0
            self._max_desk_hold_ticks = max(self._max_desk_hold_ticks, self._desk_hold_ticks)
            # Falls remain in the record; never reset or interrupt recovery.
            self._ever_fallen |= fallen
            if self.tick >= round(self.task.duration_s * 50):
                self._success = self._max_desk_hold_ticks >= 500 and not self._collision
                self._strict_result = {**status, 'success': self._success, 'max_desk_standing_s': self._max_desk_hold_ticks * .02,
                                       'ever_fallen': self._ever_fallen, 'protocol': 'climb_desk_bam_nominal_v1'}
                self._terminated = True
            return
        if self.task.suite == "community_perception":
            self._ever_fallen |= fallen
            self._terminated = self.tick >= round(self.task.duration_s * 50)
            self._success = self._terminated and not self._ever_fallen and not self._collision
            return
        if self.task.suite == "community_tricks":
            from .tricks import score_trick, CONES, SPACING
            end = self.tick >= round(self.task.duration_s * 50)
            if self.task.task_id == "cone_slalom":
                self._ever_fallen |= gravity[2] > -.5
                end |= self._ever_fallen or data.xpos[self.physics.trunk, 0] > (CONES + 2) * SPACING
            if end:
                self._strict_result = score_trick(self.task.task_id, self._trajectory, seed=self.seed, collision=self._collision)
                self._success = self._strict_result["success"]
                self._author_success = self._strict_result.get("author_terminal_metric")
                self._terminated = True
            return
        if self.task.suite == "community_basketball":
            ball = self._ball_positions()["basketball"]
            root = data.xpos[self.physics.trunk]
            # Exact released mdp.basketball_fell thresholds, free-ball hold=0.
            fell = bool(root[2] - ball[2] < 0.12 + 0.05 or np.linalg.norm(root[:2] - ball[:2]) > 0.16 or np.arccos(np.clip(-gravity[2], -1.0, 1.0)) > np.deg2rad(55.0))
            self._ever_fallen |= fell
            self._off_ball = fell
            self._terminated = fell or self.tick >= round(self.task.duration_s * 50)
            self._success = self._terminated and not self._ever_fallen and not self._collision
            self._termination_reason = "off_ball" if fell else "time_limit" if self._terminated else ""
            return
        if self.task.suite == "community_follow":
            person = self.physics._id(self.physics.mj.mjtObj.mjOBJ_BODY, "follow_person")
            distance = float(np.linalg.norm(data.xpos[person, :2] - data.xpos[self.physics.trunk, :2]))
            self._ever_fallen |= fallen
            in_band = self.task.stop_distance_m[0] <= distance <= self.task.stop_distance_m[1]
            self._follow_in_band_ticks += int(in_band)
            ready = in_band and upright and not self._collision and not self._ever_fallen
            self._goal_hold_ticks = self._goal_hold_ticks + 1 if ready else 0
            self._success = self._goal_hold_ticks >= round(self.task.stop_hold_s * 50)
            self._terminated = self._success or self._ever_fallen or self.tick >= round(self.task.duration_s * 50)
            return
        self._fallen_ticks = self._fallen_ticks + 1 if fallen else 0
        if self.task.suite == "visual_goal":
            self._ever_fallen = self._ever_fallen or fallen
        program = PROGRAMS.get(self.task.task_id)
        allowed = program is None or (self.tick - self._task_start_tick) * 0.02 >= program.recovery_after_s
        if self._recovery_active:
            self._recovery_upright_ticks = self._recovery_upright_ticks + 1 if upright else 0
            if self._recovery_upright_ticks >= 25:
                self._recovery_active = False
                self._task_start_tick = self.tick
        elif fallen and allowed and self.preset_clock != "program" and self.tick < round((self.task.duration_s - 1.5) * 50):
            self._recovery_active = True
            self._recovery_upright_ticks = 0
            self._recovery_events += 1
        if self.task.suite == "visual_goal":
            target = self._ball_positions()[self.task.target_color]
            distance = np.linalg.norm(target[:2] - data.xpos[self.physics.trunk, :2]) - 0.035
            speed = np.linalg.norm(data.qvel[self.physics.trunk_qvel : self.physics.trunk_qvel + 2])
            recent = np.asarray(self._trajectory[-11:])
            # Net heading motion over 200 ms rejects circling while tolerating
            # rapid balancing oscillations in the instantaneous IMU yaw rate.
            yaw_delta = recent[-1, 5] - recent[0, 5]
            yaw_speed = abs(np.arctan2(np.sin(yaw_delta), np.cos(yaw_delta))) / max(recent[-1, 0] - recent[0, 0], 0.02)
            ready = (
                self.task.stop_distance_m[0] <= distance <= self.task.stop_distance_m[1]
                and speed < 0.05
                and yaw_speed < 0.15
                and upright
                and not self._collision
                and not self._ever_fallen
                and self._recovery_events == 0
            )
            self._goal_hold_ticks = self._goal_hold_ticks + 1 if ready else 0
            self._success = self._goal_hold_ticks >= round(self.task.stop_hold_s * 50)
            if self._success or self._fallen_ticks >= 50:
                self._terminated = True
        if self.tick >= round(self.task.duration_s * 50):
            self._terminated = True
            if self.task.suite == "author_presets":
                self._author_success = self._preset_success()
                balls = self._ball_positions()
                displacement = float(np.linalg.norm(balls["kick_ball"][:2] - self._initial_ball["kick_ball"][:2])) if "kick_ball" in balls else 0.0
                self._strict_result = strict_preset_result(
                    self.task.task_id, np.asarray(self._trajectory), self._contact_events, collision=self._collision, recoveries=self._recovery_events, ball_displacement_m=displacement
                )
                self._success = self._strict_result["success"]

    def _preset_success(self):
        values = np.asarray(self._trajectory)
        stable = bool(np.mean((values[-38:, 4] < -0.72) & (values[-38:, 3] > 0.09)) >= 0.8 and values[-1, 4] < -0.72 and values[-1, 3] > 0.09)
        distance = float(np.linalg.norm(values[-1, 1:3] - values[0, 1:3]))
        min_height = float(values[:, 3].min())
        name = self.task.task_id
        if name in ("kick_left", "kick_right"):
            moved = np.linalg.norm(self._ball_positions()["kick_ball"][:2] - self._initial_ball["kick_ball"][:2]) > 0.03
        elif name == "roll_circle":
            moved = np.sum(np.abs(values[:, 6])) * 0.02 > 2.5
        elif name in ("ground_touch", "roller_crouch"):
            moved = min_height < values[0, 3] - 0.01
        elif name == "sit_walk":
            moved = min_height < values[0, 3] - 0.008
        else:
            moved = distance > 0.025
        return bool(stable and moved)

    def evaluator_state(self):
        if self.daemon_backend is not None:
            return self.daemon_backend.evaluator_state()
        if self.physics is None:
            raise RuntimeError("call reset before evaluator_state")
        values = np.asarray(self._trajectory)
        _, gravity, _, _ = self.physics.sensors()
        data = self.physics.data
        return {
            "episode_id": self.episode_id,
            "tick": self.tick,
            "sim_time_s": self.sim_time_s,
            "evaluator_source_sha256": EVALUATOR_SOURCE_SHA256,
            "tasks_source_sha256": TASKS_SOURCE_SHA256,
            "evaluation_origin_s": self._evaluation_origin_s,
            "evaluation_elapsed_s": self.sim_time_s - self._evaluation_origin_s,
            "robot_position": data.xpos[self.physics.trunk].copy(),
            "robot_yaw": float(values[-1, 5]),
            "robot_velocity": data.qvel[self.physics.trunk_qvel : self.physics.trunk_qvel + 3].copy(),
            "balls": self._ball_positions(),
            "obstacles": [vars(o).copy() for o in self.description.obstacles],
            "room_bounds": self.description.room_bounds,
            "target_color": self.task.target_color,
            "projected_gravity": gravity.copy(),
            "joint_targets": self.physics.last_targets.copy(),
            "applied_action": self.applied_action.copy(),
            "previous_raw_action": self.last_action.copy(),
            "motor_torques": data.ctrl[self.physics.actuator_indices].copy() if self.profile == "official_bam" else None,
            "success": self._success,
            "terminated": self._terminated,
            "fallen": bool(gravity[2] > -0.58 or values[-1, 3] < 0.073),
            "author_success": self._author_success,
            "strict_success": self._success,
            "strict_preset_result": copy.deepcopy(self._strict_result),
            "ever_fallen": self._ever_fallen,
            "contact_events": sorted(self._contact_events),
            "wheel_rotation_rad": self._wheel_rotation,
            "collision": self._collision,
            "fault": self._fault,
            "goal_hold_ticks": self._goal_hold_ticks,
            "follow_in_band_fraction": self._follow_in_band_ticks / max(self.tick, 1) if self.task.suite == "community_follow" else None,
            "off_ball": self._off_ball,
            "termination_reason": self._termination_reason,
            "recovery_active": self._recovery_active,
            "recovery_events": self._recovery_events,
            "task_start_tick": self._task_start_tick,
            "task_context": teacher_context(self.task, self.tick, recovery=self._recovery_active, task_start_tick=self._task_start_tick),
            "planar_displacement_m": float(np.linalg.norm(values[-1, 1:3] - values[0, 1:3])),
            "planar_path_length_m": float(np.linalg.norm(np.diff(values[:, 1:3], axis=0), axis=1).sum()),
            "minimum_height_m": float(values[:, 3].min()),
            "integrated_abs_pitch_rate_rad": float(np.sum(np.abs(values[:, 6])) * 0.02),
            "model_signature": self._signature,
            "actuator_profile": PROFILES[self.profile].actuator,
            "robot_signature": self._robot_signature.copy(),
            "scene": self.description.as_dict(),
            "action_delay_ticks": self.action_delay_ticks,
            "hardware_validated": False,
            "success_protocol": {
                "author_presets": "microduck_behavior_v1",
                "visual_goal": "visual_ball_stop_v2",
                "community_basketball": "community_basketball_nominal_apex_v1",
                "community_follow": "stationary_person_follow_distance_v1",
                "community_tricks": "community_tricks_bam_transfer_v1",
                "community_perception": "visible_duck_detection_iou_v1",
                "community_climb": "climb_desk_bam_nominal_v1",
            }[self.task.suite],
            "community_task_contract": (
                {
                    "ball_radius_m": 0.12,
                    "ball_mass_kg": 0.62,
                    "hold": 0.0,
                    "spawn": "nominal_apex_without_domain_randomization",
                    "source_revision": "aa5bd7909873a807fb9f4045f491519c443142e7",
                    "action_filter": "none",
                }
                if self.task.suite == "community_basketball"
                else None
            ),
        }

    def clone_state(self):
        if self.daemon_backend is not None:
            return self.daemon_backend.clone_state()
        names = (
            "seed",
            "task",
            "episode_id",
            "tick",
            "last_action",
            "applied_action",
            "_delay",
            "_recovery_active",
            "_recovery_upright_ticks",
            "_task_start_tick",
            "_recovery_events",
            "_fallen_ticks",
            "_goal_hold_ticks",
            "_success",
            "_author_success",
            "_strict_result",
            "_ever_fallen",
            "_contact_events",
            "_wheel_rotation",
            "_terminated",
            "_collision",
            "_fault",
            "_trajectory",
            "_initial_ball",
            "_initial_position",
            "_evaluation_origin_s",
            "_follow_in_band_ticks",
            "_off_ball",
            "_desk_hold_ticks",
            "_max_desk_hold_ticks",
            "_filtered_action",
            "_termination_reason",
        )
        return {
            "profile": self.profile,
            "scene": self.scene,
            "signature": self._signature,
            "action_delay_ticks": self.action_delay_ticks,
            "camera_offsamples": self.camera_offsamples,
            "physics": self.physics.clone_state(),
            "env": {name: copy.deepcopy(getattr(self, name)) for name in names},
            "rng": copy.deepcopy(self.rng.bit_generator.state),
            "policies": self.bank.clone_state(),
            # Keep the actual capture at this physical tick. A new rendering
            # context can differ from the originally measured first frame;
            # observations already consumed by the policy are part of a branch
            # anchor, not something to regenerate from a later renderer state.
            "camera_frames": {
                "tick": self.tick,
                "head": (self._rgb_cache.copy() if self._rgb_tick == self.tick
                         and self._rgb_cache is not None else None),
                "observer": (self._observer_rgb_cache.copy() if self._observer_rgb_tick == self.tick
                             and self._observer_rgb_cache is not None else None),
            },
        }

    snapshot = clone_state

    def restore_state(self, snapshot):
        if self.daemon_backend is not None:
            return self.daemon_backend.restore_state(snapshot)
        frames = snapshot.get("camera_frames")
        if frames is not None and frames["tick"] != snapshot["env"]["tick"]:
            raise ValueError("snapshot camera frames belong to another physical tick")
        if snapshot["profile"] != self.profile or snapshot["scene"] != self.scene or snapshot["action_delay_ticks"] != self.action_delay_ticks:
            raise ValueError("snapshot profile/scene/delay differs from destination environment")
        if snapshot.get("camera_offsamples") != self.camera_offsamples:
            raise ValueError("snapshot camera configuration differs from destination environment")
        if self.physics is None or self._signature != snapshot["signature"]:
            state = snapshot["env"]
            self.reset(state["seed"], state["task"], {"episode_id": state["episode_id"]})
        if self._signature != snapshot["signature"]:
            raise ValueError("snapshot model/asset signature differs from destination environment")
        self.physics.restore_state(snapshot["physics"])
        for name, value in snapshot["env"].items():
            setattr(self, name, copy.deepcopy(value))
        self.rng.bit_generator.state = copy.deepcopy(snapshot["rng"])
        self.bank.restore_state(snapshot["policies"])
        self._rgb_cache, self._rgb_tick = None, -1
        self._observer_rgb_cache, self._observer_rgb_tick = None, -1
        if frames is not None:
            if frames["head"] is not None:
                self._rgb_cache, self._rgb_tick = frames["head"].copy(), self.tick
            if frames["observer"] is not None:
                self._observer_rgb_cache, self._observer_rgb_tick = frames["observer"].copy(), self.tick
        return self.observe()

    restore = restore_state

    def close(self):
        if self.daemon_backend is not None:
            self.daemon_backend.close()
        if self.physics is not None:
            self.physics.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
