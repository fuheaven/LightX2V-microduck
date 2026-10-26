"""CPU MuJoCo physics with the actual upstream BAM M6 actuator.

BAM setup follows pollen-robotics/microduck_rl scripts/infer_policy.py at
2b581c641406a48346e696212930ea881c222c52 (Apache-2.0). The author compatibility
profile intentionally uses the earlier MJCF position actuators, not BAM.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from dataclasses import dataclass

import numpy as np

from .contracts import HOME_POLICY, JOINT_NAMES, PHYSICS_DT_S, POLICY_JOINT_NAMES
from .rendering import validate_camera_offsamples

BAM_REVISION = "62bd8ce12154340be97e06f7f41a0ca8f116d967"
AUTHOR_CAMERA_QUAT = np.asarray((0.7010574, -0.0922960, 0.0922960, -0.7010574), dtype=np.float64)


@dataclass(frozen=True)
class PhysicsProfile:
    name: str
    actuator: str
    rl_revision: str
    camera: str
    voltage: float = 7.4
    voltage_drop_gain: float = 0.1
    minimum_voltage: float = 6.0
    kp_fw: float = 200.0


PROFILES = {
    "official_bam": PhysicsProfile("official_bam", "bam_m6_xl330", "2b581c641406a48346e696212930ea881c222c52", "official_daemon_corrected_cw90"),
    "tinyvla_author_compat": PhysicsProfile("tinyvla_author_compat", "mjcf_position_pd", "d424a0c899f6b33cbd3daeb279913134349c0b63", "author_stage25_head_quaternion"),
    "official_daemon": PhysicsProfile("official_daemon", "robotd_duck_body_transport", "2b581c641406a48346e696212930ea881c222c52", "official_mediad"),
}


class Physics:
    def __init__(self, spec, *, profile="official_bam", robot_prefix="", image_size=256, tof_source=None, camera_offsamples=None):
        import mujoco

        self.mj = mujoco
        self.profile = PROFILES[profile]
        if profile == "official_daemon":
            raise ValueError("official_daemon requires the real duck-sim/robotd transport backend; it is not a local PD physics profile")
        self.robot_prefix = robot_prefix
        self.image_size = int(image_size)
        self.camera_offsamples = validate_camera_offsamples(camera_offsamples)
        self.renderer = None
        self.observer_renderer = None
        self.observer_camera = None
        self.bam_controller = None
        self.bam_model = None
        wanted = {robot_prefix + name for name in POLICY_JOINT_NAMES}
        actuator_by_joint = {}
        for actuator in spec.actuators:
            target = actuator.target
            joint_name = target.name if hasattr(target, "name") else str(target)
            if joint_name in wanted:
                actuator_by_joint[joint_name] = actuator
        if set(actuator_by_joint) != wanted:
            raise ValueError(f"robot binding is missing policy joints: {sorted(wanted - set(actuator_by_joint))}")
        names = [actuator_by_joint[robot_prefix + n].name for n in POLICY_JOINT_NAMES]
        if profile == "official_bam":
            try:
                from bam.model import load_model
                from bam.mujoco import MujocoController
            except ImportError as exc:
                raise RuntimeError(f"official_bam needs better-actuator-models from Rhoban/bam@{BAM_REVISION}; refusing a PD fallback") from exc
            self.bam_model = load_model(motor_name="xl330", model="m6")
            actuator = self.bam_model.actuator
            actuator.kp, actuator.vin, actuator.max_current = self.profile.kp_fw, self.profile.voltage, None
            force_limit = actuator.vin * self.bam_model.kt.value / self.bam_model.R.value
            for joint_name, act in actuator_by_joint.items():
                act.set_to_motor()
                act.forcelimited = True
                act.forcerange = (-force_limit, force_limit)
                act.ctrllimited = False
                act.gear = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                joint = next(j for j in spec.joints if j.name == joint_name)
                joint.damping = np.zeros((3, 1))
                joint.frictionloss = 0.0
                joint.solref_friction = (-5e4, -2e2)
                joint.solimp_friction = (0.99, 0.9999, 0.001, 0.5, 2.0)
        self.model = spec.compile()
        if self.camera_offsamples is not None:
            self.model.vis.quality.offsamples = self.camera_offsamples
        self.model.opt.timestep = PHYSICS_DT_S
        self.data = mujoco.MjData(self.model)
        if profile == "official_bam":
            self.bam_controller = MujocoController(self.bam_model, names, self.model, self.data, vin_drop_gain=self.profile.voltage_drop_gain, vin_min=self.profile.minimum_voltage)
        self.actuator_indices = np.asarray([self._id(mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in names], dtype=np.int32)
        joint_ids = np.asarray([self._id(mujoco.mjtObj.mjOBJ_JOINT, robot_prefix + n) for n in POLICY_JOINT_NAMES], dtype=np.int32)
        self.qpos_indices = self.model.jnt_qposadr[joint_ids].copy()
        self.qvel_indices = self.model.jnt_dofadr[joint_ids].copy()
        self.trunk = self._id(mujoco.mjtObj.mjOBJ_BODY, robot_prefix + "trunk_base")
        self.free_joint = self._id(mujoco.mjtObj.mjOBJ_JOINT, robot_prefix + "trunk_base_freejoint")
        self.trunk_qpos = int(self.model.jnt_qposadr[self.free_joint])
        self.trunk_qvel = int(self.model.jnt_dofadr[self.free_joint])
        self.gyro = self._id(mujoco.mjtObj.mjOBJ_SENSOR, robot_prefix + "imu_ang_vel")
        if self.model.sensor_dim[self.gyro] != 3:
            raise ValueError("gyro must have dimension 3")
        self.camera = self._id(mujoco.mjtObj.mjOBJ_CAMERA, robot_prefix + "head_camera")
        if profile == "tinyvla_author_compat":
            self.model.cam_quat[self.camera] = AUTHOR_CAMERA_QUAT
        else:
            # Exactly the correction in upstream sim/camera.py: the raw MJCF
            # camera points backwards; mediad then rotates its mounted image.
            fixed = np.zeros(4)
            mujoco.mju_mulQuat(fixed, self.model.cam_quat[self.camera], np.array((0.0, 1.0, 0.0, 0.0)))
            self.model.cam_quat[self.camera] = fixed
        self.last_targets = np.asarray(HOME_POLICY, dtype=np.float32).astype(np.float64)
        self.last_contacts = []
        self.tof_sensor = None
        self._tof_cache, self._tof_metadata, self._tof_bucket = None, {}, -1
        site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, robot_prefix + "tof")
        if site >= 0 and tof_source is not None:
            module_spec = importlib.util.spec_from_file_location("lightx2v_pinned_microduck_tof", tof_source)
            module = importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(module)
            self.tof_sensor = module.Tof(self.model, site)

    def _id(self, kind, name):
        value = self.mj.mj_name2id(self.model, kind, name)
        if value < 0:
            raise ValueError(f"required robot binding {name!r} missing")
        return int(value)

    def reset(self, *, robot_xy=(0.0, 0.0), robot_yaw=0.0, robot_height=0.125, ball_positions=None, kick_side=None, free_bodies=None, seed=0):
        self.mj.mj_resetData(self.model, self.data)
        q = self.trunk_qpos
        self.data.qpos[q : q + 7] = (*robot_xy, robot_height, np.cos(robot_yaw / 2), 0.0, 0.0, np.sin(robot_yaw / 2))
        self.data.qpos[self.qpos_indices] = np.asarray(HOME_POLICY, dtype=np.float32)
        for name, home in zip(POLICY_JOINT_NAMES, HOME_POLICY):
            joint = self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_JOINT, 'target_' + name)
            if joint >= 0:
                self.data.qpos[self.model.jnt_qposadr[joint]] = home
                for actuator in range(self.model.nu):
                    if self.model.actuator_trnid[actuator, 0] == joint:
                        self.data.ctrl[actuator] = home
        for color, pos in (ball_positions or {}).items():
            self.set_free_body(f"lx_ball_{color}_free", pos)
        for name, pos in (free_bodies or {}).items():
            self.set_free_body(name, pos)
        if kick_side:
            offset = np.array((0.13, 0.042 if kick_side == "left" else -0.042))
            c, s = np.cos(robot_yaw), np.sin(robot_yaw)
            offset = np.array(((c, -s), (s, c))) @ offset + robot_xy
            self.set_free_body("ball_free", (*offset, 0.035))
        self.last_targets = np.asarray(HOME_POLICY, dtype=np.float32).astype(np.float64)
        if self.bam_controller is not None:
            self.bam_controller.reset(self.data.qpos)
            self.bam_controller.last_ts = self.data.time
            self.bam_controller.q_target[:] = self.last_targets
        else:
            self.data.ctrl[self.actuator_indices] = self.last_targets
        self.mj.mj_forward(self.model, self.data)
        self._tof_cache, self._tof_metadata, self._tof_bucket = None, {}, -1
        if self.tof_sensor is not None:
            self.tof_sensor.random = np.random.default_rng(int(seed))

    def set_free_body(self, joint_name, position):
        joint = self._id(self.mj.mjtObj.mjOBJ_JOINT, joint_name)
        if self.model.jnt_type[joint] != self.mj.mjtJoint.mjJNT_FREE:
            raise ValueError(f"{joint_name} is not free")
        start = int(self.model.jnt_qposadr[joint])
        self.data.qpos[start : start + 7] = (*position, 1.0, 0.0, 0.0, 0.0)
        velocity = int(self.model.jnt_dofadr[joint])
        self.data.qvel[velocity : velocity + 6] = 0.0

    def step(self, targets) -> None:
        targets = np.asarray(targets, dtype=np.float64)
        if targets.shape != (14,) or not np.isfinite(targets).all():
            raise ValueError("joint targets must be finite 14D")
        self.last_targets = targets.copy()
        if self.bam_controller is not None:
            self.bam_controller.q_target[:] = targets
        else:
            self.data.ctrl[self.actuator_indices] = targets
        self.last_contacts = []
        for _ in range(4):
            if self.bam_controller is not None:
                self.bam_controller.update()
            self.mj.mj_step(self.model, self.data)
            self.last_contacts.extend((int(c.geom1), int(c.geom2)) for c in self.data.contact[: self.data.ncon])
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise FloatingPointError("MuJoCo produced non-finite physical state")

    def sensors(self):
        adr = int(self.model.sensor_adr[self.gyro])
        gyro = self.data.sensordata[adr : adr + 3].copy()
        rotation = np.zeros(9)
        self.mj.mju_quat2Mat(rotation, self.data.xquat[self.trunk])
        gravity = -rotation.reshape(3, 3).T[:, 2]
        pos, vel = np.zeros(15), np.zeros(15)
        for index, name in enumerate(JOINT_NAMES):
            if name != "mouth":
                k = POLICY_JOINT_NAMES.index(name)
                pos[index], vel[index] = self.data.qpos[self.qpos_indices[k]], self.data.qvel[self.qvel_indices[k]]
        return gyro, gravity, pos, vel

    def head_renderer(self):
        """Initialize the camera context without replacing a captured frame."""
        if self.renderer is None:
            self.renderer = self.mj.Renderer(self.model, height=self.image_size, width=self.image_size)
            if (self.camera_offsamples is not None
                    and self.renderer._mjr_context.offSamples != self.camera_offsamples):
                self.renderer.close()
                self.renderer = None
                raise RuntimeError("renderer did not apply the requested camera_offsamples")
        return self.renderer

    def camera_configuration(self):
        """Report the actual context after capture; do not initialize a renderer."""
        return dict(schema="microduck_camera_rendering_v1", requested_offsamples=self.camera_offsamples,
                    model_offsamples=int(self.model.vis.quality.offsamples),
                    context_offsamples=(None if self.renderer is None else int(self.renderer._mjr_context.offSamples)),
                    image_size=self.image_size, backend=os.environ.get("MUJOCO_GL", "auto"),
                    mujoco_version=self.mj.__version__, orientation=self.profile.camera)

    def render(self):
        renderer = self.head_renderer()
        renderer.update_scene(self.data, camera=self.camera)
        rgb = renderer.render().copy()
        if self.profile.name == "official_bam":
            rgb = np.rot90(rgb, k=-1).copy()
        return rgb

    def render_observer(self):
        """Third-person review pixels, never a robot sensor or policy input."""
        if self.observer_renderer is None:
            size = max(384, self.image_size)
            self.observer_renderer = self.mj.Renderer(self.model, height=size, width=size)
            self.observer_camera = self.mj.MjvCamera()
            self.mj.mjv_defaultCamera(self.observer_camera)
            self.observer_camera.type = self.mj.mjtCamera.mjCAMERA_FREE
            self.observer_camera.distance = 1.0
            self.observer_camera.azimuth = 135.0
            self.observer_camera.elevation = -25.0
        # A world-facing camera follows the trunk position so turning, rolling
        # and foot motion remain visible alongside nearby targets/obstacles.
        self.observer_camera.lookat[:] = self.data.xpos[self.trunk]
        self.observer_camera.lookat[2] = max(float(self.observer_camera.lookat[2]), 0.10) + 0.04
        self.observer_renderer.update_scene(self.data, camera=self.observer_camera)
        return self.observer_renderer.render().copy()

    def camera_metadata(self):
        fovy = float(self.model.cam_fovy[self.camera])
        focal = self.image_size / (2 * np.tan(np.radians(fovy) / 2))
        trunk_rotation = self.data.xmat[self.trunk].reshape(3, 3)
        camera_rotation = self.data.cam_xmat[self.camera].reshape(3, 3)
        # A relative sensor transform is robot kinematics (the simulated TF
        # counterpart), not a world pose or privileged object measurement.
        relative_rotation = trunk_rotation.T @ camera_rotation
        relative_position = trunk_rotation.T @ (self.data.cam_xpos[self.camera] - self.data.xpos[self.trunk])
        return {
            "profile": self.profile.camera,
            "width": self.image_size,
            "height": self.image_size,
            "fovy_degrees": fovy,
            "fx": focal,
            "fy": focal,
            "cx": self.image_size / 2,
            "cy": self.image_size / 2,
            "canonical_rotation_cw": 90 if self.profile.name == "official_bam" else 0,
            "source": "MuJoCo head_camera",
            "physical_camera_calibrated": False,
            "camera_to_trunk_rotation": relative_rotation.tolist(),
            "camera_to_trunk_translation": relative_position.tolist(),
            "camera_axes": "opengl_right_up_backward",
        }

    def robot_signature(self):
        """Canonical per-body dynamics identity, independent of scene indices."""
        m, mj = self.model, self.mj
        records = []
        for body in range(1, m.nbody):
            ancestor = body
            while ancestor and ancestor != self.trunk:
                ancestor = int(m.body_parentid[ancestor])
            if ancestor != self.trunk:
                continue
            name = mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, body).removeprefix(self.robot_prefix)
            geoms = []
            for geom in range(int(m.body_geomadr[body]), int(m.body_geomadr[body] + m.body_geomnum[body])):
                geoms.append(
                    {k: np.asarray(getattr(m, k)[geom]).tolist() for k in ("geom_type", "geom_pos", "geom_quat", "geom_size", "geom_friction", "geom_contype", "geom_conaffinity", "geom_condim")}
                )
            records.append(
                {"name": name, "mass": float(m.body_mass[body]), "inertia": m.body_inertia[body].tolist(), "pos": m.body_pos[body].tolist(), "quat": m.body_quat[body].tolist(), "geoms": geoms}
            )
        records.sort(key=lambda r: r["name"])
        signature = hashlib.sha256(json.dumps(records, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return {"sha256": signature, "bodies": len(records), "profile": self.profile.name, "normalized_prefix": self.robot_prefix}

    def read_tof(self, tick):
        if self.tof_sensor is None:
            return None, {"available": False, "reason": "missing pinned sensor implementation or MJCF site"}
        bucket = int(tick) * 15 // 50
        if bucket != self._tof_bucket:
            distances, statuses = self.tof_sensor.frame(self.data)
            status = np.asarray(statuses, dtype=np.uint8).reshape(8, 8)
            meters = np.asarray(distances, dtype=np.float32).reshape(8, 8) / 1000.0
            self._tof_cache = np.where(status == 5, np.minimum(meters, 4.0), 4.0).astype(np.float32)
            self._tof_metadata = {
                "available": True,
                "capture_tick": int(tick),
                "capture_time_s": float(self.data.time),
                "statuses": status.tolist(),
                "valid_status": 5,
                "no_target_status": 255,
                "max_range_m": 4.0,
                "rate_hz": 15,
                "units": "m",
                "source": "official_tof_mj_ray",
            }
            self._tof_bucket = bucket
        metadata = {**self._tof_metadata, "frame_age_s": (int(tick) - self._tof_metadata["capture_tick"]) * 0.02}
        return self._tof_cache.copy(), metadata

    def clone_state(self):
        data = self.mj.MjData(self.model)
        self.mj.mj_copyData(data, self.model, self.data)
        result = {
            "data": data,
            "targets": self.last_targets.copy(),
            "frictionloss": self.model.dof_frictionloss.copy(),
            "damping": self.model.dof_damping.copy(),
            "last_contacts": list(self.last_contacts),
        }
        result["tof"] = {
            "cache": copy.deepcopy(self._tof_cache),
            "metadata": copy.deepcopy(self._tof_metadata),
            "bucket": self._tof_bucket,
            "rng": None if self.tof_sensor is None else copy.deepcopy(self.tof_sensor.random.bit_generator.state),
        }
        if self.bam_controller is not None:
            ctrl = self.bam_controller
            result["bam"] = {"q_target": ctrl.q_target.copy(), "prev_torque": ctrl._prev_motor_torque.copy(), "last_ts": ctrl.last_ts, "actuator": copy.deepcopy(ctrl.model.actuator.__dict__)}
        return result

    def restore_state(self, snapshot):
        self.model.dof_frictionloss[:] = snapshot["frictionloss"]
        self.model.dof_damping[:] = snapshot["damping"]
        # Full MjData includes contact solver warm starts/constraint forces used
        # by BAM. Reconstructing qpos/qvel alone is not a DAgger state clone.
        self.mj.mj_copyData(self.data, self.model, snapshot["data"])
        self.last_targets = snapshot["targets"].copy()
        self.last_contacts = list(snapshot["last_contacts"])
        tof = snapshot["tof"]
        self._tof_cache, self._tof_metadata, self._tof_bucket = copy.deepcopy(tof["cache"]), copy.deepcopy(tof["metadata"]), tof["bucket"]
        if self.tof_sensor is not None and tof["rng"] is not None:
            self.tof_sensor.random.bit_generator.state = copy.deepcopy(tof["rng"])
        if self.bam_controller is not None:
            state, ctrl = snapshot["bam"], self.bam_controller
            ctrl.q_target = state["q_target"].copy()
            ctrl._prev_motor_torque = state["prev_torque"].copy()
            ctrl.last_ts = state["last_ts"]
            ctrl.model.actuator.__dict__.clear()
            ctrl.model.actuator.__dict__.update(copy.deepcopy(state["actuator"]))
        # MjData does not include the renderer's scene/context history. After
        # jumping back from a rendered branch, reusing that history can change
        # the first RGB frame at an otherwise identical physical state.
        # Recreate both cameras lazily; do not advance or recompute physics.
        self._close_renderers()

    def _close_renderers(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        if self.observer_renderer is not None:
            self.observer_renderer.close()
            self.observer_renderer = None
            self.observer_camera = None

    def close(self):
        self._close_renderers()
