"""A bounded deployment stress profile, separate from official mjlab training DR.

Sensor amplitudes, IMU mounting and BAM friction scales use the pinned official
velocity recipe. Photometry, camera jitter and 0--40 ms action delay are explicit
LightX2V stress assumptions. This does not reproduce terrain, pushes, backlash,
mass/inertia, curriculum, or the complete official training distribution.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass, replace

import numpy as np

from .contracts import JointResidualChunk, SkillIntent
from .env import MicroDuckEnv


@dataclass(frozen=True)
class DeploymentNoiseConfig:
    name: str = "deployment_noise_v1"
    brightness_range: tuple[float, float] = (0.8, 1.2)
    camera_rotation_max_deg: float = 2.0
    camera_translation_max_m: float = 0.002
    imu_mounting_max_deg: float = 6.0
    encoder_bias_max_rad: float = 0.015
    gyro_white_max_rad_s: float = 0.03
    gravity_white_max: float = 0.01
    encoder_white_max_rad: float = 0.001
    velocity_white_max_rad_s: float = 0.25
    action_delay_max_ticks: int = 2
    bam_friction_scale_range: tuple[float, float] = (0.9, 1.1)


class RandomizedEnv:
    """Compose a real BAM env with replayable physics and sensor perturbations."""

    def __init__(self, env=None, *, config=None, **env_kwargs):
        self.env = env or MicroDuckEnv(**env_kwargs)
        if self.env.profile != "official_bam":
            raise ValueError("deployment_noise_v1 currently requires the real official_bam backend")
        self.config = config or DeploymentNoiseConfig()
        self.rng = np.random.default_rng(0)
        self._cache_key, self._cached_state, self._previous_clean = None, None, None
        self.sampled = {}

    def __getattr__(self, name):
        return getattr(self.env, name)

    @property
    def render_enabled(self):
        return self.env.render_enabled

    @render_enabled.setter
    def render_enabled(self, value):
        self.env.render_enabled = bool(value)

    def _rotation(self, maximum_deg):
        axis = self.rng.normal(size=3)
        axis /= max(np.linalg.norm(axis), 1e-12)
        angle = np.deg2rad(self.rng.uniform(-maximum_deg, maximum_deg))
        quat = np.r_[np.cos(angle / 2), axis * np.sin(angle / 2)]
        matrix = np.empty(9)
        self.env.physics.mj.mju_quat2Mat(matrix, quat)
        return quat, matrix.reshape(3, 3)

    def reset(self, seed=0, task="walk_forward", options=None):
        cfg = self.config
        if cfg.name != "deployment_noise_v1" or not 0 <= cfg.action_delay_max_ticks <= 50:
            raise ValueError("invalid deployment noise profile")
        self.rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0xD370]))
        self.env.action_delay_ticks = int(self.rng.integers(cfg.action_delay_max_ticks + 1))
        self.env.reset(seed, task, options)
        physics = self.env.physics
        _, self._imu_rotation = self._rotation(cfg.imu_mounting_max_deg)
        self._encoder_bias = self.rng.uniform(-cfg.encoder_bias_max_rad, cfg.encoder_bias_max_rad, 14)
        self._brightness = float(self.rng.uniform(*cfg.brightness_range))
        camera_delta, _ = self._rotation(cfg.camera_rotation_max_deg)
        camera_quat = np.zeros(4)
        physics.mj.mju_mulQuat(camera_quat, physics.model.cam_quat[physics.camera], camera_delta)
        camera_translation = self.rng.uniform(-cfg.camera_translation_max_m, cfg.camera_translation_max_m, 3)
        physics.model.cam_quat[physics.camera] = camera_quat
        physics.model.cam_pos[physics.camera] += camera_translation
        friction_scale = float(self.rng.uniform(*cfg.bam_friction_scale_range))
        self._bam_parameters = {}
        for name, parameter in vars(physics.bam_model).items():
            if (name.startswith("friction_") and name != "friction_viscous") or name.startswith("load_friction_"):
                parameter.value *= friction_scale
                self._bam_parameters[name] = float(parameter.value)
        physics.mj.mj_forward(physics.model, physics.data)
        self.env._rgb_tick = -1
        self._cache_key, self._cached_state, self._previous_clean = None, None, None
        self._imu_lag = np.zeros(2, dtype=int)
        self.sampled = {
            "profile": cfg.name,
            "config": asdict(cfg),
            "seed": int(seed),
            "action_delay_ticks": self.env.action_delay_ticks,
            "brightness_gain": self._brightness,
            "camera_delta_quaternion": camera_delta.tolist(),
            "camera_translation_delta_m": camera_translation.tolist(),
            "imu_rotation": self._imu_rotation.tolist(),
            "encoder_bias_rad": self._encoder_bias.tolist(),
            "bam_friction_scale": friction_scale,
            "range_source": "microduck_rl@2b581c641406a48346e696212930ea881c222c52/src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py",
            "not_full_official_dr": True,
        }
        self.sampled["sha256"] = hashlib.sha256(json.dumps(self.sampled, sort_keys=True).encode()).hexdigest()
        return self.observe()

    def _sense(self, observation):
        key = observation.episode_id, observation.sim_time_s
        cfg = self.config
        if key != self._cache_key:
            clean = observation.state.copy()
            previous = clean if self._previous_clean is None else self._previous_clean
            tick = round(observation.sim_time_s * 50)
            if tick % 64 == 0:
                self._imu_lag = self.rng.integers(0, 2, 2)
            state = clean.copy()
            for i, amplitude in ((0, cfg.gyro_white_max_rad_s), (1, cfg.gravity_white_max)):
                source = previous if self._imu_lag[i] else clean
                state[i * 3 : i * 3 + 3] = self._imu_rotation @ source[i * 3 : i * 3 + 3] + self.rng.uniform(-amplitude, amplitude, 3)
            state[6:20] += self._encoder_bias + self.rng.uniform(-cfg.encoder_white_max_rad, cfg.encoder_white_max_rad, 14)
            state[20:34] = previous[20:34] + self.rng.uniform(-cfg.velocity_white_max_rad_s, cfg.velocity_white_max_rad_s, 14)
            self._cache_key, self._cached_state, self._previous_clean = key, state.copy(), clean
        # Commands are task context, not a noisy sensor; preserve a newly begun
        # task window even if the physical-time sensor cache is unchanged.
        state = self._cached_state.copy()
        state[34:] = observation.state[34:]
        rgb = None if observation.rgb is None else self._photometry(observation.rgb)
        return replace(observation, state=state, rgb=rgb, camera_metadata={**observation.camera_metadata, "deployment_noise_profile": cfg.name})

    def _photometry(self, rgb):
        return np.clip(np.rint(np.asarray(rgb, np.float32) * self._brightness), 0, 255).astype(np.uint8)

    def observe(self):
        return self._sense(self.env.observe())

    def render(self):
        return self._photometry(self.env.render())

    def step(self, action):
        if self.env.terminated:
            raise RuntimeError("episode terminated; reset before stepping")
        if isinstance(action, JointResidualChunk):
            if action.episode_id != self.env.episode_id or action.source_tick > self.env.tick or action.source_time_s > self.env.sim_time_s + 1e-8:
                raise ValueError("cross-episode or future action chunk")
            if action.action_profile != "raw_residual":
                raise ValueError("decode/denormalize actions before stepping")
            offset = self.env.tick - action.source_tick
            stop = len(action.actions) if action.valid_until_tick is None else min(len(action.actions), action.valid_until_tick - action.source_tick)
            if offset >= stop:
                raise ValueError("expired action chunk or no unexecuted steps")
            rows, intent = action.actions[offset:stop], None
        elif isinstance(action, SkillIntent):
            if action.episode_id and action.episode_id != self.env.episode_id or action.source_tick > self.env.tick:
                raise ValueError("cross-episode or future skill intent")
            rows, intent = range(action.duration_ticks), action
        else:
            rows, intent = [action], None
        rendering = self.render_enabled
        try:
            self.render_enabled = False
            for row in rows:
                if intent is not None:
                    state = self.observe().state.copy()
                    state[48:] = intent.command
                    row = self.env.bank.infer(intent.skill, state)
                self._sense(self.env.step(row))
                if self.env.terminated:
                    break
        finally:
            self.render_enabled = rendering
        return self.observe()

    def evaluator_state(self):
        return {**self.env.evaluator_state(), "deployment_randomization": copy.deepcopy(self.sampled)}

    def clone_state(self):
        physics = self.env.physics
        return {
            "wrapper": self.config.name,
            "config": asdict(self.config),
            "env": self.env.clone_state(),
            "sampled": copy.deepcopy(self.sampled),
            "rng": copy.deepcopy(self.rng.bit_generator.state),
            "cache_key": self._cache_key,
            "cached_state": copy.deepcopy(self._cached_state),
            "previous_clean": copy.deepcopy(self._previous_clean),
            "imu_lag": self._imu_lag.copy(),
            "imu_rotation": self._imu_rotation.copy(),
            "encoder_bias": self._encoder_bias.copy(),
            "brightness": self._brightness,
            "bam_parameters": self._bam_parameters.copy(),
            "camera_quat": physics.model.cam_quat[physics.camera].copy(),
            "camera_pos": physics.model.cam_pos[physics.camera].copy(),
        }

    snapshot = clone_state

    def restore_state(self, snapshot):
        if snapshot["wrapper"] != self.config.name or snapshot["config"] != asdict(self.config):
            raise ValueError("snapshot belongs to a different deployment noise profile")
        self.env.action_delay_ticks = snapshot["env"]["action_delay_ticks"]
        self.env.restore_state(snapshot["env"])
        physics = self.env.physics
        physics.model.cam_quat[physics.camera] = snapshot["camera_quat"]
        physics.model.cam_pos[physics.camera] = snapshot["camera_pos"]
        self._bam_parameters = snapshot["bam_parameters"].copy()
        for name, value in self._bam_parameters.items():
            getattr(physics.bam_model, name).value = value
        self.sampled = copy.deepcopy(snapshot["sampled"])
        self.rng.bit_generator.state = copy.deepcopy(snapshot["rng"])
        for name in ("cache_key", "cached_state", "previous_clean", "imu_lag", "imu_rotation", "encoder_bias", "brightness"):
            setattr(self, "_" + name, copy.deepcopy(snapshot[name]))
        self.env._rgb_tick = -1
        return self.observe()

    restore = restore_state

    def close(self):
        self.env.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
