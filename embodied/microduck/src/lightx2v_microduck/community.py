"""Pinned community ONNX adapters with explicit observation/action contracts.

Finite ONNX output is a loader check, not a reproduced behavior. In particular,
follow-v4 needs its upstream 80D sensor/temporal encoder, and basketball needs
its ball-apex start and unfiltered control profile.
"""

from __future__ import annotations

import copy
import importlib
import json
import math
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

from .assets import resolve_asset_root, sha256_file
from .contracts import JointResidualChunk, Observation, PolicyManifest, SkillIntent

CATALOG_PATH = Path(__file__).parent / "resources" / "community-policies.json"


def load_catalog(path: str | Path | None = None) -> dict[str, Any]:
    catalog = json.loads(Path(path or CATALOG_PATH).read_text())
    if catalog.get("schema_version") != 1:
        raise ValueError("unsupported community policy catalog")
    return catalog["policies"]


def community_cache_root(asset_root=None) -> Path:
    return resolve_asset_root(asset_root).parent / "community-policies"


def verify_community_policy(policy_id, asset_root=None):
    record = load_catalog()[policy_id]
    root = community_cache_root(asset_root) / policy_id
    errors = []
    for name, artifact in {**record['files'], **record.get('generated_files', {})}.items():
        path = root / name
        if not path.is_file():
            errors.append({'file': name, 'reason': 'missing'})
        elif sha256_file(path) != artifact['sha256']:
            errors.append({'file': name, 'reason': 'sha256_mismatch'})
    return {'policy_id': policy_id, 'revision': record['revision'], 'ok': not errors, 'errors': errors}


def fetch_community_policy(policy_id: str, asset_root=None, *, catalog_path=None) -> dict[str, Any]:
    record = load_catalog(catalog_path).get(policy_id)
    if record is None:
        raise ValueError(f"unknown community policy: {policy_id}")
    destination = community_cache_root(asset_root) / policy_id
    destination.mkdir(parents=True, exist_ok=True)
    for filename, artifact in record["files"].items():
        path = destination / filename
        if Path(filename).name != filename:
            raise ValueError("community artifacts must be plain filenames")
        if path.exists():
            if sha256_file(path) != artifact["sha256"]:
                raise ValueError(f"existing community artifact hash mismatch: {path}")
            continue
        with tempfile.NamedTemporaryFile(prefix=".download-", dir=destination) as temporary:
            request = urllib.request.Request(artifact["url"], headers={"User-Agent": "lightx2v-microduck/0.1"})
            with urllib.request.urlopen(request, timeout=90) as response:
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    temporary.write(block)
            temporary.flush()
            if sha256_file(Path(temporary.name)) != artifact["sha256"]:
                raise ValueError(f"downloaded community artifact hash mismatch: {filename}")
            path.write_bytes(Path(temporary.name).read_bytes())
    return {
        "policy_id": policy_id,
        "revision": record["revision"],
        "model_path": str(destination / record["model_file"]),
        "sha256": record["files"][record["model_file"]]["sha256"],
        "license": record["license"],
    }


class CommunityPolicy:
    """One inference session, including LSTM state for exactly one robot."""

    def __init__(self, policy_id: str, asset_root=None, *, catalog_path=None):
        import onnxruntime as ort

        catalog = load_catalog(catalog_path)
        if policy_id not in catalog:
            raise ValueError(f"unknown community policy: {policy_id}")
        self.record = catalog[policy_id]
        if 'adapter' in self.record:
            raise ValueError(f"{policy_id} requires its dedicated {self.record['adapter']} adapter")
        root = community_cache_root(asset_root) / policy_id
        self.model_path = root / self.record["model_file"]
        if not self.model_path.exists():
            raise FileNotFoundError(f"missing community weights; call fetch_community_policy({policy_id!r})")
        model_hash = self.record["files"][self.record["model_file"]]["sha256"]
        if sha256_file(self.model_path) != model_hash:
            raise ValueError("community model hash mismatch")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(self.model_path), sess_options=options, providers=["CPUExecutionProvider"])
        recurrent = self.record.get("recurrent")
        self.state_pairs = recurrent["pairs"] if recurrent else []
        inputs = {self.record["observation"]["name"]: self.record["observation"]["shape"]}
        outputs = {self.record["action"]["name"]: self.record["action"]["shape"]}
        for pair in self.state_pairs:
            inputs[pair["input"]] = pair["shape"]
            outputs[pair["output"]] = pair["shape"]
        for actual, expected, direction in ((self.session.get_inputs(), inputs, "input"), (self.session.get_outputs(), outputs, "output")):
            if set(node.name for node in actual) != set(expected):
                raise ValueError(f"community ONNX {direction} names differ from pinned contract")
            if any(list(node.shape) != expected[node.name] or node.type != "tensor(float)" for node in actual):
                raise ValueError(f"community ONNX {direction} shapes/dtypes differ from pinned contract")
        self.manifest = PolicyManifest(
            policy_id=policy_id,
            action_kind=self.record["action"]["kind"],
            observation_profile=self.record["observation"]["profile"],
            control_profile="community_unfiltered_raw" if policy_id == "basketball" else "community_declared",
            embodiment=self.record["embodiment"],
            recurrent=bool(recurrent),
            uses_privileged_state=bool(self.record.get("uses_privileged_state")),
            artifacts={"checkpoint_sha256": model_hash},
            metadata={
                "source": self.record["source"],
                "revision": self.record["revision"],
                "required_scene": self.record.get("required_scene"),
                "requires_external_observation_encoder": self.record["observation"]["shape"][-1] != 61,
            },
        )
        self.reset()

    def reset(self, episode_id: str = "") -> None:
        self.episode_id = str(episode_id)
        self.last_tick = -1
        self.state = {pair["input"]: np.zeros(pair["shape"], np.float32) for pair in self.state_pairs}

    def infer(self, observation: np.ndarray) -> np.ndarray:
        values = np.asarray(observation, np.float32)
        shape = tuple(self.record["observation"]["shape"])
        if values.shape == shape[1:]:
            values = values[None]
        if values.shape != shape or not np.isfinite(values).all():
            raise ValueError(f"{self.record['policy_id']} requires finite observation shape {shape}")
        for first, last in self.record.get("zero_command_slices", []):
            if np.any(values[:, first:last] != 0):
                raise ValueError(f"{self.record['policy_id']} requires zero command slots {first}:{last}")
        feed = {self.record["observation"]["name"]: values, **self.state}
        names = [self.record["action"]["name"], *[pair["output"] for pair in self.state_pairs]]
        returned = self.session.run(names, feed)
        if not all(np.isfinite(value).all() for value in returned):
            raise FloatingPointError("community policy returned nonfinite action or recurrent state")
        action = returned[0][0].copy()
        if "bounds_low" in self.record["action"]:
            action = np.clip(action, self.record["action"]["bounds_low"], self.record["action"]["bounds_high"]).astype(np.float32)
        for pair, value in zip(self.state_pairs, returned[1:]):
            self.state[pair["input"]] = value.copy()
        return action

    def clone_state(self) -> dict[str, Any]:
        return {"episode_id": self.episode_id, "last_tick": self.last_tick, "states": {name: value.copy() for name, value in self.state.items()}}

    def restore_state(self, snapshot: dict[str, Any]) -> None:
        if set(snapshot["states"]) != set(self.state):
            raise ValueError("recurrent state names differ")
        state = {}
        for name, reference in self.state.items():
            value = np.asarray(snapshot["states"][name], np.float32)
            if value.shape != reference.shape or not np.isfinite(value).all():
                raise ValueError("recurrent state shape/value mismatch")
            state[name] = value.copy()
        self.state = state
        self.episode_id, self.last_tick = str(snapshot["episode_id"]), int(snapshot["last_tick"])

    def predict(self, observation: Observation) -> JointResidualChunk:
        if self.record["observation"]["shape"] != [1, 61] or self.record["action"]["shape"] != [1, 14]:
            raise ValueError("this community policy needs its own sensor/intent adapter; a 61D joint-policy adapter is not valid")
        if observation.episode_id != self.episode_id:
            self.reset(observation.episode_id)
        if observation.tick <= self.last_tick:
            raise ValueError("policy observation ticks must strictly increase")
        state = observation.state.copy()
        for first, last in self.record.get("zero_command_slices", []):
            state[first:last] = 0
        raw = self.infer(state)
        self.last_tick = observation.tick
        return JointResidualChunk(
            raw[None],
            observation.episode_id,
            observation.tick,
            observation.sim_time_s,
            valid_until_tick=observation.tick + 1,
            diagnostics={"policy_id": self.record["policy_id"], "required_scene": self.record.get("required_scene"), "hardware_validated": False},
        )


def _follow_modules(asset_root=None):
    """Import the actual pinned encoder/tracker, without vendoring a variant."""
    root = resolve_asset_root(asset_root) / "microduck-lab" / "microduck_local" / "src"
    record = load_catalog()["follow-v4"]
    for relative, expected in record["encoder_files"].items():
        path = root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"follow-v4 pinned encoder source mismatch or absent: {path}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        modules = {name: importlib.import_module(f"microduck_local.{name}") for name in ("brain.brain_env", "brain.runtime", "sensors.detector", "sensors.tof")}
    except ModuleNotFoundError as error:
        raise ImportError("follow-v4 requires its pinned microduck-lab sources and gymnasium==1.3.0") from error
    for module in modules.values():
        if not Path(module.__file__).resolve().is_relative_to(root.resolve()):
            raise ValueError("another microduck_local installation shadowed the pinned encoder")
    return modules


class FollowOraclePolicy:
    """Real follow-v4 + upstream ObsBuilder v2 on geometric detections/true odom.

    This is the privileged community comparison group. It is neither an RGB
    detector nor a VLA, and is never eligible for the non-oracle visual score.
    Robot geometry is untouched: camera axes are adapted to the upstream
    detector's x-forward/y-left/z-up interface.
    """

    def __init__(self, env, asset_root=None, *, target_body="follow_person", target_radius=0.20, target_height=1.6):
        self.env = env
        self.asset_root = asset_root or env.asset_root
        self.network = CommunityPolicy("follow-v4", self.asset_root)
        self.modules = _follow_modules(self.asset_root)
        self.builder = self.modules["brain.brain_env"].ObsBuilder("person", version=2)
        self.target_body, self.target_radius, self.target_height = target_body, float(target_radius), float(target_height)
        self.manifest = PolicyManifest(
            policy_id="follow-v4-oracle",
            action_kind="skill_intent",
            observation_profile="lab_obs_v2_80d_oracle_detector_true_odom",
            control_profile="official_walker_skill",
            horizon=5,
            execution_prefix=5,
            uses_privileged_state=True,
            artifacts=self.network.manifest.artifacts,
            metadata={
                "source": self.network.record["source"],
                "revision": self.network.record["revision"],
                "encoder": "upstream ObsBuilder v2 and Tracker",
                "perception": "geometric_oracle",
                "odometry": "simulator_truth",
                "requires_rgb": False,
                "required_scene": "community_follow",
                "hardware_validated": False,
            },
        )
        self.reset()

    def reset(self, episode_id=""):
        self.network.reset(episode_id)
        self.builder.reset()
        self.episode_id, self.last_tick = str(episode_id), -1
        self.last_action = np.zeros(3, np.float32)
        self.last_observation = None

    def clone_state(self):
        return {
            "network": self.network.clone_state(),
            "builder": copy.deepcopy(self.builder),
            "episode_id": self.episode_id,
            "last_tick": self.last_tick,
            "last_action": self.last_action.copy(),
            "last_observation": copy.deepcopy(self.last_observation),
        }

    def restore_state(self, state):
        self.network.restore_state(state["network"])
        self.builder = copy.deepcopy(state["builder"])
        self.episode_id, self.last_tick = state["episode_id"], state["last_tick"]
        self.last_action, self.last_observation = state["last_action"].copy(), copy.deepcopy(state["last_observation"])

    def _senses(self, observation):
        physics = self.env.physics
        if physics is None:
            raise RuntimeError("reset the environment before follow-v4 inference")
        mj, model, data = physics.mj, physics.model, physics.data
        detector_module = self.modules["sensors.detector"]
        target_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, self.target_body)
        if target_id < 0:
            raise ValueError(f"follow-v4 requires the explicit person body {self.target_body!r}")
        rotation_gl = data.cam_xmat[physics.camera].reshape(3, 3)
        # mediad-compatible clockwise image rotation changes the image axes,
        # but preserves the optical forward direction.
        if observation.camera_metadata.get("canonical_rotation_cw") == 90:
            rotation = np.column_stack((-rotation_gl[:, 2], -rotation_gl[:, 1], -rotation_gl[:, 0]))
        else:
            rotation = np.column_stack((-rotation_gl[:, 2], -rotation_gl[:, 0], rotation_gl[:, 1]))
        origin = np.ascontiguousarray(data.cam_xpos[physics.camera], dtype=np.float64)
        spec = detector_module.DetectorSpec(fov_h_deg=62.0, fov_v_deg=48.0)
        # _visible is the pinned upstream geometric frustum/occlusion routine.
        # Its normal constructor requires a site; our official model exposes
        # the equivalent camera pose, so provide those operands directly.
        detector = object.__new__(detector_module.Detector)
        detector.model, detector.spec = model, spec
        detector.mount_body = int(model.cam_bodyid[physics.camera])
        detector._geomgroup = np.ones(6, np.uint8)
        detector._geomgroup[4] = 0
        target = detector_module.Target(self.target_body, "person", target_id, self.target_radius, self.target_height)
        visible = detector._visible(data, target, origin, rotation)
        detections = []
        if visible is not None:
            bearing, elevation, width, distance = visible
            detections.append(detector_module.Detection("person", self.target_body, bearing, elevation, width, distance, 1.0))
        privileged = self.env.evaluator_state()
        yaw = float(privileged["robot_yaw"])
        camera_yaw = math.atan2(rotation[1, 0], rotation[0, 0]) - yaw
        camera_yaw = math.atan2(math.sin(camera_yaw), math.cos(camera_yaw))
        frame = detector_module.DetectionFrame(observation.sim_time_s, detections, cam_z=float(origin[2]), cam_yaw=camera_yaw)
        tof = None
        if observation.tof is not None:
            status = np.asarray(observation.tof_metadata.get("statuses", np.full((8, 8), 5)))
            valid = status == 5
            # Official sensor represents no target as max range; lab expects 0.
            depths = np.where(valid, np.clip(observation.tof, 0, 4) * 1000, 0).astype(np.uint16)
            tof = self.modules["sensors.tof"].TofFrame(
                float(observation.tof_metadata.get("capture_time_s", observation.sim_time_s)),
                depths,
                valid,
            )
        position, velocity = privileged["robot_position"], privileged["robot_velocity"]
        speed = float(velocity[0] * math.cos(yaw) + velocity[1] * math.sin(yaw))
        return self.modules["brain.runtime"].Senses(
            t=observation.sim_time_s,
            tof=tof,
            tof_age=float(observation.tof_metadata.get("frame_age_s", 0)),
            det=frame,
            det_age=0.0,
            speed=speed,
            odom=(float(position[0]), float(position[1]), yaw),
        )

    def predict(self, observation):
        if observation.episode_id != self.episode_id:
            self.reset(observation.episode_id)
        if observation.tick <= self.last_tick:
            raise ValueError("follow-v4 observation ticks must strictly increase")
        if self.last_tick >= 0 and observation.tick - self.last_tick < 5:
            raise ValueError("follow-v4 decides at 10Hz; execute the five-tick SkillIntent")
        senses = self._senses(observation)
        values = self.builder(senses, self.last_action)
        self.last_action = self.network.infer(values)
        self.last_observation, self.last_tick = values.copy(), observation.tick
        command = np.zeros(13, np.float32)
        command[:3] = self.last_action
        return SkillIntent("walk", command, observation.episode_id, observation.tick, duration_ticks=5)
