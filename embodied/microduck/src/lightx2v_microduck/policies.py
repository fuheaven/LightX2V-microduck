"""Explicit feed-forward and recurrent ONNX locomotion policy adapters."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .contracts import JointResidualChunk, Observation, PolicyManifest

POLICY_FILES = {
    "walk": "alpha_walking.onnx",
    "stand": "alpha_stand.onnx",
    "sitstand": "alpha_sitstand.onnx",
    "ground_pick": "alpha_ground_pick.onnx",
    "kick_left": "ball_kick_left.onnx",
    "kick_right": "ball_kick_right.onnx",
    "roulade": "roulade.onnx",
    "roller": "roller.onnx",
    "roller_crouch": "roller_crouch.onnx",
}


class OnnxLocomotionPolicy:
    """A raw 61D -> 14D actor; hidden-state wiring must be declared explicitly.

    state_pairs maps each recurrent input name to its next-state output name.
    Unknown additional inputs are rejected instead of silently feeding zeros
    to a policy with a different sensor or morphology contract.
    """

    def __init__(self, path, *, expected_sha256=None, state_pairs=None, input_name="obs", output_name="actions", providers=None, profile="official_bam", embodiment="legs"):
        import onnxruntime as ort

        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        sha = hashlib.sha256(self.path.read_bytes()).hexdigest()
        if expected_sha256 is not None and sha != expected_sha256:
            raise ValueError(f"policy SHA256 mismatch: {self.path}")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(self.path), sess_options=opts, providers=providers or ["CPUExecutionProvider"])
        self.input_name, self.output_name = input_name, output_name
        self.state_pairs = dict(state_pairs or {})
        inputs = {i.name: i for i in self.session.get_inputs()}
        outputs = {o.name: o for o in self.session.get_outputs()}
        if set(inputs) != {input_name, *self.state_pairs} or set(outputs) != {output_name, *self.state_pairs.values()}:
            raise ValueError(f"undeclared ONNX interface: inputs={list(inputs)}, outputs={list(outputs)}; recurrent policies require state_pairs")
        if tuple(inputs[input_name].shape) != (1, 61) or tuple(outputs[output_name].shape) != (1, 14):
            raise ValueError("locomotion actor must expose obs[1,61] -> actions[1,14]")
        if any(i.type != "tensor(float)" for i in list(inputs.values()) + list(outputs.values())):
            raise ValueError("locomotion actor tensors must be float32")
        self._state_shapes = {}
        for source, destination in self.state_pairs.items():
            shape = tuple(inputs[source].shape)
            if any(not isinstance(d, int) or d <= 0 for d in shape) or tuple(outputs[destination].shape) != shape:
                raise ValueError(f"recurrent state shape must be fixed and match: {source} -> {destination}")
            self._state_shapes[source] = shape
        self.manifest = PolicyManifest(self.path.stem, "joint_residual", "official_61d", profile, embodiment=embodiment, recurrent=bool(self.state_pairs), artifacts={str(self.path): sha})
        self.reset("")

    def reset(self, episode_id: str = "") -> None:
        self.episode_id = str(episode_id)
        self._state = {name: np.zeros(shape, dtype=np.float32) for name, shape in self._state_shapes.items()}

    def infer(self, observation: np.ndarray) -> np.ndarray:
        value = np.asarray(observation, dtype=np.float32).reshape(1, -1)
        if value.shape != (1, 61) or not np.isfinite(value).all():
            raise ValueError("policy observation must be finite 61D")
        output_names = [self.output_name, *self.state_pairs.values()]
        values = self.session.run(output_names, {self.input_name: value, **self._state})
        action = np.asarray(values[0], dtype=np.float32)
        if action.shape != (1, 14) or not np.isfinite(action).all():
            raise ValueError("ONNX actor returned invalid 14D action")
        new_state = {}
        for name, result in zip(self.state_pairs, values[1:]):
            array = np.asarray(result, dtype=np.float32)
            if array.shape != self._state_shapes[name] or not np.isfinite(array).all():
                raise ValueError(f"invalid recurrent state: {name}")
            new_state[name] = array.copy()
        self._state = new_state
        return action[0].copy()

    def predict(self, observation: Observation) -> JointResidualChunk:
        if self.episode_id != observation.episode_id:
            self.reset(observation.episode_id)
        return JointResidualChunk(self.infer(observation.state)[None], observation.episode_id, observation.tick, observation.sim_time_s)

    def clone_state(self) -> dict:
        return {"episode_id": self.episode_id, "state": {k: v.copy() for k, v in self._state.items()}}

    def restore_state(self, snapshot: dict) -> None:
        if set(snapshot["state"]) != set(self._state_shapes):
            raise ValueError("recurrent policy snapshot belongs to another interface")
        self.episode_id = str(snapshot["episode_id"])
        self._state = {k: np.asarray(v, dtype=np.float32).copy() for k, v in snapshot["state"].items()}


class PolicyBank:
    def __init__(self, directory, *, hashes=None, profile="official_bam"):
        self.directory = Path(directory)
        self.hashes = dict(hashes or {})
        self.profile = profile
        self._policies: dict[str, OnnxLocomotionPolicy] = {}
        self.episode_id = ""

    def get(self, role: str) -> OnnxLocomotionPolicy:
        if role not in POLICY_FILES:
            raise ValueError(f"unknown locomotion role {role!r}")
        if role not in self._policies:
            filename = POLICY_FILES[role]
            policy = OnnxLocomotionPolicy(self.directory / filename, expected_sha256=self.hashes.get(filename), profile=self.profile, embodiment="rollers" if role.startswith("roller") else "legs")
            policy.reset(self.episode_id)
            self._policies[role] = policy
        return self._policies[role]

    def infer(self, role: str, observation: np.ndarray) -> np.ndarray:
        return self.get(role).infer(observation)

    def reset(self, episode_id="") -> None:
        self.episode_id = str(episode_id)
        for policy in self._policies.values():
            policy.reset(self.episode_id)

    def clone_state(self) -> dict:
        return {"episode_id": self.episode_id, "policies": {k: v.clone_state() for k, v in self._policies.items()}}

    def restore_state(self, snapshot: dict) -> None:
        self.reset(snapshot["episode_id"])
        for name, state in snapshot["policies"].items():
            self.get(name).restore_state(state)
