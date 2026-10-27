import json
import os
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from loguru import logger

from lightx2v.models.input_encoders.hf.wan.t5.model import T5EncoderModel
from lightx2v.models.networks.wan.fastwam_model import FastWAMNativeModel
from lightx2v.models.runners.base_runner import BaseRunner
from lightx2v.models.video_encoders.hf.wan.vae_2_2 import Wan2_2_VAE
from lightx2v.utils.envs import GET_DTYPE
from lightx2v.utils.registry_factory import RUNNER_REGISTER

AGENTVIEW_IMAGE_NAME = "agentview_image.png"
WRIST_IMAGE_NAME = "wrist_image.png"
HEAD_IMAGE_NAME = "head_camera.png"


def resize_rgb(image, width, height):
    pil = Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB")
    return np.asarray(pil.resize((width, height), resample=Image.BILINEAR), dtype=np.uint8)


def resize_rgb_area(image, width, height):
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("RoboDojo FastWAM preprocessing requires OpenCV (`cv2`).") from exc

    array = np.asarray(image, dtype=np.uint8)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"expected HWC RGB image with 3 channels, got shape {array.shape}")
    resized = cv2.resize(array, (width, height), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(resized, dtype=np.uint8)


def _canonical_norm_mode(mode):
    compact = str(mode).strip().lower().replace("-", "").replace("_", "").replace("/", "")
    if compact == "minmax":
        return "min/max"
    if compact == "zscore":
        return "z-score"
    if compact == "q01q99":
        return "q01/q99"
    raise ValueError(f"unsupported normalize_mode: {mode!r}")


class LinearNormalizer:
    """Affine normalizer matching FastWAM's SingleFieldLinearNormalizer (global stats).

    Modes used by the released checkpoints:
      - "min/max"  (LIBERO):   maps [global_min, global_max] -> [-1, 1]
      - "q01/q99":             maps [global_q01, global_q99] -> [-1, 1]
      - "z-score"  (RoboTwin): (x - global_mean) / global_std
    """

    std_reg = 1e-8
    range_tol = 1e-4
    output_min = -1.0
    output_max = 1.0

    def __init__(self, stats, mode="min/max"):
        mode = _canonical_norm_mode(mode)
        g = {key[len("global_") :]: torch.as_tensor(value, dtype=torch.float32) for key, value in stats.items() if key.startswith("global_")}
        if mode == "z-score":
            mean, std = g["mean"], g["std"]
            self.scale = 1.0 / (std + self.std_reg)
            self.offset = -mean / (std + self.std_reg)
        else:
            low, high = (g["min"].clone(), g["max"].clone()) if mode == "min/max" else (g["q01"].clone(), g["q99"].clone())
            input_range = high - low
            ignore = input_range < self.range_tol
            input_range[ignore] = self.output_max - self.output_min
            self.scale = (self.output_max - self.output_min) / input_range
            self.offset = self.output_min - self.scale * low
            self.offset[ignore] = (self.output_max + self.output_min) / 2 - low[ignore]
        self.mode = mode

    def forward(self, x):
        x = torch.as_tensor(x, dtype=torch.float32)
        return torch.clamp(x * self.scale + self.offset, -5.0, 5.0)

    def backward(self, x):
        x = torch.as_tensor(x, dtype=torch.float32)
        return (x - self.offset) / self.scale


class MinMaxNormalizer(LinearNormalizer):
    """Backwards-compatible alias: LIBERO used a global min/max normalizer."""

    def __init__(self, stats):
        super().__init__(stats, mode="min/max")


class FastWAMPolicy:
    def __init__(
        self,
        adapter_model_path=None,
        dataset_stats_path=None,
        device="cuda",
        model_path=None,
        action_chunk_size: Optional[int] = None,
        actions_per_plan=10,
        action_infer_steps=20,
        action_sample_shift=5.0,
        action_dim_hidden=1024,
        action_dim=7,
        robot_state_dim=8,
        seed: Optional[int] = 0,
        binarize_gripper=True,
        default_prompt=None,
        camera_size=224,
        policy_profile="libero",
        normalize_mode="min/max",
        gripper_postprocess=None,
        config=None,
    ):
        self.device = torch.device(device)
        if not default_prompt:
            raise ValueError("FastWAM requires `default_prompt`.")
        if config is None:
            raise ValueError("FastWAM requires `config`.")

        paths = {
            "model_path": model_path,
            "adapter_model_path": adapter_model_path,
            "dataset_stats_path": dataset_stats_path,
        }
        for name, value in paths.items():
            if value is None or str(value).strip() == "":
                raise ValueError(f"FastWAM requires `{name}`.")
            path = Path(str(value)).expanduser()
            if not path.is_absolute():
                raise ValueError(f"FastWAM requires absolute `{name}`, got: {path}")
            if not path.exists():
                raise FileNotFoundError(str(path))
            paths[name] = path.resolve()

        self.model_path = paths["model_path"]
        self.adapter_model_path = paths["adapter_model_path"]
        self.dataset_stats_path = paths["dataset_stats_path"]

        self.camera_size = int(camera_size)
        self.action_chunk_size = int(action_chunk_size) if action_chunk_size and int(action_chunk_size) > 0 else 32
        self.actions_per_plan = int(max(1, min(int(actions_per_plan), self.action_chunk_size)))
        self.action_infer_steps = int(action_infer_steps)
        self.action_sample_shift = float(action_sample_shift)
        self.action_dim_hidden = int(action_dim_hidden)
        self.action_dim = int(action_dim)
        self.robot_state_dim = int(robot_state_dim)
        self.seed = None if seed is None or int(seed) < 0 else int(seed)
        self.binarize_gripper = bool(binarize_gripper)
        self.default_prompt = str(default_prompt)
        self.policy_profile = str(policy_profile).strip().lower()
        self.normalize_mode = _canonical_norm_mode(normalize_mode)
        self.t5_cpu_offload = bool(config.get("t5_cpu_offload", False))
        self.vae_cpu_offload = bool(config.get("vae_cpu_offload", False))
        # LIBERO post-processes the single gripper channel; RoboTwin (dual-arm qpos) does not.
        self.gripper_postprocess = (self.policy_profile == "libero") if gripper_postprocess is None else bool(gripper_postprocess)
        if self.policy_profile == "microduck_navigation":
            observation = config.get("studio_observation")
            expected_state = 48
            if observation is not None:
                if observation != "rgb_history_300_150_50_0+proprio48+tof64+valid64_v1":
                    raise ValueError("unsupported Studio observation contract")
                expected_state = 176
            if (self.action_dim, self.robot_state_dim) != (5, expected_state):
                raise ValueError("MicroDuck dimensions must match its explicit observation contract")
            history = config.get("navigation_history")
            expected_size = 256
            if history is not None:
                if not isinstance(history, dict):
                    raise ValueError("MicroDuck history grid requires an explicit matching input contract")
                if history.get("schema") in ("microduck_rgb_target_keyframe_v1", "microduck_rgb_target_keyframe_v2"):
                    # Optional MicroDuck input adapter owns detector provenance
                    # and causal keyframe semantics. Equal image size alone is
                    # insufficient to admit this separately trained contract.
                    from lightx2v_microduck.navigation_history import history_spec_from_contract

                    expected_size = history_spec_from_contract(history).output_size
                else:
                    offsets = history.get("offsets_ticks", [])
                    side = 2 if len(offsets) == 4 else 3 if len(offsets) == 9 else 0
                    expected = {"schema": "microduck_causal_rgb_grid_v1", "offsets_ticks": offsets,
                                "camera_size": 256, "output_size": 256 * side,
                                "max_sample_age_ticks": history.get("max_sample_age_ticks"),
                                "layout": "row_major_oldest_to_current", "startup_padding": "repeat_episode_first_frame",
                                "selector": "fixed_timestamps_only", "control_hz": 50}
                    if (not side or history != expected or any(type(t) is not int or t < 0 or t % 5 for t in offsets)
                            or sorted(set(offsets), reverse=True) != offsets or offsets[-1] != 0
                            or type(history.get("max_sample_age_ticks")) is not int
                            or not 0 <= history["max_sample_age_ticks"] <= 5):
                        raise ValueError("MicroDuck history grid requires an explicit matching input contract")
                    expected_size = expected["output_size"]
            if self.gripper_postprocess or self.camera_size != expected_size:
                raise ValueError("MicroDuck navigation camera size must match its input contract, without gripper postprocessing")
        self.config = config
        self._check_model_config()
        self.pending_actions = deque()
        self._prompt_cache = None

        self.state_normalizer, self.action_normalizer = self._load_normalizers()
        self.text_encoder = self._load_text_encoder()
        self.vae = self._load_vae()
        self.model = FastWAMNativeModel(
            model_path=str(self.model_path),
            config=self.config,
            device=self.device,
        )

    @classmethod
    def from_config(cls, config):
        action_chunk_size = int(config.get("action_chunk_size", 0) or 0)
        return cls(
            adapter_model_path=config.get("adapter_model_path"),
            dataset_stats_path=config.get("dataset_stats_path"),
            device=config.get("device", "cuda"),
            model_path=config.get("model_path"),
            action_chunk_size=None if action_chunk_size <= 0 else action_chunk_size,
            actions_per_plan=config.get("actions_per_plan", 10),
            action_infer_steps=config.get("action_infer_steps", 20),
            action_sample_shift=config.get("action_sample_shift", 5.0),
            action_dim_hidden=config.get("action_dim_hidden", 1024),
            action_dim=config.get("action_dim", 7),
            robot_state_dim=config.get("robot_state_dim", 8),
            seed=config.get("seed", 0),
            binarize_gripper=config.get("binarize_gripper", True),
            default_prompt=config.get("default_prompt"),
            camera_size=config.get("camera_size", 224),
            policy_profile=config.get("policy_profile", "libero"),
            normalize_mode=config.get("normalize_mode", "min/max"),
            gripper_postprocess=config.get("gripper_postprocess"),
            config=config,
        )

    def _check_model_config(self):
        for key in ["dim", "num_layers", "num_heads", "freq_dim", "eps", "action_dim_hidden", "action_dim", "robot_state_dim"]:
            if key not in self.config:
                raise ValueError(f"FastWAM requires `{key}` in config.")

    def _load_normalizers(self):
        with open(self.dataset_stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)
        return (
            LinearNormalizer(stats["state"]["default"], self.normalize_mode),
            LinearNormalizer(stats["action"]["default"], self.normalize_mode),
        )

    def _load_text_encoder(self):
        t5_path = self._find_model_file("models_t5_umt5-xxl-enc-bf16.pth")
        tokenizer_path = self._find_model_dir("google/umt5-xxl")
        t5_device = torch.device("cpu") if self.t5_cpu_offload else self.device
        return T5EncoderModel(
            text_len=128,
            dtype=GET_DTYPE(),
            device=t5_device,
            checkpoint_path=t5_path,
            tokenizer_path=str(tokenizer_path),
            cpu_offload=self.t5_cpu_offload,
        )

    def _load_vae(self):
        vae_path = self._find_model_file("Wan2.2_VAE.pth")
        vae_device = torch.device("cpu") if self.vae_cpu_offload else self.device
        return Wan2_2_VAE(
            vae_path=vae_path,
            device=vae_device,
            dtype=GET_DTYPE(),
            vae_type="wan2.2",
            cpu_offload=self.vae_cpu_offload,
            fp32_std_reciprocal=bool(self.config.get("vae_std_reciprocal_fp32", False)),
            rms_norm_mode=self.config.get("vae_rms_norm_mode", "wan"),
        )

    def _find_model_file(self, filename):
        path = self.model_path / filename
        if path.exists():
            return str(path)
        raise FileNotFoundError(f"Cannot find {filename} under {self.model_path}")

    def _find_model_dir(self, dirname):
        path = self.model_path / dirname
        if path.exists():
            return path
        raise FileNotFoundError(f"Cannot find {dirname} under {self.model_path}")

    def reset(self):
        self.pending_actions.clear()
        self._prompt_cache = None

    def next_action(self, images, state, task_description):
        if not self.pending_actions:
            action_chunk = self.predict_action_chunk(images, state, task_description)
            for action in action_chunk[: self.actions_per_plan]:
                self.pending_actions.append(np.asarray(action, dtype=np.float32))
        if not self.pending_actions:
            raise RuntimeError("FastWAM produced an empty action chunk")
        return self.pending_actions.popleft()

    @torch.inference_mode()
    def predict_action_chunk(self, images, state, task_description, seed=None):
        image = self.build_image_tensor(images)
        first_frame_latents = self.encode_image_latents(image)
        context, context_mask = self.encode_prompt(self.default_prompt.format(task_prompt=task_description))
        robot_state = self.state_normalizer.forward(np.asarray(state, dtype=np.float32))
        inputs, action_shape = self.model.prepare_action_inputs(
            first_frame_latents=first_frame_latents,
            context=context,
            context_mask=context_mask,
            action_chunk_size=self.action_chunk_size,
            robot_state=robot_state,
        )
        action = self._run_action_denoising(
            inputs=inputs,
            action_shape=action_shape,
            action_infer_steps=self.action_infer_steps,
            seed=self.seed if seed is None else seed,
        )
        action = self.action_normalizer.backward(action).numpy()
        if self.gripper_postprocess:
            # LIBERO single-gripper channel: map [0,1] -> [-1,1], flip sign, optional binarize.
            action[..., -1] = action[..., -1] * 2 - 1
            action[..., -1] = action[..., -1] * -1.0
            if self.binarize_gripper:
                action[..., -1] = np.sign(action[..., -1])
        return action.astype(np.float32)

    def _run_action_denoising(self, inputs, action_shape, action_infer_steps, seed):
        scheduler = self.model.scheduler
        scheduler.prepare_loop(
            action_shape,
            seed=seed,
            device=self.device,
            dtype=GET_DTYPE(),
            infer_steps=action_infer_steps,
        )
        if self.config.get("fastwam_action_cuda_graph", False):
            from lightx2v.models.networks.wan.infer.fastwam.action_graph import FastWAMActionGraph

            if getattr(self, "_action_graph", None) is None:
                self._action_graph = FastWAMActionGraph(self.model, inputs, scheduler.latents, scheduler.timesteps, scheduler.deltas)
            scheduler.latents = self._action_graph.replay(inputs, scheduler.latents, scheduler.timesteps, scheduler.deltas)
        else:
            for step_index in range(scheduler.infer_steps):
                scheduler.step_pre(step_index)
                self.model.infer(inputs)
                scheduler.step_post()
        action = scheduler.latents[0].detach().to(device="cpu", dtype=torch.float32)
        scheduler.clear()
        return action

    def build_image_tensor(self, images):
        """Compose the policy image from a dict of {camera_name: HxWx3 RGB}.

        Layouts mirror the released FastWAM checkpoints:
          - LIBERO   : [agentview | wrist] side-by-side, each camera_size x camera_size.
          - RoboTwin : head (320x256) on top, [left | right] (each 160x128) below,
                       i.e. final [384, 320, 3]; matches deploy_policy.py.
          - RoboDojo : first normalizes each source RGB to 320x240 with OpenCV
                       INTER_AREA, then applies the official RoboDojo FastWAM
                       384x320 three-camera layout.
          - MicroDuck navigation: a single 256x256 head camera, no wrist view.
        """
        rgb = self._compose_rgb(images)
        if self.policy_profile == "microduck_navigation":
            # Match the training/reference image path: normalize in FP32,
            # then cast. Doing arithmetic on uint8 values already cast to
            # BF16 changes camera pixels before they reach the shared VAE.
            tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().div(127.5).sub(1)
            return tensor.to(device=self.device, dtype=GET_DTYPE())
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).to(device=self.device, dtype=GET_DTYPE())
        return tensor * (2.0 / 255.0) - 1.0

    def _compose_rgb(self, images):
        def _get(name):
            if name not in images or images[name] is None:
                raise KeyError(f"FastWAM profile '{self.policy_profile}' requires camera '{name}'")
            return images[name]

        if self.policy_profile == "microduck_navigation":
            return resize_rgb(_get("head_camera"), self.camera_size, self.camera_size)

        if self.policy_profile == "robotwin":
            head = resize_rgb(_get("head_camera"), 320, 256)
            left = resize_rgb(_get("left_camera"), 160, 128)
            right = resize_rgb(_get("right_camera"), 160, 128)
            bottom = np.concatenate([left, right], axis=1)
            return np.concatenate([head, bottom], axis=0)

        if self.policy_profile == "robodojo":
            head_src = resize_rgb_area(_get("head_camera"), 320, 240)
            left_src = resize_rgb_area(_get("left_camera"), 320, 240)
            right_src = resize_rgb_area(_get("right_camera"), 320, 240)
            head = resize_rgb(head_src, 320, 256)
            left = resize_rgb(left_src, 160, 128)
            right = resize_rgb(right_src, 160, 128)
            bottom = np.concatenate([left, right], axis=1)
            image = np.concatenate([head, bottom], axis=0)
            if image.shape != (384, 320, 3):
                raise RuntimeError(f"RoboDojo FastWAM image must be (384, 320, 3), got {image.shape}")
            return image

        if self.policy_profile != "libero":
            raise ValueError(f"unsupported FastWAM policy_profile: {self.policy_profile!r}")

        primary = resize_rgb(_get("agentview"), self.camera_size, self.camera_size)
        wrist = resize_rgb(_get("wrist"), self.camera_size, self.camera_size)
        return np.concatenate([primary, wrist], axis=1)

    def encode_image_latents(self, image):
        image = image.unsqueeze(1)
        latents = self.vae.encode(image.unsqueeze(0))
        if latents.ndim == 4:
            latents = latents.unsqueeze(0)
        return latents.to(device=self.device, dtype=GET_DTYPE())

    def encode_prompt(self, prompt):
        # Only language is reusable across observations. Camera, proprioception,
        # and the appended state token must still be rebuilt for every plan.
        cache_enabled = bool(self.config.get("fastwam_cache_prompt", False))
        key = (prompt, self.device, GET_DTYPE(), id(self.text_encoder),
               id(self.text_encoder.model), id(self.text_encoder.tokenizer))
        cached = self._prompt_cache
        if cache_enabled and cached is not None and cached[0] == key:
            return cached[1]
        # A single entry bounds memory and disabling the option invalidates it.
        self._prompt_cache = None
        ids, mask = self.text_encoder.tokenizer([prompt], return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        with torch.no_grad():
            context = self.text_encoder.model(ids, mask)
        for i, seq_len in enumerate(seq_lens):
            context[i, seq_len:] = 0
        mask = torch.ones_like(mask, dtype=torch.bool)
        result = context[0].to(device=self.device, dtype=GET_DTYPE()), mask[0].to(device=self.device)
        if cache_enabled:
            self._prompt_cache = (key, result)
        return result

    def close(self):
        self.pending_actions.clear()
        self._prompt_cache = None
        self._action_graph = None


@RUNNER_REGISTER("fastwam")
class FastWAMRunner(BaseRunner):
    def init_modules(self):
        logger.info("Loading FastWAM policy...")
        self.policy = FastWAMPolicy.from_config(self.config)
        self.config.lock()
        logger.info("FastWAM policy loaded.")

    def _resolve_action_output_path(self):
        if getattr(self.input_info, "save_action_path", ""):
            return str(Path(self.input_info.save_action_path).expanduser().resolve())
        if getattr(self.input_info, "save_result_path", ""):
            return str(Path(self.input_info.save_result_path).expanduser().resolve().with_suffix(".actions.npy"))
        raise ValueError("FastWAM requires `save_action_path` or `save_result_path`.")

    def _load_image_pair(self):
        image_path = getattr(self.input_info, "image_path", "") or ""
        if not image_path:
            raise ValueError("FastWAM requires `image_path`.")

        image_path = os.path.expanduser(str(image_path))
        if os.path.isdir(image_path):
            agentview = os.path.join(image_path, AGENTVIEW_IMAGE_NAME)
            wrist = os.path.join(image_path, WRIST_IMAGE_NAME)
        else:
            items = [item.strip() for item in image_path.split(",") if item.strip()]
            if len(items) != 2:
                raise ValueError("FastWAM `image_path` must be a directory or two comma-separated image paths.")
            agentview, wrist = items

        return self._load_rgb(agentview), self._load_rgb(wrist)

    @staticmethod
    def _load_rgb(path):
        return np.asarray(Image.open(os.path.expanduser(str(path))).convert("RGB"))

    def _load_state(self):
        state_path = getattr(self.input_info, "state_path", "") or ""
        if not state_path:
            raise ValueError("FastWAM requires `state_path` matching robot_state_dim.")
        state_path = str(Path(state_path).expanduser().resolve())
        suffix = Path(state_path).suffix.lower()

        if suffix == ".json":
            with open(state_path, "r") as f:
                payload = json.load(f)
        elif suffix in [".npy", ".npz"]:
            payload = np.load(state_path, allow_pickle=True)
            if isinstance(payload, np.lib.npyio.NpzFile):
                payload = {key: payload[key] for key in payload.files}
            elif payload.shape == () and isinstance(payload.item(), dict):
                payload = payload.item()
        else:
            payload = np.loadtxt(state_path, delimiter=",", dtype=np.float32)

        if isinstance(payload, dict):
            for key in ["state", "qpos", "robot_state", "observation.state"]:
                if key in payload:
                    payload = payload[key]
                    break
        state = np.asarray(payload, dtype=np.float32).reshape(-1)
        expected = self.policy.robot_state_dim
        if state.size != expected or not np.isfinite(state).all():
            raise ValueError(f"FastWAM state must contain {expected} finite floats, got {state.size}.")
        return state

    def _save_actions(self, actions):
        action_path = self._resolve_action_output_path()
        os.makedirs(os.path.dirname(action_path) or ".", exist_ok=True)
        if action_path.endswith(".json"):
            with open(action_path, "w") as f:
                json.dump(actions.tolist(), f, ensure_ascii=False, indent=2)
        else:
            np.save(action_path, actions)
        logger.info("Saved FastWAM actions to {}", action_path)

    def run_pipeline(self, input_info):
        self.input_info = input_info
        if not self.input_info.prompt:
            raise ValueError("FastWAM requires `prompt` as the task description.")

        if self.policy.policy_profile == "microduck_navigation":
            path = Path(str(getattr(self.input_info, "image_path", ""))).expanduser()
            if path.is_dir():
                path = path / HEAD_IMAGE_NAME
            images = {"head_camera": self._load_rgb(path)}
        else:
            agentview, wrist = self._load_image_pair()
            images = {"agentview": agentview, "wrist": wrist}
        state = self._load_state()
        actions = self.policy.predict_action_chunk(
            images=images,
            state=state,
            task_description=self.input_info.prompt,
            seed=self.input_info.seed,
        )
        self._save_actions(actions)

        if self.input_info.return_result_tensor:
            return {"actions": actions}
        return {"actions": None}

    def end_run(self):
        if hasattr(self, "policy"):
            self.policy.close()
