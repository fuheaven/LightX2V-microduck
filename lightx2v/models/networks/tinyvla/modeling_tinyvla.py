# Copyright AlexWortega and LightX2V contributors. SPDX-License-Identifier: Apache-2.0
# Adapted from AlexWortega/tinyvla @ 603f5045b45126af011418befc899e3c67d04346.
"""TinyVLA policy: dual-rate VLA on a single shared Qwen3.5 vision encoder.

Slow path: image -> Qwen3.5 vision tower -> language stack (first 12 layers) +
instruction -> cached latent (16 action readouts in public duckft unified mode;
8 readouts in the legacy semantic mode), refreshed at 0.5-2 Hz.

Fast path: Qwen3.5 vision tower spatial tokens (64/camera, no language stack)
+ proprio + embodiment token + cached semantic latent -> ~40M flow-matching
action expert -> 50-step action chunk.

During training both paths run jointly (single-rate); staleness augmentation
is added by the Stage-2 training loop.
"""

from __future__ import annotations

from collections import deque
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .configuration_tinyvla import FeatureType, TinyVLAConfig
from .modules.embodiment import EmbodimentEmbedding, pad_vector
from .modules.expert import FlowMatchingExpert
from .modules.semantic import SemanticPath

ACTION = "action"
OBS_LANGUAGE_ATTENTION_MASK = "observation.language.attention_mask"
OBS_LANGUAGE_TOKENS = "observation.language.tokens"
OBS_STATE = "observation.state"


class TinyVLAModel(nn.Module):
    config_class = TinyVLAConfig
    name = "tinyvla"

    def __init__(self, config: TinyVLAConfig, *, semantic_path=None):
        super().__init__()
        self.config = config

        self.image_keys = sorted(k for k, v in config.input_features.items() if v.type is FeatureType.VISUAL)
        state_ft = config.input_features.get(OBS_STATE)
        self.state_dim = state_ft.shape[0] if state_ft is not None else 0
        self.action_dim = config.output_features[ACTION].shape[0]

        d = config.expert_dim

        self.semantic = (
            semantic_path
            if semantic_path is not None
            else SemanticPath(
                model_name=config.lm_model_name,
                num_layers=config.lm_num_layers,
                num_readout=config.num_readout_tokens,
                out_dim=d,
                image_size=config.image_size,
                freeze_lm=config.freeze_lm,
                lm_unfreeze_last_n=config.lm_unfreeze_last_n,
                freeze_vision=config.freeze_vision_encoder,
                morph_to_slow=config.morph_to_slow,
                morph_dim=16,
                morph_qwen=(config.conditioning == "morph_qwen"),
                num_morph_readout=config.num_morph_readout,
                vlm_native=config.vlm_native,
                num_action_readout=config.num_action_readout,
                max_state_dim=config.max_state_dim,
                max_action_dim=config.max_action_dim,
                revision=config.backbone_revision,
                vision_backend=config.vision_backend,
                semantic_padding_policy=config.semantic_padding_policy,
            )
        )

        # fast-path conditioning projections
        self.spatial_proj = nn.Linear(self.semantic.visual_hidden_size, d)
        self.camera_emb = nn.Embedding(config.max_cameras, d)
        self.state_proj = nn.Linear(config.max_state_dim, d)
        self.embodiment_emb = EmbodimentEmbedding(config.num_embodiments, d)
        if config.conditioning == "morph":
            from .modules.embodiment import MorphologyEncoder

            self.morph_encoder = MorphologyEncoder(d, n_tokens=config.morph_tokens)
        if config.use_demo_conditioning:
            from .modules.embodiment import DemoConditioner

            self.demo_encoder = DemoConditioner(
                d,
                visual_dim=self.semantic.visual_hidden_size,
                action_dim=config.max_action_dim,
                n_tokens_per_example=config.demo_tokens_per_example,
                n_visual_groups=config.demo_visual_groups,
                n_action_keys=config.demo_action_keys,
                hidden_mult=config.demo_hidden_mult,
            )

        self.spatial_head = nn.Linear(d, config.spatial_vocab) if config.spatial_loss_weight > 0 else None

        self.expert = FlowMatchingExpert(
            action_dim=config.max_action_dim,
            chunk_size=config.chunk_size,
            d_model=d,
            n_layers=config.expert_layers,
            n_heads=config.expert_heads,
            min_period=config.min_period,
            max_period=config.max_period,
        )

        self._action_queue: deque = deque()

    def reset(self):
        self._action_queue.clear()

    def get_optim_params(self):
        return [p for p in self.parameters() if p.requires_grad]

    # ---------------------------------------------------------------- features

    @staticmethod
    def _last_obs(x: torch.Tensor, ndim: int) -> torch.Tensor:
        """Drop the n_obs_steps dim if present: (B, S, ...) -> (B, ...)."""
        return x[:, -1] if x.dim() == ndim + 1 else x

    def _encode_spatial(self, batch: dict, feats: dict | None = None) -> torch.Tensor:
        """Per-camera spatial tokens for the fast path: (B, num_cams*64, d).

        `feats` lets the caller pass vision-tower outputs computed once per step
        (see `_vision_features`) instead of re-running the tower here.
        """
        cond = []
        for i, key in enumerate(self.image_keys):
            tokens = (feats or {}).get(key)
            if tokens is None:
                tokens = self.semantic.encode_image(self._last_obs(batch[key], 4))  # (B, 64, C)
            proj = self.spatial_proj(tokens) + self.camera_emb.weight[i][None, None, :]
            if i == 0:
                self._cam0_proj = proj  # consumed by the spatial-distillation head
            cond.append(proj)
        return torch.cat(cond, dim=1)

    def _vision_features(self, batch: dict, keys) -> dict:
        """One vision-tower pass per camera, shared by the fast and slow paths."""
        return {k: self.semantic.encode_image(self._last_obs(batch[k], 4)) for k in keys}

    def _semantic_latent(self, batch: dict, image=None, image_embeds=None) -> torch.Tensor:
        """Slow-path latent: (B, K, d), cacheable, conditions the fast path.

        vlm_native=True makes this latent RICH — the LM sequence also carries the
        robot description and the demo (image, action) exemplars, so the readout
        tokens can attend over them. The fast path is unchanged: it still gets its
        own spatial tokens + proprio + expert, just a better-informed latent.
        """
        img = image if image is not None else self._last_obs(batch[self.image_keys[0]], 4)
        if self.config.vlm_native:
            state = None
            if self.state_dim:
                state = pad_vector(self._last_obs(batch[OBS_STATE], 2), self.config.max_state_dim)
            return self.semantic.forward_unified(
                image=img,
                lang_tokens=batch[OBS_LANGUAGE_TOKENS],
                lang_mask=batch[OBS_LANGUAGE_ATTENTION_MASK],
                state=state,
                # demo_only: no robot description — demos are the sole identity cue
                robot_ids=None if self.config.demo_only else batch.get("morph_text_ids"),
                robot_mask=None if self.config.demo_only else batch.get("morph_text_mask"),
                demo_images=batch.get("support_images"),
                demo_actions=batch.get("support_actions"),
                n_demo_action_keys=self.config.demo_action_keys_lm,
            )
        morph = batch.get("morphology") if self.config.morph_to_slow else None
        return self.semantic(
            img,
            batch[OBS_LANGUAGE_TOKENS],
            batch[OBS_LANGUAGE_ATTENTION_MASK],
            morphology=morph,
            image_embeds=image_embeds,
        )

    def _conditioning(self, batch: dict) -> torch.Tensor:
        b = batch[self.image_keys[0]].shape[0]
        device = batch[self.image_keys[0]].device

        # Run the shared Qwen vision tower ONCE per camera per step. Previously
        # cam0 went through it twice: here for the spatial tokens, and again
        # inside the LM call in `_semantic_latent`. The tower is ~45 GFLOPs per
        # camera, i.e. a quarter of the forward pass thrown away every step.
        feats = None
        if self.config.use_spatial_tokens:
            feats = self._vision_features(batch, self.image_keys)
        elif self.config.use_semantic_latent and not self.config.vlm_native:
            feats = self._vision_features(batch, self.image_keys[:1])

        parts = []
        if self.config.use_semantic_latent:
            latent = batch.get("semantic_latent")
            if latent is None:
                if "latent_image" in batch:  # staleness aug: latent from an older frame,
                    latent = self._semantic_latent(batch, image=batch["latent_image"])
                else:  # ...so the cached cam0 features do NOT apply
                    cam0 = None if (feats is None or self.config.vlm_native) else feats[self.image_keys[0]]
                    latent = self._semantic_latent(batch, image_embeds=cam0)
            parts.append(latent)
        if self.config.use_spatial_tokens:
            parts.append(self._encode_spatial(batch, feats))
        if self.state_dim:
            state = pad_vector(self._last_obs(batch[OBS_STATE], 2), self.config.max_state_dim)
            parts.append(self.state_proj(state)[:, None, :])
        if self.config.demo_only:
            pass  # no identity token at all — demos in the LM latent are the only cue
        elif self.config.conditioning == "morph":
            desc = batch.get("morphology")
            if desc is None:
                from .modules.embodiment import MORPH_DIM

                desc = torch.zeros(b, MORPH_DIM, device=device)
            parts.append(self.morph_encoder(desc.float()))
        elif self.config.conditioning == "morph_qwen":
            ids = batch.get("morph_text_ids")
            mask = batch.get("morph_text_mask")
            if ids is None:
                ids = torch.zeros(b, 1, dtype=torch.long, device=device)
                mask = torch.ones(b, 1, dtype=torch.bool, device=device)
            parts.append(self.semantic.encode_robot_text(ids, mask))
        else:
            emb_id = batch.get("embodiment_id")
            if emb_id is None:
                emb_id = torch.zeros(b, dtype=torch.long, device=device)
            parts.append(self.embodiment_emb(emb_id))

        if self.config.use_demo_conditioning:
            sup_imgs = batch.get("support_images")  # (B, K, 3, H, W)
            sup_acts = batch.get("support_actions")  # (B, K, T, A)
            if sup_imgs is not None and sup_acts is not None:
                bb, k = sup_imgs.shape[:2]
                flat_imgs = sup_imgs.reshape(bb * k, *sup_imgs.shape[2:])
                spatial = self.semantic.encode_image(flat_imgs)  # (B*K, 64, visual_dim)
                # keep the spatial structure — the encoder does grouped pooling
                spatial = spatial.view(bb, k, spatial.shape[1], spatial.shape[2])
                parts.append(self.demo_encoder(spatial, sup_acts))
        return torch.cat(parts, dim=1)

    # ---------------------------------------------------------------- training

    def forward(self, batch: dict, *, noise=None, time=None, flow_loss="mse", flow_huber_delta=1.0) -> tuple[torch.Tensor, dict]:
        # Experimental training objective only; sampling and the default MSE
        # reference path remain unchanged. Delta is in normalized flow units.
        if flow_loss not in ("mse", "huber"):
            raise ValueError("flow_loss must be mse or huber")
        if isinstance(flow_huber_delta, bool) or not math.isfinite(flow_huber_delta) or flow_huber_delta <= 0:
            raise ValueError("flow_huber_delta must be positive and finite")
        cond = self._conditioning(batch)

        actions = pad_vector(batch[ACTION], self.config.max_action_dim)  # (B, T, A)
        b, t, a = actions.shape
        device = actions.device

        beta = torch.distributions.Beta(self.config.flow_beta_alpha, self.config.flow_beta_beta)
        time = beta.sample((b,)).to(device) * 0.999 + 0.001 if time is None else time.to(device)
        noise = torch.randn_like(actions) if noise is None else noise.to(device)
        if noise.shape != actions.shape or time.shape != (b,):
            raise ValueError("training noise/time shape mismatch")
        x_t = time[:, None, None] * noise + (1 - time[:, None, None]) * actions
        target = noise - actions

        pred = self.expert(x_t, time, cond)

        loss_mask = torch.ones(b, t, a, device=device)
        if "action_dim_mask" in batch:  # per-sample real dims (mixture training)
            loss_mask = loss_mask * batch["action_dim_mask"][:, None, :].float()
        else:
            loss_mask[:, :, self.action_dim :] = 0  # padded action dims
        if "action_is_pad" in batch:  # episode-end padding from delta_timestamps
            loss_mask = loss_mask * (~batch["action_is_pad"])[:, :, None].float()
        if "action_time_weights" in batch:
            weights = batch["action_time_weights"]
            if weights.shape != (t,) or not torch.isfinite(weights).all() or (weights <= 0).any():
                raise ValueError("action_time_weights must be positive finite horizon weights")
            loss_mask = loss_mask * weights.to(device=device, dtype=loss_mask.dtype)[None, :, None]

        if not loss_mask.any():
            raise ValueError("batch contains no valid action targets")
        if flow_loss == "huber":
            # Twice Huber equals squared error near zero, then bounds each
            # residual's derivative at 2*delta without discarding its target.
            element_loss = 2 * F.huber_loss(pred.float(), target.float(), reduction="none", delta=flow_huber_delta)
        else:
            element_loss = F.mse_loss(pred, target, reduction="none")
        loss = (element_loss * loss_mask).sum() / loss_mask.sum()
        info = {"flow_loss": loss.item()}

        if self.spatial_head is not None and "spatial_cell" in batch and self.config.use_spatial_tokens:
            cells = batch["spatial_cell"]
            labeled = cells >= 0
            if labeled.any():
                logits = self.spatial_head(self._cam0_proj.mean(dim=1))  # (B, vocab)
                ce = F.cross_entropy(logits[labeled], cells[labeled])
                loss = loss + self.config.spatial_loss_weight * ce
                info["spatial_ce"] = ce.item()

        info["loss"] = loss.item()
        return loss, info

    # --------------------------------------------------------------- inference

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict, *, noise=None, generator=None, return_trace=False, active_action_dim: int | None = None) -> torch.Tensor:
        """Sample an action chunk; optionally constrain inactive action channels.

        The default preserves the author's sampler. With an explicit active
        dimension, the remaining training targets are assumed to be zero. Their
        flow path is therefore x_pad(t) = t * initial_noise, instead of integrating
        output channels excluded from the training loss. This is an experiment
        switch, not an assertion that closed-loop behavior improves. Return
        width remains the configured output width.
        """
        if active_action_dim is not None and (
            isinstance(active_action_dim, bool)
            or not isinstance(active_action_dim, int)
            or not 1 <= active_action_dim <= self.config.max_action_dim
        ):
            raise ValueError("active_action_dim must be an integer within 1..max_action_dim")
        cond = self._conditioning(batch)
        b = cond.shape[0]
        device = cond.device

        shape = (b, self.config.chunk_size, self.config.max_action_dim)
        if noise is None:
            x = torch.randn(shape, device=device, generator=generator)
        else:
            if tuple(noise.shape) != shape or not torch.isfinite(noise).all():
                raise ValueError(f"noise must be finite with shape {shape}")
            x = noise.to(device=device, dtype=torch.float32).clone()
        constrain_padding = active_action_dim is not None and active_action_dim < self.config.max_action_dim
        padded_noise = x[:, :, active_action_dim:].clone() if constrain_padding else None
        trace = [x.clone()] if return_trace else None
        n = self.config.num_denoise_steps
        dt = 1.0 / n
        for i in range(n):
            t = torch.full((b,), 1.0 - i * dt, device=device)
            v = self.expert(x, t, cond)
            x = x - dt * v
            if constrain_padding:
                # Match the forward process at the NEXT time, including exact
                # zero padding at t=0. Never mutate caller-supplied noise.
                next_time = 1.0 - (i + 1) / n
                x[:, :, active_action_dim:] = padded_noise * next_time
            if trace is not None:
                trace.append(x.clone())
        result = x[:, :, : self.action_dim]
        return (result, trace) if return_trace else result

    @torch.no_grad()
    def select_action(self, batch: dict) -> torch.Tensor:
        if len(self._action_queue) == 0:
            chunk = self.predict_action_chunk(batch)  # (B, T, A)
            self._action_queue.extend(chunk.transpose(0, 1)[: self.config.n_action_steps])
        return self._action_queue.popleft()

    @classmethod
    def from_pretrained(cls, directory, *, backbone_path=None, backbone_revision=None):
        from pathlib import Path

        from safetensors.torch import load_file

        config = TinyVLAConfig.from_pretrained(directory)
        if config.chunk_size != 50:
            raise ValueError("Only the public 50-step TinyVLA architecture is supported; Stage 2.5 requires unavailable versioned model artifacts")
        if backbone_path is not None:
            config.lm_model_name = str(backbone_path)
        if backbone_revision is not None:
            config.backbone_revision = backbone_revision
        model = cls(config)
        weights = load_file(str(Path(directory) / "model.safetensors"), device="cpu")
        model.load_state_dict(weights, strict=True, assign=True)
        return model

    def save_pretrained(self, directory):
        import json
        from pathlib import Path

        from safetensors.torch import save_file

        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "config.json").write_text(json.dumps(self.config.to_dict(), indent=2))
        tensors = {key: value.detach().cpu().contiguous() for key, value in self.state_dict().items()}
        save_file(tensors, str(destination / "model.safetensors"))


# Explicit compatibility alias for data tooling; no LeRobot runtime dependency.
TinyVLAPolicy = TinyVLAModel
