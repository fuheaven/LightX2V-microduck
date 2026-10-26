# Copyright AlexWortega and LightX2V contributors. SPDX-License-Identifier: Apache-2.0
# Adapted from AlexWortega/tinyvla @ 603f5045b45126af011418befc899e3c67d04346.
"""Slow semantic path: Qwen3.5-0.8B VLM over [image | instruction | readout slots].

Qwen3.5-0.8B has a native vision tower (12-layer ViT, patch 16, spatial merge 2:
a 256x256 image becomes 64 visual tokens), so the slow path consumes the raw
image directly — no hand-rolled projector. The language stack is truncated to
the first `num_layers` blocks (L/2 layer-skip, SmolVLA-style). The semantic
latent is the hidden state of K trailing readout slots (pad tokens after the
instruction; causal attention lets them aggregate the full prefix), projected
to the expert dimension. At deployment this runs at 0.5-2 Hz and the latent is
cached between refreshes.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticPath(nn.Module):
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3.5-0.8B",
        num_layers: int = 12,
        num_readout: int = 8,
        out_dim: int = 512,
        image_size: int = 256,
        freeze_lm: bool = True,
        lm_unfreeze_last_n: int = 0,
        freeze_vision: bool = True,
        morph_to_slow: bool = False,
        morph_dim: int = 16,
        morph_qwen: bool = False,
        num_morph_readout: int = 4,
        vlm_native: bool = False,
        num_action_readout: int = 16,
        max_state_dim: int = 16,
        max_action_dim: int = 8,
        revision: str | None = None,
        vision_backend: str = "fixed_eager",
        semantic_padding_policy: str = "legacy",
    ):
        super().__init__()
        import json

        from transformers import AutoConfig, AutoModelForImageTextToText, AutoTokenizer
        from transformers.utils.hub import cached_file

        # The complete VLA checkpoint owns the weights. Fetch only backbone
        # configuration/processor, never a second pretrained weight checkpoint.
        config = AutoConfig.from_pretrained(model_name, revision=revision)
        config.text_config.num_hidden_layers = num_layers
        if hasattr(config.text_config, "layer_types"):
            config.text_config.layer_types = config.text_config.layer_types[:num_layers]
        vlm = AutoModelForImageTextToText.from_config(config, dtype=torch.float32)
        self.vision_backend = vision_backend
        model = vlm.model  # Qwen3_5Model: visual tower + language stack
        # L/2 layer-skip: keep the first num_layers language blocks; lm_head unused
        model.language_model.layers = model.language_model.layers[:num_layers]
        model.language_model.config.num_hidden_layers = num_layers
        self.vlm = model
        from .semantic_padding import install_semantic_padding

        install_semantic_padding(self.vlm.language_model, semantic_padding_policy)
        self.semantic_padding_policy = semantic_padding_policy
        cfg = vlm.config
        self.hidden_size = cfg.text_config.hidden_size
        self.image_token_id = cfg.image_token_id
        self.vision_start_id = cfg.vision_start_token_id
        self.vision_end_id = cfg.vision_end_token_id

        tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        # Only the processor constants are needed: our patchifier implements
        # the transform, so importing a torchvision FastImageProcessor adds
        # an unnecessary binary dependency.
        processor_file = cached_file(model_name, "preprocessor_config.json", revision=revision)
        with open(processor_file) as stream:
            ip = json.load(stream)
        self.patch_size = ip["patch_size"]
        self.merge_size = ip["merge_size"]
        self.temporal_patch_size = ip["temporal_patch_size"]
        self.register_buffer("pixel_mean", torch.tensor(ip["image_mean"]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("pixel_std", torch.tensor(ip["image_std"]).view(1, 3, 1, 1), persistent=False)
        self.image_size = image_size
        grid = image_size // self.patch_size
        self.num_image_tokens = (grid // self.merge_size) ** 2  # 64 @ 256px

        self.out_proj = nn.Linear(self.hidden_size, out_dim)
        self.num_readout = num_readout
        self._fixed_fn = None  # lazily compiled fixed-grid tower path
        self.morph_to_slow = morph_to_slow
        if morph_to_slow:
            # inject robot morphology into the slow LM prefix so the semantic
            # latent is morphology-aware ("grasp X" differs per embodiment)
            self.morph_to_lm = nn.Linear(morph_dim, self.hidden_size)

        # LLaVA-style unified mode: every modality becomes tokens inside the LM
        # sequence, and the action policy reads latent tokens off the LM output.
        self.vlm_native = vlm_native
        self.num_action_readout = num_action_readout
        if vlm_native:
            self.state_to_lm = nn.Linear(max_state_dim, self.hidden_size)
            self.action_to_lm = nn.Linear(max_action_dim, self.hidden_size)
            self.action_readout_proj = nn.Linear(self.hidden_size, out_dim)
            self.readout_emb = nn.Parameter(torch.zeros(num_action_readout, self.hidden_size))
            nn.init.normal_(self.readout_emb, std=0.02)

        self.morph_qwen = morph_qwen
        self.num_morph_readout = num_morph_readout
        if morph_qwen:
            # encode the robot TEXT description through the SAME Qwen language
            # stack (shared weights, text-only, no image) instead of an MLP over
            # numeric fields — lets the (unfrozen) LM's language understanding
            # do the morphology encoding instead of a from-scratch projector.
            self.morph_out_proj = nn.Linear(self.hidden_size, out_dim)

        if not 0 <= lm_unfreeze_last_n <= num_layers:
            raise ValueError("lm_unfreeze_last_n exceeds the available LM layers")
        if freeze_lm or lm_unfreeze_last_n:
            for p in self.vlm.language_model.parameters():
                p.requires_grad = False
        if lm_unfreeze_last_n:
            for block in self.vlm.language_model.layers[-lm_unfreeze_last_n:]:
                for parameter in block.parameters():
                    parameter.requires_grad = True
        # vocab embeddings stay frozen either way (254M tied params, negligible gain)
        self.vlm.get_input_embeddings().weight.requires_grad = False
        if freeze_vision:
            for p in self.vlm.visual.parameters():
                p.requires_grad = False

    @property
    def visual_hidden_size(self) -> int:
        return self.vlm.visual.config.out_hidden_size

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Qwen vision tower spatial tokens: (B, 3, H, W) -> (B, 64, out_hidden).

        Shared encoder for both paths; the fast path cross-attends to these
        directly, bypassing the language stack. The SAME tensor can be handed to
        `forward(..., image_embeds=...)` so cam0 goes through the tower once per
        step instead of twice (once here, once inside the LM call).

        Uses a dense fixed-grid fast path: every image here is the same
        (1, g, g) grid, so the HF packed-varlen tower (which runs B separate
        SDPA calls PER LAYER and rebuilds pos/rotary tables in Python on every
        call) is replaced by one batched SDPA per layer over (B, L, D), with the
        per-image positional constants computed once and cached. Numerically
        equivalent — per-image attention with equal lengths IS batched attention.
        """
        # Fixed-grid positional caching is an inference optimization. The
        # position embedding is trainable when vision is unfrozen; detaching
        # and caching it would silently discard its gradient during fine-tuning.
        if self.training and any(parameter.requires_grad for parameter in self.vlm.visual.parameters()):
            self._fixed_ctx = None
            return self._encode_image_hf(images)
        if self.vision_backend == "hf":
            return self._encode_image_hf(images)
        pixel_values, _ = self._patchify(images)
        if self.vision_backend == "fixed_compile":
            if self._fixed_fn is None:
                self._fixed_fn = torch.compile(self._encode_image_fixed, dynamic=False)
            return self._fixed_fn(images.shape[0], pixel_values)
        return self._encode_image_fixed(images.shape[0], pixel_values)

    def _encode_image_hf(self, images: torch.Tensor) -> torch.Tensor:
        """Reference path through the stock HF tower (kept for verification)."""
        b = images.shape[0]
        pixel_values, grid_thw = self._patchify(images)
        out = self.vlm.get_image_features(pixel_values.to(self.pixel_mean.dtype), grid_thw)
        embeds = torch.cat(out.pooler_output, dim=0)  # per-image tuple -> (B*64, C)
        return embeds.reshape(b, self.num_image_tokens, -1)

    def _fixed_grid_ctx(self, device):
        """Per-image positional constants for the fixed (1, g, g) grid, computed
        ONCE via the tower's own functions (so the layout can't drift) and cached."""
        g = self.image_size // self.patch_size
        ctx = getattr(self, "_fixed_ctx", None)
        if ctx is not None and ctx[0] == (g, str(device)):
            return ctx[1]
        visual = self.vlm.visual
        grid1 = torch.tensor([[1, g, g]], device=device)
        with torch.no_grad():
            pos = visual.fast_pos_embed_interpolate(grid1)  # (g*g, hidden)
            rot = visual.rot_pos_emb(grid1)  # (g*g, head_dim/2)
        emb = torch.cat((rot, rot), dim=-1)
        val = (pos.detach(), emb.cos().detach(), emb.sin().detach())
        self._fixed_ctx = ((g, str(device)), val)
        return val

    def _encode_image_fixed(self, b: int, pixel_values: torch.Tensor) -> torch.Tensor:
        from transformers.models.qwen3_5.modeling_qwen3_5 import rotate_half

        visual = self.vlm.visual
        g = self.image_size // self.patch_size
        L = g * g
        x = visual.patch_embed(pixel_values.to(self.pixel_mean.dtype))  # (B*L, D)
        pos, cos, sin = self._fixed_grid_ctx(x.device)
        d = x.shape[-1]
        x = x.view(b, L, d) + pos[None].to(x.dtype)
        # (1, 1, L, hd) fp32 — same upcast-to-fp32 rotary as the HF path
        cosf = cos[None, None].float()
        sinf = sin[None, None].float()
        for blk in visual.blocks:
            attn = blk.attn
            h = blk.norm1(x)
            qkv = attn.qkv(h).view(b, L, 3, attn.num_heads, attn.head_dim)
            q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # (B, nh, L, hd)
            qf, kf = q.float(), k.float()
            q = ((qf * cosf) + (rotate_half(qf) * sinf)).to(q.dtype)
            k = ((kf * cosf) + (rotate_half(kf) * sinf)).to(k.dtype)
            o = F.scaled_dot_product_attention(q, k, v, scale=attn.scaling)
            o = o.transpose(1, 2).reshape(b, L, d)
            x = x + attn.proj(o)
            x = x + blk.mlp(blk.norm2(x))
        merged = visual.merger(x.reshape(b * L, d))  # (B*L/m^2, out_hidden)
        return merged.reshape(b, self.num_image_tokens, -1)

    def _grid_thw(self, b: int, device) -> torch.Tensor:
        g = self.image_size // self.patch_size
        return torch.tensor([[1, g, g]], device=device).expand(b, -1)

    def _patchify(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, 3, H, W) in [0,1] -> Qwen-VL flattened patches + grid_thw (B, 3).

        Mirrors the Qwen VL image processor: resize to image_size, normalize,
        repeat temporally, unfold into merge-ordered patches.
        """
        b = images.shape[0]
        if images.shape[-2:] != (self.image_size, self.image_size):
            images = F.interpolate(images, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        images = (images - self.pixel_mean) / self.pixel_std

        p, m, tp = self.patch_size, self.merge_size, self.temporal_patch_size
        g = self.image_size // p
        # (B, tp, C, H, W)
        x = images.unsqueeze(1).expand(-1, tp, -1, -1, -1)
        # split H,W into (g/m, m, p) blocks, order patches so that the m*m
        # merge-group is contiguous: (gh/m, gw/m, m, m) patch order
        x = x.reshape(b, tp, 3, g // m, m, p, g // m, m, p)
        x = x.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)  # b, gh/m, gw/m, m, m, C, tp, p, p
        x = x.reshape(b * g * g, 3 * tp * p * p)
        grid_thw = torch.tensor([[1, g, g]], device=images.device).expand(b, -1)
        return x, grid_thw

    def forward(
        self,
        images: torch.Tensor | None,  # (B, 3, H, W) in [0, 1]; may be None with image_embeds
        lang_tokens: torch.Tensor,  # (B, L)
        lang_mask: torch.Tensor,  # (B, L) bool, True = valid
        morphology: torch.Tensor | None = None,  # (B, morph_dim) for morph_to_slow
        image_embeds: torch.Tensor | None = None,  # (B, n_img, C) already through the tower
    ) -> torch.Tensor:
        """Returns semantic latent (B, num_readout, out_dim).

        `image_embeds` lets the caller reuse the vision-tower output the fast path
        already computed for the same frame — the tower is ~45 GFLOPs/camera and
        cam0 was otherwise encoded twice per training step.
        """
        if image_embeds is not None:
            b, device = image_embeds.shape[0], image_embeds.device
            pixel_values, grid_thw = None, self._grid_thw(b, device)
        else:
            b, device = images.shape[0], images.device
            pixel_values, grid_thw = self._patchify(images)
        n_img = self.num_image_tokens

        # <vision_start> <image>*n <vision_end> instruction <pad>*K
        ids = torch.cat(
            [
                torch.full((b, 1), self.vision_start_id, dtype=torch.long, device=device),
                torch.full((b, n_img), self.image_token_id, dtype=torch.long, device=device),
                torch.full((b, 1), self.vision_end_id, dtype=torch.long, device=device),
                lang_tokens,
                torch.full((b, self.num_readout), self.pad_token_id, dtype=torch.long, device=device),
            ],
            dim=1,
        )
        attn = torch.cat(
            [
                torch.ones(b, n_img + 2, dtype=torch.bool, device=device),
                lang_mask.bool(),
                torch.ones(b, self.num_readout, dtype=torch.bool, device=device),
            ],
            dim=1,
        )
        mm_type = (ids == self.image_token_id).int()  # text 0, image 1

        use_embeds = image_embeds is not None or (self.morph_to_slow and morphology is not None)
        if use_embeds:
            # embed tokens, scatter image features, optionally add morphology to the
            # readout slots, then run the LM on inputs_embeds (keeps M-RoPE via
            # mm_token_type_ids). This is also the path that consumes precomputed
            # `image_embeds` — identical maths, one fewer vision-tower pass.
            inputs_embeds = self.vlm.get_input_embeddings()(ids)
            if image_embeds is not None:
                img = image_embeds.reshape(-1, image_embeds.shape[-1])
            else:
                img = self.vlm.get_image_features(pixel_values.to(self.pixel_mean.dtype), grid_thw)
                img = torch.cat(img.pooler_output, dim=0)  # (B*n_img, hidden)
            img = img.to(inputs_embeds.dtype)
            imask = (ids == self.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(imask, img)
            if self.morph_to_slow and morphology is not None:
                morph_tok = self.morph_to_lm(morphology.to(inputs_embeds.dtype))  # (B, hidden)
                inputs_embeds[:, -self.num_readout :] = inputs_embeds[:, -self.num_readout :] + morph_tok[:, None, :]
            # With inputs_embeds and no input_ids the model cannot recompute
            # M-RoPE and falls back to a CACHED rope_deltas from a previous call
            # (crashes on a different batch size; silently wrong on the same one).
            # Compute the 3D position ids explicitly from the token layout.
            position_ids, _ = self.vlm.get_rope_index(
                ids,
                image_grid_thw=grid_thw,
                attention_mask=attn,
                mm_token_type_ids=mm_type,
            )
            out = self.vlm(
                inputs_embeds=inputs_embeds,
                image_grid_thw=grid_thw,
                attention_mask=attn,
                mm_token_type_ids=mm_type,
                position_ids=position_ids,
                use_cache=False,
            )
        else:
            out = self.vlm(
                input_ids=ids,
                pixel_values=pixel_values.to(self.pixel_mean.dtype),
                image_grid_thw=grid_thw,
                attention_mask=attn,
                mm_token_type_ids=mm_type,
                use_cache=False,
            )
        readout_h = out.last_hidden_state[:, -self.num_readout :]
        return self.out_proj(readout_h)

    def encode_robot_text(self, desc_ids: torch.Tensor, desc_mask: torch.Tensor) -> torch.Tensor:
        """Encode a robot-description text through the SAME (shared-weight) Qwen
        language stack, text-only (no image, no M-RoPE needed). Trailing pad-token
        readout slots aggregate the description via causal attention; their hidden
        states are projected to `num_morph_readout` conditioning tokens.

        desc_ids/desc_mask: (B, L) tokenized "Robot: X, a N-DOF arm on ..." text.
        Returns (B, num_morph_readout, out_dim).
        """
        b = desc_ids.shape[0]
        device = desc_ids.device
        ids = torch.cat(
            [desc_ids, torch.full((b, self.num_morph_readout), self.pad_token_id, dtype=torch.long, device=device)],
            dim=1,
        )
        attn = torch.cat(
            [desc_mask.bool(), torch.ones(b, self.num_morph_readout, dtype=torch.bool, device=device)],
            dim=1,
        )
        out = self.vlm.language_model(
            input_ids=ids,
            attention_mask=attn,
            use_cache=False,
        )
        readout_h = out.last_hidden_state[:, -self.num_morph_readout :]
        return self.morph_out_proj(readout_h)

    def forward_unified(
        self,
        image: torch.Tensor,  # (B, 3, H, W) current observation
        lang_tokens: torch.Tensor,  # (B, L) task instruction
        lang_mask: torch.Tensor,  # (B, L)
        state: torch.Tensor | None = None,  # (B, max_state_dim) proprio
        robot_ids: torch.Tensor | None = None,  # (B, Lr) robot description text
        robot_mask: torch.Tensor | None = None,
        demo_images: torch.Tensor | None = None,  # (B, K, 3, H, W)
        demo_actions: torch.Tensor | None = None,  # (B, K, T, A)
        n_demo_action_keys: int = 4,
    ) -> torch.Tensor:
        """LLaVA-style unified pass: ALL modalities become tokens in one LM
        sequence, and the policy reads `num_action_readout` latent tokens off the
        LM output. Demo exemplars live INSIDE the attention stream, so the readout
        tokens can attend to them selectively (the thing that "conditioning-set"
        placement could not do).

        Sequence layout:
          [<vis> main-image <-vis>] [instruction] [robot description]
          [ <vis> demo_k image <-vis> demo_k action-keyframe tokens ]*K
          [state token] [action readout slots]
        Returns (B, num_action_readout, out_dim).
        """
        b = image.shape[0]
        device = image.device
        emb = self.vlm.get_input_embeddings()
        n_img = self.num_image_tokens

        # ---- encode all images (main + demos) in one vision pass -------------
        imgs = [image]
        k = 0
        if demo_images is not None:
            k = demo_images.shape[1]
            imgs.append(demo_images.reshape(b * k, *demo_images.shape[2:]))
        all_imgs = torch.cat(imgs, dim=0)
        pixel_values, grid_thw = self._patchify(all_imgs)
        feats = self.vlm.get_image_features(pixel_values.to(self.pixel_mean.dtype), grid_thw)
        feats = torch.cat(feats.pooler_output, dim=0)  # (B*(1+K)*n_img, hidden)
        feats = feats.view(b * (1 + k), n_img, -1)
        main_feat = feats[:b]  # (B, n_img, hidden)
        demo_feat = feats[b:].view(b, k, n_img, -1) if k else None

        seq, mm, msk = [], [], []

        def push(x, is_img=False, mask=None):
            seq.append(x)
            n = x.shape[1]
            mm.append(torch.full((b, n), 1 if is_img else 0, dtype=torch.int, device=device))
            msk.append(mask if mask is not None else torch.ones(b, n, dtype=torch.bool, device=device))

        vs = emb(torch.full((b, 1), self.vision_start_id, dtype=torch.long, device=device))
        ve = emb(torch.full((b, 1), self.vision_end_id, dtype=torch.long, device=device))

        push(vs)
        push(main_feat, is_img=True)
        push(ve)
        push(emb(lang_tokens), mask=lang_mask.bool())
        if robot_ids is not None:
            push(emb(robot_ids), mask=robot_mask.bool())
        if demo_feat is not None:
            keys = torch.linspace(0, demo_actions.shape[2] - 1, n_demo_action_keys, device=device).long()
            for i in range(k):
                push(vs)
                push(demo_feat[:, i], is_img=True)
                push(ve)
                act_tok = self.action_to_lm(demo_actions[:, i, keys].to(main_feat.dtype))
                push(act_tok)  # (B, n_keys, hidden)
        if state is not None:
            push(self.state_to_lm(state.to(main_feat.dtype))[:, None, :])
        push(self.readout_emb[None].expand(b, -1, -1).to(main_feat.dtype))

        inputs_embeds = torch.cat(seq, dim=1)
        mm_type = torch.cat(mm, dim=1)
        attn = torch.cat(msk, dim=1)
        # image_grid_thw must cover only the images actually present in-sequence
        grid_for_seq = grid_thw[: b * (1 + k)]

        # The released unified checkpoint was trained with inputs_embeds only:
        # Transformers 5.5.4 then uses sequential 1D positions on a fresh model.
        # Make that convention explicit so another session's rope_deltas can
        # never alter it. Switching to true multimodal RoPE needs retraining.
        positions = torch.arange(inputs_embeds.shape[1], device=device)
        positions = positions.view(1, 1, -1).expand(4, b, -1)
        out = self.vlm(
            inputs_embeds=inputs_embeds,
            image_grid_thw=grid_for_seq,
            attention_mask=attn,
            mm_token_type_ids=mm_type,
            position_ids=positions,
            use_cache=False,
        )
        readout = out.last_hidden_state[:, -self.num_action_readout :]
        return self.action_readout_proj(readout)
