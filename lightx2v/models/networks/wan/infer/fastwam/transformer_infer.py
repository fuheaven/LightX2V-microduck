import torch
import torch.nn.functional as F


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class FastWAMTransformerInfer:
    def __init__(self, config):
        self.config = config
        self.num_layers = int(config["num_layers"])
        self.num_heads = int(config["num_heads"])
        self.head_dim = int(config["dim"]) // self.num_heads

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def _split_modulation(self, self_attn, t_mod):
        mod = self_attn.modulation.tensor.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)
        return (
            shift_msa.squeeze(1),
            scale_msa.squeeze(1),
            gate_msa.squeeze(1),
            shift_mlp.squeeze(1),
            scale_mlp.squeeze(1),
            gate_mlp.squeeze(1),
        )

    def _reshape_heads(self, x):
        return x.reshape(x.shape[0], self.num_heads, self.head_dim)

    def _rms(self, norm, x, *, action=False):
        if not action or not self.config.get("fastwam_action_rms_one_pass", False):
            return norm.apply(x)
        # Preserve the original full projection axis, before splitting heads.
        # The experimental kernel preserves BF16's intermediate cast but may
        # still differ at reduction-rounding boundaries; it is opt-in only.
        if type(norm).__name__ != "RMSWeight" or not x.is_cuda or x.dtype != torch.bfloat16:
            raise ValueError("action RMS fusion requires the torch RMS backend and CUDA BF16")
        w = norm._get_actual_weight()
        if w is None or w.dtype != x.dtype or w.device != x.device or w.numel() != x.shape[-1]:
            raise ValueError("action RMS fusion requires matching full-projection BF16 weights")
        if norm.sensitive_layer_dtype != norm.infer_dtype:
            raise ValueError("action RMS fusion does not support a sensitive-layer precision override")
        from lightx2v.common.ops.norm.triton_ops import rms_norm_kernel

        return rms_norm_kernel(x, w, norm.eps, match_torch_rms_cast=True)

    def _build_self_attention_io(self, block, x, freqs, t_mod, *, action=False):
        self_attn = block.self_attn
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self._split_modulation(self_attn, t_mod)
        attn_input = modulate(self_attn.norm1.apply(x), shift_msa, scale_msa)

        q = self._reshape_heads(self._rms(self_attn.norm_q, self_attn.q.apply(attn_input), action=action))
        k = self._reshape_heads(self._rms(self_attn.norm_k, self_attn.k.apply(attn_input), action=action))
        v = self._reshape_heads(self_attn.v.apply(attn_input))
        q, k = self_attn.rope.apply(q, k, freqs)
        return q, k, v, x, gate_msa, shift_mlp, scale_mlp, gate_mlp

    def _cross_attn(self, block, x, context, context_mask, context_kv=None, *, action=False):
        cross_attn = block.cross_attn
        q = self._reshape_heads(self._rms(cross_attn.norm_q, cross_attn.q.apply(cross_attn.norm3.apply(x)), action=action))
        if context_kv is None:
            k = self._reshape_heads(self._rms(cross_attn.norm_k, cross_attn.k.apply(context), action=action))
            v = self._reshape_heads(cross_attn.v.apply(context))
        else:
            k, v = context_kv
        out = cross_attn.attn.apply(q, k, v, attn_mask=context_mask)
        return cross_attn.o.apply(out)

    def _post_block(self, block, residual_x, mixed_attn_out, gate_msa, shift_mlp, scale_mlp, gate_mlp, context, context_mask, context_kv=None, *, action=False):
        x = residual_x + gate_msa * block.self_attn.o.apply(mixed_attn_out)
        if context is not None:
            x = x + self._cross_attn(block, x, context, context_mask, context_kv=context_kv, action=action)
        mlp_input = modulate(block.ffn.norm2.apply(x), shift_mlp, scale_mlp)
        x = x + gate_mlp * block.ffn.fc2.apply(F.gelu(block.ffn.fc0.apply(mlp_input), approximate="tanh"))
        return x

    def prefill_video_cache(self, weights, video_pre):
        x = video_pre.tokens
        video_mask = torch.ones((x.shape[0], x.shape[0]), dtype=torch.bool, device=x.device)
        kv_cache = []
        for layer_idx in range(self.num_layers):
            block = weights.video.blocks[layer_idx]
            q, k, v, residual_x, gate_msa, shift_mlp, scale_mlp, gate_mlp = self._build_self_attention_io(
                block,
                x,
                video_pre.freqs,
                video_pre.t_mod,
            )
            mixed = block.self_attn.attn.apply(q, k, v, attn_mask=video_mask)
            x = self._post_block(
                block,
                residual_x,
                mixed,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                video_pre.context,
                video_pre.context_mask,
            )
            kv_cache.append({"k": k, "v": v})
        return kv_cache

    def prefill_action_context(self, weights, context):
        """Project the current observation's conditions once per action plan.

        Context already includes the current proprioception token. This cache
        belongs only to this plan and must never be reused across observations.
        """
        cache = []
        for layer_idx in range(self.num_layers):
            cross = weights.action.blocks[layer_idx].cross_attn
            k = self._reshape_heads(self._rms(cross.norm_k, cross.k.apply(context), action=True))
            v = self._reshape_heads(cross.v.apply(context))
            cache.append((k, v))
        return cache

    def action_with_video_cache(self, weights, action_pre, video_kv_cache, video_seq_len, attention_mask, context_kv_cache=None):
        x = action_pre.tokens
        action_mask = attention_mask[video_seq_len:, : video_seq_len + x.shape[0]]
        for layer_idx in range(self.num_layers):
            block = weights.action.blocks[layer_idx]
            q, k_action, v_action, residual_x, gate_msa, shift_mlp, scale_mlp, gate_mlp = self._build_self_attention_io(
                block,
                x,
                action_pre.freqs,
                action_pre.t_mod,
                action=True,
            )
            k = torch.cat([video_kv_cache[layer_idx]["k"], k_action], dim=0)
            v = torch.cat([video_kv_cache[layer_idx]["v"], v_action], dim=0)
            mixed = block.self_attn.attn.apply(q, k, v, attn_mask=action_mask)
            x = self._post_block(
                block,
                residual_x,
                mixed,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                action_pre.context,
                action_pre.context_mask,
                context_kv=None if context_kv_cache is None else context_kv_cache[layer_idx],
                action=True,
            )
        return weights.action_head.apply(x)

    @staticmethod
    def build_mot_attention_mask(video_seq_len, action_seq_len, video_tokens_per_frame, device):
        total = int(video_seq_len) + int(action_seq_len)
        mask = torch.zeros((total, total), dtype=torch.bool, device=device)
        mask[:video_seq_len, :video_seq_len] = True
        mask[video_seq_len:, video_seq_len:] = True
        mask[video_seq_len:, : min(video_tokens_per_frame, video_seq_len)] = True
        return mask
