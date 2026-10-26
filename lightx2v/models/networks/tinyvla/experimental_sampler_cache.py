"""Opt-in eager sampler experiment; no cache survives an observation request.

Only the repository's native sampler owns the immutable conditioning tensor.
Training, direct expert calls, masked attention and hooked modules retain their
original implementation. Nothing enables this helper in an entrypoint.
"""
from collections import Counter
import threading

import torch
from torch.nn import functional as F

from lightx2v.models.networks.tinyvla.modeling_tinyvla import TinyVLAModel
from lightx2v.models.networks.tinyvla.modules.expert import CrossAttnBlock, _MHA


class SamplerConditioningCache:
    def __init__(self, model):
        if getattr(model.predict_action_chunk, "__func__", None) is not TinyVLAModel.predict_action_chunk:
            raise ValueError("conditioning reuse requires the unchanged native sampler")
        if "forward" in model.expert.__dict__:
            raise ValueError("compiled or replaced experts require a separate parity experiment")
        self.model = model
        self.blocks = [block for block in model.expert.blocks if type(block) is CrossAttnBlock]
        if not self.blocks or any(type(block.attn) is not _MHA for block in self.blocks):
            raise ValueError("conditioning reuse requires native cross-attention blocks")
        self.counts, self.entries = Counter(), {}
        self.installed = self.active = False
        self.lock = threading.RLock()
        self.peak_bytes = 0

    def __enter__(self):
        if self.installed or "predict_action_chunk" in self.model.__dict__ or any("forward" in b.__dict__ for b in self.blocks):
            raise ValueError("refusing to overwrite an existing sampler experiment")
        self.original_predict = self.model.predict_action_chunk
        self.originals = {block: block.forward for block in self.blocks}
        self.wrappers = {}
        for block in self.blocks:
            def forward(x, cond, cond_mask, current=block):
                return self._forward(current, x, cond, cond_mask)
            self.wrappers[block] = forward
            block.forward = forward
        self.model.predict_action_chunk = self._predict
        self.installed = True
        return self

    def __exit__(self, *exception):
        with self.lock:
            if self.model.predict_action_chunk != self._predict or any(block.forward is not self.wrappers[block] for block in self.blocks):
                raise RuntimeError("sampler implementation changed during the experiment")
            del self.model.predict_action_chunk
            for block in self.blocks:
                del block.forward
            self.entries.clear()
            self.active = self.installed = False

    def snapshot(self):
        return {"protocol": "sampler_conditioning_cache_experiment_v1", "enabled": self.installed,
                "entries": len(self.entries), "counts": dict(self.counts), "peak_cached_bytes": self.peak_bytes,
                "cross_attention_blocks": len(self.blocks), "cross_request_reuse": False,
                "changes_semantic_refresh": False, "counts_as_skill_success": False}

    def _predict(self, *args, **kwargs):
        # A model's sessions must never share request-local K/V tensors.
        with self.lock:
            if self.active:
                raise RuntimeError("nested sampling requests are unsupported")
            self.active = True
            self.entries.clear()
            self.counts["requests"] += 1
            try:
                return self.original_predict(*args, **kwargs)
            finally:
                self.entries.clear()
                self.active = False

    def _forward(self, block, x, cond, cond_mask):
        if not self.active or torch.is_grad_enabled() or cond_mask is not None:
            self.counts["original_path"] += 1
            return self.originals[block](x, cond, cond_mask)
        # Replacing the internal attention call would otherwise suppress hooks.
        if any(module._forward_hooks or module._forward_pre_hooks for module in block.modules()):
            self.counts["hook_bypass"] += 1
            return self.originals[block](x, cond, cond_mask)
        try:
            weights = tuple((id(p), p._version) for p in (block.norm_cond.weight, block.norm_cond.bias,
                                                         block.attn.in_proj_weight, block.attn.in_proj_bias))
        except RuntimeError:
            self.counts["untracked_weight_bypass"] += 1
            return self.originals[block](x, cond, cond_mask)
        h = block.norm1(x)
        b, length, dim = h.shape
        wq, wk, wv = block.attn.in_proj_weight.split(dim, dim=0)
        bq, bk, bv = block.attn.in_proj_bias.split(dim, dim=0)
        context = (weights, block.norm_cond.eps, block.attn.n_heads, str(h.dtype), str(h.device),
                   torch.is_autocast_enabled(h.device.type), str(torch.get_autocast_dtype(h.device.type)))
        entry = self.entries.get(block)
        reuse = entry is not None and entry[0] is cond and entry[1] == context
        # Preserve the native cold-call operator order (norm_cond, Q, K, V).
        normalized = None if reuse else block.norm_cond(cond)
        q = F.linear(h, wq, bq)

        def split(value):
            return value.view(b, -1, block.attn.n_heads, block.attn.head_dim).transpose(1, 2)

        q = split(q)
        if reuse:
            self.counts["hits"] += 1
            k, v = entry[2:]
        else:
            k, v = split(F.linear(normalized, wk, bk)), split(F.linear(normalized, wv, bv))
            self.entries[block] = (cond, context, k, v)
            self.counts["misses"] += 1
            size = sum(t.numel() * t.element_size() for value in self.entries.values() for t in value[2:])
            self.peak_bytes = max(self.peak_bytes, size)
        result = F.scaled_dot_product_attention(q, k, v)
        result = result.transpose(1, 2).reshape(b, length, dim)
        x = x + block.attn.out_proj(result)
        return x + block.mlp(block.norm2(x))
