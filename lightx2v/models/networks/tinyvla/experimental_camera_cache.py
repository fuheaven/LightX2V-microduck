"""Opt-in experiment: reuse only frozen, constant-black auxiliary-camera tokens.

Nothing imports or enables this helper in production/training entrypoints. It
does not cache main-camera features, language outputs, state or trainable
projections. Native model and gradient parity must precede any promotion.
"""
from collections import Counter, OrderedDict

import torch
from torch import nn


class FrozenAuxiliaryCameraCache:
    key = "observation.images.cam1"

    def __init__(self, model, *, max_entries=2, max_bytes=64 * 1024 * 1024):
        if type(max_entries) is not int or not 1 <= max_entries <= 16 or type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError("camera cache requires bounded positive capacity")
        if self.key not in model.image_keys or model.image_keys[0] == self.key:
            raise ValueError("only the explicit auxiliary cam1 input may be cached")
        self.model, self.max_entries, self.max_bytes = model, max_entries, max_bytes
        self.entries, self.counts = OrderedDict(), Counter()
        self.installed = False
        self.fingerprint = None
        self.bytes = 0

    def __enter__(self):
        if self.installed or "_vision_features" in self.model.__dict__:
            raise ValueError("refusing to overwrite an existing per-instance vision implementation")
        self.original = self.model._vision_features
        self.model._vision_features = self._features
        self.installed = True
        return self

    def __exit__(self, *exception):
        if self.installed:
            if self.model._vision_features != self._features:
                raise RuntimeError("vision implementation changed during the cache experiment")
            del self.model._vision_features
            self.installed = False
            self.clear()

    def clear(self):
        self.entries.clear()
        self.bytes = 0

    def snapshot(self):
        return {"protocol": "frozen_auxiliary_camera_cache_experiment_v1", "enabled": self.installed,
                "entries": len(self.entries), "bytes": self.bytes, "counts": dict(self.counts),
                "changes_semantic_refresh": False, "counts_as_skill_success": False}

    @staticmethod
    def _tensor_identity(tensor):
        # An inference-mode-created parameter has no version counter. Refuse
        # caching in that case instead of pretending mutation can be detected.
        return (id(tensor), tensor._version, tuple(tensor.shape), str(tensor.dtype), str(tensor.device))

    def _context(self, image):
        semantic = self.model.semantic
        vision = getattr(getattr(semantic, "vlm", None), "visual", None)
        if vision is None or semantic.vision_backend not in ("fixed_eager", "hf"):
            return None, "unsupported_encoder"
        if image.requires_grad or any(p.requires_grad for p in vision.parameters()):
            return None, "trainable_vision_or_input"
        for module in vision.modules():
            if module.training and ((isinstance(module, nn.modules.dropout._DropoutNd) and module.p)
                                    or isinstance(module, nn.modules.batchnorm._BatchNorm)):
                return None, "stateful_or_stochastic_vision"
            if module._forward_hooks or module._forward_pre_hooks:
                return None, "vision_hooks"
        if not image.is_floating_point() or bool(torch.count_nonzero(image).item()):
            return None, "nonconstant_auxiliary_image"
        try:
            identity = tuple((name, self._tensor_identity(value)) for name, value in
                             [*vision.named_parameters(), *vision.named_buffers(),
                              ("pixel_mean", semantic.pixel_mean), ("pixel_std", semantic.pixel_std)])
        except RuntimeError:
            return None, "untracked_tensor_version"
        if identity != self.fingerprint:
            if self.fingerprint is not None:
                self.counts["weight_or_buffer_invalidations"] += 1
            self.clear()
            self.fingerprint = identity
        device_type = image.device.type
        if device_type not in ("cpu", "cuda"):
            return None, "unsupported_device"
        context = (tuple(image.shape), tuple(image.stride()), str(image.dtype), str(image.device),
                   vision.training, torch.is_grad_enabled(), torch.is_inference_mode_enabled(),
                   torch.is_autocast_enabled(device_type), str(torch.get_autocast_dtype(device_type)),
                   torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32,
                   torch.backends.cudnn.allow_tf32, semantic.vision_backend,
                   id(getattr(semantic.encode_image, "__func__", semantic.encode_image)),
                   tuple(getattr(semantic, name, None) for name in
                         ("image_size", "patch_size", "merge_size", "temporal_patch_size", "num_image_tokens")))
        return context, None

    def _features(self, batch, keys):
        keys = tuple(keys)
        if self.key not in keys:
            return self.original(batch, keys)
        image = self.model._last_obs(batch[self.key], 4)
        context, reason = self._context(image)
        if reason is not None:
            self.counts["bypass_" + reason] += 1
            return self.original(batch, keys)
        if context in self.entries:
            self.counts["hits"] += 1
            tokens = self.entries.pop(context)
            self.entries[context] = tokens
            features = self.original(batch, [key for key in keys if key != self.key])
            features[self.key] = tokens
            return {key: features[key] for key in keys}
        self.counts["misses"] += 1
        features = self.original(batch, keys)
        tokens = features[self.key]
        size = tokens.numel() * tokens.element_size()
        if tokens.requires_grad:
            self.counts["bypass_gradient_output"] += 1
        elif size > self.max_bytes:
            self.counts["bypass_capacity"] += 1
        else:
            while self.entries and (len(self.entries) >= self.max_entries or self.bytes + size > self.max_bytes):
                _, previous = self.entries.popitem(last=False)
                self.bytes -= previous.numel() * previous.element_size()
                self.counts["evictions"] += 1
            self.entries[context] = tokens
            self.bytes += size
        return features
