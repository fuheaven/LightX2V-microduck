"""Per-model correction for Qwen3.5's batch-dependent linear-attention mask.

Transformers 5.5.4 skips apply_mask_to_padding_states when B == 1. TinyVLA
places padded language tokens before its readout, so this changes the actual
conditioning at deployment. Match the existing B > 1 path for single-example
prefills without patching the installed library or another model instance.
"""


def _mask_single_prefill(_module, args, kwargs):
    hidden = kwargs.get("hidden_states") if "hidden_states" in kwargs else args[0]
    mask = kwargs.get("attention_mask") if "attention_mask" in kwargs else (args[2] if len(args) > 2 else None)
    # The upstream batched path and single-token cached decoding stay intact.
    if hidden.shape[0] != 1 or mask is None:
        return None
    if mask.ndim != 2 or tuple(mask.shape) != tuple(hidden.shape[:2]):
        raise ValueError("semantic padding requires a matching 2D token mask")
    if mask.shape[1] <= 1:
        return None
    masked = (hidden * mask[:, :, None]).to(hidden.dtype)
    if "hidden_states" in kwargs:
        return args, {**kwargs, "hidden_states": masked}
    return (masked, *args[1:]), kwargs


def install_semantic_padding(language_model, policy):
    """Install once during model construction; config persists the policy."""
    if policy == "legacy":
        return
    if policy != "batch_consistent_v1":
        raise ValueError("unsupported semantic_padding_policy")
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet

    layers = [layer for layer in language_model.layers if layer.layer_type == "linear_attention"]
    if not layers or any(not isinstance(layer.linear_attn, Qwen3_5GatedDeltaNet) for layer in layers):
        raise ValueError("batch_consistent_v1 requires Qwen3.5 gated-delta layers")
    for layer in layers:
        layer.linear_attn.register_forward_pre_hook(_mask_single_prefill, with_kwargs=True)
