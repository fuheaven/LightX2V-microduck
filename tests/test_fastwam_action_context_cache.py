"""Numerical cache checks using the real native attention implementation."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
import pytest


def module(name):
    path = Path(__file__).resolve().parents[1] / f"lightx2v/models/networks/wan/infer/fastwam/{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def linear(input_dim=8, output_dim=8):
    w, b = torch.randn(output_dim, input_dim) / 3, torch.randn(output_dim) / 10
    return SimpleNamespace(apply=lambda x: F.linear(x, w, b))


def attention(q, k, v, attn_mask):
    return F.scaled_dot_product_attention(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1),
                                         attn_mask=attn_mask).transpose(0, 1).reshape(q.shape[0], -1)


def test_cross_attention_cache_matches_and_new_state_requires_new_projections():
    torch.manual_seed(137)
    infer = module("transformer_infer").FastWAMTransformerInfer({"dim": 8, "num_heads": 2, "num_layers": 2})
    norm = SimpleNamespace(apply=lambda x: F.rms_norm(x, (8,), eps=1e-6))
    blocks = [SimpleNamespace(cross_attn=SimpleNamespace(
        q=linear(), k=linear(), v=linear(), o=linear(), norm3=norm, norm_q=norm, norm_k=norm,
        attn=SimpleNamespace(apply=attention))) for _ in range(2)]
    weights = SimpleNamespace(action=SimpleNamespace(blocks=blocks))
    context = torch.randn(7, 8)
    mask = torch.ones(3, 7, dtype=torch.bool)
    mask[:, 1] = False
    cache = infer.prefill_action_context(weights, context)
    for _ in range(3):
        x = torch.randn(3, 8)
        for block, kv in zip(blocks, cache):
            torch.testing.assert_close(infer._cross_attn(block, x, context, mask, kv),
                                       infer._cross_attn(block, x, context, mask), atol=0, rtol=0)
    # The appended state token changes with the next observation. A cache from
    # the old state demonstrably gives a different result in this test.
    updated = context.clone()
    updated[-1] += torch.arange(8).float()
    new_cache = infer.prefill_action_context(weights, updated)
    expected = infer._cross_attn(blocks[0], x, updated, mask)
    torch.testing.assert_close(infer._cross_attn(blocks[0], x, updated, mask, new_cache[0]), expected, atol=0, rtol=0)
    assert not torch.allclose(infer._cross_attn(blocks[0], x, updated, mask, cache[0]), expected)


def test_preembedded_conditions_preserve_action_preparation_at_different_flow_times():
    torch.manual_seed(138)
    infer = module("pre_infer").FastWAMPreInfer({"dim": 8, "num_heads": 2, "num_layers": 2,
                                               "freq_dim": 8, "action_dim_hidden": 8})
    weights = SimpleNamespace(action=SimpleNamespace(
        action_encoder=linear(5, 8), time_embedding_0=linear(), time_embedding_2=linear(),
        time_projection_1=linear(8, 48), text_embedding_0=linear(), text_embedding_2=linear()))
    context = torch.randn(7, 8)
    embedded = infer._text_embedding(weights.action, context)
    for time in (.1, .5, .9):
        latents = torch.randn(1, 3, 5)
        inputs = (weights, latents, torch.tensor([time]), context, torch.ones(7, dtype=torch.bool))
        ordinary = infer.infer_action(*inputs)
        cached = infer.infer_action(*inputs, embedded_context=embedded)
        for key in ("tokens", "t_mod", "context", "context_mask", "freqs"):
            torch.testing.assert_close(getattr(cached, key), getattr(ordinary, key), atol=0, rtol=0)


def test_experimental_action_rms_leaves_video_and_default_path_unchanged():
    infer = module("transformer_infer").FastWAMTransformerInfer({"dim": 8, "num_heads": 2, "num_layers": 2})
    x = torch.randn(3, 8)
    norm = SimpleNamespace(apply=lambda t: F.rms_norm(t, (8,), eps=1e-6))
    expected = norm.apply(x)
    torch.testing.assert_close(infer._rms(norm, x, action=True), expected, atol=0, rtol=0)
    infer.config["fastwam_action_rms_one_pass"] = True
    torch.testing.assert_close(infer._rms(norm, x, action=False), expected, atol=0, rtol=0)
    with pytest.raises(ValueError, match="torch RMS backend and CUDA BF16"):
        infer._rms(norm, x, action=True)


def test_action_graph_rejects_a_changed_rms_selection():
    graph = module("action_graph")
    captured = graph.clone_inputs({"action_rms_one_pass": False, "context": torch.randn(3, 8)})
    with pytest.raises(ValueError, match="static parameter changed"):
        graph.copy_inputs(captured, {"action_rms_one_pass": True, "context": torch.randn(3, 8)})
