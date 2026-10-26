"""CPU cache-contract checks using the production policy methods.

Extract the class without optional GPU imports. The tiny text encoder exercises
real tensor values; full T5/action parity is checked separately on the H200.
"""

import ast
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
import pytest
import torch


def policy_class():
    path = Path(__file__).resolve().parents[1] / "lightx2v/models/runners/wan/fastwam_runner.py"
    node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "FastWAMPolicy")
    namespace = {"torch": torch, "np": np, "Optional": Optional, "GET_DTYPE": lambda: torch.float32}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["FastWAMPolicy"], namespace


def encoder(offset=0):
    calls = []
    weights = torch.arange(256 * 4).reshape(256, 4).float() / 100 + offset

    def tokenize(prompts, **kwargs):
        ids = torch.zeros(1, 40, dtype=torch.int64)
        tokens = list(prompts[0].encode("utf-8"))[:40]
        ids[0, :len(tokens)] = torch.tensor(tokens)
        return ids, ids.ne(0).to(torch.int64)

    def model(ids, mask):
        calls.append(ids.clone())
        return weights[ids].clone()

    return SimpleNamespace(tokenizer=tokenize, model=model, calls=calls)


def make_policy(cache):
    cls, namespace = policy_class()
    core = object.__new__(cls)
    core.device = torch.device("cpu")
    core.config = {"fastwam_cache_prompt": cache}
    core.text_encoder = encoder()
    core._prompt_cache = None
    core.pending_actions = deque()
    return core, namespace


def test_cached_context_and_mask_match_uncached_and_skip_encoder():
    cached, _ = make_policy(True)
    ordinary, _ = make_policy(False)
    for _ in range(3):
        actual = cached.encode_prompt("find red")
        expected = ordinary.encode_prompt("find red")
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
        assert torch.count_nonzero(actual[0][8:]) == 0
        assert actual[1].all()  # Preserve the native/reference padded-mask contract.
    assert len(cached.text_encoder.calls) == 1
    assert len(ordinary.text_encoder.calls) == 3


def test_new_instruction_replaces_the_single_entry_and_uses_new_tokens():
    core, _ = make_policy(True)
    red = core.encode_prompt("find red")[0].clone()
    blue = core.encode_prompt("find blue")[0].clone()
    again = core.encode_prompt("find red")[0]
    assert not torch.equal(red, blue)
    torch.testing.assert_close(again, red, atol=0, rtol=0)
    assert len(core.text_encoder.calls) == 3


@pytest.mark.parametrize("method", ["reset", "close"])
def test_episode_reset_and_close_release_language_and_pending_actions(method):
    core, _ = make_policy(True)
    core.encode_prompt("find red")
    core.pending_actions.append(np.ones(5))
    getattr(core, method)()
    assert core._prompt_cache is None and not core.pending_actions
    core.encode_prompt("find red")
    assert len(core.text_encoder.calls) == 2


def test_disabling_clears_cache_before_it_is_enabled_again():
    core, _ = make_policy(True)
    core.encode_prompt("find red")
    core.config["fastwam_cache_prompt"] = False
    core.encode_prompt("find red")
    assert core._prompt_cache is None
    core.config["fastwam_cache_prompt"] = True
    core.encode_prompt("find red")
    assert len(core.text_encoder.calls) == 3


def test_dtype_change_reencodes_and_returns_requested_precision():
    core, namespace = make_policy(True)
    expected = core.encode_prompt("find red")[0].clone()
    namespace["GET_DTYPE"] = lambda: torch.float64
    actual = core.encode_prompt("find red")[0]
    assert actual.dtype == torch.float64 and len(core.text_encoder.calls) == 2
    torch.testing.assert_close(actual, expected.double(), atol=0, rtol=0)


def test_replacing_encoder_does_not_reuse_old_context():
    core, _ = make_policy(True)
    before = core.encode_prompt("find red")[0].clone()
    core.text_encoder = encoder(offset=3)
    after = core.encode_prompt("find red")[0]
    assert not torch.equal(before, after) and len(core.text_encoder.calls) == 1


def test_encoder_error_cannot_leave_stale_successful_entry():
    core, _ = make_policy(True)
    core.encode_prompt("find red")
    def failed_encoder(ids, mask):
        raise ValueError("text encoder failed")
    core.text_encoder.model = failed_encoder
    with pytest.raises(ValueError, match="text encoder failed"):
        core.encode_prompt("find blue")
    assert core._prompt_cache is None
