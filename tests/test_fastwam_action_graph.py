"""Graph inputs must refresh observation, language, state and sample noise."""

import importlib.util
from pathlib import Path

import pytest
import torch


def graph_module():
    path = Path(__file__).resolve().parents[1] / "lightx2v/models/networks/wan/infer/fastwam/action_graph.py"
    spec = importlib.util.spec_from_file_location("action_graph", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_refresh_all_nested_conditions_without_aliasing_previous_observation():
    module = graph_module()
    original = {"context": torch.ones(3, 4), "context_mask": torch.ones(3, dtype=torch.bool),
                "video_kv_cache": [{"k": torch.arange(12).reshape(3, 4), "v": torch.zeros(3, 4)}],
                "action_context_kv_cache": [(torch.zeros(3, 4), torch.ones(3, 4))], "video_seq_len": 3}
    destination = module.clone_inputs(original)
    source = module.clone_inputs(original)
    source["context"][-1] = 9
    source["context_mask"][0] = False
    source["video_kv_cache"][0]["k"] += 30
    source["action_context_kv_cache"][0][1].fill_(7)
    pointers = (destination["context"].data_ptr(), destination["video_kv_cache"][0]["k"].data_ptr())
    module.copy_inputs(destination, source)
    assert pointers == (destination["context"].data_ptr(), destination["video_kv_cache"][0]["k"].data_ptr())
    torch.testing.assert_close(destination["context"], source["context"], atol=0, rtol=0)
    assert not destination["context_mask"][0]
    assert torch.all(destination["action_context_kv_cache"][0][1] == 7)
    assert torch.all(original["context"] == 1)
    torch.testing.assert_close(destination["video_kv_cache"][0]["k"], source["video_kv_cache"][0]["k"], atol=0, rtol=0)
    with pytest.raises(ValueError, match="fields changed"):
        module.copy_inputs(destination, {"context": source["context"]})
    with pytest.raises(ValueError, match="shape, dtype or device"):
        module.copy_inputs(destination["context"], torch.ones(4, 4))
    with pytest.raises(ValueError, match="static parameter"):
        module.copy_inputs(destination, {**source, "video_seq_len": 4})
