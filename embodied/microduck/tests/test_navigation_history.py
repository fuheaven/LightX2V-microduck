"""No future leakage, offline/online equality and explicit model-contract gates."""
import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
import pytest
import torch

from lightx2v_microduck.contracts import Observation
from lightx2v_microduck.navigation_history import (
    CausalCameraHistory, HistoryGridSpec, compose_history_grid,
    training_history_contract, verify_resume_history, verify_evaluation_history,
)
from lightx2v_microduck.wam_history_data import FastWAMNavigationHistoryDataset
from lightx2v_microduck.wam_history_policy import FastWAMNavigationHistoryPolicy
from lightx2v_microduck.wam_policy import FastWAMNavigationPolicy
from lightx2v_microduck.wam_data import FastWAMNavigationDataset, build_index
from fixtures_navigation import episode


def frames(n):
    return np.stack([np.full((256, 256, 3), i % 256, np.uint8) for i in range(n)])


def test_recipe_and_resume_cannot_change_input_meaning_at_equal_tensor_shapes():
    ordinary = {'_target_': 'lightx2v_microduck.wam_data.FastWAMNavigationDataset'}
    history = {'_target_': 'lightx2v_microduck.wam_history_data.FastWAMNavigationHistoryDataset',
               'history_offsets_ticks': [300, 150, 50, 0]}
    alternative = {**history, 'history_offsets_ticks': [400, 200, 100, 0]}
    assert training_history_contract(ordinary, ordinary) is None
    contract = training_history_contract(history, history)
    verify_resume_history({'input_history': contract}, contract)
    verify_resume_history({}, None)  # Historical single-frame checkpoints.
    with pytest.raises(ValueError, match='histories differ'):
        training_history_contract(history, ordinary)
    with pytest.raises(ValueError, match='histories differ'):
        training_history_contract(history, alternative)
    with pytest.raises(ValueError, match='dataset implementation'):
        training_history_contract({**ordinary, 'history_offsets_ticks': [300, 150, 50, 0]}, ordinary)
    for receipt in ({}, {'input_history': training_history_contract(alternative, alternative)}):
        with pytest.raises(ValueError, match='resume observation history'):
            verify_resume_history(receipt, contract)


def test_future_mutation_cannot_change_conditioning_image():
    spec = HistoryGridSpec((15, 10, 5, 0))
    images, ticks = frames(30), np.arange(30) * 5
    before = compose_history_grid(images, ticks, 50, spec)
    images[11:] = 255
    after = compose_history_grid(images, ticks, 50, spec)
    assert all(np.array_equal(a, b) for a, b in zip(before, after))
    assert before[1].tolist() == [35, 40, 45, 50]


def test_evaluation_cannot_relabel_single_frame_weights_as_trained_history():
    spec = HistoryGridSpec((300, 150, 50, 0))
    receipt = {'input_history': spec.as_dict(), 'weights_sha256': 'weights',
               'stats_sha256': 'stats', 'step': 1}
    verify_evaluation_history(receipt, spec, weights_sha256='weights', stats_sha256='stats')
    for changed in ({'input_history': None}, {'weights_sha256': 'old'},
                    {'stats_sha256': 'other'}, {'step': 0}, {'step': True}):
        with pytest.raises(ValueError):
            verify_evaluation_history({**receipt, **changed}, spec,
                                      weights_sha256='weights', stats_sha256='stats')


def test_online_pruning_padding_reset_and_order_match_offline():
    spec = HistoryGridSpec((100, 50, 25, 0))
    images, ticks = frames(200), np.arange(200) * 5
    history = CausalCameraHistory(spec)
    history.reset('a')
    for tick, image in zip(ticks, images):
        history.append('a', int(tick), image)
        actual = history.compose(int(tick))
        expected = compose_history_grid(images, ticks, int(tick), spec)
        assert all(np.array_equal(a, b) for a, b in zip(actual, expected))
    assert len(history._ticks) < 25
    with pytest.raises(ValueError, match='rewrite or reorder'):
        history.append('a', 0, images[0])
    history.reset('b')
    with pytest.raises(ValueError, match='prefix'):
        history.append('b', 5, images[1])
    with pytest.raises(ValueError, match='reset'):
        history.append('a', 0, images[0])
    history.append('b', 0, images[0])
    assert history.compose(0)[2].tolist() == [True, True, True, False]


def test_observation_copy_duplicate_and_missing_samples():
    spec = HistoryGridSpec((30, 20, 10, 0))
    history = CausalCameraHistory(spec)
    history.reset('a')
    image = frames(1)[0]
    history.append('a', 0, image)
    image[:] = 42
    assert not history.compose(0)[0].any()
    with pytest.raises(ValueError, match='rewrite'):
        history.append('a', 0, image)
    history.append('a', 40, image)
    with pytest.raises(ValueError, match='stale'):
        history.compose(40)


@pytest.mark.parametrize('offsets', [(0,), (15, 10, 0, 5), (15, 10, 5, 5), (15, 10, 1, 0), (15, 10, 5, False)])
def test_invalid_history_is_rejected(offsets):
    with pytest.raises(ValueError, match='history requires'):
        HistoryGridSpec(offsets)


def test_training_dataset_equals_live_grid_without_changing_actions_or_masks(tmp_path):
    episode(tmp_path, 0)
    build_index(tmp_path)
    ordinary = FastWAMNavigationDataset(tmp_path, sample_stride=1)
    history = FastWAMNavigationHistoryDataset(tmp_path, sample_stride=1, history_offsets_ticks=[15, 10, 5, 0])
    sample = history[6]
    expected, ticks, padding = compose_history_grid(history.cache['rgb'], history.cache['ticks'], 30, history.history_spec)
    expected = torch.from_numpy(expected).permute(2, 0, 1).float().div(127.5).sub(1)
    torch.testing.assert_close(sample['video'][:, 0], expected, atol=0, rtol=0)
    assert sample['video'].shape == (3, 9, 512, 512)
    assert sample['history_source_ticks'][0].tolist() == ticks.tolist()
    for key in ('action', 'proprio', 'action_is_pad', 'image_is_pad', 'proprio_is_pad'):
        torch.testing.assert_close(sample[key], ordinary[6][key], atol=0, rtol=0)
    with pytest.raises(ValueError, match='prefixes'):
        FastWAMNavigationHistoryDataset(tmp_path, history_offsets_ticks=[15, 10, 5, 0], recovery_roots=['missing'])


def test_policy_requires_explicit_history_core_and_clears_episode_memory():
    spec = HistoryGridSpec((15, 10, 5, 0))
    captured = []
    core = SimpleNamespace(policy_profile='microduck_navigation', action_dim=5, robot_state_dim=48,
                           camera_size=512, gripper_postprocess=False, history_contract=spec.as_dict(), reset=lambda: None)
    def predict(**kwargs):
        captured.append(kwargs)
        return np.zeros((32, 5), np.float32)
    core.predict_action_chunk = predict
    with pytest.raises(ValueError, match='contract'):
        FastWAMNavigationPolicy(core, model_id='wrong', artifact_sha256='x')
    policy = FastWAMNavigationHistoryPolicy(core, model_id='experimental', artifact_sha256='x')
    policy.reset('e')
    images = frames(4)
    for i in range(4):
        observation = Observation('e', i * 5, i * .1, np.zeros(61), images[i], 'Find red', camera_tick=i * 5, monotonic_ns=i + 1)
        policy.observe(observation)
    chunk = policy.predict(observation)
    expected = compose_history_grid(images, np.arange(4) * 5, 15, spec)[0]
    assert np.array_equal(captured[-1]['images']['head_camera'], expected)
    assert captured[-1]['state'].shape == (48,)
    assert chunk.diagnostics['history_source_ticks'] == [0, 5, 10, 15]
    assert chunk.diagnostics['input'] == 'head_rgb_history_grid+instruction+proprio48'
    policy.reset('new')
    with pytest.raises(ValueError, match='reset'):
        policy.predict(observation)
    del core.history_contract
    with pytest.raises(ValueError, match='single-frame'):
        FastWAMNavigationHistoryPolicy(core, model_id='wrong', artifact_sha256='x')


@pytest.mark.parametrize('fallback_ticks', [None, 300])
def test_native_core_rejects_history_resolution_without_matching_contract(tmp_path, fallback_ticks):
    # Execute production constructor validation, stopping before optional GPU
    # model loading. Full reference/native model alignment remains a GPU gate.
    path = Path(__file__).resolve().parents[3] / 'lightx2v/models/runners/wan/fastwam_runner.py'
    node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == 'FastWAMPolicy')
    namespace = {'torch': torch, 'np': np, 'Optional': Optional, 'Path': Path,
                 '_canonical_norm_mode': lambda mode: mode}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    cls = namespace['FastWAMPolicy']
    class ReachedModelLoading(Exception):
        pass
    def stop(self):
        raise ReachedModelLoading()
    cls._check_model_config = stop
    kwargs = dict(adapter_model_path=tmp_path, dataset_stats_path=tmp_path, model_path=tmp_path,
                  device='cpu', default_prompt='test', policy_profile='microduck_navigation', action_dim=5,
                  robot_state_dim=48, camera_size=512, gripper_postprocess=False)
    with pytest.raises(ValueError, match='input contract'):
        cls(**kwargs, config={})
    config = {'navigation_history': HistoryGridSpec((300, 150, 50, 0)).as_dict()}
    with pytest.raises(ReachedModelLoading):
        cls(**kwargs, config=config)
    config['navigation_history']['selector'] = 'privileged_target_visible_frames'
    with pytest.raises(ValueError, match='input contract'):
        cls(**kwargs, config=config)
    from lightx2v_microduck.navigation_visual_memory import TargetMemoryGridSpec
    memory = TargetMemoryGridSpec(fallback_history_ticks=fallback_ticks).as_dict()
    with pytest.raises(ReachedModelLoading):
        cls(**kwargs, config={'navigation_history': memory})
    for changed in ({'detector_source_sha256': 'unverified'},
                    {'selector': 'privileged_target_visible_frames'},
                    {'capture_age': 'hidden_current_target_position'}):
        with pytest.raises(ValueError, match='input contract'):
            cls(**kwargs, config={'navigation_history': dict(memory, **changed)})
    with pytest.raises(ValueError, match='camera size'):
        cls(**dict(kwargs, camera_size=256), config={'navigation_history': memory})
