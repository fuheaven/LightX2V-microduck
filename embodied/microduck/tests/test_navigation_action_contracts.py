"""An opt-in command extension must never silently change legacy execution."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import os

import numpy as np
import pytest

from lightx2v_microduck.contracts import Observation
from lightx2v_microduck.navigation import (
    ACTION_HIGH, NavigationBuffer, NavigationChunk, intent_from_action)
from lightx2v_microduck.navigation_action_contracts import (
    LEGACY_NAVIGATION_CONTRACT as OLD,
    FORWARD04_NAVIGATION_CONTRACT as NEW,
    action_contract, verify_artifact_action_contract)
from lightx2v_microduck.wam_policy import FastWAMNavigationPolicy


def observed(tick=0, ns=None):
    return Observation('test-episode', tick, tick * .02, np.arange(61, dtype=np.float32),
        np.zeros((256, 256, 3), np.uint8), 'Find the red ball', 'find_red_ball',
        camera_tick=tick, monotonic_ns=1_000_000_000 + tick * 20_000_000 if ns is None else ns)


def chunk(contract=OLD, speed=.2, tick=0):
    obs = observed(tick)
    return NavigationChunk(np.tile([speed, 0, .1, 0, 0], (32, 1)), obs.episode_id,
        tick, obs.monotonic_ns, tick + 25, contract=contract)


def core(contract=OLD, history=False):
    def predict(images, state, task_description):
        assert set(images) == {'head_camera'} and state.shape == (48,)
        assert task_description == 'Find the red ball'
        return np.tile([.45, 0, .1, 0, 0], (32, 1))
    value = SimpleNamespace(policy_profile='microduck_navigation', action_dim=5,
        robot_state_dim=48, camera_size=256, gripper_postprocess=False, reset=lambda: None,
        predict_action_chunk=predict, config={'navigation_contract': contract})
    if history:
        from lightx2v_microduck.navigation_history import HistoryGridSpec
        spec = HistoryGridSpec((300, 150, 50, 0))
        value.camera_size = spec.output_size
        value.history_contract = spec.as_dict()
    return value


def test_legacy_rejects_extension_and_bounds_cannot_be_mutated_globally():
    assert ACTION_HIGH[0] == np.float32(.3)
    with pytest.raises(ValueError, match='bounds'):
        chunk(speed=.4)
    with pytest.raises(ValueError, match='bounds'):
        intent_from_action([.4, 0, 0, 0, 0], episode_id='test-episode', tick=0)
    _, high = action_contract(NEW).bounds()
    high[0] = 100
    assert action_contract(NEW).bounds()[1][0] == np.float32(.4)
    assert chunk(NEW, .4).actions[0, 0] == np.float32(.4)
    assert not chunk(NEW, .4).actions.flags.writeable
    for speed in (.401, float('nan'), float('inf')):
        with pytest.raises(ValueError):
            chunk(NEW, speed)
    with pytest.raises(ValueError, match='unsupported'):
        chunk('unregistered-contract')


@pytest.mark.parametrize('receiver,sender', [(OLD, NEW), (NEW, OLD)])
def test_buffer_rejects_mixed_contracts_even_when_numeric_action_fits_both(receiver, sender):
    buffer = NavigationBuffer(contract=receiver)
    buffer.reset('test-episode')
    assert not buffer.submit(chunk(sender, .2), observed())
    assert buffer.last_reason == 'wrong_contract'
    intent, evidence = buffer.consume(observed())
    assert intent.skill == 'stand' and evidence['execution_source'] == 'official_stand_fallback'


@pytest.mark.parametrize('expiry', ['physical', 'wall'])
def test_extended_actions_keep_source_alignment_and_expire_to_official_stand(expiry):
    buffer = NavigationBuffer(contract=NEW)
    buffer.reset('test-episode')
    assert buffer.submit(chunk(NEW, .4), observed(11))
    intent, evidence = buffer.consume(observed(11))
    assert intent.command[0] == np.float32(.4) and evidence['action_index'] == 2
    late = observed(25) if expiry == 'physical' else observed(12, ns=1_500_000_000)
    intent, evidence = buffer.consume(late)
    assert intent.skill == 'stand' and not intent.command.any() and evidence['reason'] == 'expired'
    assert buffer.chunk is None
    buffer.reset('different-episode')
    assert buffer.contract == NEW and not buffer.submit(chunk(NEW), observed())


@pytest.mark.parametrize('history', [False, True])
@pytest.mark.parametrize('contract,speed', [(OLD, .3), (NEW, .4)])
def test_policy_clipping_and_chunk_metadata_match_explicit_core_contract(history, contract, speed):
    from lightx2v_microduck.wam_history_policy import FastWAMNavigationHistoryPolicy
    cls = FastWAMNavigationHistoryPolicy if history else FastWAMNavigationPolicy
    policy = cls(core(contract, history), model_id='contract-test-double', artifact_sha256='not-weights')
    policy.reset('test-episode')
    result = policy.predict(observed())
    assert result.contract == result.diagnostics['contract'] == policy.metadata['contract'] == contract
    assert result.actions[0, 0] == np.float32(speed) and result.diagnostics['clipped_values'] == 32
    assert result.source_tick == 0 and result.valid_until_tick == 25
    mismatched = NEW if contract == OLD else OLD
    with pytest.raises(ValueError, match='differs'):
        cls(core(contract, history), model_id='test', artifact_sha256='test', contract=mismatched)


def binding_fixture():
    # Coordinate provenance remains the old train-only schema; this is not a
    # trained artifact and is never admitted as one by the test.
    binding = dict(execution_contract=NEW, coordinate_contract=OLD, stats_sha256='a' * 64)
    manifest = dict(contract=NEW, embodiment='legs', files={'dataset_stats.json': 'a' * 64},
                    normalization_binding=binding)
    config = dict(navigation_contract=NEW, normalization_binding=dict(binding))
    stats = dict(schema=OLD, normalization_split='train')
    return manifest, config, stats


def test_explicit_binding_preserves_original_normalizer_schema():
    manifest, config, stats = binding_fixture()
    assert verify_artifact_action_contract(manifest, config, stats) == NEW
    assert stats == dict(schema=OLD, normalization_split='train')
    assert verify_artifact_action_contract(dict(contract=OLD, embodiment='legs'), {}, stats) == OLD


@pytest.mark.parametrize('change', ['config', 'missing_binding', 'coordinate', 'digest', 'split', 'body', 'no_digest'])
def test_artifact_rejects_implicit_or_incompatible_normalizer_rebinding(change):
    manifest, config, stats = binding_fixture()
    if change == 'config':
        config['navigation_contract'] = OLD
    elif change == 'missing_binding':
        manifest.pop('normalization_binding')
        config.pop('normalization_binding')
    elif change == 'coordinate':
        stats['schema'] = NEW
    elif change == 'digest':
        manifest['files']['dataset_stats.json'] = 'b' * 64
    elif change == 'split':
        stats['normalization_split'] = 'validation'
    elif change == 'body':
        manifest['embodiment'] = 'roller'
    else:
        manifest['normalization_binding']['stats_sha256'] = None
        config['normalization_binding']['stats_sha256'] = None
        manifest['files']['dataset_stats.json'] = None
    with pytest.raises(ValueError):
        verify_artifact_action_contract(manifest, config, stats)


def test_direct_native_config_cannot_bypass_normalization_binding(tmp_path):
    from lightx2v_microduck.wam_policy import native_navigation_config
    backbone = tmp_path / 'backbone'
    backbone.mkdir()
    architecture = dict(dim=3072, num_heads=24, num_layers=30, in_dim=48,
        ffn_dim=14336, out_dim=48, freq_dim=256, eps=1e-6)
    (backbone / 'config.json').write_text(json.dumps(architecture))
    stats = tmp_path / 'dataset_stats.json'
    stats.write_text(json.dumps(dict(schema=OLD, normalization_split='train')))
    with pytest.raises(ValueError, match='binding'):
        native_navigation_config(backbone, tmp_path / 'model.pt', stats, contract=NEW)
    binding = dict(execution_contract=NEW, coordinate_contract=OLD,
        stats_sha256=hashlib.sha256(stats.read_bytes()).hexdigest())
    config = native_navigation_config(backbone, tmp_path / 'model.pt', stats,
        contract=NEW, normalization_binding=binding)
    assert config['navigation_contract'] == NEW and config['normalization_binding'] == binding


def test_reference_core_exposes_same_explicit_contract_without_refitting_stats(monkeypatch):
    import lightx2v_microduck.wam_reference as reference
    model = SimpleNamespace(proprio_dim=48, action_expert=SimpleNamespace(action_dim=5))
    model.eval = lambda: model
    model.requires_grad_ = lambda enabled: model
    monkeypatch.setattr(reference, 'reference_processor', lambda stats, horizon: (stats, horizon))
    manifest, config, stats = binding_fixture()
    with pytest.raises(ValueError, match='binding'):
        reference.ReferenceNavigationCore(model, stats, navigation_contract=NEW)
    wrapped = reference.ReferenceNavigationCore(model, stats, navigation_contract=NEW,
        normalization_binding=manifest['normalization_binding'], stats_sha256='a' * 64)
    policy = FastWAMNavigationPolicy(wrapped, model_id='test-double', artifact_sha256='not-weights')
    assert policy.contract == NEW and wrapped.processor[0] is stats and stats['schema'] == OLD


def test_reference_checkpoint_rejects_bad_binding_before_loading_weights(tmp_path):
    from lightx2v_microduck.wam_reference import ReferenceNavigationCore
    stats = tmp_path / 'stats.json'
    stats.write_text(json.dumps(dict(schema=OLD, normalization_split='train')))
    with pytest.raises(ValueError, match='binding'):
        ReferenceNavigationCore.from_checkpoint('missing-model-config', 'missing-weights', stats,
            device='cpu', navigation_contract=NEW)


@pytest.mark.parametrize('history', [False, True])
def test_wrong_normalization_binding_is_rejected_before_importing_model(tmp_path, history):
    from lightx2v_microduck.wam_history_policy import FastWAMNavigationHistoryPolicy
    manifest, config, stats = binding_fixture()
    (tmp_path / 'model.pt').write_bytes(b'test-double-not-model-weights')
    (tmp_path / 'dataset_stats.json').write_text(json.dumps(stats))
    (tmp_path / 'lightx2v_config.json').write_text(json.dumps(config))
    manifest['files'] = {name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
        for name in ('model.pt', 'dataset_stats.json', 'lightx2v_config.json')}
    (tmp_path / 'navigation_model.json').write_text(json.dumps(manifest))
    cls = FastWAMNavigationHistoryPolicy if history else FastWAMNavigationPolicy
    with pytest.raises(ValueError, match='normalization binding'):
        cls.from_artifact(tmp_path, device='cpu')


def test_experimental_execution_support_does_not_admit_teacher_probe_dataset(tmp_path):
    from lightx2v_microduck.wam_data import FastWAMNavigationDataset, build_index
    episode = tmp_path / 'case'
    episode.mkdir()
    (episode / 'episode.json').write_text(json.dumps({
        'action_contract': 'raw_official_guarded_occlusion_teacher_probe_v1',
        'eligible_training_demonstration': False, 'training_release_allowed': False}))
    with pytest.raises(ValueError, match='schema'):
        build_index(tmp_path)
    (tmp_path / 'index.json').write_text(json.dumps(dict(schema=NEW, episodes=[])))
    with pytest.raises(ValueError, match='dataset contract'):
        FastWAMNavigationDataset(tmp_path)


@pytest.mark.integration
def test_extended_command_reaches_official_actor_every_tick_and_is_recorded(tmp_path):
    if not (Path(os.environ.get('MICRODUCK_ASSET_ROOT', '/nonexistent')) / 'microduck_rl').is_dir():
        pytest.skip('pinned official assets required')
    from lightx2v_microduck.env import MicroDuckEnv
    from lightx2v_microduck.navigation_eval import run_navigation_episode

    class Policy:
        metadata = dict(model_id='execution-test-double', contract=NEW)
        def reset(self, episode_id):
            self.episode_id = episode_id
        def predict(self, observation):
            return NavigationChunk(np.tile([.4, 0, 0, 0, 0], (32, 1)),
                observation.episode_id, observation.tick, observation.monotonic_ns,
                observation.tick + 25, contract=NEW)

    with MicroDuckEnv(scene='room_clear', image_size=256) as env:
        original = env.bank.infer
        calls = []
        def infer(role, state):
            calls.append((env.tick, state[48:61].copy()))
            return original(role, state)
        env.bank.infer = infer
        report = run_navigation_episode(env, Policy(), tmp_path / 'episode',
            seed=2300, realtime=False, max_ticks=15, record_predictions=True)
    assert report['action_contract'] == NEW and report['fallback_control_ticks'] == 0
    assert [tick for tick, _ in calls[-15:]] == list(range(15))
    assert all(command[0] == np.float32(.4) for _, command in calls[-15:])
    controls = json.loads((tmp_path / 'episode/control.json').read_text())
    assert len(controls) == 15 and all(c['command'][0] == np.float32(.4) for c in controls)
    predictions = json.loads((tmp_path / 'episode/predictions.json').read_text())
    assert len(predictions) == 3 and all(p['action_contract'] == NEW for p in predictions)
