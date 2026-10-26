import numpy as np
import pytest
from lightx2v_microduck.contracts import HOME_POLICY, HOME_POSE, MOUTH_INDEX, POLICY_INDICES, JointResidualChunk, Observation, pack_observation, scatter_action
from lightx2v_microduck.execution import ActionBuffer


def test_official_mouth_mapping_and_unscaled_history():
    joints = np.asarray(HOME_POSE).copy()
    joints[MOUTH_INDEX] = 100
    joints[10] += 0.2
    velocities = np.arange(15)
    raw = np.linspace(-1, 1, 14)
    command = np.zeros(13)
    command[9:12] = [0.02, 0.1, 0.2]
    state = pack_observation([1, 2, 3], [0, 0, -1], joints, velocities, raw, command)
    assert state.shape == (61,)
    assert state[15] == pytest.approx(0.2)
    np.testing.assert_array_equal(state[20:34], velocities[list(POLICY_INDICES)])
    np.testing.assert_allclose(state[34:48], raw)
    target = scatter_action(raw, mouth=0.3, scale=0.9)
    assert target[9] == pytest.approx(0.3)
    np.testing.assert_allclose(target[list(POLICY_INDICES)] - HOME_POLICY, raw * 0.9, atol=1e-7)


def test_actor_observation_owns_its_arrays():
    state = np.zeros(61)
    observation = Observation("episode", 0, 0.0, state)
    state[0] = 9
    assert observation.state[0] == 0
    with pytest.raises(ValueError):
        observation.state[0] = 1


def chunk(tick=0, episode="a", rows=50):
    return JointResidualChunk(np.repeat(np.arange(rows)[:, None], 14, axis=1), episode, tick)


def test_late_packet_skips_elapsed_rows_and_expires():
    buffer = ActionBuffer()
    buffer.reset("a")
    assert buffer.submit(chunk(), current_tick=3, now=10.0)
    decision = buffer.consume(3, now=10.01)
    assert decision.action_index == 3
    assert np.all(decision.action == 3)
    assert buffer.consume(10, now=10.1).reason == "expired"
    assert buffer.consume(11, now=10.11).source == "official_fallback"


def test_wrong_episode_and_reordering_never_replace_good_chunk():
    buffer = ActionBuffer()
    buffer.reset("a")
    assert buffer.submit(chunk(5), current_tick=5, now=1.0)
    assert not buffer.submit(chunk(4), current_tick=5, now=1.0)
    assert not buffer.submit(chunk(6, "old"), current_tick=6, now=1.0)
    assert buffer.consume(6, now=1.01).source_tick == 5
    buffer.reset("new")
    assert not buffer.submit(chunk(7), current_tick=7, now=1.0)
    assert buffer.consume(7, now=1.02).action is None


def test_stalled_clock_cannot_extend_action_wall_lifetime():
    buffer = ActionBuffer()
    buffer.reset("a")
    assert buffer.submit(chunk(), current_tick=0, now=1.0)
    assert buffer.consume(0, now=1.21).source == "official_fallback"


def test_bad_actions_and_non_50hz_contract_rejected():
    with pytest.raises(ValueError):
        JointResidualChunk(np.full((5, 14), np.nan), "a", 0)
    with pytest.raises(ValueError):
        JointResidualChunk(np.zeros((5, 14)), "a", 0, control_dt_s=0.1)
