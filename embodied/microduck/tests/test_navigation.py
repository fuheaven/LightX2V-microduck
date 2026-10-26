import time

import numpy as np
import pytest

from lightx2v_microduck.contracts import Observation
from lightx2v_microduck.navigation import NavigationBuffer, NavigationChunk, intent_from_action, student_state


def obs(tick=0, episode="episode", ns=None):
    return Observation(episode, tick, tick * .02, np.arange(61, dtype=np.float32),
                       monotonic_ns=1_000_000_000 + tick * 20_000_000 if ns is None else ns)


def plan(source=0, episode="episode", count=10):
    actions = np.zeros((count, 5), np.float32)
    actions[:, 0] = np.arange(count) * .01
    return NavigationChunk(actions, episode, source, 1_000_000_000 + source * 20_000_000,
                           source + count * 5)


def test_late_reply_skips_elapsed_decisions_and_expires_without_holding_velocity():
    buffer = NavigationBuffer()
    buffer.reset("episode")
    assert buffer.submit(plan(), obs(11))
    intent, evidence = buffer.consume(obs(11))
    assert evidence["action_index"] == 2
    assert intent.command[0] == pytest.approx(.02)
    assert intent.duration_ticks == 1
    intent, evidence = buffer.consume(obs(25))
    assert intent.skill == "stand" and not intent.command.any()
    assert evidence["reason"] == "expired"


def test_old_episode_out_of_order_and_wall_time_expiry_are_rejected():
    buffer = NavigationBuffer()
    buffer.reset("episode")
    assert not buffer.submit(plan(episode="other"), obs())
    assert buffer.submit(plan(5), obs(5))
    assert not buffer.submit(plan(0), obs(6))
    assert not buffer.submit(plan(5), obs(6))
    intent, evidence = buffer.consume(obs(6, ns=1_650_000_000))
    assert intent.skill == "stand" and evidence["reason"] == "expired"
    buffer.reset("new")
    assert not buffer.submit(plan(10), obs(10, episode="new"))
    assert buffer.consume(obs(10, episode="new"))[0].skill == "stand"


def test_student_excludes_teacher_commands_and_stop_overrides_translation():
    observation = obs()
    assert np.array_equal(student_state(observation), np.arange(48))
    intent = intent_from_action([.3, .1, 1., .2, 1.], episode_id="episode", tick=0)
    assert intent.skill == "stand" and not intent.command[:3].any()
    assert intent.command[5] == pytest.approx(.2)
    for invalid in ([1, 0, 0, 0, 0], [0, 0, float("nan"), 0, 0]):
        with pytest.raises(ValueError):
            intent_from_action(invalid, episode_id="episode", tick=0)


@pytest.mark.integration
def test_executor_recomputes_official_actor_each_control_tick():
    import os
    from pathlib import Path
    from lightx2v_microduck.env import MicroDuckEnv

    if not (Path(os.environ.get("MICRODUCK_ASSET_ROOT", "/nonexistent")) / "microduck_rl").is_dir():
        pytest.skip("pinned official assets required")
    with MicroDuckEnv(scene="room_clear", render=False) as env:
        observation = env.reset(0, "find_red_ball")
        original = env.bank.infer
        ticks, commands = [], []

        def infer(role, state):
            ticks.append(env.tick)
            commands.append(state[48:61].copy())
            return original(role, state)

        env.bank.infer = infer
        env.step(intent_from_action([.3, 0, 0, .2, 0], episode_id=observation.episode_id, tick=0, duration_ticks=5))
        assert ticks == [0, 1, 2, 3, 4]
        assert all(c[0] == pytest.approx(.3) and c[5] == pytest.approx(.2) for c in commands)
        assert not env.observe().state[48:61].any()


def test_fastwam_adapter_only_passes_sensors_and_language():
    from dataclasses import replace
    from lightx2v_microduck.wam_policy import FastWAMNavigationPolicy

    class Core:
        policy_profile, action_dim, robot_state_dim, camera_size = "microduck_navigation", 5, 48, 256
        gripper_postprocess = False

        def reset(self):
            pass

        def predict_action_chunk(self, images, state, task_description):
            assert set(images) == {"head_camera"}
            assert state.shape == (48,) and task_description == "Find the blue ball"
            return np.tile([.4, 0, 0, 0, 0], (32, 1))

    policy = FastWAMNavigationPolicy(Core(), model_id="test-double", artifact_sha256="test")
    policy.reset("episode")
    observation = replace(obs(), rgb=np.zeros((256, 256, 3), np.uint8), camera_tick=0,
                          instruction="Find the blue ball")
    chunk = policy.predict(observation)
    assert chunk.actions.shape == (32, 5) and chunk.diagnostics["clipped_values"] == 32
    assert chunk.valid_until_tick == 25
    with pytest.raises(ValueError, match="fresh"):
        policy.predict(replace(observation, camera_tick=1))
    with pytest.raises(ValueError, match="reset"):
        policy.predict(replace(observation, episode_id="other"))


@pytest.mark.parametrize("teacher_kind", ["basic", "obstacle_aware", "initial_scan", "local_views", "latched_view", "pivot_view", "camera_frustum", "occluded_pockets", "opportunity_weighted"])
def test_search_decision_is_independent_of_hidden_target_coordinates(teacher_kind):
    from dataclasses import replace
    from types import SimpleNamespace
    from lightx2v_microduck.navigation import VisualSearchTeacher

    observation = replace(obs(), rgb=np.zeros((256, 256, 3), np.uint8), camera_tick=0)
    actions = []
    for target in ([.7, .3, .035], [-.9, -.8, .035]):
        env = SimpleNamespace(episode_id="episode", tick=0, profile="official_bam",
                              task=SimpleNamespace(target_color="red"),
                              evaluator_state=lambda: {"balls": {"red": target}, "obstacles": [],
                                  "room_bounds": [-1.5, 1.5, -1.5, 1.5],
                                  "robot_position": [0., 0., .12], "robot_yaw": 0.},
                              physics=SimpleNamespace(camera=0,
                                  data=SimpleNamespace(cam_xpos=np.array([[0., 0., .2]]),
                                      cam_xmat=np.array([[0, 0, -1, -1, 0, 0, 0, 1, 0]])),
                                  model=SimpleNamespace(cam_fovy=[80])))
        if teacher_kind == "initial_scan":
            from lightx2v_microduck.navigation_teacher_experiments import InitialScanVisualSearchTeacher

            teacher = InitialScanVisualSearchTeacher(env)
        elif teacher_kind == "occluded_pockets":
            from lightx2v_microduck.navigation_teacher_experiments import OccludedPocketVisualSearchTeacher

            teacher = OccludedPocketVisualSearchTeacher(env)
        elif teacher_kind == "local_views":
            from lightx2v_microduck.navigation_teacher_experiments import LocalViewVisualSearchTeacher

            teacher = LocalViewVisualSearchTeacher(env)
        elif teacher_kind == "latched_view":
            from lightx2v_microduck.navigation_teacher_experiments import LatchedViewVisualSearchTeacher

            teacher = LatchedViewVisualSearchTeacher(env)
        elif teacher_kind == "pivot_view":
            from lightx2v_microduck.navigation_teacher_experiments import PivotViewVisualSearchTeacher

            teacher = PivotViewVisualSearchTeacher(env)
        elif teacher_kind == "camera_frustum":
            from lightx2v_microduck.navigation_teacher_experiments import CameraFrustumVisualSearchTeacher

            teacher = CameraFrustumVisualSearchTeacher(env)
        elif teacher_kind == "opportunity_weighted":
            from lightx2v_microduck.navigation_teacher_experiments import OpportunityWeightedVisualSearchTeacher

            teacher = OpportunityWeightedVisualSearchTeacher(env)
        else:
            teacher = VisualSearchTeacher(env, obstacle_aware=teacher_kind == "obstacle_aware")
        teacher._visible_pixels = lambda color: 0
        actions.append(teacher.predict(observation))
        assert teacher.last_seen_position is None
        assert teacher.last_diagnostics["hidden_target_coordinates_used"] is False
    np.testing.assert_array_equal(*actions)


def test_arrived_search_finishes_orientation_despite_small_walking_drift():
    from types import SimpleNamespace
    from lightx2v_microduck.navigation_teacher_experiments import LatchedViewCoverage

    world = {"obstacles": [], "room_bounds": [-1.5, 1.5, -1.5, 1.5],
             "robot_position": [.1, 0., .12], "robot_yaw": 0.}
    coverage = LatchedViewCoverage(world)
    coverage.destination = np.array([0., 0.])
    coverage.look_yaw = np.pi / 2
    coverage.chosen_tick = 0
    physics = SimpleNamespace(camera=0,
                              data=SimpleNamespace(cam_xpos=np.array([[0., 0., .2]]),
                                                   cam_xmat=np.array([[0, 0, -1, -1, 0, 0, 0, 1, 0]])),
                              model=SimpleNamespace(cam_fovy=[45]))
    coverage.command(world, physics, 0)
    world.update(robot_position=[.25, 0, .12], robot_yaw=1.2)
    action, diagnostic = coverage.command(world, physics, 5)
    assert diagnostic["viewpoint_orientation_latched"]
    assert action[2] == pytest.approx(2 * (np.pi / 2 - 1.2))
    # A large excursion must return to navigation, rather than retaining a
    # stale arrival state indefinitely.
    world["robot_position"] = [.6, 0, .12]
    _, diagnostic = coverage.command(world, physics, 10)
    assert not diagnostic["viewpoint_orientation_latched"]


def test_obstacle_teacher_remembers_visible_distractor_without_tracking_it_through_truth():
    from dataclasses import replace
    from types import SimpleNamespace
    from lightx2v_microduck.navigation import VisualSearchTeacher

    world = {"balls": {"red": [-.9, -.8, .035], "blue": [.8, .4, .035]}, "obstacles": [],
             "room_bounds": [-1.5, 1.5, -1.5, 1.5], "robot_position": [0., 0., .12], "robot_yaw": 0.}
    env = SimpleNamespace(episode_id="episode", tick=0, profile="official_bam",
                          task=SimpleNamespace(target_color="red"), evaluator_state=lambda: dict(world),
                          physics=SimpleNamespace(camera=0,
                              data=SimpleNamespace(cam_xpos=np.array([[0., 0., .2]]),
                                  cam_xmat=np.array([[0, 0, -1, -1, 0, 0, 0, 1, 0]])),
                              model=SimpleNamespace(cam_fovy=[45])))
    teacher = VisualSearchTeacher(env, obstacle_aware=True)
    teacher._visible_pixels = lambda color: 20 if color == "blue" else 0
    teacher.predict(replace(obs(), rgb=np.zeros((256, 256, 3), np.uint8), camera_tick=0))
    np.testing.assert_array_equal(teacher.observed_balls["blue"], [.8, .4, .035])
    assert teacher.last_seen_position is None
    # Move both hidden balls in the simulator. Neither movement is available
    # to the teacher until the actual camera observes that ball again.
    world["balls"] = {"red": [1., 1., .035], "blue": [-1., -.5, .035]}
    env.tick = 5
    teacher._visible_pixels = lambda color: 0
    teacher.predict(replace(obs(5), rgb=np.zeros((256, 256, 3), np.uint8), camera_tick=5))
    remembered = [o for o in teacher.coverage.obstacles if o["name"] == "observed_ball_blue"]
    assert len(remembered) == 1 and remembered[0]["xy"] == [.8, .4]
    assert all(o["name"] != "observed_ball_red" for o in teacher.coverage.obstacles)
    assert teacher.last_seen_position is None


def test_search_visibility_does_not_see_through_boxes_or_behind_camera():
    from lightx2v_microduck.navigation_search import SearchCoverage, clear_sight

    obstacles = [{"xy": [.5, 0], "half_size": [.1, .1], "height": .3}]
    assert clear_sight([0, 0, .2], [[1, 0, .035], [1, .8, .035]], obstacles).tolist() == [False, True]
    coverage = SearchCoverage({"obstacles": obstacles, "room_bounds": [-1.5, 1.5, -1.5, 1.5]})
    coverage.cells = np.array([[1., 0, .035], [1., .5, .035], [-1., 0, .035]])
    coverage.seen = np.zeros(3, dtype=bool)
    rotation = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
    coverage.update_camera([0, 0, .2], rotation, 80)
    assert coverage.seen.tolist() == [False, True, False]


def test_teacher_stays_stopped_through_small_balance_motion_but_reapproaches_outside_band():
    from types import SimpleNamespace
    from lightx2v_microduck.navigation import NavigationTeacher

    env = SimpleNamespace(profile="official_bam", task=SimpleNamespace(target_color="red", stop_distance_m=(.25, .40)))
    teacher = NavigationTeacher(env)
    world = {"robot_position": [0., 0., .12], "robot_yaw": 0., "obstacles": [],
             "room_bounds": [-1.5, 1.5, -1.5, 1.5], "recovery_active": False}
    # Enter at a safe stopping distance. Ordinary balancing then crosses the
    # entry threshold while staying within the task's unchanged stop band.
    for center_distance in (.39, .40, .42):
        world["balls"] = {"red": [center_distance, 0., .035]}
        assert teacher._predict_world(obs(), world)[4] == 1
    world["balls"] = {"red": [.47, 0., .035]}
    assert teacher._predict_world(obs(), world)[4] == 0


@pytest.mark.integration
def test_delayed_navigation_inference_does_not_pause_mujoco(tmp_path):
    import os
    from pathlib import Path
    from lightx2v_microduck.env import MicroDuckEnv
    from lightx2v_microduck.navigation_eval import run_navigation_episode

    if not (Path(os.environ.get("MICRODUCK_ASSET_ROOT", "/nonexistent")) / "microduck_rl").is_dir():
        pytest.skip("pinned official assets required")

    class DelayedPolicy:
        def reset(self, episode_id):
            self.episode_id = episode_id

        def predict(self, observation):
            time.sleep(.55)
            return NavigationChunk(np.tile([0, 0, 0, 0, 1], (32, 1)), self.episode_id,
                                   observation.tick, observation.monotonic_ns, observation.tick + 25)

    with MicroDuckEnv(scene="room_clear", render=False) as env:
        # This check isolates clock/worker behavior from OpenGL driver setup.
        env.render = lambda: np.zeros((256, 256, 3), np.uint8)
        report = run_navigation_episode(env, DelayedPolicy(), tmp_path / "slow", max_ticks=40,
                                        warmup_ticks=0, initialize_plan=False)
    assert report["sim_s"] == .8
    assert .79 <= report["wall_s"] < 1.2
    assert report["rejected_plans"] >= 1 and report["fallback_control_ticks"] == 40
    assert not report["strict_model_success"]


@pytest.mark.integration
def test_kernel_warmup_discards_old_plan_and_records_actual_camera_ticks(tmp_path):
    import os
    import threading
    from pathlib import Path
    from lightx2v_microduck.env import MicroDuckEnv
    from lightx2v_microduck.navigation_eval import run_navigation_episode

    if not (Path(os.environ.get("MICRODUCK_ASSET_ROOT", "/nonexistent")) / "microduck_rl").is_dir():
        pytest.skip("pinned official assets required")

    class ColdPolicy:
        def __init__(self):
            self.calls, self.resets = [], 0
            self.threads = []

        def reset(self, episode_id):
            self.resets += 1

        def predict(self, observation):
            self.calls.append(observation)
            identity = threading.get_ident()
            cold = identity not in self.threads
            self.threads.append(identity)
            if cold:
                time.sleep(.52)
            return NavigationChunk(np.tile([0, 0, 0, 0, 1], (32, 1)), observation.episode_id,
                                   observation.tick, observation.monotonic_ns, observation.tick + 25)

    policy = ColdPolicy()
    with MicroDuckEnv(scene="room_clear", render=False) as env:
        env.render = lambda: np.zeros((256, 256, 3), np.uint8)
        report = run_navigation_episode(env, policy, tmp_path / "warm", max_ticks=10,
                                        warmup_ticks=0, warmup_predictions=1, record_frames=True)
    assert policy.resets == 3
    assert policy.calls[1].monotonic_ns > policy.calls[0].monotonic_ns + 500_000_000
    assert policy.calls[2].monotonic_ns > policy.calls[1].monotonic_ns + 500_000_000
    assert policy.threads[1] == policy.threads[-1] != policy.threads[0]
    assert report["fallback_control_ticks"] == 0
    assert report["warmup_inference_ms"][0] >= 500
    assert report["worker_warmup_inference_ms"][0] >= 500
    with np.load(tmp_path / "warm" / "recording.npz") as data:
        assert data["camera_ticks"].tolist() == [0, 5]
        assert data["physics_ticks"].tolist() == list(range(10))
