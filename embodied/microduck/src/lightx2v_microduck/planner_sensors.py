"""Name existing actor sensors without introducing simulator pose or task truth."""

import hashlib
import json
import math

from .contracts import HOME_POLICY, POLICY_JOINT_NAMES


NAMED_SENSOR_VERSION = "microduck_named_actor_sensors_v1"
HEAD_JOINTS = ("neck_pitch", "head_pitch", "head_yaw", "head_roll")
NAMED_SENSOR_CONTRACT = {
    "version": NAMED_SENSOR_VERSION,
    "source": "existing_state48_and_fixed_robot_contract",
    "head_joints": {
        name: {"offset_index": 6 + POLICY_JOINT_NAMES.index(name),
               "velocity_index": 20 + POLICY_JOINT_NAMES.index(name),
               "home_rad": HOME_POLICY[POLICY_JOINT_NAMES.index(name)]}
        for name in HEAD_JOINTS
    },
    "camera_optical_pose_computed": False,
    "world_pose_observed": False,
    "target_pose_observed": False,
}
NAMED_SENSOR_CONTRACT_SHA256 = hashlib.sha256(json.dumps(
    NAMED_SENSOR_CONTRACT, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def name_actor_sensors(context):
    """Translate offsets to encoder angles; retain separate image/sensor ticks.

    A head encoder angle is not a global bearing or the complete camera pose.
    An older image must not be labeled as sharing the current encoder sample.
    """
    if (context.get("schema") != "microduck_planner_feedback_v1"
            or context.get("state_layout") != "gyro3,gravity3,joint_offset14,joint_velocity14,previous_raw_action14"):
        raise ValueError("named sensors require the declared official actor layout")
    state = context.get("state48")
    if (not isinstance(state, list) or len(state) != 48
            or any(isinstance(v, bool) or not isinstance(v, (int, float))
                   or not math.isfinite(v) for v in state)):
        raise ValueError("named actor sensors require finite state48")
    tick, camera_tick = context.get("tick"), context.get("camera_tick")
    if (type(tick) is not int or type(camera_tick) is not int
            or not 0 <= camera_tick <= tick):
        raise ValueError("named sensors require ordered image and sensor ticks")
    return {
        "version": NAMED_SENSOR_VERSION,
        "source": NAMED_SENSOR_CONTRACT["source"],
        "contract_sha256": NAMED_SENSOR_CONTRACT_SHA256,
        "sensor_tick": tick,
        "image_tick": camera_tick,
        "image_and_sensor_same_tick": tick == camera_tick,
        "head_joint_angle_rad": {
            name: state[spec["offset_index"]] + spec["home_rad"]
            for name, spec in NAMED_SENSOR_CONTRACT["head_joints"].items()
        },
        "head_joint_velocity_rad_s": {
            name: state[spec["velocity_index"]]
            for name, spec in NAMED_SENSOR_CONTRACT["head_joints"].items()
        },
        "interpretation": "joint_encoder_angles_not_camera_optical_or_world_pose",
    }
