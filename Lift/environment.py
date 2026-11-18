"""Default robosuite Panda Lift, shared by the viewer, conversion and inference."""

import time

import numpy as np
import robosuite as suite
from robosuite import macros

CAMERAS = ("agentview", "robot0_eye_in_hand")
IMAGE_SIZE = 84
STATE_NAMES = [f"joint_{i}" for i in range(1, 8)] + ["finger_1", "finger_2"]
ACTION_NAMES = ["dx", "dy", "dz", "rx", "ry", "rz", "gripper"]


def make_lift(*, viewer=False, images=False, seed=None):
    """Keep the default scene and controller; optionally seed the resets."""
    if images:
        macros.IMAGE_CONVENTION = "opencv"  # RGB with the top row first.
    return suite.make(
        "Lift",
        robots="Panda",
        has_renderer=viewer,
        render_camera=None,
        has_offscreen_renderer=images,
        use_camera_obs=images,
        camera_names=list(CAMERAS),
        camera_heights=IMAGE_SIZE,
        camera_widths=IMAGE_SIZE,
        seed=seed,
    )


def policy_observation(obs):
    """Use the same observation fields for offline data and live inference."""
    return {
        "observation.state": np.concatenate(
            (obs["robot0_joint_pos"], obs["robot0_gripper_qpos"]), dtype=np.float32
        ),
        **{
            f"observation.images.{camera}": np.ascontiguousarray(obs[f"{camera}_image"])
            for camera in CAMERAS
        },
    }


def controller_action(env, world_action):
    """Rotate world-frame deltas into the default Panda controller's base frame."""
    _, base_to_world = env.robots[0].composite_controller.get_controller_base_pose("right")
    action = np.asarray(world_action, dtype=np.float32).reshape(7).copy()
    # Translation and axis-angle use the same rotation; the gripper is independent.
    action[:3] = base_to_world.T @ action[:3]
    action[3:6] = base_to_world.T @ action[3:6]
    lo, hi = env.action_spec
    return np.clip(action, lo, hi)


def main():
    env = make_lift(viewer=True)
    try:
        env.reset()
        env.viewer.update()  # robosuite opens its passive viewer on the first update.
        while env.viewer.viewer.is_running():
            env.viewer.update()
            time.sleep(1 / 30)
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()
