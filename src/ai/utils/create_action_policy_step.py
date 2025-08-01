import tensorflow as tf
from tf_agents.trajectories import policy_step
from typing import Any


def create_action_policy_step(action: dict[str, Any]):
    action_tensors = {
        key: tf.convert_to_tensor(action_element, dtype=tf.float32)
        for key, action_element in action.items()
    }

    action_step = policy_step.PolicyStep(
        action=action_tensors,
        state=(),
        info={
            "dist_params": {
                "move": {
                    "loc": action_tensors["move"],
                    "scale": tf.ones_like(action_tensors["move"]),
                },
                "move_angle_cos": {
                    "loc": action_tensors["move_angle_cos"],
                    "scale": tf.ones_like(action_tensors["move_angle_cos"]),
                },
                "move_angle_sin": {
                    "loc": action_tensors["move_angle_sin"],
                    "scale": tf.ones_like(action_tensors["move_angle_sin"]),
                },
            }
        },
    )
    return action_step
