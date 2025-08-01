import tensorflow as tf
from tf_agents.policies import tf_policy
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts
import numpy as np
from tf_agents.specs import tensor_spec
import random
import json
from ai.utils import create_action_policy_step

with open("src/ai/utils/config.json", "r") as f:
    config = json.load(f)

ENV_BATCH_SIZE = config["env_batch_size"]
SEED = config["seed"]

random.seed(SEED)

# TODO: study this thing
# I had a good idea: if the Delver's last action was to move, the probability of changing to closer angles is higher than to further ones


class ContinuityRandomPolicy(tf_policy.TFPolicy):
    def __init__(self, time_step_spec, action_spec, env):
        policy_info_spec = {
            "dist_params": {
                "move": {
                    "loc": tensor_spec.TensorSpec(shape=(), dtype=tf.float32),
                    "scale": tensor_spec.TensorSpec(shape=(), dtype=tf.float32),
                },
                "move_angle_sin": {
                    "loc": tensor_spec.TensorSpec(shape=(), dtype=tf.float32),
                    "scale": tensor_spec.TensorSpec(shape=(), dtype=tf.float32),
                },
                "move_angle_cos": {
                    "loc": tensor_spec.TensorSpec(shape=(), dtype=tf.float32),
                    "scale": tensor_spec.TensorSpec(shape=(), dtype=tf.float32),
                },
            }
        }
        super().__init__(
            time_step_spec,
            action_spec,
            policy_state_spec=(),
            info_spec=policy_info_spec,
        )
        self.env = env

    def _action(self, time_step, policy_state=(), seed=None):
        move_values = self._get_random_action_elements(
            [0.0, 1.0], probs=[0.05, 0.15, 0.8], action_name="move"
        )
        move_angle_sines = self._get_random_action_elements(
            [random.uniform(-1.0, 1.0)],
            probs=[0.2, 0.8],
            action_name="move_angle_sin",
        )
        move_angle_cosines = self._get_random_action_elements(
            [random.uniform(-1.0, 1.0)],
            probs=[0.2, 0.8],
            action_name="move_angle_cos",
        )

        actions = {
            "move": move_values,
            "move_angle_sin": move_angle_sines,
            "move_angle_cos": move_angle_cosines,
        }

        for i, env in enumerate(self.env.pyenv.envs):
            env.last_action = {
                action_name: actions[action_name][i] for action_name in actions
            }
        return create_action_policy_step(actions)

    def _get_random_action_elements(
        self,
        possible_action_elements: list[float],
        probs: list[float],
        action_name: str,
    ):
        """Get n random action elements, being n the environment batch size. The last prob is the probability of using the previous action."""
        random_action_elements = []
        for i in range(ENV_BATCH_SIZE):
            pyenv = self.env.pyenv.envs[i]
            possibilities = [*possible_action_elements, pyenv.last_action[action_name]]
            random_action_elements.append(possibilities[self._get_action_index(probs)])

        return random_action_elements

    def _get_action_index(self, probs: list[float]):
        probs_array = np.array(probs)
        normalized_probs = probs / np.sum(probs_array)
        action = np.random.choice(len(probs), p=normalized_probs)
        return action
