import numpy as np
from typing import TYPE_CHECKING
from tf_agents.trajectories import trajectory, policy_step
import tensorflow as tf
from tf_agents.trajectories import time_step as ts
from ai.utils import create_action_policy_step

if TYPE_CHECKING:
    from tf_agents.typing.types import NestedArraySpec


class RingReplayBuffer:
    def __init__(
        self,
        capacity,
        observation_spec: "NestedArraySpec",
        action_spec: "NestedArraySpec",
    ):
        self.capacity = capacity
        self.observation_spec = observation_spec
        self.action_spec = action_spec

        # Convert TF dtypes to numpy dtypes
        def to_numpy_dtype(tf_dtype):
            return (
                tf_dtype.as_numpy_dtype
                if hasattr(tf_dtype, "as_numpy_dtype")
                else tf_dtype
            )

        self.observations = {
            key: np.zeros((capacity, *spec.shape), dtype=to_numpy_dtype(spec.dtype))
            for key, spec in observation_spec.items()
        }

        self.actions = {
            key: np.zeros((capacity, *spec.shape), dtype=to_numpy_dtype(spec.dtype))
            for key, spec in action_spec.items()
        }

        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=bool)
        self.next_observations = {
            key: np.zeros((capacity, *spec.shape), dtype=to_numpy_dtype(spec.dtype))
            for key, spec in observation_spec.items()
        }

        self.index = 0
        self.full = False

    def add(self, obs, action, reward, next_obs, done):
        for key in self.observation_spec.keys():
            self.observations[key][self.index] = obs[key]
            self.next_observations[key][self.index] = next_obs[key]

        for key in self.action_spec.keys():
            self.actions[key][self.index] = action[key]

        self.rewards[self.index] = reward
        self.dones[self.index] = done

        self.index = (self.index + 1) % self.capacity
        if self.index == 0:
            self.full = True

    def sample(self, batch_size):
        max_index = self.capacity if self.full else self.index

        # Handle case where buffer doesn't have enough samples yet
        if batch_size > max_index:
            indices = np.random.choice(max_index, size=batch_size, replace=True)
        else:
            indices = np.random.choice(max_index, size=batch_size, replace=False)

        time_step = ts.TimeStep(
            step_type=tf.fill([batch_size], ts.StepType.FIRST),
            reward=tf.convert_to_tensor(self.rewards[indices]),
            discount=tf.convert_to_tensor(1.0 - self.dones[indices].astype(np.float32)),
            observation=self._get_sampled_obs(indices),
        )

        next_time_step = ts.TimeStep(
            step_type=tf.where(self.dones[indices], ts.StepType.LAST, ts.StepType.MID),
            reward=tf.zeros_like(self.rewards[indices]),
            discount=tf.convert_to_tensor(1.0 - self.dones[indices].astype(np.float32)),
            observation={
                "walls": tf.convert_to_tensor(self.next_observations["walls"][indices]),
                "delver_position": tf.convert_to_tensor(
                    self.next_observations["delver_position"][indices]
                ),
                "goal_position": tf.convert_to_tensor(
                    self.next_observations["goal_position"][indices]
                ),
            },
        )

        sampled_actions = {
            key: action_element[indices] for key, action_element in self.actions.items()
        }
        action_step = create_action_policy_step(sampled_actions)

        return trajectory.Trajectory(
            time_step.step_type,
            time_step.observation,
            action_step.action,
            action_step.info,
            next_time_step.step_type,
            next_time_step.reward,
            next_time_step.discount,
        )

    def _get_sampled_obs(self, indices):
        return {
            "walls": tf.convert_to_tensor(self.observations["walls"][indices]),
            "delver_position": tf.convert_to_tensor(
                self.observations["delver_position"][indices]
            ),
            "goal_position": tf.convert_to_tensor(
                self.observations["goal_position"][indices]
            ),
        }

    def __len__(self):
        return self.capacity if self.full else self.index
