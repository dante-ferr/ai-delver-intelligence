import tensorflow as tf
import logging
import gc
from tf_agents.agents import TFAgent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common
from tf_agents.specs import tensor_spec
from tf_agents.policies import random_tf_policy, tf_policy
from ai.agents import PPOAgentFactory
from ai.config import config
from typing import Optional, Tuple, Any


class TrainerAgentManager:
    """
    Manages the Agent (PPO), Replay Buffer, and Driver.
    Handles the training steps and collection loops.
    """

    def __init__(self, learning_rate: float, gamma: float):
        self.learning_rate = learning_rate
        self.gamma = gamma
        self._agent: Optional[TFAgent] = None
        self._replay_buffer: Optional[
            tf_uniform_replay_buffer.TFUniformReplayBuffer
        ] = None
        self.driver: Optional[dynamic_step_driver.DynamicStepDriver] = None
        self._train_fn: Optional[Any] = None

    def setup(
        self, tf_env, metrics_observers: list, benchmark_mode: bool = False
    ) -> None:
        """Initializes Agent, Buffer, and Driver."""

        self._agent = PPOAgentFactory(
            tf_env, learning_rate=self.learning_rate, gamma=self.gamma
        ).get_agent()
        self._train_fn = common.function(self._agent.train, reduce_retracing=True)
        self._agent.train_step_counter.assign(0)

        if benchmark_mode:
            collect_policy = random_tf_policy.RandomTFPolicy(
                tf_env.time_step_spec(), tf_env.action_spec()
            )
            collect_data_spec = collect_policy.trajectory_spec
        else:
            collect_policy = self._agent.collect_policy
            collect_data_spec = self._agent.collect_data_spec

        self._setup_replay_buffer(collect_data_spec, tf_env.batch_size)

        observers = [self.replay_buffer.add_batch] + metrics_observers
        self.driver = dynamic_step_driver.DynamicStepDriver(
            tf_env,
            collect_policy,
            observers=observers,
            num_steps=config.COLLECT_STEPS_PER_ITERATION,
        )
        self.driver.run = common.function(self.driver.run)

    def _setup_replay_buffer(self, data_spec: Any, batch_size: int) -> None:

        def _selective_cast(spec):
            if hasattr(spec, "dtype") and spec.dtype == tf.float32:
                if not isinstance(spec, tensor_spec.BoundedTensorSpec):
                    return tensor_spec.TensorSpec(
                        shape=spec.shape, dtype=tf.float32, name=spec.name
                    )
            return spec

        # Handle structured specs
        if hasattr(data_spec, "observation"):
            observation_spec = tf.nest.map_structure(
                _selective_cast, data_spec.observation
            )
            data_spec = data_spec._replace(observation=observation_spec)
        else:
            data_spec = tf.nest.map_structure(_selective_cast, data_spec)

        self._replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=data_spec,
            batch_size=batch_size,
            max_length=config.REPLAY_BUFFER_CAPACITY,
        )

    def run_collection_step(self, time_step, policy_state):
        """Runs the driver to collect experience."""
        if not self.driver:
            raise RuntimeError("Driver not initialized")
        return self.driver.run(time_step=time_step, policy_state=policy_state)

    def run_training_step(self) -> Any:
        """Gathers experience and trains the _agent."""
        if not self._replay_buffer or not self._train_fn:
            raise RuntimeError("Agent not initialized")

        experience = self._replay_buffer.gather_all()
        loss_info = self._train_fn(experience)
        return loss_info

    def clear_buffer(self):
        if self._replay_buffer:
            self._replay_buffer.clear()

    def get_policy(self) -> tf_policy.TFPolicy:
        return self.agent.policy

    def get_step_count(self):
        return self.agent.train_step_counter.numpy()

    @property
    def replay_buffer(self) -> tf_uniform_replay_buffer.TFUniformReplayBuffer:
        if not self._replay_buffer:
            raise RuntimeError("Replay Buffer not initialized")
        return self._replay_buffer

    @property
    def agent(self) -> TFAgent:
        if not self._agent:
            raise RuntimeError("Agent not initialized")
        return self._agent
