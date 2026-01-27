from tf_agents.agents.ppo import ppo_agent
from tf_agents.networks import (
    actor_distribution_rnn_network,
    value_rnn_network,
    categorical_projection_network,
    network,
)
from tf_agents.specs import tensor_spec
import tensorflow as tf
import tensorflow_probability as tfp
import functools
from typing import TYPE_CHECKING
from ..utils import get_specs_from
import keras
from ai.config import config

if TYPE_CHECKING:
    from tf_agents.environments.tf_py_environment import TFPyEnvironment


class CategoricalOutputSpec:
    """
    A helper class to act as a distribution spec without instantiating a real distribution.
    This avoids issues with TensorSpecs in newer TensorFlow Probability versions.
    """

    def __init__(self, sample_spec):
        self.sample_spec = sample_spec
        self.dtype = sample_spec.dtype
        num_atoms = sample_spec.maximum - sample_spec.minimum + 1
        self.input_params_spec = {
            "logits": tensor_spec.TensorSpec(
                shape=(num_atoms,), dtype=tf.float32, name="logits"
            )
        }


class Float32CategoricalProjectionNetwork(network.DistributionNetwork):
    """
    A custom projection network that forces float32 logits for numerical stability,
    even when the rest of the model uses mixed precision (float16).
    """

    def __init__(
        self,
        sample_spec,
        logits_init_output_factor=0.1,
        name="Float32CategoricalProjectionNetwork",
    ):
        self._sample_spec = sample_spec
        self._num_atoms = sample_spec.maximum - sample_spec.minimum + 1

        output_spec = CategoricalOutputSpec(sample_spec)

        super().__init__(
            input_tensor_spec=None, state_spec=(), output_spec=output_spec, name=name
        )

        # Force float32 for the logits layer to avoid float16/float32 mismatches
        self._projection_layer = keras.layers.Dense(
            self._num_atoms,
            kernel_initializer=tf.compat.v1.initializers.random_uniform(
                minval=-logits_init_output_factor, maxval=logits_init_output_factor
            ),
            dtype=tf.float32,
            name="logits",
        )

    def call(self, inputs, outer_rank=1, training=False, mask=None):
        logits = self._projection_layer(inputs, training=training)
        distribution = tfp.distributions.Categorical(
            logits=logits, dtype=self._sample_spec.dtype
        )
        return distribution, ()


class PPOAgentFactory:
    """
    Factory class responsible for building the PPO Agent, including the Actor and Value
    RNN networks and their respective preprocessing layers.
    """

    def __init__(
        self,
        train_env: "TFPyEnvironment",
        learning_rate=config.LEARNING_RATE,
        gamma=config.GAMMA,
    ):
        policy = keras.mixed_precision.global_policy()
        compute_dtype = policy.compute_dtype

        local_view_preprocessing = keras.Sequential(
            [
                keras.layers.Flatten(),
                keras.layers.Dense(128, activation="relu"),
            ],
            name="local_view_preprocessing",
        )

        global_state_preprocessing = keras.Sequential(
            [
                keras.layers.LayerNormalization(axis=-1),
                keras.layers.Dense(64, activation="relu"),
            ],
            name="global_state_preprocessing",
        )

        preprocessing_layers = {
            "local_view": local_view_preprocessing,
            "global_state": global_state_preprocessing,
            "replay_json": keras.layers.Lambda(
                # replay_json is for external observers; we zero it out for the agent's input
                lambda x: tf.zeros(shape=(tf.shape(x)[0], 0), dtype=compute_dtype)
            ),
        }

        preprocessing_combiner = keras.layers.Concatenate()
        time_step_spec, action_spec, observation_spec = get_specs_from(train_env)

        input_fc_layer_params = (256, 128)
        lstm_size = (128,)
        output_fc_layer_params = (128,)

        discrete_projection_net = Float32CategoricalProjectionNetwork

        actor_net = actor_distribution_rnn_network.ActorDistributionRnnNetwork(
            observation_spec,
            action_spec,
            preprocessing_layers=preprocessing_layers,
            preprocessing_combiner=preprocessing_combiner,
            input_fc_layer_params=input_fc_layer_params,
            lstm_size=lstm_size,
            output_fc_layer_params=output_fc_layer_params,
            discrete_projection_net=discrete_projection_net,
        )

        value_net = value_rnn_network.ValueRnnNetwork(
            observation_spec,
            preprocessing_layers=preprocessing_layers,
            preprocessing_combiner=preprocessing_combiner,
            input_fc_layer_params=input_fc_layer_params,
            lstm_size=lstm_size,
            output_fc_layer_params=output_fc_layer_params,
        )

        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

        self.agent = ppo_agent.PPOAgent(
            time_step_spec,
            action_spec,
            actor_net=actor_net,
            value_net=value_net,
            optimizer=optimizer,
            normalize_observations=False,
            normalize_rewards=True,
            discount_factor=gamma,
            train_step_counter=tf.Variable(0, dtype=tf.int64),
            entropy_regularization=config.ENTROPY_REGULARIZATION,
            use_gae=True,
            use_td_lambda_return=True,
            num_epochs=5,
            check_numerics=False,
            debug_summaries=False,
        )

        self.agent.initialize()

    def get_agent(self):
        """Returns the initialized PPO agent."""
        return self.agent
