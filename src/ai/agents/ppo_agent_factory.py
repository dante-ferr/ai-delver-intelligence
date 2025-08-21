from tf_agents.agents.ppo import ppo_agent
from tf_agents.networks import (
    actor_distribution_network,
    value_network,
)
from tf_agents.networks import encoding_network
import tensorflow as tf
from typing import TYPE_CHECKING
from ..utils import get_specs_from
import json
import keras
from ai.config import config

if TYPE_CHECKING:
    from tf_agents.environments.tf_py_environment import TFPyEnvironment

def check_gpu_available():
    if tf.test.is_gpu_available():
        print("GPU is available and being used for training.")
    else:
        print("WARNING: GPU is NOT available. Training will be slow.")


class PPOAgentFactory:

    def __init__(
        self,
        train_env: "TFPyEnvironment",
        learning_rate=config.LEARNING_RATE,
        gamma=config.GAMMA,
    ):
        walls_spec = train_env.observation_spec()["walls"]
        walls_shape = walls_spec.shape

        walls_preprocessing = keras.Sequential(
            [
                keras.layers.Rescaling(1.0),
                keras.layers.Reshape((*walls_shape, 1)),
                keras.layers.Conv2D(16, 3, activation="relu"),
                keras.layers.Flatten(),
                keras.layers.Dense(32, activation="relu"),
            ]
        )
        position_preprocessing = keras.Sequential(
            [
                keras.layers.LayerNormalization(axis=-1),
                keras.layers.Dense(
                    32, activation="relu", name="position_preprocessing"
                ),
            ],
            name="position_preprocessing",
        )
        preprocessing_layers = {
            "walls": walls_preprocessing,
            "delver_position": position_preprocessing,
            "goal_position": position_preprocessing,
            "replay_json": keras.layers.Lambda(
                lambda x: tf.zeros(shape=(tf.shape(x)[0], 0))  # type: ignore
            ),
        }

        preprocessing_combiner = keras.layers.Concatenate()
        time_step_spec, action_spec, observation_spec = get_specs_from(train_env)

        actor_net = actor_distribution_network.ActorDistributionNetwork(
            observation_spec,
            action_spec,
            preprocessing_layers=preprocessing_layers,
            preprocessing_combiner=preprocessing_combiner,
            fc_layer_params=(32, 16),
        )
        value_net = value_network.ValueNetwork(
            observation_spec,
            preprocessing_layers=preprocessing_layers,
            preprocessing_combiner=preprocessing_combiner,
            fc_layer_params=(32, 16),
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
        )

        self.agent.initialize()
        check_gpu_available()

    def get_agent(self):
        return self.agent
