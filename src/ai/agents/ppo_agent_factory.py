from tf_agents.agents.ppo import ppo_agent
from tf_agents.networks import (
    actor_distribution_network,
    value_network,
)
import tensorflow as tf
from typing import TYPE_CHECKING
from ..utils import get_specs_from
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
        platforms_spec = train_env.observation_spec()["platforms"]
        platforms_shape = platforms_spec.shape

        platforms_preprocessing = keras.Sequential(
            [
                keras.layers.Reshape((*platforms_shape, 1)),
                keras.layers.Conv2D(16, 3, activation="relu"),
                keras.layers.Conv2D(32, 3, activation="relu"),
                keras.layers.Flatten(),
                keras.layers.Dense(32, activation="relu"),
            ],
            name="platforms_preprocessing",
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
            "platforms": platforms_preprocessing,
            "delver_position": position_preprocessing,
            "goal_position": position_preprocessing,
            "replay_json": keras.layers.Lambda(
                lambda x: tf.zeros(shape=(tf.shape(x)[0], 0))  # type: ignore
            ),
        }

        preprocessing_combiner = keras.layers.Concatenate()
        time_step_spec, action_spec, observation_spec = get_specs_from(train_env)

        fc_layer_params = (128, 64)

        actor_net = actor_distribution_network.ActorDistributionNetwork(
            observation_spec,
            action_spec,
            preprocessing_layers=preprocessing_layers,
            preprocessing_combiner=preprocessing_combiner,
            fc_layer_params=fc_layer_params,
        )
        value_net = value_network.ValueNetwork(
            observation_spec,
            preprocessing_layers=preprocessing_layers,
            preprocessing_combiner=preprocessing_combiner,
            fc_layer_params=fc_layer_params,
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
