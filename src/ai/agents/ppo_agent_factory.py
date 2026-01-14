from tf_agents.agents.ppo import ppo_agent
from tf_agents.networks import (
    actor_distribution_rnn_network,
    value_rnn_network,
)
import tensorflow as tf
from typing import TYPE_CHECKING
from ..utils import get_specs_from
import keras
from ai.config import config
import logging

if TYPE_CHECKING:
    from tf_agents.environments.tf_py_environment import TFPyEnvironment


def configure_gpu_memory():
    """
    Prevents TensorFlow from allocating ALL VRAM at once, which causes
    system freezes (REISUB) on laptops when running alongside heavy simulations.
    """
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logging.info(f"GPU Memory Growth enabled for: {gpus}")
        except RuntimeError as e:
            logging.error(f"Failed to set GPU memory growth: {e}")
    else:
        logging.warning("No GPU found. Training will run on CPU (SLOW).")


class PPOAgentFactory:

    def __init__(
        self,
        train_env: "TFPyEnvironment",
        learning_rate=config.LEARNING_RATE,
        gamma=config.GAMMA,
    ):
        configure_gpu_memory()

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
                lambda x: tf.zeros(shape=(tf.shape(x)[0], 0))
            ),
        }

        preprocessing_combiner = keras.layers.Concatenate()
        time_step_spec, action_spec, observation_spec = get_specs_from(train_env)

        # Layers before LSTM (Input Processing)
        input_fc_layer_params = (256, 128)

        # LSTM Cell Size (Memory)
        lstm_size = (128,)

        # Layers after LSTM (Decision)
        output_fc_layer_params = (128,)

        actor_net = actor_distribution_rnn_network.ActorDistributionRnnNetwork(
            observation_spec,
            action_spec,
            preprocessing_layers=preprocessing_layers,
            preprocessing_combiner=preprocessing_combiner,
            input_fc_layer_params=input_fc_layer_params,
            lstm_size=lstm_size,
            output_fc_layer_params=output_fc_layer_params,
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
        )

        self.agent.initialize()

    def get_agent(self):
        return self.agent
