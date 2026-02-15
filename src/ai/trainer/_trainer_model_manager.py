from tf_agents.policies import policy_saver, actor_policy
import tempfile
import shutil
import os
import logging
import tensorflow as tf
from tensorflow.python.trackable import autotrackable
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from .trainer import Trainer


class TrainerModelManager:
    def __init__(self, trainer: "Trainer"):
        self.trainer = trainer

    def get_serialized_model(self) -> None | bytes:
        """
        Saves the agent's policy to a temporary directory, archives it as a zip file,
        and returns the resulting bytes.
        """
        try:
            agent = self.trainer.agent_manager.agent
            if not agent:
                logging.error("Cannot serialize model: Agent not initialized.")
                return None

            saver = policy_saver.PolicySaver(agent.policy)

            with tempfile.TemporaryDirectory() as temp_dir:
                policy_dir = os.path.join(temp_dir, "policy")

                saver.save(policy_dir)

                archive_base_name = os.path.join(temp_dir, "model_archive")
                archive_path = shutil.make_archive(
                    base_name=archive_base_name, format="zip", root_dir=policy_dir
                )

                with open(archive_path, "rb") as f:
                    model_bytes = f.read()

                logging.info(
                    f"Successfully serialized model policy to {len(model_bytes)} bytes."
                )
                return model_bytes
        except Exception as e:
            logging.error(f"Failed to serialize model: {e}")
            return None

    def load_serialized_model(self, model_bytes: bytes) -> None:
        try:
            agent = self.trainer.agent_manager.agent
            if not agent:
                logging.error("Cannot load model: Agent not initialized.")
                return

            with tempfile.TemporaryDirectory() as temp_dir:
                zip_path = os.path.join(temp_dir, "model.zip")
                with open(zip_path, "wb") as f:
                    f.write(model_bytes)

                policy_dir = os.path.join(temp_dir, "policy")
                shutil.unpack_archive(zip_path, policy_dir)

                loaded_policy = tf.saved_model.load(policy_dir)

                if not isinstance(loaded_policy, autotrackable.AutoTrackable):
                    raise ValueError(
                        "Loaded policy is not an instance of AutoTrackable."
                    )

                # type: ignore - Pylance cannot infer .variables on AutoTrackable
                loaded_weights = [v.numpy() for v in loaded_policy.variables]

                policy = cast(actor_policy.ActorPolicy, agent.policy)

                # type: ignore - .network is part of ActorPolicy
                policy.network.set_weights(loaded_weights)

                logging.info(
                    "Successfully loaded weights from the provided model into the agent's policy network."
                )
        except Exception as e:
            logging.error(f"Failed to load serialized model: {e}")
