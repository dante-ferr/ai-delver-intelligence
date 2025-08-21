from tf_agents.policies import policy_saver, actor_policy
import tempfile
import shutil
import os
import logging
import tensorflow as tf
from tf_agents.networks import network
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

        Returns:
            None | bytes:   A byte string representing the zipped model policy,
                            or None if an error occurs.
        """
        try:
            # Create a PolicySaver instance to handle the saving process.
            saver = policy_saver.PolicySaver(self.trainer.agent.policy)

            # Use a temporary directory that will be automatically cleaned up.
            with tempfile.TemporaryDirectory() as temp_dir:
                policy_dir = os.path.join(temp_dir, "policy")

                # Save the policy to the temporary directory.
                saver.save(policy_dir)

                # Archive the contents of the policy directory into a zip file.
                archive_base_name = os.path.join(temp_dir, "model_archive")
                archive_path = shutil.make_archive(
                    base_name=archive_base_name, format="zip", root_dir=policy_dir
                )

                # Read the generated zip file into memory as bytes.
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
        with tempfile.TemporaryDirectory() as temp_dir:
            # Write the model bytes to a temporary zip file.
            zip_path = os.path.join(temp_dir, "model.zip")
            with open(zip_path, "wb") as f:
                f.write(model_bytes)

            # Unpack the archive into a dedicated policy directory.
            policy_dir = os.path.join(temp_dir, "policy")
            shutil.unpack_archive(zip_path, policy_dir)

            loaded_policy = tf.saved_model.load(policy_dir)

            # The loaded_policy object is an AutoTrackable, but at runtime it contains
            # the model's learned parameters in its .variables attribute. Pylance
            # cannot infer this specific structure, so we use # type: ignore to
            # suppress the false-positive error.
            loaded_weights = [v.numpy() for v in loaded_policy.variables]  # type: ignore

            # For an agent like PPO, the policy is an ActorPolicy, which is known
            # to have a .network attribute. We cast the general TFPolicy to this
            # specific type to resolve the linter error.
            policy = cast(actor_policy.ActorPolicy, self.trainer.agent.policy)

            # The network attribute of the policy is the underlying Keras model.
            policy.network.set_weights(loaded_weights)  # type: ignore

            logging.info(
                "Successfully loaded weights from the provided model into the agent's policy network."
            )
