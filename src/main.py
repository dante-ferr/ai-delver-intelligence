import uvicorn
from api.server import app as api_app
from tf_agents.system import multiprocessing as tf_mp
import logging
import tensorflow as tf
from tensorflow.keras import mixed_precision

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

def run_server():
    """Starts the Uvicorn server."""
    print("🚀 Starting Uvicorn server...")

    uvicorn.run(api_app, host="0.0.0.0", port=8001, log_level="info")


def setup_and_check_hardware():
    """Configures TF hardware and checks availability."""

    gpus = tf.config.list_physical_devices("GPU")

    if not gpus:
        logging.warning("⚠️ NO GPU DETECTED. Running on pure CPU.")
        logging.info(f"Devices found: {tf.config.list_physical_devices()}")
        return

    try:
        # Configure memory growth
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            details = tf.config.experimental.get_device_details(gpu)
            logging.info(
                f"✅ GPU DETECTED: {details.get('device_name', 'Unknown')} ({gpu.name})"
            )

        # Configure mixed precision
        policy = mixed_precision.Policy("mixed_float16")
        mixed_precision.set_global_policy(policy)
        logging.info("✅ Mixed Precision (float16) enabled.")

        # Tensor Core Check
        with tf.device("/device:GPU:0"):
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
            c = tf.matmul(a, b)
            logging.info(f"⚡ Tensor Core Test: Success. Result shape: {c.shape}")

    except RuntimeError as e:
        logging.error(f"❌ Hardware Setup Failed: {e}")


def main(argv):
    """
    Main entry point managed by TF-Agents.
    """
    print("✅ TF-Agents multiprocessing context active.")

    setup_and_check_hardware()

    run_server()

if __name__ == "__main__":
    tf_mp.handle_main(main)
