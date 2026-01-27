import logging
import os
import tensorflow as tf
from tensorflow.keras import mixed_precision

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def setup_hardware():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logging.info(f"✅ GPU Memory Growth enabled for: {gpus}")

            policy = mixed_precision.Policy("mixed_float16")
            mixed_precision.set_global_policy(policy)
            logging.info("✅ Mixed Precision (float16) enabled. VRAM usage optimized.")

        except RuntimeError as e:
            logging.error(f"❌ Hardware Setup Failed: {e}")
    else:
        logging.warning("⚠️ No GPU found. Training will run on CPU (SLOW).")


setup_hardware()

from fastapi import FastAPI
from .routes import router

app = FastAPI(title="AI Delver Intelligence API")
app.include_router(router)
