import uvicorn
from api import app as api_app
import tensorflow as tf


def run_api():
    uvicorn.run(api_app, host="0.0.0.0", port=8001, log_level="info")


if __name__ == "__main__":
    run_api()
