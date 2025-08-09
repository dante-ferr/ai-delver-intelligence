import uvicorn
import functools
from api.server import app as api_app
from tf_agents.system import multiprocessing as tf_mp


def run_server():
    """
    Starts the Uvicorn server.
    """
    print("🚀 Starting Uvicorn server...")
    uvicorn.run(api_app, host="0.0.0.0", port=8001, log_level="info")


def main(argv):
    """
    This function is now the main entry point for our application,
    which will be managed by the TF-Agents multiprocessing handler.
    The 'argv' argument is required by the handler but we don't need to use it.
    """
    print("✅ TF-Agents multiprocessing context is being set up by handle_main.")
    run_server()


if __name__ == "__main__":
    tf_mp.handle_main(main)
