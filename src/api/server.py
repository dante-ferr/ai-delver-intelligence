from fastapi import FastAPI
from .routes import router
from .auto_train import handle_auto_train
import os

app = FastAPI(title="AI Delver Intelligence API")


@app.on_event("startup")
async def startup_event():
    """
    This function will be executed automatically when the server starts.
    """
    # Check an environment variable to see if we should start training automatically.
    if os.getenv("AUTO_TRAIN_ON_STARTUP") == "true":
        handle_auto_train()


app.include_router(router)
