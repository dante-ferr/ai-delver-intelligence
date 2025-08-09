import logging
from fastapi import FastAPI
from .routes import router

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

app = FastAPI(title="AI Delver Intelligence API")

app.include_router(router)
