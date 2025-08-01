from fastapi import FastAPI
from api import router

app = FastAPI(title="AI Delver Intelligence API")

app.include_router(router)
