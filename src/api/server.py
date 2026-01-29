from fastapi import FastAPI
from .routes import router

app = FastAPI(title="AI Delver Intelligence API")
app.include_router(router)
