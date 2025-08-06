from fastapi import APIRouter, HTTPException
from ai import Trainer
from level_holder import level_holder
from pydantic import BaseModel
import base64, dill
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from .connection_manager import manager  # Import the manager

class TrainPayload(BaseModel):
    level: str


router = APIRouter()


@router.post("/train")
def train_agent(payload: TrainPayload):
    try:
        raw_bytes = base64.b64decode(payload.level)
        level_obj = dill.loads(raw_bytes)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid level data")

    level_holder.level = level_obj
    Trainer().train()

    return {"message": "Training step complete."}


@router.websocket("/ws/replays")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect()
