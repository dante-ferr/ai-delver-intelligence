from fastapi import APIRouter, HTTPException
from ai import TrainerController
from level_holder import level_holder
from pydantic import BaseModel
import base64, dill


class TrainPayload(BaseModel):
    level: str


router = APIRouter()


@router.post("/train")
def train_agent(payload: TrainPayload):
    try:
        raw_bytes = base64.b64decode(payload.level)
        level_obj = dill.loads(raw_bytes)
    except Exception:
        raise Exception
        raise HTTPException(status_code=400, detail="Invalid level data")

    level_holder.level = level_obj
    TrainerController().train()

    return {"message": "Training step complete."}
