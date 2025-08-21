from typing import TYPE_CHECKING
import asyncio

if TYPE_CHECKING:
    import asyncio
    from ai.sessions.session_manager import TrainingSession


def trainer_factory(
    session: "TrainingSession",
    model_bytes: None | bytes = None,
):
    from ai.trainer import Trainer

    trainer = Trainer(
        session=session,
        model_bytes=model_bytes,
        loop=asyncio.get_running_loop(),
    )
    session.trainer = trainer

    return trainer
