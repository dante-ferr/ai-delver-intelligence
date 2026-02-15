from typing import TYPE_CHECKING
import asyncio

if TYPE_CHECKING:
    import asyncio
    from ai.sessions.session_manager import TrainingSession


def trainer_factory(
    session: "TrainingSession",
    model_bytes: None | bytes = None,
):
    from ai.trainer.trainer import Trainer
    from ai.trainer.dynamic_trainer.dynamic_trainer import DynamicTrainer

    if session.level_transitioning_mode == "dynamic":
        trainer_class = DynamicTrainer
    else:
        trainer_class = Trainer

    trainer = trainer_class(
        session=session,
        model_bytes=model_bytes,
        loop=asyncio.get_running_loop(),
    )
    session.trainer = trainer

    return trainer
