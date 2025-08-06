from ai import Trainer
import threading
from .routes import manager


def auto_train(_):
    from level_holder.level_loader import level_loader
    from level_holder.level_holder import level_holder

    level_loader.load_level("data/level_saves/test_1.dill")
    level_holder.level = level_loader.level

    trainer = Trainer()
    trainer.train()


def handle_auto_train():
    print(
        "🚀 AUTO_TRAIN_ON_STARTUP is true. Internally sending a training request on the 'test_1.dill' level..."
    )
    # Run the training task in a separate thread so it doesn't block the server.
    # The 'manager' is the connection manager for WebSockets.

    thread = threading.Thread(target=auto_train, args=(manager,), daemon=True)
    thread.start()
