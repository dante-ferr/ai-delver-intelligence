from ai import TrainerController

if __name__ == "__main__":
    from level_holder.level_loader import level_loader
    from level_holder.level_holder import level_holder

    level_loader.load_level("data/level_saves/test_1.dill")

    level_holder.level = level_loader.level
    TrainerController().train()
