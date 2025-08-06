from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from level import Level

class LevelHolder:
    def __init__(self):
        self._level: "Level | None" = None

    @property
    def level(self):
        return self._level

    @level.setter
    def level(self, level):
        self._level = level


level_holder = LevelHolder()
