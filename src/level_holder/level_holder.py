class LevelHolder:
    def __init__(self):
        self._level = None

    @property
    def level(self):
        return self._level

    @level.setter
    def level(self, level):
        self._level = level


level_holder = LevelHolder()
