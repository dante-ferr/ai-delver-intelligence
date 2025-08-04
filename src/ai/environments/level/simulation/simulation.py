from runtime import Runtime
import json
from os import path
from .delver_action import DelverAction
from ai.config import FPS


DT = 1 / FPS * 3


class Simulation(Runtime):
    def __init__(self, level):
        super().__init__(level, render=False)

        self.elapsed_time = 0.0
        self.delver_actions: list[DelverAction] = []

        self.last_action: None | DelverAction = None

    def step(self, action: DelverAction):
        self.add_delver_action(action)

        if action["move"]:
            self.delver.move(DT, action["move_angle"])

        ended = self.delver.check_collision(self.goal)
        reward = 500.0 if ended else 0

        if self.last_action and self.last_action["move"] and action["move"]:
            angle_diff = (
                action["move_angle"] - self.last_action["move_angle"] + 180
            ) % 360 - 180

            # Penalize for turning. The penalty is proportional to the change in angle.
            # Max penalty is 1.0 for a 180 degree turn.
            reward -= abs(angle_diff) / 180.0

        self.last_action = action
        self.update(DT)

        return reward, ended, self.elapsed_time

    def update(self, dt):
        super().update(dt)

        self.elapsed_time += dt

    def add_delver_action(self, action: DelverAction):
        self.delver_actions.append(DelverAction(**action))
