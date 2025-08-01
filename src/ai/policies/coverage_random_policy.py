import numpy as np
from tf_agents.policies import py_policy
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import policy_step

import random


class CoverageRandomPolicy(py_policy.PyPolicy):
    def __init__(self, time_step_spec, action_spec, environment):
        super().__init__(time_step_spec, action_spec)
        self.env = environment
        self.map_size = self.env.walls_grid.shape
        self.visited_map = np.zeros(self.map_size, dtype=bool)

    def _action(self, time_step: ts.TimeStep):
        agent_pos = self.env.delver_position()  # Should return (x, y)

        reward = self._update_visited_map(agent_pos)

        legal_actions = list(range(self.action_spec.maximum + 1))
        random.shuffle(legal_actions)

        # Prefer actions that lead to unvisited areas (soft bias)
        best_action = self._biased_action(agent_pos, legal_actions)

        return policy_step.PolicyStep(
            action=np.array(best_action, dtype=self.action_spec.dtype)
        )

    def _update_visited_map(self, pos):
        x, y = pos
        count = 0
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.map_width and 0 <= ny < self.map_height:
                    if not self.visited_map[nx][ny]:
                        self.visited_map[nx][ny] = True
                        count += 1
        return count

    def _biased_action(self, pos, legal_actions):
        """
        Apply soft bias toward directions that expand visited coverage.
        """
        action_scores = []
        for action in legal_actions:
            dx, dy = self.env.get_action_direction(action)
            new_x, new_y = pos[0] + dx, pos[1] + dy
            score = self._score_position(new_x, new_y)
            action_scores.append((score, action))

        action_scores.sort(reverse=True)  # Highest score first

        # Randomly choose between top-N options to preserve exploration
        top_choices = [a for _, a in action_scores[:3]]
        return random.choice(top_choices)

    def _score_position(self, x, y):
        if not (0 <= x < self.map_width and 0 <= y < self.map_height):
            return -1  # Penalize out-of-bounds

        # Score based on new tiles in the 3x3 kernel
        score = 0
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.map_width and 0 <= ny < self.map_height:
                    if not self.visited_map[nx][ny]:
                        score += 1
        return score
