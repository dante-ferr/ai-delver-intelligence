import numpy as np


class Pathfinder:
    def __init__(self, env):
        self.env = env

    def _get_path_direction(self) -> np.ndarray:
        # Se o caminho não existe ou já terminou, retorne um vetor nulo.
        if not self.astar_path or self.current_waypoint_index >= len(self.astar_path):
            return np.zeros(2, dtype=np.float32)

        agent_pos = np.array(self.simulation.delver.position, dtype=np.float32)
        waypoint_pos = np.array(
            self.astar_path[self.current_waypoint_index], dtype=np.float32
        )

        # Atualiza para o próximo waypoint se o agente estiver perto o suficiente do atual
        if (
            np.linalg.norm(agent_pos - waypoint_pos) < self.WAYPOINT_THRESHOLD
        ):  # Ex: 0.5
            self.current_waypoint_index += 1
            # Checa novamente se o caminho terminou
            if self.current_waypoint_index >= len(self.astar_path):
                return np.zeros(2, dtype=np.float32)
            # Atualiza para o novo waypoint
            waypoint_pos = np.array(
                self.astar_path[self.current_waypoint_index], dtype=np.float32
            )

        # Calcula o vetor de direção e o normaliza (vetor unitário)
        direction_vector = waypoint_pos - agent_pos
        norm = np.linalg.norm(direction_vector)
        if norm > 0:
            return (direction_vector / norm).astype(np.float32)
        else:
            return np.zeros(2, dtype=np.float32)
