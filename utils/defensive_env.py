from .base_env import BaseStreetFighterEnv

class DefensiveStreetFighterEnv(BaseStreetFighterEnv):
    def __init__(self):
        super().__init__()

    def step(self, action):
        frame_delta, reward, done, info = super().step(action)
        current_health = self.previous_health

        # Penalización por baja salud
        health_penalty = -0.5 * (176 - current_health)
        reward += health_penalty

        return frame_delta, reward, done, info
