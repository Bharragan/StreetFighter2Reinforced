from .base_env import BaseStreetFighterEnv
import math

class AggressiveStreetFighterEnv(BaseStreetFighterEnv):
    def __init__(self):
        super().__init__()

    def step(self, action):
        frame_delta, reward, done, info = super().step(action)
        distance_x = info['enemy_x_position'] - info['x_position']
        distance_y = info['enemy_y_position'] - info['y_position']
        distance = math.sqrt(distance_x**2 + distance_y**2)
        current_health = self.previous_health
        opponent_current_health = self.opponent_previous_health
        round_timer = info['round_timer']
        time_since_last_hit = (39208 - self.last_hit_time) / 1000

        # Penalización por tiempo sin atacar
        time_penalty = -0.05 * time_since_last_hit
        reward += time_penalty

        # Recompensa por estar cerca del enemigo
        distance_reward = -0.1 * distance
        reward += distance_reward

        # Recompensa por reducir la salud del enemigo
        enemy_health_reward = 0.5 * (176 - opponent_current_health)
        reward += enemy_health_reward

        # Penalización por baja salud
        health_penalty = -0.2 * (176 - current_health)
        reward += health_penalty

        return frame_delta, reward, done, info
