from .base_env import BaseStreetFighterEnv
import math

class BalancedStreetFighterEnv(BaseStreetFighterEnv):
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

        # Recompensa por mantener una distancia óptima del enemigo
        if opponent_current_health != 0:
            optimal_distance = 100  # Supongamos que la distancia óptima es 100
            distance_reward = -0.01 * abs(distance - optimal_distance)
            reward += distance_reward

        if opponent_current_health == 0:
            health_penalty = 0
            if current_health < 85:
                health_penalty = -0.5 * (85 - current_health)
            else:
                health_penalty = 0.5 * (current_health - 85)
            reward += health_penalty

        # Penalización por tiempo sin pegar un golpe que hizo daño
        if opponent_current_health != 0:
            time_penalty = -0.005 * time_since_last_hit
            reward += time_penalty

        return frame_delta, reward, done, info
