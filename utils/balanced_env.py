from .base_env import BaseStreetFighterEnv
import math

class BalancedStreetFighterEnv(BaseStreetFighterEnv):
    def __init__(self):
        super().__init__()
        self.previous_enemy_health = 175
        self.previous_enemy_matches_won = 0
        self.previous_score = 0
        self.previous_health = 175
        self.previous_matches_won = 0

    def step(self, action):
        frame_delta, reward, done, info = super().step(action)
        
        # Extract information
        current_health = self.previous_health
        opponent_current_health = self.previous_enemy_health
        matches_won = info['matches_won']
        opponent_matches_won = info['enemy_matches_won']
        distance_x = info['enemy_x_position'] - info['x_position']
        distance_y = info['enemy_y_position'] - info['y_position']
        distance = math.sqrt(distance_x**2 + distance_y**2)
        round_timer = info['round_timer']
        time_since_last_hit = (39208 - self.last_hit_time) / 1000
        current_score = info['score']

        # Agresividad: Distancia óptima y penalización por tiempo sin pegar
        distance_reward = -0.1 * abs(distance - 100)
        time_penalty = -0.5 * time_since_last_hit
        
        aggressiveness_signal = 0.5 * distance_reward + 0.5 * time_penalty

        # Normales: Daño al oponente, daño recibido y puntuación
        opponent_health_reward = 0
        if opponent_current_health < self.previous_enemy_health:
            opponent_health_reward = (self.previous_enemy_health - opponent_current_health)
            self.previous_enemy_health = opponent_current_health
        
        damage_taken_reward = 0
        if current_health < self.previous_health:
            damage_taken_reward = (current_health - self.previous_health)
            self.previous_health = current_health
        
        '''score_reward = 0
        if current_score > self.previous_score:
            score_reward = current_score - self.previous_score
            self.previous_score = current_score'''

        normal_signal = opponent_health_reward + damage_taken_reward #+ 0.2 * score_reward

        # Combine in-game rewards
        reward += aggressiveness_signal + normal_signal

        # End-game rewards
        if done:
            health_reward = 0.5 * (current_health - 88) if current_health >= 88 else -0.5 * (88 - current_health)
            match_win_reward = 0
            if matches_won > self.previous_matches_won:
                match_win_reward = 200
                self.previous_matches_won = matches_won

            end_game_reward = health_reward + match_win_reward
            reward += end_game_reward

        # Update previous values
        self.previous_health = current_health
        self.previous_enemy_matches_won = opponent_matches_won

        return frame_delta, reward, done, info
