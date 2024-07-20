from .base_env import BaseStreetFighterEnv
import math

class BalancedStreetFighterEnv(BaseStreetFighterEnv):
    def __init__(self):
        super().__init__()
        self.previous_enemy_health = 176
        self.previous_enemy_matches_won = 0
        self.previous_score = 0
        self.previous_health = 176
        self.previous_matches_won = 0

    def step(self, action):
        frame_delta, reward, done, info = super().step(action)
        reward = 0
        
        # Extract information
        current_health = info['health']
        opponent_current_health = info['enemy_health']
        matches_won = info['matches_won']
        opponent_matches_won = info['enemy_matches_won']
        distance_x = info['enemy_x_position'] - info['x_position']
        distance_y = info['enemy_y_position'] - info['y_position']
        distance = math.sqrt(distance_x**2 + distance_y**2)
        round_timer = info['round_timer']
        time_since_last_hit =  self.last_hit_time
        current_score = info['score']

        # Agresividad: Distancia óptima y penalización por tiempo sin pegar
        distance_reward = -0.1 * abs(distance - 100)
        #print("time_since_last_hit: ", time_since_last_hit)
        time_penalty = -0.001 * time_since_last_hit

        aggressiveness_signal = 0.5 * distance_reward + 0.5 * time_penalty

        # Normales: Daño al oponente, daño recibido y puntuación
        opponent_health_reward = 0
        if opponent_current_health < self.previous_enemy_health:
            opponent_health_reward = (self.previous_enemy_health - opponent_current_health) * 2.5
            #print("Oponent damage-------------------------------------------- ", opponent_health_reward)
            self.previous_enemy_health = opponent_current_health
        
        print("Previous ", self.previous_health, " Current " , current_health)
        damage_taken_reward = 0
        if current_health < self.previous_health:
            damage_taken_reward = (current_health - self.previous_health) * 2.5
            print("USER damage-------------------------------------------- ", damage_taken_reward)

            self.previous_health = current_health
        
        '''score_reward = 0
        if current_score > self.previous_score:
            score_reward = current_score - self.previous_score
            self.previous_score = current_score'''

        normal_signal = opponent_health_reward + damage_taken_reward #+ 0.2 * score_reward

        # Combine in-game rewards
        reward += aggressiveness_signal + normal_signal

        match_reward = 0
        if self.previous_matches_won < matches_won:
            match_reward = 200
        if self.previous_enemy_matches_won < opponent_matches_won:
            match_reward = -200
        reward += match_reward

        # Update previous values
        self.previous_health = current_health
        self.previous_enemy_matches_won = opponent_matches_won
        self.previous_matches_won = matches_won
        self.previous_enemy_health = opponent_current_health

        return frame_delta, reward, done, info
