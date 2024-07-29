from .base_env import BaseStreetFighterEnv
import math
import pandas as pd
import os

class BalancedStreetFighterEnv(BaseStreetFighterEnv):
    def __init__(self, save_video=False, state='guile'):
        super().__init__(save_video=save_video, state=state)
        self.previous_enemy_health = 176
        self.previous_enemy_matches_won = 0
        self.previous_score = 0
        self.previous_health = 176
        self.previous_matches_won = 0
        self.rewards_data = []  # List to store rewards information
        self.save_data_flag = False  # Flag to control when to save the data

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
        time_since_last_hit = self.last_hit_time
        current_score = info['score']

        # Agresividad: Distancia óptima y penalización por tiempo sin pegar
        distance_reward = 0
        if distance <= 100:
            distance_reward = 0.004 * (distance)  # Recompensa positiva por estar dentro del rango
        else:
            distance_reward = -0.004 * (distance - 100)  # Penalización por estar fuera del rango

        time_penalty = -0.0005 * time_since_last_hit
        
        aggressiveness_signal = 0.5 * distance_reward + 0.5 * time_penalty

        # Normales: Daño al oponente, daño recibido y puntuación
        opponent_health_reward = 0
        if opponent_current_health < self.previous_enemy_health:
            opponent_health_reward = (self.previous_enemy_health - opponent_current_health) * 2.5
            self.previous_enemy_health = opponent_current_health
        
        damage_taken_reward = 0
        if current_health < self.previous_health:
            damage_taken_reward = (current_health - self.previous_health) * 2.5
            self.previous_health = current_health
        
        normal_signal = opponent_health_reward + damage_taken_reward

        # Combine in-game rewards
        reward += aggressiveness_signal + normal_signal

        match_reward = 0
        if self.previous_matches_won < matches_won:
            if current_health > (176 / 2):
                match_reward = 300 * (current_health / 176)
            else:
                match_reward = 200 * (current_health / 176)
            self.previous_matches_won = matches_won

        if self.previous_enemy_matches_won < opponent_matches_won:
            match_reward = -200
        reward += match_reward

        # Store rewards information
        self.rewards_data.append({
            'reward': reward,
            'aggressiveness_signal': aggressiveness_signal,
            'normal_signal': normal_signal,
            'match_reward': match_reward,
            'distance_reward': distance_reward,
            'time_penalty': time_penalty,
            'current_health': current_health,
            'opponent_current_health': opponent_current_health,
            'done': done
        })
        if done and self.save_data_flag:
            self.save()
            self.rewards_data = []

        # Update previous values
        self.previous_health = current_health
        self.previous_enemy_matches_won = opponent_matches_won
        self.previous_matches_won = matches_won
        self.previous_enemy_health = opponent_current_health

        return frame_delta, reward, done, info

    def save(self):
        # Save rewards data to an Excel file with incremental numbering in the specified directory
        rewards_df = pd.DataFrame(self.rewards_data)
        file_index = 0
        dir_path = './data/reward_data'
        os.makedirs(dir_path, exist_ok=True)  # Ensure the directory exists
        file_name = os.path.join(dir_path, f'rewards_data_{file_index}.xlsx')
        
        # Check if file exists and increment the index if it does
        while os.path.exists(file_name):
            file_index += 1
            file_name = os.path.join(dir_path, f'rewards_data_{file_index}.xlsx')
        
        rewards_df.to_excel(file_name, index=False)
        print(f"Data saved to {file_name}")

    def enable_save(self):
        """Enable saving data to Excel on termination"""
        self.save_data_flag = True

    def disable_save(self):
        """Disable saving data to Excel on termination"""
        self.save_data_flag = False

    def close(self):
        super().close()  # Ensure that the base class's close method is called
