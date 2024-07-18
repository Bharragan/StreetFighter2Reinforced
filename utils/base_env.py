import retro
import numpy as np
import cv2
from gym import Env
from gym.spaces import Box, MultiBinary
import math

class BaseStreetFighterEnv(Env):
    def __init__(self):
        super().__init__()
        self.observation_space = Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
        self.action_space = MultiBinary(12)
        self.game = retro.make(game='StreetFighterIISpecialChampionEdition-Genesis', use_restricted_actions=retro.Actions.FILTERED)
        self.previous_frame = None
        self.score = 0
        self.previous_health = None  # Valor inicial de salud
        self.opponent_previous_health = None
        self.last_hit_time = None

    def reset(self):
        obs = self.game.reset()
        obs = self.preprocess(obs)
        self.previous_frame = obs
        self.score = 0
        self.previous_health = 176
        self.opponent_previous_health = 176
        self.last_hit_time = 39208  # Inicializamos al valor máximo del round timer
        return obs

    def preprocess(self, observation):
        gray = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
        resize = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_CUBIC)
        channels = np.reshape(resize, (84, 84, 1))
        return channels

    def step(self, action):
        obs, reward, done, info = self.game.step(action)
        obs = self.preprocess(obs)
        frame_delta = obs - self.previous_frame
        self.previous_frame = obs

        current_health = info.get('health', self.previous_health)
        opponent_current_health = info.get('enemy_health', self.opponent_previous_health)

        # Penalizaciones y recompensas comunes
        distance_x = info['enemy_x_position'] - info['x_position']
        distance_y = info['enemy_y_position'] - info['y_position']
        distance = math.sqrt(distance_x**2 + distance_y**2)
        round_timer = info['round_timer']

        # Verificamos si se hizo daño al oponente
        if opponent_current_health < self.opponent_previous_health:
            self.last_hit_time = round_timer

        time_since_last_hit = (39208 - self.last_hit_time) / 1000  # Convertimos a segundos

        self.previous_health = current_health
        self.opponent_previous_health = opponent_current_health
        return frame_delta, reward, done, info

    def render(self, *args, **kwargs):
        self.game.render()

    def close(self):
        self.game.close()
