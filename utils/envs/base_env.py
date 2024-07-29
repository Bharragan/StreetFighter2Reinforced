import retro
import numpy as np
import cv2
from gym import Env
from gym.spaces import Box, MultiBinary
import math
import os

class BaseStreetFighterEnv(Env):
    def __init__(self, save_video=False, state='guile'):
        super().__init__()
        self.observation_space = Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
        self.action_space = MultiBinary(12)
        self.state = state
        self.game = retro.make(game='StreetFighterIISpecialChampionEdition-Genesis', state=self.state, use_restricted_actions=retro.Actions.FILTERED)
        self.previous_frame = None
        self.score = 0
        self.previous_health = None
        self.opponent_previous_health = None
        self.last_hit_time = None
        self.frames = []  # List to store frames
        self.save_video = save_video  # Flag to enable/disable video saving

    def reset(self):
        obs = self.game.reset()
        preprocessed_obs = self.preprocess(obs)
        self.previous_frame = preprocessed_obs
        self.score = 0
        self.previous_health = 176
        self.opponent_previous_health = 176
        self.last_hit_time = 0
        self.frames = [obs] if self.save_video else []  # Initialize frames list if save_video is enabled
        return preprocessed_obs

    def preprocess(self, observation):
        gray = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
        resize = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_CUBIC)
        channels = np.reshape(resize, (84, 84, 1))
        return channels

    def step(self, action):
        obs, reward, done, info = self.game.step(action)
        preprocessed_obs = self.preprocess(obs)
        frame_delta = preprocessed_obs - self.previous_frame
        self.previous_frame = preprocessed_obs

        # Store the original frame if save_video is enabled
        if self.save_video:
            self.frames.append(obs)

        opponent_current_health = info.get('enemy_health', self.opponent_previous_health)

        # Penalizaciones y recompensas comunes
        distance_x = info['enemy_x_position'] - info['x_position']
        distance_y = info['enemy_y_position'] - info['y_position']
        distance = math.sqrt(distance_x**2 + distance_y**2)
        round_timer = info['round_timer']

        # Verificamos si se hizo daño al oponente
        if opponent_current_health < self.opponent_previous_health:
            self.last_hit_time = 0
        else:
            self.last_hit_time += 1

        self.opponent_previous_health = opponent_current_health

        # If the game is done and save_video flag is enabled, save the video
        if done and self.save_video:
            self.save_video_to_file()
            self.frames = []

        return frame_delta, reward, done, info

    def render(self, *args, **kwargs):
        self.game.render()

    def save_video_to_file(self):
        if not self.save_video:
            return  # If save_video flag is not enabled, do nothing

        # Ensure the directory exists
        output_dir = 'reports/videos/'
        os.makedirs(output_dir, exist_ok=True)

        # Create a video writer with normalized filename
        file_index = 0
        file_name = os.path.join(output_dir, f'output_video_{file_index}.mp4')
        
        # Check if file exists and increment the index if it does
        while os.path.exists(file_name):
            file_index += 1
            file_name = os.path.join(output_dir, f'output_video_{file_index}.mp4')

        # Create a video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
        height, width, _ = self.frames[0].shape
        video_writer = cv2.VideoWriter(file_name, fourcc, 30, (width, height))

        # Write each frame to the video file with the frame number
        for i, frame in enumerate(self.frames):
            frame_with_text = frame.copy()
            cv2.putText(frame_with_text, f'Frame: {i}', (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            video_writer.write(frame_with_text)

        video_writer.release()
        print(f'Video saved as {file_name}')

    def close(self):
        self.game.close()
