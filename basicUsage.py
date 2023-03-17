import gymnasium as gym
#import pygame

import os

os.environ["SDL_VIDEODRIVER"]="x11"
#pygame.init()
#pygame.display.list_modes()
env = gym.make("CartPole-v1", render_mode="human")

observation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()
env.close()
