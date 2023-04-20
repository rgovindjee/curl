# From https://github.com/AnuraagRath/A.I-learns-to-play-Atari-Breakout-ReinforcementLearning
import os
import gym
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_atari_env

environment = 'Breakout-v4'
playGround = gym.make(environment)
playGround.reset()

print(playGround.action_space)
print(playGround.observation_space)

playGround = make_atari_env('Breakout-v4', n_envs=4, seed=0)
playGround = VecFrameStack(playGround, n_stack=4)

logPath = os.path.join('/root/src', 'runs')
a2cPath = os.path.join('/root/src', 'models')

laModel = A2C('CnnPolicy', playGround ,verbose=1, tensorboard_log=logPath, device='cuda')

laModel.learn(total_timesteps=500000)
laModel.save(a2cPath)