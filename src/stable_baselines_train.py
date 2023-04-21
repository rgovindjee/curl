# From https://github.com/AnuraagRath/A.I-learns-to-play-Atari-Breakout-ReinforcementLearning
import os
import gym
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.callbacks import CheckpointCallback

# Save a checkpoint every set number of steps
n_envs = 32  # Number of environments to run in parallel.
save_freq = 2000  # in env.step() calls. This takes a long time, so we'll do it less often.
total_steps = 10000  # in env.step() calls.

checkpoint_callback = CheckpointCallback(
  save_freq=max(save_freq // n_envs, 1),
  save_path="/root/src/models",
  name_prefix="rl_model",
)

environment = 'Breakout-v4'
playGround = gym.make(environment)
playGround.reset()

print(playGround.action_space)
print(playGround.observation_space)

playGround = make_atari_env('Breakout-v4', n_envs=n_envs, seed=0)
playGround = VecFrameStack(playGround, n_stack=4)  # Stack 4 frames

logPath = os.path.join('/root/src', 'runs')
a2cPath = os.path.join('/root/src', 'models')

laModel = A2C('CnnPolicy', playGround ,verbose=1, tensorboard_log=logPath, device='cuda')

laModel.learn(total_timesteps=total_steps, callback=checkpoint_callback)