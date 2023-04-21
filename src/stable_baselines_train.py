# From https://github.com/AnuraagRath/A.I-learns-to-play-Atari-Breakout-ReinforcementLearning
import os
import gym
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.callbacks import CheckpointCallback

# Save a checkpoint every set number of steps
n_envs = 16  # Number of environments to run in parallel.
save_freq = 200_000  # in env.step() calls. This takes a long time, so we'll do it less often.
total_steps = 100_000_000  # in env.step() calls.
n_stack = 4  # Number of frames to stack.
src_dir = "/home/rajivg/curl/src"
use_cuda = True
use_multiprocessing = True

if __name__ == "__main__":
    a2cPath = os.path.join(src_dir, 'models')

    checkpoint_callback = CheckpointCallback(
      save_freq=max(save_freq // n_envs, 1),
      save_path=a2cPath,
      name_prefix="rl_model",
    )

    environment = 'Breakout-v4'
    playGround = gym.make(environment)
    playGround.reset()

    device_str = "cuda" if use_cuda else "cpu"
    env_cls = SubprocVecEnv if use_multiprocessing else None

    print(playGround.action_space)
    print(playGround.observation_space)

    playGround = make_atari_env('Breakout-v4', n_envs=n_envs, vec_env_cls=SubprocVecEnv, seed=0)
    playGround = VecFrameStack(playGround, n_stack=n_stack)

    logPath = os.path.join(src_dir, 'runs')

    laModel = A2C('CnnPolicy', playGround ,verbose=1, tensorboard_log=logPath, device=device_str)

    laModel.learn(total_timesteps=total_steps, callback=checkpoint_callback)
    