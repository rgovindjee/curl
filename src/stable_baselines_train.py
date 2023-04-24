# From https://github.com/AnuraagRath/A.I-learns-to-play-Atari-Breakout-ReinforcementLearning
import os
import gym
from torch import nn
import torch as th
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.callbacks import CheckpointCallback
from icm_frame_stack import IcmFrameStack
from a2c_embeddings import IcmCnn

# Save a checkpoint every set number of steps
n_envs = 6  # Number of environments to run in parallel.
save_freq = 10_000  # in env.step() calls. This takes a long time, so we'll do it less often.
total_steps = 20_500  # in env.step() calls.
n_stack = 4  # Number of frames to stack.
src_dir = "/root/src"
use_cuda = False
use_multiprocessing = True
use_icm = True

if __name__ == "__main__":
    logPath = os.path.join(src_dir, 'runs')
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
    env_wrapper = IcmFrameStack if use_icm else VecFrameStack

    # print(playGround.action_space)

    playGround = make_atari_env('Breakout-v4', n_envs=n_envs, vec_env_cls=env_cls, seed=0)
    playGround = env_wrapper(playGround, n_stack=n_stack, log_path=logPath, device=device_str)

    print(playGround.observation_space.shape)

    # Create feature extractor
    # TODO(rgg): This is a hack. We should be able to pass in a custom feature extractor loaded from file.
    n_input_channels = playGround.observation_space.shape[2]
    cnn = nn.Sequential(
        nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
        nn.ReLU(),
        nn.Flatten(),
    )
    # Compute shape by doing one forward pass
    linear = nn.Sequential(nn.Linear(7*7*64, 512), nn.ReLU())
    icm = nn.Sequential(cnn, linear)

    kwargs = {"features_extractor_kwargs":{"icm_embeddings":icm},
              "features_extractor_class":IcmCnn}
    laModel = A2C('CnnPolicy', playGround, policy_kwargs=kwargs, verbose=1, tensorboard_log=logPath, device=device_str)
  
    laModel.learn(total_timesteps=total_steps, callback=checkpoint_callback)
