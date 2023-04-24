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
# in env.step() calls. This takes a long time, so we'll do it less often.
save_freq = 10_000
total_steps = 150  # in env.step() calls.
n_stack = 4  # Number of frames to stack.
src_dir = "/root/src"
log_dir = os.path.join(src_dir, 'runs')
model_dir = os.path.join(src_dir, 'models')
use_cuda = False
use_multiprocessing = True
use_icm = True
# Leave as None to train A2C feature extractor from scratch.
# Otherwise, provide a path to a pretrained feature extractor (e.g. from ICM).
embeddings_load_path = os.path.join(model_dir, "icm_102.model")  

def create_embeddings_module(n_input_channels=4):
    feature_output = 7 * 7 * 64
    feature = nn.Sequential(
        nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(feature_output, 512)
    )
    return feature

if __name__ == "__main__":

    checkpoint_callback = CheckpointCallback(
        save_freq=max(save_freq // n_envs, 1),
        save_path=model_dir,
        name_prefix="rl_model",
    )

    environment = 'Breakout-v4'
    playGround = gym.make(environment)
    playGround.reset()

    device_str = "cuda" if use_cuda else "cpu"
    env_cls = SubprocVecEnv if use_multiprocessing else None
    env_wrapper = IcmFrameStack if use_icm else VecFrameStack

    # print(playGround.action_space)

    playGround = make_atari_env(
        'Breakout-v4', n_envs=n_envs, vec_env_cls=env_cls, seed=0)
    playGround = env_wrapper(playGround, n_stack=n_stack, log_path=log_dir,
                             device=device_str, saving_freq=save_freq, model_path=model_dir)

    print(playGround.observation_space.shape)

    # Create feature extractor
    icm = create_embeddings_module(n_input_channels=n_stack)
    if embeddings_load_path is not None:
        icm.load_state_dict(th.load(embeddings_load_path))
    kwargs = {"features_extractor_kwargs": {"icm_embeddings": icm},
              "features_extractor_class": IcmCnn}
    laModel = A2C('CnnPolicy', playGround, policy_kwargs=kwargs,
                  verbose=1, tensorboard_log=log_dir, device=device_str)

    laModel.learn(total_timesteps=total_steps, callback=checkpoint_callback)




   