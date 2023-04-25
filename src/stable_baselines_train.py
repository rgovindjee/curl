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
from stable_baselines3.common.torch_layers import NatureCNN

# Save a checkpoint every set number of steps
n_envs = 2  # Number of environments to run in parallel.
# in env.step() calls. This takes a long time, so we'll do it less often.
save_freq = 500_000
total_steps = 100_000_000  # in env.step() calls.
n_stack = 4  # Number of frames to stack.
src_dir = "/root/src"
log_dir = os.path.join(src_dir, 'runs')
model_dir = os.path.join(src_dir, 'models')
use_cuda = False
use_multiprocessing = True
# Leave as None to train A2C feature extractor from scratch.
# Otherwise, provide a path to a pretrained feature extractor (e.g. from ICM).
embeddings_load_path = None # os.path.join(model_dir, "icm_102.model")
rewards = "extrinsic"  # "extrinsic" or "intrinsic" (curiosity) or "both" (straight sum)

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
        nn.Linear(feature_output, 512),
        nn.ReLU(),
    )
    return feature

if __name__ == "__main__":

    checkpoint_callback = CheckpointCallback(
        save_freq=max(save_freq // n_envs, 1),
        save_path=model_dir,
        name_prefix="rl_model",
    )

    environment = 'Breakout-v4'
    training_env = gym.make(environment)
    training_env.reset()

    device_str = "cuda" if use_cuda else "cpu"
    env_cls = SubprocVecEnv if use_multiprocessing else None

    # print(training_env.action_space)
    training_env = make_atari_env(
        'Breakout-v4', n_envs=n_envs, vec_env_cls=env_cls, seed=0)
    if rewards == "extrinsic":
        training_env = VecFrameStack(training_env, n_stack=n_stack)
    elif rewards == "intrinsic":
        training_env = IcmFrameStack(training_env, n_stack=n_stack, log_path=log_dir,
                                device=device_str, saving_freq=save_freq, model_path=model_dir,
                                learning_rate=5e-6)
    elif rewards == "both":
        raise NotImplementedError
    else:
        raise ValueError(f"Invalid rewards argument: {rewards}")
    
    print(training_env.observation_space.shape)

    # Create feature extractor for A2C. May be pretrained, e.g. from an ICM module.
    icm_feature_extractor = create_embeddings_module(n_input_channels=n_stack)
    if embeddings_load_path is not None:
        icm_feature_extractor.load_state_dict(th.load(embeddings_load_path))
    kwargs = {"features_extractor_kwargs": {"icm_embeddings": icm_feature_extractor},
              "features_extractor_class": IcmCnn}
    # kwargs = {"features_extractor_class": NatureCNN}

    a2c_model = A2C('CnnPolicy', training_env, policy_kwargs=kwargs,
                  verbose=1, tensorboard_log=log_dir, device=device_str)

    a2c_model.learn(total_timesteps=total_steps, callback=checkpoint_callback)




   