import os
import sys
from time import sleep
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from pathlib import Path
from a2c_embeddings import IcmCnn
from stable_baselines_train import create_embeddings_module

model_folder = "/home/tpriya/simplePython/stable_baseline_embedding/src/models"

highest_mean=sys.float_info.min
lowest_mean=sys.float_info.max
highest_std=0
lowest_std=0
n_envs = 16 # Not currently running in parallel.
n_stack=4

singlePlayGround = make_atari_env('Breakout-v4', n_envs=n_envs, seed=0, env_kwargs=None)
singlePlayGround = VecFrameStack(singlePlayGround, n_stack=n_stack)
icm = create_embeddings_module(n_input_channels=n_stack)
kwargs = {"features_extractor_kwargs": {"icm_embeddings": icm},
            "features_extractor_class": IcmCnn}
trained_model = A2C('CnnPolicy', singlePlayGround, policy_kwargs=kwargs, verbose=1)

zip_files = Path(model_folder).glob('*.zip')
for zip_file in zip_files:
    #print(zip_file)
    model_name = zip_file
    model_path = os.path.join(model_folder, model_name)
    #trained_model = A2C.load(model_path, verbose=1)
    trained_model.set_parameters(load_path_or_dict=model_path)

    mean_reward, std_reward = evaluate_policy(trained_model, singlePlayGround, n_eval_episodes=5)
    if mean_reward > highest_mean:
        highest_mean=mean_reward
        highest_std=std_reward
    if mean_reward < lowest_mean:
        lowest_mean=mean_reward
        lowest_std=std_reward

print(f"Highest mean_reward and std deviation are:{highest_mean:.2f} +/- {highest_std:.2f}")
print(f"Lowest mean_reward and std deviation are:{lowest_mean:.2f} +/- {lowest_std:.2f}")

singlePlayGround.close()
