import os
from time import sleep
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy

model_folder = "/home/tpriya/simplePython/curl-stable-baselines/src/models"
model_name = "rl_model_1999992_steps"
model_path = os.path.join(model_folder, model_name)
n_envs = 4 # Not currently running in parallel.

trained_model = A2C.load(model_path, verbose=1)


singlePlayGround = make_atari_env('Breakout-v4', n_envs=n_envs, seed=0, env_kwargs=None)
singlePlayGround = VecFrameStack(singlePlayGround, n_stack=4)

mean_reward, std_reward = evaluate_policy(trained_model, singlePlayGround, n_eval_episodes=100)

print(f"mean_reward and std deviation are:{mean_reward:.2f} +/- {std_reward:.2f}")

singlePlayGround.close()
