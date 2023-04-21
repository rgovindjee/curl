import os
from time import sleep
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
model_folder = "/root/src/models"
model_name = "rl_model_8000_steps.zip"
model_path = os.path.join(model_folder, model_name)
n_envs = 1 # Not currently running in parallel.

trained_model = A2C.load(model_path, verbose=1)

singlePlayGround = make_atari_env('Breakout-v4', n_envs=1, seed=0, env_kwargs={'render_mode':'human'})
singlePlayGround = VecFrameStack(singlePlayGround, n_stack=4)

episodes = 10
for episode in range(1, episodes+1):
    done = False
    obs = singlePlayGround.reset()
    score = 0
    
    while not done:
        action, _ = trained_model.predict(obs)
        obs, reward, done, info = singlePlayGround.step(action)
        score += reward
    print("Episode:{} Score:{}".format(episode, score))
singlePlayGround.close()
