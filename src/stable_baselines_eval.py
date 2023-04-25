import os
from time import sleep
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from a2c_embeddings import IcmCnn
from stable_baselines_train import create_embeddings_module

model_folder = "/root/src/models"
model_name = "rl_model_8000_steps.zip"
model_path = os.path.join(model_folder, model_name)
n_envs = 1 # Not currently running in parallel.
n_stack = 4  # Number of frames to stack.

singlePlayGround = make_atari_env('Breakout-v4', n_envs=n_envs, seed=0, env_kwargs={'render_mode':'human'})
singlePlayGround = VecFrameStack(singlePlayGround, n_stack=n_stack)

# trained_model = A2C.load(model_path, verbose=1)
icm = create_embeddings_module(n_input_channels=n_stack)
kwargs = {"features_extractor_kwargs": {"icm_embeddings": icm},
            "features_extractor_class": IcmCnn}
trained_model = A2C('CnnPolicy', singlePlayGround, policy_kwargs=kwargs, verbose=1)
trained_model.set_parameters(load_path_or_dict=model_path)

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
