from agents import *
from envs import *
from utils import *
from config import *
from torch.multiprocessing import Pipe
import os

from tensorboardX import SummaryWriter

import numpy as np
import pickle


def main():
    print({section: dict(config[section]) for section in config.sections()})
    env_id = default_config['EnvID']
    env_type = default_config['EnvType']

    if env_type == 'mario':
        env = JoypadSpace(gym_super_mario_bros.make(env_id, render_mode="human"), COMPLEX_MOVEMENT)
    elif env_type == 'atari':
        env = gym.make(env_id, render_mode="human")
    else:
        raise NotImplementedError
    input_size = env.observation_space.shape  # 4
    output_size = env.action_space.n  # 2

    if 'Breakout' in env_id:
        output_size -= 1

    env.close()

    is_render = True
    model_path = 'models/{}.model'.format(env_id)
    #predictor_path = 'models/{}.pred'.format(env_id)
    #target_path = 'models/{}.target'.format(env_id)

    use_cuda = False
    use_gae = default_config.getboolean('UseGAE')
    use_noisy_net = default_config.getboolean('UseNoisyNet')

    lam = float(default_config['Lambda'])
    num_worker = 1

    num_step = int(default_config['NumStep'])

    ppo_eps = float(default_config['PPOEps'])
    epoch = int(default_config['Epoch'])
    mini_batch = int(default_config['MiniBatch'])
    batch_size = int(num_step * num_worker / mini_batch)
    learning_rate = float(default_config['LearningRate'])
    entropy_coef = float(default_config['Entropy'])
    gamma = float(default_config['Gamma'])
    clip_grad_norm = float(default_config['ClipGradNorm'])

    sticky_action = False
    life_done = default_config.getboolean('LifeDone')

    if default_config['EnvType'] == 'atari':
        env_type = AtariEnvironment
    elif default_config['EnvType'] == 'mario':
        env_type = MarioEnvironment
    else:
        raise NotImplementedError

    agent = ICMAgent

    agent = agent(
        input_size,
        output_size,
        num_worker,
        num_step,
        gamma,
        lam=lam,
        learning_rate=learning_rate,
        ent_coef=entropy_coef,
        clip_grad_norm=clip_grad_norm,
        epoch=epoch,
        batch_size=batch_size,
        ppo_eps=ppo_eps,
        use_cuda=use_cuda,
        use_gae=use_gae,
        use_noisy_net=use_noisy_net
    )

    print('Loading Pre-trained model....')
    if use_cuda:
        agent.model.load_state_dict(torch.load(model_path))
        #agent.icm.predictor.load_state_dict(torch.load(predictor_path))
        #agent.icm.target.load_state_dict(torch.load(target_path))
    else:
        agent.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        #agent.icm.predictor.load_state_dict(torch.load(predictor_path, map_location='cpu'))
        #agent.icm.target.load_state_dict(torch.load(target_path, map_location='cpu'))
    print('End load...')

    works = []
    parent_conns = []
    child_conns = []
    for idx in range(num_worker):
        parent_conn, child_conn = Pipe()
        work = env_type(env_id, is_render, idx, child_conn)
        work.start()
        works.append(work)
        parent_conns.append(parent_conn)
        child_conns.append(child_conn)

    states = np.zeros([num_worker, 4, 84, 84])

    steps = 0
    max_steps = 5000
    rall = 0
    done = False
    intrinsic_reward_list = []
    while steps < max_steps:
        steps += 1
        actions, value, policy = agent.get_action(np.float32(states) / 255.)

        for parent_conn, action in zip(parent_conns, actions):
            parent_conn.send(action)

        next_states = []
        for parent_conn in parent_conns:
            history, reward, force_done, done, log_reward = parent_conn.recv()
            rall += reward
            next_states.append(history.reshape([4, 84, 84]))
        next_states = np.stack(next_states)

        # total reward = int reward + ext Reward
        intrinsic_reward = agent.compute_intrinsic_reward(states, next_states, actions)
        intrinsic_reward_list.append(intrinsic_reward)
        states = next_states[:, :, :, :]

        if done:
            print("Saving reward data to file...")
            intrinsic_reward_list = (intrinsic_reward_list - np.mean(intrinsic_reward_list)) / np.std(
                intrinsic_reward_list)
            with open('int_reward', 'wb') as f:
                pickle.dump(intrinsic_reward_list, f)
            steps = 0
            rall = 0
    print("Done evaluating, closing environments")
    for p in parent_conns:
        p.send("SIGTERM")  # Hack to kill blocking env processes, should be a more graceful way
        p.close()

if __name__ == '__main__':
    main()
