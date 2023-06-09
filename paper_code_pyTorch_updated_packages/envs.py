import gymnasium as gym
import cv2

import numpy as np

from abc import abstractmethod
from collections import deque
from copy import copy

import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT

from torch.multiprocessing import Pipe, Process

from model import *
from config import *
from PIL import Image

train_method = default_config['TrainMethod']
max_step_per_episode = int(default_config['MaxStepPerEpisode'])


class Environment(Process):
    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def pre_proc(self, x):
        pass

    @abstractmethod
    def get_init_state(self, x):
        pass


def unwrap(env):
    if hasattr(env, "unwrapped"):
        return env.unwrapped
    elif hasattr(env, "env"):
        return unwrap(env.env)
    elif hasattr(env, "leg_env"):
        return unwrap(env.leg_env)
    else:
        return env

class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.integers(1, self.noop_max + 1) #pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, reward, terminated, truncated, info = self.env.step(self.noop_action)
            if terminated or truncated:
                # Assuming env is a v26 or newer gym env that returns two args
                obs, info = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, is_render, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip
        self.is_render = is_render

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            if self.is_render:
                # Deprecated, do when making gym
                #self.env.render()
                pass
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return obs


class MontezumaInfoWrapper(gym.Wrapper):
    def __init__(self, env, room_address):
        super(MontezumaInfoWrapper, self).__init__(env)
        self.room_address = room_address
        self.visited_rooms = set()

    def get_current_room(self):
        ram = unwrap(self.env).ale.getRAM()
        assert len(ram) == 128
        return int(ram[self.room_address])

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        self.visited_rooms.add(self.get_current_room())
        if done:
            if 'episode' not in info:
                info['episode'] = {}
            info['episode'].update(visited_rooms=copy(self.visited_rooms))
            self.visited_rooms.clear()
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        return obs


class AtariEnvironment(Environment):
    def __init__(
            self,
            env_id,
            is_render,
            env_idx,
            child_conn,
            history_size=4,
            h=84,
            w=84,
            life_done=True):
        super(AtariEnvironment, self).__init__()
        self.daemon = True
        render_mode = "human" if is_render else None
        self.env = MaxAndSkipEnv(NoopResetEnv(gym.make(env_id, render_mode=render_mode)), is_render)
        if 'Montezuma' in env_id:
            self.env = MontezumaInfoWrapper(self.env, room_address=3 if 'Montezuma' in env_id else 1)
        # TODO(rgg): make all gym environments use v26 for step() and reset() and remove compatibility layer
        self.env = gym.wrappers.EnvCompatibility(self.env, is_render)
        self.env_id = env_id
        self.is_render = is_render
        self.env_idx = env_idx
        self.steps = 0
        self.episode = 0
        self.rall = 0
        self.recent_rlist = deque(maxlen=100)
        self.child_conn = child_conn

        self.history_size = history_size
        self.history = np.zeros([history_size, h, w])
        self.h = h
        self.w = w

        self.reset()

    def run(self):
        super(AtariEnvironment, self).run()
        while True:
            try:
                action = self.child_conn.recv()
                if action == "SIGTERM": 
                    raise Exception("Received termination signal")

                if 'Breakout' in self.env_id:
                    action += 1

                s, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                if max_step_per_episode < self.steps:
                    done = True

                log_reward = reward
                force_done = done

                # Previous 3 frames
                self.history[:3, :, :] = self.history[1:, :, :]
                # Current frame
                self.history[3, :, :] = self.pre_proc(s)

                self.rall += reward
                self.steps += 1

                if done:
                    self.recent_rlist.append(self.rall)
                    print("[Episode {}({})] Step: {}  Reward: {}  Recent Reward: {}  Visited Room: [{}]".format(
                        self.episode, self.env_idx, self.steps, self.rall, np.mean(self.recent_rlist),
                        info.get('episode', {}).get('visited_rooms', {})))

                    self.history = self.reset()

                self.child_conn.send(
                    [self.history[:, :, :], reward, force_done, done, log_reward])
            except:
                break

    def reset(self):
        self.last_action = 0
        self.steps = 0
        self.episode += 1
        self.rall = 0
        s, info = self.env.reset()
        self.get_init_state(
            self.pre_proc(s))
        return self.history[:, :, :]

    def pre_proc(self, X):
        X = np.array(Image.fromarray(X).convert('L')).astype('float32')
        x = cv2.resize(X, (self.h, self.w))
        return x

    def get_init_state(self, s):
        for i in range(self.history_size):
            self.history[i, :, :] = self.pre_proc(s)


class MarioEnvironment(Process):
    def __init__(
            self,
            env_id,
            is_render,
            env_idx,
            child_conn,
            history_size=4,
            life_done=True,
            h=84,
            w=84, movement=COMPLEX_MOVEMENT, sticky_action=True,
            p=0.25):
        super(MarioEnvironment, self).__init__()
        self.daemon = True
        self.env = JoypadSpace(
            gym_super_mario_bros.make(env_id), COMPLEX_MOVEMENT)

        self.is_render = is_render
        self.env_idx = env_idx
        self.steps = 0
        self.episode = 0
        self.rall = 0
        self.recent_rlist = deque(maxlen=100)
        self.child_conn = child_conn

        self.life_done = life_done

        self.history_size = history_size
        self.history = np.zeros([history_size, h, w])
        self.h = h
        self.w = w

        self.reset()

    def run(self):
        super(MarioEnvironment, self).run()
        while True:
            try:
                action = self.child_conn.recv()
                if action == "SIGTERM": 
                    raise Exception("Received termination signal")
                if self.is_render:
                    # Deprecated, do when making gym
                    #self.env.render()
                    pass

                obs, reward, done, info = self.env.step(action)

                # when Mario loses life, changes the state to the terminal
                # state.
                if self.life_done:
                    if self.lives > info['life'] and info['life'] > 0:
                        force_done = True
                        self.lives = info['life']
                    else:
                        force_done = done
                        self.lives = info['life']
                else:
                    force_done = done

                # reward range -15 ~ 15
                log_reward = reward / 15
                self.rall += log_reward

                r = log_reward

                self.history[:3, :, :] = self.history[1:, :, :]
                self.history[3, :, :] = self.pre_proc(obs)

                self.steps += 1

                if done:
                    self.recent_rlist.append(self.rall)
                    print(
                        "[Episode {}({})] Step: {}  Reward: {}  Recent Reward: {}  Stage: {} current x:{}   max x:{}".format(
                            self.episode,
                            self.env_idx,
                            self.steps,
                            self.rall,
                            np.mean(
                                self.recent_rlist),
                            info['stage'],
                            info['x_pos'],
                            self.max_pos))

                    self.history = self.reset()

                self.child_conn.send([self.history[:, :, :], r, force_done, done, log_reward])
            except:
                break

    def reset(self):
        self.last_action = 0
        self.steps = 0
        self.episode += 1
        self.rall = 0
        self.lives = 3
        self.stage = 1
        self.max_pos = 0
        obs, info = self.env.reset()
        self.get_init_state(obs)
        return self.history[:, :, :]

    def pre_proc(self, X):
        # grayscaling
        x = cv2.cvtColor(X, cv2.COLOR_RGB2GRAY)
        # resize
        x = cv2.resize(x, (self.h, self.w))

        return x

    def get_init_state(self, s):
        for i in range(self.history_size):
            self.history[i, :, :] = self.pre_proc(s)

class AtariEnvironmentSimple():
    def __init__(
            self,
            env_id,
            is_render,
            history_size=4,
            h=84,
            w=84,
            life_done=True):
        render_mode = "human" if is_render else None
        self.env = gym.make(env_id, render_mode=render_mode)
        self.env_id = env_id
        self.is_render = is_render
        self.steps = 0
        self.episode = 0
        self.rall = 0
        self.recent_rlist = deque(maxlen=100)

        self.history_size = history_size
        self.history = np.zeros([history_size, h, w])
        self.h = h
        self.w = w

        self.reset()

    def step(self, action):
        if 'Breakout' in self.env_id:
            action += 1

        s, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated

        if max_step_per_episode < self.steps:
            done = True

        log_reward = reward
        force_done = done

        # Previous 3 frames
        self.history[:3, :, :] = self.history[1:, :, :]
        # Current frame
        self.history[3, :, :] = self.pre_proc(s)

        self.rall += reward
        self.steps += 1

        if done:
            self.recent_rlist.append(self.rall)
            print("[Episode {}] Step: {}  Reward: {}  Recent Reward: {}  Visited Room: [{}]".format(
                self.episode, self.steps, self.rall, np.mean(self.recent_rlist),
                info.get('episode', {}).get('visited_rooms', {})))

            self.history = self.reset()

        return (self.history[:, :, :], reward, force_done, done, log_reward)

    def reset(self):
        self.last_action = 0
        self.steps = 0
        self.episode += 1
        self.rall = 0
        s, info = self.env.reset()
        self.get_init_state(
            self.pre_proc(s))
        return self.history[:, :, :]

    def pre_proc(self, X):
        X = np.array(Image.fromarray(X).convert('L')).astype('float32')
        x = cv2.resize(X, (self.h, self.w))
        return x

    def get_init_state(self, s):
        for i in range(self.history_size):
            self.history[i, :, :] = self.pre_proc(s)