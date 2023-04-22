from typing import Any, Dict, List, Mapping, Optional, Tuple, Union
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvWrapper
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.optim as optim
from torch.nn import init
import numpy as np

class IcmFrameStack(VecFrameStack):
    """
    Frame stacking wrapper for vectorized environment that contains an intrinsic curiosity module.
    When environments are rolled out, the ICM calculates the intrinsic reward as well as
    performs a gradient descent step on the ICM's parameters.
    """
    def __init__(self,
                 venv: VecEnv,
                 n_stack: int,
                 channels_order = None,
                 learning_rate=1e-4) -> None:
        super().__init__(venv, n_stack, channels_order)
        # Initialize the ICM module.
        self.lr = learning_rate
        # TODO(rgg): get input and output sizes from the environment
        self.n_envs = venv.num_envs
        print(f"IcmFrameStack action space: {venv.action_space.shape}")
        self.n_actions = 4
        self.icm = ICMModel(output_size=self.n_actions, use_cuda=False)
        self.state = None  # Save previous state for use in ICM module
        self.optimizer = optim.Adam(list(self.icm.parameters()),
                                    lr=self.lr)

    def step_async(self, actions: np.ndarray) -> None:
        # actions is a numpy array with one action per env for SubprocVecEnvs
        self.action = actions
        return super().step_async(actions)

    def step_wait(self):
        """
        Perform a step in the environment and calculate the intrinsic reward.
        Perform gradient descent on the ICM's parameters.
        """
        observations, rewards, dones, infos = self.venv.step_wait()
        observations, infos = self.stacked_obs.update(observations, dones, infos)
        # For a SubprocVecEnv, observations is a tuple of numpy arrays with one per env.

        # print(f"observations: {len(observations)}, \n \
        #       shape of first obs: {observations[0].shape}")
        # print(f"actions shape: {self.action.shape}")

        # Calculate the intrinsic reward
        # Update s_t and s_t+1 for the ICM.
        # TODO(rgg): handle first step gracefully
        if self.state is None:
            self.state = observations
        next_state = observations
        icm_inputs = (self.state, next_state, self.action) 
        # Will return N x ... tensors where N=n_envs
        real_next_state_feature, pred_next_state_feature, pred_action = self.icm(icm_inputs)
        # Perform gradient descent on the ICM's parameters
        # Replace rewards with intrinsic rewards so training maximizes intrinsic reward
        ce = nn.CrossEntropyLoss()
        forward_mse = nn.MSELoss()
        y_action = self.icm.actions_to_tensor(self.action)  # Use this to train the inverse model
        inverse_loss = ce(
            pred_action, y_action.detach())

        forward_loss = forward_mse(
            pred_next_state_feature, real_next_state_feature.detach())
        
        # print(f"forward loss: {forward_loss}")
        # print(f"inverse loss: {inverse_loss}")

        self.optimizer.zero_grad()
        # Note this is different from the Pathak paper where they use a weighted sum 
        # including the actor-critic loss.
        loss = forward_loss + inverse_loss  
        loss.backward()
        self.optimizer.step()

        # Record previous state for use in next step
        self.state = observations
        return observations, rewards, dones, infos

class ICMModel(nn.Module):
    """
    PyTorch implementation of the Intrinsic Curiosity Module (ICM) from Pathak et al. (2017).
    Based on code implementation in jwcleo/curiosity-driven-exploration-pytorch
    """
    def __init__(self, input_size=0, output_size=0, use_cuda=True):
        """
        Args:
            input_size (int): dimensions of observation space. NOT USED
            output_size (int): dimensions of action space
        """
        super(ICMModel, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.device = torch.device('cuda' if use_cuda else 'cpu')

        feature_output = 7 * 7 * 64
        self.feature = nn.Sequential(
            nn.Conv2d(
                in_channels=4,
                out_channels=32,
                kernel_size=8,
                stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1),
            nn.LeakyReLU(),
            Flatten(),
            nn.Linear(feature_output, 512)
        )

        self.inverse_net = nn.Sequential(
            nn.Linear(512 * 2, 512),
            nn.ReLU(),
            nn.Linear(512, output_size)
        )

        self.residual = [nn.Sequential(
            nn.Linear(output_size + 512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
        ).to(self.device)] * 8

        self.forward_net_1 = nn.Sequential(
            nn.Linear(output_size + 512, 512),
            nn.LeakyReLU()
        )
        self.forward_net_2 = nn.Sequential(
            nn.Linear(output_size + 512, 512),
        )

        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                init.kaiming_uniform_(p.weight)
                p.bias.data.zero_()

            if isinstance(p, nn.Linear):
                init.kaiming_uniform_(p.weight, a=1.0)
                p.bias.data.zero_()

    def observations_to_tensor(self, obs):
        """
        Convert Tuple with length=n_envs to single tensor
        """
        # Reorder to C x H x W
        tensors = [torch.FloatTensor(arr.transpose(2, 0, 1)) for arr in obs]
        t_unsqueezed = [torch.unsqueeze(t, dim=0) for t in tensors] # Add extra dimension for batch size
        t_cat = torch.cat(t_unsqueezed, dim=0)
        return t_cat

    def actions_to_tensor(self, actions):
        """
        Convert numpy array of actions to one-hot encoding tensor
        Args:
            actions (np.ndarray): batch of actions, shape (batch_size, )
        Returns: tensor of one-hot encoded actions, shape (batch_size, n_actions)
        """
        # TODO(rgg): handle discrete actions
        # Convert to tensor
        t = torch.LongTensor(actions)
        # Convert to one-hot encoding
        t_onehot = torch.FloatTensor(actions.shape[0], self.output_size)
        t_onehot.zero_()
        t_onehot.scatter_(1, t.unsqueeze(1), 1)
        return t_onehot

    def forward(self, inputs):
        state, next_state, action = inputs
        # Convert input to tensor
        state = self.observations_to_tensor(state)
        next_state = self.observations_to_tensor(next_state)
        action = self.actions_to_tensor(action) 

        encode_state = self.feature(state)
        # print(f"encode_state: {encode_state.shape}")
        encode_next_state = self.feature(next_state)
        # get pred action
        pred_action = torch.cat((encode_state, encode_next_state), 1)
        pred_action = self.inverse_net(pred_action)
        # ---------------------

        # get pred next state
        pred_next_state_feature_orig = torch.cat((encode_state, action), 1)
        pred_next_state_feature_orig = self.forward_net_1(pred_next_state_feature_orig)

        # residual
        for i in range(4):
            pred_next_state_feature = self.residual[i * 2](torch.cat((pred_next_state_feature_orig, action), 1))
            pred_next_state_feature_orig = self.residual[i * 2 + 1](
                torch.cat((pred_next_state_feature, action), 1)) + pred_next_state_feature_orig

        pred_next_state_feature = self.forward_net_2(torch.cat((pred_next_state_feature_orig, action), 1))

        real_next_state_feature = encode_next_state
        return real_next_state_feature, pred_next_state_feature, pred_action


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)