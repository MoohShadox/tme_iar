from typing import List

import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.generic_net import GenericNet
from utils.rlnn import RLNN

device = torch.device("cpu" if torch.cuda.is_available() else "cpu")


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class TD3_Actor(RLNN):

    def __init__(self, state_dim, action_dim, max_action, layer_norm=False, init=True,learning_rate=0.01):
        super(TD3_Actor, self).__init__(state_dim, action_dim, max_action)

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)
        self.layer_norm = layer_norm
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)


    def forward(self, x):
        x = torch.tanh(self.l1(x))
        x = torch.tanh(self.l2(x))
        x = self.max_action * torch.tanh(self.l3(x))
        return x

    #@torch.jit.export
    #def select_action(self, state, deterministic=False):
    #    """
    #    Compute an action or vector of actions given a state or vector of states
    #    :param state: the input state(s)
    #    :param deterministic: whether the policy should be considered deterministic or not
    #    :return: the resulting action(s)
    #    """
    #    state = torch.from_numpy(state).float().to(device)
    #    action = self(state).cpu().data.numpy()
    #    return np.clip(action, -1, 1)

    @torch.jit.export
    def select_action(self, state: List[float], deterministic: bool=False) -> List[float]:
        """
        Compute an action or vector of actions given a state or vector of states
        :param state: the input state(s)
        :param deterministic: whether the policy should be considered deterministic or not
        :return: the resulting action(s)
        """
        state = torch.tensor(state)
        action = self.forward(state)
        #action = np.clip(action,-1,1)
        act: List[float] = action.data.tolist()
        return act

