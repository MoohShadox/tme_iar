import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

from utils.generic_net import GenericNet
from utils.rlnn import RLNN


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)



class Critic(RLNN):
    def __init__(self, state_dim, action_dim, layer_norm=False,learning_rate=0.001):
        super(Critic, self).__init__(state_dim, action_dim, 1)

        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        if layer_norm:
            self.n1 = nn.LayerNorm(400)
            self.n2 = nn.LayerNorm(300)
        self.layer_norm = layer_norm

    def forward(self, x, u):

        if not self.layer_norm:
            x = torch.relu(self.l1(torch.cat([x, u], 1)))
            x = torch.relu(self.l2(x))
            x = self.l3(x)

        else:
            x = torch.relu(self.n1(self.l1(torch.cat([x, u], 1))))
            x = torch.relu(self.n2(self.l2(x)))
            x = self.l3(x)

        return x


class TD3_Critic(RLNN):
    def __init__(self, state_dim, action_dim, layer_norm=False):
        super(TD3_Critic, self).__init__(state_dim, action_dim, 1)

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

        if layer_norm:
            self.n1 = nn.LayerNorm(400)
            self.n2 = nn.LayerNorm(300)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 400)
        self.l5 = nn.Linear(400, 300)
        self.l6 = nn.Linear(300, 1)

        if layer_norm:
            self.n4 = nn.LayerNorm(400)
            self.n5 = nn.LayerNorm(300)
        self.layer_norm = layer_norm

    def forward(self, x, u):

        if not self.layer_norm:
            x1 = torch.relu(self.l1(torch.cat([x, u], 1)))
            x1 = torch.relu(self.l2(x1))
            x1 = self.l3(x1)

        else:
            x1 = torch.relu(self.n1(self.l1(torch.cat([x, u], 1))))
            x1 = torch.relu(self.n2(self.l2(x1)))
            x1 = self.l3(x1)

        if not self.layer_norm:
            x2 = torch.relu(self.l4(torch.cat([x, u], 1)))
            x2 = torch.relu(self.l5(x2))
            x2 = self.l6(x2)

        else:
            x2 = torch.relu(self.n4(self.l4(torch.cat([x, u], 1))))
            x2 = torch.relu(self.n5(self.l5(x2)))
            x2 = self.l6(x2)

        return x1, x2