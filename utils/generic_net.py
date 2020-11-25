import torch
import torch.nn as nn
import random

import gym
import torch as t
import numpy as np
import matplotlib.pyplot as plt


def plot_policy_ND(policy, env=None, plot=False, figname='stoch_actor.pdf', save_figure=True, definition=50):
    """
    Plot a policy for a ND environment like pendulum or cartpole
    :param policy: the policy to be plotted
    :param env: the evaluation environment
    :param deterministic: whether the deterministic version of the policy should be plotted
    :param plot: whether the plot should be interactive
    :param figname: the name of the file to save the figure
    :param save_figure: whether the figure should be saved
    :param definition: the resolution of the plot
    :return: nothing
    """
    if(not env):
        env = gym.make("Pendulum-v0")
    if env.observation_space.shape[0] <= 2:
        raise(ValueError("Observation space dimension {}, should be > 2".format(env.observation_space.shape[0])))
    portrait = np.zeros((definition, definition))
    state_min = env.observation_space.low
    state_max = env.observation_space.high
    for index_x, x in enumerate(np.linspace(state_min[0], state_max[0], num=definition)):
        for index_y, y in enumerate(np.linspace(state_min[1], state_max[1], num=definition)):
            state = np.array([[x, y]])
            for i in range(2, len(state_min)):
                z = random.random() - 0.5
                state = np.append(state, z)
            state = t.tensor(np.array(state), dtype=t.float32).view(len(state))
            action = policy.select_action(state, False)
            portrait[definition - (1 + index_y), index_x] = action[0]
    return portrait


class GenericNet(nn.Module):
    """
    The super class of all policy and critic networks
    Contains general behaviors like loading and saving, and updating from a loss
    The stardnard loss function used is the Mean Squared Error (MSE)
    """
    def __init__(self):
        super(GenericNet, self).__init__()
        self.loss_func = torch.nn.MSELoss()

    def save_model(self, filename) -> None:
        """
        Save a neural network model into a file
        :param filename: the filename, including the path
        :return: nothing
        """
        torch.save(self, filename)

    def load_model(self, filename):
        """
        Load a neural network model from a file
        :param filename: the filename, including the path
        :return: the resulting pytorch network
        """
        net = torch.load(filename)
        net.eval()
        return net

    def update(self, loss) -> None:
        """
        Apply a loss to a network using gradient backpropagation
        :param loss: the applied loss
        :return: nothing
        """
        self.optimizer.zero_grad()
        loss.sum().backward()
        self.optimizer.step()

    def getHeatMap(self,env=None):
        return plot_policy_ND(self,env)
