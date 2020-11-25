import random

import gym
import torch as t
import numpy as np
import matplotlib.pyplot as plt




class Agent():
    """Interacts with and learns from the environment."""
    policy_type = "UNKNOWN"
    def __init__(self):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.stats = {}

    def get_actor_to_plot(self,*args):
        pass

    def save_stats(self,**kwargs):
        for kw,val in kwargs.items():
            self.stats[kw] = self.stats.get(kw,[]) + [val]

    def plot_curve(self,kw,title=None, x_label = None, y_label = None,save=None, path_to_save=None ):
        plt.plot(np.arange(len(self.stats[kw])), self.stats[kw])
        plt.title(title if title else kw)
        plt.xlabel(x_label if x_label else kw+"_x")
        plt.ylabel(y_label if y_label else kw+"_y")
        plt.legend(loc="lower right")
        if(save):
            plt.savefig(path_to_save if path_to_save else "../data/fig/"+self.policy_type+"_"+((x_label+"_"+y_label) if x_label and y_label else kw))
        # plt.savefig(path + '/../results/rewards_' + make_full_string(params) + '.pdf')
        plt.show()


