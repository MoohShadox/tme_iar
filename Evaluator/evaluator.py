import os

import numpy as np
from itertools import count

import torch

from Evaluator.environment import make_env
from utils.generic_net import GenericNet
from utils.policy_wrapper import PolicyWrapper

device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

def evaluate_pol(env, policy, deterministic):
    """
    Function to evaluate a policy over 900 episodes
    :param env: the evaluation environment
    :param policy: the evaluated policy
    :param deterministic: whether the evaluation uses a deterministic policy
    :return: the obtained vector of 900 scores
    """
    scores = []
    for i in range(900):
        state = env.reset()
        # env.render(mode='rgb_array')
        # print("new episode")
        total_reward = 0
        for _ in count():
            action = policy.select_action(state, deterministic)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            state = next_state

            if done:
                scores.append(total_reward)
                break
    scores = np.array(scores)
    return scores


def evaluate_pol_gym(env, policy, deterministic):
    """
    Function to evaluate a policy over 900 episodes
    :param env: the evaluation environment
    :param policy: the evaluated policy
    :param deterministic: whether the evaluation uses a deterministic policy
    :return: the obtained vector of 900 scores
    """
    scores = []
    stats = []
    for i in range(1,1001):
        state = env.reset()
        # env.render(mode='rgb_array')
        # print("new episode")
        total_reward = 0
        for _ in count():
            action = policy.select_action(state, deterministic)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            state = next_state

            if done:
                scores.append(total_reward)
                break
        if(i%300 == 0):
            scores = np.array(scores)
            scores.sort()
            scores = scores[-100:-1]
            print("Mean of top 100 : ",scores.mean())
            stats.append(scores.mean())
            scores = []

    scores = np.array(scores)
    stats = np.array(stats)
    print("Finally : mean = ",stats.mean()," std : ", stats.std())
    # print("team: ", policy.team_name, "mean: ", scores.mean(), "std:", scores.std())
    return scores


class Evaluator:
    """
    A class to evaluate a set of policies stored into the same folder and ranking them accordin to their scores
    """
    def __init__(self):
        self.env_dict = {}
        self.score_dict = {}

    def load_policies(self, folder) -> None:
        """
         :param: folder : name of the folder containing policies
         Output : none (policies of the folder stored in self.env_dict)        
         """
        listdir = os.listdir(folder)
        for policy_file in listdir:
            pw = PolicyWrapper(GenericNet(), "", "", "", 0)
            policy = pw.load(folder + policy_file)
            if pw.env_name in self.env_dict:
                env = make_env(pw.env_name, pw.policy_type, pw.max_steps)
                env.set_reward_flag(False)
                env.set_duration_flag(False)
                scores = evaluate_pol(env, policy, False)
                self.score_dict[pw.env_name][scores.mean()] = [pw.team_name, scores.std()]
            else:
                env = make_env(pw.env_name, pw.policy_type, pw.max_steps)
                env.set_reward_flag(False)
                env.set_duration_flag(False)
                self.env_dict[pw.env_name] = env
                scores = evaluate_pol(env, policy, False)
                tmp_score_dict = {scores.mean(): [pw.team_name, scores.std()]}
                self.score_dict[pw.env_name] = tmp_score_dict

    def display_hall_of_fame(self) -> None:
        """
        Display the hall of fame of all the evaluated policies
        :return: nothing
        """
        print("Hall of fame")
        for k, v in self.score_dict.items():
            print("Environment :", k)
            for k2, v2 in sorted(v.items()):
                print("team: ", v2[0], "mean: ", k2, "std: ", v2[1])


if __name__ == '__main__':
    #directory = os.getcwd() + '/data/policies/'
    directory = "/home/mohamed/PycharmProjects/SIGAUD_INSPIRED/data/policies/"
    ev = Evaluator()
    ev.load_policies(directory)
    ev.display_hall_of_fame()
