from itertools import count
from typing import List

import torch
from machin.frame.algorithms import SAC
from machin.utils.logging import default_logger as logger
from torch.nn.functional import softplus
from torch.distributions import Normal
import torch as t
import torch.nn as nn
import gym
import numpy as np

import pandas as pd
# configurations
env = gym.make("Pendulum-v0")
observe_dim = 3
action_dim = 1
action_range = 2
max_episodes = 500
max_steps = 200
noise_param = (0, 0.2)
noise_mode = "normal"
solved_reward = -140
solved_repeat = 5


def atanh(x):
    return 0.5 * t.log((1 + x) / (1 - x))


# model definition
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_range):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 320)
        self.fc2 = nn.Linear(320, 640)
        self.mu_head = nn.Linear(640, action_dim)
        self.sigma_head = nn.Linear(640, action_dim)
        self.action_range = action_range

    @torch.jit.unused
    def forward(self, state, action=None):
        a = t.relu(self.fc1(state))
        a = t.relu(self.fc2(a))
        mu = self.mu_head(a)
        sigma = softplus(self.sigma_head(a))
        dist = Normal(mu, sigma)
        act = (atanh(action / self.action_range)
               if action is not None
               else dist.rsample())
        act_entropy = dist.entropy()
        # the suggested way to confine your actions within a valid range
        # is not clamping, but remapping the distribution
        act_log_prob = dist.log_prob(act)
        act_tanh = t.tanh(act)
        act = act_tanh * self.action_range
        # the distribution remapping process used in the original essay.
        act_log_prob -= t.log(self.action_range *
                              (1 - act_tanh.pow(2)) +
                              1e-6)
        act_log_prob = act_log_prob.sum(1, keepdim=True)
        return act, act_log_prob, act_entropy

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
        act: List[float] = action[0].data.tolist()
        return act

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(state_dim + action_dim, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, state, action):
        state_action = t.cat([state, action], 1)
        q = t.relu(self.fc1(state_action))
        q = t.relu(self.fc2(q))
        q = self.fc3(q)
        return q

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
            state = t.tensor(state, dtype=t.float32).view(1, observe_dim)
            action = policy.select_action(state, deterministic)
            next_state, reward, done, _ = env.step(action)
            state = t.tensor(next_state, dtype=t.float32).view(1, observe_dim)
            total_reward += reward
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
        # env.render(mode='rgb_array')
        total_reward = 0
        state = t.tensor(env.reset(), dtype=t.float32).view(1, observe_dim)
        for _ in count():
            action = policy.select_action(state, deterministic)
            next_state, reward, done, _ = env.step(action)
            state = t.tensor(next_state, dtype=t.float32).view(1, observe_dim)
            total_reward += reward
            if done:
                scores.append(total_reward)
                break
    scores = np.array(scores)
    print( "mean: ", scores.mean(), "std:", scores.std())
    return scores

if __name__ == "__main__":
    actor = Actor(observe_dim, action_dim, action_range)
    critic = Critic(observe_dim, action_dim)
    critic_t = Critic(observe_dim, action_dim)
    critic2 = Critic(observe_dim, action_dim)
    critic2_t = Critic(observe_dim, action_dim)

    sac = SAC(actor, critic, critic_t, critic2, critic2_t,
              t.optim.Adam,
              nn.MSELoss(reduction='sum'))

    episode, step, reward_fulfilled = 0, 0, 0
    smoothed_total_reward = 0
    D = {
        "episode":[],
        "reward":[]
    }
    while episode < max_episodes:
        episode += 1
        total_reward = 0
        terminal = False
        step = 0
        state = t.tensor(env.reset(), dtype=t.float32).view(1, observe_dim)

        while not terminal and step <= max_steps:
            step += 1
            with t.no_grad():
                old_state = state
                # agent model inference
                action = sac.act({"state": old_state})[0]
                state, reward, terminal, _ = env.step(action.numpy())
                state = t.tensor(state, dtype=t.float32).view(1, observe_dim)
                total_reward += reward[0]

                sac.store_transition({
                    "state": {"state": old_state},
                    "action": {"action": action},
                    "next_state": {"state": state},
                    "reward": reward[0],
                    "terminal": terminal or step == max_steps
                })

        # update, update more if episode is longer, else less
        if episode > 100:
            for _ in range(step):
                sac.update()

        # show reward
        smoothed_total_reward = (smoothed_total_reward * 0.9 +
                                 total_reward * 0.1)
        logger.info("Episode {} total reward={:.2f}"
                    .format(episode, smoothed_total_reward))
        D["episode"].append(episode)
        D["reward"].append(smoothed_total_reward)
        if smoothed_total_reward > solved_reward:
            reward_fulfilled += 1
            if reward_fulfilled >= solved_repeat:
                logger.info("Environment solved!")
                evaluate_pol_gym(env, actor, False)
                df = pd.DataFrame(D)
                df.to_csv("SAC_logs.csv")
                exit(0)
        else:
            reward_fulfilled = 0


