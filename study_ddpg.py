
import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

from agents.DDPG_Agent import DDPG_Agent
# Press the green button in the gutter to run the script.


def ddpg(n_episodes=2000, max_t=200, print_every=200):
    scores_deque = deque(maxlen=print_every)
    agent = DDPG_Agent(state_size=3, action_size=1, random_seed=2)
    scores = []
    for i_episode in range(1, n_episodes + 1):
        state = env.reset()
        agent.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_deque.append(score)
        scores.append(score)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)), end="")
        torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
        torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')

        if i_episode % print_every == 0:
            agent.store_policy('Pendulum-v0', score=np.mean(scores_deque))

            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))



    return scores




if __name__ == '__main__':
    env = gym.make('Pendulum-v0')
    env.seed(2)
    scores = ddpg()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores) + 1), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
