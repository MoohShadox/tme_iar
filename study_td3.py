# !/usr/bin/env python3
import argparse
from copy import deepcopy

import gym
import gym.spaces
import numpy as np
import pandas as pd
import torch.multiprocessing as mp

from agents.TD3_Agent import DTD3
from utils.random_process import *
from utils.util import *
from utils.memory import Memory, SharedMemory

#USE_CUDA = torch.cuda.is_available()
USE_CUDA = False

if USE_CUDA:
    FloatTensor = torch.cuda.FloatTensor
else:
    FloatTensor = torch.FloatTensor


def evaluate(actor, env, memory=None, n_episodes=1, random=False, noise=None, render=False):
    """
    Computes the score of an actor on a given number of runs
    """
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = int(env.action_space.high[0])
    if not random:
        def policy(state):
            state = FloatTensor(state.reshape(-1))
            action = actor(state).cpu().data.numpy().flatten()

            if noise is not None:
                action += noise.sample()

            return np.clip(action, -max_action, max_action)

    else:
        def policy(state):
            return env.action_space.sample()

    scores = []
    steps = 0

    for _ in range(n_episodes):

        score = 0
        obs = deepcopy(env.reset())
        done = False

        while not done:
            # get next action and act
            action = policy(obs)
            n_obs, reward, done, info = env.step(action)
            done_bool = 0 if steps + \
                1 == env._max_episode_steps else float(done)
            score += reward
            steps += 1

            # adding in memory
            if memory is not None:
                memory.add((obs, n_obs, action, reward, done_bool))
            obs = n_obs

            # render if needed
            if render:
                env.render()

            # reset when done
            if done:
                env.reset()

        scores.append(score)

    return np.mean(scores), steps



def train(agent,n_episodes, max_steps, debug=False, render=False, ou_noise = "ou_noise",
          ou_theta = 0.15,
          ou_sigma = 0.2,
          ou_mu = 0.0,
          gauss_sigma = 0.1,
          n_actor = 1,
          n_eval = 1,
          training_per_agent = 100,
          animate=False,
          ):
    """
    Train the whole process
    """
    envs = [gym.make("Pendulum-v0") for _ in range(n_actor)]
    state_dim = envs[0].observation_space.shape[0]
    action_dim = envs[0].action_space.shape[0]
    max_action = int(envs[0].action_space.high[0])
    total_steps = 0
    step_cpt = 0
    n = 0
    # action noise
    if ou_noise:
        a_noise = OrnsteinUhlenbeckProcess(
            action_dim, mu=ou_mu, theta=ou_theta, sigma=ou_sigma)
    else:
        a_noise = GaussianNoise(action_dim, sigma=gauss_sigma)

    while total_steps < max_steps:
        random = total_steps < 10000
        actor_steps = 0

        # training all agents
        for i in range(n_actor):
            f, s = evaluate(agent.actors[i], envs[i], n_episodes=n_eval,
                            noise=a_noise, random=random, memory=agent.memory, render=render)
            actor_steps += s
            total_steps += s
            step_cpt += s
            # print score
            prCyan('noisy RL agent fitness:{}'.format(f))

        for i in range(n_actor):
            agent.train(training_per_agent, i,animate = animate)


        # saving models and scores
        if step_cpt >= 5000:

            step_cpt = 0

            fs = []
            for i in range(n_actor):
                f, _ = evaluate(
                    agent.actors[i], envs[i], n_episodes=n_eval)
                fs.append(f)

                # print score
                prRed('RL agent fitness:{}'.format(f))
            # saving scores
            res = {"total_steps": total_steps,
                   "average_score": np.mean(fs), "best_score": np.max(fs)}
            for i in range(n_actor):
                agent.store_policy('Pendulum-v0', score=fs[i],index=i)
            n += 1

        # printing iteration resume
        if debug:
            prPurple('Iteration#{}: Total steps:{} \n'.format(
                n, total_steps))


if __name__ == "__main__":


    # The environment
    env = gym.make("Pendulum-v0")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = int(env.action_space.high[0])


    # replay buffer
    memory = Memory(100000, state_dim, action_dim)



    # agent
    agent = DTD3(state_dim, action_dim, max_action, memory,n_actor=5)
    print("starting")
    train(agent,n_episodes=1000,max_steps=1000000,debug=True,n_eval=100,n_actor=5)
