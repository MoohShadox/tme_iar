import numpy as np
import matplotlib.pyplot as plt
from csv import reader
import random
from importlib.machinery import SourceFileLoader
generic_net = SourceFileLoader("generic_net", "/home/nadym/Documents/Nadym/IAR/ProjetSigaud/tme_iar/utils/generic_net.py").load_module()
environment = SourceFileLoader("environment", "/home/nadym/Documents/Nadym/IAR/ProjetSigaud/tme_iar/environment.py").load_module()
policy_wrapper = SourceFileLoader("policy_wrapper", "/home/nadym/Documents/Nadym/IAR/ProjetSigaud/tme_iar/utils/policy_wrapper.py").load_module()
from environment import make_env
from generic_net import GenericNet
from policy_wrapper import PolicyWrapper
# -------------------- Rewards/Episode ----------------- #

#ddpg

def plot_rewards(filename):
    with open (filename, "r") as f:
        csv = reader(f)
        length = sum(1 for row in csv)
    with open (filename, "r") as f:
        csv = reader(f)
        x = np.zeros(length)
        y = np.zeros(length)
        i = 0
        for row in csv:
            x[i] = row[0]
            y[i] = row[1] 
            i+=1

    plt.plot(x,y)    
    plt.title("DDPG_Pendelum-V0")
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.legend(loc="lower right")
    #plt.savefig(path + '/../results/rewards_' + make_full_string(params) + '.pdf')
    plt.show()

def plot_losses(filename):
    with open (filename, "r") as f:
        csv = reader(f)
        length = sum(1 for row in csv)
    with open (filename, "r") as f:
        csv = reader(f)
        critics = np.zeros(length)
        actors = np.zeros(length)
        i = 0
        for row in csv:
            critics[i] = row[0]
            actors[i] = row[1]
            i+=1
    #actors = np.cumsum(actors)
    #critics = np.cumsum(critics)
    iterations = [i+1 for i in range(length)]
    plt.plot(iterations,actors, label = "actors loss sum")    
    plt.title("DDPG_Pendelum-V0")
    plt.xlabel("Iterations")
    plt.ylabel("Actor loss sum")
    plt.legend(loc="lower right")

    #plt.savefig(path + '/../results/rewards_' + make_full_string(params) + '.pdf')
    plt.show()

    plt.plot(iterations,critics, label = "critics loss sum")    
    plt.title("DDPG_Pendelum-V0")
    plt.xlabel("Iterations")
    plt.ylabel("Critic loss sum")
    plt.legend(loc="lower right")
    plt.show()



#plot_losses("saveDDPG_critic-actor_loss.csv")

pw = PolicyWrapper(GenericNet(), "", "", "", 0)
folder = "/home/nadym/Documents/Nadym/IAR/ProjetSigaud/tme_iar/data/policies2"
policy_file = "#TD3AgentPendulum-v0#-157.3553242564293.zip"
policy = pw.load(folder +"/"+ policy_file)
env = make_env(pw.env_name, pw.policy_type, pw.max_steps)
deterministic = True
def plot_policy_ND(policy, env, deterministic, plot=True, figname='stoch_actor.pdf', save_figure=True, definition=50) -> None:
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
    if env.observation_space.shape[0] <= 2:
        raise(ValueError("Observation space dimension {}, should be > 2".format(env.observation_space.shape[0])))

    portrait = np.zeros((definition, definition))
    state_min = env.observation_space.low
    state_max = env.observation_space.high
    # Use the dimension names if given otherwise default to "x" and "y"

    for index_x, x in enumerate(np.linspace(state_min[0], state_max[0], num=definition)):
        for index_y, y in enumerate(np.linspace(state_min[1], state_max[1], num=definition)):
            state = np.array([[x, y]])
            for i in range(2, len(state_min)):
                z = random.random() - 0.5
                state = np.append(state, z)
            action = policy.select_action(state, deterministic)
            portrait[definition - (1 + index_y), index_x] = action[0]
    plt.figure(figsize=(10, 10))
    plt.imshow(portrait, cmap="inferno", extent=[state_min[0], state_max[0], state_min[1], state_max[1]], aspect='auto')
    plt.colorbar(label="action")
    # Add a point at the center
    plt.scatter([0], [0])
    
    plt.xlabel("pos")
    plt.ylabel("angle")
    #final_show(save_figure, plot, figname, x_label, y_label, "Actor phase portrait", '/plots/')
    plt.show()


plot_policy_ND(policy, env, deterministic, figname="test")
