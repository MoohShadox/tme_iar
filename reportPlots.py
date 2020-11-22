import numpy as np
import matplotlib.pyplot as plt
from csv import reader
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
    actors = np.cumsum(actors)
    critics = np.cumsum(critics)
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



plot_losses("saveDDPG_critic-actor_loss.csv")

