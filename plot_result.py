import pickle
import matplotlib.pyplot as plt
import numpy as np


def generate_all_state_action_pairs():
    state_action_space = dict()
    for s1 in range(0, 21):
        for s2 in range(0, 21):
            state_action_space[(s1, s2)] = list(range(max(-5, -s2, -(20 - s1)), min(s1, 5, 20 - s2)+1))
    return state_action_space


def generate_final_policy(episode, T):
    with open(f'{episode}_episode_{T}_days_final_policy.pkl', 'rb') as f:
        policy = pickle.load(f)
    state_action_space = generate_all_state_action_pairs()
    final_policy = dict()
    for state in state_action_space:
        max_index = np.argmax(policy[state])
        final_policy[state] = state_action_space[state][max_index]
    return final_policy


def plot_policy(episode, T):
    final_policy = generate_final_policy(episode, T)
    print(final_policy)
    x = np.linspace(0, 20, 21)
    y = np.linspace(0, 20, 21)
    X, Y = np.meshgrid(x, y)
    Z = []
    for i in range(0, 21):
        l = []
        for j in range(0, 21):
            l.append(final_policy[(i,j)])
        Z.append(l)

    # ax = plt.axes(projection='3d')
    # ax.scatter3D(X, Y, Z, c=Z, cmap='Greens')
    # plt.show()

    fig,ax = plt.subplots(1,1)
    cp = ax.contourf(X,Y,Z, [-5,-4,-3,-2,-1,0,1,2,3,4,5])
    # [-5,-4,-3,-2,-1,0,1,2,3,4,5], colors='k'
    fig.colorbar(cp)  # Add a colorbar to a plot
    contour = ax.contour(X, Y, Z, [-5,-4,-3,-2,-1,0,1,2,3,4,5], colors='k')
    plt.clabel(contour, fontsize=5, colors='k')
    plt.xticks(list(range(21)), list(range(0,21)))
    plt.yticks(list(range(21)), list(range(0,21)))
    ax.set_title('final policy')
    ax.set_xlabel('Car Park 2')
    ax.set_ylabel('Car Park 1')
    plt.savefig(f'{episode}_episode_{T}_T_final_policy.png')
    plt.show()


if __name__ == "__main__":
    episode = 5000
    T = 20000
    plot_policy(episode, T)
