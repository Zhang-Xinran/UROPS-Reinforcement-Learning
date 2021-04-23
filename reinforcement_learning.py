import pickle
import random
import numpy as np


# generate all possible states
def generate_all_states():
    S_full = []
    for s1 in range(0, 21):
        for s2 in range(0, 21):
            S_full.append([s1, s2])
    return S_full


def generate_action(s1, s2):
    return random.randint(max(-5, -s2, -(20 - s1)), min(s1, 5, 20 - s2))


def generate_all_state_action_pairs():
    state_action_space = dict()
    for s1 in range(0, 21):
        for s2 in range(0, 21):
            state_action_space[(s1, s2)] = list(range(max(-5, -s2, -(20 - s1)), min(s1, 5, 20 - s2)+1))
    return state_action_space


def generate_action_from_policy(s1, s2, policy):
    return np.random.choice(state_action_space[(s1, s2)], 1, policy[(s1, s2)])[0]


# generate one episode
def generate_sequence_for_one_episode(T, policy):
    S = [[random.randint(1, 20), random.randint(1, 20)]]
    A = []
    Rent = []
    Return = []
    Reward = []
    for t in range(T):
        A.append(generate_action_from_policy(S[t][0], S[t][1], policy))
        Rent.append(np.random.poisson(lam=(3, 4), size=(1, 2)).tolist()[0])
        Return.append(np.random.poisson(lam=(3, 2), size=(1, 2)).tolist()[0])
        if S[t][0] - A[t] - Rent[t][0] >= 0 and S[t][1] + A[t] - Rent[t][1] >= 0:
            Reward.append(10 * (Rent[t][0] + Rent[t][1]) - abs(A[t]) * 2)
            s1 = min(S[t][0] - A[t] - Rent[t][0] + Return[t][0], 20)
            s2 = min(S[t][1] + A[t] - Rent[t][1] + Return[t][1], 20)
            S.append([s1, s2])
        elif S[t][0] - A[t] - Rent[t][0] < 0 and S[t][1] + A[t] - Rent[t][1] >= 0:
            Reward.append(10 * (S[t][0] - A[t] + Rent[t][1]) - abs(A[t]) * 2)
            s1 = min(0 + Return[t][0], 20)
            s2 = min(S[t][1] + A[t] - Rent[t][1] + Return[t][1], 20)
            S.append([s1, s2])
        elif S[t][0] - A[t] - Rent[t][0] >= 0 and S[t][1] + A[t] - Rent[t][1] < 0:
            Reward.append(10 * (Rent[t][0] + S[t][1] + A[t]) - abs(A[t]) * 2)
            s1 = min(S[t][0] - A[t] - Rent[t][0] + Return[t][0], 20)
            s2 = min(0 + Return[t][1], 20)
            S.append([s1, s2])
        else:
            Reward.append(10 * (S[t][0] - A[t] + S[t][1] + A[t]) - abs(A[t]) * 2)
            s1 = min(0 + Return[t][0], 20)
            s2 = min(0 + Return[t][1], 20)
            S.append([s1, s2])

    return S, A, Reward, policy


# print("State for one episode:", S)
# print("Action for one episode:", A)
# print("Reward for one episode:", Reward)


# for each episode
def loop_for_one_episode(S, A, Reward, policy):
    G = 0
    SA_pair = tuple([tuple(S[i] + [A[i]]) for i in range(T)])
    for t in range(T - 1, -1, -1):
        G = gamma * G + Reward[t]
        if not tuple(S[t] + [A[t]]) in SA_pair[:t]:
            if tuple(S[t] + [A[t]]) in Return:
                Return[tuple(S[t] + [A[t]])] += G
                Count[tuple(S[t] + [A[t]])] += 1
            else:
                Return[tuple(S[t] + [A[t]])] = G
                Count[tuple(S[t] + [A[t]])] = 1

    for state in Return.keys():
        Average_return[state] = Return[state] / Count[state]

    # print(Return)
    # print(Count)
    # print(Average_return)

    argmax = dict()
    for state in Average_return:
        if state[:2] not in argmax:
            argmax[state[:2]] = [state[2], Average_return[state]]
        else:
            if Average_return[state] > argmax[state[:2]][1]:
                argmax[state[:2]] = [state[2], Average_return[state]]

    for state in argmax:
        l = []
        for i in range(len(policy[state])):
            if state_action_space[state][i] == argmax[state][0]:
                l.append(1-epsilon+epsilon/len(policy[state]))
            else:
                l.append(epsilon/len(policy[state]))
        policy[state] = l

    return policy


if __name__ == "__main__":
    gamma = 0.9
    epsilon = 0.1
    T = 20000
    state_action_space = generate_all_state_action_pairs()
    policy = dict()
    for key in state_action_space:
        actions = state_action_space[key]
        policy[key] = [1/len(actions) for i in actions]

    Return = dict()
    Count = dict()
    Average_return = dict()

    episode = 5000
    for i in range(episode):
        S, A, Reward, policy = generate_sequence_for_one_episode(T, policy)
        # update the policy in each iteration
        policy = loop_for_one_episode(S, A, Reward, policy)

    print(len(Count))
    print(sum(Count.values()))
    print(policy)

    filehandler = open(f'{episode}_episode_{T}_days_final_policy.pkl', 'wb')
    pickle.dump(policy, filehandler)



