import random
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, cm

N_NODES = 15
N_CLUSTERS = 3
RANGE = 1000
CLUSTER_RANGE = 150
BALANCED = True
IMBALANCED_RANGE = N_NODES // (2 * N_CLUSTERS)
COLORS = cm.rainbow(np.linspace(0, 1, N_CLUSTERS))
OMEGA = 0.2
ALPHA = 0.1
BETA = 0.2
GAMMA = 0.3
SMOOTHING_COEFFICIENT = 0.5
SMOOTHING_PROBABILITY = 0.995
# SIMILARITY = None
N_EPISODES = 1
MAX_N_LOCAL_SEARCH_EPISODES = 100
MAX_F = pow(10, 9)


def calculate_variance(nodes):
    x_nodes, y_nodes = tuple(list(zip(*nodes)))
    center = (np.mean(x_nodes), np.mean(y_nodes))
    variance = 0
    for node in nodes:
        variance += distance(node, center)
    variance /= N_NODES
    return variance


def display_nodes(nodes, title):
    x_nodes, y_nodes = tuple(list(zip(*nodes)))
    plt.figure()
    plt.scatter(x_nodes, y_nodes, color='b')
    plt.title(title)
    plt.show()
    return


def display_cluster_nodes(cluster_nodes, title):
    plt.figure()
    for nodes, color in list(zip(cluster_nodes, COLORS)):
        x_nodes, y_nodes = tuple(list(zip(*nodes)))
        plt.scatter(x_nodes, y_nodes, color=color)
    plt.title(title)
    plt.show()
    return


def create_centers():
    x_centers = random.sample(range(-RANGE, RANGE), N_CLUSTERS)
    y_centers = random.sample(range(-RANGE, RANGE), N_CLUSTERS)
    centers = list(zip(x_centers, y_centers))
    return centers


def create_nodes(centers):
    temp_1 = N_NODES // N_CLUSTERS
    if BALANCED:
        n_cluster_nodes = [temp_1 for _ in range(N_CLUSTERS)]
    else:
        temp = random.sample(range(1, N_NODES), N_CLUSTERS - 1)
        temp.extend([0, N_NODES])
        temp = sorted(temp)
        n_cluster_nodes = [temp[i + 1] - temp[i] for i in range(N_CLUSTERS)]
    nodes = []
    cluster_nodes = []
    for center, n_nodes in list(zip(centers, n_cluster_nodes)):
        x_nodes = random.sample(range(center[0] - CLUSTER_RANGE, center[0] + CLUSTER_RANGE), n_nodes)
        y_nodes = random.sample(range(center[1] - CLUSTER_RANGE, center[1] + CLUSTER_RANGE), n_nodes)
        temp = list(zip(x_nodes, y_nodes))
        nodes.extend(temp)
        cluster_nodes.append(temp)
    return nodes, cluster_nodes


def initialize_P():
    P = [[1 / N_CLUSTERS for _ in range(N_CLUSTERS)] for _ in range(N_NODES)]
    return P


def group_selection(P, method='hybrid'):
    S = []
    for i in range(N_NODES):
        if method == 'hybrid':
            if random.uniform(0, 1) > OMEGA:
                temp = pd.Series(P[i])
                temp = temp.reindex(np.random.permutation(temp.index))
                cluster = temp.idxmax()
            else:
                cluster = random.sample(range(0, N_CLUSTERS), 1)[0]
        elif method == 'greedy':
            temp = pd.Series(P[i])
            temp = temp.reindex(np.random.permutation(temp.index))
            cluster = temp.idxmax()
        else:
            raise ValueError('Undefined Group Selection Method')
        S.append(cluster)
    return S


def distance(node_1, node_2, method='Euclidean', variance=None):
    if method == 'Euclidean':
        return np.sqrt(np.power(node_1[0] - node_2[0], 2) + np.power(node_1[1] - node_2[1], 2))
    elif method == 'Gaussian':
        return np.exp(-1 * (pow(distance(node_1, node_2), 2)) / (2 * variance))


def F(state, nodes, variance):
    cluster_nodes = []
    for i in range(N_CLUSTERS):
        temp_nodes = [node for node, cluster in enumerate(state) if cluster == i]
        cluster_nodes.append(temp_nodes)
    f = 0
    for i in range(N_CLUSTERS):
        s_1, s_2 = 0, 0
        for internal_node in cluster_nodes[i]:
            for j in range(N_CLUSTERS):
                if j == i:
                    continue
                for external_node in cluster_nodes[j]:
                    # print(internal_node, external_node)
                    s_1 += distance(nodes[internal_node], nodes[external_node], method='Gaussian', variance=variance)
        for internal_node_1 in cluster_nodes[i]:
            for internal_node_2 in cluster_nodes[i]:
                if internal_node_1 == internal_node_2:
                    continue
                s_2 += distance(nodes[internal_node_1], nodes[internal_node_2], method='Gaussian', variance=variance)
        s_2 /= 2
        if s_1 == 0 and s_2 == 0:
            f += MAX_F
        elif s_2 == 0:
            f += s_1
        else:
            f += s_1 / s_2
    return f


def best_neighbor(state, nodes, variance):
    min_f = MAX_F
    node_index = -1
    cluster_index = -1
    counter = 0
    for i, node in enumerate(nodes):
        for c in range(N_CLUSTERS):
            counter += 1
            if c == state[i]:
                continue
            state_ = state.copy()
            state_[i] = c
            f = F(state_, nodes, variance)
            if f <= min_f:
                min_f = f
                node_index = i
                cluster_index = c
    # print(f'counter = {counter}')
    if min_f < F(state, nodes, variance):
        best_neighbor_state = state.copy()
        best_neighbor_state[node_index] = cluster_index
    else:
        best_neighbor_state = None
    return best_neighbor_state


def local_search(S, nodes, variance):
    current_state = S
    for i in range(MAX_N_LOCAL_SEARCH_EPISODES):
        print(i + 1, F(current_state, nodes, variance))

        temp_cluster_nodes = []
        for j in range(N_CLUSTERS):
            temp_nodes = [nodes[node] for node, cluster in enumerate(current_state) if cluster == j]
            temp_cluster_nodes.append(temp_nodes)
        display_cluster_nodes(temp_cluster_nodes, 'Temp Cluster-Nodes')

        next_state = best_neighbor(current_state, nodes, variance)
        if next_state is None:
            return current_state
        current_state = next_state
    return current_state


def reinforcement_learning(P, S, S_):
    P_ = P.copy()
    for i in range(N_NODES):
        if S[i] == S_[i]:
            for c in range(N_CLUSTERS):
                if c == S[i]:
                    P_[i][c] = ALPHA + (1 - ALPHA) * P[i][c]
                else:
                    P_[i][c] = (1 - ALPHA) * P[i][c]
        else:
            for c in range(N_CLUSTERS):
                if c == S[i]:
                    P_[i][c] = (1 - GAMMA) * (1 - BETA) * P[i][c]
                elif c == S_[i]:
                    P_[i][c] = GAMMA + (1 - GAMMA) * BETA / (N_CLUSTERS - 1) + (1 - GAMMA) * (1 - BETA) * P[i][c]
                else:
                    P_[i][c] = (1 - GAMMA) * BETA / (N_CLUSTERS - 1) + (1 - GAMMA) * (1 - BETA) * P[i][c]
    return P_


def probability_smoothing(P):
    P_ = P.copy()
    for i in range(N_NODES):
        max_p = max(P[i])
        cluster_index = P[i].index(max_p)
        if max_p > SMOOTHING_PROBABILITY:
            for c in range(N_CLUSTERS):
                if c == cluster_index:
                    P_[i][c] = SMOOTHING_COEFFICIENT * P[i][c]
                else:
                    P_[i][c] = (1 - SMOOTHING_COEFFICIENT) / (N_CLUSTERS - 1) * P[i][cluster_index] + P[i][c]
    return P_


def main():
    centers = create_centers()
    display_nodes(centers, 'Centers')
    nodes, initial_cluster_nodes = create_nodes(centers)
    display_cluster_nodes(initial_cluster_nodes, 'Initial Cluster-Nodes')
    P = initialize_P()
    variance = calculate_variance(nodes)
    initial_state = []
    for node in nodes:
        for c in range(N_CLUSTERS):
            if node in initial_cluster_nodes[c]:
                initial_state.append(c)
    best_S = None
    for i in range(N_EPISODES):
        print(150 * '#')
        print(f'EPISODE = {i + 1}')
        S = group_selection(P)
        print('Group Selection Is Done!')
        S_ = local_search(S, nodes, variance)
        if best_S is None:
            best_S = S_
        else:
            if F(best_S, nodes, variance) > F(S_, nodes, variance):
                best_S = S_
        print('Local Search Is Done!')
        P = reinforcement_learning(P, S, S_)
        print('Reinforcement Learning Is Done!')
        P = probability_smoothing(P)
        print('Probability Smoothing Is Done!')
    S = group_selection(P, 'greedy')
    final_cluster_nodes = []
    best_cluster_nodes = []
    for i in range(N_CLUSTERS):
        temp_nodes = [nodes[node] for node, cluster in enumerate(S) if cluster == i]
        final_cluster_nodes.append(temp_nodes)
    display_cluster_nodes(final_cluster_nodes, 'Final Cluster-Nodes')
    for i in range(N_CLUSTERS):
        temp_nodes = [nodes[node] for node, cluster in enumerate(best_S) if cluster == i]
        best_cluster_nodes.append(temp_nodes)
    display_cluster_nodes(best_cluster_nodes, 'Best Cluster-Nodes')
    print(f'Initial Cluster-Nodes F Value = {F(initial_state, nodes, variance)}')
    print(f'Final Cluster-Nodes F Value = {F(S, nodes, variance)}')
    print(f'Best Cluster-Nodes F Value = {F(best_S, nodes, variance)}')


if __name__ == '__main__':
    main()
