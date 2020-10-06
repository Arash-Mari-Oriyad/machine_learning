import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import random
from itertools import combinations

N_VERTICES = 10
N_EDGES = 25
ORIGIN_VERTEX = 0
DESTINATION_VERTEX = N_VERTICES - 1
DESTINATION_VERTEX_REWARD = 100
NON_EDGE_REWARD = -100
EPSILON = 0.8
ALPHA = 0.1
GAMMA = 0.9
WALK_COUNT = 10000


def create_graph():
    vertices = list(range(N_VERTICES))
    all_edges = list(combinations(vertices, 2))
    edges = random.choices(all_edges, k=N_EDGES)
    G = nx.Graph()
    G.add_edges_from(edges)
    return G


def show_graph(G):
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos)
    plt.show()
    return


def show_q_table(q_table):
    print('q_table:')
    print(pd.DataFrame(q_table))
    return


def show_reward_table(reward_table):
    print('reward_table:')
    print(pd.DataFrame(reward_table))
    return


def show_shortest_path(shortest_path):
    print('shortest_path:')
    print(shortest_path)
    return


def create_reward_table(G):
    reward_table = np.matrix(np.zeros(shape=(N_VERTICES, N_VERTICES)))
    for x in G[DESTINATION_VERTEX]:
        reward_table[x, DESTINATION_VERTEX] = DESTINATION_VERTEX_REWARD
    return reward_table


def create_q_table(G):
    q_table = np.matrix(np.zeros(shape=(N_VERTICES, N_VERTICES)))
    q_table += NON_EDGE_REWARD
    for vertex in G.nodes:
        for x in G[vertex]:
            q_table[vertex, x] = 0
            q_table[x, vertex] = 0
    return q_table


def choose_next_vertex(G, q_table, current_vertex):
    random_value = random.uniform(0, 1)
    if random_value < EPSILON:
        next_vertices = np.where(q_table[current_vertex,] == np.max(q_table[current_vertex,]))[1]
    else:
        next_vertices = G[current_vertex]
    next_vertex = int(np.random.choice(next_vertices, 1))
    return next_vertex


def update_q_table(q_table, reward_table, current_vertex, next_vertex):
    max_index = np.where(q_table[next_vertex,] == np.max(q_table[next_vertex,]))[1]
    if max_index.shape[0] > 1:
        max_index = int(np.random.choice(max_index, 1))
    else:
        max_index = int(max_index)
    max_value = q_table[next_vertex, max_index]
    q_table[current_vertex, next_vertex] = (1 - ALPHA) * q_table[current_vertex, next_vertex] + \
                                           ALPHA * (reward_table[current_vertex, next_vertex] + GAMMA * max_value)
    return


def learn(G, q_table, reward_table):
    for i in range(WALK_COUNT):
        print('walk {0}/{1}:'.format(i+1, WALK_COUNT))
        current_vertex = np.random.randint(0, N_VERTICES)
        next_vertex = choose_next_vertex(G, q_table, current_vertex)
        update_q_table(q_table, reward_table, current_vertex, next_vertex)
    return


def find_shortest_path(q_table):
    path = [ORIGIN_VERTEX]
    next_vertex = np.argmax(q_table[ORIGIN_VERTEX,])
    path.append(next_vertex)
    while next_vertex != DESTINATION_VERTEX:
        next_vertex = np.argmax(q_table[next_vertex,])
        path.append(next_vertex)
    return path


def main():
    G = create_graph()
    show_graph(G)
    reward_table = create_reward_table(G)
    show_reward_table(reward_table)
    q_table = create_q_table(G)
    show_q_table(q_table)
    learn(G, q_table, reward_table)
    show_q_table(q_table)
    shortest_path = find_shortest_path(q_table)
    show_shortest_path(shortest_path)


if __name__ == '__main__':
    main()
