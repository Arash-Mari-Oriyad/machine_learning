import numpy as np
import pandas as pd


class Agent:
    def __init__(self, n_states, action_space, alpha, gamma, epsilon):
        self.n_states = n_states
        self.action_space = action_space
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = pd.DataFrame(data=np.zeros((n_states, len(action_space.keys()))),
                                    columns=list(action_space.keys()),
                                    dtype=np.float64)
        return

    def choose_action(self, state):
        if np.random.uniform() > self.epsilon:
            action = np.random.choice(list(self.action_space.keys()))
        else:
            action_values = self.q_table.loc[state, :]
            action_values = action_values.reindex(np.random.permutation(action_values.index))
            action = action_values.idxmax()
        return action

    def learn(self, current_state, action, reward, next_state):
        self.q_table.loc[current_state, action] = (1 - self.alpha) * self.q_table.loc[current_state, action] + \
                                                  self.alpha * (reward + self.gamma *
                                                                self.q_table.loc[next_state, :].max())
        return


if __name__ == '__main__':
    agent = Agent(n_states=6 * 6,
                  action_space={0: 'up', 1: 'down', 2: 'right', 3: 'left'},
                  alpha=0.1, gamma=0.9, epsilon=0.5)
    agent.learn(0, 0, -1000, 20)
    print(agent.q_table)
