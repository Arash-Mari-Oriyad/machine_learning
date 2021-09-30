import numpy as np
# from scipy import stats
import random
import matplotlib.pyplot as plt

N_ACTIONS = 10
MAX_VALUE = 10
EPSILON_GREEDY = None
N_EPISODES = 1000


def get_reward(action, probs):
    reward = 0
    for i in range(MAX_VALUE):
        if random.random() <= probs[action]:
            reward += 1
    return reward


def get_best_arm(record):
    return np.argmax(record[:, 1], axis=0)


def update_record(record, action, reward):
    new_reward = (record[action, 0] * record[action, 1] + reward) / (record[action, 0] + 1)
    record[action, 0] += 1
    record[action, 1] = new_reward
    return record


def main():
    probs = np.random.rand(N_ACTIONS)
    record = np.zeros((N_ACTIONS, 2))
    rewards = [0]
    for i in range(N_EPISODES):
        EPSILON_GREEDY = (1 - i / N_EPISODES) * 0.3
        print(f'episode={i + 1}')
        print(EPSILON_GREEDY)
        if random.random() >= EPSILON_GREEDY:
            action = get_best_arm(record)
        else:
            action = np.random.randint(N_ACTIONS)
        reward = get_reward(action, probs)
        record = update_record(record, action, reward)
        rewards.append(((i + 1) * rewards[-1] + reward) / (i + 2))
    print(100 * '#')
    for i, prob in enumerate(probs):
        print(i, prob)
    print(100 * '#')
    for i, re in enumerate(record):
        print(i, re[0], re[1])
    print(100 * '#')
    print((N_EPISODES * np.max(probs) * MAX_VALUE) - (rewards[-1] * (len(rewards) - 1)))
    # fig, ax = plt.subplots(1, 1)
    # ax.set_xlabel("Plays")
    # ax.set_ylabel("Avg Reward")
    # fig.set_size_inches(9, 5)
    # ax.scatter(np.arange(len(rewards)), rewards)
    return


if __name__ == '__main__':
    main()
