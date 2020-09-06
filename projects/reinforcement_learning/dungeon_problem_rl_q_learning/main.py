from environment import Dungeon, calculate_state_number
from agent import Agent

ACTION_SPACE = {0: 'left', 1: 'right', }
LEFT_REWARD = 10
RIGHT_REWARD = 100
ALPHA = 0.1
GAMMA = 0.95
EPSILON = 0.5
DELAY_TIME = 0.1
WALK_COUNT = 1000
EPISODE_COUNT = 1000
movements = []


def run_experiment():
    environment.reset()
    for walk in range(WALK_COUNT):
        current_state = calculate_state_number(environment.canvas.coords(environment.agent))
        print('Walk {0}/{1}'.format(walk + 1, WALK_COUNT))
        environment.render(DELAY_TIME)
        action = agent.choose_action(current_state)
        reward = environment.get_reward(action)
        next_state = calculate_state_number(environment.next_coordination(action))
        agent.learn(current_state, action, reward, next_state)
        environment.move(action)
    print('Game is finished!')
    print(agent.q_table)
    return


def total_run_experiment():
    for episode in range(EPISODE_COUNT):
        print('Episode {0}/{1}'.format(episode + 1, EPISODE_COUNT))
        temp = agent.q_table
        for current_state in temp.index.values:
            for action in temp.columns.values:
                next_state = environment.get_next_state(current_state, action)
                reward = environment.get_state_action_reward(current_state, action)
                agent.learn(current_state, action, reward, next_state)
                temp.loc[current_state, action] = (1 - agent.alpha) * agent.q_table.loc[current_state, action] + \
                                                  agent.alpha * \
                                                  (reward + agent.gamma * agent.q_table.loc[next_state, :].max())
        agent.q_table = temp
    print('Game is finished!')
    print(agent.q_table)
    return


if __name__ == '__main__':
    environment = Dungeon(action_space=ACTION_SPACE,
                          left_reward=LEFT_REWARD,
                          right_reward=RIGHT_REWARD)
    agent = Agent(n_states=environment.length,
                  action_space=ACTION_SPACE,
                  alpha=ALPHA,
                  gamma=GAMMA,
                  epsilon=EPSILON)
    environment.window.after(10, run_experiment)
    environment.window.mainloop()
    # total_run_experiment()
