from environment import Maze, calculate_state_number
from agent import Agent
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')

ACTION_SPACE = {0: 'up', 1: 'down', 2: 'right', 3: 'left'}
PIT_REWARD = -100
WALL_REWARD = -10
DESTINATION_REWARD = 1000
ALPHA = 0.01
GAMMA = 0.9
EPSILON = 0.8
DELAY_TIME = 0.1
EPISODE_COUNT = 200
movements = []


def run_experiment():
    for episode in range(EPISODE_COUNT):
        print('Episode {0}/{1}'.format(episode + 1, EPISODE_COUNT))
        environment.reset()
        n_movements = 0
        while True:
            current_state = calculate_state_number(environment.canvas.coords(environment.agent))
            environment.render(DELAY_TIME)
            action = agent.choose_action(current_state)
            reward = environment.get_reward(action, 0)
            next_state = calculate_state_number(environment.next_coordination(action))
            agent.learn(current_state, action, reward, next_state)
            environment.move(action)
            n_movements += 1
            if environment.is_finished():
                movements.append(n_movements)
                print('n_movements: {0}'.format(n_movements))
                break
    print('Game is finished!')
    print(agent.q_table)
    plot_rewards_movements()
    return


def plot_rewards_movements():
    plt.figure()
    plt.plot(list(range(EPISODE_COUNT)), movements)
    plt.xlabel('Episode')
    plt.ylabel('#Movements')
    plt.show()
    return


if __name__ == '__main__':
    environment = Maze(action_space=ACTION_SPACE,
                       pit_reward=PIT_REWARD,
                       destination_reward=DESTINATION_REWARD,
                       wall_reward=WALL_REWARD)
    agent = Agent(n_states=environment.width * environment.height,
                  action_space=ACTION_SPACE,
                  alpha=ALPHA,
                  gamma=GAMMA,
                  epsilon=EPSILON)
    environment.window.after(10, run_experiment)
    environment.window.mainloop()
