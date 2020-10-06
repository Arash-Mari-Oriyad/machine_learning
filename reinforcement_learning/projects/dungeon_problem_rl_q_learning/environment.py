import numpy as np
import time
import tkinter as tk

UNIT = 100
DUNGEON_LENGTH = 5
ORIGIN = np.array([UNIT / 2, UNIT / 2])


def calculate_state_number(coordination):
    coordination = ((coordination[0] + coordination[2]) / 2, (coordination[1] + coordination[3]) / 2)
    coordination = (coordination[0] // UNIT, coordination[1] // UNIT)
    return int(coordination[0] + coordination[1] * DUNGEON_LENGTH)


class Dungeon:
    def __init__(self, action_space, left_reward, right_reward):
        self.n_walk = 0
        self.length = DUNGEON_LENGTH
        self.window = tk.Tk()
        self.window.title('DUNGEON')
        self.window.geometry('{0}x{1}'.format(DUNGEON_LENGTH * UNIT, 1 * UNIT))
        self.canvas = None
        self.agent = None
        self.action_space = action_space
        self.left_reward = left_reward
        self.right_reward = right_reward
        self.build()
        return

    def build(self):
        self.canvas = tk.Canvas(master=self.window, bg='white', width=DUNGEON_LENGTH * UNIT, height=1 * UNIT)
        for c in range(0, DUNGEON_LENGTH * UNIT, UNIT):
            self.canvas.create_line(c, 0, c, DUNGEON_LENGTH * UNIT)
        self.reset()
        self.canvas.pack()
        return

    def render(self, delay_time):
        time.sleep(delay_time)
        self.window.update()
        return

    def reset(self):
        self.window.update()
        time.sleep(0.5)
        self.canvas.delete(self.agent)
        self.agent = self.canvas.create_rectangle(
            ORIGIN[0] - 1 / 4 * UNIT, ORIGIN[1] - 1 / 4 * UNIT,
            ORIGIN[0] + 1 / 4 * UNIT, ORIGIN[1] + 1 / 4 * UNIT,
            fill='red'
        )
        return

    def next_coordination(self, action):
        current_agent_coordination = self.canvas.coords(self.agent)
        base_action = [0, 0]
        if self.action_space[action] == 'left':
            next_agent_coordination = [UNIT/4, UNIT/4, 3*UNIT/4, 3*UNIT/4]
        else:
            base_action[0] += UNIT if current_agent_coordination[0] < (DUNGEON_LENGTH - 1) * UNIT else 0
            next_agent_coordination = [sum(x) for x in zip(current_agent_coordination, 2 * base_action)]
        return next_agent_coordination

    def move(self, action):
        current_agent_coordination = self.canvas.coords(self.agent)
        next_agent_coordination = self.next_coordination(action)
        base_action = [int(next_agent_coordination[i] - current_agent_coordination[i]) for i in range(2)]
        self.canvas.move(self.agent, base_action[0], base_action[1])
        self.n_walk += 1
        return

    def get_reward(self, action):
        if self.action_space[action] == 'left':
            reward = self.left_reward
        else:
            next_coordination = self.next_coordination(action)
            next_state = calculate_state_number(next_coordination)
            if next_state == self.length - 1:
                reward = self.right_reward
            else:
                reward = 0
        return reward

    def get_next_state(self, current_state, action):
        if self.action_space[action] == 'left':
            return 0
        return current_state + 1 if current_state < DUNGEON_LENGTH - 1 else current_state

    def get_state_action_reward(self, current_state, action):
        if self.action_space[action] == 'left':
            return self.left_reward
        if self.get_next_state(current_state, action) == self.length - 1:
            return self.right_reward
        return 0


if __name__ == '__main__':
    dungeon = Dungeon(action_space={0: 'left', 1: 'right'},
                      left_reward=10,
                      right_reward=100)
    print(dungeon.get_next_state(2, 0))
    dungeon.window.mainloop()
