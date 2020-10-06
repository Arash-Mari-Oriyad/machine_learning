import numpy as np
import time
import tkinter as tk

UNIT = 100
MAZE_HEIGHT = 6
MAZE_WIDTH = 6
ORIGIN = np.array([UNIT / 2, UNIT / 2])
PITS_COORDINATION = [(2, 1), (1, 2), (4, 3), (3,3)]
DESTINATION_COORDINATION = (5, 3)


def calculate_state_number(coordination):
    coordination = ((coordination[0] + coordination[2]) / 2, (coordination[1] + coordination[3]) / 2)
    coordination = (coordination[0] // UNIT, coordination[1] // UNIT)
    return int(coordination[0] + coordination[1] * MAZE_WIDTH)


class Maze:
    def __init__(self, action_space, pit_reward, wall_reward, destination_reward):
        self.width = MAZE_WIDTH
        self.height = MAZE_HEIGHT
        self.window = tk.Tk()
        self.window.title('MAZE')
        self.window.geometry('{0}x{1}'.format(MAZE_WIDTH * UNIT, MAZE_HEIGHT * UNIT))
        self.canvas = None
        self.pits = []
        self.destination = None
        self.agent = None
        self.action_space = action_space
        self.pit_reward = pit_reward
        self.wall_reward = wall_reward
        self.destination_reward = destination_reward
        self.build()
        return

    def build(self):
        self.canvas = tk.Canvas(master=self.window, bg='white', width=MAZE_WIDTH * UNIT, height=MAZE_HEIGHT * UNIT)
        for c in range(0, MAZE_WIDTH * UNIT, UNIT):
            self.canvas.create_line(c, 0, c, MAZE_WIDTH * UNIT)
        for r in range(0, MAZE_HEIGHT * UNIT, UNIT):
            self.canvas.create_line(0, r, MAZE_HEIGHT * UNIT, r)
        self.reset()
        for pit_coordination in PITS_COORDINATION:
            pit_center = ORIGIN + np.array([pit_coordination[0] * UNIT, pit_coordination[1] * UNIT])
            self.pits.append(self.canvas.create_rectangle(
                pit_center[0] - 1 / 3 * UNIT, pit_center[1] - 1 / 3 * UNIT,
                pit_center[0] + 1 / 3 * UNIT, pit_center[1] + 1 / 3 * UNIT,
                fill='black'
            ))
        destination_center = ORIGIN + np.array([DESTINATION_COORDINATION[0] * UNIT, DESTINATION_COORDINATION[1] * UNIT])
        self.destination = self.canvas.create_oval(
            destination_center[0] - 1 / 3 * UNIT, destination_center[1] - 1 / 3 * UNIT,
            destination_center[0] + 1 / 3 * UNIT, destination_center[1] + 1 / 3 * UNIT,
            fill='yellow'
        )
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
        # self.canvas.create_text(ORIGIN[0], ORIGIN[1], fill="green", font="Times 20 italic bold",
        #                         text="3")
        # return self.calculate_state_number(self.canvas.coords(self.agent))
        return

    def next_coordination(self, action):
        current_agent_coordination = self.canvas.coords(self.agent)
        base_action = [0, 0]
        if self.action_space[action] == 'up':
            base_action[1] -= UNIT if current_agent_coordination[1] > UNIT else 0
        if self.action_space[action] == 'down':
            base_action[1] += UNIT if current_agent_coordination[1] < (MAZE_HEIGHT - 1) * UNIT else 0
        if self.action_space[action] == 'right':
            base_action[0] += UNIT if current_agent_coordination[0] < (MAZE_WIDTH - 1) * UNIT else 0
        if self.action_space[action] == 'left':
            base_action[0] -= UNIT if current_agent_coordination[0] > UNIT else 0
        next_agent_coordination = [sum(x) for x in zip(current_agent_coordination, 2 * base_action)]
        return next_agent_coordination

    def move(self, action):
        current_agent_coordination = self.canvas.coords(self.agent)
        next_agent_coordination = self.next_coordination(action)
        base_action = [int(next_agent_coordination[i] - current_agent_coordination[i]) for i in range(2)]
        self.canvas.move(self.agent, base_action[0], base_action[1])
        return

    def get_reward(self, action):
        current_coordination = self.canvas.coords(self.agent)
        current_state = calculate_state_number(current_coordination)
        next_coordination = self.next_coordination(action)
        next_state = calculate_state_number(next_coordination)
        reward = 0
        if current_state == next_state:
            reward = self.wall_reward
        elif next_state == calculate_state_number(self.canvas.coords(self.destination)):
            reward = self.destination_reward
        elif next_state in [calculate_state_number(self.canvas.coords(pit)) for pit in self.pits]:
            reward = self.pit_reward
        return reward

    def is_finished(self):
        current_coordination = self.canvas.coords(self.agent)
        if calculate_state_number(current_coordination) == calculate_state_number(self.canvas.coords(self.destination)) \
                or calculate_state_number(current_coordination) in [calculate_state_number(self.canvas.coords(pit))
                                                                    for pit in self.pits]:
            return True
        return False


if __name__ == '__main__':
    maze = Maze(action_space={0: 'up', 1: 'down', 2: 'right', 3: 'left'},
                pit_reward=-1,
                destination_reward=1,
                wall_reward=-1)
    maze.window.mainloop()
