import numpy as np


class GridWorld:
    def __init__(self, size=5, start=(0, 0), goal=(4, 4), num_obstacles=4, seed=None):
        self.size = size
        self.start = start
        self.goal = goal
        self.num_obstacles = num_obstacles

        if seed is not None:
            np.random.seed(seed)

        self.obstacles = self._generate_obstacles()
        self.state = start

    def _generate_obstacles(self):
        obstacles = set()
        while len(obstacles) < self.num_obstacles:
            pos = (np.random.randint(0, self.size), np.random.randint(0, self.size))
            if pos != self.start and pos != self.goal:
                obstacles.add(pos)
        return list(obstacles)

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        actions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        dr, dc = actions[action]
        new_r, new_c = self.state[0] + dr, self.state[1] + dc

        if (new_r, new_c) in self.obstacles:
            reward = -1
            done = False
            return self.state, reward, done

        if 0 <= new_r < self.size and 0 <= new_c < self.size:
            self.state = (new_r, new_c)
        else:
            reward = -1
            done = False
            return self.state, reward, done

        if self.state == self.goal:
            reward = 20
            done = True
        else:
            reward = -1
            done = False

        return self.state, reward, done

    def render(self):
        grid = np.full((self.size, self.size), ".")
        for obs in self.obstacles:
            grid[obs] = "X"
        grid[self.start] = "S"
        grid[self.goal] = "G"
        grid[self.state] = "A"

        for row in grid:
            print(" ".join(row))
        print()
