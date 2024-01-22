from pettingzoo.utils import wrappers, agent_selector
from pettingzoo import AECEnv
from gymnasium.spaces import Discrete, Box
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

class CleaningEnv(AECEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, num_agents=5):
        super().__init__()
        self.grid_size = 16
        #self.num_agents = 2
        self.agents = [f"hover_{i}" for i in range(num_agents)]
        #self._action_spaces = {agent: Discrete(5) for agent in self.agents}  # 4 directions + stay
        #elf._observation_spaces = {agent: Box(low=0, high=1, shape=(self.grid_size, self.grid_size, 1), dtype=np.float32) for agent in self.agents}
        self.reset()

    def observation_space(self, agent):
        return Box(low=0, high=1, shape=(self.grid_size, self.grid_size, 1))
    
    def action_space(self, agent):
        return Discrete(5)

    def reset(self):
        # Initialize the positions of the hovers
        self.positions = {f"hover_{i}": np.random.randint(0, self.grid_size, size=(2,)) for i in range(len(self.agents))}

        # Initialize dirty spots
        # For simplicity, let's randomly choose a few spots to be dirty
        self.dirty_spots = np.random.choice([0, 1], size=(self.grid_size, self.grid_size), p=[0.2, 0.8])

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        
        # Return the initial observation
        return self.observe(self.agent_selection)

    def step(self, actions):
        # Update positions for all agents based on their actions
        for agent, action in actions.items():
            possible_moves = [(0, 0), (-1, 0), (0, 1), (1, 0), (0, -1)]
            move = possible_moves[action]

            current_position = self.positions[agent]
            new_position = (current_position[0] + move[0], current_position[1] + move[1])

            # Keep the agent within boundaries
            new_position = (max(0, min(new_position[0], self.grid_size - 1)),
                            max(0, min(new_position[1], self.grid_size - 1)))

            self.positions[agent] = new_position

        # Check and update dirty spots, calculate rewards
        rewards = {}
        for agent in self.agents:
            new_position = self.positions[agent]
            reward = 0
            if self.dirty_spots[new_position]:
                self.dirty_spots[new_position] = False
                reward = 1  # Positive reward for cleaning
            rewards[agent] = reward

        # Check if the episode is done (e.g., all spots are clean)
        done = not self.dirty_spots.any()
        dones = {agent: done for agent in self.agents}

        # Prepare info and observation for each agent
        infos = {agent: {} for agent in self.agents}
        observations = {agent: self.observe(agent) for agent in self.agents}
        #print(rewards)
        return observations, rewards, dones, infos

    def observe(self, agent):
        # Create an observation grid
        obs_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

        # Mark dirty spots
        obs_grid[self.dirty_spots] = 1

        agent_values = {a: i + 2 for i, a in enumerate(self.agents)}

        for a, pos in self.positions.items():
            agent_value = agent_values[a]
            obs_grid[pos[0], pos[1]] = agent_value
        
        return obs_grid

    def render(self, fig, ax, mode='human', show_grid=False):
        # Create a color map for different entities in the grid
        colors = ['white', 'gray'] + ['C{}'.format(i) for i in range(len(self.agents))]
        cmap = mcolors.ListedColormap(colors)
        bounds = list(range(len(colors) + 1))
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        # Create a grid to display
        display_grid = np.zeros((self.grid_size, self.grid_size))

        # Mark dirty spots
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                if self.dirty_spots[row, col]:
                    display_grid[row, col] = 1  # Mark dirty spots as 1

        agent_values = {a: i+2 for i, a in enumerate(self.agents)}
        
        # Mark hovers' positions
        for hover, pos in self.positions.items():
            print(hover)
            print(pos)
            symbol = 2 if hover == hover else 3  # Differentiate hover 1 and 2
            display_grid[pos[0], pos[1]] = symbol

                # Mark all agents' positions
        for agent, pos in self.positions.items():
            agent_value = agent_values[agent]
            display_grid[pos[0], pos[1]] = agent_value

        ax.clear()
        ax.imshow(display_grid, cmap=cmap, norm=norm)

        # Redraw the canvas 
        fig.canvas.draw()
        plt.pause(0.1)

    def close(self):
        # Clean up the environment
        pass

if __name__ == "__main__":
    env = CleaningEnv()
    obs = env.reset()

    fig, ax = plt.subplots(figsize=(8, 8))

    while env.agents:
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        env.step(actions)
        env.render(fig, ax)
