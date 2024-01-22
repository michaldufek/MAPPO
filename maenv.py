from pettingzoo import AECEnv
from pettingzoo.utils import wrappers

import gymnasium as gym
from gymnasium.spaces import Discrete, Box
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

class VacuumCleanerEnv(AECEnv):
    """
    Environment Representation:
    ---------------------------
        0: Clean space
        1: Dirty space
        -1: Obstacle
        2: "Point of view" agent
        3: Other agents in environment
    """
    #metadata = {'render.modes': ['human']}

    def __init__(self):
        super(VacuumCleanerEnv, self).__init__()
        self.num_agents = 1
        self.grid_size = 32
        self.max_cycles = 10000 #max_cycles
        self.cycle_count = 0
        self.agents = [f'agent_{i}' for i in range(self.num_agents)]
        self.grid = np.ones((self.grid_size, self.grid_size))
        self.valid_moves = {f'agent_{i}': True for i in range(self.num_agents) }

        self.action_space = Discrete(5)
        self.observation_space = Box(low=0, high=3, shape=(self.grid_size, self.grid_size), dtype=np.float32)
        self.action_spaces = {f"agent_{i}": Discrete(5) for i in range(self.num_agents)}
        self.observation_spaces = {f"agent_{i}": Box(low=0, high=1, shape=(self.grid_size, self.grid_size), dtype=np.float32) for i in range(self.num_agents)}
        self.agent_size = 1 # size of the agent (agent occupies n x n cells)
        self.obstacle_size = 5
        self.num_obstacles = 6
        self.dirt_size = 3
        self.num_dirt = 25

    def _check_space_is_free(self, x, y, n):
        # Check if an NxN area is free of obstacles for agent initialization
        for i in range(x, x + n):
            for j in range(y, y + n):
                if self._is_obstacle(i, j):  # Assuming obstacles are already initialized
                    return False
        return True
    
    def _initialize_agents(self):
        self.agent_positions = {}
        for agent_id in self.agents:
            robots_placed = False
            while not robots_placed:
                x, y = (np.random.randint(0, self.grid_size - self.agent_size + 1),
                        np.random.randint(0, self.grid_size - self.agent_size + 1))
                if self._check_space_is_free(x, y, self.agent_size):
                    self.agent_positions[agent_id] = (x, y)
                    robots_placed = True

    def _initialize_obstacles(self, obstacle_size, num_obstacles):
        #self.grid[0]
        # Initialize walls in the room
        for index in [0, -1]:
            self.grid[index, :] = -1
            self.grid[:, index] = -1
        for _ in range(num_obstacles):
            obstacles_placed = False
            while not obstacles_placed:
                x, y = (np.random.randint(0, self.grid_size - obstacle_size + 1),
                        np.random.randint(0, self.grid_size - obstacle_size + 1))
                if self._check_space_is_free(x, y, obstacle_size):
                    # Place the obstacle
                    for i in range(x, x + obstacle_size):
                        for j in range(y, y + obstacle_size):
                            self.grid[i, j] = -1  # Marking the obstacle
                    obstacles_placed = True

    def _initialize_dirt(self, dirt_size, num_dirt):
        for _ in range(num_dirt):
            dirt_placed = False
            while not dirt_placed:
                # Place garbage
                x, y = (np.random.randint(0, self.grid_size - dirt_size + 1),
                        np.random.randint(0, self.grid_size - self.dirt_size + 1))
                if self._check_space_is_free(x, y, dirt_size):
                    # Place dirty places
                    for i in range(x, x + dirt_size):
                        for j in range(y, y + dirt_size):
                            self.grid[i, j] = 1 # Marking as dirt
                    dirt_placed = True
 
    def reset(self, **kwargs):
        self.cycle_count = 0
        # Initialize a clean grid
        self.grid = np.zeros((self.grid_size, self.grid_size))

        # Reset the list of agents
        self.agents = [f"agent_{i}" for i in range(self.num_agents)]

        # Randomly place dirt
        num_dirt_cells = int(self.grid_size * self.grid_size * 0.5)  # Example: N% of the grid
        dirt_positions = np.random.choice(self.grid_size * self.grid_size, num_dirt_cells, replace=False)
        for pos in dirt_positions:
            x, y = pos // self.grid_size, pos % self.grid_size
            self.grid[x, y] = 1  # 1 indicates dirt

        # Initialize agents' positions
        self._initialize_agents()

        # Initialize obstacles
        self._initialize_obstacles(obstacle_size=self.obstacle_size, num_obstacles=self.num_obstacles)

        # Initialize dirt
        self._initialize_dirt(dirt_size=self.dirt_size, num_dirt=self.num_dirt)

        return self.observe(), {}

    def _get_new_position(self, x, y, action, n):
        # Adjust bounds to prevent the NxN agent from going out of bounds
        if action == 1:  # up
            new_x = max(x - 1, 0)
            new_y = y
        elif action == 2:  # down
            new_x = min(x + 1, self.grid_size - n)
            new_y = y
        elif action == 3:  # left
            new_x = x
            new_y = max(y - 1, 0)
        elif action == 4:  # right
            new_x = x
            new_y = min(y + 1, self.grid_size - n)
        else:  # stay
            new_x = x
            new_y = y
        return new_x, new_y

    def _clean_area_around(self, x, y, n):
        for i in range(x, x + n):
            for j in range(y, y + n):
                if 0 <= i < self.grid_size and 0 <= j < self.grid_size and not self._is_obstacle(i, j):
                    self.grid[i, j] = 0  # Cleaning the cell

    def step(self, actions):
        print("*********************************************")
        print("************ S T E P**************************")
        self.cycle_count += 1
        rewards = {agent_id:0 for agent_id in self.agents}

        for agent_id, action in actions.items():
            x, y = self.agent_positions[agent_id]  # Get the current position
            action = actions[agent_id] # Take the action of the current agent

            # Get the new position based on the agent's action
            new_x, new_y = self._get_new_position(x, y, action, self.agent_size)

            # Move the agent if the new position is valid
            if self._check_new_position_is_valid(new_x, new_y, self.agent_size):
                # Set a flag for current move as valid -- important for further reward calculation
                self.valid_moves[agent_id] = True
                self.agent_positions[agent_id] = (new_x, new_y)
            else:
                self.valid_moves[agent_id] = False # assign the move as invalid for reward calculation
            
            reward = self._calculate_rewards(action, agent_id)
            rewards[agent_id] = reward

            self._clean_area_around(new_x, new_y, self.agent_size)

        observations = self.observe()
        dones = {agent: self.cycle_count >= self.max_cycles for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        return observations, rewards, dones, infos

    def _check_new_position_is_valid(self, x, y, n):
        for i in range(x, x + n):
            for j in range(y, y + n):
                if i >= self.grid_size or j >= self.grid_size or self._is_obstacle(i, j):
                    return False  # Invalid position if out of bounds or on an obstacle
        return True

    def observe(self):
        observations = {}
        for agent_id in self.agents:
            # Create an observation that includes the grid and the agent's position
            obs = np.copy(self.grid)
            
            # Mark the position of the current agent
            agent_x, agent_y = self.agent_positions[agent_id]
            obs[agent_x, agent_y] = 2

            # Mark positions of other agents
            for other_agent_id, pos in self.agent_positions.items():
                if other_agent_id != agent_id:
                    other_agent_x, other_agent_y = pos
                    obs[other_agent_x, other_agent_y] = 3

            observations[agent_id] = obs
        return observations
    
    def _is_obstacle(self, x, y):
        return self.grid[x, y] == -1 # Assuming that obstacle is represented as -1

    def _calculate_rewards(self, action:int, agent_id:str):
        #rewards = {agent_id: 0 for agent_id in self.agents}
        step_penalty = -0.1  # Reduced penalty for each step
        cleaning_reward = 1
        obstacle_penalty = -1  # penalty for hitting an obstacle
        redundant_cleaning_penalty = -0.1

        reward = step_penalty  # Start with no penalty

        x, y = self.agent_positions[agent_id]

        # Reward for cleaning a dirty cell
        if self.grid[x, y] == 1: # prevent receiving reward while staying at the walls
            reward += cleaning_reward

        if self.grid[x, y] == 0:
                reward += redundant_cleaning_penalty

        if self.valid_moves[agent_id] == False:
            reward += obstacle_penalty

        # Small penalty for staying still (which is an intentional lack of movement)
        if action == 0:
            reward += -1

            #rewards[agent_id] += reward

        return reward

    def render(self, fig, ax, mode='human', show_grid=False):
        # Create a color map for rendering
        cmap = mcolors.ListedColormap(['red', 'white', 'grey', 'blue'])
        bounds = [-1, 0, 1, 2, 3]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        # Create a render grid
        render_grid = np.copy(self.grid)

        # Reset the cells to clean for rendering purposes
        # If the cell is not dirty (represented by 1), set it to clean (0)
        render_grid[render_grid > 1] = 0

        # Mark the current positions of the robots with their coverage area
        n = self.agent_size  # The coverage area of the robots, NxN
        for agent_id, (x, y) in self.agent_positions.items():
            agent_marker = 3 if agent_id == 'agent_0' else 4  # Markers for the robots
            for i in range(x - n // 2, x + (n + 1) // 2):
                for j in range(y - n // 2, y + (n + 1) // 2):
                    if 0 <= i < self.grid_size and 0 <= j < self.grid_size:  # Check grid boundaries
                        render_grid[i, j] = agent_marker

        # Update the plot
        ax.clear()
        ax.imshow(render_grid, cmap=cmap, norm=norm)

        if show_grid:
            ax.set_xticks(np.arange(-.5, self.grid_size, 1), minor=True)
            ax.set_yticks(np.arange(-.5, self.grid_size, 1), minor=True)
            ax.grid(which="minor", color="black", linestyle='-', linewidth=0.1)
            ax.tick_params(which="major", bottom=False, left=False, labelbottom=False, labelleft=False)
        else:
            # Hide grid lines
            ax.grid(False)
        # Redraw the canvas
        fig.canvas.draw()

    def close(self):
        # Close any resources if necessary
        pass

if __name__=="__main__":
    # Create the environment
    env = VacuumCleanerEnv() #env_creator()

    # Reset the environment to start a new episode
    observations = env.reset()

    fig, ax = plt.subplots(figsize=(10, 10))

    for i in range(10000):  # Simulate n steps as an example
        # Sample a random action for each agent
        actions = {agent: env.action_spaces[agent].sample() for agent in env.agents}

        # Perform a step in the environment with the sampled actions
        observations, rewards, dones, infos = env.step(actions)

        # Optional: Print observations, rewards, etc. for debugging
        #print("Observations:", observations)
        #print(f"{i}: Rewards: {rewards}")
        #print("Dones:", dones)
        #env.render()

        # Mark the positions of the agents
        for agent_id, (x, y) in env.agent_positions.items():
            agent_marker = int(agent_id.split('_')[-1]) + 3  # Assign a unique value for each agent
            env.grid[x, y] = agent_marker

        # Check if the episode is done for any agent
        if any(dones.values()):
            break
        
        # Render the current state of the environment
        env.render(fig, ax, show_grid=False)
        plt.pause(0.01)

    plt.show()