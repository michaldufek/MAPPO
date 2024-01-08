import numpy as np


class MultiAgentReplayBuffer:
    def __init__(self, max_size, actor_dims,
                n_actions, n_agents, batch_size):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.n_agents = n_agents
        self.actor_dims = actor_dims # [(channel, grid_size, grid_size)]*n_agents
        self.batch_size = batch_size
        self.n_actions = n_actions
        # Memory for combined state for critic
        self.state_memory = np.zeros((self.mem_size, n_agents, *actor_dims[0]))
        self.new_state_memory = np.zeros((self.mem_size, n_agents, *actor_dims[0]))
        self.reward_memory = np.zeros((self.mem_size, n_agents))
        self.terminal_memory = np.zeros((self.mem_size, n_agents), dtype=bool)

        self.init_actor_memory()

    def init_actor_memory(self):
        self.actor_state_memory = []
        self.actor_new_state_memory = []
        self.actor_action_memory = []
        self.actor_log_probs_memory = []

        for i in range(self.n_agents):
            self.actor_state_memory.append(
                np.zeros((self.mem_size, *self.actor_dims[i]))
            )
            self.actor_new_state_memory.append(
                np.zeros((self.mem_size, *self.actor_dims[i]))
            )
            self.actor_action_memory.append(
                np.zeros((self.mem_size))
            )
            self.actor_log_probs_memory.append(
                np.zeros((self.mem_size))
            )

    def store_transition(self, raw_obs, action, reward, raw_obs_, done, log_probs):
        index = self.mem_cntr % self.mem_size

        # Store each agent's observations and actions
        for agent_idx in range(self.n_agents):
            self.actor_state_memory[agent_idx][index] = raw_obs[agent_idx] # agent's obs is (channel, height, width)
            self.actor_new_state_memory[agent_idx][index] = raw_obs_[agent_idx]
            self.actor_action_memory[agent_idx][index] = action[agent_idx]
            self.actor_log_probs_memory[agent_idx][index] = log_probs[agent_idx]
        
        #print(index)
        self.state_memory[index] = raw_obs
        self.new_state_memory[index] = raw_obs_
        # Store rewards and terminal flags
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def standardize_rewards(self, rewards):
        rewards_mean = rewards.mean()
        rewards_std = rewards.std()
        standardized_rewards = (rewards - rewards_mean) / (rewards_std + 1e-8)
        return standardized_rewards

    def sample_buffer(self, agent_idx:int):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False) # random indexes for pulling batch of data

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        rewards = self.reward_memory[batch, agent_idx]
        terminal = self.terminal_memory[batch, agent_idx]

        actor_states = self.actor_state_memory[agent_idx][batch]
        actor_new_states = self.actor_new_state_memory[agent_idx][batch]
        actions = self.actor_action_memory[agent_idx][batch]
        log_probs = self.actor_log_probs_memory[agent_idx][batch]

        # Batch standardization of rewards
        rewards = self.standardize_rewards(rewards=rewards)
        return actor_states, states, actions, log_probs, rewards, actor_new_states, states_, terminal

    def ready(self):
        return self.mem_cntr >= self.batch_size

# EoF