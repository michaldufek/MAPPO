import torch as T
import numpy as np
import torch.nn.functional as F
from torch.distributions import Categorical
from networks import ActorNetwork, CriticNetwork

from tqdm import tqdm

class Agent:
    def __init__(self, actor_dims, n_actions, agent_idx, chkpt_dir, alpha=0.01, beta=0.01, gamma=0.95, entropy_c=1e-3):
        self.gamma = gamma
        self.n_actions = n_actions
        self.agent_idx = agent_idx
        self.agent_name = f'agent_{agent_idx}'
        self.entropy_coefficient = entropy_c

        self.actor = ActorNetwork(alpha, actor_dims, 64, 64, n_actions, self.agent_name + '_actor', chkpt_dir=chkpt_dir)
        self.critic = CriticNetwork(beta, 64, 64, name=self.agent_name+'_critic', chkpt_dir=chkpt_dir)

    def choose_action(self, observation):
        with T.no_grad():
            state = T.tensor(observation, dtype=T.float).unsqueeze(0).unsqueeze(0).to(self.actor.device)
            action_probs = self.actor(state)
            action_dist = Categorical(action_probs)
            action = action_dist.sample()
            probs = action_dist.log_prob(action)
        return action.item(), probs.item()

    def save_models(self):
        print("************* Saving model *************")
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def _calculate_returns_and_advantages(self, rewards, dones, values, next_values, gamma=0.99, gae_lambda=0.95):
        """
        Calculate GAE and returns.

        Parameters:
        - rewards: Tensor of shape [batch_size]
        - dones: Tensor of shape [batch_size]
        - values: Tensor of shape [batch_size]
        - next_values: Tensor of shape [batch_size]
        - gamma: Discount factor
        - gae_lambda: GAE lambda parameter

        Returns:
        - advantages: Tensor of shape [batch_size]
        - returns: Tensor of shape [batch_size]
        """
        batch_size = rewards.size(0)
        advantages = T.zeros(batch_size, dtype=T.float32, device=rewards.device)
        returns = T.zeros(batch_size, dtype=T.float32, device=rewards.device)
        
        last_gae_lam = 0
        for t in reversed(range(batch_size)):
            if t == batch_size - 1:
                next_non_terminal = 1.0 - dones[t]
                next_values_t = next_values[t]
            else:
                next_non_terminal = 1.0 - dones[t + 1]
                next_values_t = values[t + 1]
            
            delta = rewards[t] + gamma * next_values_t * next_non_terminal - values[t]
            last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
            #print(last_gae_lam)
            advantages[t] = last_gae_lam
            returns[t] = advantages[t] + values[t]

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def learn(self, memory, clip_param, ppo_epochs, mini_batch_size):
        if not memory.ready():
            print('-- Memory not ready yet')
            return

        # Assume memory collects states, actions, old log probs, rewards, next_states, dones

        actor_states, states, actions, old_log_probs, rewards, \
            actor_new_states, next_states, dones = memory.sample_buffer(agent_idx=self.agent_idx)

        # Convert to PyTorch tensors
        actor_states = T.tensor(actor_states, dtype=T.float).to(self.actor.device)
        states = T.tensor(states, dtype=T.float).to(self.actor.device)
        actions = T.tensor(actions).to(self.actor.device)
        old_log_probs = T.tensor(old_log_probs).to(self.actor.device)
        rewards = T.tensor(rewards, dtype=T.float).to(self.actor.device)
        next_states = T.tensor(next_states, dtype=T.float).to(self.actor.device)
        dones = T.tensor(dones, dtype=T.float).to(self.actor.device)

        # Calculate advantages and returns
        with T.no_grad():
            values = self.critic(states)
            next_values = self.critic(next_states)
        returns, advantages = self._calculate_returns_and_advantages(rewards, dones, values, next_values)

        # Optimize policy and value network for a number of PPO epochs
        for epoch in tqdm(range(ppo_epochs), desc='Training Progress'):
            # Get mini-batch indices
            mini_batch_indices = np.random.randint(0, len(rewards), mini_batch_size)

            # Mini-batch update
            mini_batch_states = states[mini_batch_indices]
            mini_batch_actor_states = actor_states[mini_batch_indices]
            mini_batch_actions = actions[mini_batch_indices]
            mini_batch_old_log_probs = old_log_probs[mini_batch_indices]
            mini_batch_returns = returns[mini_batch_indices]
            mini_batch_advantages = advantages[mini_batch_indices]

            # Calculate new log probabilities and state values
            dist = Categorical(self.actor(mini_batch_actor_states))
            new_log_probs = dist.log_prob(mini_batch_actions)
            state_values = self.critic(mini_batch_states)

            # Calculate policy loss
            ratios = T.exp(new_log_probs - mini_batch_old_log_probs)
            surr1 = ratios * mini_batch_advantages
            surr2 = T.clamp(ratios, 1.0 - clip_param, 1.0 + clip_param) * mini_batch_advantages
            entropy = dist.entropy()
            actor_loss = -T.min(surr1, surr2).mean()
            actor_loss -= self.entropy_coefficient * entropy.mean()

            # Calculate value loss
            critic_loss = F.mse_loss(mini_batch_returns, state_values)

            # Perform backpropagation
            self.actor.optimizer.zero_grad()
            self.critic.optimizer.zero_grad()
            actor_loss.backward(retain_graph=False)
            critic_loss.backward(retain_graph=False)
            self.actor.optimizer.step()
            self.critic.optimizer.step()
            #print(f'epoch {epoch}, loss actor loss {actor_loss}, critic loss {critic_loss}')
            tqdm.write(f'Epoch {epoch}, Actor Loss: {actor_loss.item()}, Critic Loss: {critic_loss.item()}')
# EoF