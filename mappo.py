import torch as T
import torch.nn.functional as F
from torch.distributions import Categorical
from agent import Agent

from torchviz import make_dot

T.autograd.set_detect_anomaly(True)


class MAPPO:
    def __init__(self, actor_dims, n_agents, n_actions, 
                alpha=0.01, beta=0.01, fc1=64, 
                fc2=64, gamma=0.99, tau=0.01, chkpt_dir='tmp/maddpg/'):

        self.n_agents = n_agents
        self.n_actions = n_actions
        self.agents = {f"agent_{idx}":
            Agent(actor_dims[idx],  
                        n_actions, idx, alpha=alpha, beta=beta,
                        chkpt_dir=chkpt_dir) for idx in range(self.n_agents)
        }

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        for agent in self.agents.values():
            agent.save_models()

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        for agent in self.agents.values():
            agent.load_models()

    def choose_action(self, raw_obs):
        actions = {}
        log_probs = {}
        for agent_idx, agent in self.agents.items():
            action, log_prob = agent.choose_action(raw_obs[agent_idx])
            actions[agent_idx] = action
            log_probs[agent_idx] = log_prob
        return actions, log_probs

    def learn(self, memory, clip_param, ppo_epochs, mini_batch_size):
        for agent in self.agents.values():
            #print(f"agent {agent}")
            #print("***************************************")
            agent.learn(memory, clip_param=clip_param, ppo_epochs=ppo_epochs, mini_batch_size=mini_batch_size)
        
# EoF