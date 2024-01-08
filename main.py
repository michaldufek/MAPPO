import numpy as np
from mappo import MAPPO
from buffer import MultiAgentReplayBuffer
from env import VacuumCleanerEnv
import matplotlib.pyplot as plt

def dict_to_list(obs_dict):
    # Assuming obs_dict is ordered by agent keys, like 'agent_0', 'agent_1', etc.
    obs_list = [obs_dict[agent_id] for agent_id in sorted(obs_dict.keys())]
    return np.array(obs_list)

if __name__ == '__main__':
    #scenario = 'simple_adversary'
    env = VacuumCleanerEnv()
    n_agents = env.num_agents
    actor_dims = []
    for i in range(n_agents):
        # Assuming single-channel 2D observations, e.g., (1, 128, 128)
        actor_dims.append((1, env.grid_size, env.grid_size)) # 1 because each cell keeps just one piece of information

    # action space is a list of arrays, assume each agent has same action space
    n_actions = 5
    mappo_agents = MAPPO(actor_dims, n_agents, n_actions, 
                           fc1=64, fc2=64,  
                           alpha=0.01, beta=0.01, 
                           chkpt_dir='tmp/maddpg/')

    memory = MultiAgentReplayBuffer(512, actor_dims, 
                        n_actions, n_agents, batch_size=128)

    PRINT_INTERVAL = 500
    N_GAMES = 50000
    MAX_STEPS = 50000
    total_steps = 0
    score_history = []
    evaluate = False
    render_while_training = True
    total_score = 0
    best_score = 0

    if evaluate:
        mappo_agents.load_checkpoint()

    fig, ax = plt.subplots(figsize=(10, 10))
    #plt.show(block=False)

    for i in range(N_GAMES):
        obs_dict = env.reset()
        obs_list = np.expand_dims(dict_to_list(obs_dict), axis=1) # to have (5, 1, 128, 128)

        score = 0
        done = [False]*n_agents
        episode_step = 0
        while not any(done):
            if render_while_training or evaluate:
                env.render(fig, ax, show_grid=False)
                plt.pause(0.001)
            #print(obs)
            actions_dict, log_probs_dict = mappo_agents.choose_action(obs_dict)
            #print(actions_dict)
            obs_dict_, reward_dict, done_dict, info = env.step(actions_dict)

            obs_list_ = np.expand_dims(dict_to_list(obs_dict_), axis=1)
            actions_list = dict_to_list(actions_dict)
            log_probs_list = dict_to_list(log_probs_dict)
            reward_list = dict_to_list(reward_dict)
            done_list = dict_to_list(done_dict)

            if episode_step >= MAX_STEPS:
                done = [True]*n_agents
            memory.store_transition(obs_list, actions_list, reward_list, obs_list_, done_list, log_probs_list)

            obs_dict = obs_dict_
            obs_list = obs_list_

            score = sum(reward_list)
            total_score += score
            score_history.append(score)

            if total_steps % 25 == 0:
                print(actions_list)

            if total_steps % 100 == 0 and not evaluate:
                print("******* action list ************")
                print(actions_list)
                print("******* reward list ************")
                print(reward_list)

                print(f"best score {best_score}")
                avg_score = np.mean(score_history[-100:])
                print(f"avg score {avg_score}")
                if avg_score > best_score:
                    mappo_agents.save_checkpoint()
                    best_score = avg_score
                
                print("learning ...")
                mappo_agents.learn(memory, clip_param=0.1, ppo_epochs=8, mini_batch_size=128)

                #print(f"episode {i}, average score {score}")

            total_steps += 1
            episode_step += 1

# EoF