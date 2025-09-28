import numpy as np
import torch

from dqn import Agent as AgentVanilla, DQN as DQNVanilla, FloatTensor as FloatTensorVanilla
from dqn_softmax import Agent as AgentSoftmax, DQN as DQNSoftmax, FloatTensor as FloatTensorSoftmax
from gym_repoman.envs import CollectEnv
from wrappers import WarpFrame

import deepdish as dd

# Helper functions for each agent type
def train_vanilla(path, env):
    agent = AgentVanilla(env, path=path)
    agent.train()
    return agent

def train_softmax(path, env):
    agent = AgentSoftmax(env, path=path)
    agent.train()
    return agent

start_positions = {'crate_beige': (3, 4),
                   'player': (6, 3),
                   'circle_purple': (7, 7),
                   'circle_beige': (1, 7),
                   'crate_blue': (1, 1),
                   'crate_purple': (8, 1),
                   'circle_blue': (1, 8)}


# --- Experiment: train DQN for multiple goal conditions and collect stats ---

# Define your tasks as different goal conditions
Tasks = [
    ('purple', '', lambda x: x.colour == 'purple'),
    # ('blue', '', lambda x: x.colour == 'blue'),
    # ('', 'circle', lambda x: x.shape == 'circle'),
    # Add more (colour, shape, condition) tuples as needed
]

# --- Run experiment for both vanilla DQN and softmax DQN ---
num_runs = 1
data_stats_vanilla = np.empty((num_runs, len(Tasks)), dtype=object)
data_stats_softmax = np.empty((num_runs, len(Tasks)), dtype=object)

for i in range(num_runs):
    print("run:", i)
    for j, (colour, shape, goal_condition) in enumerate(Tasks):
        print("Task:", j)
        name = colour + shape
        base_path_vanilla = f'./models/vanilla_{name}/'
        base_path_softmax = f'./models/softmax_{name}/'
        env = WarpFrame(CollectEnv(start_positions=start_positions, goal_condition=goal_condition))

        # Vanilla DQN
        agent_vanilla = train_vanilla(base_path_vanilla, env)
        data_stats_vanilla[i, j] = agent_vanilla.training_stats
        torch.save(agent_vanilla.q_func.state_dict(), base_path_vanilla + 'model.dqn')
        # Save vanilla stats after training
        dd.io.save('exps_data/vanilla/prime_experiment_stats.h5', data_stats_vanilla)

        # Softmax DQN
        agent_softmax = train_softmax(base_path_softmax, env)
        data_stats_softmax[i, j] = agent_softmax.training_stats
        torch.save(agent_softmax.q_func.state_dict(), base_path_softmax + 'model.dqn')
        # Save softmax stats after training
        dd.io.save('exps_data/softmax/prime_experiment_stats.h5', data_stats_softmax)
