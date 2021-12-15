import numpy as np
import os
import torch
from problems.mlp.problem_mlp import MLP_S1, MLPDataset_S1
import pickle
import ipdb

with open('../data/test-optimal/S1/10_0.npz', 'rb') as f:
    fdict = np.load(f)
    nodes, cost_matrix, coords = fdict['nodes'], fdict['cost_matrix'], fdict['coords']

costs = cost_matrix

def make_instance_directed(args):
    depot, loc, service_time = args
    return {
        'loc': torch.tensor(loc, dtype=torch.float).view(1, 10, 2),
        'depot': torch.tensor(depot, dtype=torch.float).view(1, 2),
        'service_time': torch.tensor(service_time, dtype=torch.float).view(1, -1)
    }

with open('./data/mlp11_s1_test.pkl', 'rb') as f:
    data = pickle.load(f)
ipdb.set_trace()
#dataset = MLP_S1.make_dataset(filename='./data/mlp11_s0_test_optimcosts.pkl', num_samples=1, offset=0)
dataset = [make_instance_directed(data[0])]
ipdb.set_trace()

#dataset = MLPDataset_S1(size=11, num_samples=1)
#ipdb.set_trace()

pi = torch.tensor([8, 2, 4, 10, 7, 3, 9, 6, 1, 5]).view(1, -1)
rl_costs  = MLP_S1.get_costs(dataset[0], pi)
ipdb.set_trace()


def compute_cost(perm):
    cost = costs[(0, perm[0])] * 10
    for index, edge in enumerate(zip(perm[:-1], perm[1:])):
        cost += costs[edge] * (10 - (index + 1))
    return cost

cplex_costs = compute_cost([8, 2, 4, 10, 7, 3, 9, 6, 1, 5])
print(rl_costs)
print(cplex_costs)
