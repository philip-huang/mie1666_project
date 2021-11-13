import os
import numpy as np
import torch

from torch.utils.data import DataLoader
from generate_data import generate_vrp_data
from utils import load_model
from problems import MLP
from itertools import permutations


device = 'cuda:0'

model, _ = load_model('outputs/mlp_5/mlp5_rollout_20211111T114657/best.pt')
model = model.to(device)
seed = torch.randint(1234, (1,))[0].item()
torch.manual_seed(seed)

# toy example
#num_test = 1
#batch = {'loc': torch.Tensor([[0.1, 0.0], [0.7, 0.0], [-0.2, 0.0], [-1.3, 0.0]]).view(1, 4, 2).to(device), \
#        'depot': torch.Tensor([0.0, 0.0]).view(1, 2).to(device)}

num_test = 10000
batch = {'loc': torch.FloatTensor(num_test, 4, 2).uniform_(0, 1).to(device),\
        'depot': torch.FloatTensor(num_test, 2).uniform_(0, 1).to(device)}


ind = [1, 2, 3, 4]
perms = list(set(permutations(ind)))
print(len(perms))

# enumerate all the solutions and find the optimal solution
costs = []
for i in range(len(perms)):
    cost = MLP.get_costs(batch, torch.tensor(perms[i]).view(1, -1).repeat(num_test, 1).to(device))[0]
    costs.append(cost)
min_costs = torch.stack(costs).t().min(1)[0] 
min_costs = min_costs.view(-1, 1)
print("Optimal solution:", min_costs[:10])


model.eval()
model.set_decode_type('greedy')
rl_costs = []
with torch.no_grad():
    for _ in range(1):
        cost, log_p, pi = model(batch, return_pi=True)
        rl_costs.append(cost)
rl_costs = torch.stack(rl_costs).view(-1, 1)
print("RL solution:", rl_costs[:10])

print("Optimality gap: {:.5f}".format((((rl_costs-min_costs)/min_costs).mean()).item()))
