import numpy as np
import itertools
from mlp_cplex import solve_model_A

def solve_mlp_brute_force(nodes, costs):
    min_cost = np.inf
    best = None
    N = len(nodes) - 1
    def compute_cost(perm):
        cost = costs[(0, perm[0])] * N
        for index, edge in enumerate(zip(perm[:-1], perm[1:])):
            cost += costs[edge] * (N - (index + 1))
        return cost

    for permutation in itertools.permutations(list(range(1, N + 1))):
        cost = compute_cost(permutation)
        if cost < min_cost:
            best = permutation
            min_cost = cost
    return list(best), min_cost

def test_mlp_model_A(fpath='data/S2/10_0.npz'):
    with open(fpath, 'rb') as f:
        fdict = np.load(f)
        nodes, cost_matrix = fdict['nodes'], fdict['cost_matrix']
        
    mlp_perm, mlp_cost = solve_model_A(nodes, cost_matrix)

    brute_force_perm, brute_force_cost = solve_mlp_brute_force(nodes, cost_matrix)
    assert np.isclose(mlp_cost, brute_force_cost)
    assert mlp_perm == brute_force_perm
