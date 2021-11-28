from mlp_data_gen import load_mlp_instance_from_fpath
from nearest_neighbor import compute_cost, solve_nearest_neighbor, solve_nearest_neighbor_softmax_temp, solve_sampling
import time

def solve_from_fpath(solve_fn, fpath):
    nodes, cost_matrix = load_mlp_instance_from_fpath(fpath)
    solution = solve_fn(cost_matrix)
    cost = compute_cost(solution, cost_matrix)
    return solution, cost


def evaluate(mode, fpath):
    solve_fn = solve_nearest_neighbor if mode == 'normal' else lambda cost: solve_sampling(solve_nearest_neighbor_softmax_temp, cost, N=5000, T=5)
    start_time = time.time()
    solution, cost = solve_from_fpath(solve_fn, fpath)
    elapsed = time.time() - start_time

    with open(fpath + f'.nn_{mode}.solution.txt', 'w') as f:
        f.write('\n'.join(map(str, solution)))
        f.write(f'\n{cost}')
        f.write(f'\n{elapsed}')

if __name__ == '__main__':
    import sys, os
    mode = sys.argv[1]
    assert mode in ['normal', 'softmax']
    fpath = sys.argv[2]
    assert os.path.exists(fpath)
    evaluate(mode, fpath)