import argparse
import pickle
import numpy as np


def compute_optimality_gap(rl_results_path, optim_results_path):

    with open(optim_results_path, 'rb') as f:
        optim_costs = pickle.load(f)

    with open(rl_results_path, 'rb') as f:
       results, eval_batch_size = pickle.load(f)
    rl_costs, tours, durations = zip(*results)

    optim_costs = np.array(optim_costs)
    rl_costs = np.array(rl_costs)

    #print(optim_costs)
    #print(rl_costs)

    avg_optim_gap = ((rl_costs-optim_costs)/optim_costs).mean()
    print("Optimality gap: {:.5f}".format(avg_optim_gap))


if __name__ == "__main__":

    ''' Example command:
        python optimality_gap.py --rl_results_path
        results/mlp/mlp21_s0_test/mlp21_s0_test-mlp_21_rollout_20211126T154811_best-greedy-t1-0-25.pkl
        --optim_results_path data/mlp21_s0_test_optimcosts.pkl
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--rl_results_path', help="path to rl eval results", required=True)
    parser.add_argument('--optim_results_path', help="path to optimal cost results", required=True)

    args = parser.parse_args()

    compute_optimality_gap(args.rl_results_path, args.optim_results_path)

