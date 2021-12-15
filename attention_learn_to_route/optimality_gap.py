import argparse
import pickle
import numpy as np
import os


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

def write_tour_to_txt(rl_results_path, txt_path):
    with open(rl_results_path, 'rb') as f:
        results, eval_batch_size = pickle.load(f)
        rl_costs, tours, durations = zip(*results)

        rl_costs = np.array(rl_costs)

        ids = sorted([str(i) for i in range(len(tours))])
        for i in range(len(tours)):
            size = len(tours[i])
            path = os.path.join(txt_path, f'{size}_{ids[i]}.npz.rl_a.solution.txt')
            print(path)
            with open(path, 'w') as f:
                for node in tours[i]:
                    f.write('%d\n' % node)
                f.write('%f\n' % rl_costs[i])
                f.write('%f\n' % durations[i])

        print("Tour written to txt files.")

if __name__ == "__main__":

    ''' Example command:
        python optimality_gap.py --rl_results_path
        results/mlp/mlp21_s0_test/mlp21_s0_test-mlp_21_rollout_20211126T154811_best-greedy-t1-0-25.pkl
        --optim_results_path data/mlp21_s0_test_optimcosts.pkl
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--rl_results_path', help="path to rl eval results", required=True)
    parser.add_argument('--optim_results_path', help="path to optimal cost results", required=False)
    parser.add_argument('--txt_path', help="path to write tour to txt files", required=False)

    args = parser.parse_args()

    if args.optim_results_path:
        compute_optimality_gap(args.rl_results_path, args.optim_results_path)

    if args.txt_path:
        write_tour_to_txt(args.rl_results_path, args.txt_path)
