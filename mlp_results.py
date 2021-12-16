import os
import numpy as np
import glob

def load(folder_path, txt_suffix):
    for graph_size in [10, 20, 30, 50, 100]:
        print("Graph size: {}".format(graph_size))
        print("-----------------------------------------------------")
        avg_cost = []
        avg_time = []
        for fpath in glob.glob(os.path.join(folder_path, "{}_*{}".format(graph_size, txt_suffix))):
            costs = np.loadtxt(fpath)

            cost = costs[-2]
            t = costs[-1]
            avg_cost.append(cost)
            avg_time.append(t)
        print(np.mean(avg_cost))
        print(np.mean(avg_time))


if __name__ == "__main__":
    #load("./data/test-optimal/S0", txt_suffix=".rl_a.solution.txt")
    load("./data/test-optimal/S0", txt_suffix=".rl_a_gilsSol.txt")
    #load("./data/test-optimal/S2/", txt_suffix='.npz.gilsrvndSol.txt')