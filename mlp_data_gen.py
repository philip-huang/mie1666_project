"""Scripts to generate symmetric and asymmetric instances of the Minimum Latency Problem.
Problem classifications are obtained from https://www.sciencedirect.com/science/article/pii/S0307904X12003459"""

import numpy as np
import os

import glob
from attention_learn_to_route.utils.data_utils import save_dataset

np.random.seed(1)

def generate_tsp_instance(num_nodes, grid_size=1):
    """Generates num_nodes points by uniform sampling on a 2D square of size grid_size.

    Returns a dict:
        nodes - a list of nodes
        costs - a dictionary of symmetric pairwise distances between every pair of nodes. 
        coords - (N, 2) matrix of x,y coordinates for each node
    """
    nodes = list(range(num_nodes))
    coord_x = np.random.rand(num_nodes) * grid_size
    coord_y = np.random.rand(num_nodes) * grid_size
    distances = {(i, j): np.hypot(coord_x[i] - coord_x[j], coord_y[i] - coord_y[j]) for i in nodes for j in nodes if i != j}
    return dict(nodes=nodes, costs=distances, coords=np.stack([coord_x, coord_y]))

def generate_mlp_instance_S1(num_nodes, grid_size=1):
    """Generates an instance of a Minumum Latency Problem with service times drawn from
    the interval [0, (max_travel_time - min_travel_time) / 2].

    Returns:
        nodes - a list of nodes which has length num_nodes + 1 (for the starting point).
            The starting point is the node at index 0.
        costs - a dictionary of assymetric edge costs for every pair of nodes.
        coords - (N, 2) matrix of x,y coordinates for each node
    """
    instance = generate_tsp_instance(num_nodes + 1, grid_size)
    distances = instance['costs']

    max_service_time = (max(distances.values()) - min(distances.values()))/2
    # there is no service time for the starting point
    service_times = [0] + list(np.random.rand(num_nodes) * max_service_time)

    instance['costs'] = {(i, j): distances[(i, j)] + service_times[i] for (i,j) in distances}

    return instance

def generate_mlp_instance_S2(num_nodes, grid_size=1):
    """Generates an instance of a Minumum Latency Problem with service times drawn from
    the interval [(max_travel_time + min_travel_time) / 2, (3max_travel_time - min_travel_time)/2].

    Returns:
        nodes - a list of nodes which has length num_nodes + 1 (for the starting point).
            The starting point is the node at index 0.
        costs - a dictionary of assymetric edge costs for every pair of nodes.
    """
    instance = generate_tsp_instance(num_nodes + 1, grid_size)
    distances = instance['costs']

    tmax, tmin = max(distances.values()), min(distances.values())
    min_service_time = (tmax + tmin)/2
    max_service_time = (3*tmax - tmin)/2
    # there is no service time for the starting point
    service_times = [0] + list(min_service_time + ( np.random.rand(num_nodes) * (max_service_time - min_service_time)))

    instance['costs'] = {(i, j): distances[(i, j)] + service_times[i] for (i,j) in distances}

    return instance

def generate_mlp_instance_S0(num_nodes, grid_size=1):
    """No service times - this is equivalent to a symmetric TSP instance."""
    return generate_tsp_instance(num_nodes + 1)

def dict_to_matrix(num_nodes, costs):
    cost_matrix = np.zeros((num_nodes, num_nodes))
    for i,j in costs:
        cost_matrix[i,j] = costs[(i, j)]
    return cost_matrix

def ensure_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def make_test_set(min_size=50, max_size=100, size_increment=50, num_per_size=25, data_path="./data"):
    """Generates test data and stores it in compressed numpy files on disk. One folder per problem type."""
    ensure_exists(data_path)
    for problem_type, problem_generator in zip(
        ['S0', 'S1', 'S2'],
        [generate_mlp_instance_S0, generate_mlp_instance_S1, generate_mlp_instance_S2]
    ):
        ptype_data_path = os.path.join(data_path, problem_type)
        ensure_exists(ptype_data_path)
        for size in range(min_size, max_size + size_increment, size_increment):
            for index in range(num_per_size):
                instance_name = f"{size}_{index}.npz"
                file_path = os.path.join(ptype_data_path, instance_name)
                instance = problem_generator(size)
                with open(file_path, "wb") as f:
                    np.savez(f,
                        nodes=instance['nodes'],
                        cost_matrix=dict_to_matrix(size+1, instance['costs']),
                        coords=instance['coords']
                    )

def load_mlp_instance_from_fpath(fpath):
    with open(fpath, 'rb') as f:
        fdict = np.load(f)
        nodes, cost_matrix, coords = fdict['nodes'], fdict['cost_matrix'], fdict['coords']
    return nodes, cost_matrix, coords


def convert_mlp_dataset_format(folder_path, graph_size, output_folder):
    ''' Example call:
        convert_mlp_dataset_format('./data/test-optimal/S0', 30,
                                   './attention_learn_to_route/data/')
    '''

    depots = []
    locs = []
    soln_costs = []

    for fpath in sorted(glob.glob(os.path.join(folder_path, "{}*.npz".format(graph_size)))):

        # Read input data
        with open(fpath, 'rb') as f:
            fdict = np.load(f)
            nodes, cost_matrix, coords = fdict['nodes'], fdict['cost_matrix'], fdict['coords']

            #print(coords)
            #print(coords.shape)

            coords = coords.T
            depots.append(coords[0].tolist())
            locs.append(coords[1:].tolist())

        # Read soln data
        soln_path = fpath + '.solution.txt'
        if os.path.exists(soln_path):
            with open(soln_path, 'r') as f:
                lines = f.readlines()
                cost = float(lines[-2].strip())
                soln_costs.append(cost)

    dataset = list(zip(depots, locs))
    save_dataset(dataset, os.path.join(output_folder, 'mlp{}_s0_test.pkl'.format(graph_size+1)))
    if len(soln_costs) != 0:
        save_dataset(soln_costs, os.path.join(output_folder, 'mlp{}_s0_test_optimcosts.pkl'.format(graph_size+1)))


if __name__ == '__main__':
    ensure_exists('./data')
    #make_test_set(data_path='./data/test-optimal')

    #nodes, cost_matrix, coords = load_mlp_instance_from_fpath('./data/test-optimal/S0/10_0.npz')

    # for converting generated data to format readable by RL code
    #convert_mlp_dataset_format('./data/test-optimal/S0', 100, './attention_learn_to_route/data/')
