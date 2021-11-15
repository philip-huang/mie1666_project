"""Integer programming formulation in CPLEX for the Minimum Latency Problem"""

import numpy as np
from docplex.mp.model import Model

def create_model_A(nodes, costs):
    """
    Input:
        nodes: a list of node identifiers. The first node is assumed to be the "source".
        costs: a dictionary of costs for each pair of distinct nodes.
    """
    # setup
    num_destinations = len(nodes) - 1  # nodes also includes the "source node"
    arc_layers = {}
    for k in range(1, num_destinations): # position in permutation
        for i in range(1, num_destinations + 1): # source
            for j in range(1, num_destinations + 1): # destination
                if i == j:
                    continue
                arc_layers[(i, j, k)] = costs[(i, j)] * (num_destinations - k)

    choice_at_level = {(i, k) for i in range(1, num_destinations + 1) for k in range(1, num_destinations + 1)}

    mdl=Model('MLP')

    # decision variables
    x = mdl.binary_var_dict(choice_at_level,name='x')
    y = mdl.continuous_var_dict(arc_layers,name='y')

    # objective
    z = num_destinations * mdl.sum(costs[(0, i)] * x[(i, 1)] for i in range(1, num_destinations+1)) + \
         mdl.sum(arc_layers[i] * y[i] for i in arc_layers)
    mdl.minimize(z)

    # constraints
    # only one node can be visited in a particular layer (i.e. index in the path)
    for k in range(1, num_destinations + 1):
        mdl.add_constraint(
            mdl.sum(x[(i,k)] for i in range(1, num_destinations + 1)) == 1
        )
    # every node has to visited in a exactly one position
    for i in range(1, num_destinations + 1):
        mdl.add_constraint(
            mdl.sum(x[(i,k)] for k in range(1, num_destinations + 1)) == 1
        )
    # a node selected in a particular layer must have an outgoing edge to some other node
    for i in range(1, num_destinations + 1):
        for k in range(1, num_destinations):
            mdl.add_constraint(
                mdl.sum(y[(i, j, k)] for j in range(1, num_destinations + 1) if j != i) == x[(i, k)]
            )
    # a node selected in a particular layer must have an incoming edge from some other node
    for j in range(1, num_destinations + 1):
        for k in range(1, num_destinations):
            mdl.add_constraint(
                mdl.sum(y[(i, j, k)] for i in range(1, num_destinations + 1) if j != i) == x[(j, k+1)]
            )

    return mdl

def parse_solution_model_A(solution):
    solution_values = solution.as_name_dict()
    permutation = []
    for var_name in solution_values:
        if var_name.startswith('x'):
            _, node, index = var_name.split('_')
            permutation.append((index, node))
    _, nodes = zip(*sorted(permutation))
    return nodes


def solve_model_A(nodes, costs, timelimit=120):
    model = create_model_A(nodes, costs)
    model.parameters.timelimit=timelimit
    # accept approximations.
    model.parameters.mip.tolerances.mipgap=0

    solution = model.solve(log_output=True)
    permutation = parse_solution_model_A(nodes, solution)
    value = solution.objective_value

    return (permutation, value)

if __name__ == '__main__':
    with open('data/test/S0/10_0.npz', 'rb') as f:
        fdict = np.load(f)
        nodes, cost_matrix = fdict['nodes'], fdict['cost_matrix']
        
    print(solve_model_A(nodes, cost_matrix))