import numpy as np
import time

def get_cost_matrix(N, costs):
    matrix = np.ones((N+1, N+1)) * np.inf
    for (i,j) in costs:
        matrix[i,j] = costs[(i,j)]
    return matrix

def compute_cost(perm, cost_matrix):
    N = len(perm)
    cost = cost_matrix[(0, perm[0])] * N
    for index, edge in enumerate(zip(perm[:-1], perm[1:])):
        cost += cost_matrix[edge] * (N - (index + 1))
    return cost

def solve_nearest_neighbor(cost_matrix):
    cost_matrix = cost_matrix.copy()
    sol = []
    node = 0

    for i in range(len(cost_matrix) - 1):
        node = np.argmin(cost_matrix[node,1:]) + 1
        sol.append(node)
        cost_matrix[:,node] = np.inf
    return sol

def solve_nearest_neighbor_epsilon_greedy(cost_matrix, epsilon=1e-3):
    cost_matrix = cost_matrix.copy()
    sol = []
    node = 0
    nodes_remaining = set(range(1, len(cost_matrix)))
    for i in range(len(cost_matrix) - 1):
        if np.random.random() <= epsilon:
            node = np.random.choice(list(nodes_remaining))
        else:
            node = np.argmin(cost_matrix[node,1:]) + 1
        sol.append(node)
        cost_matrix[:,node] = np.inf
        nodes_remaining.remove(node)
    return sol

def stable_softmax(x, t):
    z = x - max(x)
    numerator = np.exp(z/t)
    denominator = np.sum(numerator)
    softmax = numerator/denominator

    return softmax

def solve_nearest_neighbor_softmax_temp(cost_matrix, temp=1e-2):
    cost_matrix = cost_matrix.copy()
    sol = []
    node = 0

    for i in range(len(cost_matrix) - 1):
        dists = cost_matrix[node,1:]
        p = stable_softmax(-dists, temp)
        node = np.random.choice(len(dists), p=p) + 1
        sol.append(node)
        cost_matrix[:,node] = np.inf

    return sol

def solve_sampling(solve_fn, cost_matrix, N=5000, T=5):
    start = time.time()
    sol = None
    best_cost = np.inf
    for i in range(N):
        if time.time() - start > T:
            break
        perm = solve_fn(cost_matrix)
        cost = compute_cost(perm, cost_matrix)
        if cost < best_cost:
            best_cost = cost
            sol = perm
    return sol
