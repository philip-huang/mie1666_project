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

def stable_softmax(x, t, axis=None):
    z = x - np.max(x, axis=axis, keepdims=True)
    numerator = np.exp(z/t)
    denominator = np.sum(numerator, axis=axis, keepdims=True)
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

def random_choice_prob_index(a, axis=1):
    r = np.expand_dims(np.random.rand(a.shape[1-axis]), axis=axis)
    return (a.cumsum(axis=axis) > r).argmax(axis=axis)

def solve_nearest_neighbor_softmax_parallel(costs, M=1280, temp=1e-2):
    N = costs.shape[0] - 1
    inds = np.arange(M)
    solutions = np.zeros((M, N), dtype=int)
    masks = np.ones((M, N + 1))
    masks[:, 0] = np.inf 
    current_state = np.zeros(M, dtype=int)
    obj_values = np.zeros((M,))
    for i in range(N):
        cost_from_here = (1e-16 + costs[current_state, :]) * masks
        probs = stable_softmax(-cost_from_here, t=temp, axis=-1)
        selected = random_choice_prob_index(probs, axis=1)
        masks[inds, selected] = np.inf
        solutions[:, i] = selected
        obj_values[:] += (N-i)*costs[current_state, selected]
        current_state = selected
    
    best_index = np.argmin(obj_values)
    return solutions[best_index]

