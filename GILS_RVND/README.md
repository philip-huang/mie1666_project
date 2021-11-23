# Minimum Latency Problem (MLP)

The Minimum Latency Problem (MLP) is a variant of the traditional optimization problem [TSP](https://github.com/renatamendesc/TSP), which aims to minimize the sum of arrival times at all vertices.

## Metaheuristics
For instances which the number of vertices that we need to cover is very big, it's necessary the use of a metaheuristic to solve the issue, which doesn't guarantee the optimal solution, but in general results on good anwsers, otherwise, the computer can't give the optimal solution in a feasible time. Therefore, it was used the metaheuristic **GILS-RVND**, which combines components of Greedy Randomized Adaptive Search Procedure (GRASP), Iterated Local Search (ILS) and Variable Neighborhood Descent with Random neighborhood ordering (RVND).

## Running

To execute the program you just have to open your terminal and type the following command:

```./mlp instances/(INSTANCE)```

The instance that you choose must exist on the folder [instances](https://github.com/renatamendesc/MLP/tree/main/instances).

## Results
Once implemented correctly, optimal solutions to instances with up to 107 customers are obtained in a few seconds. All the execution results were registered on the  folder [benchmark](https://github.com/renatamendesc/MLP/blob/main/benchmark/bm_final.txt).
