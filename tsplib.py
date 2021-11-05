import tsplib95 as tsp
import os
import networkx as nx
import matplotlib.pyplot as plt

def get_all_tsp_file():
    file_list = []
    for file in os.listdir("./data"):
        if file.endswith(".tsp"):
            file_list.append(os.path.join("./data", file))

    tour_list = []
    for file in os.listdir("./data"):
        if file.endswith(".opt.tour"):
            tour_list.append(os.path.join("./data", file))

    return file_list, tour_list

def load_tsp(filename):
    """Load a TSPLIB instance from a file.

    return: the tsp problem instance
    and the networkx graph instance
    """
    problem =  tsp.load_problem(filename)
    G = problem.get_graph()
    return problem, G

### Example
def example():
    path = './data/att48.tsp'
    problem, G = load_tsp(path)

    # render (give a list of nodes and their coordinates)
    print(problem.render())

    # get list of nodes
    print(list(problem.get_nodes()))

    # get the list of nodes from networkx graph
    print(G.nodes())

    # edge weight between node 1 and 10
    print(problem.get_weight(1, 10))
    
    # also accessiable from networkx graph
    print(G.edges[1, 10]['weight'])

    # visualize the graph on 
    nx.draw_networkx(G, with_labels=True)
    plt.show()

if __name__ == "__main__":
    # testing
    example()

