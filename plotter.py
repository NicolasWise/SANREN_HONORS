import json
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigvalsh
import itertools

def compute_core_strength(G):
    '''Initialise core stregnth dictionary
       and compute original core numbers of nodes  '''
    core_strength ={}
    original_core = nx.core_number(G)
    '''Iterate over each node in the graph'''
    for node in G:
        '''Return a list of a node's neighbors'''
        neighbors = list(G.neighbors(node))
        '''Obtain core number of this node'''
        original_cn = original_core[node]
        
        ''' Start testing by removing 1 neighbor, then 2, etc.
            A nodes core strength is the minimum number of neighbors
            that must be removed inorder to reduce that node's core number. '''
        found = False
        for r in range(1, len(neighbors) + 1):
            ''' r is the number of neighbors to remove at once for each iteration.
                The below gives you all combinations of r neighbors. No repitition.'''
            for subset in itertools.combinations(neighbors, r):
                '''Create a copy of the graph and remove the subset of neighbors'''
                G_temp = G.copy()
                G_temp.remove_nodes_from(subset)
                
                try:
                    ''' Now recompute the core numbers and compare
                        this nodes new core number to its old core number.
                        If its core number decreased then this nodes core strength is 
                        initialised as r.
                        '''
                    new_core = nx.core_number(G_temp)
                    if new_core.get(node, 0) < original_cn:
                        core_strength[node] = r
                        found = True
                        break
                except nx.NetworkXError:
                    continue 
                
            if found:
                break
        '''Max value if no subset reduces core number'''
        if not found:
            core_strength[node] = len(neighbors)  

    return core_strength

def compute_core_influence(G):
    core_influence = {}
    for node in G.nodes:
        '''Compute original core numbers, create copy of graph, remove this node,
            and calculate updated core numbers'''
        original_core_numbers = nx.core_number(G)
        G_removed = G.copy()
        G_removed.remove_node(node)
        updated_core_numbers = nx.core_number(G_removed)

        influence = 0
        ''' For each neighbor of this node'''
        for neighbor in G.neighbors(node):
            '''If the neighbor has updated core number'''
            if neighbor in updated_core_numbers:
                '''Record the change between the neighbors new and old core numbers'''
                change = original_core_numbers[neighbor] - updated_core_numbers[neighbor]
                influence += max(change, 0)
        core_influence[node] = influence

    return core_influence

def compute_spectral_analysis(G):
    L = nx.laplacian_matrix(G).todense()
    print("Laplacian Matrix:")
    print(L)

    # Sorted to make algebraic connectivity derivation easier.
    eigenvalues = np.sort(np.linalg.eigvalsh(L))
    print("Eigenvalues:")
    print(eigenvalues)

    # Second highest eiganvalue
    algebraic_connectivity = eigenvalues[1]
    print(f"Algebraic Connectivity: {algebraic_connectivity}")

    # Count how many eigenvalues fall in the range [0.9, 1.1]
    cluster_count = np.sum((eigenvalues >= 0.9) & (eigenvalues <= 1.1))
    total_eigenvalues = len(eigenvalues)
    # Density is just proportion of eigenvalues in the range
    cluster_density = cluster_count / total_eigenvalues
    print(f"Number of eigenvalues clustered around 1: {cluster_count}")
    print(f"Density (proportion) of clustered eigenvalues: {cluster_density:.2f}")

    plt.figure(figsize=(8, 4))
    plt.plot(range(len(eigenvalues)), eigenvalues, marker='o')
    plt.axhline(1, color='red', linestyle='--', label='λ = 1')
    plt.title("Eigenvalues of Laplacian Matrix")
    plt.xlabel("Index")
    plt.ylabel("Eigenvalue")
    plt.legend()
    plt.grid(True)
    plt.savefig("Eigenvalues_of_Laplacian_Matrix")

    # Plot eigenvalue distribution
    plt.figure(figsize=(8, 4))
    plt.hist(eigenvalues, bins=10, edgecolor='black')
    plt.title("Eigenvalue Distribution of the Laplacian")
    plt.xlabel("Eigenvalue")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig("Eigenvalue_Distribution_of_the_Laplacian")

def populate_graph(filename):
    with open(filename) as f:
        data = json.load(f)

    G = nx.Graph()

    for node in data["nodes"]:
        G.add_node(node["ID"])

    for edge in data['edges']:
        G.add_edge(edge["SOURCE"], edge["DESTINATION"])

    return G

def compute_core_resilience(G):
    core_number = nx.core_number(G)
    print(core_number)

    core_strength = compute_core_strength(G)
    print(core_strength)

    core_influence = compute_core_influence(G)
    print(core_influence)

def main():
    Graph = populate_graph('CapeTown_Graph_Abbreviated.json')
    compute_core_resilience(Graph)
    

if __name__=="__main__":
    main()




