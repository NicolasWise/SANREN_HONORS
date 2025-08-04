import networkx as nx
import itertools
import csv
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs

''' Exhaustive search: This method gives a more accurate approximation for core strengths.
    But, its complexity is O(n2^d) because it generates all combinations of neighbors for each node and is therefore ineffecient for large graphs'''
def compute_core_strength_v0(G):
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

''' Greedy Approach: This core strength method removes the weaker neighbors first to reduce 
    complexity. Its complexity is O(n.d). The core strengths are less accurate, but decent approximations 
    of a node's core strenght but much faster for large graphs
    
    Core strength measures how many nodes must be removed in order to reduce a node's core number.
    This is a resilience metric and gives an approximation of a node's resilience against network failures.
    
    The core strength of a node is a measure of how likely its core number will decreas 
    when edges are deleted from the network. '''
def compute_core_strength_v1(G):
    core_strength = {}
    original_core = nx.core_number(G)

    for node in G:
        '''Sort the neighbors of each node in increasing order (smallest core number to largest core number).
            This allows us to remove the weaker neighbors first to save time. '''
        neighbors = sorted(G.neighbors(node), key=lambda x: G.degree(x))
        original_core_number = original_core[node]
        removed_count = 0

        G_temp = G.copy()
        for neighbor in neighbors:
            G_temp.remove_node(neighbor)
            removed_count+=1
            new_core = nx.core_number(G_temp)
            if new_core.get(node, 0)  <  original_core_number:
                core_strength[node] = removed_count
                break
            else:
                core_strength[node] = len(neighbors)
    return core_strength

# --- Core Strength (theoretical formula) ---
def compute_core_strength(G, core_num):
    """
    Core Strength: CS(u) = |{v ∈ Γ(u): κ(v) ≥ κ(u)}| - κ(u) + 1
    """
    CS = {}
    for u in G.nodes():
        neighbors_ge = sum(1 for v in G.neighbors(u) if core_num[v] >= core_num[u])
        CS[u] = neighbors_ge - core_num[u] + 1
    return CS

'''Core influence measures a node's impact on the core numbers of its neighbours. A centrality measure.
    The core influence of a node is a measure of the extent to which nodes with lower core 
    numbers depend on that node for their own core numbers. '''
def compute_core_influence_v0(G):
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


# --- Core Influence (eigenvector-based) ---
def compute_core_influence(G, core_num, approximate=False):
    """
    Compute Core Influence using eigenvector of M matrix.
    If approximate=True, ignore contributions from equal-core neighbors.
    """
    n = len(G.nodes())
    node_index = {node: i for i, node in enumerate(G.nodes())}
    
    # Build sparse matrix M
    data, rows, cols = [], [], []
    for u, v in G.edges():
        ku, kv = core_num[u], core_num[v]
        # u influences v
        if ku <= kv:
            denom = sum(1 for w in G.neighbors(v) if core_num[w] >= core_num[v])
            if denom > 0:
                if not (approximate and ku == kv):
                    rows.append(node_index[u])
                    cols.append(node_index[v])
                    data.append(1.0/denom)
        # v influences u
        if kv <= ku:
            denom = sum(1 for w in G.neighbors(u) if core_num[w] >= core_num[u])
            if denom > 0:
                if not (approximate and kv == ku):
                    rows.append(node_index[v])
                    cols.append(node_index[u])
                    data.append(1.0/denom)
    
    # Add diagonal 1
    for u in G.nodes():
        idx = node_index[u]
        rows.append(idx)
        cols.append(idx)
        data.append(1.0)
    
    M = csr_matrix((data, (rows, cols)), shape=(n, n))
    
    # Eigenvector (largest eigenvalue)
    vals, vecs = eigs(M, k=1, which='LM')
    r = np.real(vecs[:, 0])
    r = np.abs(r)  # ensure non-negative
    r = r / np.linalg.norm(r)
    
    return {node: r[i] for node, i in node_index.items()}

# --- Core Influence-Strength metric ---
def compute_CIS(G, f=0.9, approximate_CI=False):
    """
    Compute Core Influence-Strength (CIS_f)
    f: percentile threshold (0 < f <= 1)
    """
    core_num = nx.core_number(G)
    CS = compute_core_strength(G, core_num)
    CI = compute_core_influence(G, core_num, approximate=approximate_CI)
    
    # f-th percentile of CI
    CI_values = np.array(list(CI.values()))
    threshold = np.percentile(CI_values, f * 100)
    
    # Nodes with CI >= threshold
    S_f = [u for u, ci in CI.items() if ci >= threshold]
    if not S_f:
        return 0
    
    return np.mean([CS[u] for u in S_f])

def compute_Core_Influence_Strength_metric_v0(core_strength, core_influence, graph_name):
    '''The Core Influence-Strength (CIS) metric, defined as the average core strength of the top
      r % most influential nodes provides a measure of network resilience.'''
    top_influential_nodes = {}
    r = 10
    count = len(core_influence)
    top_n = max(1, int(count*r/100))

    sorted_nodes = sorted(core_influence.items(), key=lambda x:x[1], reverse=True)
    for node, _ in sorted_nodes[:top_n]:
        top_influential_nodes[node] = _

    total_core_strength = 0
    for node_key, node_value in top_influential_nodes.items():
        total_core_strength+= core_strength[node_key]

    CIS = total_core_strength/top_n
    
    return CIS

def write_core_resilience_to_csv(graph, core_number, core_strength, core_influence, CIS, sample_name='',sample=False, tgf=False, json=False):
    # Extract all unique nodes from the keys of one of the dictionaries
    nodes = list(core_number.keys())
    if sample:
        output_file = f'Analyses/Samples/{sample_name}_{graph.name}_core_resilience_metrics.csv'
        CIS_output_filename = f'Analyses/Samples/CIS_metric.csv'
    elif tgf:
        output_file= f'Analyses/TGF_Files/{graph.name}_core_resilience_metrics.csv'
        CIS_output_filename = f'Analyses/TGF_Files/CIS_metric.csv'
    elif json:
        output_file = f'Analyses/JSON_Files/{graph.name}_core_resilience_metrics.csv'
        CIS_output_filename = f'Analyses/TGF_Files/CIS_metric.csv'
    # Define header
    header = ['Node', 'Core Number', 'Core Strength', 'Core Influence']

    # Prepare rows
    rows = []
    for node in nodes:
        row = {
            'Node': node,
            'Core Number': core_number.get(node, 'N/A'),
            'Core Strength': core_strength.get(node, 'N/A'),
            'Core Influence': core_influence.get(node, 'N/A')
        }
        rows.append(row)

    write_header = not os.path.exists(output_file)
    # Write to CSV (use semicolon for Excel compatibility)
    with open(f"{output_file}", 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header, delimiter=';')
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


    
    file_exists = os.path.isfile(f'{CIS_output_filename}')
    print(f"Core_Influence Strength Metric: {CIS}")
    header = ['Name', 'Core-Influence Strength Metric']
    with open(f'{CIS_output_filename}', "a" ,newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, delimiter=';', fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerow({'Name': f'{graph.name}', 'Core-Influence Strength Metric': CIS})

    print(f"Core resilience data written to {CIS_output_filename}")

def compute_core_resilience(G):
    '''A node's core number is the highest k for which it remains in the k-core, reflecting its structural depth and resilience.'''
    core_number = nx.core_number(G)

    core_strength = compute_core_strength(G, core_number)

    core_influence = compute_core_influence(G, core_number)
    
    CIS = compute_CIS(G)

    return core_number, core_strength, core_influence, CIS

