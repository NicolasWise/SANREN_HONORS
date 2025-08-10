import csv
import os
from itertools import combinations
import networkx as nx
import plotter as plot
import node_removals as ndr
import pandas as pd



def parse_tgf(filepath):
    with open(filepath, encoding="utf-8") as f:
        nodes_part, edges_part = f.read().split("\n#\n", 1)

    # id -> label (labels can contain spaces)
    id2label = {}
    for line in nodes_part.strip().splitlines():
        parts = line.strip().split(maxsplit=1)
        if not parts: 
            continue
        node_id  = parts[0]
        label = parts[1].strip() if len(parts) == 2 else node_id
        id2label[node_id] = label

    # edges as label pairs (ignore weights if present)
    edges = []
    for line in edges_part.strip().splitlines():
        parts = line.strip().split()
        if len(parts) >= 2:
            u = id2label.get(parts[0], parts[0])
            v = id2label.get(parts[1], parts[1])
            edges.append((u, v))
    return id2label, edges

def build_union_graph(tgf_files):
    G = nx.Graph()
    G.name = "TGF_UNION"
    for fp in tgf_files:
        _, edges = parse_tgf(fp)
        src = os.path.basename(fp)
        for u, v in edges:
            if G.has_edge(u, v):
                # track which files contributed this edge
                G[u][v].setdefault("sources", set()).add(src)
            else:
                G.add_edge(u, v, sources={src})
        # optionally: track per-node provenance
        for u, v in edges:
            G.nodes[u].setdefault("files", set()).add(src)
            G.nodes[v].setdefault("files", set()).add(src)
    return G


def parse_tgf_nodes(filepath):
    """Return a set of node labels from a TGF file."""
    with open(filepath, "r", encoding="utf-8") as f:
        node_section = f.read().split("\n#\n")[0]
    nodes = set()
    for line in node_section.strip().splitlines():
        parts = line.strip().split(maxsplit=1)  # id, label
        if len(parts) == 2:
            nodes.add(parts[1].strip())
        elif len(parts) == 1:  # in case there's no label
            nodes.add(parts[0].strip())
    return nodes

def identify_connecting_nodes():
    subgraphs = ['pta.tgf', 'bfn.tgf', 'cpt.tgf', 'dur.tgf', 'els.tgf', 'jnb.tgf', 'pzb.tgf', 'vdp.tgf']
    path = 'Graph_Files/TGF_Files/'
    # read all nodes at once
    graph_nodes = {g: parse_tgf_nodes(os.path.join(path,g)) for g in subgraphs}
    all_nodes = set().union(*graph_nodes.values())
    connecting_nodes = set()
    not_connecting_nodes = set()

    for g1, g2 in combinations(subgraphs, 2):
        shared = graph_nodes[g1] & graph_nodes[g2]
        not_connecting_nodes = all_nodes - shared

        connecting_nodes.update(shared)

    return connecting_nodes, not_connecting_nodes

def find_in_sub_graph(node_name):
    subgraphs = ['pta.tgf', 'bfn.tgf', 'cpt.tgf', 'dur.tgf', 'els.tgf', 'jnb.tgf', 'pzb.tgf', 'vdp.tgf']
    path = 'Graph_Files/TGF_Files/'
    for subgraph in subgraphs:
        filename = f"{path}{subgraph}"
        with open(filename) as file:
            data = file.read().split('\n#\n')
            nodes = data[0].strip().split('\n')
            
            for node in nodes:
                new_node = node.split(' ')[1]
                if new_node == node_name:
                    print(f'Found {node_name} in {filename}')

def read_data(file):
    nodes = []
    with open(file, 'r') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=';')
        for row in reader:
            # search in sub graphs
            find_in_sub_graph(row['Node'])
        


def main():
    '''metrics = ['_bottom_r_core_influences', '_bottom_r_betweeness_centrality', '_bottom_r_closeness_centrality', '_bottom_r_degree_centrality']
    path = 'Analyses/JSON_Files/'
    filename = "isis-links.json"'''

    #A method to create one representative graph using all the tgf files
    subgraphs = ['pta.tgf', 'bfn.tgf', 'cpt.tgf', 'dur.tgf', 'els.tgf', 'jnb.tgf', 'pzb.tgf', 'vdp.tgf']
    path = 'Graph_Files/TGF_Files/'
    # read all nodes at once
    files = [os.path.join(path, fn) for fn in subgraphs]
    graph = build_union_graph(files)
    results = plot.analyze_graph(graph)
    plot.plot_graph(graph, tgf=True)
    plot.export_all_results(graph, results['spectral'], results['core'], results['classical'])
    print("did it work?")

    out_dir = "Removals/Union"
    os.makedirs(out_dir, exist_ok=True)

    all_dfs = []

    for fn, name in ndr.strategies:
        df = ndr.simulate_strategy(graph, fn, name)
        df.to_csv(os.path.join(out_dir, f'{graph.name}_{name}.csv'), index=False)
        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)

    # plot each metric
    for metric in ['aG','e1_mult','e0_mult','CIS']:
        ndr.plot_metric_small_multiples(
            combined,
            metric,
            os.path.join(out_dir, f'{graph.name}_compare_{metric}.png')
        )


    '''
    connecting_nodes, non_connecting_nodes = identify_connecting_nodes()
    print(f"Connecting nodes: {connecting_nodes}]")
    print(isinstance(connecting_nodes, set))
    print(isinstance(non_connecting_nodes, set))
    print(f"Non-connecting nodes: {non_connecting_nodes}\n{len(non_connecting_nodes)}")
    for metric in metrics:
        file = f"{path}{filename}{metric}.csv"
        read_data(file)'''


if __name__=="__main__":
    main()