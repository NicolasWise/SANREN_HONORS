import csv
import os
from itertools import combinations


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
    connecting_nodes = set()

    for g1, g2 in combinations(subgraphs, 2):
        shared = graph_nodes[g1] & graph_nodes[g2]
        for node in shared:
            print(f"{node} is a connecting node between {g1} and {g2}")
        connecting_nodes.update(shared)

    return sorted(connecting_nodes)

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
    metrics = ['_bottom_r_core_influences', '_bottom_r_betweeness_centrality', '_bottom_r_closeness_centrality', '_bottom_r_degree_centrality']
    path = 'Analyses/JSON_Files/'
    filename = 'isis-links.json'
    connecting_nodes = identify_connecting_nodes()
    print(connecting_nodes)
    '''for metric in metrics:
        file = f"{path}{filename}{metric}.csv"
        read_data(file)'''


if __name__=="__main__":
    main()