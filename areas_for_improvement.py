import csv

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

    for metric in metrics:
        file = f"{path}{filename}{metric}.csv"
        read_data(file)


if __name__=="__main__":
    main()