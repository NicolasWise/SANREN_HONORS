import spectral_analysis
import core_resilience
import classical_graph_measures
import json
import networkx as nx
import csv
import matplotlib.pyplot as plt

def populate_graph_v0(filename):
    with open(filename) as f:
        data = json.load(f)

    G = nx.Graph()
    #Note this change may need to be
    G.name = filename

    for node in data["nodes"]:
        G.add_node(node["ID"])

    for edge in data['edges']:
        G.add_edge(edge["SOURCE"], edge["DESTINATION"])

    return G

def populate_graph_json(filename):
    with open(filename) as f:
        data = json.load(f)

    G = nx.Graph()
    name = filename.split('/')[1]
    G.name = name

    for link_pair in data:
        if len(link_pair) == 2:
            source = link_pair[0]['hostname']
            target = link_pair[1]['hostname']
            G.add_edge(source, target)
    
    return G

def populate_graph_tgf(filename):

    G = nx.Graph()
    G.name = (filename.split('/'))[2]

    with open(filename) as file:
        data = file.read().split('\n#\n')
        nodes = data[0].strip().split('\n')
        edges = data[1].strip().split('\n')
        for edge in edges:
            if edge:
                parts = edge.split(' ')

                edge1 = parts[0]
                edge2 = parts[1]

                G.add_edge(edge1, edge2)

    return G

def load_graph(filename, filetype):
    if filetype == 'tgf':
        return populate_graph_tgf(filename)
    elif filetype == 'json':
        return populate_graph_json(filename)
    elif filetype == 'custom':
        return populate_graph_v0(filename)
    else:
        raise ValueError(f"Unsupported file type: {filetype}")

def plot_graph(graph, tgf=False, sample = False, json = False):
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    layout =  nx.spring_layout(graph)
    node_size = 150
    font_size = 8
    plt.figure(figsize=(10,8))
    nx.draw(
        graph, layout, with_labels=True,
        node_size=node_size,
        font_size=font_size,
        node_color='skyblue',
        edge_color='gray'
    )
    plt.title(f"Graph: {graph.name} ({num_nodes} nodes, {num_edges} edges)")
    plt.tight_layout()
    if tgf:
        plt.savefig(f"Plots/TGF_Files/{graph.name}.png", dpi=300)
    elif sample:
        plt.savefig(f"Plots/Samples/{graph.name}.png")
    elif json:
        plt.savefig(f"Plots/JSON_Files/{graph.name}.png")
    plt.close()

def write_measure_to_csv(data_dict, filename, metric_name):
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        writer.writerow(['Node', metric_name])
        for node, value in data_dict.items():
            writer.writerow([node, value])

def identify_top_r_nodes_v0(dict, r = 45):
    top_nodes = {}
    count = len(dict)
    top_n = max(1, int(count*r/100))
    sorted_nodes = sorted(dict.items(), key=lambda x:x[1], reverse=True)
    
    for node, value in sorted_nodes[:top_n]:
        top_nodes[node] = value

    return top_nodes

def identify_top_r_nodes(dict, r = 45):
    return {node: value for node, value in sorted(dict.items(), key=lambda x:-x[1])}

def identify_bottom_r_nodes(dict, r = 45):
    return {node: value for node, value in sorted(dict.items(), key = lambda x:x[1])}

def identify_bottom_r_nodes_v0(dict, r = 45):
    bottom_nodes = {}
    count = len(dict)
    top_n = max(1, int(count*r/100))
    sorted_nodes = sorted(dict.items(), key=lambda x:x[1], reverse=False)
    
    for node, value in sorted_nodes[:top_n]:
        bottom_nodes[node] = value

    return bottom_nodes


def export_all_results(graph, spectral, core, classical, folder='Analyses/TGF_Files'):
    # Spectral
    eigenvalues, e1_cluster, a_G, e1_mult, e0_mult = spectral
    spectral_analysis.write_spectral_to_output_file(graph, a_G, e1_cluster, e1_mult, e0_mult, tgf=True, json=False)
    spectral_analysis.plot_spectral_graphs(eigenvalues, graph_name=graph.name, tgf=True)
    
    # Core resilience
    core_number, core_strength, core_influence, CIS, top_core, bottom_core = core
    core_resilience.write_core_resilience_to_csv(graph, core_number, core_strength, core_influence, CIS, sample_name=graph.name, tgf=True)
    
    for metric, data in top_core.items():
        write_measure_to_csv(data, f'{folder}/{graph.name}_top_r_core_{metric}.csv', f'Core {metric.capitalize()}')
    for metric, data in bottom_core.items():
        write_measure_to_csv(data, f'{folder}/{graph.name}_bottom_r_core_{metric}.csv', f'Core {metric.capitalize()}')
    
    # Classical
    degree_dict, close_dict, bet_dict, top_classical, bottom_classical = classical
    for metric, data in top_classical.items():
        write_measure_to_csv(data, f'{folder}/{graph.name}_top_r_{metric}_centrality.csv', f'{metric.capitalize()} Centrality')
    for metric, data in bottom_classical.items():
        write_measure_to_csv(data, f'{folder}/{graph.name}_bottom_r_{metric}_centrality.csv', f'{metric.capitalize()} Centrality')

def analyze_graph(graph, r=45):
    # Spectral analysis
    eigenvalues, e1_cluster, a_G, e1_mult, e0_mult = spectral_analysis.compute_spectral_analysis(graph)
    
    # Core resilience
    core_number, core_strength, core_influence, CIS = core_resilience.compute_core_resilience(graph)
    top_core = { 
        'numbers': identify_top_r_nodes(core_number, r),
        'strengths': identify_top_r_nodes(core_strength, r),
        'influences': identify_top_r_nodes(core_influence, r)
    }
    bottom_core = {
        'numbers': identify_bottom_r_nodes(core_number, r),
        'strengths': identify_bottom_r_nodes(core_strength, r),
        'influences': identify_bottom_r_nodes(core_influence, r)
    }
    
    # Classical metrics
    degree_dict, close_dict, bet_dict = classical_graph_measures.compute_classical_graph_measures(graph)
    top_classical = {
        'degree': identify_top_r_nodes(degree_dict, r),
        'closeness': identify_top_r_nodes(close_dict, r),
        'betweenness': identify_top_r_nodes(bet_dict, r)
    }
    bottom_classical = {
        'degree': identify_bottom_r_nodes(degree_dict, r),
        'closeness': identify_bottom_r_nodes(close_dict, r),
        'betweenness': identify_bottom_r_nodes(bet_dict, r)
    }

    return {
        'spectral': (eigenvalues, e1_cluster, a_G, e1_mult, e0_mult),
        'core': (core_number, core_strength, core_influence, CIS, top_core, bottom_core),
        'classical': (degree_dict, close_dict, bet_dict, top_classical, bottom_classical)
    }


def process_graphs(inputs, r=45):
    for filename in inputs:
        filetype = filename.split('.')[-1]
        path, r = ((f'Graph_files/TGF_Files/{filename}', 45) if filetype == 'tgf' else (f'Graph_files/{filename}', 25))
        graph = load_graph(path, filetype)
        plot_graph(graph, tgf=(filetype=='tgf'), json=(filetype=='json'))
        results = analyze_graph(graph, r)
        export_all_results(graph, results['spectral'], results['core'], results['classical'])

def main():
    inputs = ['bfn.tgf', 'cpt.tgf', 'dur.tgf', 'els.tgf', 'jnb.tgf', 'pta.tgf', 'pzb.tgf', 'vdp.tgf', 'isis-links.json',]
    process_graphs(inputs)

if __name__ == "__main__":
    main()