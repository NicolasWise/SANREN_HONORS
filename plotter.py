import spectral_analysis
import core_resilience
import classical_graph_measures
import json
import networkx as nx
import csv
import matplotlib.pyplot as plt
import os


def populate_graph_json(filename):
    """ Populate a NetworkX graph from a JSON file containing link pairs. """
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
    """ Populate a NetworkX graph from a TGF (Trivial Graph Format) file. """
    G = nx.Graph()
    G.name = os.path.splitext(os.path.basename(filename))[0]

    with open(filename, encoding="utf-8") as f:
        content = f.read()

    # Split node and edge sections (handles the standard '\n#\n')
    try:
        nodes_part, edges_part = content.split('\n#\n', 1)
    except ValueError:
        # Fallback if line endings are different or '#' line is unusual
        parts = content.split('#', 1)
        nodes_part = parts[0]
        edges_part = parts[1] if len(parts) > 1 else ''

    # Build ID -> label map and add nodes by label
    id2label = {}
    for line in nodes_part.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        # TGF: "ID<space>Label..." (label may have spaces)
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            nid, label = parts[0], parts[1].strip()
        else:
            nid, label = parts[0], parts[0]  # no separate label provided
        id2label[nid] = label
        G.add_node(label)

    # Add edges using labels
    for line in edges_part.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        a_id, b_id = parts[0], parts[1]
        u = id2label.get(a_id, a_id)
        v = id2label.get(b_id, b_id)
        if u == v:
            continue  # skip self-loops; remove if you want to keep them
        G.add_edge(u, v)

    return G

def load_graph(filename, filetype):
    """ Load a graph from a file, determining the format by file extension. """
    if filetype == 'tgf':
        return populate_graph_tgf(filename)
    elif filetype == 'json':
        return populate_graph_json(filename)
    else:
        raise ValueError(f"Unsupported file type: {filetype}")

def plot_graph(graph, tgf=False, sample = False, json = False):
    """ Plot and save the graph structure. """
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


def identify_top_r_nodes(dict, r = 45):
    """ Identify top r-percentile nodes based on the given metric dictionary. """
    return {node: value for node, value in sorted(dict.items(), key=lambda x:-x[1])}

def identify_bottom_r_nodes(dict, r = 45):
    """ Identify bottom r-percentile nodes based on the given metric dictionary. """
    return {node: value for node, value in sorted(dict.items(), key = lambda x:x[1])}


def export_all_results(graph, spectral, core, classical, tgf=False, json=False, folder = None):
    """ Export all analysis results to CSV files and plots. """
    if tgf:
        subdir = 'TGF_Files'
    elif json:
        subdir = 'JSON_Files'
    else:
        subdir = ''
    # Spectral
    eigenvalues, e1_cluster, a_G, e1_mult, e0_mult = spectral
    spectral_analysis.write_spectral_to_output_file(graph, a_G, e1_cluster, e1_mult, e0_mult, tgf=tgf, json=json)
    spectral_analysis.plot_spectral_graphs(eigenvalues, graph_name=graph.name, tgf=tgf, json=json)
    
    # Core resilience
    core_number, core_strength, core_influence, CIS, top_core, bottom_core = core
    core_resilience.write_core_resilience_to_csv(graph, core_number, core_strength, core_influence, CIS, sample_name=graph.name, tgf=tgf, json=json)
    
    for metric, data in top_core.items():
        write_measure_to_csv(data, f'{folder}/{subdir}/{graph.name}_top_r_core_{metric}.csv', f'Core {metric.capitalize()}')
    for metric, data in bottom_core.items():
        write_measure_to_csv(data, f'{folder}/{subdir}/{graph.name}_bottom_r_core_{metric}.csv', f'Core {metric.capitalize()}')
    
    # Classical
    degree_dict, close_dict, bet_dict, top_classical, bottom_classical = classical
    for metric, data in top_classical.items():
        write_measure_to_csv(data, f'{folder}/{subdir}/{graph.name}_top_r_{metric}_centrality.csv', f'{metric.capitalize()} Centrality')
    for metric, data in bottom_classical.items():
        write_measure_to_csv(data, f'{folder}/{subdir}/{graph.name}_bottom_r_{metric}_centrality.csv', f'{metric.capitalize()} Centrality')

def analyze_graph(graph, r=45):
    """ Perform full analysis on the graph, returning all computed metrics."""
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
        'betweeness': identify_top_r_nodes(bet_dict, r)
    }
    bottom_classical = {
        'degree': identify_bottom_r_nodes(degree_dict, r),
        'closeness': identify_bottom_r_nodes(close_dict, r),
        'betweeness': identify_bottom_r_nodes(bet_dict, r)
    }

    return {
        'spectral': (eigenvalues, e1_cluster, a_G, e1_mult, e0_mult),
        'core': (core_number, core_strength, core_influence, CIS, top_core, bottom_core),
        'classical': (degree_dict, close_dict, bet_dict, top_classical, bottom_classical)
    }


def process_graphs(inputs, r=45):
    """ Process a list of graph files, performing analysis and exporting results. """
    for filename in inputs:
        filetype = filename.split('.')[-1]
        path, r, tgf, json= ((f'Graph_files/TGF_Files/{filename}', 45, True, False) if filetype == 'tgf' else (f'Graph_files/{filename}', 25, False, True))
        graph = load_graph(path, filetype)
        plot_graph(graph, tgf=(filetype=='tgf'), json=(filetype=='json'))
        results = analyze_graph(graph, r)
        export_all_results(graph, results['spectral'], results['core'], results['classical'],tgf=tgf, json=json, folder="Analyses")

def main():
    """ Main function to process predefined graph files. """
    inputs = ['bfn.tgf', 'cpt.tgf', 'dur.tgf', 'els.tgf', 'jnb.tgf', 'pta.tgf', 'pzb.tgf', 'vdp.tgf', 'isis-links.json',]
    process_graphs(inputs)

if __name__ == "__main__":
    main()