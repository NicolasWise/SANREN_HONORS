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
    G.name = filename

    for node in data["nodes"]:
        G.add_node(node["ID"])

    for edge in data['edges']:
        G.add_edge(edge["SOURCE"], edge["DESTINATION"])

    return G

def populate_graph_v1(filename):
    with open(filename) as f:
        data = json.load(f)

    G = nx.Graph()
    G.name = filename

    for link_pair in data:
        if len(link_pair) == 2:
            source = link_pair[0]['hostname']
            target = link_pair[1]['hostname']
            G.add_edge(source, target)
    
    return G


def export_graph(Graph):
    with open(f"{Graph.name}_graph_edges.csv", "w", newline='') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow(['Source', 'Target'])
        for u, v in Graph.edges():
            writer.writerow([u, v])

def plot_graph(Graph, sample_name=''):
    num_nodes = Graph.number_of_nodes()
    num_edges = Graph.number_of_edges()
    layout =  nx.spring_layout(Graph)
    node_size = 150
    font_size = 8
    plt.figure(figsize=(10,8))
    nx.draw(
        Graph, layout, with_labels=True,
        node_size=node_size,
        font_size=font_size,
        node_color='skyblue',
        edge_color='gray'
    )
    plt.title(f"Graph: {sample_name} ({num_nodes} nodes, {num_edges} edges)")
    plt.tight_layout()
    plt.savefig(f"Plots/graph_plot_{sample_name}.png", dpi=300)
    plt.close()

def write_measure_to_csv(data_dict, filename, metric_name):
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        writer.writerow(['Node', metric_name])
        for node, value in data_dict.items():
            writer.writerow([node, value])

def identify_top_r_nodes(dict):
    top_nodes = {}
    r = 10
    count = len(dict)
    top_n = max(1, int(count*r/100))
    sorted_nodes = sorted(dict.items(), key=lambda x:x[1], reverse=True)
    
    for node, value in sorted_nodes[:top_n]:
        top_nodes[node] = value

    return top_nodes

def main():
    filename = 'Graph_files/isis-links.json'
    Graph = populate_graph_v1(filename)
    graph_size = Graph.size()
    print(f'{graph_size} edges')

    plot_graph(Graph, sample_name='main')

    
    #Compute Classical Graph Measures
    degree_centrality_dict, closeness_centrality_dict, betweeness_centrality_dict = classical_graph_measures.compute_classical_graph_measures(Graph)
    write_measure_to_csv(degree_centrality_dict, 'Analyses/degree_centrality.csv', 'Degree Centrality')
    write_measure_to_csv(closeness_centrality_dict, 'Analyses/closeness_centrality.csv', 'Closeness Centrality')
    write_measure_to_csv(betweeness_centrality_dict, 'Analyses/betweeness_centrality.csv', 'Betweeness Centrality')
    top_degree_nodes = identify_top_r_nodes(degree_centrality_dict)
    top_closeness_nodes = identify_top_r_nodes(closeness_centrality_dict)
    top_betweeness_nodes = identify_top_r_nodes(betweeness_centrality_dict)
    write_measure_to_csv(top_betweeness_nodes, 'Analyses/top_r_betweeness_centrality.csv', 'Betweeness Centrality')
    write_measure_to_csv(top_degree_nodes, 'Analyses/top_r_degree_centrality.csv', 'Degree Centrality')
    write_measure_to_csv(top_closeness_nodes, 'Analyses/top_r_closeness_centrality.csv', 'Closeness Centrality')


    #Spectral Analysiss
    eigenvalue_one_cluster_density, algebraic_connectivity, eigenvalue_one_multiplicity, eigenvalue_zero_multiplicity = spectral_analysis.compute_spectral_analysis(Graph)
    spectral_analysis.write_spectral_to_output_file(Graph, algebraic_connectivity, eigenvalue_one_cluster_density, eigenvalue_one_multiplicity, eigenvalue_zero_multiplicity)
    
    #Core Resilience Analysis
    core_number, core_strength, core_influence, CIS = core_resilience.compute_core_resilience(Graph)
    core_resilience.write_core_resilience_to_csv(Graph, core_number, core_strength, core_influence, CIS)
    top_core_numbers = identify_top_r_nodes(core_number)
    top_core_influences = identify_top_r_nodes(core_influence)
    top_core_strengths = identify_top_r_nodes(core_strength)
    write_measure_to_csv(top_core_numbers,'Analyses/top_r_core_numbers.csv', 'Core Number')
    write_measure_to_csv(top_core_influences,'Analyses/top_r_core_influences.csv', 'Core Influence')
    write_measure_to_csv(top_core_strengths,'Analyses/top_r_core_strengths.csv', 'Core Strength')
    
    
if __name__=="__main__":
    main()




