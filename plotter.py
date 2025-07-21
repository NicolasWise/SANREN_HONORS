import spectral_analysis
import core_resilience
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
        if len(link_pair) ==2:
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

def plot_graph(Graph):
    plt.plot(Graph)

def main():
    filename = 'isis-links.json'
    Graph = populate_graph_v1(filename)
    #Spectral Analysis
    eigenvalue_one_cluster_density, algebraic_connectivity, eigenvalue_one_multiplicity, eigenvalue_zero_multiplicity = spectral_analysis.compute_spectral_analysis(Graph)
    spectral_analysis.write_spectral_to_output_file(Graph, algebraic_connectivity, eigenvalue_one_cluster_density, eigenvalue_one_multiplicity, eigenvalue_zero_multiplicity)
    
    #Core Resilience Analysis
    core_number, core_strength, core_influence, CIS = core_resilience.compute_core_resilience(Graph)
    core_resilience.write_core_resilience_to_csv(Graph, core_number, core_strength, core_influence, CIS)

    
if __name__=="__main__":
    main()




