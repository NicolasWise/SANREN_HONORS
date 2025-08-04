import json
import networkx as nx
from plotter import write_measure_to_csv
from plotter import identify_top_r_nodes
import spectral_analysis
import core_resilience
import classical_graph_measures
from plotter import plot_graph

def populate_graph_v1_sampling(filename, graph_size, graph_name):
    with open(filename) as f:
        data = json.load(f)

    G = nx.Graph()
    G.name = graph_name

    for link_pair in data:
        if len(link_pair) == 2:
            source = link_pair[0]['hostname']
            target = link_pair[1]['hostname']
            G.add_edge(source, target)
            if G.size() == graph_size:
                break
    
    return G

def main():
    filename = 'Graph_files/isis-links.json'
    Graph_sizes = [50, 100, 150, 200, 250, 300, 350, 400]
    value = 1
    for size in Graph_sizes:  
        sample_name = f"sample{value}"
        name = filename.split('/')[1]
        graph_name = f'{sample_name}_{name}'
        graph = populate_graph_v1_sampling(filename, size, graph_name)
        # Number of edges
        num_edges = graph.size()
        print(f'{num_edges} edges')

        #Plot the graph
        plot_graph(graph, sample=True, tgf=False)
        #Compute Classical Graph Measures
        degree_centrality_dict, closeness_centrality_dict, betweeness_centrality_dict = classical_graph_measures.compute_classical_graph_measures(graph)
        write_measure_to_csv(degree_centrality_dict, f'Analyses/Samples/degree_centrality_{graph_name}.csv', 'Degree Centrality')
        write_measure_to_csv(closeness_centrality_dict, f'Analyses/Samples/closeness_centrality_{graph_name}.csv', 'Closeness Centrality')
        write_measure_to_csv(betweeness_centrality_dict, f'Analyses/Samples/betweeness_centrality_{graph_name}.csv', 'Betweeness Centrality')
        top_degree_nodes = identify_top_r_nodes(degree_centrality_dict)
        top_closeness_nodes = identify_top_r_nodes(closeness_centrality_dict)
        top_betweeness_nodes = identify_top_r_nodes(betweeness_centrality_dict)
        write_measure_to_csv(top_betweeness_nodes, f'Analyses/Samples/top_r_betweeness_centrality_{graph_name}.csv', 'Betweeness Centrality')
        write_measure_to_csv(top_degree_nodes, f'Analyses/Samples/top_r_degree_centrality_{graph_name}.csv', 'Degree Centrality')
        write_measure_to_csv(top_closeness_nodes, f'Analyses/Samples/top_r_closeness_centrality_{graph_name}.csv', 'Closeness Centrality')
        
        #Spectral Analysiss
        eigenvalues, eigenvalue_one_cluster_density, algebraic_connectivity, eigenvalue_one_multiplicity, eigenvalue_zero_multiplicity = spectral_analysis.compute_spectral_analysis(graph)
        # MUST CHANGE WRITING 
        spectral_analysis.plot_spectral_graphs(eigenvalues, graph_name, sample=True)
        spectral_analysis.write_spectral_to_output_file(graph, algebraic_connectivity, eigenvalue_one_cluster_density, eigenvalue_one_multiplicity, eigenvalue_zero_multiplicity, sample_name, sample=True)
        
        #Core Resilience Analysis
        core_number, core_strength, core_influence, CIS = core_resilience.compute_core_resilience(graph)
        top_core_numbers = identify_top_r_nodes(core_number)
        top_core_influences = identify_top_r_nodes(core_influence)
        top_core_strengths = identify_top_r_nodes(core_strength)
        write_measure_to_csv(top_core_numbers,f'Analyses/Samples/top_r_core_numbers_{graph_name}.csv', 'Core Number')
        write_measure_to_csv(top_core_influences,f'Analyses/Samples/top_r_core_influences_{graph_name}.csv', 'Core Influence')
        write_measure_to_csv(top_core_strengths,f'Analyses/Samples/top_r_core_strengths_{graph_name}.csv', 'Core Strength')
        # MUST CHANGE WRITING
        core_resilience.write_core_resilience_to_csv(graph, core_number, core_strength, core_influence, CIS, graph_name, sample=True)
        
        value+=1
        

if __name__=="__main__":
    main()