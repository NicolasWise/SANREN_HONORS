import json
import networkx as nx
from plotter import write_measure_to_csv
from plotter import identify_top_r_nodes
import spectral_analysis
import core_resilience
import classical_graph_measures

def populate_graph_v1_sampling(filename, graph_size):
    with open(filename) as f:
        data = json.load(f)

    G = nx.Graph()
    G.name = filename

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
        Graph = populate_graph_v1_sampling(filename, size)
        # Number of edges
        num_edges = Graph.size()
        print(f'{num_edges} edges')
        
        #Compute Classical Graph Measures
        degree_centrality_dict, closeness_centrality_dict, betweeness_centrality_dict = classical_graph_measures.compute_classical_graph_measures(Graph)
        write_measure_to_csv(degree_centrality_dict, f'Analyses/degree_centrality_{sample_name}.csv', 'Degree Centrality')
        write_measure_to_csv(closeness_centrality_dict, f'Analyses/closeness_centrality_{sample_name}.csv', 'Closeness Centrality')
        write_measure_to_csv(betweeness_centrality_dict, f'Analyses/betweeness_centrality_{sample_name}.csv', 'Betweeness Centrality')
        top_degree_nodes = identify_top_r_nodes(degree_centrality_dict)
        top_closeness_nodes = identify_top_r_nodes(closeness_centrality_dict)
        top_betweeness_nodes = identify_top_r_nodes(betweeness_centrality_dict)
        write_measure_to_csv(top_betweeness_nodes, f'Analyses/top_r_betweeness_centrality_{sample_name}.csv', 'Betweeness Centrality')
        write_measure_to_csv(top_degree_nodes, f'Analyses/top_r_degree_centrality_{sample_name}.csv', 'Degree Centrality')
        write_measure_to_csv(top_closeness_nodes, f'Analyses/top_r_closeness_centrality_{sample_name}.csv', 'Closeness Centrality')
        
        #Spectral Analysiss
        eigenvalue_one_cluster_density, algebraic_connectivity, eigenvalue_one_multiplicity, eigenvalue_zero_multiplicity = spectral_analysis.compute_spectral_analysis(Graph)
        # MUST CHANGE WRITING 
        spectral_analysis.write_spectral_to_output_file(Graph, algebraic_connectivity, eigenvalue_one_cluster_density, eigenvalue_one_multiplicity, eigenvalue_zero_multiplicity, sample_name)
        
        #Core Resilience Analysis
        core_number, core_strength, core_influence, CIS = core_resilience.compute_core_resilience(Graph)
        top_core_numbers = identify_top_r_nodes(core_number)
        top_core_influences = identify_top_r_nodes(core_influence)
        top_core_strengths = identify_top_r_nodes(core_strength)
        write_measure_to_csv(top_core_numbers,f'Analyses/top_r_core_numbers_{sample_name}.csv', 'Core Number')
        write_measure_to_csv(top_core_influences,f'Analyses/top_r_core_influences_{sample_name}.csv', 'Core Influence')
        write_measure_to_csv(top_core_strengths,f'Analyses/top_r_core_strengths_{sample_name}.csv', 'Core Strength')
        # MUST CHANGE WRITING
        core_resilience.write_core_resilience_to_csv(Graph, core_number, core_strength, core_influence, CIS, sample_name)
        
        value+=1
        

if __name__=="__main__":
    main()