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

def export_graph(Graph):
    with open(f"{Graph.name}_graph_edges.csv", "w", newline='') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow(['Source', 'Target'])
        for u, v in Graph.edges():
            writer.writerow([u, v])

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



def identify_top_r_nodes(dict, r = 45):
    top_nodes = {}
    count = len(dict)
    top_n = max(1, int(count*r/100))
    sorted_nodes = sorted(dict.items(), key=lambda x:x[1], reverse=True)
    
    for node, value in sorted_nodes[:top_n]:
        top_nodes[node] = value

    return top_nodes

def identify_bottom_r_nodes(dict, r = 45):
    bottom_nodes = {}
    count = len(dict)
    top_n = max(1, int(count*r/100))
    sorted_nodes = sorted(dict.items(), key=lambda x:x[1], reverse=False)
    
    for node, value in sorted_nodes[:top_n]:
        bottom_nodes[node] = value

    return bottom_nodes

def main():
    #input = 'bfn.tgf'
    inputs = ['bfn.tgf', 'cpt.tgf', 'dur.tgf', 'els.tgf', 'jnb.tgf', 'pta.tgf', 'pzb.tgf', 'vdp.tgf', 'isis-links.json',]
    for input in inputs:
        filename = f'Graph_files/{input}'

        splitname = filename.split('.',2)
        filetype = splitname[1]

        if filetype == 'tgf':
            filename = f'Graph_files/TGF_Files/{input}'
            graph = populate_graph_tgf(filename)
            plot_graph(graph, tgf=True)

            eigenvalues, e1_cluster, a_G, e1_mult, e0_mult=spectral_analysis.compute_spectral_analysis(graph)
            spectral_analysis.plot_spectral_graphs(eigenvalues, graph_name=graph.name, tgf=True)
            spectral_analysis.write_spectral_to_output_file(graph, a_G, e1_cluster, e1_mult,e0_mult, tgf=True)

            core_number, core_strength, core_influence, CIS = core_resilience.compute_core_resilience(graph)

            top_core_numbers = identify_top_r_nodes(core_number, r = 45)
            top_core_influences = identify_top_r_nodes(core_influence, r = 45)
            top_core_strengths = identify_top_r_nodes(core_strength, r = 45)

            bottom_core_numbers = identify_bottom_r_nodes(core_number, r = 45)
            bottom_core_strengths = identify_bottom_r_nodes(core_strength, r = 45)
            bottom_core_influence = identify_bottom_r_nodes(core_influence, r = 45)

            write_measure_to_csv(top_core_numbers,f'Analyses/TGF_Files/{graph.name}_top_r_core_numbers.csv', 'Core Number')
            write_measure_to_csv(top_core_influences,f'Analyses/TGF_Files/{graph.name}_top_r_core_influences.csv', 'Core Influence')
            write_measure_to_csv(top_core_strengths,f'Analyses/TGF_Files/{graph.name}_top_r_core_strengths.csv', 'Core Strength')

            write_measure_to_csv(bottom_core_numbers,f'Analyses/TGF_Files/{graph.name}_bottom_r_core_numbers.csv', 'Core Number')
            write_measure_to_csv(bottom_core_strengths,f'Analyses/TGF_Files/{graph.name}_bottom_r_core_influences.csv', 'Core Influence')
            write_measure_to_csv(bottom_core_influence,f'Analyses/TGF_Files/{graph.name}_bottom_r_core_strengths.csv', 'Core Strength')


        
            core_resilience.write_core_resilience_to_csv(graph, core_number, core_strength, core_influence, CIS, sample_name=input, tgf=True)
            
            degree_dict, close_dict, bet_dict = classical_graph_measures.compute_classical_graph_measures(graph)
            
            top_degree_nodes = identify_top_r_nodes(degree_dict, r = 45)
            top_closeness_nodes = identify_top_r_nodes(close_dict, r = 45)
            top_betweeness_nodes = identify_top_r_nodes(bet_dict, r = 45)

            bottom_degree = identify_bottom_r_nodes(degree_dict, r = 45)
            bottom_closeness = identify_bottom_r_nodes(close_dict, r = 45)
            bottom_betweeness = identify_bottom_r_nodes(bet_dict, r = 45)

            write_measure_to_csv(top_betweeness_nodes, f'Analyses/TGF_Files/{graph.name}_top_r_betweeness_centrality.csv', 'Betweeness Centrality')
            write_measure_to_csv(top_degree_nodes, f'Analyses/TGF_Files/{graph.name}_top_r_degree_centrality.csv', 'Degree Centrality')
            write_measure_to_csv(top_closeness_nodes, f'Analyses/TGF_Files/{graph.name}_top_r_closeness_centrality.csv', 'Closeness Centrality')

            write_measure_to_csv(bottom_degree,f'Analyses/TGF_Files/{graph.name}_bottom_r_betweeness_centrality.csv', 'Betweeness Centrality')
            write_measure_to_csv(bottom_closeness,f'Analyses/TGF_Files/{graph.name}_bottom_r_degree_centrality.csv', 'Degree Centrality')
            write_measure_to_csv(bottom_betweeness,f'Analyses/TGF_Files/{graph.name}_bottom_r_closeness_centrality.csv', 'Closenss Centrality')

            dicts = {'Degree Centrality': degree_dict,  'Closeness Centrality':close_dict, 'Betweeness Centrality': bet_dict}
            for value, key in dicts.items():
                write_measure_to_csv(key, f'Analyses/TGF_Files/{value}_{input}.csv', value)
    
        elif filetype == 'json':
            print('json')
            
            filename = 'Graph_files/isis-links.json'
            graph = populate_graph_json(filename)
            graph_size = graph.size()
            print(f'{graph_size} edges')

            plot_graph(graph, json=True)

            
            #Compute Classical Graph Measures
            degree_centrality_dict, closeness_centrality_dict, betweeness_centrality_dict = classical_graph_measures.compute_classical_graph_measures(graph)
            write_measure_to_csv(degree_centrality_dict, f'Analyses/JSON_Files/{graph.name}_degree_centrality.csv', 'Degree Centrality')
            write_measure_to_csv(closeness_centrality_dict, f'Analyses/JSON_Files/{graph.name}_closeness_centrality.csv', 'Closeness Centrality')
            write_measure_to_csv(betweeness_centrality_dict, f'Analyses/JSON_Files/{graph.name}_betweeness_centrality.csv', 'Betweeness Centrality')
            top_degree_nodes = identify_top_r_nodes(degree_centrality_dict, r = 25)
            top_closeness_nodes = identify_top_r_nodes(closeness_centrality_dict, r = 25)
            top_betweeness_nodes = identify_top_r_nodes(betweeness_centrality_dict, r = 25)

            bottom_degree_nodes = identify_bottom_r_nodes(degree_centrality_dict, r=25)
            bottom_close_nodes = identify_bottom_r_nodes(closeness_centrality_dict, r =25)
            bottom_betweeness_nodes = identify_bottom_r_nodes(betweeness_centrality_dict, r = 25)

            write_measure_to_csv(top_betweeness_nodes, f'Analyses/JSON_Files/{graph.name}_top_r_betweeness_centrality.csv', 'Betweeness Centrality')
            write_measure_to_csv(top_degree_nodes, f'Analyses/JSON_Files/{graph.name}_top_r_degree_centrality.csv', 'Degree Centrality')
            write_measure_to_csv(top_closeness_nodes, f'Analyses/JSON_Files/{graph.name}_top_r_closeness_centrality.csv', 'Closeness Centrality')

            write_measure_to_csv(bottom_betweeness_nodes, f'Analyses/JSON_Files/{graph.name}_bottom_r_betweeness_centrality.csv', 'Betweeness Centrality')
            write_measure_to_csv(bottom_degree_nodes, f'Analyses/JSON_Files/{graph.name}_bottom_r_degree_centrality.csv', 'Degree Centrality')
            write_measure_to_csv(bottom_close_nodes, f'Analyses/JSON_Files/{graph.name}_bottom_r_closeness_centrality.csv', 'Closeness Centrality')

            #Spectral Analysiss
            eigenvalues, eigenvalue_one_cluster_density, algebraic_connectivity, eigenvalue_one_multiplicity, eigenvalue_zero_multiplicity = spectral_analysis.compute_spectral_analysis(graph)
            spectral_analysis.write_spectral_to_output_file(graph, algebraic_connectivity, eigenvalue_one_cluster_density, eigenvalue_one_multiplicity, eigenvalue_zero_multiplicity, json=True)
            spectral_analysis.plot_spectral_graphs(eigenvalues, graph.name, json=True)
            #Core Resilience Analysis
            core_number, core_strength, core_influence, CIS = core_resilience.compute_core_resilience(graph)
            core_resilience.write_core_resilience_to_csv(graph, core_number, core_strength, core_influence, CIS, json=True)
            top_core_numbers = identify_top_r_nodes(core_number, r = 25)
            top_core_influences = identify_top_r_nodes(core_influence, r = 25)
            top_core_strengths = identify_top_r_nodes(core_strength, r = 25)

            bottom_core_numbers = identify_bottom_r_nodes(core_number, r = 25)
            bottom_core_influences = identify_bottom_r_nodes(core_influence, r = 25)
            bottom_core_strengths = identify_bottom_r_nodes(core_strength, r = 25)

            write_measure_to_csv(top_core_numbers,f'Analyses/JSON_Files/{graph.name}_top_r_core_numbers.csv', 'Core Number')
            write_measure_to_csv(top_core_influences,f'Analyses/JSON_Files/{graph.name}_top_r_core_influences.csv', 'Core Influence')
            write_measure_to_csv(top_core_strengths,f'Analyses/JSON_Files/{graph.name}_top_r_core_strengths.csv', 'Core Strength')

            write_measure_to_csv(bottom_core_numbers,f'Analyses/JSON_Files/{graph.name}_bottom_r_core_numbers.csv', 'Core Number')
            write_measure_to_csv(bottom_core_influences,f'Analyses/JSON_Files/{graph.name}_bottom_r_core_influences.csv', 'Core Influence')
            write_measure_to_csv(bottom_core_strengths,f'Analyses/JSON_Files/{graph.name}_bottom_r_core_strengths.csv', 'Core Strength')

            ### Must apply bottom r nodes
            
        elif filetype == 'txt':
            print('txt')
if __name__=="__main__":
    main()




