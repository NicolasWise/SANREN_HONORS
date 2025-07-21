import networkx as nx
import csv

def compute_classical_graph_measures(Graph):
    #Compute Classical Graph-Theoretic Measures
    degree_centrality_dict = nx.degree_centrality(Graph)
 
    closeness_centrality_dict = nx.closeness_centrality(Graph)
 
    betweeness_centrality_dict = nx.betweenness_centrality(Graph)

    return degree_centrality_dict, closeness_centrality_dict, betweeness_centrality_dict


def write_measured_to_csv(data_dict, filename, metric_name):
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

