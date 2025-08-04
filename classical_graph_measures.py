import networkx as nx
import csv

def compute_classical_graph_measures(graph):
    #Compute Classical Graph-Theoretic Measures
    degree_centrality_dict = nx.degree_centrality(graph)
 
    closeness_centrality_dict = nx.closeness_centrality(graph)
 
    betweeness_centrality_dict = nx.betweenness_centrality(graph)

    return degree_centrality_dict, closeness_centrality_dict, betweeness_centrality_dict




