import networkx as nx
import csv

def compute_classical_graph_measures(Graph):
    #Compute Classical Graph-Theoretic Measures
    degree_centrality_dict = nx.degree_centrality(Graph)
 
    closeness_centrality_dict = nx.closeness_centrality(Graph)
 
    betweeness_centrality_dict = nx.betweenness_centrality(Graph)

    return degree_centrality_dict, closeness_centrality_dict, betweeness_centrality_dict




