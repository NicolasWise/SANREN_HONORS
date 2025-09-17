import networkx as nx
import csv

def compute_classical_graph_measures(graph):
    """ Compute degree, closeness, and betweenness centrality for the graph. """
    degree_centrality_dict = nx.degree_centrality(graph)
    closeness_centrality_dict = nx.closeness_centrality(graph)
    betweeness_centrality_dict = nx.betweenness_centrality(graph)

    return degree_centrality_dict, closeness_centrality_dict, betweeness_centrality_dict




