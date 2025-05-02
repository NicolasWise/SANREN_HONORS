import json
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

with open('honors_project_sanren/CapeTown_Graph_Abbreviated.json') as f:
    data = json.load(f)

G = nx.Graph()

for node in data["nodes"]:
    G.add_node(node["ID"])

for edge in data['edges']:
    G.add_edge(edge["SOURCE"], edge["DESTINATION"])

pos = nx.get_node_attributes(G, 'graph')

L = nx.laplacian_matrix(G)

L_dense = L.todense()
print("Laplacian Matrix:")
print(L)

nx.draw(G, with_labels = True, node_size=400, font_size = 8)
plt.show()


