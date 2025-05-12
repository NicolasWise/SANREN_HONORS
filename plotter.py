import json
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigvalsh

with open('CapeTown_Graph_Abbreviated.json') as f:
    data = json.load(f)

G = nx.Graph()

for node in data["nodes"]:
    G.add_node(node["ID"])

for edge in data['edges']:
    G.add_edge(edge["SOURCE"], edge["DESTINATION"])

L = nx.laplacian_matrix(G).todense()
print("Laplacian Matrix:")
print(L)

# Sorted to make algebraic connectivity derivation easier.
eigenvalues = np.sort(np.linalg.eigvalsh(L))
print("Eigenvalues:")
print(eigenvalues)

# Second highest eiganvalue
algebraic_connectivity = eigenvalues[1]
print(f"Algebraic Connectivity: {algebraic_connectivity}")

# Count how many eigenvalues fall in the range [0.9, 1.1]
cluster_count = np.sum((eigenvalues >= 0.9) & (eigenvalues <= 1.1))
total_eigenvalues = len(eigenvalues)

# Density is just proportion of eigenvalues in the range
cluster_density = cluster_count / total_eigenvalues

print(f"Number of eigenvalues clustered around 1: {cluster_count}")
print(f"Density (proportion) of clustered eigenvalues: {cluster_density:.2f}")


plt.figure(figsize=(8, 4))
plt.plot(range(len(eigenvalues)), eigenvalues, marker='o')
plt.axhline(1, color='red', linestyle='--', label='λ = 1')
plt.title("Eigenvalues of Laplacian Matrix")
plt.xlabel("Index")
plt.ylabel("Eigenvalue")
plt.legend()
plt.grid(True)
plt.savefig("Eigenvalues_of_Laplacian_Matrix")

# Plot eigenvalue distribution
plt.figure(figsize=(8, 4))
plt.hist(eigenvalues, bins=10, edgecolor='black')
plt.title("Eigenvalue Distribution of the Laplacian")
plt.xlabel("Eigenvalue")
plt.ylabel("Frequency")
plt.grid(True)
plt.savefig("Eigenvalue_Distribution_of_the_Laplacian")


