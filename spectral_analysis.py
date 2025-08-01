import networkx as nx
import csv
import os
import numpy as np
import matplotlib.pyplot as plt

def compute_spectral_analysis(G):
    print(f"{G.name}:")
    L = nx.laplacian_matrix(G).todense()
   
    # Sorted to make algebraic connectivity derivation easier.
    eigenvalues = np.sort(np.linalg.eigvalsh(L))

    # Second highest eigenvalue
    algebraic_connectivity = eigenvalues[1]

    # Count how many eigenvalues fall in the range [0.9, 1.1]
    eigenvalue_one_multiplicity = np.sum((eigenvalues >= 0.9) & (eigenvalues <= 1.1))

    # Set a small threshold to account for the floating-point imprecision
    threshold = 1e-10
    eigenvalue_zero_multiplicity = np.sum(np.isclose(eigenvalues, 0.0, atol=threshold))
    
    total_eigenvalues = len(eigenvalues)
    # Density is just proportion of eigenvalues in the range
    eigenvalue_one_cluster_density = eigenvalue_one_multiplicity / total_eigenvalues

    return eigenvalue_one_cluster_density, algebraic_connectivity, eigenvalue_one_multiplicity, eigenvalue_zero_multiplicity

def write_spectral_to_output_file(G, algebraic_connectivity, eigenvalue_one_cluster_density, eigenvalue_one_multiplicity, eigenvalue_zero_multiplicity, sample_name=''):
    output_file = 'spectral_results.csv'
    # Prepare data for CSV
    data = [{'Name': f'{G.name}_{sample_name}', 'Algebraic Connectivity': algebraic_connectivity, 'Multiplicity of the one eigenvalue':eigenvalue_one_multiplicity,
           'Density of eigenvalues around 1':eigenvalue_one_cluster_density, 'Multiplicity of the zero eigenvalue':eigenvalue_zero_multiplicity}]

    header = ['Name', 'Algebraic Connectivity', 'Multiplicity of the one eigenvalue',
              'Density of eigenvalues around 1', 'Multiplicity of the zero eigenvalue']

    # Write to CSV (append if exists, else create)
    file_exists = os.path.isfile(output_file)

    with open(f"Analyses/{output_file}", 'a', newline='') as csvfile:
        csv_writer = csv.DictWriter(csvfile, fieldnames=header, delimiter=';')
        if not file_exists:
            csv_writer.writeheader() # Writes the header row
        csv_writer.writerows(data)

def plot_spectral_graphs(eigenvalues):
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