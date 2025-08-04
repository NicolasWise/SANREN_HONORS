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

    return eigenvalues, eigenvalue_one_cluster_density, algebraic_connectivity, eigenvalue_one_multiplicity, eigenvalue_zero_multiplicity

def write_spectral_to_output_file(graph, algebraic_connectivity, eigenvalue_one_cluster_density, eigenvalue_one_multiplicity, eigenvalue_zero_multiplicity, sample_name='',sample=False, tgf= False, json=False):
    if sample:
        output_file = f'Analyses/Samples/Sample_spectral_results.csv'
        data = [{'Name': f'{graph.name}_{sample_name}', 'Algebraic Connectivity': algebraic_connectivity, 'Multiplicity of the one eigenvalue':eigenvalue_one_multiplicity,
           'Density of eigenvalues around 1':eigenvalue_one_cluster_density, 'Multiplicity of the zero eigenvalue':eigenvalue_zero_multiplicity}]
    elif tgf:
        output_file= f'Analyses/TGF_Files/TGF_spectral_results.csv'
        data = [{'Name': f'{graph.name}', 'Algebraic Connectivity': algebraic_connectivity, 'Multiplicity of the one eigenvalue':eigenvalue_one_multiplicity,
           'Density of eigenvalues around 1':eigenvalue_one_cluster_density, 'Multiplicity of the zero eigenvalue':eigenvalue_zero_multiplicity}]
    elif json:
        output_file = f'Analyses/JSON_Files/JSON_spectral_results.csv'
        data = [{'Name': f'{graph.name}', 'Algebraic Connectivity': algebraic_connectivity, 'Multiplicity of the one eigenvalue':eigenvalue_one_multiplicity,
           'Density of eigenvalues around 1':eigenvalue_one_cluster_density, 'Multiplicity of the zero eigenvalue':eigenvalue_zero_multiplicity}]
    # Prepare data for CSV
    

    header = ['Name', 'Algebraic Connectivity', 'Multiplicity of the one eigenvalue',
              'Density of eigenvalues around 1', 'Multiplicity of the zero eigenvalue']

    # Write to CSV (append if exists, else create)
    file_exists = not os.path.isfile(output_file)

    with open(f"{output_file}", 'a', newline='') as csvfile:
        csv_writer = csv.DictWriter(csvfile, fieldnames=header, delimiter=';')
        if file_exists:
            csv_writer.writeheader() # Writes the header row
        csv_writer.writerows(data)

def plot_spectral_graphs(eigenvalues, graph_name, tgf=False, sample = False, json=False):
    plt.figure(figsize=(8, 4))
    plt.plot(range(len(eigenvalues)), eigenvalues, marker='o')
    plt.axhline(1, color='red', linestyle='--', label='λ = 1')
    if tgf:
        plt.title(f"Plots/TGF_Files/Eigenvalues_of_Laplacian_Matrix_{graph_name}.png")
    elif sample:
        plt.title(f"Plots/Samples/Eigenvalues_of_Laplacian_Matrix_{graph_name}.png")
    elif json:
        plt.title(f"Plots/JSON_files/Eigenvalues_of_Laplacian_Matrix_{graph_name}.png")
    plt.xlabel("Index")
    plt.ylabel("Eigenvalue")
    plt.legend()
    plt.grid(True)
    if tgf:
        plt.savefig(f"Plots/TGF_Files/Eigenvalues_of_Laplacian_Matrix_{graph_name}.png")
    elif sample:
        plt.savefig(f"Plots/Samples/Eigenvalues_of_Laplacian_Matrix_{graph_name}.png")
    elif json:
        plt.savefig(f"Plots/JSON_files/Eigenvalues_of_Laplacian_Matrix_{graph_name}.png")

    # Plot eigenvalue distribution
    plt.figure(figsize=(8, 4))
    plt.hist(eigenvalues, bins=10, edgecolor='black')
    if tgf:
        plt.title(f"Plots/TGF_Files/Eigenvalue_Distribution_of_Laplacian_Matrix_{graph_name}.png")
    elif sample:
        plt.title(f"Plots/Samples/Eigenvalue_Distribution_of_Laplacian_Matrix_{graph_name}.png")
    elif json:
        plt.title(f"Plots/JSON_files/Eigenvalue_Distribution_of_Laplacian_Matrix_{graph_name}.png")
    plt.xlabel("Eigenvalue")
    plt.ylabel("Frequency")
    plt.grid(True)
    if tgf:
        plt.savefig(f"Plots/TGF_Files/Eigenvalue_Distribution_of_Laplacian_Matrix_{graph_name}.png")
    elif sample:
        plt.savefig(f"Plots/Samples/Eigenvalue_Distribution_of_Laplacian_Matrix_{graph_name}.png")
    elif json:
        plt.savefig(f"Plots/JSON_files/Eigenvalue_Distribution_of_Laplacian_Matrix_{graph_name}.png")