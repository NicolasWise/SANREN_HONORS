import networkx as nx
import itertools
import csv
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs
try:
    from scipy.sparse.linalg import ArpackNoConvergence
except Exception:  # older SciPy
    class ArpackNoConvergence(Exception):
        pass

"""
Core Resilience Metrics:
- Core Number: k-core decomposition number.
- Core Strength: CS(u) = |{v ∈ Γ(u): κ(v)   ≥ κ(u)}| - κ(u) + 1
- Core Influence: Eigenvector-based measure of influence within core structure.
- Core Influence-Strength (CIS): Average Core Strength of top f-percentile Core Influence

Nicolas Wise
"""


# --- Core Strength (theoretical formula) ---
def compute_core_strength(G, core_num):
    """
    Core Strength: CS(u) = |{v ∈ Γ(u): κ(v) ≥ κ(u)}| - κ(u) + 1
    """
    CS = {}
    for u in G.nodes():
        neighbors_ge = sum(1 for v in G.neighbors(u) if core_num[v] >= core_num[u])
        CS[u] = neighbors_ge - core_num[u] + 1
    return CS



# --- Core Influence (eigenvector-based) ---
def compute_core_influence(G, core_num, approximate=False, tol =1e-6, maxiter=5000):
    """
    Compute Core Influence using eigenvector of M matrix.
    If approximate=True, ignore contributions from equal-core neighbors.
    """
    n = len(G.nodes())
    node_index = {node: i for i, node in enumerate(G.nodes())}
    
    # Build sparse matrix M
    data, rows, cols = [], [], []
    for u, v in G.edges():
        ku, kv = core_num[u], core_num[v]
        # u influences v
        if ku <= kv:
            denom = sum(1 for w in G.neighbors(v) if core_num[w] >= core_num[v])
            if denom > 0:
                if not (approximate and ku == kv):
                    rows.append(node_index[u])
                    cols.append(node_index[v])
                    data.append(1.0/denom)
        # v influences u
        if kv <= ku:
            denom = sum(1 for w in G.neighbors(u) if core_num[w] >= core_num[u])
            if denom > 0:
                if not (approximate and kv == ku):
                    rows.append(node_index[v])
                    cols.append(node_index[u])
                    data.append(1.0/denom)
    
    # Add diagonal 1
    for u in G.nodes():
        idx = node_index[u]
        rows.append(idx)
        cols.append(idx)
        data.append(1.0)
    
    M = csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float64)

    # -------- Leading eigenvector of M (robust) --------
    def _leading_eigvec(M_csr: csr_matrix) -> np.ndarray:
        # tiny graphs: use dense eig directly
        if n < 3:
            A = M_csr.toarray()
            vals, vecs = np.linalg.eig(A)
            idx = np.argmax(np.real(vals))
            return np.real(vecs[:, idx])

        # ARPACK attempt
        # workspace size (ncv) > k; grow mildly with n
        ncv = min(n, max(20, int(2 * np.sqrt(max(n, 1))) + 1))
        rng = np.random.default_rng(42)
        v0 = rng.standard_normal(n)

        try:
            # 'LR' = largest real part; better for non-symmetric positive matrices
            vals, vecs = eigs(M_csr, k=1, which='LR', ncv=ncv, maxiter=maxiter, tol=tol, v0=v0)
            return np.real(vecs[:, 0])
        except (ArpackNoConvergence, Exception):
            # Power iteration fallback
            x = v0 / (np.linalg.norm(v0) or 1.0)
            for _ in range(maxiter):
                y = M_csr @ x
                yn = np.linalg.norm(y)
                if yn == 0:
                    break
                y /= yn
                if np.linalg.norm(y - x) < 1e-8:
                    x = y
                    break
                x = y
            return np.real(x)

    vec = _leading_eigvec(M)

    # -------- Normalize nonnegative principal vector --------
    r = np.abs(vec)
    norm = np.linalg.norm(r)
    if norm == 0:
        r = np.ones_like(r) / np.sqrt(n)
    else:
        r /= norm

    return {node: float(r[i]) for node, i in node_index.items()}

# --- Core Influence-Strength metric ---
def compute_CIS(G, f=0.9, approximate_CI=False):
    """
    Compute Core Influence-Strength (CIS_f)
    f: percentile threshold (0 < f <= 1)
    """
    core_num = nx.core_number(G)
    CS = compute_core_strength(G, core_num)
    CI = compute_core_influence(G, core_num, approximate=approximate_CI)
    
    # f-th percentile of CI
    CI_values = np.array(list(CI.values()))
    threshold = np.percentile(CI_values, f * 100)
    
    # Nodes with CI >= threshold
    S_f = [u for u, ci in CI.items() if ci >= threshold]
    if not S_f:
        return 0
    
    return np.mean([CS[u] for u in S_f])


def write_core_resilience_to_csv(graph, core_number, core_strength, core_influence, CIS, sample_name='',sample=False, tgf=False, json=False):
    # Extract all unique nodes from the keys of one of the dictionaries
    nodes = list(core_number.keys())
    if sample:
        output_file = f'Analyses/Samples/{sample_name}_{graph.name}_core_resilience_metrics.csv'
        CIS_output_filename = f'Analyses/Samples/CIS_metric.csv'
    elif tgf:
        output_file= f'Analyses/TGF_Files/{graph.name}_core_resilience_metrics.csv'
        CIS_output_filename = f'Analyses/TGF_Files/CIS_metric.csv'
    elif json:
        output_file = f'Analyses/JSON_Files/{graph.name}_core_resilience_metrics.csv'
        CIS_output_filename = f'Analyses/TGF_Files/CIS_metric.csv'
    # Define header
    header = ['Node', 'Core Number', 'Core Strength', 'Core Influence']

    # Prepare rows
    rows = []
    for node in nodes:
        row = {
            'Node': node,
            'Core Number': core_number.get(node, 'N/A'),
            'Core Strength': core_strength.get(node, 'N/A'),
            'Core Influence': core_influence.get(node, 'N/A')
        }
        rows.append(row)

    write_header = not os.path.exists(output_file)
    # Write to CSV (use semicolon for Excel compatibility)
    with open(f"{output_file}", 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header, delimiter=';')
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


    
    file_exists = os.path.isfile(f'{CIS_output_filename}')
    print(f"Core_Influence Strength Metric: {CIS}")
    header = ['Name', 'Core-Influence Strength Metric']
    with open(f'{CIS_output_filename}', "a" ,newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, delimiter=';', fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerow({'Name': f'{graph.name}', 'Core-Influence Strength Metric': CIS})

    print(f"Core resilience data written to {CIS_output_filename}")

def compute_core_resilience(G):
    '''A node's core number is the highest k for which it remains in the k-core, reflecting its structural depth and resilience.'''
    core_number = nx.core_number(G)

    core_strength = compute_core_strength(G, core_number)

    core_influence = compute_core_influence(G, core_number)
    
    CIS = compute_CIS(G)

    return core_number, core_strength, core_influence, CIS

