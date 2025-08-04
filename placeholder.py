import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs

def algebraic_connectivity(G):
    if len(G) < 2 or not nx.is_connected(G):
        return 0
    return nx.algebraic_connectivity(G)

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
def compute_core_influence(G, core_num, approximate=False):
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
    
    M = csr_matrix((data, (rows, cols)), shape=(n, n))
    
    # Eigenvector (largest eigenvalue)
    vals, vecs = eigs(M, k=1, which='LM')
    r = np.real(vecs[:, 0])
    r = np.abs(r)  # ensure non-negative
    r = r / np.linalg.norm(r)
    
    return {node: r[i] for node, i in node_index.items()}

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

# --- Simulation and AUC ---
def simulate_and_auc(G, node_list):
    G_copy = G.copy()

    ac0 = algebraic_connectivity(G_copy)
    cis0 = compute_CIS(G_copy)  # Use formal CIS metric

    ac_values = [1.0]
    cis_values = [1.0]

    for node in node_list:
        if node in G_copy:
            G_copy.remove_node(node)

        ac = algebraic_connectivity(G_copy)
        cis = compute_CIS(G_copy)

        ac_values.append(ac / ac0 if ac0 > 0 else 0)
        cis_values.append(cis / cis0 if cis0 > 0 else 0)

    auc_ac = integrate.trapezoid(ac_values, dx=1)
    auc_cis = integrate.trapezoid(cis_values, dx=1)

    return auc_ac, auc_cis, ac_values, cis_values
