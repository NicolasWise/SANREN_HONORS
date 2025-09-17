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
    Definition: Core Strength: CS(u) = |{v ∈ Γ(u): κ(v) ≥ κ(u)}| - κ(u) + 1
    
    Intuition:
        - A node with core number κ(u) sits in a κ(u)-core, which loosely
          means it needs ~κ(u) peers of comparable robustness to keep its status.
        - CS(u) counts how many of u's neighbors are at least as “deep in the
          core” as u is (κ(v) ≥ κ(u)), then subtracts the “bare minimum”
          κ(u) - 1. What's left is the *slack*: how much extra high-core
          support u has beyond just scraping by.
        - Larger CS(u) ⇒ u is harder to dislodge locally; you'd need to delete
          more of its strong neighbors before u's own core level can drop.

    Notes:
        - CS(u) is a *local redundancy* indicator: it looks only at u's immediate
          neighborhood and their core numbers, not the global topology.
        - Non-negativity: Values near 0 indicate “just enough” support;
          higher values indicate redundancy headroom.

    Parameters:
        G: NetworkX graph (simple, undirected).
        core_num: dict {node -> κ(node)} typically from nx.core_number(G).

    Returns:
        dict {node -> CS(node)} (float/int).
    """

    # Compute Core Strength for each node
    CS = {}
    # For each node u in G
    for u in G.nodes():
        # Count neighbors v with core_num[v] >= core_num[u]
        neighbors_ge = sum(1 for v in G.neighbors(u) if core_num[v] >= core_num[u])
        # Compute CS(u)
        CS[u] = neighbors_ge - core_num[u] + 1
    # Return Core Strength for all nodes
    return CS



# --- Core Influence (eigenvector-based) ---
def compute_core_influence(G, core_num, approximate=False, tol =1e-6, maxiter=5000):
    """
    Compute Core Influence using eigenvector of M matrix.
    If approximate=True, ignore contributions from equal-core neighbors.
    Goal:
        Measure how much “core support” a node receives when support mass
        preferentially flows from lower/equal-core nodes toward equal/higher-core
        neighbors. We then take the leading eigenvector of this flow matrix
        to identify structurally influential nodes inside the core-periphery.

    Construction of M (support-flow matrix):
        - For each edge (u, v):
            * If κ(u) ≤ κ(v), then u can funnel support into v (upwards or lateral).
              We add a contribution to M[u, v].
              The contribution is 1 / denom_v, where:
                  denom_v = |{ w in Γ(v) : κ(w) ≥ κ(v) }|
              i.e., v's pool of high-core “anchors”. This normalizes incoming
              support: a v with many strong neighbors dilutes each neighbor's share.
            * Symmetrically, if κ(v) ≤ κ(u), add a contribution to M[v, u]
              with denom_u defined analogously.
        - We then add an identity (I) to M (i.e., M ← M + I). This keeps the
          matrix strictly positive/diagonally dominant, avoiding zero rows and
          improving numerical stability so the Perron-Frobenius principal vector
          is well-behaved.

    Parameter `approximate`:
        - If True, we *ignore equal-core* contributions (κ(u) = κ(v)). This
          reduces closed reinforcement loops within the same k-shell and focuses
          the score on *cross-shell* (lower → higher) support. Use this if you
          want CI to emphasize “pull” into higher cores rather than within-shell
          popularity.

    Output vector:
        - We compute the leading eigenvector of M. Entries are then L2-normalized,
          and absolute-valued to guard against numerical sign flips.
        - Relative magnitudes matter: larger CI(u) ⇒ u is more “core-influential”.

    Practical interpretation:
        - CI highlights nodes that are (i) well supported by strong neighbors and
          (ii) sit in positions where core support tends to concentrate.
        - It's complementary to Core Strength: CS is immediate slack; CI captures
          how support *circulates and concentrates* through the core structure.

    Returns:
        dict {node -> CS(node)} (float/int).
    """

    # Number of nodes
    n = len(G.nodes())
    # Node index mapping
    node_index = {node: i for i, node in enumerate(G.nodes())}
    
    # Build sparse matrix M
    data, rows, cols = [], [], []
    # For each edge (u, v) in G
    for u, v in G.edges():
        # Get core numbers of each node
        ku, kv = core_num[u], core_num[v]
        # u influences v
        # if the core number of u is less than or equal to that of v
        if ku <= kv:
            # Count neighbors v with core_num[v] >= core_num[u]
            denom = sum(1 for w in G.neighbors(v) if core_num[w] >= core_num[v])
            if denom > 0:
                # Add contribution to M[u, v]
                if not (approximate and ku == kv):
                    # Normalize by the number of high-core neighbors of v
                    rows.append(node_index[u])
                    # Normalize by the number of high-core neighbors of v
                    cols.append(node_index[v])
                    # Normalize by the number of high-core neighbors of v
                    data.append(1.0/denom)
        # v influences u
        if kv <= ku:
            # Count neighbors u with core_num[u] >= core_num[v]
            denom = sum(1 for w in G.neighbors(u) if core_num[w] >= core_num[u])
            if denom > 0:
                # Normalize by the number of high-core neighbors of u
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
    # Create sparse matrix M
    M = csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float64)

    # find leading eigenvector of M
    # using ARPACK with fallback to power iteration
    def _leading_eigvec(M_csr: csr_matrix) -> np.ndarray:
        # tiny graphs: use dense eig directly
        if n < 3:
            A = M_csr.toarray()
            vals, vecs = np.linalg.eig(A)
            idx = np.argmax(np.real(vals))
            return np.real(vecs[:, idx])

        # ARPACK attempt
        # workspace size (ncv) > k; grow mildly with n for stability
        # (see scipy.sparse.linalg.eigs docs)
        ncv = min(n, max(20, int(2 * np.sqrt(max(n, 1))) + 1))
        # max iterations
        rng = np.random.default_rng(42)
        # random initial vector
        v0 = rng.standard_normal(n)

        try:
            # 'LR' = largest real part; better for non-symmetric positive matrices
            # (see scipy.sparse.linalg.eigs docs)
            vals, vecs = eigs(M_csr, k=1, which='LR', ncv=ncv, maxiter=maxiter, tol=tol, v0=v0)
            return np.real(vecs[:, 0])
        # ARPACK fallback
        except (ArpackNoConvergence, Exception):
            # Power iteration fallback
            # Initialize with random vector
            x = v0 / (np.linalg.norm(v0) or 1.0)
            # For up to maxiter iterations
            for _ in range(maxiter):
                # Multiply by M
                y = M_csr @ x
                # Normalize
                yn = np.linalg.norm(y)

                if yn == 0:
                    break
                # Normalize
                y /= yn
                # Check convergence
                if np.linalg.norm(y - x) < 1e-8:
                    x = y
                    break
                x = y
            # Return real part of converged vector - the leading eigenvector
            return np.real(x)
        
    # Compute leading eigenvector
    vec = _leading_eigvec(M)

    # Normalize nonnegative principal vector to unit L2 norm
    r = np.abs(vec)
    # L2 norm
    norm = np.linalg.norm(r)
    # Handle zero vector case
    if norm == 0:
        r = np.ones_like(r) / np.sqrt(n)
    else:
        # Normalize
        r /= norm
    # Map back to nodes
    return {node: float(r[i]) for node, i in node_index.items()}


def compute_CIS(G, f=0.9, approximate_CI=False):
    """
    Compute Core Influence-Strength (CIS_f)
    f: percentile threshold (0 < f <= 1)
    
    Definition:
        1) Compute κ(·) via k-core; compute CS(·) and CI(·).
        2) Let τ be the f-th percentile of CI scores (f=0.9 ⇒ top 10%).
        3) Let S_f = { u : CI(u) ≥ τ } be the “keystone” cohort.
        4) CIS_f = average_u∈S_f CS(u).

    Intuition:
        - CI picks out nodes that are structurally influential inside the core.
        - CS tells you how much *local slack* those nodes have.
        - CIS_f says: “On average, how redundant (hard to dislodge) are our most
          core-influential nodes?” Higher CIS_f ⇒ the network's key anchors are
          locally well backed, improving robustness to targeted removals.
        - Treat CIS_f as a *local redundancy* summary at the network level.
        - Pair with global redundancy such as m1 (multiplicity of eigenvalue 1 of
          the normalized Laplacian) to show: “local slack” vs “global duplication”.

    Returns:
        dict {node -> CI(node)} (float in [0,1] after L2 normalization).
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
    """ Write core resilience metrics to CSV file."""
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
    """ Compute core resilience metrics for graph G.   
        Returns:
        core_number: dict {node -> κ(node)}
        core_strength: dict {node -> CS(node)}
        core_influence: dict {node -> CI(node)}
        CIS: float (Core Influence-Strength metric)
    """
    core_number = nx.core_number(G)
    core_strength = compute_core_strength(G, core_number)
    core_influence = compute_core_influence(G, core_number)
    CIS = compute_CIS(G)
    return core_number, core_strength, core_influence, CIS

