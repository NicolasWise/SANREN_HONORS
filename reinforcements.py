"""
reinforcements.py

Purpose
-------
Add edges to a given SANReN subgraph using several reinforcement strategies,
and after each step of node additions, re-run node-removal experiments to measure
resilience. We summarize resilience with AUC metrics of:

  - AUC_aG      : algebraic connectivity a(G)   (higher is better)
  - AUC_e0_mult : multiplicity of eigenvalue 0 (lower is better; fewer components)
  - AUC_e1_mult : multiplicity of eigenvalue 1 (higher is better; more redundancy)
  - AUC_CIS     : core-influence strength      (higher is better)

Workflow
--------
1) Load graph (TGF or JSON).
2) Choose an adaptive budget: (total_additions, per_step) via node count.
   - Large graphs (JSON or n >= 200): total 50 edges, 10 per step.
   - Small graphs (TGFs)            : total  5 edges,  1 per step.
3) For each reinforcement strategy:
   a) At each step, add edges picked by the strategy by the per_step value.
   b) Compute Area Under the Curve for that step for each metric(before vs after additions).
   c) Run the full removal suite (random, CI, degree, betweenness, closeness).
   d) Compute per-strategy AUCs and save plots/CSV summaries.
4) Repeat steps until the budget is exhausted or no non-edges remain.

Outputs
-------
Reinforcements/<TGF|JSON>/<graph>/<strategy>/
  - step_XX_removals.csv              : all trajectories from the removal runs
  - step_XX_AUC_summary.csv           : AUCs per removal strategy at this step
  - step_XX_compare_<metric>.jpg      : small-multiples plots per metric
  - _summary_steps.csv                : one-row-per-step summary (edges added, ΔaG, AUCs)

Notes
-----
- “Step” means one evaluation cycle; on large graphs a step can add multiple edges.
- AUC is the mean of the trajectory values over removals.

Nicolas Wise
"""
import os
import math
import random
from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh

import spectral_analysis as spec
import core_resilience as core
import classical_graph_measures as classic
import plotter as plot



def removal_random(graph: nx.Graph):
    nodes = list(graph.nodes())
    random.shuffle(nodes)
    return nodes

def removal_top_k_core_influence(graph: nx.Graph):
    _, _, ci, _ = core.compute_core_resilience(graph)
    return {n: v for n, v in sorted(ci.items(), key=lambda x: -x[1])}

def removal_top_k_degree(graph: nx.Graph):
    deg = dict(graph.degree())
    return {n: v for n, v in sorted(deg.items(), key=lambda x: -x[1])}

def removal_top_k_betweeness(graph: nx.Graph):
    _, _, bet = classic.compute_classical_graph_measures(graph)
    return {n: v for n, v in sorted(bet.items(), key=lambda x: -x[1])}

def removal_top_k_closeness(graph: nx.Graph):
    _, clo, _ = classic.compute_classical_graph_measures(graph)
    return {n: v for n, v in sorted(clo.items(), key=lambda x: -x[1])}

REMOVAL_STRATEGIES = [
    (removal_random, 'random'),
    (removal_top_k_core_influence, 'core influence'),
    (removal_top_k_degree, 'degree'),
    (removal_top_k_betweeness, 'betweeness'),
    (removal_top_k_closeness, 'closeness'),
]

def _auc_series(s: pd.Series) -> float:
    """ Compute AUC of a pandas Series using the trapezoidal rule. """
    return float(s.mean()) if len(s) else float('nan')

def add_cumulative_auc(df: pd.DataFrame, metrics=('aG', 'e0_mult', 'e1_mult', 'CIS')) -> pd.DataFrame:
    """ Add cumAUC_<metric> columns to the DataFrame. """
    df = df.sort_values(['strategy', 'removed']).copy()
    for m in metrics:
        df[f'cumAUC_{m}'] = (
            df.groupby('strategy')[m]
              .apply(lambda s: s.expanding().mean())
              .reset_index(level=0, drop=True)
        )
    return df

def summarize_auc(df: pd.DataFrame, graph_name: str, metrics=('aG','e0_mult','e1_mult','CIS')) -> pd.DataFrame:
    """ Summarize AUC per strategy as one-row DataFrame. """
    rows = []
    for strat, sub in df.groupby('strategy'):
        row = {'graph': graph_name, 'strategy': strat, 'steps': int(sub['removed'].max())}
        for m in metrics:
            row[f'AUC_{m}'] = _auc_series(sub[m])
        rows.append(row)
    return pd.DataFrame(rows)

def add_vs_random(summary_df: pd.DataFrame, metrics=('aG','e0_mult','e1_mult','CIS')) -> pd.DataFrame:
    """ Add Delta_X_vs_random columns to the summary DataFrame.     """
    out = summary_df.copy()
    if 'random' not in set(out['strategy']):
        return out
    base = out[out['strategy']=='random'].iloc[0]
    for m in metrics:
        out[f'Delta_{m}_vs_random'] = out[f'AUC_{m}'] - base[f'AUC_{m}']
    return out


def plot_metric_small_multiples(df, metric, outpath, max_cols=3):
    """ Small-multiples plot of the given metric over removal steps, one subplot per strategy. """
    strats = df['strategy'].unique()
    n = len(strats)
    cols = min(n, max_cols)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows), sharex=True, sharey=True)
    axes = axes.flatten() if n > 1 else [axes]

    for ax, strat in zip(axes, strats):
        sub = df[df['strategy'] == strat]
        ax.plot(sub['removed'], sub[metric], marker='o', linestyle='-')
        ax.set_title(strat)
        ax.grid(True)
        ax.set_xlabel('Removed')
        ax.set_ylabel(metric)

    for ax in axes[n:]:
        fig.delaxes(ax)

    fig.tight_layout(rect=[0,0,1,0.95])
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def simulate_strategy(graph: nx.Graph, strategy_fn, strategy_name: str) -> pd.DataFrame:
    """ Simulate node removals on the graph using the given removal strategy function."""
    order_raw = strategy_fn(graph)
    order = list(order_raw.keys()) if isinstance(order_raw, dict) else list(order_raw)

    G = graph.copy()
    records = []

    for k, node in enumerate(order, start=1):
        if G.number_of_nodes() < 2:
            break

        eigs, cluster, aG, e1_mult, e0_mult = spec.compute_spectral_analysis(G)
        _, _, _, CIS = core.compute_core_resilience(G)

        records.append({
            'node_id': node,
            'removed': k,
            'remaining': G.number_of_nodes(),
            'aG': aG,
            'e1_mult': e1_mult,
            'e0_mult': e0_mult,
            'CIS': CIS,
            'strategy': strategy_name
        })

        G.remove_node(node)
        if G.number_of_nodes() == 0:
            break

    return pd.DataFrame(records)

def run_all_removals(graph: nx.Graph) -> pd.DataFrame:
    """ Run all removal strategies on the graph and return concatenated results. """
    per = []
    for fn, name in REMOVAL_STRATEGIES:
        df = simulate_strategy(graph, fn, name)
        df = add_cumulative_auc(df, metrics=('aG','e0_mult','e1_mult','CIS'))
        per.append(df)
    return pd.concat(per, ignore_index=True)


# Fiedler greedy (edge add)
def fiedler_vector(G: nx.Graph):
    """ Compute a Fiedler vector: the eigenvector associated with the second-smallest
    eigenvalue (λ2) of the *combinatorial Laplacian* L = D - A.

    Intuition:
      - The Fiedler vector (a.k.a. algebraic connectivity vector) encodes the
        graph's “tightest bottleneck”. Nodes with very negative
        vs very positive Fiedler entries tend to lie on opposite sides of a
        spectral cut; adding edges across these extremes is a principled way to
        raise λ2 (algebraic connectivity), improving global connectivity and improving resilience.

    Implementation details:
      - For tiny graphs (n<3), fall back to dense eigendecomposition.
      - For general n, use ARPACK (eigsh) with `which='SM'` (smallest magnitude)
        and k=2; the smallest eigenvalue is 0 with eigenvector 1, so the second
        is the Fiedler value/vector (provided the graph is connected).
      - `v0` is an optional warm-start vector. If you add edges incrementally,
        the Laplacian changes only slightly; reusing the previous Fiedler vector
        as a starting guess can speed up convergence significantly.

    Caveats:
      - If the graph is *disconnected*, λ=0 has multiplicity > 1, and the
        “second-smallest” eigenpair may still correspond to 0. The returned
        vector will then encode component structure rather than an intra-component
        bottleneck. Practically, that still helps: it pushes candidates to connect
        components—often a good first step—but you can also ensure connectivity
        first if you want a pure bottleneck view."""
    L = nx.laplacian_matrix(G).astype(float)
    n = G.number_of_nodes()
    if n < 3:
        # For tiny graphs, fall back to dense eigendecomposition
        arr = L.toarray()
        # Compute eigenvalues and eigenvectors
        vals, vecs = np.linalg.eigh(arr)
        # Get the index of the second smallest eigenvalue
        idx = np.argsort(vals)
        # Return the corresponding eigenvector (Fiedler vector)
        return vecs[:, idx[1]]
    try:
        # Compute the two smallest magnitude eigenvalues and their eigenvectors
        # which SM = smallest magnitude
        vals, vecs = eigsh(L, k=2, which='SM')
        # The Fiedler vector corresponds to the second smallest eigenvalue
        # Sort eigenvalues and get the index of the second smallest
        order = np.argsort(vals)
        # Return the corresponding eigenvector (Fiedler vector)
        return vecs[:, order[1]]
    except Exception:
        # Fallback to dense method if ARPACK fails (e.g., for very small graphs)
        arr = L.toarray()
        # Compute eigenvalues and eigenvectors
        vals, vecs = np.linalg.eigh(arr)
        # Get the index of the second smallest eigenvalue
        idx = np.argsort(vals)
        # Return the corresponding eigenvector (Fiedler vector)
        return vecs[:, idx[1]]

def next_edge_fiedler_greedy(G: nx.Graph):
    """ Return the non-edge (u,v) that maximizes (f[u]-f[v])^2 where f is the Fiedler vector.
    Choose up to k non-edges guided by the Fiedler vector:
        - Sort nodes by Fiedler coordinate.
        - Consider only the 'extreme' ends (lowest and highest values) to cap the
        candidate set size (O(Lcap^2) pairs vs O(n^2)).
        - Score each candidate (u, v) by (f[u] - f[v])^2 and pick the top-k.

    Intuition:
        - (f[u] - f[v])^2 measures how far apart u and v are along the Fiedler line.
        Connecting large-gap pairs approximates adding “springs” across the weakest
        cut, which tends to increase λ2 and reduce bottlenecks.

    Design choices:
        - We compute the Fiedler vector once per step (fast). The strictly optimal
        greedy strategy would recompute f after every added edge (slow but best).
        Empirically, “once per step” gives a strong improvement at a fraction of
        the cost; you can trade up to recompute if k is large or graphs are small.
        - Lcap bounds runtime: pair candidates only among the Lcap most negative and
        Lcap most positive nodes. If n < 2*Lcap, it shrinks automatically.

    Returns:
        - chosen: list of edges added [(u, v), ...]
        - f: the Fiedler vector used (returning it lets callers warm-start next step)"""
    
    # Limit candidate pairs to the Lcap most extreme nodes
    nodes = list(G.nodes())
    # Sort nodes by Fiedler coordinate.
    idx = {u:i for i,u in enumerate(nodes)}
    # Compute Fiedler vector
    f = fiedler_vector(G)
    # Cap candidate set size to Lcap most extreme nodes
    best_pair = None
    # Limit candidates to the Lcap most extreme nodes
    best_score = -1.0
    # For each pair of nodes (u, v) among the extreme nodes
    for u, v in combinations(nodes, 2):
        # Skip if (u, v) is already an edge
        if G.has_edge(u, v):
            continue
        # Score the pair by (f[u] - f[v])^2
        score = (f[idx[u]] - f[idx[v]])**2
        # Update best pair if this one is better
        if score > best_score:
            best_score = score
            # Update best pair
            best_pair = (u, v)
    # Return the best pair found and its score
    return best_pair, best_score


# Baselines (random)
def next_edge_random(G: nx.Graph):
    """ Return a random non-edge (u,v). """
    nodes = list(G.nodes())
    attempts = 0
    while attempts < 10000:
        u, v = random.sample(nodes, 2)
        if not G.has_edge(u, v):
            return (u, v), None
        attempts += 1
    return None, None


# MRKC (heuristic)
def next_edge_mrkc_heuristic(G: nx.Graph):
    """ Minimum-Redundancy k-Core heuristic for adding one edge. 
    k-core anchoring heuristic:
      - Identify the most vulnerable shell (min core number).
      - Identify the most robust anchors (max core number).
      - Connect low-κ 'vulnerable' nodes to high-κ 'anchor' nodes, tie-breaking
        by degree (promote high-degree anchors; visit low-degree vuln first).

    Intuition:
      - In k-core theory, a node's core number κ(u) depends on how many neighbors
        of comparable-or-stronger κ it has. By wiring periphery nodes (low κ) to
        anchors (high κ), you increase their high-κ neighborhood count—lifting
        local redundancy (Core Strength) and making it harder to peel them off.

    Notes & trade-offs:
      - We recompute cores once per call (not after each added edge). This is a
        fast approximation; recomputing κ after each addition would be more
        precise but slower.
      - This heuristic targets *local* resilience (CS/CIS) more directly than
        global λ2. It complements the Fiedler method, which targets spectral
        cohesion. In practice, mix-and-match or compare both via AUC metrics.

    Fallback behavior:
      - If we run out of “obvious” vuln→anchor pairs (e.g., small graphs), we
        fall back to adding arbitrary non-edges to satisfy the requested budget.

    Returns:
      - added: next edge added (u, v) or None if no non-edges remain"""
    
    # Compute core numbers and degrees
    core_num = nx.core_number(G)
    # Compute degrees
    deg = dict(G.degree())
    # Identify vuln and anchor sets
    min_k = min(core_num.values())
    # Vulnerable nodes are those with the minimum core number
    vuln = [u for u, k in core_num.items() if k == min_k]
    # Sort vuln by increasing degree (tie-break by node ID for determinism)
    vuln.sort(key=lambda u: (deg[u], u))
    # Identify anchors
    max_k = max(core_num.values())
    # Anchors are those with the maximum core number
    anchors = [v for v, k in core_num.items() if k == max_k]
    # Sort anchors by decreasing degree (tie-break by node ID for determinism)
    anchors.sort(key=lambda v: (-deg[v], v))
    # For each vuln node, try to connect to each anchor node
    for u in vuln:
        # For each anchor node
        for v in anchors:
            # Skip if u == v or (u, v) is already an edge
            if u != v and not G.has_edge(u, v):
                # Return the first valid (u, v) found
                return (u, v), (core_num[u], core_num[v])
    # Fallback: if no vuln→anchor pair found, return any random non-edge
    for u, v in combinations(G.nodes(), 2):
        if not G.has_edge(u, v):
            return (u, v), (core_num[u], core_num[v])
    return None, None



def choose_budget_by_size(filename: str, G: nx.Graph):
    """
    Return (total_additions, edges_per_step).
    Policy:
      - Large graphs (JSON or >= 200 nodes): total 50, add 10 per step.
      - Small graphs (TGF): total 5, add 1 per step.
    Adjust thresholds as you like.
    """
    ext = os.path.splitext(filename)[1].lower()
    n = G.number_of_nodes()
    if ext == '.json' or n >= 200:
        return 100, 10
    else:
        return 10, 1


def reinforce_and_evaluate(graph: nx.Graph,
                           strategy_name: str,
                           next_edge_fn,
                           total_additions: int,
                           per_step: int,
                           outdir: str,
                           improvements_csv: str = None):
    """
    Adds edges iteratively; at each step add up to `per_step` edges (until `total_additions` reached),
    then run removals and write AUC/plots. Returns per-step summary DataFrame.
    """
    os.makedirs(outdir, exist_ok=True)
    G = graph.copy()

    _, _, aG0, _, _ = spec.compute_spectral_analysis(G)

    step_rows = []
    added_total = 0
    num_steps = math.ceil(total_additions / per_step)

    for step in range(1, num_steps + 1):
        if added_total >= total_additions:
            break

        # metrics before this step's additions
        _, _, aG_before, _, _ = spec.compute_spectral_analysis(G)

        edges_this_step = []
        # inner loop: add up to `per_step` edges
        for _ in range(per_step):
            if added_total >= total_additions:
                break
            else:
                pair, score = next_edge_fn(G)

            if not pair:
                # no more candidates
                break

            u, v = pair
            G.add_edge(u, v)
            edges_this_step.append((u, v))
            added_total += 1

        if not edges_this_step:
            print(f"[{strategy_name}] No more candidate non-edges at step {step}.")
            break

        # metrics after additions
        _, _, aG_after, e1_after, e0_after = spec.compute_spectral_analysis(G)
        delta_aG = aG_after - aG_before

        # run removals & AUC
        rem_df = run_all_removals(G)
        auc_summary = summarize_auc(rem_df, graph_name=graph.name)
        auc_summary = add_vs_random(auc_summary)

        # plots for this step
        for metric in ['aG', 'e0_mult', 'e1_mult', 'CIS']:
            plot_metric_small_multiples(
                rem_df, metric,
                os.path.join(outdir, f'step_{step:02d}_compare_{metric}.jpg')
            )

        # persist CSVs
        rem_df.to_csv(os.path.join(outdir, f'step_{step:02d}_removals.csv'), index=False)
        auc_summary.round(5).to_csv(os.path.join(outdir, f'step_{step:02d}_AUC_summary.csv'), index=False)

        # summarize this step
        row = {
            'graph': graph.name,
            'strategy': strategy_name,
            'step': step,
            'num_edges_added_this_step': len(edges_this_step),
            'edges_added': ';'.join([f'{u}--{v}' for (u, v) in edges_this_step]),
            'aG_before': aG_before,
            'aG_after': aG_after,
            'Delta_aG': delta_aG,
            'aG_baseline0': aG0
        }
        for _, r in auc_summary.iterrows():
            tag = r['strategy'].replace(' ', '_')
            for col in ['AUC_aG','AUC_e0_mult','AUC_e1_mult','AUC_CIS',
                        'Delta_aG_vs_random','Delta_e0_mult_vs_random','Delta_e1_mult_vs_random','Delta_CIS_vs_random']:
                if col in r:
                    row[f'{col}__{tag}'] = r[col]
        step_rows.append(row)

    summary_df = pd.DataFrame(step_rows)
    if not summary_df.empty:
        summary_df_rounded = summary_df.copy()
        for c in summary_df_rounded.columns:
            if np.issubdtype(summary_df_rounded[c].dtype, np.floating):
                summary_df_rounded[c] = summary_df_rounded[c].round(5)
        summary_df_rounded.to_csv(os.path.join(outdir, f'_summary_steps.csv'), index=False)

    return summary_df


def run_reinforcements_for_graph(filename: str,
                                 base_graph_dir_tgf='Graph_files/TGF_Files',
                                 base_graph_dir_json='Graph_files',
                                 improvements_dir='Improvements'):
    """ Run all reinforcement strategies on the given graph file. """

    filetype = filename.split('.')[-1]
    path, subdir = ((os.path.join(base_graph_dir_tgf, filename), 'TGF_Files')
                    if filetype == 'tgf'
                    else (os.path.join(base_graph_dir_json, filename), 'JSON_Files'))
    G = plot.load_graph(path, filetype)
    graph_name = os.path.splitext(os.path.basename(filename))[0]

    # adaptive budget
    total_additions, per_step = choose_budget_by_size(filename, G)

    base_out = os.path.join('Reinforcements', subdir, graph_name)
    os.makedirs(base_out, exist_ok=True)

    imp_csv = os.path.join(improvements_dir, f'{graph_name}_lowest_connectivity.csv')

    # 1) Fiedler-greedy
    reinforce_and_evaluate(
        G, 'fiedler_greedy', next_edge_fiedler_greedy,
        total_additions, per_step,
        outdir=os.path.join(base_out, 'fiedler_greedy'),
    )

    # 2) Baseline: random edge adds
    random.seed(42)
    reinforce_and_evaluate(
        G, 'random_add', next_edge_random,
        total_additions, per_step,
        outdir=os.path.join(base_out, 'random_add'),
    )

    # 3) MRKC heuristic
    reinforce_and_evaluate(
        G, 'mrkc_heuristic', next_edge_mrkc_heuristic,
        total_additions, per_step,
        outdir=os.path.join(base_out, 'mrkc_heuristic'),
    )

# ----------------------------
# Batch runner
# ----------------------------
def main():
    inputs = [
        'bfn.tgf', 'cpt.tgf', 'dur.tgf', 'els.tgf',
        'jnb.tgf', 'pta.tgf', 'pzb.tgf', 'vdp.tgf',
        'isis-links.json'
    ]
    for fn in inputs:
        run_reinforcements_for_graph(fn)

if __name__ == "__main__":
    main()
