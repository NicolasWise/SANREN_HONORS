# reinforce_no_removal.py
#
# Minimal runner: add edges via several reinforcement strategies and,
# after each step, compute metrics on the *reinforced* graph only
# (no removal experiments). Then plot a(G), m0, m1, CIS trajectories
# with one line per reinforcement strategy.

import os
import math
import random
from itertools import combinations

import argparse
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh

# --- Your project modules (rename if your paths differ)
import plotter as plot
import spectral_analysis as spec
import core_resilience as core


# ----------------------------
# Utility: metrics on current graph
# ----------------------------
def compute_metrics(G: nx.Graph):
    """
    Returns dict: {'aG': ..., 'e0_mult': ..., 'e1_mult': ..., 'CIS': ...}
    """
    _, _, aG, e1_mult, e0_mult = spec.compute_spectral_analysis(G)
    _, _, _, CIS = core.compute_core_resilience(G)
    return {"aG": float(aG), "e0_mult": float(e0_mult), "e1_mult": float(e1_mult), "CIS": float(CIS)}


# ----------------------------
# Reinforcement strategies
# ----------------------------
def fiedler_vector(G: nx.Graph, v0=None):
    """Fiedler vector (2nd smallest eigenvector of Laplacian). Uses eigsh with optional warm-start."""
    L = nx.laplacian_matrix(G).astype(float)
    n = G.number_of_nodes()
    if n < 3:
        arr = L.toarray()
        vals, vecs = np.linalg.eigh(arr)
        idx = np.argsort(vals)
        return vecs[:, idx[1]]
    try:
        vals, vecs = eigsh(L, k=2, which='SM', v0=v0)
        order = np.argsort(vals)
        return vecs[:, order[1]]
    except Exception:
        arr = L.toarray()
        vals, vecs = np.linalg.eigh(arr)
        idx = np.argsort(vals)
        return vecs[:, idx[1]]

def pick_k_edges_fiedler(G: nx.Graph, k: int, Lcap: int = 200, prev_f=None):
    """
    Add up to k non-edges that maximize (f_i - f_j)^2 using extremes of Fiedler values.
    Computes the Fiedler vector once per step for speed.
    """
    if k <= 0:
        return []
    nodes = list(G.nodes())
    idx = {u: i for i, u in enumerate(nodes)}
    f = fiedler_vector(G, v0=prev_f)
    # extremes
    order = sorted(nodes, key=lambda u: f[idx[u]])
    Lcap = min(Lcap, len(order) // 2) if len(order) >= 2 else 0
    left = order[:Lcap] if Lcap else order[:1]
    right = order[-Lcap:] if Lcap else order[-1:]
    # candidate pairs (score high -> better)
    cand = []
    for u in left:
        iu = idx[u]
        for v in right:
            if u == v or G.has_edge(u, v):
                continue
            s = (f[iu] - f[idx[v]]) ** 2
            cand.append(((u, v), s))
    cand.sort(key=lambda x: -x[1])

    chosen = []
    for (u, v), _ in cand:
        if len(chosen) >= k:
            break
        if not G.has_edge(u, v):
            G.add_edge(u, v)
            chosen.append((u, v))
    return chosen, f  # return f so the caller can warm-start next step

def next_edge_random(G: nx.Graph):
    nodes = list(G.nodes())
    for _ in range(20000):
        u, v = random.sample(nodes, 2)
        if not G.has_edge(u, v):
            return u, v
    return None, None

def add_k_edges_random(G: nx.Graph, k: int):
    added = []
    for _ in range(k):
        u, v = next_edge_random(G)
        if u is None:
            break
        G.add_edge(u, v)
        added.append((u, v))
    return added

def add_k_edges_mrkc_heuristic(G: nx.Graph, k: int):
    """
    Heuristic: connect min-core nodes to max-core nodes (tie-broken by degree).
    Recomputes cores once per step; fast and good enough for a simple runner.
    """
    added = []
    core_num = nx.core_number(G)
    deg = dict(G.degree())

    vuln = [u for u, ku in core_num.items() if ku == min(core_num.values())]
    vuln.sort(key=lambda u: (deg[u], u))
    anchors = [v for v, kv in core_num.items() if kv == max(core_num.values())]
    anchors.sort(key=lambda v: (-deg[v], v))

    for u in vuln:
        for v in anchors:
            if len(added) >= k:
                break
            if u != v and not G.has_edge(u, v):
                G.add_edge(u, v)
                added.append((u, v))
        if len(added) >= k:
            break

    # Fallback if we ran out
    if len(added) < k:
        for u, v in combinations(G.nodes(), 2):
            if len(added) >= k:
                break
            if not G.has_edge(u, v):
                G.add_edge(u, v)
                added.append((u, v))
    return added


# ----------------------------
# Runner: no-removal trajectories
# ----------------------------
def choose_budget_by_size(filename: str, G: nx.Graph):
    ext = os.path.splitext(filename)[1].lower()
    n = G.number_of_nodes()
    if ext == '.json' or n >= 200:
        return 100, 10   # simpler defaults for speed
    else:
        return 10, 1

def run_strategy_no_removal(G_in: nx.Graph, strategy_name: str, k_per_step: int, steps: int):
    """
    Return a DataFrame with rows: step, num_edges_added, edges_added, aG, e0_mult, e1_mult, CIS.
    Includes step=0 baseline.
    """
    G = G_in.copy()
    rows = []

    # baseline
    m0 = compute_metrics(G)
    rows.append({
        "step": 0, "num_edges_added": 0, "edges_added": "",
        **m0
    })

    prev_f = None  # warm-start for Fiedler

    for step in range(1, steps + 1):
        if strategy_name == "fiedler_greedy":
            added, prev_f = pick_k_edges_fiedler(G, k_per_step, prev_f=prev_f)
        elif strategy_name == "random_add":
            added = add_k_edges_random(G, k_per_step)
        elif strategy_name == "mrkc_heuristic":
            added = add_k_edges_mrkc_heuristic(G, k_per_step)
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")

        if not added:  # no more non-edges
            break

        m = compute_metrics(G)
        rows.append({
            "step": step,
            "num_edges_added": len(added),
            "edges_added": ";".join([f"{u}--{v}" for (u, v) in added]),
            **m
        })

    df = pd.DataFrame(rows)
    # Optional: add cumulative mean (AUC-over-steps, including baseline)
    for col in ["aG", "e0_mult", "e1_mult", "CIS"]:
        df[f"AUC_{col}"] = df[col].expanding().mean()
    return df


# ----------------------------
# Plotting
# ----------------------------
def plot_metric_across_strategies(dfs_by_strategy: dict, metric: str, outpath: str, title: str):
    """
    One figure for a given metric; one line per strategy.
    """
    plt.figure(figsize=(8, 5))
    for strat, df in dfs_by_strategy.items():
        if metric not in df.columns:
            continue
        plt.plot(df["step"], df[metric], marker="o", linestyle="-", label=strat)
    plt.xlabel("Reinforcement step")
    plt.ylabel(metric)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


# ----------------------------
# Orchestration
# ----------------------------
def run_for_graph(filename: str,
                  base_graph_dir_tgf='Graph_files/TGF_Files',
                  base_graph_dir_json='Graph_files',
                  out_root='Reinforcements_NoRemoval',
                  strategies=("fiedler_greedy", "random_add", "mrkc_heuristic")):

    # Load
    filetype = filename.split('.')[-1].lower()
    path, subdir = ((os.path.join(base_graph_dir_tgf, filename), 'TGF_Files')
                    if filetype == 'tgf'
                    else (os.path.join(base_graph_dir_json, filename), 'JSON_Files'))
    G = plot.load_graph(path, filetype)
    G.name = os.path.splitext(os.path.basename(filename))[0]

    total, per_step = choose_budget_by_size(filename, G)
    steps = math.ceil(total / per_step)

    outdir = os.path.join(out_root, subdir, G.name)
    os.makedirs(outdir, exist_ok=True)

    # Run strategies
    dfs = {}
    for strat in strategies:
        df = run_strategy_no_removal(G, strat, k_per_step=per_step, steps=steps)
        dfs[strat] = df
        df.to_csv(os.path.join(outdir, f"{G.name}_no_removal_{strat}.csv"), index=False)

    # Plot raw metric trajectories (one plot per metric; one line per strategy)
    for metric, pretty in [("aG", "a(G)"),
                           ("e0_mult", "Multiplicity m0 (zero eigenvalue)"),
                           ("e1_mult", "Multiplicity m1 (one eigenvalue)"),
                           ("CIS", "Core Influence Strength")]:
        plot_metric_across_strategies(
            dfs, metric,
            os.path.join(outdir, f"{G.name}_{metric}_no_removal.jpg"),
            f"{G.name}: {pretty} (No-removal trajectories)"
        )

    # (Optional) also plot the cumulative mean (“AUC over steps”)
    for metric, pretty in [("AUC_aG", "AUC a(G)"),
                           ("AUC_e0_mult", "AUC m0"),
                           ("AUC_e1_mult", "AUC m1"),
                           ("AUC_CIS", "AUC CIS")]:
        plot_metric_across_strategies(
            dfs, metric,
            os.path.join(outdir, f"{G.name}_{metric}_no_removal.jpg"),
            f"{G.name}: {pretty} (No-removal trajectories)"
        )

    return dfs


def main():
    parser = argparse.ArgumentParser(description="Reinforce graphs without removals and plot metrics.")
    parser.add_argument("--inputs", nargs="+",
                        default=['bfn.tgf', 'cpt.tgf', 'dur.tgf', 'els.tgf',
                                 'jnb.tgf', 'pta.tgf', 'pzb.tgf', 'vdp.tgf',
                                 'isis-links.json'])
    parser.add_argument("--tgf_dir", default="Graph_files/TGF_Files")
    parser.add_argument("--json_dir", default="Graph_files")
    parser.add_argument("--out", default="Reinforcements_NoRemoval")
    parser.add_argument("--strategies", nargs="+",
                        default=["fiedler_greedy", "random_add", "mrkc_heuristic"])
    args = parser.parse_args()

    for fn in args.inputs:
        run_for_graph(fn, base_graph_dir_tgf=args.tgf_dir, base_graph_dir_json=args.json_dir,
                      out_root=args.out, strategies=tuple(args.strategies))


if __name__ == "__main__":
    main()
