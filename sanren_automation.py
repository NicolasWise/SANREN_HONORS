
"""
sanren_automation.py

A consolidated runner that:
1) Accepts a filename (.tgf or .json).
2) Loads a graph using project's plotter.load_graph.
3) Computes spectral & core resilience metrics and writes summaries to Automation/.
4) Shows a short textual summary to the user.
5) Prompts optionally to run node-removal simulations (like node_removals module) and saves outputs to Automation/.
6) Prompts optionally to identify low-connectivity nodes within this graph and saves outputs to Automation/.

Relies on project modules:
- plotter.py                (load_graph)
- spectral_analysis.py      (compute_spectral_analysis, optional plotting/writing helpers)
- core_resilience.py        (compute_core_resilience)
- classical_graph_measures.py
- node_removals.py          (optional; if present, strategies + simulate_strategy)
- areas_for_improvement.py  (compute_connectivity_scores, lowest_connectivity_nodes)

Author: ChatGPT (automation helper)
"""

from __future__ import annotations
import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt

import networkx as nx

# Project imports
import plotter as plot
import spectral_analysis as spec
import core_resilience as core
import classical_graph_measures as classic

# Optional imports
try:
    import node_removals as nr
    HAS_NODE_REMOVALS = True
except Exception:
    HAS_NODE_REMOVALS = False

try:
    import areas_for_improvement as afi
    HAS_AFI = True
except Exception:
    HAS_AFI = False


AUTODIR = "Automation"
os.makedirs(AUTODIR, exist_ok=True)


def summarize_spectral(graph_name: str, spectral_tuple) -> pd.DataFrame:
    """
    spectral_tuple expected: (eigenvalues, e1_cluster, a_G, e1_mult, e0_mult)
    """
    eigenvalues, e1_cluster, a_G, e1_mult, e0_mult = spectral_tuple
    df = pd.DataFrame({
        "graph": [graph_name],
        "a_G":   [a_G],
        "e1_mult": [e1_mult],
        "e0_mult": [e0_mult],
        "num_eigs": [len(eigenvalues) if hasattr(eigenvalues, "__len__") else None]
    })
    return df


def write_spectral_outputs(graph, spectral_tuple, outdir: str):
    """Write spectral summary CSV; also call project plotting helpers if available."""
    os.makedirs(outdir, exist_ok=True)
    summary_csv = os.path.join(outdir, f"{graph.name}_spectral_summary.csv")
    summarize_spectral(graph.name, spectral_tuple).to_csv(summary_csv, index=False)

    # Try project helper(s) if present
    try:
        eigenvalues, e1_cluster, a_G, e1_mult, e0_mult = spectral_tuple
        # If user module exposes these, they will create plots/files in their usual locations
        spec.write_spectral_to_output_file(graph, a_G, e1_cluster, e1_mult, e0_mult, tgf=(graph.name.endswith(".tgf")), json=(graph.name.endswith(".json")))
        spec.plot_spectral_graphs(eigenvalues, graph_name=graph.name, tgf=(graph.name.endswith(".tgf")), json=(graph.name.endswith(".json")))
    except Exception:
        pass
    return summary_csv


def write_core_outputs(graph, core_tuple, outdir: str):
    """Write core-resilience CSV with node-level core_number, core_strength, core_influence, CIS."""
    os.makedirs(outdir, exist_ok=True)
    core_number, core_strength, core_influence, CIS = core_tuple
    df = pd.DataFrame({
        "node": list(core_number.keys())
    }).set_index("node")
    df["core_number"]   = pd.Series(core_number, dtype=float)
    df["core_strength"] = pd.Series(core_strength, dtype=float)
    df["core_influence"]= pd.Series(core_influence, dtype=float)
    df["CIS"]           = pd.Series(CIS, dtype=float)
    out_csv = os.path.join(outdir, f"{graph.name}_core_resilience.csv")
    df.sort_values(["core_number","core_strength","core_influence"], ascending=[False, False, False]).to_csv(out_csv)
    return out_csv


def run_node_removals(graph, outdir: str, r: int):
    """
    Run node removals either via node_removals module (if available),
    or a minimal internal fallback (random + degree only).
    Writes combined CSV and comparison plots to outdir.
    """
    os.makedirs(outdir, exist_ok=True)
    all_dfs = []

    if HAS_NODE_REMOVALS:
        strategies = getattr(nr, "strategies", [])
        simulate = getattr(nr, "simulate_strategy", None)
        if not strategies or simulate is None:
            print("[automation] node_removals module found but missing attributes; using fallback.")
        else:
            for fn, name in strategies:
                df = simulate(graph, fn, name)
                df.to_csv(os.path.join(outdir, f"{graph.name}_{name}.csv"), index=False)
                all_dfs.append(df)

    if not all_dfs:
        # Minimal fallback: random & degree
        import random
        from copy import deepcopy
        def order_random(G):
            nodes = list(G.nodes())
            random.shuffle(nodes)
            return nodes
        def order_degree(G):
            deg = dict(G.degree())
            return [n for n,_ in sorted(deg.items(), key=lambda x: -x[1])]

        def simulate(G, order_fn, label):
            H = deepcopy(G)
            rows = []
            order = order_fn(H)
            for k, node in enumerate(order, start=1):
                if H.number_of_nodes() < 2:
                    break
                eigs, cluster, aG, e1_mult, e0_mult = spec.compute_spectral_analysis(H)
                rows.append({"removed": k, "remaining": H.number_of_nodes(), "aG": aG, "e1_mult": e1_mult, "e0_mult": e0_mult, "strategy": label})
                H.remove_node(node)
            return pd.DataFrame(rows)

        for fn, name in [(order_random, "random"), (order_degree, "degree")]:
            df = simulate(graph, fn, name)
            df.to_csv(os.path.join(outdir, f"{graph.name}_{name}.csv"), index=False)
            all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)
    combined_csv = os.path.join(outdir, f"{graph.name}_removals_combined.csv")
    combined.to_csv(combined_csv, index=False)

    

    # Simple comparison plots
    for metric in ["aG", "e1_mult", "e0_mult"]:
        nr.plot_metric_small_multiples(
            combined,
            metric,
            os.path.join(outdir, f'{graph.name}_compare_{metric}.png')
        )

    return combined_csv


def run_low_connectivity_analysis(graph, outdir: str, bottom_frac: float=0.15, min_k: int=3):
    """Compute per-node aggregate connectivity (degree, closeness, betweenness, core-influence) and write CSV."""
    os.makedirs(outdir, exist_ok=True)
    if HAS_AFI:
        scores = afi.compute_connectivity_scores(graph)
        low    = afi.lowest_connectivity_nodes(scores, bottom_frac=bottom_frac, min_k=min_k)
    else:
        # Minimal reimplementation
        degree, closeness, betweenness = classic.compute_classical_graph_measures(graph)
        _, _, core_influence, _ = core.compute_core_resilience(graph)
        df = pd.DataFrame({
            "degree": pd.Series(degree, dtype=float),
            "closeness": pd.Series(closeness, dtype=float),
            "betweenness": pd.Series(betweenness, dtype=float),
            "core_influence": pd.Series(core_influence, dtype=float),
        })
        def minmax(s):
            if s.empty: return s
            lo, hi = s.min(), s.max()
            if hi-lo == 0: return pd.Series(0.0, index=s.index)
            return (s-lo)/(hi-lo)
        df["agg_score"] = (minmax(df["degree"]) + minmax(df["closeness"]) + minmax(df["betweenness"]) + minmax(df["core_influence"])) / 4.0
        scores = df
        k = max(min_k, int(len(df)*bottom_frac))
        low = df.sort_values("agg_score", ascending=True).head(k)

    scores_csv = os.path.join(outdir, f"{graph.name}_connectivity_scores.csv")
    scores.round(6).reset_index(names="node").to_csv(scores_csv, index=False)

    low_csv = os.path.join(outdir, f"{graph.name}_lowest_connectivity.csv")
    low.round(6).reset_index(names="node").to_csv(low_csv, index=False)

    return scores_csv, low_csv


def main():
    parser = argparse.ArgumentParser(description="SANReN automation: spectral/core summaries, node removals, and low-connectivity nodes.")
    parser.add_argument("filename", help="Path to a .tgf or .json graph file.")
    parser.add_argument("--r", type=int, default=45, help="Percent for 'top-r' style thresholds (if used by node-removals code).")
    args = parser.parse_args()
    #args = 'Graph_Files/TGF_files/bfn.tgf'

    filetype = os.path.splitext(args.filename)[1].lstrip(".").lower()
    if filetype not in ("tgf", "json"):
        print(f"[automation] Unsupported file type: {filetype}")
        sys.exit(1)

    # Load graph using project plotter
    G = plot.load_graph(args.filename, filetype)
    # Set a name useful for outputs
    if not G.name:
        G.name = os.path.basename(args.filename)

    outdir = os.path.join(AUTODIR, os.path.splitext(os.path.basename(args.filename))[0])
    os.makedirs(outdir, exist_ok=True)

    # 1) Spectral and Core Resilience
    spectral = spec.compute_spectral_analysis(G)
    core_res = core.compute_core_resilience(G)

    spectral_csv = write_spectral_outputs(G, spectral, outdir)
    core_csv     = write_core_outputs(G, core_res, outdir)

    print("[automation] Wrote:")
    print(" ", spectral_csv)
    print(" ", core_csv)

    # 2) Ask to run node removals
    try:
        ans = input("Compute node removals and comparison plots? [y/N]: ").strip().lower()
    except EOFError:
        ans = "n"
    if ans == "y":
        rem_csv = run_node_removals(G, outdir, r=args)
        print(" ", rem_csv)

    # 3) Ask to identify low-connectivity nodes
    try:
        ans2 = input("Identify lowest-connectivity nodes in this graph? [y/N]: ").strip().lower()
    except EOFError:
        ans2 = "n"
    if ans2 == "y":
        scores_csv, low_csv = run_low_connectivity_analysis(G, outdir)
        print(" ", scores_csv)
        print(" ", low_csv)

    print(f"[automation] Done. Outputs in {outdir}")

if __name__ == "__main__":
    main()
