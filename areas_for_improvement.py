
"""
areas_for_improvement.py

Utilities to:
1) Identify connecting nodes across multiple TGF subgraphs (nodes that appear in ≥ 2 files).
2) Compute per-graph connectivity scores (aggregate of core-influence + classical centralities).
3) Write the lowest-connectivity nodes per TGF to a folder (default: Improvements/).

Relies on project modules:
- core_resilience.py    (must expose: compute_core_resilience(G) -> (_, _, core_influence, _))
- classical_graph_measures.py (must expose: compute_classical_graph_measures(G) -> (degree, closeness, betweenness))
- plotter.py            (must expose: load_graph(path, filetype))

"""

import os
from collections import defaultdict
from typing import Dict, Set, List
import networkx as nx
import pandas as pd
import core_resilience as core
import classical_graph_measures as classic
import plotter as plot

def parse_tgf_nodes(filepath: str) -> Set[str]:
    """
    Return a set of node labels from a .tgf file.
    Handles labels with spaces (ID <space> label...).
    """
    with open(filepath, encoding="utf-8") as f:
        content = f.read()
    parts = content.split("\\n#\\n", 1)
    nodes_part = parts[0]
    labels: Set[str] = set()
    for line in nodes_part.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        pieces = line.split(maxsplit=1)
        if len(pieces) == 2:
            labels.add(pieces[1].strip())
        else:
            labels.add(pieces[0].strip())
    return labels


def load_tgf_node_sets(subgraphs: List[str], directory: str) -> Dict[str, Set[str]]:
    """
    Read each TGF once and return mapping: file -> set(node_labels)
    """
    out: Dict[str, Set[str]] = {}
    for name in subgraphs:
        path = os.path.join(directory, name)
        out[name] = parse_tgf_nodes(path)
    return out


def connecting_nodes_global(subgraphs: List[str], directory: str) -> Dict[str, List[str]]:
    """
    Build a global mapping of node_label -> list of TGF files it appears in (>= 2 only).
    """
    node_sets = load_tgf_node_sets(subgraphs, directory)
    appears_in = defaultdict(set)
    for g, nodes in node_sets.items():
        for n in nodes:
            appears_in[n].add(g)
    # Keep only nodes present in 2+ files
    multi = {n: sorted(list(gs)) for n, gs in appears_in.items() if len(gs) >= 2}
    return multi


# ----------------------------
# Connectivity scoring
# ----------------------------

def minmax(series: pd.Series) -> pd.Series:
    """Safe min-max normalization to [0,1]. If constant, returns zeros."""
    if series.empty:
        return series
    lo, hi = series.min(), series.max()
    if hi - lo == 0:
        return pd.Series(0.0, index=series.index)
    return (series - lo) / (hi - lo)


def compute_connectivity_scores(G: nx.Graph) -> pd.DataFrame:
    """
    Compute classical centralities and core influence for all nodes,
    then return a DataFrame with normalized columns and an aggregate score.

    Columns: degree, closeness, betweenness, core_influence
             deg_n, clo_n, bet_n, ci_n, agg_score
    """
    degree, closeness, betweenness = classic.compute_classical_graph_measures(G)
    _, _, core_influence, _ = core.compute_core_resilience(G)

    df = pd.DataFrame({
        "degree": pd.Series(degree, dtype=float),
        "closeness": pd.Series(closeness, dtype=float),
        "betweenness": pd.Series(betweenness, dtype=float),
        "core_influence": pd.Series(core_influence, dtype=float),
    })

    # Normalize each to [0,1]
    df["deg_n"] = minmax(df["degree"])
    df["clo_n"] = minmax(df["closeness"])
    df["bet_n"] = minmax(df["betweenness"])
    df["ci_n"]  = minmax(df["core_influence"])

    # Aggregate low-connectivity score (lower is worse)
    # We interpret "connectivity" as the mean of the normalized measures.
    df["agg_score"] = (df["deg_n"] + df["clo_n"] + df["bet_n"] + df["ci_n"]) / 4.0

    return df


def lowest_connectivity_nodes(df: pd.DataFrame, bottom_frac: float = 0.15, min_k: int = 3) -> pd.DataFrame:
    """
    Return the lowest-connectivity nodes by aggregate score.
    - bottom_frac: select bottom X% (default 15%)
    - min_k: at least this many nodes
    """
    k = max(min_k, int(len(df) * bottom_frac))
    return df.sort_values("agg_score", ascending=True).head(k)


def write_improvement_reports_for_tgfs(
    subgraphs: List[str],
    directory: str,
    outdir: str = "Improvements",
    bottom_frac: float = 0.15,
    min_k: int = 3,
) -> str:
    """
    For each TGF in `subgraphs` (found under `directory`), load the graph via plotter.load_graph,
    compute connectivity scores, identify lowest-connectivity nodes, and write CSV reports to `outdir`.

    Also writes a global file listing connecting nodes across TGFs.
    Returns the path to the output folder.
    """
    os.makedirs(outdir, exist_ok=True)

    # Identify global connecting nodes (optional: write to CSV)
    connectors = connecting_nodes_global(subgraphs, directory)
    # Write connectors
    conn_csv = os.path.join(outdir, "connecting_nodes_global.csv")
    pd.DataFrame([{"node": n, "appears_in": ";".join(gs), "count": len(gs)} for n, gs in sorted(connectors.items())]).to_csv(conn_csv, index=False)

    for tgf in subgraphs:
        path = os.path.join(directory, tgf)
        G = plot.load_graph(path, "tgf")
        scores = compute_connectivity_scores(G)
        low    = lowest_connectivity_nodes(scores, bottom_frac=bottom_frac, min_k=min_k)

        # Flag whether node is a connector
        low = low.copy()
        low["is_connecting_node"] = low.index.to_series().apply(lambda n: n in connectors)

        out_csv = os.path.join(outdir, f"{os.path.splitext(tgf)[0]}_lowest_connectivity.csv")
        low.round(6).reset_index(names="node").to_csv(out_csv, index=False)

    return outdir
