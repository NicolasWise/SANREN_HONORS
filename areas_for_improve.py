"""
areas_for_improvement.py

Utilities to:
1) Identify connecting nodes across multiple TGF subgraphs (nodes that appear in ≥ 2 files).
2) Compute per-graph connectivity scores (aggregate of core-influence + classical centralities).
3) Write the lowest-connectivity nodes per TGF to a folder (default: Improvements/).

Relies on project modules:
- core_resilience.py
- classical_graph_measures.py
- plotter.py
"""

import os
from typing import Dict, Set, List, Tuple
from collections import defaultdict
import networkx as nx
import pandas as pd

import core_resilience as core
import classical_graph_measures as classic
import plotter as plot


# ----------------------------
# TGF parsing helpers
# ----------------------------

def parse_tgf_id_label_map(filepath: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Parse a .tgf file and return:
      - id2label: mapping from numeric/string ID -> human-readable label
      - label2id: inverse mapping (label -> ID in THIS file)
    Robust to CRLF and stray whitespace.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\r\n") for ln in f]

    # Find '#' separator
    try:
        sep_idx = lines.index('#')
    except ValueError:
        sep_idx = next((i for i, ln in enumerate(lines) if ln.strip() == '#'), len(lines))

    node_lines = lines[:sep_idx]

    id2label: Dict[str, str] = {}
    label2id: Dict[str, str] = {}

    for line in node_lines:
        line = line.strip()
        if not line or line.startswith('%'):
            continue
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            nid, label = parts[0], parts[1].strip()
        else:
            nid, label = parts[0], parts[0]
        id2label[nid] = label
        # if duplicate labels appear (rare), last one wins
        label2id[label] = nid

    return id2label, label2id


def parse_tgf_nodes(filepath: str) -> Set[str]:
    """
    Return the set of node labels from a .tgf file (wrapper using the map above).
    """
    _, label2id = parse_tgf_id_label_map(filepath)
    return set(label2id.keys())


def load_tgf_label2id_maps(subgraphs: List[str], directory: str) -> Dict[str, Dict[str, str]]:
    """
    Read each TGF once and return mapping: file -> {label -> id_in_that_file}
    """
    out: Dict[str, Dict[str, str]] = {}
    for name in subgraphs:
        path = os.path.join(directory, name)
        _, label2id = parse_tgf_id_label_map(path)
        out[name] = label2id
    return out


# ----------------------------
# Connecting nodes (appear in >= 2 files)
# ----------------------------

def connecting_nodes_global(subgraphs: List[str], directory: str):
    """
    Build global info for nodes (by LABEL) that appear in >= 2 TGFs.

    Returns:
      connecting_labels: set of labels with multiplicity >= 2
      rows: list of dict rows for a CSV with per-label details
            (node_label, count, appears_in_files, file_ids)
    """
    label2id_maps = load_tgf_label2id_maps(subgraphs, directory)

    # Accumulate where each LABEL appears and its ID in that file
    where_files: Dict[str, List[str]] = defaultdict(list)
    where_fileids: Dict[str, List[str]] = defaultdict(list)

    for fname, l2i in label2id_maps.items():
        for label, nid in l2i.items():
            where_files[label].append(fname)
            where_fileids[label].append(f"{fname}:{nid}")

    rows = []
    connecting_labels = set()
    for label, files in where_files.items():
        if len(files) >= 2:
            connecting_labels.add(label)
            rows.append({
                "node_label": label,
                "count": len(files),
                "appears_in_files": ";".join(sorted(files)),
                "file_ids": ";".join(sorted(where_fileids[label]))
            })

    # Sort rows for stable output
    rows.sort(key=lambda r: (-r["count"], r["node_label"]))
    return connecting_labels, rows


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

    Index: node LABELS
    Columns: degree, closeness, betweeness, core_influence,
             deg_n, clo_n, bet_n, ci_n, agg_score
    """
    degree, closeness, betweeness = classic.compute_classical_graph_measures(G)
    _, _, core_influence, _ = core.compute_core_resilience(G)

    df = pd.DataFrame({
        "degree": pd.Series(degree, dtype=float),
        "closeness": pd.Series(closeness, dtype=float),
        "betweeness": pd.Series(betweeness, dtype=float),
        "core_influence": pd.Series(core_influence, dtype=float),
    })

    # Normalize each to [0,1]
    df["deg_n"] = minmax(df["degree"])
    df["clo_n"] = minmax(df["closeness"])
    df["bet_n"] = minmax(df["betweeness"])
    df["ci_n"]  = minmax(df["core_influence"])

    # Aggregate connectivity score (higher is better)
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


# ----------------------------
# Main writer
# ----------------------------

def write_improvement_reports_for_tgfs(
    subgraphs: List[str],
    directory: str,
    outdir: str = "Improvements",
    bottom_frac: float = 0.50,
    min_k: int = 3,
) -> str:
    """
    For each TGF in `subgraphs` (under `directory`), load graph via plotter.load_graph,
    compute connectivity scores, select lowest-connectivity nodes, and write CSVs to `outdir`.

    Also writes a global 'connecting_nodes_global.csv' with both LABELS and per-file IDs.
    """
    os.makedirs(outdir, exist_ok=True)

    # Global connecting nodes (collect labels + file-local IDs)
    connecting_labels, conn_rows = connecting_nodes_global(subgraphs, directory)
    conn_csv = os.path.join(outdir, "connecting_nodes_global.csv")
    pd.DataFrame(conn_rows, columns=["node_label", "count", "appears_in_files", "file_ids"]).to_csv(conn_csv, index=False)

    # Process each TGF
    for tgf in subgraphs:
        path = os.path.join(directory, tgf)
        # Graph is assumed to be labeled by LABELS
        G = plot.load_graph(path, "tgf")
        # label2id for THIS file to add 'node_id' column
        _, label2id = parse_tgf_id_label_map(path)

        scores = compute_connectivity_scores(G)
        low = lowest_connectivity_nodes(scores, bottom_frac=bottom_frac, min_k=min_k)

        # Attach ID in this file and connector flag
        low = low.copy()
        low["node_id"] = low.index.to_series().map(label2id).fillna("")
        low["is_connecting_node"] = low.index.to_series().apply(lambda lbl: lbl in connecting_labels)

        out_csv = os.path.join(outdir, f"{os.path.splitext(tgf)[0]}_lowest_connectivity.csv")
        cols = ["node_id"] + ["degree", "closeness", "betweeness", "core_influence",
                              "deg_n", "clo_n", "bet_n", "ci_n", "agg_score",
                              "is_connecting_node"]
        low.round(6).reset_index(names="node_label")[["node_label"] + cols].to_csv(out_csv, index=False)

    return outdir


def main():
    inputs = ['bfn.tgf', 'cpt.tgf', 'dur.tgf', 'els.tgf', 'jnb.tgf', 'pta.tgf', 'pzb.tgf', 'vdp.tgf']
    write_improvement_reports_for_tgfs(subgraphs=inputs, directory='Graph_Files/TGF_Files')


if __name__ == "__main__":
    main()
