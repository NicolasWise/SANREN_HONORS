# plot_auc_by_reinforcement.py
#
# For each graph + reinforcement strategy, plot AUC over steps
# with one line per *removal* strategy. One figure per metric.

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

# Columns present in your *_AUC_summary.csv
METRICS = ["AUC_aG", "AUC_e0_mult", "AUC_e1_mult", "AUC_CIS"]

# Expected removal rows (kept in this order if present in CSVs)
REMOVAL_ROWS = ["random", "core influence", "degree", "betweeness", "closeness"]

ALL_LINES = REMOVAL_ROWS + ["no removal"]

# Map metric -> (baseline_col, after_col, cumAUC_col)
_NR_MAP = {
    "AUC_aG":      ("nr_aG_baseline",      "nr_aG_after",      "nr_cumAUC_aG"),
    "AUC_e0_mult": ("nr_e0_mult_baseline", "nr_e0_mult_after", "nr_cumAUC_e0_mult"),
    "AUC_e1_mult": ("nr_e1_mult_baseline", "nr_e1_mult_after", "nr_cumAUC_e1_mult"),
    "AUC_CIS":     ("nr_CIS_baseline",     "nr_CIS_after",     "nr_cumAUC_CIS"),
}

# Reinforcement folders created by reinforcements.py
REINF_STRATEGIES = [
    "fiedler_greedy",
    "random_add",
    "mrkc_heuristic", 
]

def _subdir_for_file(filename: str) -> str:
    return "TGF_Files" if filename.lower().endswith(".tgf") else "JSON_Files"

def _gname(filename: str) -> str:
    return os.path.splitext(os.path.basename(filename))[0]

def _pretty_metric(m: str) -> str:
    return {
        "AUC_aG": "AUC a(G)",
        "AUC_e0_mult": "AUC m0",
        "AUC_e1_mult": "AUC m1",
        "AUC_CIS": "AUC CIS",
    }.get(m, m)

def _pretty_reinf(r: str) -> str:
    return {
        "fiedler_greedy": "Fiedler Greedy",
        "random_add": "Random edge addition",
        "mrkc_heuristic": "MRKC heuristic",
    }.get(r, r)

def _no_removal_auc_series(filename: str, reinforcement_strategy: str, metric: str):
    """Return AUC-over-steps series for the 'no removal' line:
       [baseline, cumAUC_step1, cumAUC_step2, ...]"""
    subdir = _subdir_for_file(filename)
    gname = _gname(filename)
    summary_path = os.path.join("Reinforcements", subdir, gname, reinforcement_strategy, "_summary_steps.csv")
    if not os.path.isfile(summary_path):
        return None
    df = pd.read_csv(summary_path).sort_values("step")
    bcol, acol, ccol = _NR_MAP[metric]

    # baseline value
    base = float(df.iloc[0][bcol]) if bcol in df.columns else float("nan")

    # Preferred: use precomputed cumulative AUC if present
    if ccol in df.columns:
        return [base] + df[ccol].astype(float).tolist()

    # Fallback: compute expanding mean from after-values
    if acol in df.columns:
        after_vals = df[acol].astype(float).tolist()
        auc_vals = []
        running = [base]
        for v in after_vals:
            running.append(v)
            auc_vals.append(float(pd.Series(running).mean()))
        return [base] + auc_vals

    return None

def _baseline_auc_rows(filename: str) -> pd.DataFrame:
    """Return baseline AUC summary (one row per removal strategy)."""
    subdir = _subdir_for_file(filename)
    base_csv = os.path.join("Removals", subdir, f"{filename}_AUC_summary.csv")
    if not os.path.isfile(base_csv):
        raise FileNotFoundError(f"Baseline AUC file not found: {base_csv}")
    return pd.read_csv(base_csv)

def _step_auc_rows(filename: str, reinforcement_strategy: str) -> list[tuple[str, pd.DataFrame]]:
    """
    Return list of (step_tag, df) for each step_XY_AUC_summary.csv under the given strategy.
    step_tag is like 'step_01'.
    """
    subdir = _subdir_for_file(filename)
    gname = _gname(filename)
    step_dir = os.path.join("Reinforcements", subdir, gname, reinforcement_strategy)
    step_files = sorted(glob.glob(os.path.join(step_dir, "step_*_AUC_summary.csv")))
    out = []
    for path in step_files:
        try:
            tag = os.path.splitext(os.path.basename(path))[0].replace("_AUC_summary", "")
            out.append((tag, pd.read_csv(path)))
        except Exception as e:
            print(f"Warning: failed to read {path}: {e}")
    return out

def _collect_series(filename: str, reinforcement_strategy: str):
    base_df = _baseline_auc_rows(filename)
    steps_and_dfs = _step_auc_rows(filename, reinforcement_strategy)
    steps = ["baseline"] + [tag for tag, _ in steps_and_dfs]

    series = {label: {m: [float("nan")] * len(steps) for m in METRICS} for label in ALL_LINES}

    # removal baselines and steps
    for r in REMOVAL_ROWS:
        row = base_df[base_df["strategy"] == r]
        if not row.empty:
            row = row.iloc[0]
            for m in METRICS:
                if m in row.index:
                    series[r][m][0] = float(row[m])

    for idx, (tag, df) in enumerate(steps_and_dfs, start=1):
        for r in REMOVAL_ROWS:
            row = df[df["strategy"] == r]
            if not row.empty:
                row = row.iloc[0]
                for m in METRICS:
                    if m in row.index:
                        series[r][m][idx] = float(row[m])

    # no-removal AUC-over-steps
    for m in METRICS:
        nr = _no_removal_auc_series(filename, reinforcement_strategy, m)
        if nr is None:
            continue
        y = nr[:len(steps)] + [float("nan")] * max(0, len(steps) - len(nr))
        series["no removal"][m] = y

    return steps, series

def plot_auc_by_reinforcement(filename: str, reinforcement_strategy: str):
    subdir = _subdir_for_file(filename)
    gname = _gname(filename)
    outdir = os.path.join("Reinforcements", subdir, gname, "_plots_AUC_by_reinforcement", reinforcement_strategy)
    os.makedirs(outdir, exist_ok=True)

    steps, series = _collect_series(filename, reinforcement_strategy)

    for m in METRICS:
        plt.figure(figsize=(8, 5))
        x = range(len(steps))

        # removal AUC lines
        for r in REMOVAL_ROWS:
            ys = series[r][m]
            if not all(pd.isna(v) for v in ys):
                plt.plot(x, ys, marker="o", linestyle="-", label=r)

        # no-removal AUC-over-steps (dashed)
        nr = series["no removal"][m]
        if not all(pd.isna(v) for v in nr):
            plt.plot(x, nr, marker="o", linestyle="--", label="no removal")

        plt.xticks(x, steps, rotation=45, ha="right")
        plt.xlabel("Reinforcement step")
        plt.ylabel(f"{_pretty_metric(m)} AUC\n(removals: AUC over removals; no-removal: AUC over steps)")
        plt.title(f"{gname}: {_pretty_metric(m)} for {_pretty_reinf(reinforcement_strategy)}")
        plt.legend()
        plt.tight_layout()
        fout = os.path.join(outdir, f"{gname}_{m}_{reinforcement_strategy}.jpg")
        plt.savefig(fout, dpi=300)
        plt.close()
        print(f"Wrote {fout}")

def main():
    graphs = [
        "bfn.tgf", "cpt.tgf", "dur.tgf", "els.tgf",
        "jnb.tgf", "pta.tgf", "pzb.tgf", "vdp.tgf",
        "isis-links.json",
    ]
    for fn in graphs:
        for strat in REINF_STRATEGIES:
            try:
                plot_auc_by_reinforcement(fn, strat)
            except FileNotFoundError as e:
                # ok if a particular strategy hasn't been run for a graph yet
                print(e)

if __name__ == "__main__":
    main()
