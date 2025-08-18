from pathlib import Path
import pandas as pd
import re

# ---------------------------
# CONFIG
# ---------------------------
FILES = [
    'bfn.tgf', 'cpt.tgf', 'dur.tgf', 'els.tgf',
    'jnb.tgf', 'pta.tgf', 'pzb.tgf', 'vdp.tgf',
    'isis-links.json'
]

# Strategies (directory names in Reinforcements) you want tables for
STRATEGY_DIRS = {
    'fiedler_greedy'      : 'Fiedler-greedy (edge-add)',
    'mrkc_heuristic'      : 'MRKC heuristic',
    'areas_low_nonconnect': 'Areas–low-nonconnect',
    'random_add'          : 'Random edge-add'
}
# Removal strategies order (rows grouped in tables)
REMOVAL_ORDER = ['betweeness', 'closeness', 'core influence', 'degree', 'random']

# Where to search / write
REMOVALS_BASE = Path('Removals')
REINF_BASE    = Path('Reinforcements')
OUT_DIR       = Path('Automation/latex_tables')   # tables will be written here
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Column names we expect in AUC CSVs
AUC_COLS = ['AUC_aG', 'AUC_e0_mult', 'AUC_e1_mult', 'AUC_CIS']
COL_LABEL = {
    'AUC_aG'     : r'AUC $a(G)$',
    'AUC_e0_mult': r'AUC $m_0$',
    'AUC_e1_mult': r'AUC $m_1$',
    'AUC_CIS'    : r'AUC CIS',
}

# ---------------------------
# Helpers
# ---------------------------
def stem_of(filename: str) -> str:
    """'bfn.tgf' -> 'bfn' ; 'isis-links.json' -> 'isis-links' """
    return Path(filename).stem

def find_baseline_auc(filename: str) -> pd.DataFrame:
    """
    Look for Removals/TGF_Files/<file>_AUC_summary.csv
         or Removals/JSON_Files/<file>_AUC_summary.csv
         or a loose <file>_AUC_summary.csv
    Return DataFrame (may be empty).
    """
    fname = f"{filename}_AUC_summary.csv"
    candidates = [
        REMOVALS_BASE/'TGF_Files'/fname,
        REMOVALS_BASE/'JSON_Files'/fname,
        Path(fname)
    ]
    for p in candidates:
        if p.is_file():
            df = pd.read_csv(p)
            return df
    return pd.DataFrame()

def norm_strategy_name(s: str) -> str:
    """Normalise strategy strings so we can match reliably."""
    if not isinstance(s, str):
        return ''
    return re.sub(r'\s+', ' ', s.strip().lower())

def tidy_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'strategy_norm' for matching/ordering."""
    df = df.copy()
    df['strategy_norm'] = df['strategy'].apply(norm_strategy_name)
    return df

def find_step_files(base_dir: Path) -> list[Path]:
    """Return sorted list of step_XX_AUC_summary.csv files by XX numeric order."""
    if not base_dir.is_dir():
        return []
    files = list(base_dir.glob('step_*_AUC_summary.csv'))
    def key(p: Path):
        m = re.search(r'step_(\d+)', p.name)
        return int(m.group(1)) if m else 0
    return sorted(files, key=key)

def load_steps(graph_stem: str, subdir: str, strat_dir: str) -> tuple[list[int], dict[int, pd.DataFrame]]:
    """
    subdir: 'TGF_Files' or 'JSON_Files'
    Returns (steps_list, {step -> df})
    """
    based = REINF_BASE / subdir / graph_stem / strat_dir
    step_paths = find_step_files(based)
    steps, dfs = [], {}
    for p in step_paths:
        m = re.search(r'step_(\d+)', p.name)
        if not m: 
            continue
        step = int(m.group(1))
        df = pd.read_csv(p)
        dfs[step] = tidy_strategy(df)
        steps.append(step)
    return sorted(steps), dfs

def format_float(x) -> str:
    try:
        return f"{float(x):.6f}"
    except Exception:
        return ""

def make_table_latex(graph_file: str, strategy_dir: str, strategy_title: str) -> str:
    """
    Build one LaTeX table for graph_file × strategy_dir.
    """
    graph_stem = stem_of(graph_file)
    subdir = 'JSON_Files' if graph_file.endswith('.json') else 'TGF_Files'

    # load baseline and normalise names
    base = find_baseline_auc(graph_file)
    if base.empty:
        return f"% WARNING: Baseline AUC not found for {graph_file}\n"

    base = tidy_strategy(base)

    # load steps for this strategy
    step_ids, step_dfs = load_steps(graph_stem, subdir, strategy_dir)
    if not step_ids:
        return f"% NOTE: No steps found for {graph_file} / {strategy_dir}\n"

    # header cols
    step_headers = [f"Step {i}" for i in step_ids]

    # build rows
    lines = []
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'  \centering')
    lines.append(f'  \\caption{{AUC per removal strategy at baseline and after each reinforcement step for the {strategy_title} approach on \\texttt{{{graph_file}}} (no deltas).}}')
    lines.append(f'  \\label{{tab:{graph_stem}-{strategy_dir}-auc}}')
    lines.append('  \\begin{tabular}{ll' + 'r'* (1 + len(step_headers)) + '}')
    lines.append(r'    \toprule')
    header = ' & '.join(['\\textbf{Strategy}', '\\textbf{Metric}', '\\textbf{Baseline}'] + [f'\\textbf{{{h}}}' for h in step_headers])
    lines.append(f'    {header} \\\\')
    lines.append(r'    \midrule')

    for removal in REMOVAL_ORDER:
        # slice baseline row for this removal strategy
        base_row = base[base['strategy_norm'] == removal]
        # multirow only for first metric block
        first = True
        for auc_col in AUC_COLS:
            metric_label = COL_LABEL[auc_col]
            bval = format_float(base_row.iloc[0][auc_col]) if not base_row.empty and auc_col in base_row.columns else ""

            # step values
            svals = []
            for st in step_ids:
                df = step_dfs[st]
                r = df[df['strategy_norm'] == removal]
                sval = format_float(r.iloc[0][auc_col]) if not r.empty and auc_col in r.columns else ""
                svals.append(sval)

            if first:
                lines.append(f"    \\multirow{{4}}{{*}}{{{removal}}} & {metric_label} & {bval} & " + " & ".join(svals) + r" \\")
                first = False
            else:
                lines.append(f"    & {metric_label} & {bval} & " + " & ".join(svals) + r" \\")
        # after each block, a small midrule for readability (optional)
        lines.append(r'    \addlinespace')

    lines.append(r'    \bottomrule')
    lines.append(r'  \end{tabular}')
    lines.append(r'\end{table}')
    lines.append('')  # blank line
    return '\n'.join(lines)

# ---------------------------
# Main: generate all tables
# ---------------------------
def main():
    out_path = OUT_DIR / 'auc_tables_all_strategies.tex'
    with out_path.open('w', encoding='utf-8') as f:
        for graph_file in FILES:
            for strat_dir, strat_title in STRATEGY_DIRS.items():
                latex = make_table_latex(graph_file, strat_dir, strat_title)
                f.write(latex)
                # also echo a small comment separator
                f.write(f"% --- end table: {graph_file} / {strat_dir} ---\n\n")
    print(f"Wrote LaTeX tables to: {out_path}")

if __name__ == '__main__':
    main()
