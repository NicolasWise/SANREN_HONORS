import math
import random
import os
import core_resilience as core
import classical_graph_measures as classic
import spectral_analysis as spec
import plotter as plot
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


def removal_random(graph):
    nodes = list(graph.nodes())
    random.shuffle(nodes)
    return nodes

def removal_top_k_core_number(graph):
    core_number, _, _, _ = core.compute_core_resilience(graph)
    return {node: value for node, value in sorted(core_number.items(), key=lambda x:-x[1])}

def removal_top_k_core_influence(graph):
    _, _, core_influence, _ = core.compute_core_resilience(graph)
    return {node:value for node, value in sorted(core_influence.items(), key = lambda x:-x[1])}
#dont use
def removal_top_k_core_strengths(graph):
    _, core_strength, _, _ = core.compute_core_resilience(graph)
    return {node: value for node, value in sorted(core_strength.items(), key = lambda x:-x[1])}

def removal_top_k_degree(graph):
    deg = dict(graph.degree())
    return {node: value for node, value in sorted(deg.items(), key=lambda x:-x[1])}

def removal_top_k_closenss(graph):
    _, closeness, _ = classic.compute_classical_graph_measures(graph)
    return {n: v for n, v in sorted(closeness.items(), key=lambda x: -x[1])}

def removal_top_k_betweeness(graph):
    _, _, betweenness = classic.compute_classical_graph_measures(graph)
    return {n: v for n, v in sorted(betweenness.items(), key=lambda x: -x[1])}

strategies = [
    (removal_random, 'random'),
    (removal_top_k_core_influence, 'core influence'),
    (removal_top_k_degree, 'degree'),
    (removal_top_k_betweeness, 'betweeness'),
    (removal_top_k_closenss, 'closeness'),
]

def _auc_series(s: pd.Series) -> float:
    """Average over steps t=1..T (discrete AUC normalized by T)."""
    # ensure natural step order
    return float(s.mean()) if len(s) else float('nan')

### NEW: cumulative AUC columns per strategy
def add_cumulative_auc(df: pd.DataFrame, metrics=('aG', 'e0_mult', 'e1_mult', 'CIS')) -> pd.DataFrame:
    """
    For each strategy, compute cumulative AUC up to step t for each metric.
    Adds columns: cumAUC_<metric>.
    """
    df = df.sort_values(['strategy', 'removed']).copy()
    for m in metrics:
        df[f'cumAUC_{m}'] = (
            df.groupby('strategy')[m]
              .apply(lambda s: s.expanding().mean())
              .reset_index(level=0, drop=True)
        )
    return df

### NEW: final AUC summary table (one row per strategy)
def summarize_auc(df: pd.DataFrame, graph_name: str, metrics=('aG', 'e0_mult', 'e1_mult', 'CIS')) -> pd.DataFrame:
    rows = []
    for strat, sub in df.groupby('strategy'):
        row = {
            'graph': graph_name,
            'strategy': strat,
            'steps': int(sub['removed'].max())
        }
        for m in metrics:
            row[f'AUC_{m}'] = _auc_series(sub[m])
        rows.append(row)
    return pd.DataFrame(rows)

### NEW: add deltas vs random baseline (convenient for LaTeX tables)
def add_vs_random(summary_df: pd.DataFrame, metrics=('aG', 'e0_mult', 'e1_mult', 'CIS')) -> pd.DataFrame:
    out = summary_df.copy()
    if 'random' not in set(out['strategy']):
        return out  # nothing to compare
    base = out[out['strategy'] == 'random'].iloc[0]
    for m in metrics:
        out[f'Delta_{m}_vs_random'] = out[f'AUC_{m}'] - base[f'AUC_{m}']
    return out

def simulate_strategy(graph, strategy_fn, strategy_name):
    '''A list of nodes in order in which the nodes will be removed'''
    order = list(strategy_fn(graph))
    G = graph.copy()
    records = []

    for k, node in enumerate(order, start=1):
        if G.number_of_nodes() < 2:
            print(f"Stopping simulation at step {k}: only {G.number_of_nodes()} nodes left")
            break
        # Compute spectral metrics of the copy graph
        eigs, cluster, aG, e1_mult, e0_mult = spec.compute_spectral_analysis(G)

        core_num, core_str, core_inf, CIS = core.compute_core_resilience(G)

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

def plot_metric(df, metric, outpath):
    plt.figure(figsize=(8,5))
    for strat in df['strategy'].unique():
        sub = df[df['strategy']==strat]
        plt.plot(sub['removed'], sub[metric], label=strat)
    plt.xlabel('Nodes removed')
    plt.ylabel(metric)
    plt.title(f'{metric} vs Nodes Removed')
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()

def plot_metric_small_multiples(df, metric, outpath, max_cols=3):
    strategies = df['strategy'].unique()
    n = len(strategies)
    cols = min(n, max_cols)
    rows = math.ceil(n / cols)
    
    fig, axes = plt.subplots(
        rows, cols,
        figsize=(4*cols, 3*rows),
        sharex=True, sharey=True
    )
    
    # axes might be 1D or 2D
    axes = axes.flatten() if n > 1 else [axes]
    
    for ax, strat in zip(axes, strategies):
        sub = df[df['strategy'] == strat]
        ax.plot(sub['removed'], sub[metric], marker='o', linestyle='-')
        ax.set_title(strat)
        ax.grid(True)

        # only label outer edges
        ax.set_xlabel('Removed')  
        ax.set_ylabel(metric)
    
    # Remove any unused subplots
    for ax in axes[n:]:
        fig.delaxes(ax)
    
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outpath, dpi=300)
    plt.close(fig)

# ===== NEW: consistent line/marker styles per removal strategy =====
STYLE_MAP = {
    'random':        dict(linestyle='-',  marker='o'),
    'core influence':dict(linestyle='--', marker='s'),
    'degree':        dict(linestyle='-.', marker='^'),
    'betweeness':    dict(linestyle=':',  marker='D'),   # note: original spelling retained
    'closeness':     dict(linestyle=(0,(3,1,1,1)), marker='v'),
}

def plot_metric_onepanel_styled(df: pd.DataFrame, metric: str, outpath: str, title: str):
    """
    NEW: One figure per metric; one line per removal strategy with distinct styles/markers.
    Does not replace existing plotters.
    """
    plt.figure(figsize=(8, 5))
    for strat in df['strategy'].unique():
        sub = df[df['strategy'] == strat]
        style = STYLE_MAP.get(strat, dict(linestyle='-', marker='o'))
        plt.plot(
            sub['removed'], sub[metric],
            label=strat,
            **style
        )
    plt.xlabel('Nodes removed')
    plt.ylabel(metric)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def emit_latex_auc_table(graph_name: str,
                         auc_summary_df: pd.DataFrame,
                         out_tex_path: str,
                         caption: str = None,
                         label: str = None,
                         round_digits: int = 4):
    """
    NEW: Emit a LaTeX table of AUC values per removal strategy for this graph.
    Columns: strategy, AUC_aG, AUC_e0_mult, AUC_e1_mult, AUC_CIS (+ steps if present).
    """
    cols_order = ['strategy', 'steps', 'AUC_aG', 'AUC_e0_mult', 'AUC_e1_mult', 'AUC_CIS']
    cols = [c for c in cols_order if c in auc_summary_df.columns]
    tbl = auc_summary_df[cols].copy()

    # Round numeric AUC columns for clean LaTeX
    for c in tbl.columns:
        if c.startswith('AUC_') or c == 'steps':
            tbl[c] = pd.to_numeric(tbl[c], errors='coerce').round(round_digits)

    # Build LaTeX
    cap  = caption or f"AUC summary per removal strategy for {graph_name}"
    lab  = label   or f"tab:auc_{graph_name}"

    # Column spec based on present columns
    # strategy is left, all others right aligned
    align_spec = '@{}l' + 'r' * (len(cols) - 1) + '@{}'

    lines = []
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'  \centering')
    lines.append(rf'  \caption{{{cap}}}')
    lines.append(rf'  \label{{{lab}}}')
    lines.append(r'  \begin{{tabular}}{{{align_spec}}}')
    lines.append(r'    \toprule')

    # Header row
    header_pretty = {
        'strategy': 'Removal strategy',
        'steps': 'Steps',
        'AUC_aG': r'AUC $a(G)$',
        'AUC_e0_mult': r'AUC $m_0$',
        'AUC_e1_mult': r'AUC $m_1$',
        'AUC_CIS': r'AUC CIS',
    }
    header_row = ' & '.join(header_pretty.get(c, c) for c in cols) + r' \\'
    lines.append('    ' + header_row)
    lines.append(r'    \midrule')

    # Body rows
    for _, r in tbl.iterrows():
        vals = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                vals.append(f'{v:.{round_digits}f}')
            else:
                vals.append(str(v))
        lines.append('    ' + ' & '.join(vals) + r' \\')

    lines.append(r'    \bottomrule')
    lines.append(r'  \end{tabular}')
    lines.append(r'\end{table}')
    lines.append('')

    Path(out_tex_path).write_text('\n'.join(lines), encoding='utf-8')


def individual_graph_removals(inputs):
    all_auc_rows = []

    for filename in inputs:
        filetype = filename.split('.')[-1]
        path, r, subdir = ((f'Graph_files/TGF_Files/{filename}', 45, 'TGF_Files') if filetype == 'tgf' else (f'Graph_files/{filename}', 25, 'JSON_Files'))
        graph = plot.load_graph(path, filetype)

        # ensure output dir exists:
        out_dir = os.path.join('Removals', subdir)
        os.makedirs(out_dir, exist_ok=True)
        
        all_dfs = []

        for fn, name in strategies:
            df = simulate_strategy(graph, fn, name)
            df = add_cumulative_auc(df, metrics=('aG', 'e0_mult', 'e1_mult', 'CIS'))
            df.to_csv(os.path.join(out_dir, f'{filename}_{name}.csv'),index=False)
            all_dfs.append(df)

        combined = pd.concat(all_dfs, ignore_index=True)

        auc_summary = summarize_auc(combined, graph_name=filename,
                                    metrics=('aG', 'e0_mult', 'e1_mult', 'CIS'))
        auc_summary = add_vs_random(auc_summary, metrics=('aG', 'e0_mult', 'e1_mult', 'CIS'))


         # neat rounding for easy LaTeX
        auc_summary_rounded = auc_summary.copy()
        for col in auc_summary_rounded.columns:
            if col.startswith('AUC_') or col.startswith('Delta_'):
                auc_summary_rounded[col] = auc_summary_rounded[col].round(5)

        auc_path = os.path.join(out_dir, f'{filename}_AUC_summary.csv')
        auc_summary_rounded.to_csv(auc_path, index=False)

        
        for metric, pretty in [
            ('aG',       r'a(G)'),
            ('e0_mult',  r'm_0'),
            ('e1_mult',  r'm_1'),
            ('CIS',      r'CIS'),
        ]:
            out_onepanel = os.path.join(out_dir, f'{filename}_onepanel_{metric}.jpg')
            plot_metric_onepanel_styled(
                combined, metric, out_onepanel,
                title=f'{filename}: {pretty} vs Nodes Removed (all strategies)'
            )


        tex_path = os.path.join(out_dir, f'{filename}_AUC_table.tex')
        emit_latex_auc_table(
            graph_name=filename,
            auc_summary_df=auc_summary_rounded,  # use your nicely rounded version
            out_tex_path=tex_path,
            caption=f'AUC summary per removal strategy for {filename}',
            label=f'tab:auc_{os.path.splitext(filename)[0]}',
            round_digits=4
        )

        # collect for global table
        all_auc_rows.append(auc_summary)

        # plot each metric
        for metric in ['aG','e1_mult','e0_mult','CIS']:
            plot_metric_small_multiples(
                combined,
                metric,
                os.path.join(out_dir, f'{filename}_compare_{metric}.jpg')
            )

        print(f'Finished Phase 2 for {filename} → results in removals/{subdir}/')

        if all_auc_rows:
            all_auc = pd.concat(all_auc_rows, ignore_index=True)
            # order columns nicely
            cols = ['graph', 'strategy', 'steps',
                    'AUC_aG', 'AUC_e0_mult', 'AUC_e1_mult', 'AUC_CIS',
                    'Delta_aG_vs_random', 'Delta_e0_mult_vs_random',
                    'Delta_e1_mult_vs_random', 'Delta_CIS_vs_random']
            cols = [c for c in cols if c in all_auc.columns]  # guard
            all_auc = all_auc[cols]
            all_auc = all_auc.round(5)
            os.makedirs('Removals', exist_ok=True)
            all_auc.to_csv('Removals/AUC_summary_all_graphs.csv', index=False)

def main():
    inputs = ['bfn.tgf', 'cpt.tgf', 'dur.tgf', 'els.tgf', 'jnb.tgf', 'pta.tgf', 'pzb.tgf', 'vdp.tgf', 'isis-links.json',]
    individual_graph_removals(inputs)


if __name__=="__main__":
    main()