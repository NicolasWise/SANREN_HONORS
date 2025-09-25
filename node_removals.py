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

"""
Simulate node removals from a graph according to various strategies,
computing spectral and core metrics at each step.

Nicolas Wise
"""


def removal_random(graph):
    nodes = list(graph.nodes())
    random.shuffle(nodes)
    return nodes

def removal_top_k_core_number(graph):
    """ Return nodes sorted by core number descending. """
    core_number, _, _, _ = core.compute_core_resilience(graph)
    return {node: value for node, value in sorted(core_number.items(), key=lambda x:-x[1])}

def removal_top_k_core_influence(graph):
    """ Return nodes sorted by core influence descending. """
    _, _, core_influence, _ = core.compute_core_resilience(graph)
    return {node:value for node, value in sorted(core_influence.items(), key = lambda x:-x[1])}

def removal_top_k_degree(graph):
    """ Return nodes sorted by degree descending. """
    deg = dict(graph.degree())
    return {node: value for node, value in sorted(deg.items(), key=lambda x:-x[1])}

def removal_top_k_closenss(graph):
    """ Return nodes sorted by closeness descending. """
    _, closeness, _ = classic.compute_classical_graph_measures(graph)
    return {n: v for n, v in sorted(closeness.items(), key=lambda x: -x[1])}

def removal_top_k_betweeness(graph):
    """ Return nodes sorted by betweenness descending. """
    _, _, betweenness = classic.compute_classical_graph_measures(graph)
    return {n: v for n, v in sorted(betweenness.items(), key=lambda x: -x[1])}

# Define removal strategies
strategies = [
    (removal_random, 'random'),
    (removal_top_k_core_influence, 'core influence'),
    (removal_top_k_degree, 'degree'),
    (removal_top_k_betweeness, 'betweeness'),
    (removal_top_k_closenss, 'closeness'),
]

def simulate_strategy(graph, strategy_fn, strategy_name, r):
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
    """ One figure per metric; one line per removal strategy. """
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
    """ Small multiples: one subplot per removal strategy. """
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

def main():
    inputs = ['bfn.tgf', 'cpt.tgf', 'dur.tgf', 'els.tgf', 'jnb.tgf', 'pta.tgf', 'pzb.tgf', 'vdp.tgf', 'isis-links.json',]
    for filename in inputs:
        filetype = filename.split('.')[-1]
        path, r, subdir = ((f'Graph_files/TGF_Files/{filename}', 45, 'TGF_Files') if filetype == 'tgf' else (f'Graph_files/{filename}', 25, 'JSON_Files'))
        # Load graph
        graph = plot.load_graph(path, filetype)

        # ensure output dir exists:
        out_dir = os.path.join('Removals', subdir)
        os.makedirs(out_dir, exist_ok=True)
        
        all_dfs = []
        # Run each removal strategy for a given graph
        for fn, name in strategies:
            df = simulate_strategy(graph, fn, name, r)
            df.to_csv(os.path.join(out_dir, f'{filename}_{name}.csv'),index=False)
            all_dfs.append(df)

        combined = pd.concat(all_dfs, ignore_index=True)
        # Summarize AUC
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


if __name__=="__main__":
    main()