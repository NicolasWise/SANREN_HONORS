import random
import os

import core_resilience as core
import classical_graph_measures as classic
import spectral_analysis as spec
import plotter as plot

import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd


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
    (removal_top_k_core_number, 'core number'),
    (removal_top_k_core_influence, 'core influence'),
    (removal_top_k_core_strengths, 'core strength'),
    (removal_top_k_degree, 'degree'),
    (removal_top_k_betweeness, 'betweeness'),
    (removal_top_k_closenss, 'closeness'),
]

def simulate_strategy(graph, strategy_fn, strategy_name, r):
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

def main():
    inputs = ['bfn.tgf', 'cpt.tgf', 'dur.tgf', 'els.tgf', 'jnb.tgf', 'pta.tgf', 'pzb.tgf', 'vdp.tgf', 'isis-links.json',]
    for filename in inputs:
        filetype = filename.split('.')[-1]
        path, r, subdir = ((f'Graph_files/TGF_Files/{filename}', 45, 'TGF_Files') if filetype == 'tgf' else (f'Graph_files/{filename}', 25, 'JSON_Files'))
        graph = plot.load_graph(path, filetype)

        # ensure output dir exists:
        out_dir = os.path.join('Removals', subdir)
        os.makedirs(out_dir, exist_ok=True)
        
        all_dfs = []

        for fn, name in strategies:
            df = simulate_strategy(graph, fn, name, r)
            df.to_csv(os.path.join(out_dir, f'{filename}_{name}.csv'),index=False)
            all_dfs.append(df)

        combined = pd.concat(all_dfs, ignore_index=True)

        # plot each metric
        for metric in ['aG','e1_mult','e0_mult','CIS']:
            plot_metric(
                combined,
                metric,
                os.path.join(out_dir, f'{filename}_compare_{metric}.png')
            )

        print(f'Finished Phase 2 for {filename} → results in removals/{subdir}/')


if __name__=="__main__":
    main()