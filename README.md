# SANREN HONOURS PROJECT

## Overview

This project provides a framework for analyzing the structural resilience of network topologies, with a focus on the SANReN network. It uses spectral graph theory and core resilience metrics to evaluate node criticality, centrality, and overall connectivity. The codebase supports experiments on both real-world and synthetic graphs, simulating node removals and edge reinforcements to measure the impact on network robustness.

## Key Features

- **Core Resilience Metrics:** Compute core number, core strength, core influence, and the Core Influence-Strength (CIS) metric for each node.
- **Classical Graph Measures:** Degree, closeness, and betweenness centrality.
- **Spectral Analysis:** Algebraic connectivity, eigenvalue multiplicities, and spectral clustering.
- **Node Removal Simulations:** Evaluate resilience under various node removal strategies.
- **Reinforcement Strategies:** Iteratively add edges to improve resilience using multiple algorithms.
- **Automated Plotting and CSV Export:** Visualize results and export metrics for further analysis.

---

## File Descriptions

### `reinforcements.py`

Implements edge reinforcement experiments. After each step of edge additions, the code runs node-removal simulations to measure resilience using AUC metrics for algebraic connectivity (`aG`), eigenvalue multiplicities (`e0_mult`, `e1_mult`), and CIS.

**Reinforcement Strategy Algorithms:**
- **Fiedler-Greedy:** Adds edges between node pairs that maximize the squared difference in their Fiedler vector values, targeting connectivity bottlenecks.
- **Random Add:** Adds random non-edges between nodes.
- **MRKC Heuristic:** Connects nodes with minimum core number (vulnerable) to those with maximum core number (anchors), prioritizing degree for tie-breaking.

Each strategy is implemented as a function (`next_edge_fiedler_greedy`, `next_edge_random`, `next_edge_mrkc_heuristic`) and used in iterative experiments.

### `core_resilience.py`

Implements core resilience metrics:

- **Core Number:** The highest k for which a node remains in the k-core.
- **Core Strength:** For node `u`, `CS(u) = |{v ∈ Γ(u): κ(v) ≥ κ(u)}| - κ(u) + 1`, where `κ(u)` is the core number and `Γ(u)` is the neighborhood.
- **Core Influence:** Computed as the leading eigenvector of a matrix `M` that encodes influence relationships based on core numbers. The matrix is constructed so that nodes with higher or equal core numbers contribute to each other's influence.
- **Core Influence-Strength (CIS):** The average core strength of nodes in the top f-percentile of core influence values. This metric summarizes the resilience of the most influential nodes.

### `reinforce_no_removal.py`

Runs reinforcement experiments **without** node removals. After each edge addition step, it computes and plots the metrics (`aG`, `e0_mult`, `e1_mult`, `CIS`) for the reinforced graph. Supports the same reinforcement strategies as `reinforcements.py`.

### `node_removals.py`

Simulates node removals using various strategies (random, core influence, degree, betweenness, closeness) and computes resilience metrics at each step. Generates plots and summary tables for AUC values.

### `plotter.py`

Handles graph loading (from TGF and JSON), plotting, and exporting analysis results to CSV and PNG files. Also provides functions for identifying top/bottom percentile nodes by metric.

### `classical_graph_measures.py`

Computes classical centrality measures using NetworkX:
- Degree centrality
- Closeness centrality
- Betweenness centrality

---

## Methodology

1. **Graph Loading:** Supports TGF and JSON formats.
2. **Metric Computation:** Computes spectral, core resilience, and classical metrics for each graph.
3. **Node Removal Experiments:** Simulates removals using multiple strategies, records metric trajectories, and summarizes results with AUC.
4. **Reinforcement Experiments:** Adds edges using different algorithms, measures improvement in resilience, and compares strategies.
5. **Visualization and Export:** Plots metric trajectories and exports results for further analysis.

---

## How to Run

- **Node Removal Experiments:** Run `node_removals.py` to simulate removals and generate results.
- **Reinforcement Experiments:** Run `reinforcements.py` or `reinforce_no_removal.py` for edge addition experiments.
- **Graph Analysis:** Use `plotter.py` to analyze and export metrics for individual graphs.

---

## References

- [`reinforcements.py`](reinforcements.py)
- [`core_resilience.py`](core_resilience.py)
- [`reinforce_no_removal.py`](reinforce_no_removal.py)
- [`node_removals.py`](node_removals.py)
- [`plotter.py`](plotter.py)
- [`classical_graph_measures.py`](classical_graph_measures.py)

---

## Contact

Nicolas