# SANReN Network Resilience Analysis

This repository contains code and tools for analyzing and improving the resilience of network topologies, with a focus on the South African National Research and Education Network (SANReN). The project implements advanced graph-theoretic and spectral methods to simulate node removals, reinforce network connectivity, and evaluate robustness using both classical and novel metrics.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Usage](#usage)
  - [Graph Analysis](#graph-analysis)
  - [Node Removal Experiments](#node-removal-experiments)
  - [Reinforcement Experiments](#reinforcement-experiments)
  - [No-Removal Reinforcement](#no-removal-reinforcement)
  - [AUC Plotting](#auc-plotting)
- [Core Metrics and Algorithms](#core-metrics-and-algorithms)
  - [Core Strength](#core-strength)
  - [Core Influence](#core-influence)
  - [Core Influence-Strength (CIS)](#core-influence-strength-cis)
  - [Spectral Metrics](#spectral-metrics)
  - [Classical Centrality Measures](#classical-centrality-measures)
- [Reinforcement Strategies](#reinforcement-strategies)
- [File Structure](#file-structure)
- [Outputs](#outputs)
- [References](#references)
- [Contact](#contact)

---

## Project Overview

The goal of this project is to provide a reproducible framework for:
- **Quantifying network resilience** under various node removal strategies.
- **Reinforcing** (adding edges to) a network using principled algorithms to maximize robustness.
- **Comparing** the effectiveness of different reinforcement strategies using Area Under Curve (AUC) metrics for key resilience indicators.

The code supports both real-world and synthetic graphs, and outputs detailed CSVs and plots for further analysis.

---

## Key Features

- **Automated graph loading** from TGF and JSON formats.
- **Computation of advanced resilience metrics**: algebraic connectivity, eigenvalue multiplicities, core strength, core influence, and CIS.
- **Simulation of node removals** using multiple strategies (random, core-based, centrality-based).
- **Edge reinforcement algorithms**: Fiedler-greedy, MRKC heuristic, and random addition.
- **Batch processing** for multiple graphs and strategies.
- **Publication-ready plots and CSV summaries** for all experiments.

---

## Installation

1. **Clone the repository:**
   git clone https://github.com/yourusername/HONORS_PROJECT_SANREN.git
   cd HONORS_PROJECT_SANREN

2. **Install dependencies:**
    Python 3.8+
    Required packages (install via pip)

---

## Usage

**Graph Analysis**

    > python plotter.py   

 - This will:
    - load all .json or .tgf files in the Graph_Files directory.
    - Compute baseline spectral, core resilience and classical metrics baselines.
    - Save plots to Analyses directory

**Node Removal Experiments**

    > python node_removals.py

 This will:
 - Run all removal strategies for each graph given in the main function of the class.
 - Output per-step and summary CSVs and plots in Removals/

**Reinforcement Experiments**
 - To run node removal experiments at each step of the reinforcement approach run: 

    > python reinforcements.py 

 - To run node reinforcements without applying node removals:
    
    > python reinforce_no_removal.py

 - This will 
    - Add edges iteratively using each strategy 
    - After each step, run node-removal experiments and compute AUCs
    - Output results in Reinforcements/ for application of removal suites and Reinforcements_NoRemoval/ for non-removal reinforcements.

**AUC Plotting**

    > python plot_auc_graphs.py

 - This will:
    - Aggregate AUCs over steps for all strategies
    - Output comparative plots in Reinforcements/.../_plots_AUC_by_reinforcement/

--

## Core Metrics and Algorithms

**Core Strength**
    - Definition: For node ( u ), ( CS(u) = |{v \in \Gamma(u) : \kappa(v) \geq \kappa(u)}| - \kappa(u) + 1 )
    
    - Interpretation: Measures the local redundancy of a node in its k-core shell. Higher values indicate more robust local support.
    
    - Implementation: See core_resilience.py:compute_core_strength.

**Core Influence**
    - Definition: Leading eigenvector of a support-flow matrix ( M ) constructed from the core structure. ( M[u, v] ) encodes how much node ( u ) can support node ( v ) based on their core numbers.

    - Interpretation: Highlights nodes that are structurally influential in the core-periphery structure, capturing both direct and indirect support.
    
    - Implementation: See core_resilience.py:compute_core_influence.

**Core Influence-Strength (CIS)**
    - Definition: The average core strength of the top ( f )-percentile nodes by core influence.

    - Interpretation: Summarizes the local redundancy of the most influential nodes, providing a single resilience indicator.

    - Implementation: See core_resilience.py:compute_CIS.

**Spectral Metrics**
    - Algebraic Connectivity (( a(G) )): Second-smallest eigenvalue of the Laplacian matrix. Higher values indicate better global connectivity.

    - Eigenvalue Multiplicities:
        - ( m_0 ): Multiplicity of eigenvalue 0 (number of connected components).
        - ( m_1 ): Multiplicity of eigenvalue 1 (related to redundancy).
    
    - Implementation: See spectral_analysis.py:compute_spectral_analysis.

**Classical Centrality Measures**
    - Degree Centrality
    - Closeness Centrality
    - Betweenness Centrality

    - Implementation: See classical_graph_measures.py:compute_classical_graph_measures.

## Reinforcement Strategies 

Implemented in both reinforcements.py and reinforce_no_removal.py:

**Fiedler-Greedy:** Adds edges between node pairs with the largest difference in Fiedler vector values, targeting the weakest connectivity bottlenecks.

**MRKC Heuristic:** Connects nodes with minimum core number (vulnerable) to those with maximum core number (anchors), prioritizing degree for tie-breaking.

**Random Add:** Adds random non-edges between nodes.
Each strategy is modular and can be extended or replaced.

## File Structure
- [`core_resilience.py`](./core_resilience.py) — Core strength, influence, and CIS metrics  
- [`classical_graph_measures.py`](./classical_graph_measures.py) — Degree, closeness, betweenness  
- [`spectral_analysis.py`](./spectral_analysis.py) — Algebraic connectivity, eigenvalues  
- [`plotter.py`](./plotter.py) — Load/plot graphs, export CSV  
- [`node_removals.py`](./node_removals.py) — Stress tests + AUC  
- [`reinforcements.py`](./reinforcements.py) — Reinforcement + removals evaluation  
- [`reinforce_no_removal.py`](./reinforce_no_removal.py) — Reinforcement (no removals)  
- [`plot_auc_graphs.py`](./plot_auc_graphs.py) — Plot AUC across reinforcement steps  
- [`Graph_files/`](./Graph_files/) — Input graphs  
- [`Analyses/`](./Analyses/) — Metrics & plots  
- [`Removals/`](./Removals/) — Removal outputs  
- [`Reinforcements/`](./Reinforcements/) — Reinforcement outputs  
- [`Reinforcements_NoRemoval/`](./Reinforcements_NoRemoval/) — No-removal outputs

## Outputs

- CSV files: Per-step and summary metrics for all experiments.
- Plots: Metric trajectories, AUC comparisons, and eigenvalue distributions.
- LaTeX tables: For easy inclusion in publications.
- Folders: Organized by experiment type, graph, and strategy.

## Author

Nicolas Wise,
University of Cape Town, 
Bachelors of Business Science, Honors in Computer Science.