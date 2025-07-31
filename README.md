# SANREN HONOURS PROJECT

## Notes;
    - What benchmark will be used to evaluate SANReN's spectral and core resilience results?
        - Must compare the results to those of more general network topologies such as a star, bus, ring, mesh, hybrid, and tree.

    - Research Question: To what extent can spectral graph theory and core resilience analysis quantitatively evaluate the structural resilience of SANReN topology - including node criticality, node centrality, and connectivity - and how effectively can these methods inform the design of a more resilient network topology?

    Phase 1:
        - Create a framework to evaluate the effectiveness of my chosen evaluation metrics - core resilience and spectral graph theory.
        - Compute initial core resilience and spectral analyses of the given SANReN topology.
        - Compute classial graph-theoretic measures:
            - Degree centrality.
            - Betweeness centrality.
            - Closeness centrality.

        - Compute statistical correlation.
            - Compare nodes in high core influence to those with the highest degree, betweeness and closeness centrality using a Spearman Rank test to test the effectiveness of Core Influence in identifying node criticality and centrality against classical graph measures.

            - The Spearman rank correlation test identifies how how strongly two sets of rankings are correlated. For example, if we rank nodes from most to least critical using two different methods, Spearman's correlation checks how similar those rankings are.

            1. Spearman Coefficient (ρ)
                What it is:
                The Spearman rank correlation coefficient measures how well the relationship between two ranked variables can be described by a monotonic function (values moving in the same or opposite direction).

                    - ρ = 1: Perfect positive monotonic relationship (two rankings are very similar)
                    - ρ = -1: Perfect negative monotonic relationship (the rankings are very different)
                    - ρ = 0:  No monotonic relationship (there’s little to no consistent relationship between the rankings.)

                Interpretation:
                    - Closer to ±1 → Stronger relationship between rankings (nodes ranked similarly across two measures).
                    - Closer to 0 → Little to no correlation between the rankings.

            2. p-value
                What it is:
                The p-value tells you whether the observed correlation could have occurred by chance if there were really no relationship between the rankings.

                Interpretation:
                    - Small p-value (e.g., < 0.05): There is a statistically significant correlation.
                    - Large p-value: The observed correlation could be due to random chance (not statistically significant).
            
            - By evaluating the effectiveness of core influence in identifying node criticality and centrality provides a basis of authenticity of the Core-Influence Strength metric in evaluating a broader centrality measure of a given graph.

        - Experimentation:
            - Experiment with different sized graphs 
                1. Samples of the SANReN graph:
                    - Select random edges from the isis-links.json edges dictionary to plot smaller sample sized graphs.
                    - Edges = [50, 100, 150, 200, 250, 300, 350, 400]

                2. Classical graph generation methods (Not yet implemented):
                    - Edges = [10, 100, 1000, 10000, 100000]
                    - allocate each node a random name - random id value generator
                    - use graph generation methods - random graph, graph geometric, gariel graphs, waxman graphs. 
                - Create visualisations
                - Write spectral, core resilience, classical measures, top-r measures for each generated graph.

        - Findings (In progress):
            - Compare each metric's average value, e.g., betweeness centrality sample 1, 2, 3, ..., 8, to graph size.
                - Take each sample, calculate the average, store it and plot it to a line graph. 
                - This gives an average metric accross the graph samples.

            - Plot each samples CIS metric to a graph
            - Plot each samples spectral results to a graph:
                - Algebraic connectivity of each sample
                - Multiplicity of the one eigenvalue of each sample
                - Density of eigenvalues around 1 of each sample
                - Multiplicity of the zero eigenvalue of each sample
        
        - Phase 1 builds a framework to evaluate the effectiveness of core resilience analysis in identifying critical and central nodes against classical graph measures by evaluating the top-r nodes of each sample graph.
    
    Phase 2:
        - Simulate node removals:
            1. Removing random nodes (Acts as the BASELINE).
            2. Removing top-K nodes using core strength/influence.
            3. Removing top-K nodes using degree centrality.
            4. Removing top-K nodes using closeness centrality.
            5. Removing top-K nodes using betweeness centrality.

        - Recompute spectral analysis to compare changes in connectivity (a(G)), fragmentation (multiplicty of the zero eigenvalue) and redundancy (multiplicity and density of the one eigenvalue). 
        - This helps identify how well core resilience metrics identify and predict structural degredation, node criticality and centrality.



