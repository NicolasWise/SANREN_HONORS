# SANREN HONOURS PROJECT

## Methodology
    - Research Question: To what extent can spectral graph theory and core resilience analysis quantitatively evaluate the structural resilience of SANReN topology (Phase 1) - including node criticality, node centrality, and connectivity - and how effectively can these methods inform the design of a more resilient network topology (Phase2)?

    Phase 1:
        - Create a framework to evaluate the effectiveness of my chosen evaluation metrics - core resilience and spectral graph theory.
        - Compute initial core resilience and spectral analyses of the given SANReN topology.
            - Core Influence is a node centrality measure.
            - Core Number/Strength are resilience measures.
        - Compute classial graph-theoretic measures:
            - Degree centrality.
            - Betweeness centrality.
            - Closeness centrality.

        - Compute the top-r (10%) of nodes with the highes core influence, strenght, number as well as classical measures.

        - Compute statistical correlation between classical measures and core influence (Centrality measures).
            - Compare top-r (top 10%) of nodes in high core influence to degree, betweeness and closeness centrality using a Spearman Rank test to test the effectiveness of Core Influence in identifying node criticality and centrality against classical graph measures.

            - The Spearman rank correlation test identifies how how strongly two sets of rankings are correlated. For example, if we rank nodes from most to least critical using two different methods, Spearman's correlation checks how similar those rankings are.

            1. Spearman Coefficient (ρ)
                The Spearman rank correlation coefficient measures how two ranked lists are related (values moving in the same or opposite direction).

                    - ρ = 1: Perfect positive monotonic relationship (two rankings are very similar)
                    - ρ = -1: Perfect negative monotonic relationship (the rankings are very different)
                    - ρ = 0:  No monotonic relationship (there’s little to no consistent relationship between the rankings.)

                Interpretation:
                    - Closer to ±1 → Stronger relationship between rankings (nodes ranked similarly across two measures).
                    - Closer to 0 → Little to no correlation between the rankings.

            2. p-value
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

        - Findings:
            - Record each samples CIS metric to evaluate each sample graph's broader connectivity.
            - Record each samples spectral results to a graph:
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

        - Interpretation:
            - Steeper drop in a(G) means that strategy finds more “critical” nodes.
            - Faster rise in fragmentation (multiplicity of the zero eigenvalue) shows how quickly the network shatters.

            - Comparing curves tells you:
                - Does targeting by core-influence break connectivity faster than targeting by betweenness?
                - Are classical centralities nearly as predictive as your core metrics, or do core metrics consistently outperform them?


## Notes
    - Need to validate core strength
