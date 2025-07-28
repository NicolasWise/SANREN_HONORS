# SANREN HONOURS PROJECT

## Notes;
    - What benchmark will be used to evaluate SANReN's spectral and core resilience results?
        - Must compare the results to those of more general network topologies such as a star, bus, ring, mesh, hybrid, and tree.

    Phase 1:
        - Compute initial core resilience and spectral analyses.
        - Compute classial graph-theoretic measures:
            - Degree centrality.
            - Betweeness centrality.
            - Closeness centrality.
        - Compute statistical correlation.
        - Compare nodes in high core strength/influence to those with the highest degree, betweeness and closeness centrality. 
        - This phase identifies nodes with high ^ centrality and core resilience measures. This evaluates the effectivess of the core resilience analysis in identifying node criticality and centrality.

        - Experimentation:
            - Graphical visualisations
            - experiment with different sized graphs
            - create a graph generator to generate a dictionary of edges of a given input size.  
                Classical graph generation methods:
                    - edges = [10, 100, 1000, 10000, 100000]
                    - allocate each node a random name - random id value generator
                    - use graph generation methods - random graph, graph geometric, gariel graphs, waxman graphs. 
                Sample of the SANReN graph:
                    - select random edges from the isis-links.json edges dictionary to plot smaller sample sized graphs.
                    - length = Graph.size() #gives you the number of edges
                    - edges = [length, 10%(length), 25%(length), 50%(length), ..., 95%(length)]
                - write these to a json
                - create visualisations
                - write spectral, core resilience, classical measures, top r measures for each generated graph.
                
        
    
    Phase 2:
        - Simulate node removals:
            1. Removing random nodes (Acts as the BASELINE).
            2. Removing top-K nodes using core strength/influence.
            3. Removing top-K nodes using degree centrality.
            4. Removing top-K nodes using closeness centrality.
            5. Removing top-K nodes using betweeness centrality.

        - Recompute spectral analysis to compare changes in connectivity (a(G)), fragmentation (multiplicty of the zero eigenvalue) and redundancy (multiplicity and density of the one eigenvalue). 
        - This helps identify how well core resilience metrics identify and predict structural degredation, node criticality and centrality.

