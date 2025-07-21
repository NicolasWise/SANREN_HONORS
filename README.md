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
        
    
    Phase 2:
        - Simulate node removals:
            1. Removing random nodes (Acts as the BASELINE).
            2. Removing top-K nodes using core strength/influence.
            3. Removing top-K nodes using core degree centrality.
            4. Removing top-K nodes using core closeness centrality.
            5. Removing top-K nodes using core betweeness centrality.

        - Recompute spectral analysis to compare changes in connectivity (a(G)), fragmentation (multiplicty of the zero eigenvalue) and redundancy (multiplicity and density of the one eigenvalue). 
        - This helps identify how well core resilience metrics identify and predict structural degredation. 

