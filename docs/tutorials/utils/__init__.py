"""
Tutorial utilities for the Evolve Framework.

This module provides:
- Synthetic data generators for benchmarks, symbolic regression, and causal discovery
- Visualization functions for fitness, diversity, Pareto fronts, and speciation
- Mermaid diagram rendering for Jupyter notebooks
- EA-to-ML terminology glossary

Usage:
    from docs.tutorials.utils import (
        # Data generators
        get_benchmark, sphere_function, rastrigin_function,
        generate_polynomial_data, generate_causal_dag_data,

        # Visualization
        render_mermaid, plot_fitness_history, plot_pareto_3d_interactive,

        # Terminology
        get_glossary, print_glossary_table,

        # Data structures
        BenchmarkFunction, EvolutionHistory, ParetoFront, IslandConfig,
    )
"""

from .tutorial_utils import (
    EVOLUTIONARY_LOOP_DIAGRAM,
    GENOME_PHENOTYPE_DIAGRAM,
    ISLAND_MODEL_DIAGRAM,
    # Data structures
    BenchmarkFunction,
    BenchmarkResult,
    CausalDAGData,
    EvolutionHistory,
    IslandConfig,
    MigrationEvent,
    ParetoFront,
    SpeciesHistory,
    SymbolicRegressionData,
    TerminologyEntry,
    ackley_function,
    # GPU utilities
    check_gpu_available,
    compare_runs_statistical,
    # Statistical utilities (FR-010)
    compute_population_stats,
    convergence_test,
    # Island model utilities (FR-020-023)
    create_island_config,
    # Causal data (FR-003)
    generate_causal_dag_data,
    generate_chain_dag_data,
    generate_composite_data,
    # Symbolic regression data (FR-002)
    generate_polynomial_data,
    generate_trigonometric_data,
    get_benchmark,
    # Terminology (FR-009)
    get_glossary,
    # Causal graph visualization (FR-054)
    plot_causal_graph_comparison,
    plot_crowding_distance_visual,
    plot_diversity_over_generations,
    plot_fitness_comparison,
    # Fitness plots (FR-007)
    plot_fitness_history,
    # Pareto visualization (FR-048, FR-049, FR-050)
    plot_pareto_2d_projections,
    plot_pareto_3d_interactive,
    plot_pareto_evolution,
    # Diversity visualization (FR-008)
    plot_population_diversity,
    plot_species_phylogeny,
    # Speciation visualization (FR-038)
    plot_species_stacked_area,
    print_glossary_table,
    rastrigin_function,
    # Mermaid rendering (FR-006)
    render_mermaid,
    rosenbrock_function,
    # Benchmark functions (FR-001)
    sphere_function,
    visualize_topology,
)

__all__ = [
    # Data structures
    "BenchmarkFunction",
    "SymbolicRegressionData",
    "CausalDAGData",
    "EvolutionHistory",
    "SpeciesHistory",
    "ParetoFront",
    "TerminologyEntry",
    "IslandConfig",
    "MigrationEvent",
    "BenchmarkResult",
    # Benchmark functions
    "sphere_function",
    "rastrigin_function",
    "rosenbrock_function",
    "ackley_function",
    "get_benchmark",
    # Data generators
    "generate_polynomial_data",
    "generate_trigonometric_data",
    "generate_composite_data",
    "generate_causal_dag_data",
    "generate_chain_dag_data",
    # Visualization
    "render_mermaid",
    "EVOLUTIONARY_LOOP_DIAGRAM",
    "GENOME_PHENOTYPE_DIAGRAM",
    "ISLAND_MODEL_DIAGRAM",
    "plot_fitness_history",
    "plot_fitness_comparison",
    "plot_population_diversity",
    "plot_diversity_over_generations",
    "plot_pareto_2d_projections",
    "plot_pareto_3d_interactive",
    "plot_pareto_evolution",
    "plot_crowding_distance_visual",
    "plot_species_stacked_area",
    "plot_species_phylogeny",
    "plot_causal_graph_comparison",
    # Terminology
    "get_glossary",
    "print_glossary_table",
    # Statistics
    "compute_population_stats",
    "convergence_test",
    "compare_runs_statistical",
    # Island model
    "create_island_config",
    "visualize_topology",
    # GPU
    "check_gpu_available",
]
