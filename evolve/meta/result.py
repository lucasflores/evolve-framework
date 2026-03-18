"""
Meta-Evolution Result.

Provides result container for meta-evolution runs, including
best configuration, solution, and export capabilities.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from evolve.config.unified import UnifiedConfig
    from evolve.core.population import Population


@dataclass(frozen=True)
class MetaEvolutionResult:
    """
    Result of a meta-evolution run (T068).
    
    Contains the best configuration found and the solution(s)
    discovered using that configuration.
    
    Attributes:
        best_config: Best configuration found.
        best_fitness: Fitness achieved by best configuration.
        best_solution: Best individual found with best configuration.
        final_population: Final outer population of configurations.
        history: History of outer evolution (fitness per generation).
        trials_run: Total number of inner evolution trials run.
    
    Example:
        >>> result = run_meta_evolution(...)
        >>> result.export_best_config("best_config.json")
        >>> print(f"Best fitness: {result.best_fitness}")
    """
    
    best_config: "UnifiedConfig"
    """Best configuration found."""
    
    best_fitness: float
    """Fitness achieved by best configuration."""
    
    best_solution: Any | None
    """Best individual found with best configuration."""
    
    final_population: list[tuple["UnifiedConfig", float]]
    """Final outer population: list of (config, fitness) tuples."""
    
    history: list[dict[str, Any]]
    """History of outer evolution metrics."""
    
    trials_run: int
    """Total number of inner evolution trials run."""
    
    def get_pareto_configs(
        self,
        objectives: list[str] | None = None,
    ) -> list[tuple["UnifiedConfig", dict[str, float]]]:
        """
        Get Pareto-optimal configurations (T069).
        
        For multi-objective meta-evolution, filters to non-dominated
        configurations. For single-objective, returns sorted by fitness.
        
        Args:
            objectives: Objective names for multi-objective filtering.
                None for single-objective (returns all sorted).
                
        Returns:
            List of (config, fitness_dict) tuples on Pareto front.
        """
        if objectives is None:
            # Single-objective: sort by fitness
            sorted_pop = sorted(
                self.final_population,
                key=lambda x: x[1],
                reverse=not _should_minimize(self.best_config),
            )
            return [
                (cfg, {"fitness": fit})
                for cfg, fit in sorted_pop
            ]
        
        # Multi-objective: extract Pareto front
        configs_with_objectives = []
        for cfg, fit in self.final_population:
            obj_values = {}
            # Try to extract objective values from stored data
            # This is a simplified implementation
            obj_values["primary"] = fit
            configs_with_objectives.append((cfg, obj_values))
        
        # Filter non-dominated (simple implementation)
        pareto_front = []
        for i, (cfg_i, obj_i) in enumerate(configs_with_objectives):
            dominated = False
            for j, (cfg_j, obj_j) in enumerate(configs_with_objectives):
                if i != j:
                    if _dominates(obj_j, obj_i):
                        dominated = True
                        break
            if not dominated:
                pareto_front.append((cfg_i, obj_i))
        
        return pareto_front
    
    def export_best_config(
        self,
        path: str,
        include_solution: bool = False,
    ) -> None:
        """
        Export best configuration to JSON file (T070).
        
        Args:
            path: Output file path.
            include_solution: Whether to include solution in output.
        """
        data: dict[str, Any] = {
            "config": self.best_config.to_dict(),
            "fitness": self.best_fitness,
            "trials_run": self.trials_run,
        }
        
        if include_solution and self.best_solution is not None:
            # Attempt to serialize solution
            try:
                data["solution"] = _serialize_solution(self.best_solution)
            except Exception:
                data["solution"] = str(self.best_solution)
        
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    
    def summarize(self) -> str:
        """Get human-readable summary."""
        lines = [
            "Meta-Evolution Result",
            "=" * 40,
            f"Best Fitness: {self.best_fitness:.6f}",
            f"Trials Run: {self.trials_run}",
            f"Final Population Size: {len(self.final_population)}",
            "",
            "Best Configuration:",
        ]
        
        config_dict = self.best_config.to_dict()
        for key in ["population_size", "crossover_rate", "mutation_rate", 
                    "selection", "crossover", "mutation"]:
            if key in config_dict:
                lines.append(f"  {key}: {config_dict[key]}")
        
        return "\n".join(lines)


def _should_minimize(config: "UnifiedConfig") -> bool:
    """Check if configuration minimizes fitness."""
    return config.minimize


def _dominates(a: dict[str, float], b: dict[str, float]) -> bool:
    """Check if a dominates b (Pareto dominance)."""
    all_keys = set(a.keys()) | set(b.keys())
    at_least_one_better = False
    
    for key in all_keys:
        val_a = a.get(key, float("inf"))
        val_b = b.get(key, float("inf"))
        if val_a > val_b:
            return False
        if val_a < val_b:
            at_least_one_better = True
    
    return at_least_one_better


def _serialize_solution(solution: Any) -> Any:
    """Attempt to serialize solution to JSON-compatible format."""
    if hasattr(solution, "to_dict"):
        return solution.to_dict()
    if hasattr(solution, "genome"):
        genome = solution.genome
        if hasattr(genome, "genes"):
            return {"genes": list(genome.genes)}
        return {"genome": str(genome)}
    return str(solution)
