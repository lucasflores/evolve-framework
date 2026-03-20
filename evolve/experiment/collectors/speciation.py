"""
Speciation metrics collector.

Collects metrics related to species dynamics when speciation is enabled.

NO ML FRAMEWORK IMPORTS ALLOWED.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from evolve.experiment.collectors.base import MetricCollector, CollectionContext

if TYPE_CHECKING:
    from evolve.core.population import Population
    from evolve.core.types import Individual


@dataclass
class SpeciationMetricCollector(MetricCollector):
    """
    Collect species-related metrics (FR-008, FR-015).
    
    Tracks species dynamics including counts, sizes, and extinctions.
    Only active when speciation is enabled in the evolution configuration.
    
    species_info format: dict[int, list[int]] where:
        - keys are species_id
        - values are lists of individual indices in population
    
    Attributes:
        track_dynamics: Whether to track births/extinctions/stagnation
        _previous_species_ids: Species IDs from previous generation
        _previous_best_fitness: Per-species best fitness from previous gen
        _stagnation_counters: Per-species fitness improvement counters
    
    Example:
        >>> collector = SpeciationMetricCollector()
        >>> context = CollectionContext(
        ...     generation=10,
        ...     population=population,
        ...     species_info={0: [0, 1, 2], 1: [3, 4]},
        ... )
        >>> metrics = collector.collect(context)
        >>> assert "species_count" in metrics
    """
    
    track_dynamics: bool = True
    
    _previous_species_ids: set[int] = field(default_factory=set, repr=False)
    _previous_best_fitness: dict[int, float] = field(default_factory=dict, repr=False)
    _stagnation_counters: dict[int, int] = field(default_factory=dict, repr=False)
    
    def collect(self, context: CollectionContext) -> dict[str, float]:
        """
        Collect speciation metrics.
        
        Args:
            context: Collection context with species_info
            
        Returns:
            Dictionary of speciation metrics:
                - species_count: Number of active species
                - average_species_size: Mean individuals per species
                - largest_species_fitness: Best fitness in largest species
                - species_births: New species this generation (if tracking)
                - species_extinctions: Species that went extinct (if tracking)
                - stagnation_count: Species with no improvement (if tracking)
        """
        species_info = context.species_info
        
        if species_info is None:
            return {}
        
        metrics: dict[str, float] = {}
        
        # Core species metrics
        species_count = len(species_info)
        metrics["species_count"] = float(species_count)
        
        if species_count == 0:
            return metrics
        
        # Average species size
        total_individuals = sum(len(members) for members in species_info.values())
        metrics["average_species_size"] = total_individuals / species_count
        
        # Find largest species and its best fitness
        population = context.population
        largest_species_id = max(species_info.keys(), key=lambda sid: len(species_info[sid]))
        largest_members = species_info[largest_species_id]
        
        # Get best fitness from largest species
        if largest_members:
            fitness_values = []
            for i in largest_members:
                ind = population[i]
                if ind.fitness is not None:
                    fitness_values.append(ind.fitness[0])
            if fitness_values:
                metrics["largest_species_fitness"] = float(max(fitness_values))
        
        # Species IDs for this generation
        current_ids = set(species_info.keys())
        
        # Compute per-species best fitness for stagnation tracking
        current_best_fitness: dict[int, float] = {}
        for species_id, member_indices in species_info.items():
            if member_indices:
                species_fitness_values = []
                for i in member_indices:
                    ind = population[i]
                    if ind.fitness is not None:
                        species_fitness_values.append(ind.fitness[0])
                if species_fitness_values:
                    current_best_fitness[species_id] = max(species_fitness_values)
        
        # Species dynamics tracking
        if self.track_dynamics:
            # Births: new species that didn't exist before
            births = current_ids - self._previous_species_ids
            metrics["species_births"] = float(len(births))
            
            # Extinctions: species that existed before but are gone
            extinctions = self._previous_species_ids - current_ids
            metrics["species_extinctions"] = float(len(extinctions))
            
            # Stagnation: species with no fitness improvement
            stagnation_count = 0
            for species_id in current_ids:
                current_best = current_best_fitness.get(species_id)
                prev_best = self._previous_best_fitness.get(species_id)
                
                if prev_best is not None and current_best is not None:
                    if current_best <= prev_best:
                        self._stagnation_counters[species_id] = (
                            self._stagnation_counters.get(species_id, 0) + 1
                        )
                    else:
                        self._stagnation_counters[species_id] = 0
                
                if self._stagnation_counters.get(species_id, 0) > 0:
                    stagnation_count += 1
            
            metrics["stagnation_count"] = float(stagnation_count)
            
            # Update best fitness state for next generation
            self._previous_best_fitness = current_best_fitness
        
        # Update previous species IDs
        self._previous_species_ids = current_ids
        
        return metrics
    
    def reset(self) -> None:
        """Reset internal state for a new evolution run."""
        self._previous_species_ids = set()
        self._previous_best_fitness = {}
        self._stagnation_counters = {}


__all__ = ["SpeciationMetricCollector"]
