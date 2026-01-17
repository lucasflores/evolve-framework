"""
Island topology functions.

Topologies define the migration connectivity graph between islands.
Each topology function returns a mapping from island ID to list
of neighbor island IDs.

NO ML FRAMEWORK IMPORTS ALLOWED (except NumPy).
"""

from __future__ import annotations

import math


def ring_topology(n_islands: int) -> dict[int, list[int]]:
    """
    Ring topology: each island connected to two neighbors.
    
    Creates a circular arrangement where island i is connected
    to islands (i-1) and (i+1), wrapping around at the ends.
    
    Good for gradual genetic mixing with local neighborhoods.
    
    Args:
        n_islands: Number of islands (must be >= 2)
        
    Returns:
        Mapping from island ID to list of neighbor IDs
        
    Raises:
        ValueError: If n_islands < 2
        
    Example:
        >>> ring_topology(4)
        {0: [3, 1], 1: [0, 2], 2: [1, 3], 3: [2, 0]}
    """
    if n_islands < 2:
        raise ValueError(f"Ring topology requires at least 2 islands, got {n_islands}")
    
    return {
        i: [(i - 1) % n_islands, (i + 1) % n_islands]
        for i in range(n_islands)
    }


def fully_connected_topology(n_islands: int) -> dict[int, list[int]]:
    """
    Fully connected topology: all islands connected to all others.
    
    Creates a complete graph where every island can migrate
    to every other island. Maximum genetic mixing.
    
    Good for fast convergence but may reduce diversity.
    
    Args:
        n_islands: Number of islands (must be >= 2)
        
    Returns:
        Mapping from island ID to list of neighbor IDs
        
    Raises:
        ValueError: If n_islands < 2
        
    Example:
        >>> fully_connected_topology(3)
        {0: [1, 2], 1: [0, 2], 2: [0, 1]}
    """
    if n_islands < 2:
        raise ValueError(
            f"Fully connected topology requires at least 2 islands, got {n_islands}"
        )
    
    return {
        i: [j for j in range(n_islands) if j != i]
        for i in range(n_islands)
    }


def hypercube_topology(n_islands: int) -> dict[int, list[int]]:
    """
    Hypercube topology for power-of-2 islands.
    
    Each island is connected to others differing by exactly
    one bit in their binary representation. For n islands,
    each has log2(n) neighbors.
    
    Good balance between connectivity and diversity preservation.
    
    Args:
        n_islands: Number of islands (must be power of 2, >= 2)
        
    Returns:
        Mapping from island ID to list of neighbor IDs
        
    Raises:
        ValueError: If n_islands is not a power of 2 or < 2
        
    Example:
        >>> hypercube_topology(4)
        {0: [1, 2], 1: [0, 3], 2: [3, 0], 3: [2, 1]}
        
        Binary representation:
        - 0 (00) connected to 1 (01), 2 (10)
        - 1 (01) connected to 0 (00), 3 (11)
        - 2 (10) connected to 3 (11), 0 (00)
        - 3 (11) connected to 2 (10), 1 (01)
    """
    if n_islands < 2:
        raise ValueError(
            f"Hypercube topology requires at least 2 islands, got {n_islands}"
        )
    
    if n_islands & (n_islands - 1) != 0:
        raise ValueError(
            f"Hypercube topology requires power of 2 islands, got {n_islands}"
        )
    
    n_bits = int(math.log2(n_islands))
    topology: dict[int, list[int]] = {}
    
    for i in range(n_islands):
        neighbors = []
        for bit in range(n_bits):
            # Flip each bit to get neighbor
            neighbor = i ^ (1 << bit)
            neighbors.append(neighbor)
        topology[i] = neighbors
    
    return topology


def ladder_topology(n_islands: int) -> dict[int, list[int]]:
    """
    Ladder/line topology: linear chain of islands.
    
    Creates a linear arrangement where island i is connected
    to islands i-1 and i+1 (without wrapping). End islands
    have only one neighbor.
    
    Good for gradient-based exploration.
    
    Args:
        n_islands: Number of islands (must be >= 2)
        
    Returns:
        Mapping from island ID to list of neighbor IDs
        
    Example:
        >>> ladder_topology(4)
        {0: [1], 1: [0, 2], 2: [1, 3], 3: [2]}
    """
    if n_islands < 2:
        raise ValueError(
            f"Ladder topology requires at least 2 islands, got {n_islands}"
        )
    
    topology: dict[int, list[int]] = {}
    
    for i in range(n_islands):
        neighbors = []
        if i > 0:
            neighbors.append(i - 1)
        if i < n_islands - 1:
            neighbors.append(i + 1)
        topology[i] = neighbors
    
    return topology


def star_topology(n_islands: int) -> dict[int, list[int]]:
    """
    Star topology: one central hub connected to all others.
    
    Island 0 is the hub, connected to all other islands.
    Peripheral islands are only connected to the hub.
    
    Good for centralized genetic mixing while maintaining
    peripheral diversity.
    
    Args:
        n_islands: Number of islands (must be >= 2)
        
    Returns:
        Mapping from island ID to list of neighbor IDs
        
    Example:
        >>> star_topology(4)
        {0: [1, 2, 3], 1: [0], 2: [0], 3: [0]}
    """
    if n_islands < 2:
        raise ValueError(
            f"Star topology requires at least 2 islands, got {n_islands}"
        )
    
    topology: dict[int, list[int]] = {}
    
    # Hub (island 0) connected to all others
    topology[0] = list(range(1, n_islands))
    
    # All others connected only to hub
    for i in range(1, n_islands):
        topology[i] = [0]
    
    return topology
