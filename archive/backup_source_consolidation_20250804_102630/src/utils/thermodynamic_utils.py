#!/usr/bin/env python3
"""
Canonical Thermodynamic Utilities for Kimera SWM
===============================================

This module provides a centralized, single source of truth for all
thermodynamic calculations across the Kimera codebase. It is founded on the
validated, physics-compliant principles established in the corrected
Revolutionary Thermodynamic Engine.

By centralizing these functions, we eliminate the architectural flaw of
"fractured thermodynamic authority" and ensure that all components operate
under one consistent, scientifically rigorous framework.
"""

import numpy as np
import torch
from typing import List, Any

# --- Canonical Physics Constants ---
# These constants should be the single source of truth.
PHYSICS_CONSTANTS = {
    'normalized_kb': 1.0,         # Normalized Boltzmann constant for semantic systems
    'min_temperature': 0.1,       # Minimum temperature to avoid division by zero
    'carnot_tolerance': 0.01,     # Safety margin for Carnot efficiency (e.g., 99% of theoretical max)
    'max_efficiency': 0.99,       # Absolute maximum efficiency cap
}


def calculate_physical_temperature(energies: List[float]) -> float:
    """
    Physics-compliant temperature using statistical mechanics (equipartition theorem).
    
    This is the CANONICAL temperature calculation for the entire system.
    T = 2/3 * <E> / k_B, where <E> is mean kinetic energy.
    """
    if not energies:
        return PHYSICS_CONSTANTS['min_temperature']
    
    mean_energy = np.mean(energies)
    
    # T = 2 * <E> / (3 * k_B) for a 3D system.
    temperature = (2.0 * mean_energy) / (3.0 * PHYSICS_CONSTANTS['normalized_kb'])
    
    return max(temperature, PHYSICS_CONSTANTS['min_temperature'])


def calculate_total_energy(fields: List[Any]) -> float:
    """
    Canonical calculation for the total energy of a collection of fields.
    Handles various field types (tensors, geoids, raw numbers).
    """
    total_energy = 0.0
    for field in fields:
        if hasattr(field, 'embedding'):
            if hasattr(field.embedding, 'cpu'):
                energy = torch.norm(field.embedding).cpu().item()
            else:
                energy = np.linalg.norm(field.embedding)
        elif hasattr(field, 'semantic_state'):
            energy = sum(field.semantic_state.values()) if field.semantic_state else 0.0
        else:
            energy = float(field) if isinstance(field, (int, float)) else 1.0
        
        total_energy += energy
    return total_energy


def calculate_theoretical_carnot_efficiency(T_hot: float, T_cold: float) -> float:
    """
    Calculates the theoretical maximum Carnot efficiency.
    η_carnot = 1 - (T_cold / T_hot)
    """
    if T_hot <= T_cold or T_hot <= 0:
        return 0.0
    
    return 1.0 - (T_cold / T_hot) 