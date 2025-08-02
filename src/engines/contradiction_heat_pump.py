"""
CONTRADICTION HEAT PUMP ENGINE
==============================

Revolutionary thermodynamic engine that uses contradiction tensions as work input
to cool overheated semantic regions. This implements the thermodynamic principle
where contradictions can be harnessed to remove heat from cognitive conflicts.

Key Features:
- Contradiction-powered cooling cycles
- Thermal management of cognitive conflicts  
- Coefficient of Performance optimization
- Semantic temperature regulation
"""

import numpy as np
import torch
import time
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import math

logger = logging.getLogger(__name__)


@dataclass
class ContradictionField:
    """Represents a field with contradictory semantic states"""
    field_id: str
    semantic_vectors: List[np.ndarray]
    contradiction_tensor: np.ndarray
    initial_temperature: float
    target_temperature: float
    tension_magnitude: float
    coherence_score: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class HeatPumpCycle:
    """Represents a complete heat pump cycle"""
    cycle_id: str
    contradiction_work_input: float
    heat_removed: float
    heat_rejected: float
    coefficient_of_performance: float
    cooling_efficiency: float
    initial_temp: float
    final_temp: float
    cycle_duration: float
    energy_conservation_error: float
    timestamp: datetime = field(default_factory=datetime.now)


class ContradictionHeatPump:
    """
    Contradiction Heat Pump Engine
    
    Uses contradiction tensions as work input to cool overheated semantic regions.
    Based on reverse Carnot cycle principles with cognitive field adaptations.
    """
    
    def __init__(self, target_cop: float = 3.0, max_cooling_power: float = 100.0):
        """
        Initialize the Contradiction Heat Pump
        
        Args:
            target_cop: Target coefficient of performance (cooling/work ratio)
            max_cooling_power: Maximum cooling power per cycle
        """
        self.target_cop = target_cop
        self.max_cooling_power = max_cooling_power
        self.cycles_completed = 0
        self.total_heat_removed = 0.0
        self.total_work_consumed = 0.0
        self.cycle_history = []
        
        # Thermodynamic constants for cognitive fields
        self.cognitive_carnot_factor = 0.85  # Efficiency factor for cognitive systems
        self.min_temperature_ratio = 0.1    # Minimum temperature difference ratio
        self.contradiction_efficiency = 0.9  # How efficiently contradictions provide work
        
        logger.info(f"🔄 Contradiction Heat Pump initialized (target COP: {target_cop})")
    
    def analyze_contradiction_field(self, field: ContradictionField) -> Dict[str, float]:
        """
        Analyze a contradiction field to determine cooling potential
        
        Args:
            field: ContradictionField to analyze
            
        Returns:
            Analysis results including work potential and cooling capacity
        """
        # Calculate contradiction intensity from tensor
        contradiction_intensity = float(np.linalg.norm(field.contradiction_tensor))
        
        # Calculate semantic dispersion (temperature proxy)
        if len(field.semantic_vectors) > 1:
            vectors_matrix = np.array(field.semantic_vectors)
            semantic_variance = np.var(vectors_matrix, axis=0).mean()
        else:
            semantic_variance = 0.1
        
        # Calculate available work from contradiction tension
        available_work = (
            field.tension_magnitude * contradiction_intensity * 
            self.contradiction_efficiency
        )
        
        # Calculate theoretical cooling capacity
        temp_ratio = field.target_temperature / max(field.initial_temperature, 0.1)
        theoretical_cop = 1.0 / max(1.0 - temp_ratio, 0.1)
        practical_cop = min(theoretical_cop * self.cognitive_carnot_factor, self.target_cop)
        
        cooling_capacity = available_work * practical_cop
        
        # Calculate efficiency metrics
        temperature_differential = field.initial_temperature - field.target_temperature
        cooling_efficiency = min(1.0, cooling_capacity / max(temperature_differential, 0.1))
        
        return {
            'contradiction_intensity': contradiction_intensity,
            'semantic_variance': semantic_variance,
            'available_work': available_work,
            'theoretical_cop': theoretical_cop,
            'practical_cop': practical_cop,
            'cooling_capacity': cooling_capacity,
            'cooling_efficiency': cooling_efficiency,
            'temperature_differential': temperature_differential
        }
    
    def run_cooling_cycle(self, field: ContradictionField) -> HeatPumpCycle:
        """
        Run a complete cooling cycle on a contradiction field
        
        Args:
            field: ContradictionField to cool
            
        Returns:
            HeatPumpCycle results
        """
        start_time = time.time()
        cycle_id = str(uuid.uuid4())
        
        # Analyze field for cooling potential
        analysis = self.analyze_contradiction_field(field)
        
        # Calculate work input from contradiction tension
        work_input = min(
            analysis['available_work'],
            self.max_cooling_power / max(analysis['practical_cop'], 1.0)
        )
        
        # Calculate heat removal based on COP
        heat_removed = work_input * analysis['practical_cop']
        heat_removed = min(heat_removed, self.max_cooling_power)
        
        # Apply cooling to the field
        temperature_reduction = heat_removed / max(
            len(field.semantic_vectors) * field.coherence_score, 1.0
        )
        
        final_temperature = max(
            field.initial_temperature - temperature_reduction,
            field.target_temperature
        )
        
        # Calculate heat rejection (conservation of energy)
        heat_rejected = work_input + heat_removed
        
        # Calculate actual COP achieved
        actual_cop = heat_removed / max(work_input, 0.001)
        
        # Calculate energy conservation error
        energy_in = work_input
        energy_out = heat_removed + heat_rejected
        conservation_error = abs(energy_in - energy_out) / max(energy_in, 0.001)
        
        # Update field temperature
        field.initial_temperature = final_temperature
        
        # Create cycle record
        cycle = HeatPumpCycle(
            cycle_id=cycle_id,
            contradiction_work_input=work_input,
            heat_removed=heat_removed,
            heat_rejected=heat_rejected,
            coefficient_of_performance=actual_cop,
            cooling_efficiency=analysis['cooling_efficiency'],
            initial_temp=field.initial_temperature + temperature_reduction,
            final_temp=final_temperature,
            cycle_duration=time.time() - start_time,
            energy_conservation_error=conservation_error
        )
        
        # Update engine statistics
        self.cycles_completed += 1
        self.total_heat_removed += heat_removed
        self.total_work_consumed += work_input
        self.cycle_history.append(cycle)
        
        logger.info(f"🔄 Cooling cycle complete: T={final_temperature:.3f}K, COP={actual_cop:.2f}")
        
        return cycle
    
    def optimize_cooling_strategy(self, fields: List[ContradictionField]) -> Dict[str, Any]:
        """
        Optimize cooling strategy for multiple contradiction fields
        
        Args:
            fields: List of ContradictionField objects to optimize
            
        Returns:
            Optimization results and strategy
        """
        if not fields:
            return {'error': 'No fields provided for optimization'}
        
        # Analyze all fields
        field_analyses = []
        for field in fields:
            analysis = self.analyze_contradiction_field(field)
            analysis['field_id'] = field.field_id
            analysis['priority_score'] = (
                field.initial_temperature * analysis['contradiction_intensity'] *
                (1.0 / max(analysis['practical_cop'], 1.0))
            )
            field_analyses.append(analysis)
        
        # Sort by priority (highest temperature and contradiction intensity first)
        field_analyses.sort(key=lambda x: x['priority_score'], reverse=True)
        
        # Calculate total work available
        total_work_available = sum(a['available_work'] for a in field_analyses)
        
        # Allocate work based on priority and efficiency
        cooling_strategy = []
        remaining_work = total_work_available
        
        for analysis in field_analyses:
            if remaining_work <= 0:
                break
            
            # Allocate work based on efficiency and priority
            work_allocation = min(
                analysis['available_work'],
                remaining_work * (analysis['priority_score'] / 
                                sum(a['priority_score'] for a in field_analyses))
            )
            
            expected_cooling = work_allocation * analysis['practical_cop']
            
            cooling_strategy.append({
                'field_id': analysis['field_id'],
                'work_allocation': work_allocation,
                'expected_cooling': expected_cooling,
                'expected_cop': analysis['practical_cop'],
                'priority_score': analysis['priority_score']
            })
            
            remaining_work -= work_allocation
        
        # Calculate overall strategy metrics
        total_expected_cooling = sum(s['expected_cooling'] for s in cooling_strategy)
        total_work_allocated = sum(s['work_allocation'] for s in cooling_strategy)
        overall_cop = total_expected_cooling / max(total_work_allocated, 0.001)
        
        return {
            'cooling_strategy': cooling_strategy,
            'total_work_available': total_work_available,
            'total_work_allocated': total_work_allocated,
            'total_expected_cooling': total_expected_cooling,
            'overall_cop': overall_cop,
            'efficiency_rating': min(overall_cop / self.target_cop, 1.0),
            'fields_processed': len(cooling_strategy)
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics for the heat pump"""
        if self.cycles_completed == 0:
            return {'error': 'No cycles completed yet'}
        
        average_cop = self.total_heat_removed / max(self.total_work_consumed, 0.001)
        
        recent_cycles = self.cycle_history[-10:] if len(self.cycle_history) >= 10 else self.cycle_history
        recent_cop = np.mean([c.coefficient_of_performance for c in recent_cycles])
        recent_efficiency = np.mean([c.cooling_efficiency for c in recent_cycles])
        
        return {
            'cycles_completed': self.cycles_completed,
            'total_heat_removed': self.total_heat_removed,
            'total_work_consumed': self.total_work_consumed,
            'average_cop': average_cop,
            'recent_cop': recent_cop,
            'recent_efficiency': recent_efficiency,
            'target_cop': self.target_cop,
            'cop_achievement_ratio': average_cop / self.target_cop,
            'performance_rating': min(average_cop / self.target_cop, 1.0)
        }
    
    async def shutdown(self):
        """Shutdown the heat pump gracefully"""
        try:
            logger.info("🔄 Contradiction Heat Pump shutting down...")
            
            # Clear cycle history
            self.cycle_history.clear()
            
            # Reset counters
            self.cycles_completed = 0
            self.total_heat_removed = 0.0
            self.total_work_consumed = 0.0
            
            logger.info("✅ Contradiction Heat Pump shutdown complete")
            
        except Exception as e:
            logger.error(f"❌ Error during heat pump shutdown: {e}") 