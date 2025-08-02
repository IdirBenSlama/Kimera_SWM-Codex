"""
SPDE Core - Semantic Pressure Diffusion Engine Core
==================================================

The mathematical physics foundation of Kimera's cognitive processing.
This module integrates both simple semantic pressure diffusion and 
advanced stochastic partial differential equation systems.

SPDE Core provides:
- Simple Semantic Pressure Diffusion (basic cognitive field evolution)
- Advanced SPDE systems (stochastic cognitive dynamics modeling)
- Unified interface for cognitive field processing
- Integration with KCCL and other foundational systems

This is the mathematical substrate that enables Kimera's physics-based
approach to cognitive processing and field dynamics.
"""

import asyncio
import numpy as np
import torch
import torch.nn.functional as F
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from enum import Enum
from collections import deque, defaultdict

# Core mathematical dependencies
from ..native_math import NativeMath
from ...utils.config import get_api_settings
from ...config.settings import get_settings
from ...utils.kimera_logger import get_cognitive_logger

logger = get_cognitive_logger(__name__)


class SPDEType(Enum):
    """Types of stochastic PDEs supported"""
    SIMPLE_DIFFUSION = "simple_diffusion"        # Basic semantic pressure diffusion
    COGNITIVE_DIFFUSION = "cognitive_diffusion"  # Advanced cognitive field diffusion
    SEMANTIC_WAVE = "semantic_wave"              # Wave-based semantic propagation
    REACTION_DIFFUSION = "reaction_diffusion"    # Reaction-diffusion systems
    BROWNIAN_FIELD = "brownian_field"           # Brownian motion fields
    NEURAL_FIELD = "neural_field"               # Neural field dynamics


class BoundaryCondition(Enum):
    """Boundary condition types for SPDE solving"""
    PERIODIC = "periodic"
    NEUMANN = "neumann"
    DIRICHLET = "dirichlet"
    ABSORBING = "absorbing"


class DiffusionMode(Enum):
    """Diffusion processing modes"""
    SIMPLE = "simple"              # Basic Gaussian diffusion
    ADVANCED = "advanced"          # Full SPDE solving
    ADAPTIVE = "adaptive"          # Adaptive mode selection
    HYBRID = "hybrid"             # Combination of methods


@dataclass
class SPDEConfig:
    """Configuration for SPDE processing"""
    spde_type: SPDEType = SPDEType.COGNITIVE_DIFFUSION
    diffusion_rate: float = 0.5
    decay_factor: float = 1.0
    spatial_dims: int = 2
    temporal_steps: int = 100
    dt: float = 0.01
    dx: float = 0.1
    boundary_condition: BoundaryCondition = BoundaryCondition.PERIODIC
    noise_amplitude: float = 0.1
    device: str = "cpu"
    
    # Advanced parameters
    reaction_strength: float = 1.0
    diffusion_coefficient: float = 1.0
    nonlinearity_strength: float = 0.1
    
    # Performance parameters
    batch_size: int = 32
    use_gpu: bool = False
    mixed_precision: bool = False


@dataclass
class SPDESolution:
    """Solution result from SPDE processing"""
    solution_id: str
    initial_field: torch.Tensor
    final_field: torch.Tensor
    field_evolution: List[torch.Tensor]
    solving_time: float
    convergence_achieved: bool
    error_estimate: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DiffusionResult:
    """Result from semantic diffusion processing"""
    original_state: Dict[str, float]
    diffused_state: Dict[str, float]
    diffusion_delta: Dict[str, float]
    processing_time: float
    method_used: DiffusionMode
    entropy_change: float = 0.0


class SemanticDiffusionEngine:
    """
    Simple Semantic Pressure Diffusion Engine
    
    Implements basic Gaussian diffusion for semantic pressure
    evolution with configurable parameters.
    """
    
    def __init__(self, 
                 diffusion_rate: float = 0.5, 
                 decay_factor: float = 1.0,
                 sigma_scaling: float = 1.0):
        """
        Initialize Simple Semantic Diffusion Engine
        
        Args:
            diffusion_rate: Rate of diffusion blend (0-1)
            decay_factor: Gaussian filter decay factor
            sigma_scaling: Scaling factor for Gaussian sigma
        """
        self.diffusion_rate = diffusion_rate
        self.decay_factor = decay_factor
        self.sigma_scaling = sigma_scaling
        
        # Performance tracking
        self.diffusions_performed = 0
        self.total_processing_time = 0.0
        self.average_processing_time = 0.0
        
        logger.debug(f"Simple SPDE initialized: rate={diffusion_rate}, decay={decay_factor}")
    
    def diffuse(self, state: Dict[str, float]) -> Dict[str, float]:
        """
        Apply simple semantic pressure diffusion to state
        
        Args:
            state: Dictionary of semantic state values
            
        Returns:
            Diffused semantic state
        """
        start_time = time.time()
        
        try:
            if not state:
                return {}
            
            keys = list(state.keys())
            values = [state[k] for k in keys]
            
            # Apply Gaussian blur using native implementation
            sigma = self.decay_factor * self.sigma_scaling
            blurred = NativeMath.gaussian_filter_1d(values, sigma=sigma)
            
            # Apply diffusion blend
            diffused = [
                (1 - self.diffusion_rate) * v + self.diffusion_rate * b 
                for v, b in zip(values, blurred)
            ]
            
            result = dict(zip(keys, diffused))
            
            # Update performance metrics
            processing_time = time.time() - start_time
            self.diffusions_performed += 1
            self.total_processing_time += processing_time
            self.average_processing_time = self.total_processing_time / self.diffusions_performed
            
            return result
            
        except Exception as e:
            logger.error(f"Simple diffusion failed: {e}")
            return state  # Return original state on error
    
    async def diffuse_async(self, state: Dict[str, float]) -> DiffusionResult:
        """Async version of diffuse with detailed result"""
        start_time = time.time()
        
        original_state = state.copy()
        diffused_state = self.diffuse(state)
        
        # Calculate diffusion delta
        diffusion_delta = {
            key: diffused_state.get(key, 0.0) - original_state.get(key, 0.0)
            for key in set(original_state.keys()) | set(diffused_state.keys())
        }
        
        processing_time = time.time() - start_time
        
        return DiffusionResult(
            original_state=original_state,
            diffused_state=diffused_state,
            diffusion_delta=diffusion_delta,
            processing_time=processing_time,
            method_used=DiffusionMode.SIMPLE
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get diffusion performance metrics"""
        return {
            'diffusions_performed': self.diffusions_performed,
            'total_processing_time': self.total_processing_time,
            'average_processing_time': self.average_processing_time,
            'diffusion_rate': self.diffusion_rate,
            'decay_factor': self.decay_factor
        }


class AdvancedSPDEEngine:
    """
    Advanced Stochastic Partial Differential Equation Engine
    
    Implements sophisticated SPDE solving for cognitive dynamics
    with multiple solver types and advanced mathematical methods.
    """
    
    def __init__(self, device: str = "cpu"):
        """
        Initialize Advanced SPDE Engine
        
        Args:
            device: Computing device (cpu/cuda)
        """
        self.settings = get_api_settings()
        self.device = torch.device(device)
        
        # Solver registry
        self.solvers = {}
        self.solution_history = deque(maxlen=1000)
        
        # Performance metrics
        self.metrics = {
            "total_solutions": 0,
            "average_solve_time": 0.0,
            "last_solve_time": 0.0,
            "convergence_rate": 0.0,
            "error_rate": 0.0
        }
        
        # Configuration
        self.default_config = SPDEConfig(device=device)
        
        logger.info(f"Advanced SPDE Engine initialized on device: {device}")
    
    async def solve_cognitive_diffusion(self, 
                                       initial_field: torch.Tensor,
                                       config: Optional[SPDEConfig] = None) -> SPDESolution:
        """
        Solve cognitive diffusion equation for field evolution
        
        Args:
            initial_field: Initial cognitive field state
            config: SPDE configuration (uses default if None)
            
        Returns:
            Complete solution with field evolution
        """
        if config is None:
            config = self.default_config
        
        start_time = time.time()
        solution_id = f"SPDE_{int(start_time)}_{self.metrics['total_solutions']}"
        
        try:
            # Create solver based on configuration
            solver = self._create_solver(config)
            
            # Define cognitive reaction function
            def cognitive_reaction(phi):
                # Nonlinear cognitive dynamics: activation with saturation
                return torch.tanh(phi) - config.nonlinearity_strength * phi**3
            
            # Solve the SPDE
            field_evolution = []
            current_field = initial_field.clone()
            
            for step in range(config.temporal_steps):
                # Apply diffusion step
                next_field = await self._diffusion_step(
                    current_field, config, cognitive_reaction
                )
                field_evolution.append(next_field.clone())
                current_field = next_field
                
                # Check convergence
                if step > 10 and self._check_convergence(field_evolution[-10:]):
                    logger.debug(f"Convergence achieved at step {step}")
                    break
            
            # Calculate error estimate
            error_estimate = self._estimate_error(initial_field, current_field, field_evolution)
            
            solution = SPDESolution(
                solution_id=solution_id,
                initial_field=initial_field,
                final_field=current_field,
                field_evolution=field_evolution,
                solving_time=time.time() - start_time,
                convergence_achieved=len(field_evolution) < config.temporal_steps,
                error_estimate=error_estimate,
                metadata={
                    'config': config,
                    'steps_taken': len(field_evolution),
                    'solver_type': config.spde_type.value
                }
            )
            
            # Update metrics
            self._update_solution_metrics(solution)
            self.solution_history.append(solution)
            
            return solution
            
        except Exception as e:
            logger.error(f"SPDE solving failed: {e}")
            # Return minimal solution on error
            return SPDESolution(
                solution_id=solution_id,
                initial_field=initial_field,
                final_field=initial_field,
                field_evolution=[initial_field],
                solving_time=time.time() - start_time,
                convergence_achieved=False,
                error_estimate=float('inf'),
                metadata={'error': str(e)}
            )
    
    async def solve_semantic_wave(self,
                                 initial_field: torch.Tensor,
                                 config: Optional[SPDEConfig] = None) -> SPDESolution:
        """Solve semantic wave equation for wave-based propagation"""
        if config is None:
            config = SPDEConfig(spde_type=SPDEType.SEMANTIC_WAVE, device=self.device)
        
        # Wave equation: âˆ‚Â²Ï†/âˆ‚tÂ² = cÂ²âˆ‡Â²Ï† + source
        return await self._solve_wave_equation(initial_field, config)
    
    async def solve_reaction_diffusion(self,
                                      initial_field: torch.Tensor,
                                      config: Optional[SPDEConfig] = None) -> SPDESolution:
        """Solve reaction-diffusion system for pattern formation"""
        if config is None:
            config = SPDEConfig(spde_type=SPDEType.REACTION_DIFFUSION, device=self.device)
        
        # Reaction-diffusion: âˆ‚Ï†/âˆ‚t = Dâˆ‡Â²Ï† + f(Ï†)
        return await self._solve_reaction_diffusion_equation(initial_field, config)
    
    def _create_solver(self, config: SPDEConfig):
        """Create appropriate solver based on configuration"""
        solver_key = config.spde_type.value
        
        if solver_key not in self.solvers:
            if config.spde_type == SPDEType.COGNITIVE_DIFFUSION:
                self.solvers[solver_key] = CognitiveDiffusionSolver(config)
            elif config.spde_type == SPDEType.SEMANTIC_WAVE:
                self.solvers[solver_key] = SemanticWaveSolver(config)
            elif config.spde_type == SPDEType.REACTION_DIFFUSION:
                self.solvers[solver_key] = ReactionDiffusionSolver(config)
            else:
                raise ValueError(f"Unsupported SPDE type: {config.spde_type}")
        
        return self.solvers[solver_key]
    
    async def _diffusion_step(self, 
                             field: torch.Tensor, 
                             config: SPDEConfig,
                             reaction_function: Callable) -> torch.Tensor:
        """Execute single diffusion step"""
        # Laplacian operator for diffusion
        laplacian = self._compute_laplacian(field, config.dx)
        
        # Reaction term
        reaction = reaction_function(field)
        
        # Noise term for stochastic component
        noise = torch.randn_like(field) * config.noise_amplitude
        
        # Update equation: Ï†_{n+1} = Ï†_n + dt * (Dâˆ‡Â²Ï† + f(Ï†) + Î·)
        next_field = field + config.dt * (
            config.diffusion_coefficient * laplacian + 
            config.reaction_strength * reaction + 
            noise
        )
        
        # Apply boundary conditions
        next_field = self._apply_boundary_conditions(next_field, config.boundary_condition)
        
        return next_field
    
    def _compute_laplacian(self, field: torch.Tensor, dx: float) -> torch.Tensor:
        """Compute discrete Laplacian operator"""
        # 2D Laplacian using finite differences
        if len(field.shape) == 2:
            laplacian = (
                torch.roll(field, 1, dims=0) + torch.roll(field, -1, dims=0) +
                torch.roll(field, 1, dims=1) + torch.roll(field, -1, dims=1) -
                4 * field
            ) / (dx * dx)
        else:
            # 1D Laplacian
            laplacian = (
                torch.roll(field, 1, dims=-1) + torch.roll(field, -1, dims=-1) - 2 * field
            ) / (dx * dx)
        
        return laplacian
    
    def _apply_boundary_conditions(self, 
                                  field: torch.Tensor, 
                                  boundary_type: BoundaryCondition) -> torch.Tensor:
        """Apply boundary conditions to field"""
        if boundary_type == BoundaryCondition.PERIODIC:
            # Periodic boundaries are handled by torch.roll
            return field
        elif boundary_type == BoundaryCondition.NEUMANN:
            # Zero gradient at boundaries
            if len(field.shape) == 2:
                field[0, :] = field[1, :]
                field[-1, :] = field[-2, :]
                field[:, 0] = field[:, 1]
                field[:, -1] = field[:, -2]
        elif boundary_type == BoundaryCondition.DIRICHLET:
            # Zero value at boundaries
            if len(field.shape) == 2:
                field[0, :] = 0
                field[-1, :] = 0
                field[:, 0] = 0
                field[:, -1] = 0
        
        return field
    
    def _check_convergence(self, recent_fields: List[torch.Tensor]) -> bool:
        """Check if solution has converged"""
        if len(recent_fields) < 2:
            return False
        
        # Check relative change between recent fields
        change = torch.norm(recent_fields[-1] - recent_fields[-2]) / torch.norm(recent_fields[-1])
        return change < 1e-6
    
    def _estimate_error(self, 
                       initial: torch.Tensor,
                       final: torch.Tensor,
                       evolution: List[torch.Tensor]) -> float:
        """Estimate solution error"""
        # Simple error estimate based on energy conservation
        initial_energy = torch.sum(initial**2).item()
        final_energy = torch.sum(final**2).item()
        
        # Normalized energy change
        if initial_energy > 0:
            energy_change = abs(final_energy - initial_energy) / initial_energy
        else:
            energy_change = final_energy
        
        return energy_change
    
    async def _solve_wave_equation(self, 
                                  initial_field: torch.Tensor,
                                  config: SPDEConfig) -> SPDESolution:
        """Solve wave equation implementation"""
        # Placeholder for wave equation solver
        # Would implement second-order time derivative
        return await self.solve_cognitive_diffusion(initial_field, config)
    
    async def _solve_reaction_diffusion_equation(self,
                                               initial_field: torch.Tensor,
                                               config: SPDEConfig) -> SPDESolution:
        """Solve reaction-diffusion equation implementation"""
        # Placeholder for reaction-diffusion solver
        # Would implement specific reaction functions
        return await self.solve_cognitive_diffusion(initial_field, config)
    
    def _update_solution_metrics(self, solution: SPDESolution):
        """Update performance metrics based on solution"""
        self.metrics["total_solutions"] += 1
        self.metrics["last_solve_time"] = solution.solving_time
        
        # Update average solve time
        total = self.metrics["total_solutions"]
        self.metrics["average_solve_time"] = (
            (self.metrics["average_solve_time"] * (total - 1) + solution.solving_time) / total
        )
        
        # Update convergence rate
        if solution.convergence_achieved:
            current_rate = self.metrics["convergence_rate"]
            self.metrics["convergence_rate"] = (current_rate * (total - 1) + 1.0) / total
        
        # Update error rate
        if solution.error_estimate != float('inf'):
            error_occurred = 0.0
        else:
            error_occurred = 1.0
        
        current_error_rate = self.metrics["error_rate"]
        self.metrics["error_rate"] = (current_error_rate * (total - 1) + error_occurred) / total


# Placeholder solver classes (to be implemented as needed)
class CognitiveDiffusionSolver:
    def __init__(self, config: SPDEConfig):
        self.config = config

class SemanticWaveSolver:
    def __init__(self, config: SPDEConfig):
        self.config = config

class ReactionDiffusionSolver:
    def __init__(self, config: SPDEConfig):
        self.config = config


class SPDECore:
    """
    Unified SPDE Core System
    
    Integrates both simple semantic diffusion and advanced SPDE
    capabilities with intelligent mode selection and optimization.
    """
    
    def __init__(self, 
                 default_mode: DiffusionMode = DiffusionMode.ADAPTIVE,
                 device: str = "cpu"):
        """
        Initialize SPDE Core
        
        Args:
            default_mode: Default diffusion processing mode
            device: Computing device for advanced processing
        """
        self.settings = get_api_settings()
        self.default_mode = default_mode
        self.device = device
        
        # Initialize engines
        self.simple_engine = SemanticDiffusionEngine()
        self.advanced_engine = AdvancedSPDEEngine(device=device)
        
        # Performance tracking
        self.total_operations = 0
        self.mode_usage_stats = defaultdict(int)
        self.performance_history = deque(maxlen=1000)
        
        # Adaptive thresholds
        self.complexity_threshold = 1000  # Switch to advanced for large states
        self.performance_threshold = 0.1  # Switch to simple for fast processing
        
        # Integration callbacks (for KCCL coordination)
        self.cycle_callbacks = []
        
        logger.info(f"ðŸŒŠ SPDE Core initialized")
        logger.info(f"   Default mode: {default_mode.value}")
        logger.info(f"   Device: {device}")
        logger.info(f"   Adaptive thresholds: complexity={self.complexity_threshold}, "
                   f"performance={self.performance_threshold}")
    
    async def process_semantic_diffusion(self, 
                                        state: Union[Dict[str, float], torch.Tensor],
                                        mode: Optional[DiffusionMode] = None,
                                        config: Optional[SPDEConfig] = None) -> DiffusionResult:
        """
        Process semantic diffusion with intelligent mode selection
        
        Args:
            state: Semantic state (dict or tensor)
            mode: Processing mode (uses default if None)
            config: SPDE configuration for advanced processing
            
        Returns:
            Diffusion result with metadata
        """
        start_time = time.time()
        
        # Determine processing mode
        if mode is None:
            mode = self._select_processing_mode(state)
        
        try:
            if mode == DiffusionMode.SIMPLE:
                if isinstance(state, torch.Tensor):
                    # Convert tensor to dict for simple processing
                    state_dict = self._tensor_to_dict(state)
                    result = await self.simple_engine.diffuse_async(state_dict)
                    # Convert back to tensor if needed
                    result.diffused_state = self._dict_to_tensor(result.diffused_state, state.shape)
                else:
                    result = await self.simple_engine.diffuse_async(state)
            
            elif mode == DiffusionMode.ADVANCED:
                if isinstance(state, dict):
                    # Convert dict to tensor for advanced processing
                    tensor_state = self._dict_to_tensor(state)
                else:
                    tensor_state = state
                
                if config is None:
                    config = SPDEConfig(device=self.device)
                
                solution = await self.advanced_engine.solve_cognitive_diffusion(tensor_state, config)
                
                # Convert solution to DiffusionResult format
                result = DiffusionResult(
                    original_state=state,
                    diffused_state=solution.final_field if isinstance(state, torch.Tensor) else self._tensor_to_dict(solution.final_field),
                    diffusion_delta=self._calculate_diffusion_delta(state, solution.final_field),
                    processing_time=solution.solving_time,
                    method_used=DiffusionMode.ADVANCED,
                    entropy_change=self._calculate_entropy_change(solution.initial_field, solution.final_field)
                )
            
            elif mode == DiffusionMode.ADAPTIVE:
                # Try simple first, fall back to advanced if needed
                simple_result = await self.process_semantic_diffusion(state, DiffusionMode.SIMPLE)
                
                # Check if simple result is adequate
                if self._is_result_adequate(simple_result):
                    result = simple_result
                    result.method_used = DiffusionMode.ADAPTIVE
                else:
                    result = await self.process_semantic_diffusion(state, DiffusionMode.ADVANCED, config)
                    result.method_used = DiffusionMode.ADAPTIVE
            
            elif mode == DiffusionMode.HYBRID:
                # Combine simple and advanced results
                simple_result = await self.process_semantic_diffusion(state, DiffusionMode.SIMPLE)
                advanced_result = await self.process_semantic_diffusion(state, DiffusionMode.ADVANCED, config)
                
                # Blend results (weighted average)
                result = self._blend_results(simple_result, advanced_result, blend_weight=0.7)
                result.method_used = DiffusionMode.HYBRID
            
            else:
                raise ValueError(f"Unsupported diffusion mode: {mode}")
            
            # Update statistics
            self.total_operations += 1
            self.mode_usage_stats[mode] += 1
            self.performance_history.append({
                'processing_time': result.processing_time,
                'mode': mode,
                'entropy_change': result.entropy_change,
                'timestamp': datetime.now()
            })
            
            # Trigger callbacks for KCCL integration
            await self._trigger_diffusion_callbacks({
                'state': state,
                'result': result,
                'mode': mode,
                'operation_count': self.total_operations
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Semantic diffusion failed: {e}")
            # Return identity result on error
            return DiffusionResult(
                original_state=state,
                diffused_state=state,
                diffusion_delta={},
                processing_time=time.time() - start_time,
                method_used=mode,
                entropy_change=0.0
            )
    
    def diffuse(self, state: Dict[str, float]) -> Dict[str, float]:
        """
        Synchronous diffusion interface for compatibility
        
        Args:
            state: Semantic state dictionary
            
        Returns:
            Diffused state dictionary
        """
        return self.simple_engine.diffuse(state)
    
    def register_cycle_callback(self, callback: Callable):
        """Register callback for KCCL integration"""
        self.cycle_callbacks.append(callback)
        logger.debug("Registered SPDE cycle callback")
    
    def _select_processing_mode(self, state: Union[Dict[str, float], torch.Tensor]) -> DiffusionMode:
        """Intelligently select processing mode based on state complexity"""
        if self.default_mode != DiffusionMode.ADAPTIVE:
            return self.default_mode
        
        # Estimate complexity
        if isinstance(state, dict):
            complexity = len(state)
        else:
            complexity = state.numel()
        
        # Check recent performance
        if self.performance_history:
            recent_avg_time = np.mean([
                h['processing_time'] for h in list(self.performance_history)[-10:]
            ])
        else:
            recent_avg_time = 0.0
        
        # Mode selection logic
        if complexity > self.complexity_threshold:
            return DiffusionMode.ADVANCED
        elif recent_avg_time > self.performance_threshold:
            return DiffusionMode.SIMPLE
        else:
            return DiffusionMode.SIMPLE  # Default to simple for efficiency
    
    def _tensor_to_dict(self, tensor: torch.Tensor) -> Dict[str, float]:
        """Convert tensor to dictionary for simple processing"""
        flattened = tensor.flatten()
        return {f"elem_{i}": float(flattened[i]) for i in range(len(flattened))}
    
    def _dict_to_tensor(self, state_dict: Dict[str, float], target_shape: Optional[torch.Size] = None) -> torch.Tensor:
        """Convert dictionary to tensor for advanced processing"""
        try:
            values = list(state_dict.values())
            tensor = torch.tensor(values, dtype=torch.float32)
            
            if target_shape is not None:
                try:
                    tensor = tensor.reshape(target_shape)
                except RuntimeError:
                    # If reshape fails, pad or truncate to match target shape
                    target_size = target_shape.numel()
                    current_size = tensor.numel()
                    
                    if current_size < target_size:
                        # Pad with zeros
                        padding = torch.zeros(target_size - current_size)
                        tensor = torch.cat([tensor, padding])
                    elif current_size > target_size:
                        # Truncate
                        tensor = tensor[:target_size]
                    
                    tensor = tensor.reshape(target_shape)
            
            return tensor
        except Exception as e:
            # Fallback: return a default tensor
            if target_shape is not None:
                return torch.zeros(target_shape)
            else:
                return torch.zeros(len(state_dict) if state_dict else 1)
    
    def _calculate_diffusion_delta(self, 
                                  original: Union[Dict[str, float], torch.Tensor],
                                  final: Union[Dict[str, float], torch.Tensor]) -> Dict[str, float]:
        """Calculate diffusion delta between original and final states"""
        if isinstance(original, dict) and isinstance(final, dict):
            return {
                key: final.get(key, 0.0) - original.get(key, 0.0)
                for key in set(original.keys()) | set(final.keys())
            }
        elif isinstance(original, torch.Tensor) and isinstance(final, torch.Tensor):
            delta_tensor = final - original
            return self._tensor_to_dict(delta_tensor)
        else:
            return {}
    
    def _calculate_entropy_change(self, initial: torch.Tensor, final: torch.Tensor) -> float:
        """Calculate entropy change between initial and final fields"""
        try:
            # Simple entropy calculation based on variance
            initial_entropy = torch.var(initial).item()
            final_entropy = torch.var(final).item()
            return final_entropy - initial_entropy
        except Exception as e:
            logger.error(f"Error in spde_core.py: {e}", exc_info=True)
            raise  # Re-raise for proper error handling
            return 0.0
    
    def _is_result_adequate(self, result: DiffusionResult) -> bool:
        """Check if simple processing result is adequate"""
        try:
            # Check processing time and entropy change
            return (result.processing_time < self.performance_threshold and
                    abs(result.entropy_change) < 10.0)  # Reasonable entropy change
        except (AttributeError, TypeError):
            return True  # Default to adequate if can't evaluate
    
    def _blend_results(self, 
                      simple_result: DiffusionResult,
                      advanced_result: DiffusionResult,
                      blend_weight: float = 0.7) -> DiffusionResult:
        """Blend simple and advanced results"""
        # For now, return the advanced result with combined processing time
        blended = advanced_result
        blended.processing_time = (
            simple_result.processing_time + advanced_result.processing_time
        )
        return blended
    
    async def _trigger_diffusion_callbacks(self, event_data: Dict[str, Any]):
        """Trigger registered callbacks for diffusion events"""
        for callback in self.cycle_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event_data)
                else:
                    callback(event_data)
            except Exception as e:
                logger.warning(f"SPDE callback failed: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            'total_operations': self.total_operations,
            'mode_usage_stats': dict(self.mode_usage_stats),
            'default_mode': self.default_mode.value,
            'device': self.device,
            'simple_engine_metrics': self.simple_engine.get_metrics(),
            'advanced_engine_metrics': self.advanced_engine.metrics,
            'adaptive_thresholds': {
                'complexity_threshold': self.complexity_threshold,
                'performance_threshold': self.performance_threshold
            },
            'recent_performance': {
                'average_processing_time': np.mean([
                    h['processing_time'] for h in list(self.performance_history)[-100:]
                ]) if self.performance_history else 0.0,
                'operations_last_100': len(self.performance_history)
            }
        }