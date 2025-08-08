"""
Vortex Dynamics Engine
======================
Models vortex structures in cognitive fields.

This engine implements vortex dynamics that represent:
- Rotational patterns in semantic space
- Cognitive attractors and repellers
- Information circulation and mixing
- Emergent spiral structures in thought patterns
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.ndimage as ndimage
import torch
from scipy.integrate import odeint

logger = logging.getLogger(__name__)

class VortexType(Enum):
    """Types of vortex structures"""
    POINT = "point"              # Point vortex
    RANKINE = "rankine"          # Rankine vortex (solid body core)
    LAMB_OSEEN = "lamb_oseen"    # Viscous vortex
    BURGERS = "burgers"          # Burgers vortex
    COGNITIVE = "cognitive"       # Cognitive attractor vortex

@dataclass
class Vortex:
    """Auto-generated class."""
    pass
    """Represents a vortex in cognitive field"""
    position: np.ndarray      # Position in field
    circulation: float        # Vortex strength (Γ)
    radius: float            # Core radius
    vortex_type: VortexType
    velocity: np.ndarray     # Vortex movement velocity
    age: float = 0.0         # Vortex age
    stability: float = 1.0   # Vortex stability

    def update_position(self, dt: float):
        """Update vortex position"""
        self.position += self.velocity * dt
        self.age += dt

@dataclass
class VortexField:
    """Auto-generated class."""
    pass
    """Vortex field state"""
    velocity_field: np.ndarray    # Velocity field (u, v)
    vorticity_field: np.ndarray   # Vorticity ω = ∇ × v
    stream_function: np.ndarray   # Stream function ψ
    energy: float                 # Kinetic energy
    enstrophy: float             # Enstrophy (vorticity squared)
class VortexDynamicsEngine:
    """Auto-generated class."""
    pass
    """
    Engine for modeling vortex dynamics in cognitive fields
    """

    def __init__(self, grid_size: int = 128, domain_size: float = 10.0):
        try:
            self.settings = get_api_settings()
        except Exception as e:
            logger.warning(f"API settings loading failed: {e}. Using safe fallback.")
            from ..utils.robust_config import safe_get_api_settings
            self.settings = safe_get_api_settings()
        logger.debug(f"   Environment: {self.settings.environment}")
        self.grid_size = grid_size
        self.domain_size = domain_size
        self.dx = domain_size / grid_size

        # Grid coordinates
        x = np.linspace(0, domain_size, grid_size)
        y = np.linspace(0, domain_size, grid_size)
        self.X, self.Y = np.meshgrid(x, y)

        # Vortex registry
        self.vortices: List[Vortex] = []

        # Physical parameters
        self.viscosity = 0.01  # Kinematic viscosity
        self.diffusivity = 0.01  # Vorticity diffusion

        logger.info(f"Vortex Dynamics Engine initialized: {grid_size}x{grid_size} grid")

    def create_vortex(self
                     position: Tuple[float, float],
                     circulation: float
                     vortex_type: VortexType = VortexType.COGNITIVE
                     radius: float = 1.0) -> Vortex:
        """
        Create a new vortex

        Args:
            position: (x, y) position
            circulation: Vortex strength
            vortex_type: Type of vortex
            radius: Core radius

        Returns:
            Created vortex
        """
        vortex = Vortex(
            position=np.array(position),
            circulation=circulation,
            radius=radius,
            vortex_type=vortex_type,
            velocity=np.zeros(2)
        )

        self.vortices.append(vortex)
        logger.info(f"Created {vortex_type.value} vortex at {position} with Γ={circulation}")

        return vortex

    def compute_velocity_field(self) -> np.ndarray:
        """
        Compute velocity field from all vortices

        Returns:
            Velocity field (u, v) components
        """
        u = np.zeros((self.grid_size, self.grid_size))
        v = np.zeros((self.grid_size, self.grid_size))

        for vortex in self.vortices:
            u_vortex, v_vortex = self._compute_vortex_velocity(vortex)
            u += u_vortex
            v += v_vortex

        return np.stack([u, v], axis=-1)

    def _compute_vortex_velocity(self, vortex: Vortex) -> Tuple[np.ndarray, np.ndarray]:
        """Compute velocity field for a single vortex"""
        # Distance from vortex center
        dx = self.X - vortex.position[0]
        dy = self.Y - vortex.position[1]
        r = np.sqrt(dx**2 + dy**2)

        # Avoid singularity at vortex center
        r = np.maximum(r, 0.1 * self.dx)

        if vortex.vortex_type == VortexType.POINT:
            # Point vortex: v_θ = Γ/(2πr)
            v_theta = vortex.circulation / (2 * np.pi * r)

        elif vortex.vortex_type == VortexType.RANKINE:
            # Rankine vortex: solid body rotation in core
            v_theta = np.where(
                r < vortex.radius
                vortex.circulation * r / (2 * np.pi * vortex.radius**2),
                vortex.circulation / (2 * np.pi * r)
            )

        elif vortex.vortex_type == VortexType.LAMB_OSEEN:
            # Lamb-Oseen vortex: viscous decay
            v_theta = (vortex.circulation / (2 * np.pi * r)) * \
                     (1 - np.exp(-r**2 / (4 * self.viscosity * (vortex.age + 1))))

        elif vortex.vortex_type == VortexType.COGNITIVE:
            # Cognitive vortex: custom profile with attraction/repulsion
            core_factor = np.exp(-r**2 / (2 * vortex.radius**2))
            v_theta = (vortex.circulation / (2 * np.pi * r)) * \
                     (1 - core_factor) * vortex.stability
        else:
            v_theta = vortex.circulation / (2 * np.pi * r)

        # Convert to Cartesian components
        u = -v_theta * dy / r
        v = v_theta * dx / r

        return u, v

    def compute_vorticity(self, velocity_field: np.ndarray) -> np.ndarray:
        """
        Compute vorticity field ω = ∂v/∂x - ∂u/∂y

        Args:
            velocity_field: (u, v) components

        Returns:
            Vorticity field
        """
        u = velocity_field[..., 0]
        v = velocity_field[..., 1]

        # Compute derivatives using central differences
        du_dy = np.gradient(u, self.dx, axis=0)
        dv_dx = np.gradient(v, self.dx, axis=1)

        vorticity = dv_dx - du_dy

        return vorticity

    def compute_stream_function(self, vorticity: np.ndarray) -> np.ndarray:
        """
        Compute stream function from vorticity

        Solves: ∇²ψ = -ω
        """
        # Use FFT-based Poisson solver
        omega_hat = np.fft.fft2(vorticity)

        # Wave numbers
        kx = np.fft.fftfreq(self.grid_size, self.dx).reshape(-1, 1) * 2 * np.pi
        ky = np.fft.fftfreq(self.grid_size, self.dx).reshape(1, -1) * 2 * np.pi

        # Laplacian in Fourier space
        k_squared = kx**2 + ky**2
        k_squared[0, 0] = 1.0  # Avoid division by zero

        # Solve for stream function
        psi_hat = -omega_hat / k_squared
        psi_hat[0, 0] = 0  # Set mean to zero

        psi = np.real(np.fft.ifft2(psi_hat))

        return psi

    def evolve_vortices(self, dt: float):
        """
        Evolve vortex positions and properties

        Args:
            dt: Time step
        """
        # Compute velocity at each vortex position due to other vortices
        for i, vortex in enumerate(self.vortices):
            # Velocity induced by other vortices
            u_induced = 0.0
            v_induced = 0.0

            for j, other in enumerate(self.vortices):
                if i != j:
                    dx = vortex.position[0] - other.position[0]
                    dy = vortex.position[1] - other.position[1]
                    r = np.sqrt(dx**2 + dy**2)

                    if r > 0.1 * self.dx:
                        # Biot-Savart law
                        factor = other.circulation / (2 * np.pi * r**2)
                        u_induced += -factor * dy
                        v_induced += factor * dx

            vortex.velocity = np.array([u_induced, v_induced])
            vortex.update_position(dt)

            # Apply periodic boundary conditions
            vortex.position[0] = vortex.position[0] % self.domain_size
            vortex.position[1] = vortex.position[1] % self.domain_size

            # Update vortex properties
            self._update_vortex_properties(vortex, dt)

    def _update_vortex_properties(self, vortex: Vortex, dt: float):
        """Update vortex circulation and stability"""
        # Viscous decay
        if vortex.vortex_type == VortexType.LAMB_OSEEN:
            decay_factor = np.exp(-dt * self.viscosity / vortex.radius**2)
            vortex.circulation *= decay_factor

        # Cognitive vortex dynamics
        elif vortex.vortex_type == VortexType.COGNITIVE:
            # Stability decay with age
            vortex.stability *= np.exp(-dt * 0.1)

            # Circulation modulation
            vortex.circulation *= (0.99 + 0.01 * np.sin(vortex.age))

    def merge_vortices(self, merge_radius: float = 1.0):
        """
        Merge nearby vortices

        Args:
            merge_radius: Distance threshold for merging
        """
        merged = []
        i = 0

        while i < len(self.vortices):
            if i in merged:
                i += 1
                continue

            vortex_i = self.vortices[i]

            for j in range(i + 1, len(self.vortices)):
                if j in merged:
                    continue

                vortex_j = self.vortices[j]
                distance = np.linalg.norm(vortex_i.position - vortex_j.position)

                if distance < merge_radius:
                    # Merge vortices
                    total_circulation = vortex_i.circulation + vortex_j.circulation

                    # Weighted average position
                    new_position = (
                        vortex_i.position * abs(vortex_i.circulation) +
                        vortex_j.position * abs(vortex_j.circulation)
                    ) / (abs(vortex_i.circulation) + abs(vortex_j.circulation))

                    # Update vortex i
                    vortex_i.position = new_position
                    vortex_i.circulation = total_circulation
                    vortex_i.radius = np.sqrt(vortex_i.radius**2 + vortex_j.radius**2)

                    merged.append(j)
                    logger.info(f"Merged vortices: Γ_total={total_circulation}")

            i += 1

        # Remove merged vortices
        self.vortices = [v for i, v in enumerate(self.vortices) if i not in merged]

    def calculate_circulation(self) -> float:
        """Calculate total circulation in the field"""
        total_circulation = sum(vortex.circulation for vortex in self.vortices)
        return total_circulation

    def calculate_energy(self, velocity_field: np.ndarray) -> float:
        """
        Calculate kinetic energy of the flow

        E = 0.5 * ∫∫ (u² + v²) dA
        """
        u = velocity_field[..., 0]
        v = velocity_field[..., 1]

        energy_density = 0.5 * (u**2 + v**2)
        total_energy = np.sum(energy_density) * self.dx**2

        return total_energy

    def calculate_enstrophy(self, vorticity: np.ndarray) -> float:
        """
        Calculate enstrophy (vorticity squared)

        Ω = 0.5 * ∫∫ ω² dA
        """
        enstrophy_density = 0.5 * vorticity**2
        total_enstrophy = np.sum(enstrophy_density) * self.dx**2

        return total_enstrophy

    def detect_coherent_structures(self, vorticity: np.ndarray
                                 threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Detect coherent vortex structures in the field

        Args:
            vorticity: Vorticity field
            threshold: Detection threshold

        Returns:
            List of detected structures
        """
        # Normalize vorticity
        vort_norm = np.abs(vorticity) / (np.max(np.abs(vorticity)) + 1e-10)

        # Threshold and label regions
        binary = vort_norm > threshold
        labeled, num_features = ndimage.label(binary)

        structures = []
        for i in range(1, num_features + 1):
            mask = labeled == i

            # Calculate structure properties
            y_coords, x_coords = np.where(mask)
            center_x = np.mean(x_coords) * self.dx
            center_y = np.mean(y_coords) * self.dx

            circulation = np.sum(vorticity[mask]) * self.dx**2
            area = np.sum(mask) * self.dx**2

            structure = {
                "center": (center_x, center_y),
                "circulation": circulation
                "area": area
                "intensity": np.mean(np.abs(vorticity[mask])),
                "coherence": np.std(vorticity[mask]) / (np.mean(np.abs(vorticity[mask])) + 1e-10)
            }

            structures.append(structure)

        return structures

    def generate_vortex_field_state(self) -> VortexField:
        """Generate complete vortex field state"""
        velocity_field = self.compute_velocity_field()
        vorticity = self.compute_vorticity(velocity_field)
        stream_function = self.compute_stream_function(vorticity)
        energy = self.calculate_energy(velocity_field)
        enstrophy = self.calculate_enstrophy(vorticity)

        return VortexField(
            velocity_field=velocity_field
            vorticity_field=vorticity
            stream_function=stream_function
            energy=energy
            enstrophy=enstrophy
        )

    def cognitive_vortex_interaction(self, semantic_field: np.ndarray) -> np.ndarray:
        """
        Model interaction between vortices and semantic field

        Args:
            semantic_field: Input semantic activation field

        Returns:
            Modified semantic field after vortex interaction
        """
        # Get current vortex field
        vortex_state = self.generate_vortex_field_state()

        # Advect semantic field by velocity field
        u = vortex_state.velocity_field[..., 0]
        v = vortex_state.velocity_field[..., 1]

        # Semi-Lagrangian advection
        advected_field = self._semi_lagrangian_advection(semantic_field, u, v, dt=0.1)

        # Add vortex-induced mixing
        mixing_strength = 0.1
        mixed_field = advected_field + mixing_strength * vortex_state.vorticity_field

        # Apply cognitive modulation based on vortex types
        for vortex in self.vortices:
            if vortex.vortex_type == VortexType.COGNITIVE:
                # Cognitive vortices enhance nearby semantic activations
                distance_field = np.sqrt(
                    (self.X - vortex.position[0])**2 +
                    (self.Y - vortex.position[1])**2
                )

                influence = np.exp(-distance_field**2 / (2 * vortex.radius**2))
                mixed_field += vortex.stability * influence * np.sign(vortex.circulation)

        return mixed_field

    def _semi_lagrangian_advection(self, field: np.ndarray
                                  u: np.ndarray, v: np.ndarray
                                  dt: float) -> np.ndarray:
        """Semi-Lagrangian advection scheme"""
        # Backward particle tracing
        X_back = self.X - u * dt
        Y_back = self.Y - v * dt

        # Periodic boundary conditions
        X_back = X_back % self.domain_size
        Y_back = Y_back % self.domain_size

        # Interpolate field values
        from scipy.interpolate import RegularGridInterpolator

from ..config.settings import get_settings
from ..utils.robust_config import get_api_settings

        x_1d = np.linspace(0, self.domain_size, self.grid_size)
        y_1d = np.linspace(0, self.domain_size, self.grid_size)

        interpolator = RegularGridInterpolator(
            (y_1d, x_1d), field
            bounds_error=False
            fill_value=0
        )

        points = np.stack([Y_back.ravel(), X_back.ravel()], axis=-1)
        advected = interpolator(points).reshape(field.shape)

        return advected

# Convenience class for simple vortex field operations
class SimpleVortexField:
    """Auto-generated class."""
    pass
    """Simple interface for vortex field operations"""

    def __init__(self, grid_size: int = 128):
        try:
            self.settings = get_api_settings()
        except Exception as e:
            logger.warning(f"API settings loading failed: {e}. Using safe fallback.")
            from ..utils.robust_config import safe_get_api_settings
            self.settings = safe_get_api_settings()
        logger.debug(f"   Environment: {self.settings.environment}")
        self.engine = VortexDynamicsEngine(grid_size=grid_size)

    def add_vortex(self, x: float, y: float, strength: float):
        """Add a vortex to the field"""
        return self.engine.create_vortex(
            position=(x, y),
            circulation=strength
            vortex_type=VortexType.COGNITIVE
        )

    def calculate_circulation(self) -> float:
        """Calculate total circulation"""
        return self.engine.calculate_circulation()

    def evolve(self, dt: float = 0.1, steps: int = 10):
        """Evolve the vortex field"""
        for _ in range(steps):
            self.engine.evolve_vortices(dt)
            self.engine.merge_vortices()

    def get_field_state(self) -> VortexField:
        """Get current field state"""
        return self.engine.generate_vortex_field_state()

def create_vortex_field(grid_size: int = 128) -> SimpleVortexField:
    """Factory function to create vortex field"""
    return SimpleVortexField(grid_size=grid_size)
