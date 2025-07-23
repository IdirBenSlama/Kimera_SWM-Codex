# KIMERA Engine Interface API Reference
**Category**: API Documentation | **Status**: Complete | **Last Updated**: January 23, 2025

## Overview

This document provides comprehensive API documentation for all KIMERA SWM engine interfaces. All engines have been validated through systematic performance testing and are production-ready.

**Base Path**: All engines are located in `src/engines/`
**Python Package**: `kimera.engines`

---

## 🔧 Engine Classes

### ThermodynamicEngine

**File**: `src/engines/thermodynamic_engine.py`  
**Purpose**: Physics-compliant thermodynamic calculations for cognitive fields

#### Constructor

```python
from kimera.engines import ThermodynamicEngine

engine = ThermodynamicEngine()
```

**Parameters**: None  
**Returns**: ThermodynamicEngine instance  
**Raises**: 
- `ImportError`: If required dependencies are not available
- `ConfigurationError`: If system configuration is invalid

#### Methods

##### `calculate_semantic_temperature()`

Calculates the semantic temperature of a cognitive field using covariance matrix trace.

```python
def calculate_semantic_temperature(
    self, 
    cognitive_field: List[np.ndarray]
) -> float
```

**Parameters**:
- `cognitive_field` (List[np.ndarray]): List of embedding vectors representing the cognitive field

**Returns**: 
- `float`: Semantic temperature value (positive real number)

**Raises**:
- `TypeError`: If cognitive_field is not a list or contains non-numpy arrays
- `ValueError`: If arrays have incompatible shapes or are empty
- `ThermodynamicError`: If temperature calculation fails physical constraints

**Example**:
```python
import numpy as np
from kimera.engines import ThermodynamicEngine

# Initialize engine
engine = ThermodynamicEngine()

# Create sample cognitive field
field = [
    np.random.randn(100),  # Embedding vector 1
    np.random.randn(100),  # Embedding vector 2
    np.random.randn(100)   # Embedding vector 3
]

# Calculate semantic temperature
temperature = engine.calculate_semantic_temperature(field)
print(f"Semantic temperature: {temperature:.4f}")
```

##### `run_semantic_carnot_engine()`

Runs a theoretical semantic Carnot engine between two cognitive field reservoirs.

```python
def run_semantic_carnot_engine(
    self, 
    hot_reservoir: List[np.ndarray], 
    cold_reservoir: List[np.ndarray]
) -> Dict[str, float]
```

**Parameters**:
- `hot_reservoir` (List[np.ndarray]): High-temperature source embeddings
- `cold_reservoir` (List[np.ndarray]): Low-temperature sink embeddings

**Returns**: 
- `Dict[str, float]`: Dictionary containing:
  - `efficiency`: Carnot efficiency (0.0 to 1.0)
  - `work_output`: Work extracted from the process
  - `heat_absorbed`: Heat absorbed from hot reservoir
  - `heat_rejected`: Heat rejected to cold reservoir
  - `entropy_change`: Total entropy change

**Raises**:
- `TypeError`: If reservoirs are not lists of numpy arrays
- `ValueError`: If reservoirs are empty or have incompatible dimensions
- `ThermodynamicError`: If reservoirs violate thermodynamic constraints
- `TemperatureError`: If cold reservoir temperature >= hot reservoir temperature

**Example**:
```python
# Create hot and cold reservoirs
hot_reservoir = [np.random.randn(100) + 2.0 for _ in range(10)]   # Higher variance
cold_reservoir = [np.random.randn(100) * 0.5 for _ in range(10)]  # Lower variance

# Run Carnot engine
result = engine.run_semantic_carnot_engine(hot_reservoir, cold_reservoir)

print(f"Carnot efficiency: {result['efficiency']:.4f}")
print(f"Work output: {result['work_output']:.4f}")
print(f"Entropy change: {result['entropy_change']:.6f}")
```

##### `calculate_entropy()`

Calculates Shannon entropy for discrete probability distributions.

```python
def calculate_entropy(
    self, 
    distribution: np.ndarray, 
    base: float = 2.0
) -> float
```

**Parameters**:
- `distribution` (np.ndarray): Probability distribution (must sum to 1.0)
- `base` (float, optional): Logarithm base for entropy calculation (default: 2.0)

**Returns**:
- `float`: Shannon entropy in specified base units

**Raises**:
- `ValueError`: If distribution doesn't sum to 1.0 or contains negative values
- `TypeError`: If distribution is not a numpy array

**Example**:
```python
# Uniform distribution
uniform_dist = np.array([0.25, 0.25, 0.25, 0.25])
entropy_uniform = engine.calculate_entropy(uniform_dist)

# Skewed distribution  
skewed_dist = np.array([0.7, 0.2, 0.05, 0.05])
entropy_skewed = engine.calculate_entropy(skewed_dist)

print(f"Uniform entropy: {entropy_uniform:.4f} bits")
print(f"Skewed entropy: {entropy_skewed:.4f} bits")
```

---

### QuantumFieldEngine

**File**: `src/engines/quantum_field_engine.py`  
**Purpose**: Quantum-inspired information processing for cognitive modeling

#### Constructor

```python
from kimera.engines import QuantumFieldEngine

engine = QuantumFieldEngine(coherence_time=1000)
```

**Parameters**:
- `coherence_time` (int, optional): Quantum coherence preservation time (default: 1000)

**Returns**: QuantumFieldEngine instance

#### Methods

##### `create_superposition()`

Creates a quantum superposition state from multiple cognitive states.

```python
def create_superposition(
    self, 
    states: List[np.ndarray], 
    amplitudes: Optional[List[complex]] = None
) -> QuantumState
```

**Parameters**:
- `states` (List[np.ndarray]): List of basis states for superposition
- `amplitudes` (List[complex], optional): Complex amplitudes (auto-normalized if None)

**Returns**:
- `QuantumState`: Quantum superposition state object

**Example**:
```python
# Create basis states
state1 = np.array([1, 0, 0])
state2 = np.array([0, 1, 0])
state3 = np.array([0, 0, 1])

# Create superposition
superposition = engine.create_superposition(
    states=[state1, state2, state3],
    amplitudes=[1+0j, 1+0j, 1+0j]  # Equal superposition
)

print(f"Superposition fidelity: {superposition.fidelity:.4f}")
```

##### `measure_state()`

Performs a quantum measurement on a superposition state.

```python
def measure_state(
    self, 
    quantum_state: QuantumState, 
    observable: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, float]
```

**Parameters**:
- `quantum_state` (QuantumState): Quantum state to measure
- `observable` (np.ndarray, optional): Observable operator matrix

**Returns**:
- `Tuple[np.ndarray, float]`: (collapsed_state, measurement_probability)

**Example**:
```python
# Measure the superposition
collapsed_state, probability = engine.measure_state(superposition)
print(f"Collapsed to state with probability: {probability:.4f}")
```

---

### SPDEEngine

**File**: `src/engines/spde_engine.py`  
**Purpose**: Stochastic partial differential equation solving for cognitive dynamics

#### Constructor

```python
from kimera.engines import SPDEEngine

engine = SPDEEngine(dt=0.01, noise_strength=0.1)
```

**Parameters**:
- `dt` (float, optional): Time step for integration (default: 0.01)
- `noise_strength` (float, optional): Stochastic noise strength (default: 0.1)

#### Methods

##### `solve_diffusion()`

Solves diffusion equation for cognitive field evolution.

```python
def solve_diffusion(
    self, 
    initial_field: np.ndarray, 
    diffusion_coeff: float,
    time_steps: int,
    boundary_conditions: str = "periodic"
) -> np.ndarray
```

**Parameters**:
- `initial_field` (np.ndarray): Initial cognitive field configuration
- `diffusion_coeff` (float): Diffusion coefficient
- `time_steps` (int): Number of evolution steps
- `boundary_conditions` (str): Boundary condition type ("periodic", "dirichlet", "neumann")

**Returns**:
- `np.ndarray`: Evolved field at final time

**Example**:
```python
# Initial Gaussian field
x = np.linspace(0, 10, 100)
initial_field = np.exp(-(x - 5)**2)

# Solve diffusion
final_field = engine.solve_diffusion(
    initial_field=initial_field,
    diffusion_coeff=0.1,
    time_steps=1000,
    boundary_conditions="periodic"
)

print(f"Field evolution completed over {len(final_field)} points")
```

---

## Error Handling

### Exception Types

All engines use a common exception hierarchy:

```python
class KimeraEngineError(Exception):
    """Base exception for all engine errors"""
    pass

class ThermodynamicError(KimeraEngineError):
    """Thermodynamic constraint violations"""
    pass

class QuantumError(KimeraEngineError):
    """Quantum state manipulation errors"""
    pass

class SPDEError(KimeraEngineError):
    """SPDE solving errors"""
    pass
```

### Error Handling Best Practices

```python
from kimera.engines import ThermodynamicEngine, ThermodynamicError

try:
    engine = ThermodynamicEngine()
    result = engine.calculate_semantic_temperature(field)
except ThermodynamicError as e:
    print(f"Thermodynamic constraint violated: {e}")
except ValueError as e:
    print(f"Invalid input data: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Performance Considerations

### GPU Acceleration

All engines support GPU acceleration when available:

```python
# Engines automatically detect and use GPU if available
engine = ThermodynamicEngine()
print(f"Using device: {engine.device}")  # cuda:0 or cpu
```

### Memory Management

For large cognitive fields, use batch processing:

```python
# Process large fields in batches
def process_large_field(engine, large_field, batch_size=1000):
    results = []
    for i in range(0, len(large_field), batch_size):
        batch = large_field[i:i+batch_size]
        result = engine.calculate_semantic_temperature(batch)
        results.append(result)
    return results
```

## Related Documentation

- **[Architecture Overview](../../architecture/core-systems/system-overview.md)** - System architecture design
- **[Engine Specifications](../../architecture/engines/)** - Detailed engine documentation
- **[API Examples](examples.md)** - Complete usage examples
- **[Troubleshooting](../troubleshooting/)** - Common issues and solutions

---

**Navigation**: [📡 API Home](README.md) | [📖 Examples](examples.md) | [🔧 Troubleshooting](../troubleshooting/) | [🏗️ Architecture](../../architecture/) 