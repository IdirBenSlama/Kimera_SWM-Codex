"""
Layered Architecture Definition and Validation
Part of Phase 2: Architecture Refactoring

This module defines the layered architecture and provides validation
to ensure dependencies only flow in the correct direction.
"""

from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple, Any
import logging

import ast
logger = logging.getLogger(__name__)


class Layer(Enum):
    """System architecture layers"""
    INFRASTRUCTURE = 1  # GPU, Database, Config, External Services
    CORE = 2           # Embedding, System, Vault, Memory
    ENGINES = 3        # Contradiction, Thermodynamic, Cognitive, Diffusion
    API = 4            # Routers, Middleware, Handlers, WebSockets


@dataclass
class ModuleInfo:
    """Information about a module"""
    path: Path
    layer: Layer
    imports: Set[str]
    dependencies: Set['ModuleInfo'] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = set()


class LayerValidator:
    """Validates that dependencies only flow downward in the architecture"""
    
    # Define which layers can depend on which other layers
    ALLOWED_DEPENDENCIES = {
        Layer.API: [Layer.ENGINES, Layer.CORE, Layer.INFRASTRUCTURE],
        Layer.ENGINES: [Layer.CORE, Layer.INFRASTRUCTURE],
        Layer.CORE: [Layer.INFRASTRUCTURE],
        Layer.INFRASTRUCTURE: []  # Infrastructure has no dependencies on other layers
    }
    
    # Map module paths to layers
    MODULE_LAYER_MAPPING = {
        # Infrastructure Layer
        "backend/core/gpu_foundation.py": Layer.INFRASTRUCTURE,
        "backend/core/database.py": Layer.INFRASTRUCTURE,
        "backend/core/config.py": Layer.INFRASTRUCTURE,
        "backend/utils": Layer.INFRASTRUCTURE,
        "backend/monitoring": Layer.INFRASTRUCTURE,
        
        # Core Layer
        "backend/core/kimera_system.py": Layer.CORE,
        "backend/core/embedding_utils.py": Layer.CORE,
        "backend/core/vault_manager.py": Layer.CORE,
        "backend/core/memory_manager.py": Layer.CORE,
        "backend/core/lazy_initialization_manager.py": Layer.CORE,
        
        # Engines Layer
        "backend/engines/kimera_contradiction_engine.py": Layer.ENGINES,
        "backend/engines/kimera_thermodynamic_engine.py": Layer.ENGINES,
        "backend/engines/cognitive_field_dynamics.py": Layer.ENGINES,
        "backend/engines/kimera_text_diffusion_engine.py": Layer.ENGINES,
        "backend/engines/kimera_optimization_engine.py": Layer.ENGINES,
        "backend/engines/universal_translator_hub.py": Layer.ENGINES,
        
        # API Layer
        "backend/api/routers": Layer.API,
        "backend/api/middleware": Layer.API,
        "backend/api/websocket": Layer.API,
        "backend/api/handlers": Layer.API,
    }
    
    @classmethod
    def validate_dependency(cls, from_layer: Layer, to_layer: Layer) -> bool:
        """
        Check if a dependency from one layer to another is allowed
        
        Args:
            from_layer: The layer that has the dependency
            to_layer: The layer being depended upon
            
        Returns:
            True if the dependency is allowed, False otherwise
        """
        if from_layer == to_layer:
            return True  # Same layer dependencies are allowed
        
        return to_layer in cls.ALLOWED_DEPENDENCIES.get(from_layer, [])
    
    @classmethod
    def get_layer_for_module(cls, module_path: str) -> Optional[Layer]:
        """
        Determine which layer a module belongs to
        
        Args:
            module_path: Path to the module
            
        Returns:
            The layer the module belongs to, or None if not mapped
        """
        # Normalize path
        module_path = module_path.replace("\\", "/")
        
        # Check exact matches first
        for pattern, layer in cls.MODULE_LAYER_MAPPING.items():
            if module_path == pattern or module_path.endswith(pattern):
                return layer
        
        # Check prefix matches
        for pattern, layer in cls.MODULE_LAYER_MAPPING.items():
            if module_path.startswith(pattern):
                return layer
        
        return None
    
    @classmethod
    def analyze_module_dependencies(cls, module_path: Path) -> Tuple[Set[str], List[str]]:
        """
        Analyze a Python module's imports
        
        Args:
            module_path: Path to the module file
            
        Returns:
            Tuple of (imports set, violation messages)
        """
        imports = set()
        violations = []
        
        try:
            with open(module_path, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read())
            
            # Extract imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.add(node.module)
            
            # Check each import
            module_layer = cls.get_layer_for_module(str(module_path))
            if module_layer:
                for imp in imports:
                    if imp.startswith('backend'):
                        import_path = imp.replace('.', '/')
                        import_layer = cls.get_layer_for_module(import_path)
                        
                        if import_layer and not cls.validate_dependency(module_layer, import_layer):
                            violations.append(
                                f"{module_path} (Layer: {module_layer.name}) imports from "
                                f"{imp} (Layer: {import_layer.name}) - VIOLATION"
                            )
        
        except Exception as e:
            logger.error(f"Error analyzing {module_path}: {e}")
        
        return imports, violations
    
    @classmethod
    def validate_architecture(cls, root_path: Path) -> Dict[str, Any]:
        """
        Validate the entire architecture
        
        Args:
            root_path: Root path of the project
            
        Returns:
            Dictionary with validation results
        """
        results = {
            "total_modules": 0,
            "violations": [],
            "module_layers": {},
            "layer_statistics": {layer: 0 for layer in Layer},
            "circular_dependencies": []
        }
        
        # Analyze all Python files
        for py_file in root_path.rglob("*.py"):
            if ".venv" in str(py_file) or "__pycache__" in str(py_file):
                continue
            
            relative_path = py_file.relative_to(root_path)
            layer = cls.get_layer_for_module(str(relative_path))
            
            if layer:
                results["total_modules"] += 1
                results["module_layers"][str(relative_path)] = layer.name
                results["layer_statistics"][layer] += 1
                
                # Check dependencies
                imports, violations = cls.analyze_module_dependencies(py_file)
                results["violations"].extend(violations)
        
        # Check for circular dependencies
        results["circular_dependencies"] = cls.detect_circular_dependencies(root_path)
        
        return results
    
    @classmethod
    def detect_circular_dependencies(cls, root_path: Path) -> List[List[str]]:
        """
        Detect circular dependencies in the codebase
        
        Args:
            root_path: Root path of the project
            
        Returns:
            List of circular dependency chains
        """
        # Build dependency graph
        graph = {}
        
        for py_file in root_path.rglob("*.py"):
            if ".venv" in str(py_file) or "__pycache__" in str(py_file):
                continue
            
            module_name = str(py_file.relative_to(root_path)).replace("/", ".").replace("\\", ".")[:-3]
            imports, _ = cls.analyze_module_dependencies(py_file)
            
            # Filter to only backend imports
            backend_imports = [imp for imp in imports if imp.startswith("backend")]
            graph[module_name] = backend_imports
        
        # Find cycles using DFS
        cycles = []
        visited = set()
        rec_stack = set()
        
        def dfs(node, path):
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if dfs(neighbor, path):
                        return True
                elif neighbor in rec_stack:
                    # Found cycle
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:] + [neighbor]
                    cycles.append(cycle)
                    return True
            
            path.pop()
            rec_stack.remove(node)
            return False
        
        for node in graph:
            if node not in visited:
                dfs(node, [])
        
        return cycles


class ArchitectureEnforcer:
    """Enforces architectural rules at runtime"""
    
    def __init__(self):
        self.call_stack: List[Tuple[Layer, str]] = []
    
    def check_call(self, from_module: str, to_module: str) -> bool:
        """
        Check if a call from one module to another is allowed
        
        Args:
            from_module: The calling module
            to_module: The called module
            
        Returns:
            True if allowed, raises exception if not
        """
        from_layer = LayerValidator.get_layer_for_module(from_module)
        to_layer = LayerValidator.get_layer_for_module(to_module)
        
        if from_layer and to_layer:
            if not LayerValidator.validate_dependency(from_layer, to_layer):
                raise RuntimeError(
                    f"Architecture violation: {from_module} (Layer: {from_layer.name}) "
                    f"cannot call {to_module} (Layer: {to_layer.name})"
                )
        
        return True


# Global architecture enforcer
architecture_enforcer = ArchitectureEnforcer()


def layer_boundary(layer: Layer):
    """
    Decorator to mark a class or function as belonging to a specific layer
    
    Args:
        layer: The layer this component belongs to
    """
    def decorator(cls_or_func):
        cls_or_func.__layer__ = layer
        return cls_or_func
    
    return decorator


# Example usage in tests
if __name__ == "__main__":
    # Validate the architecture
    import sys
    from pathlib import Path
    
    project_root = Path(__file__).parent.parent.parent
    results = LayerValidator.validate_architecture(project_root)
    
    print(f"Total modules analyzed: {results['total_modules']}")
    print(f"Architecture violations: {len(results['violations'])}")
    
    if results['violations']:
        print("\nViolations found:")
        for violation in results['violations']:
            print(f"  - {violation}")
    
    if results['circular_dependencies']:
        print("\nCircular dependencies found:")
        for cycle in results['circular_dependencies']:
            print(f"  - {' -> '.join(cycle)}")
    
    print("\nLayer statistics:")
    for layer, count in results['layer_statistics'].items():
        print(f"  - {layer.name}: {count} modules")
    
    # Exit with error if violations found
    if results['violations'] or results['circular_dependencies']:
        sys.exit(1)