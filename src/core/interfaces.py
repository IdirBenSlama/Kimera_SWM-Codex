"""
Service Interfaces for KIMERA System
Part of Phase 2: Architecture Refactoring

This module defines abstract interfaces to break circular dependencies
and enable proper dependency injection.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union

from abc import ABC, abstractmethod
import numpy as np
import torch
class IGPUService(ABC):
    """GPU resource management interface"""
    
    @abstractmethod
    def get_device(self) -> torch.device:
        """Get the current torch device"""
        pass
    
    @abstractmethod
    def get_memory_info(self) -> Dict[str, int]:
        """Get GPU memory information"""
        pass
    
    @abstractmethod
    def allocate_tensor(self, size: Tuple[int, ...], dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Allocate a tensor on the appropriate device"""
        pass
    
    @abstractmethod
    def clear_cache(self) -> None:
        """Clear GPU cache"""
        pass


class IDatabaseService(ABC):
    """Database service interface"""
    
    @abstractmethod
    async def execute(self, query: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Execute a database query"""
        pass
    
    @abstractmethod
    async def fetch_one(self, query: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Fetch a single row"""
        pass
    
    @abstractmethod
    async def fetch_all(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Fetch all rows"""
        pass
    
    @abstractmethod
    async def transaction(self):
        """Create a database transaction context"""
        pass


class IConfigurationService(ABC):
    """Configuration service interface"""
    
    @abstractmethod
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value"""
        pass
    
    @abstractmethod
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get a configuration section"""
        pass
    
    @abstractmethod
    def reload(self) -> None:
        """Reload configuration from source"""
        pass


# Core Layer Interfaces

class IEmbeddingService(ABC):
    """Text embedding service interface"""
    
    @abstractmethod
    async def get_embedding(self, text: str) -> torch.Tensor:
        """Get embedding for text"""
        pass
    
    @abstractmethod
    async def get_embeddings_batch(self, texts: List[str]) -> torch.Tensor:
        """Get embeddings for multiple texts"""
        pass
    
    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings"""
        pass
    
    @abstractmethod
    async def similarity(self, embedding1: torch.Tensor, embedding2: torch.Tensor) -> float:
        """Calculate similarity between embeddings"""
        pass


class IVaultService(ABC):
    """Vault storage service interface"""
    
    @abstractmethod
    async def create_vault(self, vault_id: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Create a new vault"""
        pass
    
    @abstractmethod
    async def store_geoid(self, vault_id: str, geoid_id: str, data: Dict[str, Any]) -> bool:
        """Store a geoid in a vault"""
        pass
    
    @abstractmethod
    async def retrieve_geoid(self, vault_id: str, geoid_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a geoid from a vault"""
        pass
    
    @abstractmethod
    async def list_geoids(self, vault_id: str) -> List[str]:
        """List all geoids in a vault"""
        pass
    
    @abstractmethod
    async def delete_geoid(self, vault_id: str, geoid_id: str) -> bool:
        """Delete a geoid from a vault"""
        pass


class IMemoryService(ABC):
    """Memory management service interface"""
    
    @abstractmethod
    async def store_memory(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Store a memory with optional TTL"""
        pass
    
    @abstractmethod
    async def retrieve_memory(self, key: str) -> Optional[Any]:
        """Retrieve a memory"""
        pass
    
    @abstractmethod
    async def delete_memory(self, key: str) -> bool:
        """Delete a memory"""
        pass
    
    @abstractmethod
    async def clear_expired(self) -> int:
        """Clear expired memories, return count cleared"""
        pass


# Engine Layer Interfaces

class IContradictionEngine(ABC):
    """Contradiction detection engine interface"""
    
    @abstractmethod
    async def detect_contradictions(self, geoids: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect contradictions between geoids"""
        pass
    
    @abstractmethod
    async def resolve_contradiction(self, contradiction: Dict[str, Any]) -> Dict[str, Any]:
        """Attempt to resolve a contradiction"""
        pass
    
    @abstractmethod
    def get_contradiction_threshold(self) -> float:
        """Get current contradiction threshold"""
        pass
    
    @abstractmethod
    def set_contradiction_threshold(self, threshold: float) -> None:
        """Set contradiction threshold"""
        pass


class IThermodynamicEngine(ABC):
    """Thermodynamic optimization engine interface"""
    
    @abstractmethod
    async def calculate_entropy(self, system_state: Dict[str, Any]) -> float:
        """Calculate system entropy"""
        pass
    
    @abstractmethod
    async def optimize_energy(self, operations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Optimize operations for energy efficiency"""
        pass
    
    @abstractmethod
    def get_temperature(self) -> float:
        """Get current system temperature"""
        pass
    
    @abstractmethod
    def set_temperature(self, temperature: float) -> None:
        """Set system temperature"""
        pass


class ICognitiveFieldEngine(ABC):
    """Cognitive field dynamics engine interface"""
    
    @abstractmethod
    async def create_field(self, field_id: str, dimensions: int) -> bool:
        """Create a new cognitive field"""
        pass
    
    @abstractmethod
    async def add_to_field(self, field_id: str, geoid_id: str, embedding: torch.Tensor) -> bool:
        """Add a geoid to a cognitive field"""
        pass
    
    @abstractmethod
    async def calculate_field_dynamics(self, field_id: str) -> Dict[str, Any]:
        """Calculate field dynamics"""
        pass
    
    @abstractmethod
    async def find_resonance(self, field_id: str, query_embedding: torch.Tensor, top_k: int = 10) -> List[Tuple[str, float]]:
        """Find resonant geoids in field"""
        pass


class ITextDiffusionEngine(ABC):
    """Text diffusion engine interface"""
    
    @abstractmethod
    async def generate_text(self, prompt: str, max_length: int = 100, temperature: float = 1.0) -> str:
        """Generate text from prompt"""
        pass
    
    @abstractmethod
    async def diffuse_concept(self, concept: str, steps: int = 50) -> List[str]:
        """Diffuse a concept through semantic space"""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the diffusion model"""
        pass


# API Layer Interfaces

class IAuthenticationService(ABC):
    """Authentication service interface"""
    
    @abstractmethod
    async def authenticate(self, credentials: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Authenticate user"""
        pass
    
    @abstractmethod
    async def validate_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Validate authentication token"""
        pass
    
    @abstractmethod
    async def refresh_token(self, refresh_token: str) -> Optional[Dict[str, Any]]:
        """Refresh authentication token"""
        pass


class IRateLimiter(ABC):
    """Rate limiting service interface"""
    
    @abstractmethod
    async def check_rate_limit(self, key: str, limit: int, window: int) -> bool:
        """Check if rate limit exceeded"""
        pass
    
    @abstractmethod
    async def increment_counter(self, key: str) -> int:
        """Increment rate limit counter"""
        pass
    
    @abstractmethod
    async def reset_counter(self, key: str) -> None:
        """Reset rate limit counter"""
        pass


class IMonitoringService(ABC):
    """System monitoring service interface"""
    
    @abstractmethod
    def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a metric"""
        pass
    
    @abstractmethod
    def increment_counter(self, name: str, tags: Optional[Dict[str, str]] = None) -> None:
        """Increment a counter"""
        pass
    
    @abstractmethod
    def record_histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a histogram value"""
        pass
    
    @abstractmethod
    def create_span(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Create a tracing span context"""
        pass


# Composite Interfaces

class IKimeraSystem(ABC):
    """Main KIMERA system interface"""
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the system"""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the system"""
        pass
    
    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Get system status"""
        pass
    
    @abstractmethod
    def get_service(self, service_type: type) -> Any:
        """Get a registered service"""
        pass


# Lifecycle Interfaces

class IDisposable(ABC):
    """Interface for objects that need cleanup"""
    
    @abstractmethod
    def dispose(self) -> None:
        """Cleanup resources"""
        pass


class IInitializable(ABC):
    """Interface for objects that need initialization"""
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the object"""
        pass
    
    @abstractmethod
    def is_initialized(self) -> bool:
        """Check if initialized"""
        pass