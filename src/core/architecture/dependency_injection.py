"""
Dependency Injection Container for KIMERA System
Part of Phase 2: Architecture Refactoring

This module provides a centralized dependency injection container to break
circular dependencies and improve system architecture.
"""

import logging
import threading
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Callable, Dict, Generic, Optional, Type, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")
class ServiceLifetime:
    """Auto-generated class."""
    pass
    """Service lifetime options"""

    SINGLETON = "singleton"
    TRANSIENT = "transient"
    SCOPED = "scoped"
class ServiceDescriptor:
    """Auto-generated class."""
    pass
    """Describes a service registration"""

    def __init__(
        self
        interface: Type
        implementation: Optional[Type] = None
        factory: Optional[Callable] = None
        instance: Optional[Any] = None
        lifetime: str = ServiceLifetime.SINGLETON
    ):
        self.interface = interface
        self.implementation = implementation
        self.factory = factory
        self.instance = instance
        self.lifetime = lifetime
        self.created_at = datetime.now()

        # Validation
        if not any([implementation, factory, instance]):
            raise ValueError("Must provide implementation, factory, or instance")
class ServiceScope:
    """Auto-generated class."""
    pass
    """Represents a service scope for scoped services"""

    def __init__(self, container: "ServiceContainer"):
        self.container = container
        self.scoped_instances: Dict[Type, Any] = {}
        self._lock = threading.Lock()

    def resolve(self, interface: Type[T]) -> T:
        """Resolve a service within this scope"""
        descriptor = self.container.get_descriptor(interface)

        if descriptor.lifetime == ServiceLifetime.SCOPED:
            with self._lock:
                if interface in self.scoped_instances:
                    return self.scoped_instances[interface]

                instance = self.container._create_instance(descriptor)
                self.scoped_instances[interface] = instance
                return instance
        else:
            return self.container.resolve(interface)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Cleanup scoped instances if they implement IDisposable
        for instance in self.scoped_instances.values():
            if hasattr(instance, "dispose"):
                try:
                    instance.dispose()
                except Exception as e:
                    logger.error(f"Error disposing scoped instance: {e}")
class ServiceContainer:
    """Auto-generated class."""
    pass
    """Centralized dependency injection container"""

    def __init__(self):
        self._descriptors: Dict[Type, ServiceDescriptor] = {}
        self._singletons: Dict[Type, Any] = {}
        self._lock = threading.Lock()
        self._building: set = set()  # Prevent circular dependencies during construction

    def register(
        self
        interface: Type[T],
        implementation: Optional[Type[T]] = None
        factory: Optional[Callable[[], T]] = None
        instance: Optional[T] = None
        lifetime: str = ServiceLifetime.SINGLETON
    ) -> "ServiceContainer":
        """
        Register a service

        Args:
            interface: The interface type
            implementation: The implementation class
            factory: A factory function that creates the instance
            instance: A pre-created instance
            lifetime: The service lifetime (singleton, transient, scoped)

        Returns:
            Self for fluent registration
        """
        with self._lock:
            descriptor = ServiceDescriptor(
                interface=interface
                implementation=implementation
                factory=factory
                instance=instance
                lifetime=lifetime
            )

            self._descriptors[interface] = descriptor

            # If instance provided, store as singleton
            if instance and lifetime == ServiceLifetime.SINGLETON:
                self._singletons[interface] = instance

            logger.debug(f"Registered {interface.__name__} with lifetime {lifetime}")

        return self

    def register_singleton(
        self, interface: Type[T], implementation: Type[T]
    ) -> "ServiceContainer":
        """Register a singleton service"""
        return self.register(
            interface, implementation=implementation, lifetime=ServiceLifetime.SINGLETON
        )

    def register_transient(
        self, interface: Type[T], implementation: Type[T]
    ) -> "ServiceContainer":
        """Register a transient service"""
        return self.register(
            interface, implementation=implementation, lifetime=ServiceLifetime.TRANSIENT
        )

    def register_scoped(
        self, interface: Type[T], implementation: Type[T]
    ) -> "ServiceContainer":
        """Register a scoped service"""
        return self.register(
            interface, implementation=implementation, lifetime=ServiceLifetime.SCOPED
        )

    def register_factory(
        self
        interface: Type[T],
        factory: Callable[[], T],
        lifetime: str = ServiceLifetime.SINGLETON
    ) -> "ServiceContainer":
        """Register a service with a factory function"""
        return self.register(interface, factory=factory, lifetime=lifetime)

    def get_descriptor(self, interface: Type) -> ServiceDescriptor:
        """Get service descriptor"""
        if interface not in self._descriptors:
            raise ValueError(f"No registration found for {interface.__name__}")
        return self._descriptors[interface]

    def resolve(self, interface: Type[T]) -> T:
        """
        Resolve a service

        Args:
            interface: The interface type to resolve

        Returns:
            The service instance

        Raises:
            ValueError: If no registration found
            RuntimeError: If circular dependency detected
        """
        with self._lock:
            if interface not in self._descriptors:
                raise ValueError(f"No registration found for {interface.__name__}")

            descriptor = self._descriptors[interface]

            # Handle different lifetimes
            if descriptor.lifetime == ServiceLifetime.SINGLETON:
                if interface in self._singletons:
                    return self._singletons[interface]

                instance = self._create_instance(descriptor)
                self._singletons[interface] = instance
                return instance

            elif descriptor.lifetime == ServiceLifetime.TRANSIENT:
                return self._create_instance(descriptor)

            elif descriptor.lifetime == ServiceLifetime.SCOPED:
                raise RuntimeError(
                    "Scoped services must be resolved within a scope. Use create_scope()."
                )

            else:
                raise ValueError(f"Unknown lifetime: {descriptor.lifetime}")

    def _create_instance(self, descriptor: ServiceDescriptor) -> Any:
        """Create an instance from a descriptor"""
        # Check for circular dependencies
        if descriptor.interface in self._building:
            raise RuntimeError(
                f"Circular dependency detected for {descriptor.interface.__name__}"
            )

        self._building.add(descriptor.interface)

        try:
            if descriptor.instance:
                return descriptor.instance

            elif descriptor.factory:
                return descriptor.factory()

            elif descriptor.implementation:
                # Simple constructor injection - can be enhanced
                return descriptor.implementation()

            else:
                raise ValueError(
                    f"No way to create instance for {descriptor.interface.__name__}"
                )

        finally:
            self._building.discard(descriptor.interface)

    def create_scope(self) -> ServiceScope:
        """Create a new service scope"""
        return ServiceScope(self)

    def is_registered(self, interface: Type) -> bool:
        """Check if a service is registered"""
        return interface in self._descriptors

    def clear(self):
        """Clear all registrations (useful for testing)"""
        with self._lock:
            self._descriptors.clear()
            self._singletons.clear()
            self._building.clear()


# Global container instance
container = ServiceContainer()


# Decorator for automatic registration
def injectable(
    lifetime: str = ServiceLifetime.SINGLETON, interface: Optional[Type] = None
):
    """
    Decorator to mark a class as injectable

    Args:
        lifetime: Service lifetime
        interface: Optional interface to register as
    """

    def decorator(cls):
        # Register the class
        iface = interface or cls
        container.register(iface, implementation=cls, lifetime=lifetime)
        return cls

    return decorator


# Context manager for dependency injection
class DIContext:
    """Auto-generated class."""
    pass
    """Context manager for dependency injection operations"""

    def __init__(self, registrations: Optional[Dict[Type, Any]] = None):
        self.registrations = registrations or {}
        self.original_container = None

    def __enter__(self):
        # Save current container state
        self.original_container = container

        # Apply temporary registrations
        for interface, implementation in self.registrations.items():
            container.register(interface, instance=implementation)

        return container

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original container
        global container
        container = self.original_container


# Helper function for testing
def create_test_container() -> ServiceContainer:
    """Create a fresh container for testing"""
    return ServiceContainer()
