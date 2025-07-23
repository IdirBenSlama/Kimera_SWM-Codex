"""
System Validation Test
Tests the KIMERA system without mocks or simulations
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.config import get_settings, initialize_configuration
from src.main import app
from src.core.performance_integration import get_performance_manager
from src.layer_2_governance.monitoring.monitoring_integration import get_monitoring_manager
from src.security.security_integration import get_security_manager


async def validate_configuration():
    """Validate configuration system"""
    print("\n=== Validating Configuration System ===")
    try:
        # Initialize configuration
        initialize_configuration()
        settings = get_settings()
        
        print(f"✓ Configuration loaded successfully")
        print(f"  - Environment: {settings.environment}")
        print(f"  - Database URL: {settings.database.url}")
        print(f"  - Log Level: {settings.logging.level}")
        
        return True
    except Exception as e:
        print(f"✗ Configuration validation failed: {e}")
        return False


async def validate_security():
    """Validate security components"""
    print("\n=== Validating Security System ===")
    try:
        security_manager = get_security_manager()
        print(f"✓ Security manager initialized")
        
        # Check if authentication is configured
        from src.security.authentication import auth_manager
        test_password = "test_" + hashlib.sha256("test_password".encode()).hexdigest()[:16]
        hashed = auth_manager.get_password_hash(test_password)
        verified = auth_manager.verify_password(test_password, hashed)
        
        if verified:
            print(f"✓ Password hashing and verification working")
        else:
            print(f"✗ Password verification failed")
            return False
            
        return True
    except Exception as e:
        print(f"✗ Security validation failed: {e}")
        return False


async def validate_monitoring():
    """Validate monitoring components"""
    print("\n=== Validating Monitoring System ===")
    try:
        monitoring_manager = get_monitoring_manager()
        monitoring_manager.initialize()
        
        print(f"✓ Monitoring manager initialized")
        
        # Test logger
        logger = monitoring_manager.get_logger("test")
        logger.info("Test log message")
        print(f"✓ Structured logging working")
        
        # Test tracer
        tracer = monitoring_manager.get_tracer("test")
        with tracer.start_as_current_span("test_span") as span:
            span.set_attribute("test", "value")
        print(f"✓ Distributed tracing working")
        
        return True
    except Exception as e:
        print(f"✗ Monitoring validation failed: {e}")
        return False


async def validate_performance():
    """Validate performance optimization components"""
    print("\n=== Validating Performance System ===")
    try:
        perf_manager = await get_performance_manager()
        
        # Don't initialize the full system, just check components
        print(f"✓ Performance manager created")
        
        # Check cache manager
        cache_manager = perf_manager.cache_manager
        await cache_manager.set("test_key", "test_value")
        value = await cache_manager.get("test_key")
        
        if value == "test_value":
            print(f"✓ Cache layer working")
        else:
            print(f"✗ Cache layer failed")
            return False
            
        return True
    except Exception as e:
        print(f"✗ Performance validation failed: {e}")
        return False


async def validate_api():
    """Validate API endpoints"""
    print("\n=== Validating API Endpoints ===")
    try:
        from httpx import AsyncClient, ASGITransport
        
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            # Test root endpoint
            response = await client.get("/")
            if response.status_code == 200:
                print(f"✓ Root endpoint working")
            else:
                print(f"✗ Root endpoint failed: {response.status_code}")
                return False
                
            # Test health endpoint
            response = await client.get("/health")
            if response.status_code == 200:
                print(f"✓ Health endpoint working")
            else:
                print(f"✗ Health endpoint failed: {response.status_code}")
                return False
                
        return True
    except Exception as e:
        print(f"✗ API validation failed: {e}")
        return False


async def main():
    """Run all validation tests"""
    print("=" * 60)
    print("KIMERA System Validation")
    print("=" * 60)
    
    results = []
    
    # Run validation tests
    results.append(("Configuration", await validate_configuration()))
    results.append(("Security", await validate_security()))
    results.append(("Monitoring", await validate_monitoring()))
    results.append(("Performance", await validate_performance()))
    results.append(("API", await validate_api()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Validation Summary")
    print("=" * 60)
    
    all_passed = True
    for component, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{component:.<40} {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n✓ All validations PASSED - System is ready!")
    else:
        print("\n✗ Some validations FAILED - System needs fixes")
    
    return all_passed


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)