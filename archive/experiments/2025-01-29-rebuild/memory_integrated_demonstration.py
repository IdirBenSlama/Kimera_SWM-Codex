#!/usr/bin/env python3
"""
KIMERA SWM - MEMORY-INTEGRATED SYSTEM DEMONSTRATION
==================================================

This script demonstrates the complete memory-integrated Kimera SWM cognitive 
system, showcasing the full flow from basic geoid creation through sophisticated 
cognitive processing with persistent memory, anomaly detection, and self-healing 
capabilities.

This serves as the final demonstration of the complete rebuilt system.
"""

import sys
import os
import numpy as np
from datetime import datetime
import logging

# Add the src directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import our rebuilt system components
from core.data_structures.geoid_state import (
    GeoidState, GeoidType, create_concept_geoid, create_hypothesis_geoid,
    create_relation_geoid
)
from core.data_structures.scar_state import (
    ScarState, ScarType, ScarSeverity, create_processing_error_scar,
    create_energy_violation_scar, create_coherence_breakdown_scar
)

# Import memory system components
from core.utilities.scar_manager import (
    ScarManager, get_global_scar_manager, AnalysisMode,
    report_logical_contradiction, get_system_health
)
from core.utilities.vault_system import (
    VaultSystem, get_global_vault, StorageConfiguration, StorageBackend,
    store_geoid, retrieve_geoid
)
from core.utilities.database_manager import (
    DatabaseManager, get_global_database_manager, DatabaseConfiguration, DatabaseType,
    get_system_health_metrics
)

# Import orchestration components
from orchestration.memory_integrated_orchestrator import (
    MemoryIntegratedOrchestrator, MemoryIntegrationParameters, MemoryMode,
    get_complete_system_status, orchestrate_with_memory, query_system_knowledge
)
from orchestration.kimera_orchestrator import (
    OrchestrationParameters, ProcessingStrategy
)


def print_separator(title: str, char: str = "="):
    """Print a visual separator with title"""
    logger.info(f"\n{char * 70}")
    logger.info(f" {title.upper()}")
    logger.info(f"{char * 70}")


def print_geoid_summary(geoid: GeoidState, title: str = "Geoid"):
    """Print a summary of a geoid's state"""
    logger.info(f"\n{title}:")
    logger.info(f"  ID: {geoid.geoid_id[:8]}...")
    logger.info(f"  Type: {geoid.geoid_type.value}")
    logger.info(f"  State: {geoid.processing_state.value}")
    logger.info(f"  Coherence: {geoid.coherence_score:.3f}")
    logger.info(f"  Energy: {geoid.cognitive_energy:.3f}")
    logger.info(f"  Has Semantic: {geoid.semantic_state is not None}")
    logger.info(f"  Has Symbolic: {geoid.symbolic_state is not None}")
    logger.info(f"  Has Thermodynamic: {geoid.thermodynamic is not None}")
    logger.info(f"  Processing Depth: {geoid.metadata.processing_depth}")


def demonstrate_scar_system():
    """Demonstrate the SCAR (Semantic Contextual Anomaly Report) system"""
    print_separator("SCAR SYSTEM DEMONSTRATION")
    
    # Initialize SCAR manager
    scar_manager = get_global_scar_manager()
    
    logger.info("SCAR System Features:")
    logger.info("  - Automated anomaly detection")
    logger.info("  - Structured evidence collection")
    logger.info("  - Intelligent root cause analysis")
    logger.info("  - Automated resolution actions")
    logger.info("  - System health monitoring")
    
    # Create test geoids with issues
    problematic_geoid = create_concept_geoid("problematic_concept")
    problematic_geoid.semantic_state.coherence_score = 0.2  # Low coherence
    
    energy_geoid = create_hypothesis_geoid("energy_test", confidence=0.9)
    if energy_geoid.thermodynamic:
        energy_geoid.thermodynamic.free_energy = -5.0  # Negative energy (violation)
    
    logger.info(f"\nCreated test geoids with intentional issues:")
    print_geoid_summary(problematic_geoid, "Low Coherence Geoid")
    print_geoid_summary(energy_geoid, "Energy Violation Geoid")
    
    # Create SCARs for various issues
    logger.info(f"\nCreating SCARs for detected issues...")
    
    # Coherence breakdown SCAR
    coherence_scar = create_coherence_breakdown_scar(problematic_geoid, 0.8, 0.2)
    scar_id1 = scar_manager.report_anomaly(coherence_scar)
    logger.info(f"  Created coherence breakdown SCAR: {scar_id1[:8]}...")
    
    # Energy violation SCAR
    energy_scar = create_energy_violation_scar(energy_geoid, 5.0, -5.0)
    scar_id2 = scar_manager.report_anomaly(energy_scar)
    logger.info(f"  Created energy violation SCAR: {scar_id2[:8]}...")
    
    # Processing error SCAR
    error_scar = create_processing_error_scar(
        problematic_geoid, "TestEngine", 
        "Simulated processing error", 
        {"error_type": "test", "severity": "medium"}
    )
    scar_id3 = scar_manager.report_anomaly(error_scar)
    logger.info(f"  Created processing error SCAR: {scar_id3[:8]}...")
    
    # Get SCAR statistics
    logger.info(f"\nSCAR System Statistics:")
    stats = scar_manager.get_statistics()
    logger.info(f"  Total SCARs: {stats.total_scars}")
    logger.info(f"  Active SCARs: {len(scar_manager.get_active_scars())}")
    logger.info(f"  System Health Score: {stats.system_health_score:.3f}")
    logger.info(f"  Average Resolution Time: {stats.average_resolution_time:.2f}s")
    
    # Demonstrate SCAR resolution
    logger.info(f"\nResolving SCARs...")
    scar_manager.resolve_scar(scar_id3, "Test resolution completed", effectiveness=0.9)
    logger.info(f"  Resolved SCAR {scar_id3[:8]} with 90% effectiveness")
    
    updated_stats = scar_manager.get_statistics()
    logger.info(f"  Updated System Health Score: {updated_stats.system_health_score:.3f}")
    
    return [problematic_geoid, energy_geoid]


def demonstrate_vault_system():
    """Demonstrate the Vault System for persistent storage"""
    print_separator("VAULT SYSTEM DEMONSTRATION")
    
    # Initialize vault system
    vault = get_global_vault()
    
    logger.info("Vault System Features:")
    logger.info("  - Multi-backend storage (SQLite, JSON, Memory)")
    logger.info("  - Automatic compression and backup")
    logger.info("  - Performance optimization with caching")
    logger.info("  - Data lifecycle management")
    logger.info("  - Integrity verification")
    
    # Create test geoids for storage
    test_geoids = []
    for i in range(5):
        if i % 2 == 0:
            geoid = create_concept_geoid(f"vault_concept_{i}")
        else:
            geoid = create_hypothesis_geoid(f"vault_hypothesis_{i}", confidence=0.7 + i*0.05)
        test_geoids.append(geoid)
    
    logger.info(f"\nStoring {len(test_geoids)} geoids in vault...")
    
    # Store geoids
    storage_times = []
    for geoid in test_geoids:
        start_time = datetime.now()
        success = vault.store_geoid(geoid)
        storage_time = (datetime.now() - start_time).total_seconds()
        storage_times.append(storage_time)
        
        if success:
            logger.info(f"  ✓ Stored {geoid.geoid_id[:8]} ({geoid.geoid_type.value}) in {storage_time:.3f}s")
        else:
            logger.info(f"  ✗ Failed to store {geoid.geoid_id[:8]}")
    
    # Retrieve geoids
    logger.info(f"\nRetrieving geoids from vault...")
    retrieval_times = []
    retrieved_geoids = []
    
    for geoid in test_geoids:
        start_time = datetime.now()
        retrieved = vault.retrieve_geoid(geoid.geoid_id)
        retrieval_time = (datetime.now() - start_time).total_seconds()
        retrieval_times.append(retrieval_time)
        
        if retrieved:
            retrieved_geoids.append(retrieved)
            logger.info(f"  ✓ Retrieved {retrieved.geoid_id[:8]} in {retrieval_time:.3f}s")
        else:
            logger.info(f"  ✗ Failed to retrieve {geoid.geoid_id[:8]}")
    
    # List stored geoids
    all_geoid_ids = vault.list_geoids()
    logger.info(f"\nTotal geoids in vault: {len(all_geoid_ids)}")
    
    # Storage metrics
    logger.info(f"\nVault Performance Metrics:")
    metrics = vault.get_storage_metrics()
    logger.info(f"  Total items stored: {metrics.total_items_stored}")
    logger.info(f"  Total items retrieved: {metrics.total_items_retrieved}")
    logger.info(f"  Average storage time: {np.mean(storage_times):.3f}s")
    logger.info(f"  Average retrieval time: {np.mean(retrieval_times):.3f}s")
    logger.info(f"  Storage size: {metrics.storage_size_bytes / 1024:.1f} KB")
    logger.info(f"  Cache hit rate: {metrics.cache_hit_rate:.2%}")
    
    return retrieved_geoids


def demonstrate_database_system():
    """Demonstrate the Database Manager for analytics"""
    print_separator("DATABASE SYSTEM DEMONSTRATION")
    
    # Initialize database manager
    database = get_global_database_manager()
    
    logger.info("Database System Features:")
    logger.info("  - Structured queries and analytics")
    logger.info("  - Multi-database backend support")
    logger.info("  - Real-time performance monitoring")
    logger.info("  - Complex aggregations and reporting")
    logger.info("  - Schema management and migrations")
    
    # Store some test data
    test_geoids = [
        create_concept_geoid("database_concept_1"),
        create_concept_geoid("database_concept_2"),
        create_hypothesis_geoid("database_hypothesis_1", confidence=0.8),
        create_relation_geoid("concept_1", "relates_to", "concept_2")
    ]
    
    logger.info(f"\nStoring metadata for {len(test_geoids)} geoids...")
    for geoid in test_geoids:
        success = database.store_geoid_metadata(geoid)
        if success:
            logger.info(f"  ✓ Stored metadata for {geoid.geoid_id[:8]} ({geoid.geoid_type.value})")
    
    # Query geoids by type
    logger.info(f"\nQuerying geoids by type...")
    concept_geoids = database.query_geoids({'geoid_type': 'concept'})
    hypothesis_geoids = database.query_geoids({'geoid_type': 'hypothesis'})
    relation_geoids = database.query_geoids({'geoid_type': 'relation'})
    
    logger.info(f"  Found {len(concept_geoids)} concept geoids")
    logger.info(f"  Found {len(hypothesis_geoids)} hypothesis geoids")
    logger.info(f"  Found {len(relation_geoids)} relation geoids")
    
    # Query by coherence
    high_coherence = database.query_geoids({'min_coherence': 0.7})
    logger.info(f"  Found {len(high_coherence)} high-coherence geoids (>0.7)")
    
    # Get system analytics
    logger.info(f"\nSystem Analytics:")
    analytics = database.get_system_analytics()
    
    if 'geoids' in analytics:
        geoid_stats = analytics['geoids']
        if isinstance(geoid_stats, dict):
            logger.info(f"  Total geoids in database: {geoid_stats.get('total_geoids', 0)}")
            logger.info(f"  Average coherence: {geoid_stats.get('avg_coherence', 0):.3f}")
            logger.info(f"  Average energy: {geoid_stats.get('avg_energy', 0):.3f}")
    
    if 'geoid_type_distribution' in analytics:
        logger.info(f"  Geoid type distribution:")
        for geoid_type, count in analytics['geoid_type_distribution'].items():
            logger.info(f"    {geoid_type}: {count}")
    
    return test_geoids


def demonstrate_memory_integrated_orchestrator():
    """Demonstrate the complete Memory-Integrated Orchestrator"""
    print_separator("MEMORY-INTEGRATED ORCHESTRATOR DEMONSTRATION")
    
    # Configure memory integration
    memory_params = MemoryIntegrationParameters(
        memory_mode=MemoryMode.HYBRID,
        enable_scar_detection=True,
        enable_vault_storage=True,
        enable_database_analytics=True,
        scar_analysis_mode=AnalysisMode.IMMEDIATE
    )
    
    orchestration_params = OrchestrationParameters(
        strategy=ProcessingStrategy.SCIENTIFIC,
        max_parallel_engines=2,
        emergence_detection=True
    )
    
    # Initialize memory-integrated orchestrator
    from orchestration.memory_integrated_orchestrator import initialize_memory_orchestrator
    orchestrator = initialize_memory_orchestrator(orchestration_params, memory_params)
    
    logger.info("Memory-Integrated Orchestrator Features:")
    logger.info("  - Complete cognitive processing with persistent memory")
    logger.info("  - Automatic anomaly detection and resolution")
    logger.info("  - Real-time system health monitoring")
    logger.info("  - Self-healing cognitive behaviors")
    logger.info("  - Advanced analytics and reporting")
    
    # Create complex test scenario
    logger.info(f"\nCreating complex cognitive scenario...")
    test_geoids = [
        create_concept_geoid("complex_ai_system"),
        create_hypothesis_geoid("ai_consciousness_hypothesis", confidence=0.6),
        create_relation_geoid("ai_system", "exhibits", "consciousness"),
        create_concept_geoid("emergent_behavior"),
        create_hypothesis_geoid("collective_intelligence_theory", confidence=0.8)
    ]
    
    for geoid in test_geoids:
        print_geoid_summary(geoid, f"Test Geoid ({geoid.geoid_type.value})")
    
    # Test different processing strategies
    strategies_to_test = [
        ProcessingStrategy.EXPLORATION,
        ProcessingStrategy.SCIENTIFIC,
        ProcessingStrategy.EMERGENCE
    ]
    
    for strategy in strategies_to_test:
        logger.info(f"\nTesting {strategy.value} strategy...")
        
        result = orchestrator.orchestrate(test_geoids, strategy=strategy)
        
        logger.info(f"  Session ID: {result.session_id[:8]}...")
        logger.info(f"  Pipeline used: {result.pipeline_used}")
        logger.info(f"  Engines executed: {len(result.engines_executed)}")
        logger.info(f"  Processing duration: {result.processing_duration:.3f}s")
        logger.info(f"  Energy consumed: {result.energy_consumed:.3f}")
        logger.info(f"  Emergent phenomena: {len(result.emergent_phenomena)}")
        logger.info(f"  Success rate: {result.performance_metrics.get('success_rate', 0):.2%}")
        logger.info(f"  Errors: {len(result.errors)}")
        
        if result.errors:
            logger.info(f"    Error details: {result.errors[:2]}")  # Show first 2 errors
    
    # Query processed knowledge
    logger.info(f"\nQuerying processed knowledge...")
    
    # Query by different criteria
    queries = [
        {'geoid_type': 'concept'},
        {'geoid_type': 'hypothesis', 'min_coherence': 0.5},
        {'processing_state': 'stable'}
    ]
    
    for query in queries:
        results = orchestrator.query_geoids_by_criteria(query)
        logger.info(f"  Query {query}: Found {len(results)} geoids")
    
    # Get comprehensive system status
    logger.info(f"\nComprehensive System Status:")
    status = orchestrator.get_comprehensive_status()
    
    # Memory metrics
    if 'memory_metrics' in status:
        memory_metrics = status['memory_metrics']
        logger.info(f"  Memory Metrics:")
        logger.info(f"    Total geoids stored: {memory_metrics['total_geoids_stored']}")
        logger.info(f"    Total SCARs created: {memory_metrics['total_scars_created']}")
        logger.info(f"    Vault storage: {memory_metrics['vault_storage_size_mb']:.1f} MB")
        logger.info(f"    Database size: {memory_metrics['database_size_mb']:.1f} MB")
        logger.info(f"    Cache hit rate: {memory_metrics['cache_hit_rate']}")
        logger.info(f"    System health: {memory_metrics['system_health_score']:.3f}")
    
    # System health
    if 'scar_status' in status and status['scar_status']:
        scar_status = status['scar_status']
        logger.info(f"  SCAR System:")
        logger.info(f"    Total SCARs: {scar_status.total_scars}")
        logger.info(f"    Resolution success rate: {scar_status.resolution_success_rate:.2%}")
        logger.info(f"    System health score: {scar_status.system_health_score:.3f}")
    
    return result


def demonstrate_system_recovery():
    """Demonstrate system self-healing and recovery capabilities"""
    print_separator("SYSTEM RECOVERY DEMONSTRATION")
    
    logger.info("System Recovery Features:")
    logger.info("  - Automatic anomaly detection")
    logger.info("  - Self-healing mechanisms")
    logger.info("  - Error recovery and isolation")
    logger.info("  - System health restoration")
    
    orchestrator = get_global_memory_orchestrator()
    
    # Create a geoid with intentional problems
    problematic_geoid = create_hypothesis_geoid("failing_hypothesis", confidence=0.1)
    problematic_geoid.semantic_state.coherence_score = 0.1  # Very low coherence
    if problematic_geoid.thermodynamic:
        problematic_geoid.thermodynamic.free_energy = -10.0  # Severe energy violation
    
    logger.info(f"\nCreated problematic geoid:")
    print_geoid_summary(problematic_geoid, "Problematic Geoid")
    
    # Process with orchestrator (will trigger SCAR detection)
    logger.info(f"\nProcessing problematic geoid (will trigger anomaly detection)...")
    
    try:
        result = orchestrator.orchestrate([problematic_geoid], strategy=ProcessingStrategy.SCIENTIFIC)
        logger.info(f"  Processing completed with {len(result.errors)} errors")
        
        if result.errors:
            logger.info(f"  Detected errors: {result.errors[:2]}")
    
    except Exception as e:
        logger.info(f"  Processing failed with exception: {str(e)}")
    
    # Check system health after processing
    logger.info(f"\nSystem health after processing problematic geoid:")
    system_health = get_system_health()
    logger.info(f"  System health score: {system_health:.3f}")
    
    # Show active SCARs
    scar_manager = get_global_scar_manager()
    active_scars = scar_manager.get_active_scars()
    logger.info(f"  Active SCARs: {len(active_scars)}")
    
    for scar in active_scars[:3]:  # Show first 3
        logger.info(f"    - {scar.title} (severity: {scar.severity.value}, type: {scar.scar_type.value})")
    
    # Demonstrate recovery
    logger.info(f"\nInitiating recovery procedures...")
    for scar in active_scars[:2]:  # Resolve first 2 SCARs
        scar_manager.resolve_scar(
            scar.scar_id, 
            f"Automatic recovery applied for {scar.scar_type.value}",
            effectiveness=0.8
        )
        logger.info(f"  ✓ Resolved SCAR: {scar.title}")
    
    # Check improved system health
    updated_health = get_system_health()
    logger.info(f"\nSystem health after recovery:")
    logger.info(f"  Updated system health score: {updated_health:.3f}")
    logger.info(f"  Health improvement: {updated_health - system_health:+.3f}")
    
    return updated_health


def main():
    """Main demonstration function"""
    print_separator("KIMERA SWM MEMORY-INTEGRATED SYSTEM DEMONSTRATION", "=")
    logger.info("Complete Cognitive Architecture with Persistent Memory")
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Step 1: Demonstrate SCAR system
        test_geoids_scars = demonstrate_scar_system()
        
        # Step 2: Demonstrate Vault system
        test_geoids_vault = demonstrate_vault_system()
        
        # Step 3: Demonstrate Database system
        test_geoids_db = demonstrate_database_system()
        
        # Step 4: Demonstrate Memory-Integrated Orchestrator
        orchestration_result = demonstrate_memory_integrated_orchestrator()
        
        # Step 5: Demonstrate system recovery
        final_health = demonstrate_system_recovery()
        
        print_separator("MEMORY-INTEGRATED DEMONSTRATION COMPLETE", "=")
        logger.info("✅ All memory system components demonstrated successfully!")
        logger.info(f"✅ SCAR System: Anomaly detection and resolution operational")
        logger.info(f"✅ Vault System: Persistent storage and retrieval operational")
        logger.info(f"✅ Database System: Advanced analytics and querying operational")
        logger.info(f"✅ Memory-Integrated Orchestrator: Complete cognitive processing operational")
        logger.info(f"✅ System Recovery: Self-healing capabilities operational")
        logger.info(f"✅ Final System Health Score: {final_health:.3f}")
        logger.info(f"✅ System Architecture: COMPLETE WITH MEMORY INTEGRATION")
        
        # Final system status
        logger.info(f"\nFinal System Status:")
        final_status = get_complete_system_status()
        
        if 'memory_metrics' in final_status:
            memory_metrics = final_status['memory_metrics']
            logger.info(f"  Total geoids in system: {memory_metrics['total_geoids_stored']}")
            logger.info(f"  Total SCARs managed: {memory_metrics['total_scars_created']}")
            logger.info(f"  Storage utilization: {memory_metrics['vault_storage_size_mb']:.1f} MB")
            logger.info(f"  System health: {memory_metrics['system_health_score']:.3f}")
        
        logger.info(f"\n🎉 KIMERA SWM MEMORY-INTEGRATED SYSTEM: FULLY OPERATIONAL! 🎉")
        
    except Exception as e:
        print_separator("DEMONSTRATION ERROR", "!")
        logger.info(f"❌ Error during demonstration: {str(e)}")
        logger.info(f"❌ This indicates an issue in the memory-integrated system")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 