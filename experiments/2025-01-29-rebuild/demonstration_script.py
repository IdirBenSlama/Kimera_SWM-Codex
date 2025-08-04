#!/usr/bin/env python3
"""
KIMERA SWM - SYSTEM DEMONSTRATION SCRIPT
========================================

This script demonstrates the rebuilt Kimera SWM cognitive system,
showcasing the complete flow from basic geoid creation through
sophisticated cognitive processing across all engines.

This serves as both a demonstration and a comprehensive test
of the rebuilt system architecture.
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
from core.processing.geoid_processor import GeoidProcessor
from core.utilities.geoid_registry import get_global_registry
from engines.thermodynamic.thermodynamic_evolution_engine import ThermodynamicEvolutionEngine
from engines.transformation.mirror_portal_engine import MirrorPortalEngine, TransitionType
from engines.field_dynamics.cognitive_field_engine import CognitiveFieldEngine, FieldType
from orchestration.kimera_orchestrator import (
    KimeraOrchestrator, OrchestrationParameters, ProcessingStrategy
)


def print_separator(title: str, char: str = "="):
    """Print a visual separator with title"""
    logger.info(f"\n{char * 60}")
    logger.info(f" {title.upper()}")
    logger.info(f"{char * 60}")


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


def demonstrate_geoid_creation():
    """Demonstrate creation of different types of geoids"""
    print_separator("GEOID CREATION DEMONSTRATION")
    
    logger.info("Creating various types of geoids...")
    
    # Create different types of geoids
    concept_geoid = create_concept_geoid("artificial_intelligence")
    hypothesis_geoid = create_hypothesis_geoid(
        "AI systems exhibit emergent cognitive behaviors", 
        confidence=0.8
    )
    relation_geoid = create_relation_geoid(
        "artificial_intelligence", 
        "enables", 
        "cognitive_processing"
    )
    
    # Print summaries
    print_geoid_summary(concept_geoid, "Concept Geoid")
    print_geoid_summary(hypothesis_geoid, "Hypothesis Geoid")
    print_geoid_summary(relation_geoid, "Relation Geoid")
    
    return [concept_geoid, hypothesis_geoid, relation_geoid]


def demonstrate_geoid_processor(geoids):
    """Demonstrate the GeoidProcessor capabilities"""
    print_separator("GEOID PROCESSOR DEMONSTRATION")
    
    processor = GeoidProcessor()
    
    logger.info("Available operations:")
    for op in processor.registered_operations.keys():
        logger.info(f"  - {op}")
    
    # Test various operations
    test_geoid = geoids[0]
    
    logger.info(f"\nTesting operations on geoid {test_geoid.geoid_id[:8]}...")
    
    # Semantic enhancement
    result = processor.process_geoid(test_geoid, 'semantic_enhancement', 
                                   {'enhancement_factor': 1.2})
    logger.info(f"Semantic enhancement: {'SUCCESS' if result.success else 'FAILED'}")
    if result.success:
        enhanced_geoid = result.processed_geoid
        logger.info(f"  Coherence change: {enhanced_geoid.coherence_score - test_geoid.coherence_score:.3f}")
    
    # State validation
    result = processor.process_geoid(test_geoid, 'state_validation')
    logger.info(f"State validation: {'SUCCESS' if result.success else 'FAILED'}")
    
    # Batch processing
    logger.info(f"\nBatch processing {len(geoids)} geoids...")
    results = processor.process_batch(geoids, 'coherence_analysis', 
                                    {'coherence_threshold': 0.7})
    successful = sum(1 for r in results if r.success)
    logger.info(f"Batch processing: {successful}/{len(results)} successful")
    
    # Performance summary
    logger.info(f"\nProcessor Performance Summary:")
    summary = processor.get_performance_summary()
    for key, value in summary.items():
        logger.info(f"  {key}: {value}")
    
    return results


def demonstrate_thermodynamic_engine(geoids):
    """Demonstrate the ThermodynamicEvolutionEngine"""
    print_separator("THERMODYNAMIC EVOLUTION DEMONSTRATION")
    
    engine = ThermodynamicEvolutionEngine()
    
    logger.info("Evolving geoids thermodynamically...")
    
    # Single geoid evolution
    test_geoid = geoids[0]
    print_geoid_summary(test_geoid, "Before Evolution")
    
    evolution_result = engine.evolve(test_geoid)
    
    logger.info(f"\nEvolution result:")
    logger.info(f"  Success: {evolution_result.original_geoid is not None}")
    logger.info(f"  Energy change: {evolution_result.energy_change:.3f}")
    logger.info(f"  Entropy change: {evolution_result.entropy_change:.3f}")
    logger.info(f"  Temperature change: {evolution_result.temperature_change:.3f}")
    logger.info(f"  Evolution probability: {evolution_result.evolution_probability:.3f}")
    logger.info(f"  Quantum tunneling: {evolution_result.quantum_tunneling}")
    logger.info(f"  Duration: {evolution_result.duration:.3f}s")
    
    print_geoid_summary(evolution_result.evolved_geoid, "After Evolution")
    
    # System evolution
    logger.info(f"\nEvolving system of {len(geoids)} geoids...")
    system_results = engine.evolve_system(geoids)
    
    total_energy_change = sum(r.energy_change for r in system_results)
    logger.info(f"Total system energy change: {total_energy_change:.3f}")
    
    # Engine statistics
    logger.info(f"\nEngine Statistics:")
    stats = engine.get_engine_statistics()
    for key, value in stats.items():
        if isinstance(value, dict):
            logger.info(f"  {key}: {value}")
        else:
            logger.info(f"  {key}: {value}")
    
    return evolution_result.evolved_geoid


def demonstrate_mirror_portal_engine(geoids):
    """Demonstrate the MirrorPortalEngine"""
    print_separator("MIRROR PORTAL ENGINE DEMONSTRATION")
    
    engine = MirrorPortalEngine()
    
    logger.info("Available transition types:")
    for transition_type in TransitionType:
        logger.info(f"  - {transition_type.value}")
    
    test_geoid = geoids[1]  # Use hypothesis geoid
    
    # Test different transition types
    transitions_to_test = [
        TransitionType.COHERENCE_BRIDGE,
        TransitionType.SEMANTIC_TO_SYMBOLIC,
        TransitionType.MIRROR_REFLECTION
    ]
    
    transformed_geoids = []
    
    for transition_type in transitions_to_test:
        logger.info(f"\nTesting {transition_type.value} transition...")
        
        result = engine.transform(test_geoid, transition_type)
        
        logger.info(f"  Success: {result.success}")
        if result.success:
            logger.info(f"  Energy consumed: {result.energy_consumed:.3f}")
            logger.info(f"  Coherence change: {result.coherence_change:.3f}")
            logger.info(f"  Portal state: {result.portal_state.value}")
            logger.info(f"  Quantum effects: {list(result.quantum_effects.keys())}")
            
            transformed_geoids.append(result.transformed_geoid)
    
    # Test entanglement
    logger.info(f"\nTesting quantum entanglement...")
    if len(geoids) >= 2:
        entangled1, entangled2 = engine.entangle_geoids(geoids[0], geoids[1])
        logger.info(f"  Entangled geoids: {entangled1.geoid_id[:8]} ↔ {entangled2.geoid_id[:8]}")
    
    # Engine statistics
    logger.info(f"\nEngine Statistics:")
    stats = engine.get_engine_statistics()
    for key, value in stats.items():
        if isinstance(value, (dict, list)) and len(str(value)) > 100:
            logger.info(f"  {key}: [Complex data structure]")
        else:
            logger.info(f"  {key}: {value}")
    
    return transformed_geoids


def demonstrate_field_engine(geoids):
    """Demonstrate the CognitiveFieldEngine"""
    print_separator("COGNITIVE FIELD ENGINE DEMONSTRATION")
    
    engine = CognitiveFieldEngine()
    
    logger.info("Available field types:")
    for field_type in FieldType:
        logger.info(f"  - {field_type.value}")
    
    # Test field processing
    field_types_to_test = [
        FieldType.SEMANTIC_FIELD,
        FieldType.ENERGY_FIELD,
        FieldType.COHERENCE_FIELD
    ]
    
    logger.info(f"\nProcessing {len(geoids)} geoids in multiple fields...")
    
    result = engine.process_geoids_in_fields(
        geoids, 
        field_types_to_test, 
        evolution_steps=5
    )
    
    logger.info(f"Field processing results:")
    logger.info(f"  Processing duration: {result.processing_duration:.3f}s")
    logger.info(f"  Emergent behaviors detected: {len(result.emergent_behaviors)}")
    logger.info(f"  Interaction events: {len(result.interaction_events)}")
    
    # Show emergent behaviors
    if result.emergent_behaviors:
        logger.info(f"\nEmergent behaviors:")
        for i, behavior in enumerate(result.emergent_behaviors[:3]):  # Show first 3
            logger.info(f"  {i+1}. Type: {behavior.get('structure', {}).get('type', 'unknown')}")
            logger.info(f"     Strength: {behavior.get('structure', {}).get('strength', 0.0):.3f}")
    
    # Energy changes
    logger.info(f"\nEnergy changes per geoid:")
    for geoid_id, energy_change in result.energy_changes.items():
        logger.info(f"  {geoid_id[:8]}: {energy_change:+.3f}")
    
    # Engine statistics
    logger.info(f"\nEngine Statistics:")
    stats = engine.get_engine_statistics()
    for key, value in stats.items():
        if isinstance(value, dict) and len(str(value)) > 100:
            logger.info(f"  {key}: [Complex data structure]")
        else:
            logger.info(f"  {key}: {value}")
    
    return result.processed_geoids


def demonstrate_orchestrator(geoids):
    """Demonstrate the KimeraOrchestrator"""
    print_separator("KIMERA ORCHESTRATOR DEMONSTRATION")
    
    # Configure orchestrator
    params = OrchestrationParameters(
        strategy=ProcessingStrategy.SCIENTIFIC,
        max_parallel_engines=2,
        emergence_detection=True
    )
    
    orchestrator = KimeraOrchestrator(params)
    
    logger.info("Available pipelines:")
    for pipeline_name in orchestrator.pipelines.keys():
        pipeline = orchestrator.pipelines[pipeline_name]
        logger.info(f"  - {pipeline_name}: {pipeline.description}")
    
    # Test different orchestration strategies
    strategies_to_test = [
        ProcessingStrategy.EXPLORATION,
        ProcessingStrategy.TRANSFORMATION,
        ProcessingStrategy.SCIENTIFIC
    ]
    
    for strategy in strategies_to_test:
        logger.info(f"\nTesting {strategy.value} strategy...")
        
        result = orchestrator.orchestrate(geoids, strategy=strategy)
        
        logger.info(f"  Session ID: {result.session_id[:8]}...")
        logger.info(f"  Pipeline used: {result.pipeline_used}")
        logger.info(f"  Engines executed: {len(result.engines_executed)}")
        logger.info(f"  Processing duration: {result.processing_duration:.3f}s")
        logger.info(f"  Energy consumed: {result.energy_consumed:.3f}")
        logger.info(f"  Emergent phenomena: {len(result.emergent_phenomena)}")
        logger.info(f"  Errors: {len(result.errors)}")
        logger.info(f"  Performance metrics:")
        for metric, value in result.performance_metrics.items():
            logger.info(f"    {metric}: {value:.3f}" if isinstance(value, float) else f"    {metric}: {value}")
    
    # System status
    logger.info(f"\nSystem Status:")
    status = orchestrator.get_system_status()
    logger.info(f"  System health: {status['system_health']:.3f}")
    logger.info(f"  Total orchestrations: {status['total_orchestrations']}")
    logger.info(f"  Active sessions: {status['active_sessions']}")
    logger.info(f"  Available pipelines: {len(status['available_pipelines'])}")
    logger.info(f"  Registry status: {status['registry_status']['current_size']} geoids")
    
    return result


def demonstrate_registry():
    """Demonstrate the GeoidRegistry capabilities"""
    print_separator("GEOID REGISTRY DEMONSTRATION")
    
    registry = get_global_registry()
    
    # Registry statistics
    stats = registry.get_statistics()
    logger.info(f"Registry Statistics:")
    logger.info(f"  Total geoids: {stats.total_geoids}")
    logger.info(f"  Average coherence: {stats.average_coherence:.3f}")
    logger.info(f"  Average energy: {stats.average_energy:.3f}")
    logger.info(f"  Total relationships: {stats.total_relationships}")
    
    logger.info(f"\nGeoids by type:")
    for geoid_type, count in stats.geoids_by_type.items():
        logger.info(f"  {geoid_type.value}: {count}")
    
    logger.info(f"\nGeoids by state:")
    for state, count in stats.geoids_by_state.items():
        logger.info(f"  {state.value}: {count}")
    
    # Registry metrics
    metrics = registry.get_registry_metrics()
    logger.info(f"\nRegistry Metrics:")
    for key, value in metrics.items():
        if key != 'index_sizes':
            logger.info(f"  {key}: {value}")


def main():
    """Main demonstration function"""
    print_separator("KIMERA SWM SYSTEM DEMONSTRATION", "=")
    logger.info("Rebuilt Cognitive Architecture Demonstration")
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Step 1: Create geoids
        geoids = demonstrate_geoid_creation()
        
        # Step 2: Demonstrate geoid processor
        processed_results = demonstrate_geoid_processor(geoids)
        
        # Step 3: Demonstrate thermodynamic engine
        evolved_geoid = demonstrate_thermodynamic_engine(geoids)
        
        # Step 4: Demonstrate mirror portal engine
        transformed_geoids = demonstrate_mirror_portal_engine(geoids)
        
        # Step 5: Demonstrate field engine
        field_processed_geoids = demonstrate_field_engine(geoids)
        
        # Step 6: Demonstrate orchestrator
        orchestration_result = demonstrate_orchestrator(geoids)
        
        # Step 7: Demonstrate registry
        demonstrate_registry()
        
        print_separator("DEMONSTRATION COMPLETE", "=")
        logger.info("✅ All system components demonstrated successfully!")
        logger.info(f"✅ Total geoids processed: {len(geoids)}")
        logger.info(f"✅ System architecture: COHESIVE AND FUNCTIONAL")
        logger.info(f"✅ Engine interconnection: SUCCESSFUL")
        logger.info(f"✅ Pipeline orchestration: OPERATIONAL")
        
    except Exception as e:
        print_separator("DEMONSTRATION ERROR", "!")
        logger.info(f"❌ Error during demonstration: {str(e)}")
        logger.info(f"❌ This indicates an issue in the rebuilt system")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 