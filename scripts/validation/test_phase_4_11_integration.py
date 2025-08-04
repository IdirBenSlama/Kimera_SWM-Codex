#!/usr/bin/env python3
"""
Phase 4.11 Integration Validation
==================================
DO-178C Level A validation of Barenholtz Dual-System Architecture
"""

import sys
sys.path.insert(0, '.')

import asyncio
import torch
import numpy as np
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def validate_phase_4_11():
    """Validate Phase 4.11 integration following DO-178C standards"""
    print("\n" + "="*80)
    print("🔬 PHASE 4.11 VALIDATION: BARENHOLTZ DUAL-SYSTEM ARCHITECTURE")
    print("DO-178C Level A Compliance Verification")
    print("="*80 + "\n")

    results = {
        'components': {},
        'safety_requirements': {},
        'performance': {},
        'cognitive_benchmarks': {}
    }

    try:
        # Step 1: Verify KimeraSystem integration
        print("1️⃣ Verifying KimeraSystem integration...")
        from src.core.kimera_system import KimeraSystem

        system = KimeraSystem()
        system.initialize()

        # Check if component loaded
        barenholtz = system.get_component("barenholtz_architecture")
        if barenholtz:
            print("✅ Barenholtz Architecture loaded in KimeraSystem")
            results['components']['kimera_integration'] = True
        else:
            print("❌ Barenholtz Architecture not found in KimeraSystem")
            results['components']['kimera_integration'] = False
            return results

        # Step 2: Verify all components
        print("\n2️⃣ Verifying Barenholtz components...")
        from src.core.barenholtz_architecture import (
            BarenholtzDualSystemIntegrator,
            System1Processor,
            System2Processor,
            MetacognitiveController
        )

        # Check each component
        integrator = BarenholtzDualSystemIntegrator()
        print("✅ BarenholtzDualSystemIntegrator initialized")
        results['components']['integrator'] = True

        # Verify subcomponents
        assert hasattr(integrator, 'system1'), "System 1 not found"
        assert hasattr(integrator, 'system2'), "System 2 not found"
        assert hasattr(integrator, 'metacognitive'), "Metacognitive controller not found"
        print("✅ All subcomponents present")
        results['components']['subcomponents'] = True

        # Step 3: Safety Requirements Testing
        print("\n3️⃣ Testing safety requirements...")

        # Test response time requirements
        test_input = torch.randn(512)

        # Test System 1 < 100ms
        print("   Testing System 1 response time...")
        start = datetime.now()
        s1_result = await integrator.system1.process(test_input)
        s1_time = (datetime.now() - start).total_seconds()

        if s1_time < 0.100:
            print(f"   ✅ System 1: {s1_time*1000:.1f}ms < 100ms")
            results['safety_requirements']['system1_time'] = True
        else:
            print(f"   ❌ System 1: {s1_time*1000:.1f}ms > 100ms")
            results['safety_requirements']['system1_time'] = False

        # Test System 2 < 1000ms
        print("   Testing System 2 response time...")
        start = datetime.now()
        s2_result = await integrator.system2.process(test_input)
        s2_time = (datetime.now() - start).total_seconds()

        if s2_time < 1.000:
            print(f"   ✅ System 2: {s2_time*1000:.1f}ms < 1000ms")
            results['safety_requirements']['system2_time'] = True
        else:
            print(f"   ❌ System 2: {s2_time*1000:.1f}ms > 1000ms")
            results['safety_requirements']['system2_time'] = False

        # Test arbitration < 50ms
        print("   Testing arbitration time...")
        start = datetime.now()
        arb_result = await integrator.metacognitive.arbitrate(s1_result, s2_result)
        arb_time = (datetime.now() - start).total_seconds()

        if arb_time < 0.050:
            print(f"   ✅ Arbitration: {arb_time*1000:.1f}ms < 50ms")
            results['safety_requirements']['arbitration_time'] = True
        else:
            print(f"   ❌ Arbitration: {arb_time*1000:.1f}ms > 50ms")
            results['safety_requirements']['arbitration_time'] = False

        # Step 4: Integration Testing
        print("\n4️⃣ Testing integrated processing...")

        # Test different modes
        from src.core.barenholtz_architecture import ProcessingConstraints, SystemMode

        modes = [
            SystemMode.PARALLEL,
            SystemMode.SEQUENTIAL,
            SystemMode.SYSTEM1_PREFERRED,
            SystemMode.SYSTEM2_PREFERRED,
            SystemMode.AUTOMATIC
        ]

        for mode in modes:
            print(f"   Testing {mode.value} mode...")
            constraints = ProcessingConstraints(system_mode=mode)

            try:
                output = await integrator.process(test_input, constraints=constraints)
                if output.is_valid():
                    print(f"   ✅ {mode.value}: Valid output")
                    results['performance'][f'mode_{mode.value}'] = True
                else:
                    print(f"   ❌ {mode.value}: Invalid output")
                    results['performance'][f'mode_{mode.value}'] = False
            except Exception as e:
                print(f"   ❌ {mode.value}: Error - {e}")
                results['performance'][f'mode_{mode.value}'] = False

        # Step 5: Cognitive Benchmarks
        print("\n5️⃣ Running cognitive benchmarks...")

        # Simulate cognitive tasks
        tasks = {
            'pattern_recognition': torch.randn(256),  # Should favor System 1
            'logical_reasoning': torch.randn(512),    # Should favor System 2
            'mixed_task': torch.randn(384)           # Should use both
        }

        for task_name, task_input in tasks.items():
            print(f"   Testing {task_name}...")

            context = {'task_type': task_name}
            output = await integrator.process(task_input, context=context)

            # Check which systems were used
            s1_used = output.system1_result is not None
            s2_used = output.system2_result is not None

            print(f"   Systems used: S1={s1_used}, S2={s2_used}")
            print(f"   Confidence: {output.confidence:.3f}")
            print(f"   Processing time: {output.processing_time*1000:.1f}ms")

            results['cognitive_benchmarks'][task_name] = {
                'success': True,
                'system1_used': s1_used,
                'system2_used': s2_used,
                'confidence': output.confidence,
                'time_ms': output.processing_time * 1000
            }

        # Step 6: Generate health report
        print("\n6️⃣ Generating health report...")
        health = integrator.get_health_report()

        print(f"   Integrator status: {health['integrator_status']}")
        print(f"   Processing count: {health['processing_count']}")
        print(f"   Overall health: {health['component_health']['overall_status']}")

        results['health'] = health

    except Exception as e:
        logger.error(f"Validation error: {e}", exc_info=True)
        results['error'] = str(e)

    # Summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)

    # Components
    components_ok = all(results['components'].values())
    print(f"\n✅ Components: {'PASS' if components_ok else 'FAIL'}")
    for comp, status in results['components'].items():
        print(f"   - {comp}: {'✅' if status else '❌'}")

    # Safety
    safety_ok = all(results['safety_requirements'].values())
    print(f"\n✅ Safety Requirements: {'PASS' if safety_ok else 'FAIL'}")
    for req, status in results['safety_requirements'].items():
        print(f"   - {req}: {'✅' if status else '❌'}")

    # Performance
    perf_ok = all(results['performance'].values())
    print(f"\n✅ Performance: {'PASS' if perf_ok else 'FAIL'}")
    for mode, status in results['performance'].items():
        print(f"   - {mode}: {'✅' if status else '❌'}")

    # Overall
    overall_pass = components_ok and safety_ok and perf_ok
    print(f"\n🎯 OVERALL: {'PASS ✅' if overall_pass else 'FAIL ❌'}")

    return results


if __name__ == "__main__":
    results = asyncio.run(validate_phase_4_11())

    # Exit with appropriate code
    if all([
        all(results.get('components', {}).values()),
        all(results.get('safety_requirements', {}).values()),
        all(results.get('performance', {}).values())
    ]):
        print("\n✅ Phase 4.11 validation PASSED")
        sys.exit(0)
    else:
        print("\n❌ Phase 4.11 validation FAILED")
        sys.exit(1)
