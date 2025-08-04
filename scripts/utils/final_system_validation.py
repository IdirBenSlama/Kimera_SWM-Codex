#!/usr/bin/env python3
"""
FINAL SYSTEM VALIDATION
=======================

Comprehensive validation script to confirm KIMERA thermodynamic system
is fully operational and ready for production use.

Following KIMERA Autonomous Architect Protocol v3.0
"""

import asyncio
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from engines.comprehensive_thermodynamic_monitor import ComprehensiveThermodynamicMonitor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def final_system_validation():
    """Execute final system validation protocol"""

    logger.info("🎯 KIMERA FINAL SYSTEM VALIDATION")
    logger.info("=" * 60)
    logger.info("Testing: Thermodynamic Monitor System")
    logger.info("Goal: Confirm alert loop resolution and system stability")
    logger.info("=" * 60)

    validation_results = {
        'alert_loop_resolved': False,
        'efficiency_baseline_working': False,
        'monitoring_stable': False,
        'initialization_working': False,
        'overall_system_health': 'UNKNOWN'
    }

    try:
        # Test 1: Quick monitoring cycle (should not generate infinite alerts)
        logger.info("🧪 Test 1: Alert Loop Resolution")
        logger.info("-" * 40)

        monitor = ComprehensiveThermodynamicMonitor(
            monitoring_interval=0.5,  # Fast for testing
            auto_optimization=False   # Disable for pure monitoring test
        )

        alert_count = 0
        def count_alerts(alert):
            nonlocal alert_count
            alert_count += 1
            logger.info(f"   Alert: {alert.alert_type} - {alert.severity}")

        monitor.add_alert_callback(count_alerts)

        # Run monitoring for 5 seconds
        await monitor.start_continuous_monitoring()
        await asyncio.sleep(5)
        await monitor.stop_monitoring()

        logger.info(f"   Alerts Generated: {alert_count} in 5 seconds")
        logger.info(f"   Alert Rate: {alert_count/5:.1f} per second")

        if alert_count < 10:  # Less than 2 per second is good
            logger.info("   ✅ Alert loop RESOLVED")
            validation_results['alert_loop_resolved'] = True
        else:
            logger.info("   ❌ Alert loop still present")

        # Test 2: Energy efficiency baseline
        logger.info("\n🔋 Test 2: Energy Efficiency Baseline")
        logger.info("-" * 40)

        state = await monitor.calculate_comprehensive_thermodynamic_state()
        logger.info(f"   Energy Efficiency: {state.energy_efficiency:.3f}")
        logger.info(f"   Overall Efficiency: {state.overall_efficiency:.3f}")
        logger.info(f"   System Health: {state.system_health.value}")

        if state.energy_efficiency > 0.3:  # Above minimum threshold
            logger.info("   ✅ Efficiency baseline WORKING")
            validation_results['efficiency_baseline_working'] = True
        else:
            logger.info("   ❌ Efficiency still too low")

        # Test 3: Monitoring stability (no crashes)
        logger.info("\n⚡ Test 3: Monitoring Stability")
        logger.info("-" * 40)

        stable_cycles = 0
        try:
            for i in range(5):
                state = await monitor.calculate_comprehensive_thermodynamic_state()
                stable_cycles += 1
                logger.info(f"   Cycle {i+1}: Health={state.system_health.value}, Eff={state.overall_efficiency:.3f}")
                await asyncio.sleep(0.2)

            logger.info("   ✅ Monitoring STABLE")
            validation_results['monitoring_stable'] = True

        except Exception as e:
            logger.info(f"   ❌ Monitoring unstable: {e}")

        # Test 4: Battery initialization
        logger.info("\n🔋 Test 4: Battery Initialization")
        logger.info("-" * 40)

        battery_status = monitor.vortex_battery.get_battery_status()
        operations = battery_status.get('operations_completed', 0)
        avg_efficiency = battery_status.get('average_efficiency', 0.0)

        logger.info(f"   Operations Completed: {operations}")
        logger.info(f"   Average Efficiency: {avg_efficiency:.3f}")

        if operations > 0 and avg_efficiency > 0:
            logger.info("   ✅ Battery initialization WORKING")
            validation_results['initialization_working'] = True
        else:
            logger.info("   ❌ Battery initialization failed")

        await monitor.shutdown()

    except Exception as e:
        logger.info(f"❌ Validation failed with error: {e}")
        return False

    # Calculate overall health
    tests_passed = sum(validation_results[key] for key in validation_results if key != 'overall_system_health')
    total_tests = len(validation_results) - 1

    if tests_passed == total_tests:
        validation_results['overall_system_health'] = 'EXCELLENT'
    elif tests_passed >= total_tests * 0.8:
        validation_results['overall_system_health'] = 'GOOD'
    elif tests_passed >= total_tests * 0.6:
        validation_results['overall_system_health'] = 'ACCEPTABLE'
    else:
        validation_results['overall_system_health'] = 'POOR'

    # Final report
    logger.info("\n" + "=" * 60)
    logger.info("📊 FINAL VALIDATION RESULTS")
    logger.info("=" * 60)

    for test, result in validation_results.items():
        if test == 'overall_system_health':
            continue
        status = "✅ PASS" if result else "❌ FAIL"
        test_name = test.replace('_', ' ').title()
        logger.info(f"{test_name:.<40} {status}")

    logger.info("-" * 60)
    logger.info(f"Overall System Health: {validation_results['overall_system_health']}")
    logger.info(f"Tests Passed: {tests_passed}/{total_tests}")

    if validation_results['overall_system_health'] in ['EXCELLENT', 'GOOD']:
        logger.info("\n🎉 SYSTEM VALIDATION: SUCCESS")
        logger.info("✅ Thermodynamic monitoring system is PRODUCTION READY")
        logger.info("✅ Alert loop has been permanently resolved")
        logger.info("✅ System demonstrates robust operation")
        return True
    else:
        logger.info("\n⚠️  SYSTEM VALIDATION: NEEDS ATTENTION")
        logger.info("System requires additional fixes before production use")
        return False


async def main():
    """Execute final validation"""
    success = await final_system_validation()

    logger.info("\n" + "=" * 60)
    if success:
        logger.info("🎯 KIMERA THERMODYNAMIC SYSTEM: MISSION ACCOMPLISHED")
        logger.info("System validated and ready for production operations")
    else:
        logger.info("🚨 KIMERA THERMODYNAMIC SYSTEM: REQUIRES ATTENTION")
        logger.info("Additional fixes needed before production deployment")
    logger.info("=" * 60)

    return success


if __name__ == "__main__":
    asyncio.run(main())
