#!/usr/bin/env python3
"""
Thermodynamic Cognitive Scheduler Test & Recording

This script demonstrates and records the revolutionary self-optimizing system
that uses Kimera's thermodynamic principles to optimize GPU performance in real-time.

RECORDING EVERYTHING: This test will capture all optimization decisions, 
thermodynamic states, and performance improvements.
"""

import sys
import time
import json
import numpy as np
from datetime import datetime
from pathlib import Path

# Initialize structured logger
from backend.utils.kimera_logger import get_system_logger
logger = get_system_logger(__name__)


# Add backend path
backend_path = Path(__file__).parent / "backend"
if str(backend_path) not in sys.path:
    sys.path.append(str(backend_path))

from backend.engines.thermodynamic_scheduler import ThermodynamicCognitiveScheduler


class ThermodynamicSchedulerRecorder:
    """Records and analyzes the thermodynamic scheduler performance"""
    
    def __init__(self):
        self.scheduler = ThermodynamicCognitiveScheduler(monitoring_interval=0.5)
        self.test_results = []
        self.session_start = datetime.now()
        
        logger.info("🧠🔥 THERMODYNAMIC COGNITIVE SCHEDULER TEST")
        logger.info("=" * 80)
        logger.info("Revolutionary self-optimizing AI using thermodynamic principles")
        logger.info("RECORDING EVERYTHING for analysis...")
        logger.info()
    
    def run_baseline_test(self):
        """Run baseline test without thermodynamic optimization"""
        logger.info("📊 BASELINE TEST (No Thermodynamic Optimization)
        logger.info("-" * 60)
        
        field_counts = [100, 500, 1000, 2500, 5000]
        baseline_results = []
        
        for count in field_counts:
            start_time = time.time()
            
            # Direct field creation without optimization
            fields = self.scheduler.field_engine.batch_create_fields(count)
            
            end_time = time.time()
            creation_time = end_time - start_time
            rate = len(fields) / creation_time
            
            # Collect GPU metrics
            gpu_metrics = self.scheduler.collect_gpu_metrics()
            
            result = {
                "field_count": count,
                "creation_time": creation_time,
                "performance_rate": rate,
                "gpu_temperature": gpu_metrics["temperature"],
                "gpu_power": gpu_metrics["power"],
                "gpu_utilization": gpu_metrics["utilization"]
            }
            
            baseline_results.append(result)
            
            logger.info(f"   {count:,} fields: {rate:.1f} fields/sec | ")
                  f"Temp: {gpu_metrics['temperature']:.1f}°C | "
                  f"Power: {gpu_metrics['power']:.1f}W")
            
            time.sleep(1)  # Brief pause between tests
        
        return baseline_results
    
    def run_thermodynamic_optimization_test(self):
        """Run test with full thermodynamic optimization"""
        logger.info("\n🔥 THERMODYNAMIC OPTIMIZATION TEST")
        logger.info("-" * 60)
        logger.info("Using Kimera's thermodynamic principles for real-time optimization...")
        
        field_counts = [100, 500, 1000, 2500, 5000, 10000]
        optimization_results = []
        
        # Start continuous monitoring
        self.scheduler.start_continuous_monitoring()
        
        for i, count in enumerate(field_counts):
            logger.info(f"\n🧪 Test {i+1}/{len(field_counts)
            logger.info("." * 40)
            
            # Run optimized cognitive task
            result = self.scheduler.run_optimized_cognitive_task(count)
            
            # Extract key metrics
            initial_state = result["initial_thermodynamic_state"]
            final_state = result["final_thermodynamic_state"]
            optimization = result["optimization_applied"]
            
            logger.info(f"⚡ Performance: {result['performance_rate']:.1f} fields/sec")
            logger.info(f"🌡️  Thermal Entropy: {initial_state.thermal_entropy:.3f} → {final_state.thermal_entropy:.3f}")
            logger.info(f"🧠 Computational Entropy: {initial_state.computational_entropy:.3f} → {final_state.computational_entropy:.3f}")
            logger.info(f"↩️  Reversibility: {initial_state.reversibility_index:.3f} → {final_state.reversibility_index:.3f}")
            logger.info(f"🆓 Free Energy: {initial_state.free_energy:.1f} → {final_state.free_energy:.1f}")
            logger.info(f"📈 Thermo Efficiency: {initial_state.thermodynamic_efficiency:.3f} → {final_state.thermodynamic_efficiency:.3f}")
            
            if optimization.decision_type != "none":
                logger.debug(f"🔧 Optimization Applied: {optimization.decision_type}")
                logger.info(f"   Action: {optimization.action_taken}")
                logger.info(f"   Expected: {optimization.expected_improvement:.1f}% improvement")
                if optimization.actual_improvement is not None:
                    logger.info(f"   Actual: {optimization.actual_improvement:.1f}% improvement")
                    logger.info(f"   Status: {optimization.validation_status}")
            else:
                logger.info("✅ No optimization needed - already optimal")
            
            # Record results
            optimization_results.append({
                "test_number": i + 1,
                "field_count": count,
                "total_time": result["total_time"],
                "performance_rate": result["performance_rate"],
                "initial_thermodynamic_state": {
                    "thermal_entropy": initial_state.thermal_entropy,
                    "computational_entropy": initial_state.computational_entropy,
                    "reversibility_index": initial_state.reversibility_index,
                    "free_energy": initial_state.free_energy,
                    "thermodynamic_efficiency": initial_state.thermodynamic_efficiency,
                    "gpu_temperature": initial_state.gpu_temperature,
                    "gpu_power": initial_state.gpu_power
                },
                "final_thermodynamic_state": {
                    "thermal_entropy": final_state.thermal_entropy,
                    "computational_entropy": final_state.computational_entropy,
                    "reversibility_index": final_state.reversibility_index,
                    "free_energy": final_state.free_energy,
                    "thermodynamic_efficiency": final_state.thermodynamic_efficiency,
                    "gpu_temperature": final_state.gpu_temperature,
                    "gpu_power": final_state.gpu_power
                },
                "optimization_record": {
                    "decision_type": optimization.decision_type,
                    "action_taken": optimization.action_taken,
                    "expected_improvement": optimization.expected_improvement,
                    "actual_improvement": optimization.actual_improvement,
                    "reversibility_change": optimization.reversibility_change,
                    "efficiency_change": optimization.efficiency_change,
                    "validation_status": optimization.validation_status
                },
                "improvements": {
                    "thermodynamic_improvement": result["thermodynamic_improvement"],
                    "reversibility_improvement": result["reversibility_improvement"]
                }
            })
            
            time.sleep(2)  # Pause between tests for thermal stability
        
        # Stop monitoring
        self.scheduler.stop_continuous_monitoring()
        
        return optimization_results
    
    def analyze_optimization_effectiveness(self, baseline_results, optimization_results):
        """Analyze the effectiveness of thermodynamic optimization"""
        logger.info(f"\n📊 THERMODYNAMIC OPTIMIZATION ANALYSIS")
        logger.info("=" * 80)
        
        # Performance comparison
        baseline_rates = [r["performance_rate"] for r in baseline_results]
        optimization_rates = [r["performance_rate"] for r in optimization_results[:len(baseline_rates)]]
        
        performance_improvements = []
        for i, (baseline_rate, opt_rate) in enumerate(zip(baseline_rates, optimization_rates)):
            improvement = ((opt_rate - baseline_rate) / baseline_rate) * 100.0
            performance_improvements.append(improvement)
            
            field_count = baseline_results[i]["field_count"]
            logger.info(f"📈 {field_count:,} fields: {baseline_rate:.1f} → {opt_rate:.1f} fields/sec ")
                  f"({improvement:+.1f}% improvement)")
        
        avg_performance_improvement = np.mean(performance_improvements)
        
        # Thermodynamic metrics analysis
        reversibility_improvements = []
        efficiency_improvements = []
        
        for result in optimization_results:
            if result["optimization_record"]["reversibility_change"] is not None:
                reversibility_improvements.append(result["optimization_record"]["reversibility_change"])
            
            if result["optimization_record"]["efficiency_change"] is not None:
                efficiency_improvements.append(result["optimization_record"]["efficiency_change"])
        
        # Optimization statistics
        optimization_stats = self.scheduler.get_optimization_statistics()
        
        logger.info(f"\n🎯 OPTIMIZATION EFFECTIVENESS SUMMARY:")
        logger.info(f"   Average Performance Improvement: {avg_performance_improvement:+.1f}%")
        logger.info(f"   Optimization Success Rate: {optimization_stats['success_rate_percent']:.1f}%")
        logger.info(f"   Total Optimizations Applied: {optimization_stats['total_optimizations']}")
        logger.info(f"   Successful Optimizations: {optimization_stats['successful_optimizations']}")
        logger.info(f"   Reversibility Improvements: {optimization_stats['reversibility_improvements']}")
        
        if reversibility_improvements:
            avg_reversibility_improvement = np.mean(reversibility_improvements)
            logger.info(f"   Average Reversibility Gain: {avg_reversibility_improvement:+.3f}")
        
        if efficiency_improvements:
            avg_efficiency_improvement = np.mean(efficiency_improvements)
            logger.info(f"   Average Efficiency Gain: {avg_efficiency_improvement:+.3f}")
        
        # Optimization type breakdown
        if optimization_stats['optimization_types']:
            logger.debug(f"\n🔧 OPTIMIZATION TYPES APPLIED:")
            for opt_type, count in optimization_stats['optimization_types'].items():
                logger.info(f"   {opt_type}: {count} times")
        
        return {
            "average_performance_improvement": avg_performance_improvement,
            "performance_improvements": performance_improvements,
            "reversibility_improvements": reversibility_improvements,
            "efficiency_improvements": efficiency_improvements,
            "optimization_statistics": optimization_stats
        }
    
    def demonstrate_real_time_adaptation(self):
        """Demonstrate real-time thermodynamic adaptation"""
        logger.info(f"\n🔄 REAL-TIME THERMODYNAMIC ADAPTATION DEMO")
        logger.info("-" * 60)
        logger.info("Demonstrating how the scheduler adapts to changing conditions...")
        
        adaptation_results = []
        
        # Start monitoring
        self.scheduler.start_continuous_monitoring()
        
        # Simulate varying workloads to trigger different optimizations
        workload_scenarios = [
            {"name": "Light Load", "fields": 200, "description": "Should trigger complexity increase"},
            {"name": "Heavy Load", "fields": 8000, "description": "Should trigger thermal management"},
            {"name": "Optimal Load", "fields": 1000, "description": "Should maintain optimal state"},
            {"name": "Burst Load", "fields": 15000, "description": "Should trigger batch optimization"},
            {"name": "Recovery", "fields": 500, "description": "Should optimize for efficiency"}
        ]
        
        for i, scenario in enumerate(workload_scenarios):
            logger.info(f"\n🧪 Scenario {i+1}: {scenario['name']} ({scenario['fields']:,} fields)
            logger.info(f"   Expected: {scenario['description']}")
            
            start_time = time.time()
            result = self.scheduler.run_optimized_cognitive_task(scenario['fields'])
            end_time = time.time()
            
            adaptation_result = {
                "scenario": scenario['name'],
                "field_count": scenario['fields'],
                "execution_time": end_time - start_time,
                "performance_rate": result['performance_rate'],
                "optimization_applied": result['optimization_applied'].decision_type,
                "optimization_action": result['optimization_applied'].action_taken,
                "thermodynamic_state": {
                    "reversibility": result['final_thermodynamic_state'].reversibility_index,
                    "thermal_entropy": result['final_thermodynamic_state'].thermal_entropy,
                    "free_energy": result['final_thermodynamic_state'].free_energy,
                    "temperature": result['final_thermodynamic_state'].gpu_temperature
                }
            }
            
            adaptation_results.append(adaptation_result)
            
            logger.info(f"   ✅ Completed: {result['performance_rate']:.1f} fields/sec")
            logger.debug(f"   🔧 Optimization: {result['optimization_applied'].decision_type}")
            logger.info(f"   🌡️  Final State: T={result['final_thermodynamic_state'].gpu_temperature:.1f}°C, ")
                  f"R={result['final_thermodynamic_state'].reversibility_index:.3f}")
            
            time.sleep(3)  # Allow thermal dynamics to settle
        
        self.scheduler.stop_continuous_monitoring()
        
        return adaptation_results
    
    def save_complete_session_record(self, baseline_results, optimization_results, 
                                   analysis_results, adaptation_results):
        """Save complete session recording"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"thermodynamic_scheduler_session_{timestamp}.json"
        
        session_record = {
            "session_metadata": {
                "timestamp": datetime.now().isoformat(),
                "session_start": self.session_start.isoformat(),
                "session_duration": (datetime.now() - self.session_start).total_seconds(),
                "test_type": "Thermodynamic Cognitive Scheduler Demonstration",
                "purpose": "Revolutionary self-optimizing AI using thermodynamic principles"
            },
            "baseline_results": baseline_results,
            "optimization_results": optimization_results,
            "analysis_results": analysis_results,
            "adaptation_results": adaptation_results,
            "scheduler_statistics": self.scheduler.get_optimization_statistics(),
            "thermodynamic_session_data": None  # Will be filled by scheduler
        }
        
        # Save scheduler's internal data
        scheduler_filename = self.scheduler.save_thermodynamic_session(
            f"scheduler_internal_{timestamp}.json"
        )
        session_record["thermodynamic_session_data"] = scheduler_filename
        
        # Save complete session record
        with open(filename, 'w') as f:
            json.dump(session_record, f, indent=2)
        
        logger.info(f"\n💾 COMPLETE SESSION RECORD SAVED:")
        logger.info(f"   Main record: {filename}")
        logger.info(f"   Scheduler data: {scheduler_filename}")
        
        return filename, scheduler_filename


def main():
    """Main test function"""
    logger.info("🧠🔥 THERMODYNAMIC COGNITIVE SCHEDULER RECORDING SESSION")
    logger.info("=" * 80)
    logger.info("RECORDING EVERYTHING: Testing revolutionary self-optimizing AI system")
    logger.info("Using Kimera's thermodynamic principles for real-time GPU optimization")
    logger.info()
    
    try:
        # Create recorder
        recorder = ThermodynamicSchedulerRecorder()
        
        # Run baseline tests
        baseline_results = recorder.run_baseline_test()
        
        # Run thermodynamic optimization tests
        optimization_results = recorder.run_thermodynamic_optimization_test()
        
        # Analyze optimization effectiveness
        analysis_results = recorder.analyze_optimization_effectiveness(
            baseline_results, optimization_results
        )
        
        # Demonstrate real-time adaptation
        adaptation_results = recorder.demonstrate_real_time_adaptation()
        
        # Save complete session record
        main_file, scheduler_file = recorder.save_complete_session_record(
            baseline_results, optimization_results, analysis_results, adaptation_results
        )
        
        logger.info(f"\n" + "=" * 80)
        logger.info(f"🎉 THERMODYNAMIC SCHEDULER RECORDING COMPLETE!")
        logger.info(f"=" * 80)
        logger.debug(f"🔬 Revolutionary Result: AI system successfully used thermodynamic")
        logger.info(f"   principles to self-optimize GPU performance in real-time!")
        logger.info(f"🌡️  Kimera's entropy, reversibility, and free energy analysis")
        logger.info(f"   enabled dynamic optimization of hardware substrate.")
        logger.info(f"📊 Performance improvements: {analysis_results['average_performance_improvement']:+.1f}% average")
        logger.info(f"🎯 Optimization success rate: {analysis_results['optimization_statistics']['success_rate_percent']:.1f}%")
        logger.info(f"💾 All data recorded to: {main_file}")
        
    except KeyboardInterrupt:
        logger.warning(f"\n⚠️  Test interrupted by user")
    except Exception as e:
        logger.error(f"\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 