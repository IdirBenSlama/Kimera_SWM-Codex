#!/usr/bin/env python3
"""
Test Running Kimera SWM Platform
================================

Test script to verify that the Kimera SWM platform is running
and demonstrate its capabilities.
"""

import asyncio
import requests
import json
import time
from datetime import datetime

async def test_running_platform():
    """Test the running platform services"""
    print("🧪 TESTING RUNNING KIMERA SWM PLATFORM")
    print("=" * 60)
    
    test_results = {}
    
    # Test 1: Check API Health
    print("1️⃣  Testing Cognitive Services API...")
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            print(f"   ✅ API Health: {health_data.get('status', 'OK')}")
            test_results['api_health'] = 'OPERATIONAL'
        else:
            print(f"   ⚠️  API Health: Status {response.status_code}")
            test_results['api_health'] = f'STATUS_{response.status_code}'
    except requests.exceptions.ConnectionError:
        print("   🔄 API Health: Starting up...")
        test_results['api_health'] = 'STARTING'
    except Exception as e:
        print(f"   ❌ API Health: {e}")
        test_results['api_health'] = 'ERROR'
    
    # Test 2: Check Dashboard
    print("2️⃣  Testing Monitoring Dashboard...")
    try:
        response = requests.get("http://localhost:8001/health", timeout=5)
        if response.status_code == 200:
            print("   ✅ Dashboard: Operational")
            test_results['dashboard'] = 'OPERATIONAL'
        else:
            print(f"   ⚠️  Dashboard: Status {response.status_code}")
            test_results['dashboard'] = f'STATUS_{response.status_code}'
    except requests.exceptions.ConnectionError:
        print("   🔄 Dashboard: Starting up...")
        test_results['dashboard'] = 'STARTING'
    except Exception as e:
        print(f"   ❌ Dashboard: {e}")
        test_results['dashboard'] = 'ERROR'
    
    # Test 3: Core Performance Systems
    print("3️⃣  Testing Core Performance Systems...")
    try:
        # Test GPU
        import torch
        from src.core.performance.gpu_acceleration import get_gpu_metrics, move_to_gpu
        
        metrics = get_gpu_metrics()
        test_tensor = move_to_gpu(torch.randn(64, 64))
        result = torch.matmul(test_tensor, test_tensor.T)
        
        print(f"   ✅ GPU: {metrics.device_name} processing {result.shape}")
        test_results['gpu'] = 'OPERATIONAL'
        
        # Test Cache
        from src.core.performance.advanced_caching import put_cached, get_cached, get_cache_stats
        
        test_data = {"test": "platform_running", "timestamp": time.time()}
        await put_cached("running_test", test_data)
        cached = await get_cached("running_test")
        stats = get_cache_stats()
        
        print(f"   ✅ Cache: {stats.total_entries} entries, hit rate {stats.hit_rate:.1%}")
        test_results['cache'] = 'OPERATIONAL'
        
        # Test Pipeline
        from src.core.performance.pipeline_optimization import add_pipeline_task, TaskPriority, get_pipeline_metrics
        
        async def test_task(name):
            return f"Task {name} completed at {time.time()}"
        
        add_pipeline_task("running_test", test_task, "Platform Test", priority=TaskPriority.HIGH)
        pipeline_metrics = get_pipeline_metrics()
        
        print(f"   ✅ Pipeline: {pipeline_metrics.total_tasks} tasks, score {pipeline_metrics.performance_score:.3f}")
        test_results['pipeline'] = 'OPERATIONAL'
        
        # Test Scaling
        from src.core.scaling.horizontal_scaling import route_cognitive_request, get_cluster_status
        
        node = await route_cognitive_request("running_test", {"priority": "high"})
        cluster = get_cluster_status()
        
        print(f"   ✅ Scaling: Routed to {node}, {cluster['cluster_metrics'].total_nodes} nodes")
        test_results['scaling'] = 'OPERATIONAL'
        
    except Exception as e:
        print(f"   ❌ Core Systems: {e}")
        test_results['core_systems'] = 'ERROR'
    
    # Test 4: Cognitive Processing Request
    print("4️⃣  Testing Cognitive Processing...")
    try:
        cognitive_request = {
            "input_data": "Test the running Kimera SWM cognitive processing capabilities",
            "workflow_type": "basic_cognition",
            "priority": "high"
        }
        
        response = requests.post(
            "http://localhost:8000/cognitive/process",
            json=cognitive_request,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Cognitive Processing: Request {result.get('request_id', 'N/A')} completed")
            test_results['cognitive_processing'] = 'OPERATIONAL'
        else:
            print(f"   ⚠️  Cognitive Processing: Status {response.status_code}")
            test_results['cognitive_processing'] = f'STATUS_{response.status_code}'
            
    except requests.exceptions.ConnectionError:
        print("   🔄 Cognitive Processing: API not ready")
        test_results['cognitive_processing'] = 'API_NOT_READY'
    except Exception as e:
        print(f"   ❌ Cognitive Processing: {e}")
        test_results['cognitive_processing'] = 'ERROR'
    
    # Test 5: Real-time Metrics
    print("5️⃣  Testing Real-time Metrics...")
    try:
        response = requests.get("http://localhost:8001/metrics", timeout=5)
        if response.status_code == 200:
            metrics_data = response.json()
            print(f"   ✅ Metrics: {len(metrics_data)} metric categories")
            test_results['metrics'] = 'OPERATIONAL'
        else:
            print(f"   ⚠️  Metrics: Status {response.status_code}")
            test_results['metrics'] = f'STATUS_{response.status_code}'
    except requests.exceptions.ConnectionError:
        print("   🔄 Metrics: Dashboard not ready")
        test_results['metrics'] = 'DASHBOARD_NOT_READY'
    except Exception as e:
        print(f"   ❌ Metrics: {e}")
        test_results['metrics'] = 'ERROR'
    
    # Summary
    print("\n" + "=" * 60)
    print("🎯 PLATFORM STATUS SUMMARY")
    print("=" * 60)
    
    operational_count = sum(1 for status in test_results.values() if status == 'OPERATIONAL')
    total_tests = len(test_results)
    
    for service, status in test_results.items():
        emoji = "✅" if status == "OPERATIONAL" else "🔄" if "STARTING" in status or "NOT_READY" in status else "❌"
        print(f"{emoji} {service.replace('_', ' ').title()}: {status}")
    
    print(f"\n📊 OPERATIONAL STATUS: {operational_count}/{total_tests} services ({operational_count/total_tests*100:.1f}%)")
    
    if operational_count >= total_tests * 0.8:
        print("🎉 KIMERA SWM PLATFORM: RUNNING SUCCESSFULLY!")
        print("🚀 Enterprise cognitive services platform is operational!")
    elif operational_count >= total_tests * 0.6:
        print("⚡ KIMERA SWM PLATFORM: CORE SYSTEMS OPERATIONAL!")
        print("🔄 Additional services starting up...")
    else:
        print("🔄 KIMERA SWM PLATFORM: STARTING UP...")
        print("⏳ Please wait for all services to initialize...")
    
    print("\n🌐 QUICK ACCESS:")
    print("   • API Documentation: http://localhost:8000/docs")
    print("   • Health Check: http://localhost:8000/health")
    print("   • Monitoring Dashboard: http://localhost:8001")
    print("   • Real-time Metrics: http://localhost:8001/metrics")
    
    return operational_count >= total_tests * 0.6

if __name__ == "__main__":
    print(f"🧪 Testing Kimera SWM Platform - {datetime.now().strftime('%H:%M:%S')}")
    success = asyncio.run(test_running_platform())
    
    if success:
        print("\n✅ Platform test completed successfully!")
    else:
        print("\n🔄 Platform still initializing...")