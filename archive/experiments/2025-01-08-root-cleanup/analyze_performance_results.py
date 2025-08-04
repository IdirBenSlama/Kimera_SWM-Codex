#!/usr/bin/env python3
"""
Kimera Performance Results Analyzer
===================================
This script analyzes the performance test results and generates
detailed insights and visualizations.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from datetime import datetime
import statistics
import logging
logger = logging.getLogger(__name__)

class PerformanceAnalyzer:
    def __init__(self, results_dir: str = "performance_results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
    def load_latest_results(self) -> dict:
        """Load the most recent performance test results"""
        json_files = list(self.results_dir.glob("kimera_performance_report_*.json"))
        if not json_files:
            raise FileNotFoundError("No performance test results found")
            
        latest_file = max(json_files, key=lambda p: p.stat().st_mtime)
        logger.info(f"Loading results from: {latest_file}")
        
        with open(latest_file, 'r') as f:
            return json.load(f)
    
    def analyze_results(self, results: dict):
        """Perform detailed analysis of performance results"""
        logger.info("\n" + "="*60)
        logger.info("KIMERA PERFORMANCE ANALYSIS")
        logger.info("="*60)
        
        # Extract test results
        test_results = results['test_results']
        
        # Create analysis report
        analysis = {
            'summary': self._analyze_summary(test_results),
            'bottlenecks': self._identify_bottlenecks(test_results),
            'scalability': self._analyze_scalability(results),
            'reliability': self._analyze_reliability(test_results),
            'resource_efficiency': self._analyze_resource_efficiency(results)
        }
        
        # Generate detailed visualizations
        self._create_detailed_visualizations(results)
        
        return analysis
    
    def _analyze_summary(self, test_results):
        """Generate summary statistics"""
        summary = {
            'total_endpoints_tested': len(test_results),
            'total_requests_made': sum(r.get('total_requests', 0) for r in test_results),
            'total_successful_requests': sum(r.get('successful_requests', 0) for r in test_results),
            'overall_success_rate': 0,
            'fastest_endpoint': None,
            'slowest_endpoint': None,
            'most_reliable_endpoint': None,
            'least_reliable_endpoint': None
        }
        
        # Calculate overall success rate
        if summary['total_requests_made'] > 0:
            summary['overall_success_rate'] = (summary['total_successful_requests'] / 
                                              summary['total_requests_made']) * 100
        
        # Find extremes
        valid_results = [r for r in test_results if 'avg_response_time_ms' in r]
        
        if valid_results:
            fastest = min(valid_results, key=lambda x: x['avg_response_time_ms'])
            slowest = max(valid_results, key=lambda x: x['avg_response_time_ms'])
            
            summary['fastest_endpoint'] = {
                'endpoint': fastest['endpoint'],
                'avg_response_time_ms': fastest['avg_response_time_ms']
            }
            
            summary['slowest_endpoint'] = {
                'endpoint': slowest['endpoint'],
                'avg_response_time_ms': slowest['avg_response_time_ms']
            }
            
            # Reliability based on success rate
            for r in valid_results:
                r['success_rate'] = (r['successful_requests'] / r['total_requests']) * 100
                
            most_reliable = max(valid_results, key=lambda x: x['success_rate'])
            least_reliable = min(valid_results, key=lambda x: x['success_rate'])
            
            summary['most_reliable_endpoint'] = {
                'endpoint': most_reliable['endpoint'],
                'success_rate': most_reliable['success_rate']
            }
            
            summary['least_reliable_endpoint'] = {
                'endpoint': least_reliable['endpoint'],
                'success_rate': least_reliable['success_rate']
            }
        
        return summary
    
    def _identify_bottlenecks(self, test_results):
        """Identify performance bottlenecks"""
        bottlenecks = []
        
        for result in test_results:
            if 'avg_response_time_ms' not in result:
                continue
                
            # High response time bottleneck
            if result['avg_response_time_ms'] > 500:
                bottlenecks.append({
                    'type': 'high_response_time',
                    'endpoint': result['endpoint'],
                    'severity': 'high' if result['avg_response_time_ms'] > 1000 else 'medium',
                    'avg_response_time_ms': result['avg_response_time_ms'],
                    'recommendation': 'Consider caching, query optimization, or async processing'
                })
            
            # High failure rate bottleneck
            if result['failed_requests'] > result['total_requests'] * 0.05:
                failure_rate = (result['failed_requests'] / result['total_requests']) * 100
                bottlenecks.append({
                    'type': 'high_failure_rate',
                    'endpoint': result['endpoint'],
                    'severity': 'critical' if failure_rate > 10 else 'high',
                    'failure_rate_percent': failure_rate,
                    'recommendation': 'Check rate limiting, connection pooling, and error handling'
                })
            
            # High variance bottleneck (unstable performance)
            if 'p99_response_time_ms' in result and 'median_response_time_ms' in result:
                variance_ratio = result['p99_response_time_ms'] / result['median_response_time_ms']
                if variance_ratio > 5:
                    bottlenecks.append({
                        'type': 'high_variance',
                        'endpoint': result['endpoint'],
                        'severity': 'medium',
                        'variance_ratio': variance_ratio,
                        'recommendation': 'Investigate sporadic slowdowns, GC pauses, or resource contention'
                    })
        
        return bottlenecks
    
    def _analyze_scalability(self, results):
        """Analyze system scalability"""
        scalability = {
            'max_throughput_rps': 0,
            'saturation_point': None,
            'scaling_efficiency': None
        }
        
        # Calculate max throughput
        for result in results['test_results']:
            if 'requests_per_second' in result:
                scalability['max_throughput_rps'] = max(
                    scalability['max_throughput_rps'],
                    result['requests_per_second']
                )
        
        # Analyze system metrics for saturation
        if 'system_metrics_summary' in results:
            sys_metrics = results['system_metrics_summary']
            
            # Determine if system reached saturation
            if sys_metrics.get('cpu', {}).get('max_percent', 0) > 85:
                scalability['saturation_point'] = 'CPU bound'
            elif sys_metrics.get('memory', {}).get('max_percent', 0) > 85:
                scalability['saturation_point'] = 'Memory bound'
            else:
                scalability['saturation_point'] = 'Not reached'
        
        return scalability
    
    def _analyze_reliability(self, test_results):
        """Analyze system reliability"""
        reliability = {
            'endpoints_meeting_sla': 0,
            'endpoints_total': len(test_results),
            'sla_compliance_percent': 0,
            'error_patterns': []
        }
        
        # Define SLA thresholds
        sla_response_time_ms = 200
        sla_success_rate = 99.0
        
        for result in test_results:
            if 'avg_response_time_ms' not in result:
                continue
                
            success_rate = (result['successful_requests'] / result['total_requests']) * 100
            
            if (result['avg_response_time_ms'] <= sla_response_time_ms and 
                success_rate >= sla_success_rate):
                reliability['endpoints_meeting_sla'] += 1
        
        reliability['sla_compliance_percent'] = (
            reliability['endpoints_meeting_sla'] / reliability['endpoints_total']
        ) * 100
        
        return reliability
    
    def _analyze_resource_efficiency(self, results):
        """Analyze resource utilization efficiency"""
        efficiency = {
            'cpu_efficiency': 'unknown',
            'memory_efficiency': 'unknown',
            'recommendations': []
        }
        
        if 'system_metrics_summary' in results:
            sys_metrics = results['system_metrics_summary']
            
            # CPU efficiency
            avg_cpu = sys_metrics.get('cpu', {}).get('avg_percent', 0)
            max_cpu = sys_metrics.get('cpu', {}).get('max_percent', 0)
            
            if avg_cpu < 30:
                efficiency['cpu_efficiency'] = 'underutilized'
                efficiency['recommendations'].append(
                    "CPU is underutilized. Consider increasing concurrent requests or optimizing for throughput."
                )
            elif avg_cpu > 70:
                efficiency['cpu_efficiency'] = 'high_utilization'
                efficiency['recommendations'].append(
                    "CPU utilization is high. Consider horizontal scaling or CPU optimization."
                )
            else:
                efficiency['cpu_efficiency'] = 'optimal'
            
            # Memory efficiency
            avg_mem = sys_metrics.get('memory', {}).get('avg_percent', 0)
            max_mem = sys_metrics.get('memory', {}).get('max_percent', 0)
            
            if max_mem - avg_mem > 20:
                efficiency['memory_efficiency'] = 'volatile'
                efficiency['recommendations'].append(
                    "Memory usage is volatile. Check for memory leaks or inefficient caching."
                )
            elif avg_mem > 75:
                efficiency['memory_efficiency'] = 'high_utilization'
                efficiency['recommendations'].append(
                    "Memory utilization is high. Consider memory optimization or increasing resources."
                )
            else:
                efficiency['memory_efficiency'] = 'stable'
        
        return efficiency
    
    def _create_detailed_visualizations(self, results):
        """Create detailed performance visualizations"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Response Time Comparison
        ax1 = plt.subplot(3, 3, 1)
        self._plot_response_time_comparison(results['test_results'], ax1)
        
        # 2. Success Rate Heatmap
        ax2 = plt.subplot(3, 3, 2)
        self._plot_success_rate_heatmap(results['test_results'], ax2)
        
        # 3. Throughput Analysis
        ax3 = plt.subplot(3, 3, 3)
        self._plot_throughput_analysis(results['test_results'], ax3)
        
        # 4. Response Time Distribution
        ax4 = plt.subplot(3, 3, 4)
        self._plot_response_distribution(results, ax4)
        
        # 5. Percentile Analysis
        ax5 = plt.subplot(3, 3, 5)
        self._plot_percentile_analysis(results['test_results'], ax5)
        
        # 6. Load vs Response Time
        ax6 = plt.subplot(3, 3, 6)
        self._plot_load_vs_response(results['test_results'], ax6)
        
        # 7. System Resource Timeline
        ax7 = plt.subplot(3, 1, 3)
        self._plot_system_resources(results, ax7)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / f'kimera_detailed_analysis_{timestamp}.png', dpi=300)
        plt.close()
        
        logger.info(f"\n📊 Detailed visualizations saved to: {self.results_dir}/kimera_detailed_analysis_{timestamp}.png")
    
    def _plot_response_time_comparison(self, test_results, ax):
        """Plot response time comparison"""
        valid_results = [r for r in test_results if 'avg_response_time_ms' in r]
        
        endpoints = [r['endpoint'] for r in valid_results]
        avg_times = [r['avg_response_time_ms'] for r in valid_results]
        p95_times = [r.get('p95_response_time_ms', 0) for r in valid_results]
        p99_times = [r.get('p99_response_time_ms', 0) for r in valid_results]
        
        x = np.arange(len(endpoints))
        width = 0.25
        
        ax.bar(x - width, avg_times, width, label='Average', alpha=0.8)
        ax.bar(x, p95_times, width, label='P95', alpha=0.8)
        ax.bar(x + width, p99_times, width, label='P99', alpha=0.8)
        
        ax.set_xlabel('Endpoints')
        ax.set_ylabel('Response Time (ms)')
        ax.set_title('Response Time Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels([e.split('/')[-1] for e in endpoints], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_success_rate_heatmap(self, test_results, ax):
        """Plot success rate heatmap"""
        valid_results = [r for r in test_results if 'successful_requests' in r]
        
        # Create matrix for heatmap
        data = []
        labels = []
        
        for r in valid_results:
            success_rate = (r['successful_requests'] / r['total_requests']) * 100
            data.append([success_rate])
            labels.append(r['endpoint'].split('/')[-1])
        
        # Create heatmap
        im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_xticks([0])
        ax.set_xticklabels(['Success Rate %'])
        ax.set_title('Endpoint Reliability Heatmap')
        
        # Add text annotations
        for i, row in enumerate(data):
            ax.text(0, i, f'{row[0]:.1f}%', ha='center', va='center',
                   color='white' if row[0] < 50 else 'black')
    
    def _plot_throughput_analysis(self, test_results, ax):
        """Plot throughput analysis"""
        valid_results = [r for r in test_results if 'requests_per_second' in r]
        
        endpoints = [r['endpoint'].split('/')[-1] for r in valid_results]
        throughput = [r['requests_per_second'] for r in valid_results]
        
        ax.barh(endpoints, throughput, color='skyblue')
        ax.set_xlabel('Requests per Second')
        ax.set_title('Endpoint Throughput')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, v in enumerate(throughput):
            ax.text(v + 0.5, i, f'{v:.1f}', va='center')
    
    def _plot_response_distribution(self, results, ax):
        """Plot response time distribution"""
        if 'detailed_metrics' in results:
            response_times = [m['response_time_ms'] for m in results['detailed_metrics'] 
                            if m.get('status_code', 0) != 0]
            
            if response_times:
                ax.hist(response_times, bins=50, alpha=0.7, color='blue', edgecolor='black')
                ax.axvline(np.mean(response_times), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(response_times):.1f}ms')
                ax.axvline(np.median(response_times), color='green', linestyle='--',
                          label=f'Median: {np.median(response_times):.1f}ms')
                ax.set_xlabel('Response Time (ms)')
                ax.set_ylabel('Frequency')
                ax.set_title('Response Time Distribution')
                ax.legend()
                ax.grid(True, alpha=0.3)
    
    def _plot_percentile_analysis(self, test_results, ax):
        """Plot percentile analysis"""
        valid_results = [r for r in test_results if all(k in r for k in 
                        ['median_response_time_ms', 'p95_response_time_ms', 'p99_response_time_ms'])]
        
        if valid_results:
            endpoints = [r['endpoint'].split('/')[-1] for r in valid_results]
            
            percentiles = {
                '50th (Median)': [r['median_response_time_ms'] for r in valid_results],
                '95th': [r['p95_response_time_ms'] for r in valid_results],
                '99th': [r['p99_response_time_ms'] for r in valid_results]
            }
            
            x = np.arange(len(endpoints))
            width = 0.25
            
            for i, (label, values) in enumerate(percentiles.items()):
                ax.bar(x + i*width, values, width, label=label, alpha=0.8)
            
            ax.set_xlabel('Endpoints')
            ax.set_ylabel('Response Time (ms)')
            ax.set_title('Response Time Percentiles')
            ax.set_xticks(x + width)
            ax.set_xticklabels(endpoints, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    def _plot_load_vs_response(self, test_results, ax):
        """Plot load vs response time relationship"""
        valid_results = [r for r in test_results if all(k in r for k in 
                        ['total_requests', 'avg_response_time_ms'])]
        
        if valid_results:
            loads = [r['total_requests'] for r in valid_results]
            response_times = [r['avg_response_time_ms'] for r in valid_results]
            
            ax.scatter(loads, response_times, s=100, alpha=0.6)
            
            # Add labels for each point
            for i, r in enumerate(valid_results):
                ax.annotate(r['endpoint'].split('/')[-1], 
                           (loads[i], response_times[i]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.7)
            
            ax.set_xlabel('Total Requests')
            ax.set_ylabel('Avg Response Time (ms)')
            ax.set_title('Load vs Response Time')
            ax.grid(True, alpha=0.3)
    
    def _plot_system_resources(self, results, ax):
        """Plot system resource utilization over time"""
        if 'system_metrics_summary' in results:
            metrics = results['system_metrics_summary']
            
            categories = ['CPU Avg', 'CPU Max', 'Memory Avg', 'Memory Max']
            values = [
                metrics.get('cpu', {}).get('avg_percent', 0),
                metrics.get('cpu', {}).get('max_percent', 0),
                metrics.get('memory', {}).get('avg_percent', 0),
                metrics.get('memory', {}).get('max_percent', 0)
            ]
            
            colors = ['lightblue', 'blue', 'lightgreen', 'green']
            bars = ax.bar(categories, values, color=colors)
            
            # Add value labels
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.1f}%', ha='center', va='bottom')
            
            ax.set_ylabel('Usage %')
            ax.set_title('System Resource Utilization Summary')
            ax.set_ylim(0, 100)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add threshold lines
            ax.axhline(y=80, color='orange', linestyle='--', alpha=0.5, label='Warning')
            ax.axhline(y=90, color='red', linestyle='--', alpha=0.5, label='Critical')
            ax.legend()
    
    def generate_report(self, analysis: dict):
        """Generate a comprehensive analysis report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.results_dir / f'kimera_analysis_report_{timestamp}.txt'
        
        with open(report_file, 'w') as f:
            f.write("KIMERA PERFORMANCE ANALYSIS REPORT\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Summary
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-" * 40 + "\n")
            summary = analysis['summary']
            f.write(f"Total Endpoints Tested: {summary['total_endpoints_tested']}\n")
            f.write(f"Total Requests Made: {summary['total_requests_made']}\n")
            f.write(f"Overall Success Rate: {summary['overall_success_rate']:.2f}%\n")
            
            if summary['fastest_endpoint']:
                f.write(f"\nFastest Endpoint: {summary['fastest_endpoint']['endpoint']}\n")
                f.write(f"  Response Time: {summary['fastest_endpoint']['avg_response_time_ms']:.2f}ms\n")
            
            if summary['slowest_endpoint']:
                f.write(f"\nSlowest Endpoint: {summary['slowest_endpoint']['endpoint']}\n")
                f.write(f"  Response Time: {summary['slowest_endpoint']['avg_response_time_ms']:.2f}ms\n")
            
            # Bottlenecks
            f.write("\n\nPERFORMANCE BOTTLENECKS\n")
            f.write("-" * 40 + "\n")
            
            if analysis['bottlenecks']:
                for bottleneck in analysis['bottlenecks']:
                    f.write(f"\n[{bottleneck['severity'].upper()}] {bottleneck['type']}\n")
                    f.write(f"  Endpoint: {bottleneck['endpoint']}\n")
                    f.write(f"  Recommendation: {bottleneck['recommendation']}\n")
            else:
                f.write("No significant bottlenecks identified.\n")
            
            # Scalability
            f.write("\n\nSCALABILITY ANALYSIS\n")
            f.write("-" * 40 + "\n")
            scalability = analysis['scalability']
            f.write(f"Max Throughput: {scalability['max_throughput_rps']:.2f} requests/second\n")
            f.write(f"Saturation Point: {scalability['saturation_point']}\n")
            
            # Reliability
            f.write("\n\nRELIABILITY ANALYSIS\n")
            f.write("-" * 40 + "\n")
            reliability = analysis['reliability']
            f.write(f"Endpoints Meeting SLA: {reliability['endpoints_meeting_sla']}/{reliability['endpoints_total']}\n")
            f.write(f"SLA Compliance: {reliability['sla_compliance_percent']:.2f}%\n")
            
            # Resource Efficiency
            f.write("\n\nRESOURCE EFFICIENCY\n")
            f.write("-" * 40 + "\n")
            efficiency = analysis['resource_efficiency']
            f.write(f"CPU Efficiency: {efficiency['cpu_efficiency']}\n")
            f.write(f"Memory Efficiency: {efficiency['memory_efficiency']}\n")
            
            if efficiency['recommendations']:
                f.write("\nRecommendations:\n")
                for rec in efficiency['recommendations']:
                    f.write(f"  • {rec}\n")
        
        logger.info(f"\n📄 Analysis report saved to: {report_file}")
        return report_file

def main():
    """Main analysis execution"""
    analyzer = PerformanceAnalyzer()
    
    try:
        # Load latest results
        results = analyzer.load_latest_results()
        
        # Perform analysis
        analysis = analyzer.analyze_results(results)
        
        # Generate report
        report_file = analyzer.generate_report(analysis)
        
        logger.info("\n✅ Performance analysis complete!")
        
    except FileNotFoundError:
        logger.info("❌ No performance test results found. Please run the performance test first.")
    except Exception as e:
        logger.info(f"❌ Error during analysis: {e}")

if __name__ == "__main__":
    main()