#!/usr/bin/env python3
"""
KIMERA COMPETITIVE BENCHMARK RUNNER
===================================

This script runs the comprehensive competitive benchmark suite and generates
visual comparisons against industry standards.

Usage:
    python run_competitive_benchmark.py

Author: KIMERA Development Team
Date: 2025-01-27
"""

import asyncio
import sys
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime
import numpy as np

# Add the tests directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tests'))
from competitive_benchmark_suite import CompetitiveBenchmarkSuite

def create_competitive_visualizations(report_data: dict):
    """Create visual comparisons of benchmark results"""
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Extract data for visualization
    category_results = report_data["competitive_benchmark_report"]["category_results"]
    
    # 1. Overall Category Performance Comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('KIMERA vs Industry: Comprehensive Competitive Analysis', fontsize=16, fontweight='bold')
    
    # Category scores
    categories = list(category_results.keys())
    category_scores = [category_results[cat]["overall_score"] for cat in categories]
    
    # Plot 1: Category Performance Bar Chart
    ax1 = axes[0, 0]
    bars = ax1.bar(range(len(categories)), category_scores, 
                   color=['green' if score > 30 else 'orange' if score > 0 else 'red' for score in category_scores])
    ax1.set_title('Performance by Category\n(% Advantage over Industry Baseline)', fontweight='bold')
    ax1.set_xlabel('Categories')
    ax1.set_ylabel('Advantage Percentage (%)')
    ax1.set_xticks(range(len(categories)))
    ax1.set_xticklabels([cat.replace('_', ' ').title() for cat in categories], rotation=45, ha='right')
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars, category_scores)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + (1 if height >= 0 else -3),
                f'{score:.1f}%', ha='center', va='bottom' if height >= 0 else 'top', fontweight='bold')
    
    # Plot 2: Radar Chart for Key Metrics
    ax2 = axes[0, 1]
    
    # Collect key metrics for radar chart
    key_metrics = []
    kimera_scores = []
    industry_leaders = []
    
    for category in category_results.values():
        for metric in category["metrics"][:2]:  # Take first 2 metrics from each category
            key_metrics.append(metric["metric_name"])
            # Normalize scores to 0-100 scale for comparison
            if metric["unit"] == "ms":  # Lower is better for response time
                kimera_norm = max(0, 100 - (metric["kimera_score"] / 10))
                leader_norm = max(0, 100 - (metric["industry_leader"] / 10))
            elif metric["unit"] == "percentage" and "error" in metric["metric_name"].lower():
                # Lower is better for error rates
                kimera_norm = max(0, 100 - (metric["kimera_score"] * 10))
                leader_norm = max(0, 100 - (metric["industry_leader"] * 10))
            else:
                # Higher is better for most metrics
                kimera_norm = min(100, metric["kimera_score"] * 100)
                leader_norm = min(100, metric["industry_leader"] * 100)
            
            kimera_scores.append(kimera_norm)
            industry_leaders.append(leader_norm)
    
    # Create radar chart
    angles = np.linspace(0, 2 * np.pi, len(key_metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    kimera_scores += kimera_scores[:1]
    industry_leaders += industry_leaders[:1]
    
    ax2.plot(angles, kimera_scores, 'o-', linewidth=2, label='KIMERA', color='blue')
    ax2.fill(angles, kimera_scores, alpha=0.25, color='blue')
    ax2.plot(angles, industry_leaders, 'o-', linewidth=2, label='Industry Leader', color='red')
    ax2.fill(angles, industry_leaders, alpha=0.25, color='red')
    
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels([metric[:15] + '...' if len(metric) > 15 else metric for metric in key_metrics], 
                       fontsize=8)
    ax2.set_ylim(0, 100)
    ax2.set_title('KIMERA vs Industry Leaders\n(Key Metrics Comparison)', fontweight='bold')
    ax2.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
    ax2.grid(True)
    
    # Plot 3: Competitive Position Matrix
    ax3 = axes[1, 0]
    
    # Create scatter plot of competitive advantages
    x_data = []
    y_data = []
    colors = []
    labels = []
    
    for cat_name, category in category_results.items():
        for metric in category["metrics"]:
            x_data.append(metric["kimera_advantage"])
            y_data.append(metric["kimera_score"])
            
            if metric["competitive_position"] == "Leading":
                colors.append('green')
            elif metric["competitive_position"] == "Competitive":
                colors.append('orange')
            else:
                colors.append('red')
            
            labels.append(metric["metric_name"])
    
    scatter = ax3.scatter(x_data, y_data, c=colors, alpha=0.7, s=100)
    ax3.set_xlabel('Competitive Advantage (%)')
    ax3.set_ylabel('KIMERA Score')
    ax3.set_title('Competitive Position Matrix', fontweight='bold')
    ax3.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax3.grid(True, alpha=0.3)
    
    # Add quadrant labels
    ax3.text(max(x_data) * 0.7, max(y_data) * 0.9, 'Leading\nPosition', 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7),
             fontweight='bold', ha='center')
    
    # Plot 4: Market Opportunity Assessment
    ax4 = axes[1, 1]
    
    # Create a pie chart of competitive advantages
    advantage_categories = {'Leading (>30%)': 0, 'Competitive (0-30%)': 0, 'Behind (<0%)': 0}
    
    for category in category_results.values():
        if category["overall_score"] > 30:
            advantage_categories['Leading (>30%)'] += 1
        elif category["overall_score"] > 0:
            advantage_categories['Competitive (0-30%)'] += 1
        else:
            advantage_categories['Behind (<0%)'] += 1
    
    colors_pie = ['green', 'orange', 'red']
    wedges, texts, autotexts = ax4.pie(advantage_categories.values(), 
                                      labels=advantage_categories.keys(),
                                      colors=colors_pie,
                                      autopct='%1.0f%%',
                                      startangle=90)
    ax4.set_title('Competitive Position Distribution\n(Categories)', fontweight='bold')
    
    plt.tight_layout()
    
    # Save the visualization
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'kimera_competitive_analysis_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create detailed metrics comparison table
    create_detailed_comparison_table(category_results)

def create_detailed_comparison_table(category_results: dict):
    """Create a detailed comparison table"""
    
    # Prepare data for table
    table_data = []
    
    for cat_name, category in category_results.items():
        for metric in category["metrics"]:
            table_data.append({
                'Category': cat_name.replace('_', ' ').title(),
                'Metric': metric["metric_name"],
                'KIMERA Score': f"{metric['kimera_score']:.2f} {metric['unit']}",
                'Industry Baseline': f"{metric['industry_baseline']:.2f} {metric['unit']}",
                'Industry Leader': f"{metric['industry_leader']:.2f} {metric['unit']}",
                'Advantage': f"{metric['kimera_advantage']:.1f}%",
                'Position': metric["competitive_position"]
            })
    
    # Create DataFrame and save as CSV
    df = pd.DataFrame(table_data)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df.to_csv(f'kimera_detailed_comparison_{timestamp}.csv', index=False)
    
    print("\n" + "="*80)
    print("DETAILED COMPETITIVE COMPARISON TABLE")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)

def print_executive_summary(report_data: dict):
    """Print executive summary of benchmark results"""
    
    exec_summary = report_data["competitive_benchmark_report"]["executive_summary"]
    insights = report_data["competitive_benchmark_report"]["competitive_insights"]
    
    print("\n" + "🏆" * 20)
    print("KIMERA COMPETITIVE BENCHMARK RESULTS")
    print("🏆" * 20)
    
    print(f"\n📈 OVERALL COMPETITIVE ADVANTAGE: {exec_summary['overall_competitive_advantage']}")
    print(f"🎯 COMPETITIVE POSITION: {exec_summary['competitive_position']}")
    print(f"📊 TOTAL METRICS TESTED: {exec_summary['total_metrics_tested']}")
    
    print(f"\n🥇 LEADING CATEGORIES:")
    for category in exec_summary['leading_categories']:
        print(f"   • {category.replace('_', ' ').title()}")
    
    print(f"\n🏃 COMPETITIVE CATEGORIES:")
    for category in exec_summary['competitive_categories']:
        print(f"   • {category.replace('_', ' ').title()}")
    
    print(f"\n💪 KEY ADVANTAGES:")
    for advantage in insights['kimera_advantages'][:5]:
        print(f"   • {advantage}")
    
    print(f"\n🌟 MARKET OPPORTUNITIES:")
    for opportunity in insights['market_opportunities'][:3]:
        print(f"   • {opportunity}")
    
    print(f"\n🎯 STRATEGIC RECOMMENDATIONS:")
    for recommendation in insights['strategic_recommendations'][:3]:
        print(f"   • {recommendation}")
    
    print("\n" + "🏆" * 20)

async def main():
    """Main function to run competitive benchmarks"""
    
    print("🚀 STARTING KIMERA COMPETITIVE BENCHMARK SUITE")
    print("=" * 60)
    print("This comprehensive benchmark will test KIMERA against industry standards")
    print("across multiple dimensions and generate detailed competitive analysis.")
    print("=" * 60)
    
    # Initialize and run benchmark suite
    suite = CompetitiveBenchmarkSuite()
    
    print("\n⏱️  Running comprehensive benchmarks (this may take several minutes)...")
    results = await suite.run_complete_competitive_benchmark()
    
    if "error" in results:
        print(f"❌ Benchmark failed: {results['error']}")
        return
    
    print("\n✅ Benchmarks completed successfully!")
    
    # Print executive summary
    print_executive_summary(results)
    
    # Create visualizations
    print("\n📊 Generating competitive analysis visualizations...")
    try:
        create_competitive_visualizations(results)
        print("✅ Visualizations created successfully!")
    except Exception as e:
        print(f"⚠️  Could not create visualizations: {e}")
        print("   (This is normal if matplotlib is not installed)")
    
    # Print final summary
    exec_summary = results["competitive_benchmark_report"]["executive_summary"]
    print(f"\n🎉 BENCHMARK COMPLETE!")
    print(f"📈 KIMERA shows {exec_summary['overall_competitive_advantage']} advantage over industry baseline")
    print(f"🏆 Position: {exec_summary['competitive_position']}")
    
    # List generated files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n📁 Generated files:")
    print(f"   • competitive_benchmark_results_{timestamp}.json")
    print(f"   • competitive_summary_{timestamp}.txt")
    print(f"   • kimera_competitive_analysis_{timestamp}.png (if matplotlib available)")
    print(f"   • kimera_detailed_comparison_{timestamp}.csv")

if __name__ == "__main__":
    asyncio.run(main()) 