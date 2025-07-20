"""
Text-based Understanding Progress Monitor
Tracks KIMERA's journey toward genuine understanding
"""

import sys
import os

# Initialize structured logger
from backend.utils.kimera_logger import get_system_logger
logger = get_system_logger(__name__)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.vault.understanding_vault_manager import UnderstandingVaultManager
import sqlite3
from datetime import datetime
import time


def get_understanding_stats():
    """Get comprehensive understanding statistics"""
    vault = UnderstandingVaultManager()
    metrics = vault.get_understanding_metrics()
    
    # Get additional database stats
    conn = sqlite3.connect('kimera_swm.db')
    cursor = conn.cursor()
    
    # Confidence distribution
    cursor.execute("""
        SELECT 
            COUNT(CASE WHEN confidence_score >= 0.8 THEN 1 END) as high,
            COUNT(CASE WHEN confidence_score >= 0.6 AND confidence_score < 0.8 THEN 1 END) as medium,
            COUNT(CASE WHEN confidence_score < 0.6 THEN 1 END) as low,
            AVG(confidence_score) as average
        FROM multimodal_groundings
    """)
    conf_stats = cursor.fetchone()
    
    # Recent activity
    cursor.execute("""
        SELECT COUNT(*) FROM multimodal_groundings 
        WHERE created_at > datetime('now', '-1 hour')
    """)
    recent_groundings = cursor.fetchone()[0]
    
    cursor.execute("""
        SELECT COUNT(*) FROM causal_relationships 
        WHERE created_at > datetime('now', '-1 hour')
    """)
    recent_causal = cursor.fetchone()[0]
    
    # Top grounded concepts
    cursor.execute("""
        SELECT concept_id, confidence_score 
        FROM multimodal_groundings 
        ORDER BY confidence_score DESC 
        LIMIT 5
    """)
    top_concepts = cursor.fetchall()
    
    conn.close()
    
    return {
        'metrics': metrics,
        'confidence': {
            'high': conf_stats[0] or 0,
            'medium': conf_stats[1] or 0,
            'low': conf_stats[2] or 0,
            'average': conf_stats[3] or 0
        },
        'recent': {
            'groundings': recent_groundings,
            'causal': recent_causal
        },
        'top_concepts': top_concepts
    }


def display_progress_bar(value, max_value=1.0, width=30, label=""):
    """Display a text progress bar"""
    filled = int(width * value / max_value)
    bar = "█" * filled + "░" * (width - filled)
    percentage = value / max_value * 100
    return f"{label:20} [{bar}] {percentage:5.1f}%"


def display_understanding_monitor():
    """Display the understanding monitor"""
    while True:
        # Clear screen (works on most terminals)
        os.system('cls' if os.name == 'nt' else 'clear')
        
        stats = get_understanding_stats()
        metrics = stats['metrics']
        
        # Header
        logger.info("╔" + "═"*78 + "╗")
        logger.info("║" + " "*25 + "KIMERA UNDERSTANDING MONITOR" + " "*25 + "║")
        logger.info("║" + " "*78 + "║")
        logger.info("║" + f"  Last Update: {datetime.now()
        logger.info("╚" + "═"*78 + "╝")
        
        # Roadmap Progress
        logger.info("\n📊 ROADMAP PROGRESS")
        logger.info("─" * 80)
        for phase, progress in metrics['roadmap_progress'].items():
            phase_name = phase.replace('_', ' ').title()
            logger.info(display_progress_bar(progress, label=phase_name)
        
        # Understanding Components
        logger.info("\n📈 UNDERSTANDING COMPONENTS")
        logger.info("─" * 80)
        components = metrics['understanding_components']
        logger.info(f"  Multimodal Groundings: {components['multimodal_groundings']:>6}")
        logger.info(f"  Causal Relationships:  {components['causal_relationships']:>6}")
        logger.info(f"  Self Models:           {components['self_models']:>6}")
        logger.info(f"  Abstract Concepts:     {components['abstract_concepts']:>6}")
        logger.info(f"  Genuine Opinions:      {components['genuine_opinions']:>6}")
        logger.info(f"  Learned Values:        {components['learned_values']:>6}")
        
        # Confidence Distribution
        logger.info("\n🎯 GROUNDING CONFIDENCE")
        logger.info("─" * 80)
        conf = stats['confidence']
        total = conf['high'] + conf['medium'] + conf['low']
        if total > 0:
            logger.info(f"  High (≥0.8)
            logger.info(f"  Medium (0.6-0.8)
            logger.info(f"  Low (<0.6)
            logger.info(f"  Average Score:   {conf['average']:>4.3f}")
        
        # Understanding Quality
        logger.info("\n✅ UNDERSTANDING QUALITY")
        logger.info("─" * 80)
        quality = metrics['understanding_quality']
        logger.info(f"  Test Accuracy:    {quality['average_test_accuracy']:>5.1%}")
        logger.info(f"  Tests Passed:     {quality['tests_passed']}/{quality['total_tests']}")
        logger.info(f"  Quality Score:    {quality['average_understanding_quality']:>5.1%}")
        
        # Recent Activity
        logger.info("\n🔄 RECENT ACTIVITY (Last Hour)
        logger.info("─" * 80)
        logger.info(f"  New Groundings:        {stats['recent']['groundings']:>4}")
        logger.info(f"  Causal Discoveries:    {stats['recent']['causal']:>4}")
        
        # Top Concepts
        if stats['top_concepts']:
            logger.info("\n🏆 TOP GROUNDED CONCEPTS")
            logger.info("─" * 80)
            for i, (concept, confidence) in enumerate(stats['top_concepts'], 1):
                logger.info(f"  {i}. {concept:<30} (confidence: {confidence:.3f})
        
        # Status Summary
        logger.info("\n📋 STATUS SUMMARY")
        logger.info("─" * 80)
        
        # Calculate overall progress
        overall_progress = sum(metrics['roadmap_progress'].values()) / len(metrics['roadmap_progress'])
        
        if overall_progress < 0.25:
            status = "🟡 Early Stage - Building Foundation"
        elif overall_progress < 0.5:
            status = "🟢 Progressing - Developing Understanding"
        elif overall_progress < 0.75:
            status = "🔵 Advanced - Deepening Comprehension"
        else:
            status = "🟣 Mature - Approaching Genuine Understanding"
        
        logger.info(f"  Status: {status}")
        logger.info(f"  Overall Progress: {overall_progress:.1%}")
        
        # Recommendations
        logger.info("\n💡 RECOMMENDATIONS")
        logger.info("─" * 80)
        
        if components['multimodal_groundings'] < 100:
            logger.info("  • Ground more concepts to build semantic foundation")
        
        if conf['average'] < 0.7:
            logger.info("  • Focus on improving grounding quality (target: 0.7+)
        
        if components['causal_relationships'] < 20:
            logger.info("  • Discover more causal relationships for reasoning")
        
        if components['abstract_concepts'] < 10:
            logger.info("  • Form abstract concepts to generalize understanding")
        
        if quality['average_test_accuracy'] < 0.7:
            logger.info("  • Improve understanding mechanisms (target: 70% accuracy)
        
        # Footer
        logger.info("\n" + "─" * 80)
        logger.info("Press Ctrl+C to exit | Updates every 5 seconds")
        
        # Wait before refresh
        time.sleep(5)


def generate_report():
    """Generate a one-time report"""
    stats = get_understanding_stats()
    metrics = stats['metrics']
    
    logger.info("\n" + "="*80)
    logger.info("KIMERA UNDERSTANDING PROGRESS REPORT")
    logger.info(f"Generated: {datetime.now()
    logger.info("="*80)
    
    # Overall Progress
    overall = sum(metrics['roadmap_progress'].values()) / len(metrics['roadmap_progress'])
    logger.info(f"\n🎯 OVERALL PROGRESS: {overall:.1%}")
    
    # Detailed Progress
    logger.info("\n📊 PHASE PROGRESS:")
    for phase, progress in metrics['roadmap_progress'].items():
        logger.info(f"  {phase}: {progress:.1%}")
    
    # Key Metrics
    logger.info(f"\n📈 KEY METRICS:")
    logger.info(f"  Total Groundings: {metrics['understanding_components']['multimodal_groundings']}")
    logger.info(f"  Causal Relations: {metrics['understanding_components']['causal_relationships']}")
    logger.info(f"  Confidence Average: {stats['confidence']['average']:.3f}")
    logger.info(f"  Test Accuracy: {metrics['understanding_quality']['average_test_accuracy']:.1%}")
    
    # Next Milestones
    logger.info("\n🎯 NEXT MILESTONES:")
    
    groundings_needed = max(0, 100 - metrics['understanding_components']['multimodal_groundings'])
    if groundings_needed > 0:
        logger.info(f"  • Ground {groundings_needed} more concepts to reach 100")
    
    causal_needed = max(0, 50 - metrics['understanding_components']['causal_relationships'])
    if causal_needed > 0:
        logger.info(f"  • Discover {causal_needed} more causal relationships")
    
    if metrics['understanding_components']['abstract_concepts'] < 10:
        logger.info(f"  • Form {10 - metrics['understanding_components']['abstract_concepts']} abstract concepts")
    
    logger.info("\n" + "="*80)


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitor KIMERA Understanding Progress')
    parser.add_argument('--live', action='store_true', help='Show live monitoring dashboard')
    parser.add_argument('--report', action='store_true', help='Generate one-time report')
    
    args = parser.parse_args()
    
    if args.live:
        logger.info("Starting live monitoring... (Press Ctrl+C to exit)
        try:
            display_understanding_monitor()
        except KeyboardInterrupt:
            logger.info("\n\nMonitoring stopped.")
    else:
        generate_report()


if __name__ == "__main__":
    main()