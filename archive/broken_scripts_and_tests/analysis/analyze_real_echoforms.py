#!/usr/bin/env python3
"""
Real EchoForm Analysis for KIMERA Insights
==========================================

This script fetches real insights from KIMERA and analyzes their EchoForm representations
to reveal the deep cognitive structures behind the system's thinking.
"""

import requests
import json
from typing import Dict, Any, List
from echoform_analysis import EchoFormAnalyzer

# Initialize structured logger
from backend.utils.kimera_logger import get_system_logger
logger = get_system_logger(__name__)


def fetch_kimera_insights() -> List[Dict[str, Any]]:
    """Fetch real insights from KIMERA system"""
    try:
        # Try to fetch insights from KIMERA API
        response = requests.get("http://localhost:8001/insights/recent", timeout=10)
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and 'insights' in data:
                return data['insights']
            else:
                logger.warning("⚠️  Unexpected response format from KIMERA")
                return []
        else:
            logger.warning(f"⚠️  KIMERA API returned status {response.status_code}")
            return []
    except requests.exceptions.RequestException as e:
        logger.warning(f"⚠️  Could not connect to KIMERA: {e}")
        return []

def analyze_real_echoforms():
    """Analyze real EchoForm representations from KIMERA insights"""
    logger.debug("🔍 ANALYZING REAL ECHOFORMS FROM KIMERA")
    logger.info("="*60)
    logger.info("\nFetching insights from KIMERA system...")
    
    insights = fetch_kimera_insights()
    
    if not insights:
        logger.error("\n❌ No insights available from KIMERA system.")
        logger.info("   Make sure KIMERA is running on localhost:8001")
        logger.info("   and has generated some insights.\n")
        
        # Show demonstration with synthetic examples instead
        logger.info("📚 SHOWING DEMONSTRATION WITH SYNTHETIC EXAMPLES:")
        logger.info("-" * 50)
        demonstrate_with_examples()
        return
    
    logger.info(f"\n✅ Found {len(insights)
    logger.info("   Analyzing their EchoForm representations...\n")
    
    analyzer = EchoFormAnalyzer()
    
    for i, insight in enumerate(insights[:5], 1):  # Analyze first 5 insights
        logger.info(f"\n🧠 REAL INSIGHT {i}")
        logger.info("-" * 50)
        
        # Extract basic insight information
        insight_id = insight.get('insight_id', 'Unknown')
        insight_type = insight.get('insight_type', 'Unknown')
        confidence = insight.get('confidence', 0.0)
        entropy_reduction = insight.get('entropy_reduction', 0.0)
        
        logger.info(f"ID: {insight_id}")
        logger.info(f"Type: {insight_type}")
        logger.info(f"Confidence: {confidence:.3f}")
        logger.info(f"Entropy Reduction: {entropy_reduction:.3f}")
        
        # Analyze the EchoForm representation
        echoform_repr = insight.get('echoform_repr', {})
        
        if not echoform_repr:
            logger.error("❌ No EchoForm representation found in this insight")
            continue
        
        logger.debug(f"\n🔍 ECHOFORM STRUCTURE:")
        logger.info(f"   Raw EchoForm: {json.dumps(echoform_repr, indent=2)
        
        # Perform detailed analysis
        analysis = analyzer.analyze_echoform(echoform_repr)
        
        if 'error' in analysis:
            logger.error(f"❌ Analysis error: {analysis['error']}")
            continue
        
        # Display analysis results
        logger.info(f"\n📊 STRUCTURAL ANALYSIS:")
        struct = analysis['structure_analysis']
        logger.info(f"   Completeness: {struct['completeness']}")
        logger.info(f"   Components: {', '.join(struct['components'])
        logger.info(f"   Complexity Score: {struct['complexity_score']}")
        
        logger.debug(f"\n🔍 CONCEPT ANALYSIS:")
        concepts = analysis['concept_analysis']
        if 'error' not in concepts:
            logger.info(f"   Primary Concepts: {', '.join(concepts['primary_concepts'])
            logger.info(f"   Dominant Concept: {concepts['dominant_concept']}")
            if isinstance(concepts['concept_weights'], dict):
                logger.info(f"   Concept Weights: {concepts['concept_weights']}")
            logger.info(f"   Semantic Density: {concepts['semantic_density']}")
        
        logger.debug(f"\n🎭 ARCHETYPE ANALYSIS:")
        archetype = analysis['archetype_analysis']
        if 'error' not in archetype:
            logger.info(f"   Archetype: {archetype['archetype_name']}")
            logger.info(f"   Meaning: {archetype['archetype_meaning']}")
            logger.info(f"   Cognitive Pattern: {archetype['cognitive_pattern']}")
            logger.info(f"   Family: {archetype['archetypal_family']}")
        
        logger.info(f"\n⚡ PARADOX ANALYSIS:")
        paradox = analysis['paradox_analysis']
        if 'error' not in paradox:
            logger.info(f"   Paradox: {paradox['paradox_statement']}")
            logger.info(f"   Type: {paradox['paradox_type']}")
            logger.info(f"   Dynamics: {paradox['tension_dynamics']}")
            logger.info(f"   Challenge: {paradox['cognitive_challenge']}")
        
        logger.info(f"\n🧩 COGNITIVE SIGNATURE:")
        signature = analysis['cognitive_signature']
        logger.info(f"   Cognitive Type: {signature['cognitive_type']}")
        logger.info(f"   Processing Style: {signature['processing_style']}")
        logger.info(f"   Abstraction Level: {signature['abstraction_level']}")
        logger.info(f"   Complexity Class: {signature['complexity_class']}")
        
        logger.info(f"\n💡 INTERPRETATION:")
        logger.info(f"   {analysis['interpretation_summary']}")
        
        logger.info("\n" + "="*60)

def demonstrate_with_examples():
    """Demonstrate EchoForm analysis with synthetic examples when real data unavailable"""
    
    # These are realistic examples based on KIMERA's actual structure
    synthetic_insights = [
        {
            "insight_id": "INS_demo_001",
            "insight_type": "ANALOGY",
            "confidence": 0.82,
            "entropy_reduction": 0.28,
            "echoform_repr": {
                "type": "ANALOGY",
                "core_concept": {"financial_volatility": 0.85, "market_psychology": 0.78},
                "archetype": "The Stampede",
                "paradox": "Individual rationality creates collective irrationality"
            }
        },
        {
            "insight_id": "INS_demo_002", 
            "insight_type": "HYPOTHESIS",
            "confidence": 0.71,
            "entropy_reduction": 0.19,
            "echoform_repr": {
                "type": "HYPOTHESIS",
                "core_concept": {"atmospheric_pressure": 0.73, "cascade_effects": 0.69},
                "archetype": "The Hidden Trigger",
                "paradox": "Small changes create large effects"
            }
        },
        {
            "insight_id": "INS_demo_003",
            "insight_type": "META_FRAMEWORK",
            "confidence": 0.76,
            "entropy_reduction": 0.33,
            "echoform_repr": {
                "type": "META_FRAMEWORK", 
                "core_concept": {"bias_detection": 0.81, "cognitive_monitoring": 0.75},
                "archetype": "The Self-Critic",
                "paradox": "Awareness of bias can create new biases"
            }
        }
    ]
    
    analyzer = EchoFormAnalyzer()
    
    for i, insight in enumerate(synthetic_insights, 1):
        logger.info(f"\n🧠 DEMONSTRATION INSIGHT {i}")
        logger.info("-" * 50)
        
        # Extract basic insight information
        insight_id = insight['insight_id']
        insight_type = insight['insight_type']
        confidence = insight['confidence']
        entropy_reduction = insight['entropy_reduction']
        
        logger.info(f"ID: {insight_id}")
        logger.info(f"Type: {insight_type}")
        logger.info(f"Confidence: {confidence:.3f}")
        logger.info(f"Entropy Reduction: {entropy_reduction:.3f}")
        
        # Analyze the EchoForm representation
        echoform_repr = insight['echoform_repr']
        
        logger.debug(f"\n🔍 ECHOFORM STRUCTURE:")
        logger.info(f"   Raw EchoForm: {json.dumps(echoform_repr, indent=2)
        
        # Perform detailed analysis
        analysis = analyzer.analyze_echoform(echoform_repr)
        
        # Display key analysis results
        logger.info(f"\n📊 KEY FINDINGS:")
        struct = analysis['structure_analysis']
        concepts = analysis['concept_analysis']
        archetype = analysis['archetype_analysis']
        paradox = analysis['paradox_analysis']
        signature = analysis['cognitive_signature']
        
        logger.info(f"   • Structure: {struct['completeness']}")
        logger.info(f"   • Dominant Concept: {concepts['dominant_concept']}")
        logger.info(f"   • Archetype: {archetype['archetype_name']} ({archetype['cognitive_pattern']})
        logger.info(f"   • Paradox Type: {paradox['paradox_type']}")
        logger.info(f"   • Processing Style: {signature['processing_style']}")
        
        logger.info(f"\n💡 WHAT THE ECHOFORM REVEALS:")
        logger.info(f"   {analysis['interpretation_summary']}")
        
        logger.info("\n" + "="*60)

def main():
    """Main function to run EchoForm analysis"""
    logger.info("🚀 Starting Real EchoForm Analysis...")
    analyze_real_echoforms()
    
    logger.info(f"\n📚 For more detailed analysis tools, see:")
    logger.info(f"   • echoform_analysis.py - Technical analysis framework")
    logger.info(f"   • ECHOFORM_INTERPRETATION_GUIDE.md - Comprehensive guide")
    logger.info(f"   • insight_interpretation_guide.py - Practical interpretation tools")

if __name__ == "__main__":
    main() 