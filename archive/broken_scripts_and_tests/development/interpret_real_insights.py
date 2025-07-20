#!/usr/bin/env python3
"""
Script to interpret real insights from the running KIMERA system
"""

import requests
import json
from insight_interpretation_guide import KimeraInsightInterpreter

# Initialize structured logger
from backend.utils.kimera_logger import get_system_logger
logger = get_system_logger(__name__)


def interpret_real_insights():
    """Generate and interpret real insights from KIMERA"""
    logger.debug('🔍 INTERPRETING REAL KIMERA INSIGHTS')
    logger.info('='*50)
    
    # Initialize interpreter
    interpreter = KimeraInsightInterpreter()
    
    # Generate some actual insights
    logger.info('\n1. Generating insights from current geoids...')
    try:
        response = requests.post('http://localhost:8001/insights/auto_generate')
        if response.status_code == 200:
            result = response.json()
            logger.info(f'Generated {result.get("insights_generated", 0)
            
            # Get the insights
            insights_response = requests.get('http://localhost:8001/insights')
            if insights_response.status_code == 200:
                insights = insights_response.json()
                logger.info(f'Total insights available: {len(insights)
                
                if insights:
                    # Interpret the first few insights
                    for i, insight in enumerate(insights[:3], 1):
                        logger.info(f'\n🧠 REAL INSIGHT {i} INTERPRETATION:')
                        logger.info('-'*40)
                        logger.info(f'Raw insight data: {json.dumps(insight, indent=2)
                        logger.info('\n' + '-'*40)
                        
                        try:
                            report = interpreter.interpret_insight(insight)
                            logger.info(report)
                        except Exception as e:
                            logger.error(f'Error interpreting insight: {e}')
                            
                        logger.info('\n' + '='*50)
                else:
                    logger.info('No insights available for interpretation')
                    logger.info('Let me try to process some contradictions first...')
                    
                    # Try to generate some contradictions first
                    contra_response = requests.post('http://localhost:8001/process/contradictions/sync', 
                                                  json={"content": "Market volatility patterns in financial systems"})
                    if contra_response.status_code == 200:
                        contra_result = contra_response.json()
                        logger.info(f'Processed contradictions: {contra_result.get("contradictions_found", 0)
                        
                        # Try generating insights again
                        response2 = requests.post('http://localhost:8001/insights/auto_generate')
                        if response2.status_code == 200:
                            result2 = response2.json()
                            logger.info(f'Generated {result2.get("insights_generated", 0)
            else:
                logger.error(f'Failed to get insights: {insights_response.status_code}')
        else:
            logger.error(f'Failed to generate insights: {response.status_code}')
            
    except Exception as e:
        logger.error(f'Error: {e}')
        
    # Also demonstrate with a constructed example
    logger.info('\n\n🎯 DEMONSTRATION WITH CONSTRUCTED EXAMPLE:')
    logger.info('='*50)
    
    # Create a realistic example based on KIMERA's structure
    example_insight = {
        "insight_id": "INS_constructed_001",
        "insight_type": "complexity_analysis",
        "confidence": 0.72,
        "entropy_reduction": 0.15,
        "echoform_repr": {"content": "High semantic complexity detected in financial analysis"},
        "application_domains": ["understanding", "analysis"],
        "source_resonance_id": "understanding_engine",
        "utility_score": 0.108,
        "status": "provisional"
    }
    
    logger.info(f'Constructed insight: {json.dumps(example_insight, indent=2)
    logger.info('\n' + '-'*50)
    
    try:
        report = interpreter.interpret_insight(example_insight)
        logger.info(report)
    except Exception as e:
        logger.error(f'Error interpreting constructed insight: {e}')

if __name__ == "__main__":
    interpret_real_insights() 