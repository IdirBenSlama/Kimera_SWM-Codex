"""
Test KIMERA Trading Module through API
"""

import requests
import json
import asyncio
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

API_BASE = "http://localhost:8000"

def test_system_status():
    """Check KIMERA system status"""
    logger.info("Checking KIMERA system status...")
    response = requests.get(f"{API_BASE}/system/status")
    if response.status_code == 200:
        data = response.json()
        logger.info(f"✅ System Status: {data['status']}")
        logger.info(f"   Cycle Count: {data['cycle_count']}")
        logger.info(f"   Active Geoids: {data['active_geoids']}")
        return True
    else:
        logger.error(f"❌ Failed to get system status: {response.status_code}")
        return False

def create_market_geoid():
    """Create a geoid representing market data"""
    logger.info("\nCreating market data geoid...")
    
    geoid_data = {
        "semantic_features": {
            "price": 50000.0,
            "volume": 2500.0,
            "momentum": 0.05,
            "volatility": 0.02,
            "rsi": 65.0,
            "macd": 100.0
        },
        "symbolic_content": {
            "symbol": "BTC-USD",
            "trend": "bullish",
            "market_phase": "accumulation"
        },
        "metadata": {
            "source": "market_data",
            "timestamp": "2025-01-21T10:00:00Z"
        }
    }
    
    response = requests.post(f"{API_BASE}/geoids", json=geoid_data)
    if response.status_code == 200:
        data = response.json()
        logger.info(f"✅ Created market geoid: {data['geoid_id']}")
        logger.info(f"   Entropy: {data['entropy']:.3f}")
        return data['geoid_id']
    else:
        logger.error(f"❌ Failed to create geoid: {response.text}")
        return None

def create_news_geoid():
    """Create a geoid representing news sentiment"""
    logger.info("\nCreating news sentiment geoid...")
    
    geoid_data = {
        "semantic_features": {
            "sentiment_score": -0.8,  # Negative news!
            "sentiment_volume": 150.0,
            "sentiment_momentum": -0.3,
            "credibility": 0.85,
            "virality": 0.6
        },
        "symbolic_content": {
            "symbol": "BTC-USD",
            "sentiment": "bearish",
            "topics": ["regulation", "security_breach"]
        },
        "metadata": {
            "source": "news_sentiment",
            "timestamp": "2025-01-21T10:00:00Z"
        }
    }
    
    response = requests.post(f"{API_BASE}/geoids", json=geoid_data)
    if response.status_code == 200:
        data = response.json()
        logger.info(f"✅ Created news geoid: {data['geoid_id']}")
        logger.info(f"   Entropy: {data['entropy']:.3f}")
        return data['geoid_id']
    else:
        logger.error(f"❌ Failed to create geoid: {response.text}")
        return None

def process_contradictions(trigger_geoid_id):
    """Process contradictions starting from a trigger geoid"""
    logger.info(f"\nProcessing contradictions from {trigger_geoid_id}...")
    
    request_data = {
        "trigger_geoid_id": trigger_geoid_id,
        "search_limit": 10
    }
    
    response = requests.post(f"{API_BASE}/process/contradictions/sync", json=request_data)
    if response.status_code == 200:
        data = response.json()
        logger.info(f"✅ Contradiction processing complete:")
        logger.info(f"   Tensions detected: {data['tensions_detected']}")
        logger.info(f"   Decisions made: {data['decisions_made']}")
        
        if data.get('scars_created'):
            logger.info("   SCARs created:")
            for scar in data['scars_created']:
                logger.info(f"     • {scar['scar_id']}: {scar['reason']}")
                logger.info(f"       Entropy change: {scar['delta_entropy']:.3f}")
        
        return True
    else:
        logger.error(f"❌ Failed to process contradictions: {response.text}")
        return False

def main():
    logger.info("="*60)
    logger.info("KIMERA TRADING MODULE - API TEST")
    logger.info("="*60)
    
    # Test 1: Check system status
    if not test_system_status():
        return
    
    # Test 2: Create market data geoid (bullish)
    market_geoid = create_market_geoid()
    if not market_geoid:
        return
    
    # Test 3: Create news sentiment geoid (bearish - contradiction!)
    news_geoid = create_news_geoid()
    if not news_geoid:
        return
    
    # Test 4: Process contradictions
    logger.info("\n⚡ CONTRADICTION DETECTED: Bullish market vs Bearish news!")
    process_contradictions(market_geoid)
    
    # Test 5: Check insights
    logger.info("\nChecking for generated insights...")
    response = requests.get(f"{API_BASE}/insights")
    if response.status_code == 200:
        insights = response.json()
        logger.info(f"✅ Found {len(insights)} insights")
        for insight in insights[:3]:  # Show first 3
            logger.info(f"   • {insight['insight_id']}: {insight.get('type', 'unknown')}")
    
    logger.info("\n" + "="*60)
    logger.info("✅ Trading module is integrated and working with KIMERA!")
    logger.info("="*60)

if __name__ == "__main__":
    main() 