#!/usr/bin/env python3
"""
Debug Test for Enterprise Trading Components

This script debugs the specific attribute issues in the enterprise trading components.
"""

import sys
import traceback
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_smart_order_router():
    """Debug Smart Order Router attributes"""
    print("=== Debugging Smart Order Router ===")
    try:
        from src.trading.enterprise.smart_order_router import SmartOrderRouter
        router = SmartOrderRouter()
        
        print(f"Router initialized successfully")
        print(f"Router attributes: {dir(router)}")
        
        # Check specific attributes
        print(f"Has venues: {hasattr(router, 'venues')}")
        print(f"Has routing_engine: {hasattr(router, 'routing_engine')}")
        
        if hasattr(router, 'venues'):
            print(f"Venues: {router.venues}")
        if hasattr(router, 'routing_engine'):
            print(f"Routing engine: {router.routing_engine}")
            
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

def debug_market_microstructure_analyzer():
    """Debug Market Microstructure Analyzer attributes"""
    print("\n=== Debugging Market Microstructure Analyzer ===")
    try:
        from src.trading.enterprise.market_microstructure_analyzer import MarketMicrostructureAnalyzer
        analyzer = MarketMicrostructureAnalyzer()
        
        print(f"Analyzer initialized successfully")
        print(f"Analyzer attributes: {dir(analyzer)}")
        
        # Check specific attributes
        print(f"Has order_book_reconstructor: {hasattr(analyzer, 'order_book_reconstructor')}")
        print(f"Has liquidity_analyzer: {hasattr(analyzer, 'liquidity_analyzer')}")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

def debug_regulatory_compliance_engine():
    """Debug Regulatory Compliance Engine attributes"""
    print("\n=== Debugging Regulatory Compliance Engine ===")
    try:
        from src.trading.enterprise.regulatory_compliance_engine import RegulatoryComplianceEngine
        compliance = RegulatoryComplianceEngine()
        
        print(f"Compliance engine initialized successfully")
        print(f"Compliance attributes: {dir(compliance)}")
        
        # Check specific attributes
        print(f"Has compliance_rules: {hasattr(compliance, 'compliance_rules')}")
        print(f"Has jurisdictions: {hasattr(compliance, 'jurisdictions')}")
        
        if hasattr(compliance, 'compliance_rules'):
            print(f"Compliance rules: {len(compliance.compliance_rules)} rules")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

def debug_hft_infrastructure():
    """Debug HFT Infrastructure attributes"""
    print("\n=== Debugging HFT Infrastructure ===")
    try:
        from src.trading.enterprise.hft_infrastructure import HFTInfrastructure
        hft = HFTInfrastructure()
        
        print(f"HFT infrastructure initialized successfully")
        print(f"HFT attributes: {dir(hft)}")
        
        # Check specific attributes
        print(f"Has latency_monitor: {hasattr(hft, 'latency_monitor')}")
        print(f"Has execution_engine: {hasattr(hft, 'execution_engine')}")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

def debug_integrated_trading_system():
    """Debug Integrated Trading System attributes"""
    print("\n=== Debugging Integrated Trading System ===")
    try:
        from src.engines.cognitive_field_dynamics import CognitiveFieldDynamics
        from src.engines.thermodynamic_engine import ThermodynamicEngine
        from src.engines.contradiction_engine import ContradictionEngine
        from src.trading.enterprise.integrated_trading_system import IntegratedTradingSystem
        
        # Create components
        cognitive_field = CognitiveFieldDynamics(dimension=128)
        thermodynamic_engine = ThermodynamicEngine()
        contradiction_engine = ContradictionEngine()
        
        print(f"Components created successfully")
        
        # Create integrated system
        integrated_system = IntegratedTradingSystem(
            cognitive_field=cognitive_field,
            thermodynamic_engine=thermodynamic_engine,
            contradiction_engine=contradiction_engine
        )
        
        print(f"Integrated system initialized successfully")
        print(f"Integrated system attributes: {dir(integrated_system)}")
        
        # Check specific attributes
        print(f"Has cognitive_field: {hasattr(integrated_system, 'cognitive_field')}")
        print(f"Has components: {hasattr(integrated_system, 'components')}")
        
        if hasattr(integrated_system, 'components'):
            print(f"Components: {integrated_system.components}")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    print("Starting Enterprise Trading Components Debug")
    print("=" * 80)
    
    results = []
    
    # Debug each component
    results.append(debug_smart_order_router())
    results.append(debug_market_microstructure_analyzer())
    results.append(debug_regulatory_compliance_engine())
    results.append(debug_hft_infrastructure())
    results.append(debug_integrated_trading_system())
    
    print("\n" + "=" * 80)
    print("DEBUG SUMMARY")
    print("=" * 80)
    
    successful = sum(results)
    total = len(results)
    
    print(f"Successful debugs: {successful}/{total}")
    
    if successful == total:
        print("✅ All components can be debugged successfully!")
    else:
        print(f"❌ {total - successful} components have issues") 