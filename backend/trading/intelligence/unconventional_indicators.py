"""
Unconventional Market Indicators
Analyzes seemingly unrelated fields that can influence crypto markets
"""

import logging
import asyncio
import aiohttp
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

class UnconventionalIndicators:
    """Analyzes unconventional data sources that influence crypto markets"""
    
    def __init__(self):
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def get_space_weather_data(self) -> Dict[str, Any]:
        """
        Solar activity affects satellite communications and power grids
        Major solar flares can disrupt mining operations and internet infrastructure
        """
        try:
            # NOAA Space Weather data
            url = "https://services.swpc.noaa.gov/json/planetary_k_index_1m.json"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    latest = data[-1] if data else {}
                    
                    return {
                        'kp_index': float(latest.get('kp', 0)),  # Geomagnetic activity
                        'impact_level': self._assess_space_weather_impact(float(latest.get('kp', 0))),
                        'mining_risk': 'high' if float(latest.get('kp', 0)) > 6 else 'low'
                    }
        except Exception as e:
            logger.warning(f"Space weather data unavailable: {e}")
            
        return {'kp_index': 0, 'impact_level': 'unknown', 'mining_risk': 'low'}

    def _assess_space_weather_impact(self, kp_index: float) -> str:
        """Assess impact of space weather on crypto infrastructure"""
        if kp_index >= 7:
            return 'severe'  # Major infrastructure risk
        elif kp_index >= 5:
            return 'moderate'  # Some mining/communication issues
        elif kp_index >= 3:
            return 'minor'  # Minimal impact
        else:
            return 'none'  # No impact

    async def get_all_indicators(self) -> Dict[str, Any]:
        """Gather all unconventional indicators"""
        try:
            async with self:
                # For now, return simulated data
                return {
                    'space_weather': {
                        'kp_index': 2.3,
                        'impact_level': 'none',
                        'mining_risk': 'low'
                    },
                    'internet_health': {
                        'global_connectivity': 0.98,
                        'major_outages': 0,
                        'routing_anomalies': 2
                    },
                    'academic_research': {
                        'paper_velocity': 45,
                        'citation_growth': 0.12,
                        'hot_topics': ['zero-knowledge proofs', 'layer 2 scaling']
                    },
                    'energy_markets': {
                        'electricity_price_trend': 'decreasing',
                        'renewable_percentage': 0.65,
                        'mining_profitability_index': 1.2
                    },
                    'developer_activity': {
                        'github_commits_trend': 'increasing',
                        'new_developers': 1250,
                        'developer_sentiment': 'bullish'
                    },
                    'gaming_metaverse': {
                        'nft_gaming_volume': 125000000,
                        'metaverse_dau': 2500000,
                        'mainstream_gaming_adoption': 'growing'
                    },
                    'supply_chain': {
                        'semiconductor_availability': 'improving',
                        'shipping_cost_index': 0.85,
                        'mining_hardware_lead_time': 8
                    },
                    'social_psychology': {
                        'social_media_engagement': 'declining',
                        'fomo_indicator': 0.3,
                        'collective_mood': 'cautious_optimism'
                    },
                    'regulatory_intelligence': {
                        'regulatory_momentum': 'neutral',
                        'policy_uncertainty_index': 0.4,
                        'lobbying_expenditure': 45000000
                    },
                    'quantum_threat': {
                        'quantum_threat_timeline': '2035-2040',
                        'current_threat_level': 'minimal',
                        'crypto_vulnerability': 'low'
                    },
                    'timestamp': datetime.now().isoformat()
                }
        except Exception as e:
            logger.error(f"Error gathering unconventional indicators: {e}")
            return {}

    def calculate_unconventional_score(self, indicators: Dict[str, Any]) -> float:
        """
        Calculate a composite score from unconventional indicators
        Returns: -1.0 (very bearish) to +1.0 (very bullish)
        """
        try:
            score = 0.0
            factors = 0
            
            # Space weather impact (negative = bad for infrastructure)
            if 'space_weather' in indicators:
                kp = indicators['space_weather'].get('kp_index', 0)
                if kp > 6:
                    score -= 0.2  # Major solar activity = infrastructure risk
                factors += 1
            
            # Internet health (positive = good for trading)
            if 'internet_health' in indicators:
                connectivity = indicators['internet_health'].get('global_connectivity', 1.0)
                score += (connectivity - 0.95) * 2  # Above 95% is good
                factors += 1
            
            # Academic research momentum (positive = future innovation)
            if 'academic_research' in indicators:
                momentum = indicators['academic_research'].get('citation_growth', 0)
                score += min(momentum * 2, 0.3)  # Cap at 0.3
                factors += 1
            
            # Energy market favorability (lower costs = better for mining)
            if 'energy_markets' in indicators:
                trend = indicators['energy_markets'].get('electricity_price_trend', 'stable')
                if trend == 'decreasing':
                    score += 0.2
                elif trend == 'increasing':
                    score -= 0.2
                factors += 1
            
            # Developer activity (positive = ecosystem growth)
            if 'developer_activity' in indicators:
                trend = indicators['developer_activity'].get('github_commits_trend', 'stable')
                if trend == 'increasing':
                    score += 0.15
                elif trend == 'decreasing':
                    score -= 0.15
                factors += 1
            
            # Gaming/metaverse adoption (positive = utility growth)
            if 'gaming_metaverse' in indicators:
                adoption = indicators['gaming_metaverse'].get('mainstream_gaming_adoption', 'stable')
                if adoption == 'growing':
                    score += 0.1
                elif adoption == 'declining':
                    score -= 0.1
                factors += 1
            
            # Supply chain health (positive = hardware availability)
            if 'supply_chain' in indicators:
                availability = indicators['supply_chain'].get('semiconductor_availability', 'stable')
                if availability == 'improving':
                    score += 0.1
                elif availability == 'worsening':
                    score -= 0.1
                factors += 1
            
            # Social psychology (FOMO vs fear)
            if 'social_psychology' in indicators:
                fomo = indicators['social_psychology'].get('fomo_indicator', 0.5)
                score += (fomo - 0.5) * 0.4  # FOMO can be bullish short-term
                factors += 1
            
            # Regulatory clarity (positive = less uncertainty)
            if 'regulatory_intelligence' in indicators:
                uncertainty = indicators['regulatory_intelligence'].get('policy_uncertainty_index', 0.5)
                score += (0.5 - uncertainty) * 0.3  # Lower uncertainty is better
                factors += 1
            
            # Quantum threat (negative = existential risk)
            if 'quantum_threat' in indicators:
                threat_level = indicators['quantum_threat'].get('current_threat_level', 'minimal')
                if threat_level == 'severe':
                    score -= 0.5
                elif threat_level == 'moderate':
                    score -= 0.2
                factors += 1
            
            # Normalize by number of factors
            if factors > 0:
                score = score / factors
            
            # Clamp to [-1, 1]
            return max(-1.0, min(1.0, score))
            
        except Exception as e:
            logger.error(f"Error calculating unconventional score: {e}")
            return 0.0


# Comprehensive list of unconventional fields that could influence crypto markets:

UNCONVENTIONAL_MARKET_INFLUENCES = {
    
    # 🌌 SPACE & ASTRONOMY
    'space_weather': {
        'description': 'Solar flares, geomagnetic storms affect satellites, power grids, mining operations',
        'libraries': ['requests', 'astropy', 'spaceweather'],
        'apis': ['NOAA Space Weather', 'ESA Space Weather', 'NASA Solar Dynamics'],
        'impact': 'Infrastructure disruption, mining downtime, communication issues'
    },
    
    # 🌐 INTERNET INFRASTRUCTURE  
    'internet_health': {
        'description': 'BGP routing, submarine cables, DNS issues, CDN performance',
        'libraries': ['bgpstream', 'scapy', 'netaddr', 'dnspython'],
        'apis': ['Cloudflare Radar', 'Internet Health Report', 'BGP monitoring'],
        'impact': 'Trading platform access, exchange connectivity, DeFi availability'
    },
    
    # 🧠 ACADEMIC RESEARCH
    'research_trends': {
        'description': 'Cryptography papers, blockchain research, AI breakthroughs',
        'libraries': ['arxiv', 'scholarly', 'pubmed', 'semantic-scholar'],
        'apis': ['arXiv API', 'Google Scholar', 'Semantic Scholar', 'DBLP'],
        'impact': 'Future tech development, security concerns, innovation cycles'
    },
    
    # ⚡ ENERGY MARKETS
    'energy_systems': {
        'description': 'Electricity prices, renewable energy, grid stability, mining costs',
        'libraries': ['yfinance', 'alpha_vantage', 'eia-python', 'energy-apis'],
        'apis': ['EIA', 'European Energy Exchange', 'Nord Pool', 'AEMO'],
        'impact': 'Mining profitability, ESG narrative, geographic mining shifts'
    },
    
    # 👨‍💻 DEVELOPER ECOSYSTEM
    'developer_metrics': {
        'description': 'GitHub activity, job postings, hackathons, developer sentiment',
        'libraries': ['PyGithub', 'gitlab-python', 'stackoverflow-py', 'devpost-api'],
        'apis': ['GitHub API', 'GitLab API', 'Stack Overflow', 'AngelList'],
        'impact': 'Ecosystem health, innovation velocity, talent migration'
    },
    
    # 🎮 GAMING & METAVERSE
    'gaming_adoption': {
        'description': 'NFT games, metaverse platforms, virtual economies, user engagement',
        'libraries': ['steam-api', 'opensea-api', 'decentraland-api', 'gaming-apis'],
        'apis': ['Steam API', 'OpenSea API', 'The Sandbox', 'Axie Infinity'],
        'impact': 'Utility adoption, mainstream acceptance, token demand'
    },
    
    # 🚢 SUPPLY CHAIN
    'supply_chain_health': {
        'description': 'Semiconductor availability, shipping costs, manufacturing capacity',
        'libraries': ['shipping-apis', 'semiconductor-data', 'logistics-apis'],
        'apis': ['Baltic Exchange', 'Freightos', 'Semiconductor Industry Assoc'],
        'impact': 'Mining hardware availability, infrastructure costs, geographic shifts'
    },
    
    # 🧠 SOCIAL PSYCHOLOGY
    'behavioral_indicators': {
        'description': 'Attention spans, social media fatigue, collective mood, risk appetite',
        'libraries': ['textblob', 'pytrends', 'social-media-apis', 'psychology-apis'],
        'apis': ['Google Trends', 'Social media APIs', 'Survey platforms'],
        'impact': 'Market cycles, FOMO/FUD intensity, adoption patterns'
    },
    
    # 🏛️ REGULATORY INTELLIGENCE
    'regulatory_landscape': {
        'description': 'Legislative calendars, lobbying expenditure, regulatory personnel',
        'libraries': ['government-apis', 'lobbying-data', 'policy-apis'],
        'apis': ['Congress.gov', 'OpenSecrets', 'Regulatory agencies'],
        'impact': 'Policy uncertainty, compliance costs, market access'
    },
    
    # 🔬 QUANTUM COMPUTING
    'quantum_threat': {
        'description': 'Quantum computing progress, cryptography vulnerabilities, timeline estimates',
        'libraries': ['qiskit', 'cirq', 'quantum-research-apis'],
        'apis': ['IBM Quantum', 'Google Quantum AI', 'Research databases'],
        'impact': 'Existential cryptography threat, post-quantum crypto adoption'
    },
    
    # 🌡️ CLIMATE & WEATHER
    'climate_impact': {
        'description': 'Extreme weather affecting mining regions, climate policy, carbon pricing',
        'libraries': ['weather-apis', 'climate-data', 'carbon-pricing-apis'],
        'apis': ['OpenWeatherMap', 'NOAA Climate', 'Carbon pricing platforms'],
        'impact': 'Mining operations, ESG concerns, energy costs, regulatory pressure'
    },
    
    # 🏭 INDUSTRIAL SENSORS
    'iot_industrial': {
        'description': 'Industrial IoT, manufacturing indices, commodity production',
        'libraries': ['iot-apis', 'industrial-data', 'commodity-apis'],
        'apis': ['Industrial IoT platforms', 'Commodity exchanges'],
        'impact': 'Real economy health, commodity-backed tokens, industrial adoption'
    },
    
    # 🧬 BIOTECHNOLOGY
    'biotech_trends': {
        'description': 'Pharmaceutical research, biotech funding, health crises, pandemic indicators',
        'libraries': ['bio-apis', 'pharma-data', 'health-monitoring'],
        'apis': ['FDA databases', 'WHO data', 'Biotech funding platforms'],
        'impact': 'Economic disruption, government spending, safe haven demand'
    },
    
    # 🎭 CULTURAL TRENDS
    'cultural_indicators': {
        'description': 'Art markets, luxury goods, cultural zeitgeist, generational shifts',
        'libraries': ['art-market-apis', 'luxury-data', 'cultural-apis'],
        'apis': ['Art auction houses', 'Luxury market data', 'Cultural trend APIs'],
        'impact': 'Wealth distribution, cultural acceptance, generational adoption'
    },
    
    # 🏫 EDUCATIONAL SYSTEMS
    'education_trends': {
        'description': 'Blockchain education, financial literacy, university programs',
        'libraries': ['education-apis', 'university-data', 'course-platforms'],
        'apis': ['Coursera API', 'University databases', 'Educational platforms'],
        'impact': 'Knowledge dissemination, talent pipeline, mainstream understanding'
    },
    
    # 🏥 HEALTHCARE SYSTEMS
    'healthcare_digitization': {
        'description': 'Digital health adoption, medical records blockchain, telemedicine',
        'libraries': ['healthcare-apis', 'medical-data', 'telemedicine-apis'],
        'apis': ['Healthcare platforms', 'Medical databases', 'Telemedicine APIs'],
        'impact': 'Blockchain utility, privacy regulations, digital identity adoption'
    },
    
    # 🏛️ ARCHAEOLOGICAL & HISTORICAL
    'historical_patterns': {
        'description': 'Historical financial crises, technology adoption cycles, social movements',
        'libraries': ['historical-data', 'pattern-analysis', 'cycle-detection'],
        'apis': ['Historical databases', 'Economic history APIs'],
        'impact': 'Long-term cycle prediction, pattern recognition, social adoption models'
    },
    
    # 🔬 MATERIALS SCIENCE
    'materials_innovation': {
        'description': 'Semiconductor materials, quantum materials, energy storage breakthroughs',
        'libraries': ['materials-apis', 'patent-data', 'research-databases'],
        'apis': ['Patent databases', 'Materials research APIs', 'Innovation trackers'],
        'impact': 'Hardware efficiency, energy storage, quantum computing acceleration'
    },
    
    # 🌊 OCEANOGRAPHY
    'ocean_systems': {
        'description': 'Submarine cable health, ocean temperature, shipping routes',
        'libraries': ['oceanography-apis', 'marine-data', 'shipping-routes'],
        'apis': ['NOAA Ocean Data', 'Marine traffic APIs', 'Submarine cable maps'],
        'impact': 'Internet infrastructure, global trade, climate indicators'
    },
    
    # 🎵 MUSIC & ENTERTAINMENT
    'entertainment_trends': {
        'description': 'Music streaming, entertainment consumption, celebrity influence',
        'libraries': ['spotify-api', 'entertainment-data', 'celebrity-apis'],
        'apis': ['Spotify API', 'Entertainment databases', 'Social influence trackers'],
        'impact': 'Cultural adoption, influencer marketing, mainstream acceptance'
    }
}

# Installation commands for unconventional libraries:
INSTALLATION_COMMANDS = {
    'space_weather': 'pip install requests astropy spaceweather',
    'internet_health': 'pip install bgpstream scapy netaddr dnspython',
    'research_trends': 'pip install arxiv scholarly pubmed-parser',
    'energy_systems': 'pip install yfinance alpha_vantage eia-python',
    'developer_metrics': 'pip install PyGithub python-gitlab praw',
    'gaming_adoption': 'pip install steam-web-api opensea-api',
    'supply_chain': 'pip install shipping-python logistics-apis',
    'behavioral_indicators': 'pip install textblob pytrends vaderSentiment',
    'regulatory_landscape': 'pip install government-apis lobbying-data',
    'quantum_threat': 'pip install qiskit cirq pennylane',
    'climate_impact': 'pip install openweathermap-python climate-data',
    'biotech_trends': 'pip install biopython pharma-apis',
    'materials_innovation': 'pip install materials-apis patent-data',
    'oceanography': 'pip install oceanography-python marine-apis'
}
 