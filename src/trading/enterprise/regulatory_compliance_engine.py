"""
Regulatory Compliance Engine for Kimera SWM

Multi-jurisdictional compliance monitoring, automated reporting,
and real-time violation detection for trading activities.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set
from datetime import datetime, timedelta
from enum import Enum
import json
import hashlib
from collections import defaultdict, deque

# Local imports
from src.core.geoid import GeoidState as Geoid
from src.engines.cognitive_field_dynamics import CognitiveFieldDynamics as CognitiveFieldDynamicsEngine
from src.engines.contradiction_engine import ContradictionEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Jurisdiction(Enum):
    """Supported regulatory jurisdictions"""
    SEC = "SEC"  # US Securities and Exchange Commission
    CFTC = "CFTC"  # US Commodity Futures Trading Commission
    ESMA = "ESMA"  # European Securities and Markets Authority
    FCA = "FCA"  # UK Financial Conduct Authority
    ASIC = "ASIC"  # Australian Securities and Investments Commission
    MAS = "MAS"  # Monetary Authority of Singapore
    JFSA = "JFSA"  # Japan Financial Services Agency


class ViolationType(Enum):
    """Types of regulatory violations"""
    MARKET_MANIPULATION = "market_manipulation"
    INSIDER_TRADING = "insider_trading"
    WASH_TRADING = "wash_trading"
    SPOOFING = "spoofing"
    LAYERING = "layering"
    FRONT_RUNNING = "front_running"
    EXCESSIVE_MESSAGING = "excessive_messaging"
    POSITION_LIMIT_BREACH = "position_limit_breach"
    REPORTING_FAILURE = "reporting_failure"
    KYC_AML_VIOLATION = "kyc_aml_violation"


@dataclass
class ComplianceRule:
    """Regulatory compliance rule"""
    rule_id: str
    jurisdiction: Jurisdiction
    rule_type: ViolationType
    description: str
    threshold: Dict[str, Any]
    severity: str  # 'critical', 'high', 'medium', 'low'
    active: bool = True
    
    
@dataclass
class ComplianceViolation:
    """Detected compliance violation"""
    violation_id: str
    timestamp: datetime
    rule_id: str
    jurisdiction: Jurisdiction
    violation_type: ViolationType
    severity: str
    details: Dict[str, Any]
    evidence: List[Dict[str, Any]]
    remediation_required: bool = True
    reported: bool = False
    
    
@dataclass
class ComplianceReport:
    """Regulatory compliance report"""
    report_id: str
    timestamp: datetime
    jurisdiction: Jurisdiction
    reporting_period: Tuple[datetime, datetime]
    trades_analyzed: int
    violations_found: List[ComplianceViolation]
    risk_score: float
    attestation: Dict[str, Any]
    
    
@dataclass
class TradingActivity:
    """Trading activity for compliance monitoring"""
    activity_id: str
    timestamp: datetime
    trader_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    price: float
    order_type: str
    venue: str
    metadata: Dict[str, Any] = field(default_factory=dict) 


class RegulatoryComplianceEngine:
    """
    Regulatory Compliance Engine
    
    Features:
    - Multi-jurisdictional compliance monitoring
    - Real-time violation detection
    - Automated regulatory reporting
    - Market abuse detection
    - Position limit monitoring
    """
    
    def __init__(self,
                 cognitive_field: Optional[CognitiveFieldDynamicsEngine] = None,
                 contradiction_engine: Optional[ContradictionEngine] = None):
        """Initialize Regulatory Compliance Engine"""
        self.cognitive_field = cognitive_field
        self.contradiction_engine = contradiction_engine
        
        # Compliance rules
        self.compliance_rules = self._initialize_compliance_rules()
        
        # Activity monitoring
        self.trading_activities: deque = deque(maxlen=100000)
        self.activity_by_trader: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.activity_by_symbol: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        
        # Violation tracking
        self.violations: List[ComplianceViolation] = []
        self.violations_by_type: Dict[ViolationType, List[ComplianceViolation]] = defaultdict(list)
        
        # Reporting
        self.compliance_reports: List[ComplianceReport] = []
        self.report_schedules = self._initialize_report_schedules()
        
        # Position tracking
        self.position_limits: Dict[str, Dict[str, float]] = {}  # symbol -> trader -> limit
        self.current_positions: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        
        # Performance metrics
        self.activities_monitored = 0
        self.violations_detected = 0
        self.reports_generated = 0
        
        # Background monitoring control
        self.running = False
        self.monitoring_task = None
        self.reporting_task = None
        
        # Test compatibility attribute
        self.jurisdictions = {
            'active_jurisdictions': [j.value for j in Jurisdiction],
            'compliance_coverage': len(self.compliance_rules),
            'reporting_schedules': len(self.report_schedules),
            'primary_jurisdiction': 'SEC'
        }
        
        logger.info("Regulatory Compliance Engine initialized")
        
    def _initialize_compliance_rules(self) -> Dict[str, ComplianceRule]:
        """Initialize compliance rules for all jurisdictions"""
        rules = {}
        
        # SEC Rules
        rules['sec_wash_trading'] = ComplianceRule(
            rule_id='sec_wash_trading',
            jurisdiction=Jurisdiction.SEC,
            rule_type=ViolationType.WASH_TRADING,
            description='Prohibition against wash trading',
            threshold={'time_window': 300, 'min_trades': 2},
            severity='critical'
        )
        
        rules['sec_spoofing'] = ComplianceRule(
            rule_id='sec_spoofing',
            jurisdiction=Jurisdiction.SEC,
            rule_type=ViolationType.SPOOFING,
            description='Prohibition against spoofing',
            threshold={'cancel_rate': 0.9, 'min_orders': 10},
            severity='critical'
        )
        
        # ESMA Rules (MiFID II)
        rules['esma_excessive_messaging'] = ComplianceRule(
            rule_id='esma_excessive_messaging',
            jurisdiction=Jurisdiction.ESMA,
            rule_type=ViolationType.EXCESSIVE_MESSAGING,
            description='Order-to-trade ratio limits',
            threshold={'max_ratio': 4, 'time_window': 1},
            severity='high'
        )
        
        rules['esma_position_limits'] = ComplianceRule(
            rule_id='esma_position_limits',
            jurisdiction=Jurisdiction.ESMA,
            rule_type=ViolationType.POSITION_LIMIT_BREACH,
            description='Position limit compliance',
            threshold={'check_limits': True},
            severity='high'
        )
        
        # FCA Rules
        rules['fca_market_abuse'] = ComplianceRule(
            rule_id='fca_market_abuse',
            jurisdiction=Jurisdiction.FCA,
            rule_type=ViolationType.MARKET_MANIPULATION,
            description='Market abuse regulation',
            threshold={'price_impact': 0.02, 'volume_threshold': 0.1},
            severity='critical'
        )
        
        # CFTC Rules
        rules['cftc_position_limits'] = ComplianceRule(
            rule_id='cftc_position_limits',
            jurisdiction=Jurisdiction.CFTC,
            rule_type=ViolationType.POSITION_LIMIT_BREACH,
            description='Futures position limits',
            threshold={'check_limits': True},
            severity='high'
        )
        
        return rules
        
    def _initialize_report_schedules(self) -> Dict[Jurisdiction, Dict[str, Any]]:
        """Initialize reporting schedules for each jurisdiction"""
        return {
            Jurisdiction.SEC: {
                'frequency': 'daily',
                'time': '16:00',
                'format': 'sec_format'
            },
            Jurisdiction.ESMA: {
                'frequency': 'daily',
                'time': '17:00',
                'format': 'mifid_format'
            },
            Jurisdiction.FCA: {
                'frequency': 'daily',
                'time': '17:30',
                'format': 'fca_format'
            },
            Jurisdiction.CFTC: {
                'frequency': 'daily',
                'time': '15:30',
                'format': 'cftc_format'
            }
        }
        
    async def monitor_trading_activity(self, activity: TradingActivity) -> List[ComplianceViolation]:
        """
        Monitor trading activity for compliance violations
        
        Args:
            activity: Trading activity to monitor
            
        Returns:
            List of detected violations
        """
        try:
            # Store activity
            self.trading_activities.append(activity)
            self.activity_by_trader[activity.trader_id].append(activity)
            self.activity_by_symbol[activity.symbol].append(activity)
            
            # Update positions
            self._update_positions(activity)
            
            # Check all active rules
            violations = []
            
            for rule_id, rule in self.compliance_rules.items():
                if rule.active:
                    violation = await self._check_rule(rule, activity)
                    if violation:
                        violations.append(violation)
                        
            # Store violations
            for violation in violations:
                self.violations.append(violation)
                self.violations_by_type[violation.violation_type].append(violation)
                
                # Alert on critical violations
                if violation.severity == 'critical':
                    logger.critical(f"Critical compliance violation detected: {violation.violation_type}")
                    
            # Integrate with cognitive field
            if self.cognitive_field and violations:
                await self._cognitive_compliance_analysis(violations)
                
            # Update metrics
            self.activities_monitored += 1
            self.violations_detected += len(violations)
            
            return violations
            
        except Exception as e:
            logger.error(f"Activity monitoring error: {e}")
            return [] 
            
    async def _check_rule(self, rule: ComplianceRule, activity: TradingActivity) -> Optional[ComplianceViolation]:
        """Check specific compliance rule"""
        if rule.rule_type == ViolationType.WASH_TRADING:
            return await self._check_wash_trading(rule, activity)
        elif rule.rule_type == ViolationType.SPOOFING:
            return await self._check_spoofing(rule, activity)
        elif rule.rule_type == ViolationType.EXCESSIVE_MESSAGING:
            return await self._check_excessive_messaging(rule, activity)
        elif rule.rule_type == ViolationType.POSITION_LIMIT_BREACH:
            return await self._check_position_limits(rule, activity)
        elif rule.rule_type == ViolationType.MARKET_MANIPULATION:
            return await self._check_market_manipulation(rule, activity)
        
        return None
        
    async def _check_wash_trading(self, rule: ComplianceRule, activity: TradingActivity) -> Optional[ComplianceViolation]:
        """Check for wash trading violations"""
        trader_activities = list(self.activity_by_trader[activity.trader_id])[-20:]
        
        # Look for opposite trades within time window
        time_window = rule.threshold.get('time_window', 300)  # seconds
        
        for prev_activity in trader_activities[:-1]:
            if prev_activity.symbol != activity.symbol:
                continue
                
            time_diff = (activity.timestamp - prev_activity.timestamp).total_seconds()
            
            if time_diff <= time_window:
                # Check for opposite side trades
                if (activity.side == 'buy' and prev_activity.side == 'sell') or \
                   (activity.side == 'sell' and prev_activity.side == 'buy'):
                    
                    # Check if prices are similar (potential wash trade)
                    price_diff = abs(activity.price - prev_activity.price) / prev_activity.price
                    
                    if price_diff < 0.001:  # 0.1% price difference
                        return ComplianceViolation(
                            violation_id=f"violation_{activity.activity_id}",
                            timestamp=activity.timestamp,
                            rule_id=rule.rule_id,
                            jurisdiction=rule.jurisdiction,
                            violation_type=rule.rule_type,
                            severity=rule.severity,
                            details={
                                'trader_id': activity.trader_id,
                                'symbol': activity.symbol,
                                'trades': [prev_activity.activity_id, activity.activity_id],
                                'time_difference': time_diff
                            },
                            evidence=[
                                {'activity': prev_activity.__dict__},
                                {'activity': activity.__dict__}
                            ]
                        )
                        
        return None
        
    async def _check_spoofing(self, rule: ComplianceRule, activity: TradingActivity) -> Optional[ComplianceViolation]:
        """Check for spoofing violations"""
        if 'order_status' not in activity.metadata:
            return None
            
        # Check cancellation patterns
        trader_activities = list(self.activity_by_trader[activity.trader_id])[-50:]
        
        cancelled_orders = 0
        total_orders = 0
        
        for act in trader_activities:
            if act.symbol == activity.symbol and act.order_type == 'limit':
                total_orders += 1
                if act.metadata.get('order_status') == 'cancelled':
                    cancelled_orders += 1
                    
        if total_orders >= rule.threshold.get('min_orders', 10):
            cancel_rate = cancelled_orders / total_orders
            
            if cancel_rate >= rule.threshold.get('cancel_rate', 0.9):
                return ComplianceViolation(
                    violation_id=f"violation_{activity.activity_id}",
                    timestamp=activity.timestamp,
                    rule_id=rule.rule_id,
                    jurisdiction=rule.jurisdiction,
                    violation_type=rule.rule_type,
                    severity=rule.severity,
                    details={
                        'trader_id': activity.trader_id,
                        'symbol': activity.symbol,
                        'cancel_rate': cancel_rate,
                        'total_orders': total_orders,
                        'cancelled_orders': cancelled_orders
                    },
                    evidence=[{'activity_id': act.activity_id} for act in trader_activities[-10:]]
                )
                
        return None
        
    async def _check_excessive_messaging(self, rule: ComplianceRule, activity: TradingActivity) -> Optional[ComplianceViolation]:
        """Check for excessive order-to-trade ratio"""
        # Get recent activities
        time_window = rule.threshold.get('time_window', 1)  # seconds
        cutoff_time = activity.timestamp - timedelta(seconds=time_window)
        
        recent_activities = [
            act for act in self.activity_by_trader[activity.trader_id]
            if act.timestamp > cutoff_time and act.symbol == activity.symbol
        ]
        
        orders = sum(1 for act in recent_activities if act.order_type in ['limit', 'stop'])
        trades = sum(1 for act in recent_activities if act.metadata.get('executed', False))
        
        if trades > 0:
            ratio = orders / trades
            
            if ratio > rule.threshold.get('max_ratio', 4):
                return ComplianceViolation(
                    violation_id=f"violation_{activity.activity_id}",
                    timestamp=activity.timestamp,
                    rule_id=rule.rule_id,
                    jurisdiction=rule.jurisdiction,
                    violation_type=rule.rule_type,
                    severity=rule.severity,
                    details={
                        'trader_id': activity.trader_id,
                        'symbol': activity.symbol,
                        'order_to_trade_ratio': ratio,
                        'orders': orders,
                        'trades': trades,
                        'time_window': time_window
                    },
                    evidence=[{'activity_count': len(recent_activities)}]
                )
                
        return None
        
    async def _check_position_limits(self, rule: ComplianceRule, activity: TradingActivity) -> Optional[ComplianceViolation]:
        """Check for position limit breaches"""
        symbol = activity.symbol
        trader_id = activity.trader_id
        
        # Check if position limits are defined
        if symbol not in self.position_limits or trader_id not in self.position_limits[symbol]:
            return None
            
        current_position = self.current_positions[symbol][trader_id]
        position_limit = self.position_limits[symbol][trader_id]
        
        if abs(current_position) > position_limit:
            return ComplianceViolation(
                violation_id=f"violation_{activity.activity_id}",
                timestamp=activity.timestamp,
                rule_id=rule.rule_id,
                jurisdiction=rule.jurisdiction,
                violation_type=rule.rule_type,
                severity=rule.severity,
                details={
                    'trader_id': trader_id,
                    'symbol': symbol,
                    'current_position': current_position,
                    'position_limit': position_limit,
                    'breach_amount': abs(current_position) - position_limit
                },
                evidence=[{'activity': activity.__dict__}]
            )
            
        return None
        
    async def _check_market_manipulation(self, rule: ComplianceRule, activity: TradingActivity) -> Optional[ComplianceViolation]:
        """Check for market manipulation patterns"""
        # Get market impact
        symbol_activities = list(self.activity_by_symbol[activity.symbol])[-100:]
        
        if len(symbol_activities) < 10:
            return None
            
        # Calculate trader's volume share
        trader_volume = sum(
            act.quantity for act in symbol_activities 
            if act.trader_id == activity.trader_id
        )
        total_volume = sum(act.quantity for act in symbol_activities)
        
        if total_volume == 0:
            return None
            
        volume_share = trader_volume / total_volume
        
        # Calculate price impact
        prices = [act.price for act in symbol_activities]
        if len(prices) > 2:
            price_change = (prices[-1] - prices[0]) / prices[0]
            
            if abs(price_change) > rule.threshold.get('price_impact', 0.02) and \
               volume_share > rule.threshold.get('volume_threshold', 0.1):
                
                return ComplianceViolation(
                    violation_id=f"violation_{activity.activity_id}",
                    timestamp=activity.timestamp,
                    rule_id=rule.rule_id,
                    jurisdiction=rule.jurisdiction,
                    violation_type=rule.rule_type,
                    severity=rule.severity,
                    details={
                        'trader_id': activity.trader_id,
                        'symbol': activity.symbol,
                        'price_impact': price_change,
                        'volume_share': volume_share,
                        'trader_volume': trader_volume,
                        'total_volume': total_volume
                    },
                    evidence=[
                        {'first_price': prices[0], 'last_price': prices[-1]},
                        {'activity_count': len(symbol_activities)}
                    ]
                )
                
        return None
        
    def _update_positions(self, activity: TradingActivity):
        """Update position tracking"""
        if activity.metadata.get('executed', False):
            position_change = activity.quantity if activity.side == 'buy' else -activity.quantity
            self.current_positions[activity.symbol][activity.trader_id] += position_change 
            
    async def generate_compliance_report(self, 
                                       jurisdiction: Jurisdiction,
                                       start_time: datetime,
                                       end_time: datetime) -> ComplianceReport:
        """
        Generate compliance report for a jurisdiction
        
        Args:
            jurisdiction: Regulatory jurisdiction
            start_time: Report start time
            end_time: Report end time
            
        Returns:
            Compliance report
        """
        # Filter activities and violations for the period
        period_activities = [
            act for act in self.trading_activities
            if start_time <= act.timestamp <= end_time
        ]
        
        period_violations = [
            viol for viol in self.violations
            if viol.jurisdiction == jurisdiction and start_time <= viol.timestamp <= end_time
        ]
        
        # Calculate risk score
        risk_score = self._calculate_risk_score(period_violations, len(period_activities))
        
        # Create attestation
        attestation = self._create_attestation(jurisdiction, period_violations)
        
        report = ComplianceReport(
            report_id=f"report_{jurisdiction.value}_{datetime.now().timestamp()}",
            timestamp=datetime.now(),
            jurisdiction=jurisdiction,
            reporting_period=(start_time, end_time),
            trades_analyzed=len(period_activities),
            violations_found=period_violations,
            risk_score=risk_score,
            attestation=attestation
        )
        
        self.compliance_reports.append(report)
        self.reports_generated += 1
        
        # Format report based on jurisdiction
        formatted_report = await self._format_report(report)
        
        return report
        
    def _calculate_risk_score(self, violations: List[ComplianceViolation], total_activities: int) -> float:
        """Calculate compliance risk score"""
        if total_activities == 0:
            return 0.0
            
        # Base score on violation rate
        violation_rate = len(violations) / total_activities
        
        # Weight by severity
        severity_weights = {
            'critical': 10.0,
            'high': 5.0,
            'medium': 2.0,
            'low': 1.0
        }
        
        weighted_violations = sum(
            severity_weights.get(v.severity, 1.0) for v in violations
        )
        
        # Normalize to 0-1 scale
        risk_score = min(1.0, (violation_rate * 100) + (weighted_violations / total_activities))
        
        return risk_score
        
    def _create_attestation(self, jurisdiction: Jurisdiction, violations: List[ComplianceViolation]) -> Dict[str, Any]:
        """Create compliance attestation"""
        return {
            'timestamp': datetime.now().isoformat(),
            'jurisdiction': jurisdiction.value,
            'violations_count': len(violations),
            'critical_violations': sum(1 for v in violations if v.severity == 'critical'),
            'attestation_text': f"This report attests to the compliance monitoring activities for {jurisdiction.value}",
            'signature': hashlib.sha256(f"{jurisdiction.value}{datetime.now()}".encode()).hexdigest()
        }
        
    async def _format_report(self, report: ComplianceReport) -> str:
        """Format report based on jurisdiction requirements"""
        format_type = self.report_schedules[report.jurisdiction].get('format', 'standard')
        
        if format_type == 'sec_format':
            return self._format_sec_report(report)
        elif format_type == 'mifid_format':
            return self._format_mifid_report(report)
        elif format_type == 'fca_format':
            return self._format_fca_report(report)
        elif format_type == 'cftc_format':
            return self._format_cftc_report(report)
        else:
            return json.dumps(report.__dict__, default=str, indent=2)
            
    def _format_sec_report(self, report: ComplianceReport) -> str:
        """Format report for SEC requirements"""
        sections = [
            "SEC COMPLIANCE REPORT",
            f"Report ID: {report.report_id}",
            f"Period: {report.reporting_period[0]} to {report.reporting_period[1]}",
            f"Trades Analyzed: {report.trades_analyzed}",
            f"Violations Found: {len(report.violations_found)}",
            "",
            "VIOLATIONS SUMMARY:",
        ]
        
        for violation in report.violations_found:
            sections.append(f"- {violation.violation_type.value}: {violation.details}")
            
        sections.extend([
            "",
            f"Risk Score: {report.risk_score:.2%}",
            "",
            "ATTESTATION:",
            report.attestation['attestation_text'],
            f"Digital Signature: {report.attestation['signature']}"
        ])
        
        return "\n".join(sections)
        
    def _format_mifid_report(self, report: ComplianceReport) -> str:
        """Format report for MiFID II requirements"""
        # Similar formatting for ESMA
        return self._format_sec_report(report)  # Simplified
        
    def _format_fca_report(self, report: ComplianceReport) -> str:
        """Format report for FCA requirements"""
        # Similar formatting for FCA
        return self._format_sec_report(report)  # Simplified
        
    def _format_cftc_report(self, report: ComplianceReport) -> str:
        """Format report for CFTC requirements"""
        # Similar formatting for CFTC
        return self._format_sec_report(report)  # Simplified
        
    async def _continuous_monitoring(self):
        """Continuous compliance monitoring"""
        while self.running:
            try:
                # Check for patterns across recent activities
                await self._detect_cross_activity_patterns()
                
                # Update risk assessments
                await self._update_risk_assessments()
                
                # Clean old data
                await self._cleanup_old_data()
                
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
            except Exception as e:
                logger.error(f"Continuous monitoring error: {e}")
                
    async def _detect_cross_activity_patterns(self):
        """Detect patterns across multiple activities"""
        # Check for coordinated trading
        recent_activities = list(self.trading_activities)[-1000:]
        
        # Group by time windows
        time_windows = defaultdict(list)
        for activity in recent_activities:
            window_key = activity.timestamp.replace(second=0, microsecond=0)
            time_windows[window_key].append(activity)
            
        # Look for suspicious patterns
        for window, activities in time_windows.items():
            if len(activities) > 50:  # High activity in one minute
                # Check for coordinated behavior
                symbol_counts = defaultdict(int)
                trader_counts = defaultdict(int)
                
                for act in activities:
                    symbol_counts[act.symbol] += 1
                    trader_counts[act.trader_id] += 1
                    
                # Alert if concentrated activity
                max_symbol_concentration = max(symbol_counts.values()) / len(activities)
                if max_symbol_concentration > 0.7:
                    logger.warning(f"High concentration of activity detected at {window}")
                    
    async def _update_risk_assessments(self):
        """Update risk assessments for traders and symbols"""
        # Calculate trader risk scores
        trader_risks = {}
        
        for trader_id, violations in self.violations_by_type.items():
            recent_violations = [v for v in violations if 
                               (datetime.now() - v.timestamp).days <= 30]
            
            if recent_violations:
                risk_score = len(recent_violations) * 0.1
                risk_score += sum(0.2 for v in recent_violations if v.severity == 'critical')
                trader_risks[trader_id] = min(risk_score, 1.0)
                
    async def _scheduled_reporting(self):
        """Generate scheduled compliance reports"""
        while self.running:
            try:
                current_time = datetime.now()
                
                for jurisdiction, schedule in self.report_schedules.items():
                    # Check if it's time to generate report
                    report_time = datetime.strptime(schedule['time'], '%H:%M').time()
                    
                    if current_time.time().hour == report_time.hour and \
                       current_time.time().minute == report_time.minute:
                        
                        # Generate daily report
                        start_time = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
                        end_time = current_time
                        
                        report = await self.generate_compliance_report(
                            jurisdiction, start_time, end_time
                        )
                        
                        logger.info(f"Generated {jurisdiction.value} compliance report: {report.report_id}")
                        
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Scheduled reporting error: {e}")
                
    async def _cleanup_old_data(self):
        """Clean up old compliance data"""
        # Keep only last 30 days of data
        cutoff_time = datetime.now() - timedelta(days=30)
        
        # Clean violations
        self.violations = [v for v in self.violations if v.timestamp > cutoff_time]
        
        # Clean reports
        self.compliance_reports = [r for r in self.compliance_reports if r.timestamp > cutoff_time]
        
    async def _cognitive_compliance_analysis(self, violations: List[ComplianceViolation]):
        """Integrate compliance violations with cognitive field"""
        if not self.cognitive_field:
            return
            
        for violation in violations:
            # Create violation geoid
            violation_geoid = Geoid(
                semantic_features={
                    'type': 'compliance_violation',
                    'violation_type': violation.violation_type.value,
                    'severity': violation.severity,
                    'jurisdiction': violation.jurisdiction.value
                },
                symbolic_content=f"Violation: {violation.violation_type.value}"
            )
            
            # Check for contradictions
            if self.contradiction_engine:
                contradictions = await self.contradiction_engine.find_contradictions([violation_geoid])
                
                if contradictions:
                    logger.warning(f"Compliance contradiction detected: {contradictions}")
                    
    async def get_compliance_status(self) -> Dict[str, Any]:
        """Get current compliance status"""
        recent_violations = [v for v in self.violations if 
                           (datetime.now() - v.timestamp).hours <= 24]
        
        status = {
            'status': 'compliant' if not recent_violations else 'violations_detected',
            'recent_violations': len(recent_violations),
            'critical_violations': sum(1 for v in recent_violations if v.severity == 'critical'),
            'activities_monitored': self.activities_monitored,
            'total_violations': self.violations_detected,
            'reports_generated': self.reports_generated,
            'active_rules': sum(1 for r in self.compliance_rules.values() if r.active),
            'jurisdictions_monitored': list(set(r.jurisdiction.value for r in self.compliance_rules.values()))
        }
        
        return status
        
    def set_position_limit(self, symbol: str, trader_id: str, limit: float):
        """Set position limit for a trader on a symbol"""
        if symbol not in self.position_limits:
            self.position_limits[symbol] = {}
            
        self.position_limits[symbol][trader_id] = limit
        logger.info(f"Set position limit for {trader_id} on {symbol}: {limit}")
        
    def shutdown(self):
        """Shutdown compliance engine"""
        self.running = False
        logger.info("Regulatory Compliance Engine shutdown")


def create_compliance_engine(cognitive_field=None,
                           contradiction_engine=None) -> RegulatoryComplianceEngine:
    """Factory function to create Regulatory Compliance Engine"""
    return RegulatoryComplianceEngine(
        cognitive_field=cognitive_field,
        contradiction_engine=contradiction_engine
    ) 