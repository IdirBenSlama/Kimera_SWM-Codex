"""
Enhanced Anomaly Detection System for Kimera Trading
Integrates state-of-the-art fraud detection and market anomaly algorithms
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

try:
    from pyod.models.iforest import IForest
    from pyod.models.lof import LOF
    from pyod.models.ocsvm import OCSVM
    from pyod.models.knn import KNN
    from pyod.models.pca import PCA as PCA_PYOD
    PYOD_AVAILABLE = True
except ImportError:
    PYOD_AVAILABLE = False
    logging.warning("PyOD not available. Install with: pip install pyod")

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.svm import OneClassSVM
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.decomposition import PCA
    from sklearn.covariance import EllipticEnvelope
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("Scikit-learn not available")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not available for model interpretability")


class AnomalyType(Enum):
    """Types of anomalies that can be detected"""
    MARKET_MANIPULATION = "market_manipulation"
    EXECUTION_ANOMALY = "execution_anomaly"
    PORTFOLIO_RISK = "portfolio_risk"
    PRICE_ANOMALY = "price_anomaly"
    VOLUME_ANOMALY = "volume_anomaly"
    SPREAD_ANOMALY = "spread_anomaly"
    ORDER_FLOW_ANOMALY = "order_flow_anomaly"


@dataclass
class AnomalyResult:
    """Result of anomaly detection"""
    timestamp: pd.Timestamp
    anomaly_type: AnomalyType
    severity: float  # 0-1, where 1 is most severe
    confidence: float  # 0-1, confidence in detection
    features: Dict[str, float]
    explanation: str
    recommended_action: str


class EnhancedAnomalyDetector:
    """
    State-of-the-art anomaly detection system for trading
    Combines multiple algorithms for comprehensive fraud detection
    """
    
    def __init__(self, 
                 contamination: float = 0.1,
                 enable_interpretability: bool = True,
                 warmup_periods: int = 100):
        """
        Initialize enhanced anomaly detector
        
        Args:
            contamination: Expected proportion of anomalies (0.1 = 10%)
            enable_interpretability: Whether to enable SHAP explanations
            warmup_periods: Number of periods before anomaly detection becomes active
        """
        self.contamination = contamination
        self.enable_interpretability = enable_interpretability and SHAP_AVAILABLE
        self.warmup_periods = warmup_periods
        self.warmup_count = 0
        
        # Initialize scalers
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.robust_scaler = RobustScaler() if SKLEARN_AVAILABLE else None
        
        # Initialize models
        self.models = {}
        self._initialize_models()
        
        # Feature engineering
        self.feature_extractors = {}
        self._initialize_feature_extractors()
        
        # Historical data for context
        self.market_history = []
        self.execution_history = []
        self.portfolio_history = []
        
        # Anomaly tracking
        self.detected_anomalies = []
        self.anomaly_counts = {anomaly_type: 0 for anomaly_type in AnomalyType}
        
        logging.info("Enhanced Anomaly Detector initialized")
    
    def _initialize_models(self):
        """Initialize all anomaly detection models"""
        
        if SKLEARN_AVAILABLE:
            # Extended Isolation Forest (best performer in benchmarks)
            self.models['isolation_forest'] = IsolationForest(
                contamination=self.contamination,
                n_estimators=200,
                max_samples='auto',
                random_state=42
            )
            
            # Local Outlier Factor (best for local anomalies)
            self.models['lof'] = LocalOutlierFactor(
                n_neighbors=20,
                contamination=self.contamination,
                novelty=True
            )
            
            # One-Class SVM (robust for non-linear patterns)
            self.models['ocsvm'] = OneClassSVM(
                nu=self.contamination,
                kernel='rbf',
                gamma='scale'
            )
            
            # Elliptic Envelope (assumes Gaussian distribution)
            self.models['elliptic_envelope'] = EllipticEnvelope(
                contamination=self.contamination,
                random_state=42
            )
        
        if PYOD_AVAILABLE:
            # PyOD models for additional coverage
            self.models['pyod_iforest'] = IForest(contamination=self.contamination)
            self.models['pyod_lof'] = LOF(contamination=self.contamination)
            self.models['pyod_knn'] = KNN(contamination=self.contamination)
            self.models['pyod_pca'] = PCA_PYOD(contamination=self.contamination)
    
    def _initialize_feature_extractors(self):
        """Initialize feature extraction functions"""
        
        self.feature_extractors = {
            'price_features': self._extract_price_features,
            'volume_features': self._extract_volume_features,
            'spread_features': self._extract_spread_features,
            'volatility_features': self._extract_volatility_features,
            'execution_features': self._extract_execution_features,
            'portfolio_features': self._extract_portfolio_features
        }
    
    def _extract_price_features(self, data: pd.DataFrame) -> Dict[str, float]:
        """Extract price-based anomaly features"""
        features = {}
        
        if 'close' in data.columns:
            prices = data['close'].values
            
            # Price momentum features
            if len(prices) >= 2:
                features['price_change'] = (prices[-1] - prices[-2]) / prices[-2]
                features['price_acceleration'] = np.diff(np.diff(prices))[-1] if len(prices) >= 3 else 0
            
            # Price volatility features
            if len(prices) >= 10:
                returns = np.diff(prices) / prices[:-1]
                features['volatility'] = np.std(returns)
                features['skewness'] = pd.Series(returns).skew()
                features['kurtosis'] = pd.Series(returns).kurtosis()
            
            # Price level features
            if len(prices) >= 20:
                features['price_zscore'] = (prices[-1] - np.mean(prices[-20:])) / np.std(prices[-20:])
                features['price_percentile'] = np.percentile(prices[-20:], 95)
        
        return features
    
    def _extract_volume_features(self, data: pd.DataFrame) -> Dict[str, float]:
        """Extract volume-based anomaly features"""
        features = {}
        
        if 'volume' in data.columns:
            volumes = data['volume'].values
            
            # Volume anomaly features
            if len(volumes) >= 2:
                features['volume_change'] = (volumes[-1] - volumes[-2]) / volumes[-2] if volumes[-2] > 0 else 0
            
            if len(volumes) >= 20:
                features['volume_zscore'] = (volumes[-1] - np.mean(volumes[-20:])) / np.std(volumes[-20:])
                features['volume_ratio'] = volumes[-1] / np.mean(volumes[-20:])
        
        return features
    
    def _extract_spread_features(self, data: pd.DataFrame) -> Dict[str, float]:
        """Extract bid-ask spread anomaly features"""
        features = {}
        
        if 'bid' in data.columns and 'ask' in data.columns:
            spreads = data['ask'] - data['bid']
            mid_prices = (data['bid'] + data['ask']) / 2
            
            # Spread anomaly features
            if len(spreads) >= 1:
                features['spread_absolute'] = spreads.iloc[-1]
                features['spread_relative'] = spreads.iloc[-1] / mid_prices.iloc[-1] if mid_prices.iloc[-1] > 0 else 0
            
            if len(spreads) >= 20:
                features['spread_zscore'] = (spreads.iloc[-1] - spreads.tail(20).mean()) / spreads.tail(20).std()
        
        return features
    
    def _extract_volatility_features(self, data: pd.DataFrame) -> Dict[str, float]:
        """Extract volatility-based anomaly features"""
        features = {}
        
        if 'close' in data.columns and len(data) >= 20:
            prices = data['close'].values
            returns = np.diff(prices) / prices[:-1]
            
            # Volatility clustering features
            volatility_window = 10
            if len(returns) >= volatility_window:
                rolling_vol = pd.Series(returns).rolling(volatility_window).std()
                features['volatility_current'] = rolling_vol.iloc[-1]
                features['volatility_change'] = rolling_vol.iloc[-1] - rolling_vol.iloc[-2] if len(rolling_vol) >= 2 else 0
                features['volatility_zscore'] = (rolling_vol.iloc[-1] - rolling_vol.mean()) / rolling_vol.std()
        
        return features
    
    def _extract_execution_features(self, order_data: Dict) -> Dict[str, float]:
        """Extract order execution anomaly features"""
        features = {}
        
        # Execution timing features
        if 'execution_time' in order_data:
            features['execution_delay'] = order_data.get('execution_time', 0)
        
        # Slippage features
        if 'expected_price' in order_data and 'actual_price' in order_data:
            expected = order_data['expected_price']
            actual = order_data['actual_price']
            features['slippage'] = abs(actual - expected) / expected if expected > 0 else 0
        
        # Order size features
        if 'order_size' in order_data:
            features['order_size'] = order_data['order_size']
        
        return features
    
    def _extract_portfolio_features(self, portfolio_data: Dict) -> Dict[str, float]:
        """Extract portfolio-level anomaly features"""
        features = {}
        
        # Position size features
        if 'total_exposure' in portfolio_data:
            features['total_exposure'] = portfolio_data['total_exposure']
        
        # Risk features
        if 'var' in portfolio_data:
            features['portfolio_var'] = portfolio_data['var']
        
        # Concentration features
        if 'positions' in portfolio_data:
            positions = portfolio_data['positions']
            if positions:
                max_position = max(abs(pos) for pos in positions.values())
                total_exposure = sum(abs(pos) for pos in positions.values())
                features['concentration_ratio'] = max_position / total_exposure if total_exposure > 0 else 0
        
        return features
    
    def detect_market_anomalies(self, market_data: pd.DataFrame) -> List[AnomalyResult]:
        """
        Detect anomalies in market data
        
        Args:
            market_data: DataFrame with OHLCV data
            
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        # Update history
        self.market_history.append(market_data.copy())
        if len(self.market_history) > 1000:  # Keep last 1000 records
            self.market_history.pop(0)
        
        # Check if we have enough data
        if self.warmup_count < self.warmup_periods:
            self.warmup_count += 1
            return anomalies
        
        try:
            # Extract features
            features = {}
            features.update(self._extract_price_features(market_data))
            features.update(self._extract_volume_features(market_data))
            features.update(self._extract_spread_features(market_data))
            features.update(self._extract_volatility_features(market_data))
            
            if not features or not self.scaler:
                return anomalies
            
            # Convert to array for models
            feature_array = np.array(list(features.values())).reshape(1, -1)
            
            # Scale features
            feature_array_scaled = self.scaler.fit_transform(feature_array)
            
            # Run anomaly detection models
            anomaly_scores = self._run_anomaly_models(feature_array_scaled, 'market')
            
            # Combine results and determine if anomaly
            avg_score = np.mean(list(anomaly_scores.values())) if anomaly_scores else 0
            
            if avg_score > 0.7:  # Threshold for anomaly
                anomaly = AnomalyResult(
                    timestamp=pd.Timestamp.now(),
                    anomaly_type=AnomalyType.PRICE_ANOMALY,
                    severity=avg_score,
                    confidence=self._calculate_confidence(anomaly_scores),
                    features=features,
                    explanation=self._generate_explanation(features, anomaly_scores),
                    recommended_action=self._recommend_action(AnomalyType.PRICE_ANOMALY, avg_score)
                )
                anomalies.append(anomaly)
                self._log_anomaly(anomaly)
        
        except Exception as e:
            logging.error(f"Error in market anomaly detection: {str(e)}")
        
        return anomalies
    
    def detect_execution_anomalies(self, order_data: Dict) -> List[AnomalyResult]:
        """
        Detect anomalies in order execution
        
        Args:
            order_data: Dictionary with order execution details
            
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        try:
            # Extract execution features
            features = self._extract_execution_features(order_data)
            
            if not features or not self.robust_scaler:
                return anomalies
            
            # Convert to array
            feature_array = np.array(list(features.values())).reshape(1, -1)
            feature_array_scaled = self.robust_scaler.fit_transform(feature_array)
            
            # Run anomaly detection
            anomaly_scores = self._run_anomaly_models(feature_array_scaled, 'execution')
            avg_score = np.mean(list(anomaly_scores.values())) if anomaly_scores else 0
            
            if avg_score > 0.8:  # Higher threshold for execution anomalies
                anomaly = AnomalyResult(
                    timestamp=pd.Timestamp.now(),
                    anomaly_type=AnomalyType.EXECUTION_ANOMALY,
                    severity=avg_score,
                    confidence=self._calculate_confidence(anomaly_scores),
                    features=features,
                    explanation=self._generate_explanation(features, anomaly_scores),
                    recommended_action=self._recommend_action(AnomalyType.EXECUTION_ANOMALY, avg_score)
                )
                anomalies.append(anomaly)
                self._log_anomaly(anomaly)
        
        except Exception as e:
            logging.error(f"Error in execution anomaly detection: {str(e)}")
        
        return anomalies
    
    def detect_portfolio_anomalies(self, portfolio_data: Dict) -> List[AnomalyResult]:
        """
        Detect anomalies in portfolio composition and risk
        
        Args:
            portfolio_data: Dictionary with portfolio details
            
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        try:
            # Extract portfolio features
            features = self._extract_portfolio_features(portfolio_data)
            
            if not features:
                return anomalies
            
            # Check for risk limit violations
            if 'total_exposure' in features and features['total_exposure'] > 1.0:  # 100% exposure limit
                anomaly = AnomalyResult(
                    timestamp=pd.Timestamp.now(),
                    anomaly_type=AnomalyType.PORTFOLIO_RISK,
                    severity=min(features['total_exposure'], 1.0),
                    confidence=1.0,
                    features=features,
                    explanation="Portfolio exposure exceeds risk limits",
                    recommended_action="Reduce position sizes immediately"
                )
                anomalies.append(anomaly)
                self._log_anomaly(anomaly)
            
            # Check for concentration risk
            if 'concentration_ratio' in features and features['concentration_ratio'] > 0.5:  # 50% concentration limit
                anomaly = AnomalyResult(
                    timestamp=pd.Timestamp.now(),
                    anomaly_type=AnomalyType.PORTFOLIO_RISK,
                    severity=features['concentration_ratio'],
                    confidence=1.0,
                    features=features,
                    explanation="Portfolio concentration exceeds risk limits",
                    recommended_action="Diversify portfolio holdings"
                )
                anomalies.append(anomaly)
                self._log_anomaly(anomaly)
        
        except Exception as e:
            logging.error(f"Error in portfolio anomaly detection: {str(e)}")
        
        return anomalies
    
    def _run_anomaly_models(self, feature_array: np.ndarray, data_type: str) -> Dict[str, float]:
        """Run all available anomaly detection models"""
        scores = {}
        
        for model_name, model in self.models.items():
            try:
                if hasattr(model, 'decision_function'):
                    # Models with decision function (higher = more normal)
                    decision = model.decision_function(feature_array)[0]
                    # Convert to anomaly score (0-1, higher = more anomalous)
                    score = max(0, min(1, (1 - decision) / 2))
                elif hasattr(model, 'score_samples'):
                    # Models with score_samples
                    score_sample = model.score_samples(feature_array)[0]
                    score = max(0, min(1, (1 - score_sample) / 2))
                else:
                    # Default prediction
                    prediction = model.predict(feature_array)[0]
                    score = 1.0 if prediction == -1 else 0.0
                
                scores[model_name] = score
                
            except Exception as e:
                logging.warning(f"Model {model_name} failed: {str(e)}")
                continue
        
        return scores
    
    def _calculate_confidence(self, anomaly_scores: Dict[str, float]) -> float:
        """Calculate confidence in anomaly detection"""
        if not anomaly_scores:
            return 0.0
        
        # Confidence based on agreement between models
        scores = list(anomaly_scores.values())
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        # Higher confidence when models agree (low std) and score is high
        confidence = mean_score * (1 - min(std_score, 0.5))
        return max(0, min(1, confidence))
    
    def _generate_explanation(self, features: Dict[str, float], 
                            anomaly_scores: Dict[str, float]) -> str:
        """Generate human-readable explanation of anomaly"""
        explanations = []
        
        # Find most anomalous features
        if features:
            sorted_features = sorted(features.items(), key=lambda x: abs(x[1]), reverse=True)
            top_features = sorted_features[:3]
            
            for feature_name, value in top_features:
                if abs(value) > 2.0:  # Z-score threshold
                    direction = "high" if value > 0 else "low"
                    explanations.append(f"{feature_name.replace('_', ' ')} is unusually {direction}")
        
        # Add model agreement information
        high_scoring_models = [name for name, score in anomaly_scores.items() if score > 0.7]
        if high_scoring_models:
            explanations.append(f"Detected by {len(high_scoring_models)} anomaly models")
        
        return "; ".join(explanations) if explanations else "Anomaly detected by statistical models"
    
    def _recommend_action(self, anomaly_type: AnomalyType, severity: float) -> str:
        """Recommend action based on anomaly type and severity"""
        actions = {
            AnomalyType.MARKET_MANIPULATION: {
                'low': "Monitor market conditions closely",
                'medium': "Reduce position sizes and increase monitoring",
                'high': "Halt trading and investigate market conditions"
            },
            AnomalyType.EXECUTION_ANOMALY: {
                'low': "Review execution quality metrics",
                'medium': "Check exchange connectivity and latency",
                'high': "Halt automated trading and investigate execution issues"
            },
            AnomalyType.PORTFOLIO_RISK: {
                'low': "Review risk management parameters",
                'medium': "Reduce overall exposure",
                'high': "Emergency risk reduction required"
            },
            AnomalyType.PRICE_ANOMALY: {
                'low': "Monitor price movements",
                'medium': "Reduce trading frequency",
                'high': "Suspend trading until price stability returns"
            }
        }
        
        severity_level = 'high' if severity > 0.8 else 'medium' if severity > 0.5 else 'low'
        return actions.get(anomaly_type, {}).get(severity_level, "Monitor situation closely")
    
    def _log_anomaly(self, anomaly: AnomalyResult):
        """Log detected anomaly"""
        self.detected_anomalies.append(anomaly)
        self.anomaly_counts[anomaly.anomaly_type] += 1
        
        # Keep only recent anomalies
        if len(self.detected_anomalies) > 1000:
            self.detected_anomalies.pop(0)
        
        logging.warning(
            f"ANOMALY DETECTED: {anomaly.anomaly_type.value} "
            f"(Severity: {anomaly.severity:.2f}, Confidence: {anomaly.confidence:.2f}) "
            f"- {anomaly.explanation}"
        )
    
    def get_anomaly_summary(self) -> Dict[str, Any]:
        """Get summary of detected anomalies"""
        recent_anomalies = [a for a in self.detected_anomalies 
                          if (pd.Timestamp.now() - a.timestamp).total_seconds() < 3600]  # Last hour
        
        return {
            'total_anomalies': len(self.detected_anomalies),
            'recent_anomalies': len(recent_anomalies),
            'anomaly_counts': dict(self.anomaly_counts),
            'average_severity': np.mean([a.severity for a in recent_anomalies]) if recent_anomalies else 0,
            'system_status': self._get_system_status(recent_anomalies)
        }
    
    def _get_system_status(self, recent_anomalies: List[AnomalyResult]) -> str:
        """Determine overall system status based on recent anomalies"""
        if not recent_anomalies:
            return "NORMAL"
        
        high_severity_count = sum(1 for a in recent_anomalies if a.severity > 0.8)
        
        if high_severity_count >= 3:
            return "CRITICAL"
        elif high_severity_count >= 1 or len(recent_anomalies) >= 5:
            return "WARNING"
        else:
            return "CAUTION"


def create_enhanced_detector() -> Optional[EnhancedAnomalyDetector]:
    """Factory function to create enhanced anomaly detector"""
    if not SKLEARN_AVAILABLE:
        logging.error("Cannot create enhanced detector: scikit-learn not available")
        return None
    
    return EnhancedAnomalyDetector()


# Example usage and testing
if __name__ == "__main__":
    # Test the enhanced anomaly detector
    detector = create_enhanced_detector()
    
    if detector:
        # Test with sample market data
        sample_data = pd.DataFrame({
            'close': np.random.randn(100).cumsum() + 100,
            'volume': np.random.exponential(1000, 100),
            'bid': np.random.randn(100).cumsum() + 99.5,
            'ask': np.random.randn(100).cumsum() + 100.5
        })
        
        # Introduce some anomalies
        sample_data.loc[50, 'close'] = sample_data.loc[50, 'close'] * 1.1  # Price spike
        sample_data.loc[75, 'volume'] = sample_data.loc[75, 'volume'] * 10  # Volume spike
        
        logger.info("Testing market anomaly detection...")
        anomalies = detector.detect_market_anomalies(sample_data)
        logger.info(f"Detected {len(anomalies)}")
        
        for anomaly in anomalies:
            logger.info(f"- {anomaly.anomaly_type.value}: {anomaly.explanation}")
        
        # Test execution anomaly detection
        sample_execution = {
            'execution_time': 0.5,  # High execution time
            'expected_price': 100.0,
            'actual_price': 100.2,  # High slippage
            'order_size': 1000
        }
        
        logger.info("\nTesting execution anomaly detection...")
        exec_anomalies = detector.detect_execution_anomalies(sample_execution)
        logger.info(f"Detected {len(exec_anomalies)}")
        
        # Get summary
        summary = detector.get_anomaly_summary()
        logger.info(f"\nSystem Status: {summary['system_status']}")
        logger.info(f"Total Anomalies: {summary['total_anomalies']}")