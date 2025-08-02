"""
Advanced Market Manipulation Detection System for KIMERA Trading
Implements LSTM-based anomaly detection and multi-dimensional analysis
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import asyncio
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class ManipulationSignal:
    """Market manipulation detection signal"""
    timestamp: datetime
    symbol: str
    manipulation_type: str
    confidence: float
    severity: str  # 'low', 'medium', 'high', 'critical'
    evidence: Dict[str, Any]
    risk_score: float
    recommended_action: str

class LSTMManipulationDetector(nn.Module):
    """LSTM Neural Network for detecting manipulation patterns"""
    
    def __init__(self, input_size: int = 10, hidden_size: int = 64, num_layers: int = 3, dropout: float = 0.2):
        super(LSTMManipulationDetector, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Attention mechanism
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Use last output for classification
        final_output = attn_out[:, -1, :]
        
        # Classification
        manipulation_prob = self.classifier(final_output)
        
        return manipulation_prob

class AdvancedManipulationDetector:
    """
    Advanced market manipulation detection system combining:
    - LSTM neural networks for pattern recognition
    - Statistical anomaly detection
    - Volume-price analysis
    - Order flow analysis
    - Cross-market correlation analysis
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Manipulation detector using device: {self.device}")
        
        # Initialize models
        self.lstm_model = LSTMManipulationDetector().to(self.device)
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.dbscan = DBSCAN(eps=0.5, min_samples=5)
        
        # Scalers for different data types
        self.price_scaler = MinMaxScaler()
        self.volume_scaler = MinMaxScaler()
        self.feature_scaler = MinMaxScaler()
        
        # Detection thresholds
        self.thresholds = {
            'pump_dump': 0.75,
            'spoofing': 0.70,
            'wash_trading': 0.65,
            'layering': 0.60,
            'momentum_ignition': 0.55,
            'general_anomaly': 0.50
        }
        
        # Historical data for pattern learning
        self.historical_patterns = {}
        self.model_trained = False
        
    async def analyze_manipulation(self, market_data: pd.DataFrame, 
                                 order_book_data: Optional[pd.DataFrame] = None,
                                 trade_data: Optional[pd.DataFrame] = None) -> List[ManipulationSignal]:
        """
        Comprehensive manipulation analysis
        
        Args:
            market_data: OHLCV data with timestamps
            order_book_data: Order book snapshots (optional)
            trade_data: Individual trade data (optional)
            
        Returns:
            List of manipulation signals detected
        """
        try:
            signals = []
            
            # Prepare features for analysis
            features = await self._prepare_features(market_data, order_book_data, trade_data)
            
            if len(features) < 50:  # Need minimum data for reliable detection
                logger.warning("Insufficient data for manipulation detection")
                return signals
            
            # Run multiple detection methods
            detection_tasks = [
                self._detect_pump_dump(features),
                self._detect_spoofing(features, order_book_data),
                self._detect_wash_trading(features, trade_data),
                self._detect_layering(features, order_book_data),
                self._detect_momentum_ignition(features),
                self._lstm_anomaly_detection(features),
                self._statistical_anomaly_detection(features)
            ]
            
            # Run detections in parallel
            detection_results = await asyncio.gather(*detection_tasks, return_exceptions=True)
            
            # Combine results
            for result in detection_results:
                if isinstance(result, list):
                    signals.extend(result)
                elif isinstance(result, Exception):
                    logger.error(f"Detection error: {result}")
            
            # Remove duplicates and rank by confidence
            signals = self._consolidate_signals(signals)
            
            logger.info(f"Detected {len(signals)} potential manipulation signals")
            return signals
            
        except Exception as e:
            logger.error(f"Error in manipulation analysis: {e}")
            return []
    
    async def _prepare_features(self, market_data: pd.DataFrame, 
                              order_book_data: Optional[pd.DataFrame],
                              trade_data: Optional[pd.DataFrame]) -> pd.DataFrame:
        """Prepare comprehensive feature set for manipulation detection"""
        
        features = market_data.copy()
        
        # Price-based features
        features['price_change'] = features['close'].pct_change()
        features['price_volatility'] = features['price_change'].rolling(20).std()
        features['price_momentum'] = features['close'].rolling(10).apply(lambda x: (x[-1] - x[0]) / x[0])
        
        # Volume-based features
        features['volume_change'] = features['volume'].pct_change()
        features['volume_ma'] = features['volume'].rolling(20).mean()
        features['volume_ratio'] = features['volume'] / features['volume_ma']
        features['volume_price_trend'] = features['volume'] * features['price_change']
        
        # Technical indicators
        features['rsi'] = self._calculate_rsi(features['close'])
        features['bollinger_upper'], features['bollinger_lower'] = self._calculate_bollinger_bands(features['close'])
        features['macd'], features['macd_signal'] = self._calculate_macd(features['close'])
        
        # Advanced features
        features['bid_ask_spread'] = np.nan
        features['order_imbalance'] = np.nan
        features['trade_size_anomaly'] = np.nan
        
        if order_book_data is not None:
            features = await self._add_order_book_features(features, order_book_data)
        
        if trade_data is not None:
            features = await self._add_trade_features(features, trade_data)
        
        # Fill NaN values
        features = features.fillna(method='ffill').fillna(0)
        
        return features
    
    async def _detect_pump_dump(self, features: pd.DataFrame) -> List[ManipulationSignal]:
        """Detect pump and dump schemes"""
        signals = []
        
        try:
            # Look for rapid price increases followed by dumps
            price_changes = features['price_change'].rolling(10).sum()
            volume_spikes = features['volume_ratio'] > 3.0
            
            # Pump phase detection
            pump_conditions = (
                (price_changes > 0.15) &  # 15% increase in 10 periods
                (features['volume_ratio'] > 2.0) &  # High volume
                (features['rsi'] > 70)  # Overbought
            )
            
            # Dump phase detection (following pump)
            dump_conditions = (
                (price_changes < -0.10) &  # 10% decrease
                (features['volume_ratio'] > 1.5) &  # Elevated volume
                (features['rsi'] < 30)  # Oversold
            )
            
            pump_indices = features[pump_conditions].index
            dump_indices = features[dump_conditions].index
            
            # Match pumps with subsequent dumps
            for pump_idx in pump_indices:
                subsequent_dumps = dump_indices[dump_indices > pump_idx]
                if len(subsequent_dumps) > 0:
                    dump_idx = subsequent_dumps[0]
                    
                    # Calculate confidence based on pattern strength
                    confidence = min(0.95, abs(price_changes.loc[pump_idx]) * 2 + 
                                   features.loc[pump_idx, 'volume_ratio'] * 0.1)
                    
                    if confidence > self.thresholds['pump_dump']:
                        signal = ManipulationSignal(
                            timestamp=features.loc[pump_idx, 'timestamp'] if 'timestamp' in features.columns else datetime.now(),
                            symbol=features.get('symbol', 'UNKNOWN').iloc[0] if 'symbol' in features.columns else 'UNKNOWN',
                            manipulation_type='pump_dump',
                            confidence=confidence,
                            severity='high' if confidence > 0.85 else 'medium',
                            evidence={
                                'pump_price_change': price_changes.loc[pump_idx],
                                'dump_price_change': price_changes.loc[dump_idx],
                                'pump_volume_ratio': features.loc[pump_idx, 'volume_ratio'],
                                'pump_rsi': features.loc[pump_idx, 'rsi']
                            },
                            risk_score=confidence * 10,
                            recommended_action='AVOID_TRADING'
                        )
                        signals.append(signal)
            
        except Exception as e:
            logger.error(f"Error in pump-dump detection: {e}")
        
        return signals
    
    async def _detect_spoofing(self, features: pd.DataFrame, 
                             order_book_data: Optional[pd.DataFrame]) -> List[ManipulationSignal]:
        """Detect spoofing patterns in order book"""
        signals = []
        
        if order_book_data is None:
            return signals
        
        try:
            # Look for large orders that get cancelled quickly
            # This is a simplified implementation - real spoofing detection
            # requires detailed order book analysis
            
            large_order_threshold = order_book_data['bid_size'].quantile(0.95)
            
            # Detect unusual order book imbalances
            imbalance = abs(order_book_data['bid_size'] - order_book_data['ask_size']) / \
                       (order_book_data['bid_size'] + order_book_data['ask_size'])
            
            spoofing_conditions = (
                (imbalance > 0.7) &  # High imbalance
                ((order_book_data['bid_size'] > large_order_threshold) | 
                 (order_book_data['ask_size'] > large_order_threshold))
            )
            
            spoof_indices = order_book_data[spoofing_conditions].index
            
            for idx in spoof_indices:
                confidence = min(0.90, imbalance.iloc[idx] + 0.2)
                
                if confidence > self.thresholds['spoofing']:
                    signal = ManipulationSignal(
                        timestamp=order_book_data.loc[idx, 'timestamp'] if 'timestamp' in order_book_data.columns else datetime.now(),
                        symbol=order_book_data.get('symbol', 'UNKNOWN').iloc[0] if 'symbol' in order_book_data.columns else 'UNKNOWN',
                        manipulation_type='spoofing',
                        confidence=confidence,
                        severity='medium',
                        evidence={
                            'order_imbalance': imbalance.iloc[idx],
                            'bid_size': order_book_data.loc[idx, 'bid_size'],
                            'ask_size': order_book_data.loc[idx, 'ask_size']
                        },
                        risk_score=confidence * 8,
                        recommended_action='MONITOR_CLOSELY'
                    )
                    signals.append(signal)
        
        except Exception as e:
            logger.error(f"Error in spoofing detection: {e}")
        
        return signals
    
    async def _detect_wash_trading(self, features: pd.DataFrame, 
                                 trade_data: Optional[pd.DataFrame]) -> List[ManipulationSignal]:
        """Detect wash trading patterns"""
        signals = []
        
        if trade_data is None:
            return signals
        
        try:
            # Look for patterns indicating wash trading:
            # - High volume with minimal price movement
            # - Repetitive trade sizes
            # - Unusual trade timing patterns
            
            # High volume, low price movement
            volume_price_efficiency = abs(features['price_change']) / (features['volume_ratio'] + 0.001)
            
            wash_conditions = (
                (features['volume_ratio'] > 2.0) &  # High volume
                (volume_price_efficiency < 0.01) &  # Low price impact
                (abs(features['price_change']) < 0.005)  # Minimal price change
            )
            
            wash_indices = features[wash_conditions].index
            
            for idx in wash_indices:
                confidence = min(0.85, features.loc[idx, 'volume_ratio'] * 0.3)
                
                if confidence > self.thresholds['wash_trading']:
                    signal = ManipulationSignal(
                        timestamp=features.loc[idx, 'timestamp'] if 'timestamp' in features.columns else datetime.now(),
                        symbol=features.get('symbol', 'UNKNOWN').iloc[0] if 'symbol' in features.columns else 'UNKNOWN',
                        manipulation_type='wash_trading',
                        confidence=confidence,
                        severity='medium',
                        evidence={
                            'volume_ratio': features.loc[idx, 'volume_ratio'],
                            'price_change': features.loc[idx, 'price_change'],
                            'volume_price_efficiency': volume_price_efficiency.iloc[idx]
                        },
                        risk_score=confidence * 7,
                        recommended_action='REDUCE_POSITION_SIZE'
                    )
                    signals.append(signal)
        
        except Exception as e:
            logger.error(f"Error in wash trading detection: {e}")
        
        return signals
    
    async def _detect_layering(self, features: pd.DataFrame, 
                             order_book_data: Optional[pd.DataFrame]) -> List[ManipulationSignal]:
        """Detect layering manipulation"""
        signals = []
        
        if order_book_data is None:
            return signals
        
        try:
            # Layering involves placing multiple orders at different price levels
            # to create false impression of support/resistance
            
            # This is a simplified detection - real layering detection requires
            # detailed order flow analysis
            
            # Look for unusual order book depth patterns
            if 'bid_levels' in order_book_data.columns and 'ask_levels' in order_book_data.columns:
                depth_imbalance = abs(order_book_data['bid_levels'] - order_book_data['ask_levels'])
                
                layering_conditions = depth_imbalance > depth_imbalance.quantile(0.95)
                
                layer_indices = order_book_data[layering_conditions].index
                
                for idx in layer_indices:
                    confidence = min(0.80, depth_imbalance.iloc[idx] / depth_imbalance.max())
                    
                    if confidence > self.thresholds['layering']:
                        signal = ManipulationSignal(
                            timestamp=order_book_data.loc[idx, 'timestamp'] if 'timestamp' in order_book_data.columns else datetime.now(),
                            symbol=order_book_data.get('symbol', 'UNKNOWN').iloc[0] if 'symbol' in order_book_data.columns else 'UNKNOWN',
                            manipulation_type='layering',
                            confidence=confidence,
                            severity='low',
                            evidence={
                                'depth_imbalance': depth_imbalance.iloc[idx],
                                'bid_levels': order_book_data.loc[idx, 'bid_levels'],
                                'ask_levels': order_book_data.loc[idx, 'ask_levels']
                            },
                            risk_score=confidence * 6,
                            recommended_action='MONITOR'
                        )
                        signals.append(signal)
        
        except Exception as e:
            logger.error(f"Error in layering detection: {e}")
        
        return signals
    
    async def _detect_momentum_ignition(self, features: pd.DataFrame) -> List[ManipulationSignal]:
        """Detect momentum ignition patterns"""
        signals = []
        
        try:
            # Look for sudden aggressive trades that trigger momentum
            momentum_spike = (
                (abs(features['price_change']) > features['price_volatility'] * 3) &
                (features['volume_ratio'] > 2.5) &
                (features['price_momentum'].abs() > 0.05)
            )
            
            momentum_indices = features[momentum_spike].index
            
            for idx in momentum_indices:
                confidence = min(0.75, abs(features.loc[idx, 'price_change']) * 10 + 
                               features.loc[idx, 'volume_ratio'] * 0.1)
                
                if confidence > self.thresholds['momentum_ignition']:
                    signal = ManipulationSignal(
                        timestamp=features.loc[idx, 'timestamp'] if 'timestamp' in features.columns else datetime.now(),
                        symbol=features.get('symbol', 'UNKNOWN').iloc[0] if 'symbol' in features.columns else 'UNKNOWN',
                        manipulation_type='momentum_ignition',
                        confidence=confidence,
                        severity='medium',
                        evidence={
                            'price_change': features.loc[idx, 'price_change'],
                            'volume_ratio': features.loc[idx, 'volume_ratio'],
                            'price_momentum': features.loc[idx, 'price_momentum']
                        },
                        risk_score=confidence * 7,
                        recommended_action='WAIT_FOR_CONFIRMATION'
                    )
                    signals.append(signal)
        
        except Exception as e:
            logger.error(f"Error in momentum ignition detection: {e}")
        
        return signals
    
    async def _lstm_anomaly_detection(self, features: pd.DataFrame) -> List[ManipulationSignal]:
        """Use LSTM model for pattern-based anomaly detection"""
        signals = []
        
        try:
            if not self.model_trained:
                logger.warning("LSTM model not trained, skipping LSTM detection")
                return signals
            
            # Prepare data for LSTM
            feature_columns = ['price_change', 'volume_ratio', 'rsi', 'price_volatility', 
                             'volume_change', 'price_momentum', 'macd', 'macd_signal']
            
            available_columns = [col for col in feature_columns if col in features.columns]
            
            if len(available_columns) < 4:
                logger.warning("Insufficient features for LSTM detection")
                return signals
            
            # Normalize features
            feature_data = features[available_columns].values
            feature_data = self.feature_scaler.fit_transform(feature_data)
            
            # Create sequences for LSTM
            sequence_length = 20
            sequences = []
            
            for i in range(sequence_length, len(feature_data)):
                sequences.append(feature_data[i-sequence_length:i])
            
            if len(sequences) == 0:
                return signals
            
            sequences = np.array(sequences)
            sequences_tensor = torch.FloatTensor(sequences).to(self.device)
            
            # Get predictions
            self.lstm_model.eval()
            with torch.no_grad():
                predictions = self.lstm_model(sequences_tensor)
                anomaly_scores = predictions.cpu().numpy().flatten()
            
            # Identify anomalies
            anomaly_threshold = np.percentile(anomaly_scores, 95)
            anomaly_indices = np.where(anomaly_scores > anomaly_threshold)[0]
            
            for idx in anomaly_indices:
                actual_idx = idx + sequence_length
                if actual_idx < len(features):
                    confidence = min(0.90, anomaly_scores[idx])
                    
                    signal = ManipulationSignal(
                        timestamp=features.iloc[actual_idx]['timestamp'] if 'timestamp' in features.columns else datetime.now(),
                        symbol=features.get('symbol', 'UNKNOWN').iloc[0] if 'symbol' in features.columns else 'UNKNOWN',
                        manipulation_type='lstm_anomaly',
                        confidence=confidence,
                        severity='high' if confidence > 0.8 else 'medium',
                        evidence={
                            'anomaly_score': anomaly_scores[idx],
                            'threshold': anomaly_threshold,
                            'features_analyzed': available_columns
                        },
                        risk_score=confidence * 9,
                        recommended_action='INVESTIGATE'
                    )
                    signals.append(signal)
        
        except Exception as e:
            logger.error(f"Error in LSTM anomaly detection: {e}")
        
        return signals
    
    async def _statistical_anomaly_detection(self, features: pd.DataFrame) -> List[ManipulationSignal]:
        """Statistical anomaly detection using Isolation Forest"""
        signals = []
        
        try:
            # Select features for anomaly detection
            anomaly_features = ['price_change', 'volume_ratio', 'price_volatility', 'volume_change']
            available_features = [col for col in anomaly_features if col in features.columns]
            
            if len(available_features) < 2:
                logger.warning("Insufficient features for statistical anomaly detection")
                return signals
            
            feature_data = features[available_features].values
            
            # Fit isolation forest
            anomalies = self.isolation_forest.fit_predict(feature_data)
            anomaly_scores = self.isolation_forest.decision_function(feature_data)
            
            # Identify anomalous points
            anomaly_indices = np.where(anomalies == -1)[0]
            
            for idx in anomaly_indices:
                # Convert anomaly score to confidence (more negative = more anomalous)
                confidence = min(0.85, abs(anomaly_scores[idx]) * 0.5 + 0.3)
                
                if confidence > self.thresholds['general_anomaly']:
                    signal = ManipulationSignal(
                        timestamp=features.iloc[idx]['timestamp'] if 'timestamp' in features.columns else datetime.now(),
                        symbol=features.get('symbol', 'UNKNOWN').iloc[0] if 'symbol' in features.columns else 'UNKNOWN',
                        manipulation_type='statistical_anomaly',
                        confidence=confidence,
                        severity='low' if confidence < 0.6 else 'medium',
                        evidence={
                            'anomaly_score': anomaly_scores[idx],
                            'features_analyzed': available_features,
                            'feature_values': {col: features.iloc[idx][col] for col in available_features}
                        },
                        risk_score=confidence * 5,
                        recommended_action='MONITOR'
                    )
                    signals.append(signal)
        
        except Exception as e:
            logger.error(f"Error in statistical anomaly detection: {e}")
        
        return signals
    
    def _consolidate_signals(self, signals: List[ManipulationSignal]) -> List[ManipulationSignal]:
        """Remove duplicate signals and rank by confidence"""
        
        if not signals:
            return signals
        
        # Group signals by timestamp and symbol
        signal_groups = {}
        
        for signal in signals:
            key = (signal.timestamp, signal.symbol)
            if key not in signal_groups:
                signal_groups[key] = []
            signal_groups[key].append(signal)
        
        # Keep highest confidence signal from each group
        consolidated = []
        
        for group in signal_groups.values():
            if len(group) == 1:
                consolidated.append(group[0])
            else:
                # Keep signal with highest confidence
                best_signal = max(group, key=lambda s: s.confidence)
                
                # Combine evidence from all signals
                combined_evidence = {}
                manipulation_types = []
                
                for sig in group:
                    combined_evidence.update(sig.evidence)
                    manipulation_types.append(sig.manipulation_type)
                
                best_signal.manipulation_type = ','.join(set(manipulation_types))
                best_signal.evidence = combined_evidence
                
                consolidated.append(best_signal)
        
        # Sort by confidence descending
        consolidated.sort(key=lambda s: s.confidence, reverse=True)
        
        return consolidated
    
    # Technical indicator helper methods
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        ma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = ma + (std * std_dev)
        lower = ma - (std * std_dev)
        return upper, lower
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD indicator"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        return macd, macd_signal
    
    async def _add_order_book_features(self, features: pd.DataFrame, 
                                     order_book_data: pd.DataFrame) -> pd.DataFrame:
        """Add order book derived features"""
        
        # Calculate bid-ask spread
        if 'bid_price' in order_book_data.columns and 'ask_price' in order_book_data.columns:
            features['bid_ask_spread'] = (order_book_data['ask_price'] - order_book_data['bid_price']) / \
                                       order_book_data['bid_price']
        
        # Calculate order imbalance
        if 'bid_size' in order_book_data.columns and 'ask_size' in order_book_data.columns:
            total_size = order_book_data['bid_size'] + order_book_data['ask_size']
            features['order_imbalance'] = (order_book_data['bid_size'] - order_book_data['ask_size']) / total_size
        
        return features
    
    async def _add_trade_features(self, features: pd.DataFrame, 
                                trade_data: pd.DataFrame) -> pd.DataFrame:
        """Add trade-derived features"""
        
        # Calculate average trade size anomalies
        if 'trade_size' in trade_data.columns:
            avg_trade_size = trade_data['trade_size'].rolling(100).mean()
            features['trade_size_anomaly'] = (trade_data['trade_size'] - avg_trade_size) / avg_trade_size
        
        return features

    async def train_lstm_model(self, historical_data: pd.DataFrame, 
                             manipulation_labels: Optional[pd.Series] = None):
        """Train the LSTM model on historical data"""
        
        try:
            logger.info("Training LSTM manipulation detection model...")
            
            # Prepare training data
            features = await self._prepare_features(historical_data, None, None)
            
            feature_columns = ['price_change', 'volume_ratio', 'rsi', 'price_volatility', 
                             'volume_change', 'price_momentum', 'macd', 'macd_signal']
            
            available_columns = [col for col in feature_columns if col in features.columns]
            
            if len(available_columns) < 4:
                logger.error("Insufficient features for LSTM training")
                return False
            
            # Normalize features
            feature_data = features[available_columns].values
            feature_data = self.feature_scaler.fit_transform(feature_data)
            
            # Create sequences
            sequence_length = 20
            X, y = [], []
            
            for i in range(sequence_length, len(feature_data)):
                X.append(feature_data[i-sequence_length:i])
                
                # If labels provided, use them; otherwise use anomaly detection
                if manipulation_labels is not None:
                    y.append(manipulation_labels.iloc[i])
                else:
                    # Use statistical anomaly as pseudo-label
                    anomaly_score = self.isolation_forest.fit_predict(feature_data[i:i+1])[0]
                    y.append(1 if anomaly_score == -1 else 0)
            
            if len(X) == 0:
                logger.error("No training sequences created")
                return False
            
            X = torch.FloatTensor(np.array(X)).to(self.device)
            y = torch.FloatTensor(np.array(y)).to(self.device)
            
            # Training loop
            optimizer = torch.optim.Adam(self.lstm_model.parameters(), lr=0.001)
            criterion = nn.BCELoss()
            
            self.lstm_model.train()
            
            for epoch in range(50):
                optimizer.zero_grad()
                outputs = self.lstm_model(X).squeeze()
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                
                if epoch % 10 == 0:
                    logger.info(f"Training epoch {epoch}, Loss: {loss.item():.4f}")
            
            self.model_trained = True
            logger.info("LSTM model training completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error training LSTM model: {e}")
            return False

# Factory function for easy instantiation
def create_manipulation_detector() -> AdvancedManipulationDetector:
    """Create and return a configured manipulation detector"""
    return AdvancedManipulationDetector() 