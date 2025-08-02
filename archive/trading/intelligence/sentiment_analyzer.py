"""
Advanced Sentiment Analysis for Kimera Trading

Uses multiple NLP models to understand market sentiment from text.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import numpy as np

# NLP Libraries
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import nltk

# Ensure NLTK data is downloaded
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

logger = logging.getLogger(__name__)


class AdvancedSentimentAnalyzer:
    """
    Multi-model sentiment analysis combining:
    - FinBERT (financial sentiment)
    - VADER (social media sentiment)
    - TextBlob (general sentiment)
    """
    
    def __init__(self):
        """Initialize sentiment analysis models."""
        logger.info("Initializing Advanced Sentiment Analyzer...")
        
        # VADER for social media
        self.vader = SentimentIntensityAnalyzer()
        
        # Check for GPU
        self.device = 0 if torch.cuda.is_available() else -1
        if self.device == 0:
            logger.info("🚀 Using GPU for sentiment analysis")
        else:
            logger.warning("⚠️ GPU not available, using CPU for sentiment analysis")
        
        # Initialize FinBERT for financial sentiment
        try:
            self.finbert = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                device=self.device
            )
            logger.info("✅ FinBERT loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load FinBERT: {e}. Using fallback models.")
            self.finbert = None
    
    def analyze_text(self, text: str, source_type: str = 'general') -> Dict[str, Any]:
        """
        Analyze sentiment of text using multiple models.
        
        Args:
            text: Text to analyze
            source_type: Type of source ('news', 'social', 'financial', 'general')
            
        Returns:
            Comprehensive sentiment analysis
        """
        results = {
            'text': text[:200] + '...' if len(text) > 200 else text,
            'timestamp': datetime.now(),
            'source_type': source_type
        }
        
        # VADER Analysis (good for social media)
        vader_scores = self.vader.polarity_scores(text)
        results['vader'] = {
            'compound': vader_scores['compound'],
            'positive': vader_scores['pos'],
            'negative': vader_scores['neg'],
            'neutral': vader_scores['neu']
        }
        
        # TextBlob Analysis (general sentiment)
        try:
            blob = TextBlob(text)
            results['textblob'] = {
                'polarity': blob.sentiment.polarity,  # -1 to 1
                'subjectivity': blob.sentiment.subjectivity  # 0 to 1
            }
        except Exception as e:
            logger.warning(f"TextBlob analysis failed: {e}")
            results['textblob'] = {'polarity': 0, 'subjectivity': 0.5}
        
        # FinBERT Analysis (financial sentiment)
        if self.finbert and source_type in ['news', 'financial']:
            try:
                # FinBERT has max length of 512 tokens
                finbert_result = self.finbert(text[:512])[0]
                
                # Convert label to numeric score
                label_map = {'positive': 1, 'negative': -1, 'neutral': 0}
                results['finbert'] = {
                    'label': finbert_result['label'],
                    'score': finbert_result['score'],
                    'sentiment': label_map.get(finbert_result['label'], 0)
                }
            except Exception as e:
                logger.warning(f"FinBERT analysis failed: {e}")
                results['finbert'] = {'label': 'neutral', 'score': 0.5, 'sentiment': 0}
        
        # Composite sentiment score
        results['composite_sentiment'] = self._calculate_composite_sentiment(results, source_type)
        
        # Market interpretation
        results['market_signal'] = self._interpret_for_market(results['composite_sentiment'])
        
        return results
    
    def _calculate_composite_sentiment(self, results: Dict[str, Any], source_type: str) -> float:
        """
        Calculate weighted composite sentiment score.
        
        Args:
            results: Individual model results
            source_type: Type of source
            
        Returns:
            Composite sentiment score (-1 to 1)
        """
        weights = {
            'news': {'vader': 0.2, 'textblob': 0.2, 'finbert': 0.6},
            'social': {'vader': 0.6, 'textblob': 0.3, 'finbert': 0.1},
            'financial': {'vader': 0.1, 'textblob': 0.2, 'finbert': 0.7},
            'general': {'vader': 0.4, 'textblob': 0.4, 'finbert': 0.2}
        }
        
        source_weights = weights.get(source_type, weights['general'])
        composite = 0.0
        
        # VADER component
        if 'vader' in results:
            composite += source_weights['vader'] * results['vader']['compound']
        
        # TextBlob component
        if 'textblob' in results:
            composite += source_weights['textblob'] * results['textblob']['polarity']
        
        # FinBERT component
        if 'finbert' in results and self.finbert:
            composite += source_weights['finbert'] * results['finbert']['sentiment']
        
        return np.clip(composite, -1, 1)
    
    def _interpret_for_market(self, sentiment: float) -> Dict[str, Any]:
        """
        Interpret sentiment score for market action.
        
        Args:
            sentiment: Composite sentiment score
            
        Returns:
            Market interpretation
        """
        if sentiment > 0.5:
            signal = 'strong_bullish'
            confidence = min(sentiment, 0.9)
        elif sentiment > 0.2:
            signal = 'bullish'
            confidence = 0.6 + (sentiment - 0.2) * 0.5
        elif sentiment > -0.2:
            signal = 'neutral'
            confidence = 0.4 + abs(sentiment)
        elif sentiment > -0.5:
            signal = 'bearish'
            confidence = 0.6 + (abs(sentiment) - 0.2) * 0.5
        else:
            signal = 'strong_bearish'
            confidence = min(abs(sentiment), 0.9)
        
        return {
            'signal': signal,
            'confidence': confidence,
            'action_suggestion': self._get_action_suggestion(signal, confidence)
        }
    
    def _get_action_suggestion(self, signal: str, confidence: float) -> str:
        """Get trading action suggestion based on signal and confidence."""
        if confidence < 0.5:
            return 'wait'
        
        actions = {
            'strong_bullish': 'buy_aggressive',
            'bullish': 'buy_conservative',
            'neutral': 'hold',
            'bearish': 'sell_conservative',
            'strong_bearish': 'sell_aggressive'
        }
        
        return actions.get(signal, 'hold')
    
    def analyze_multiple_texts(self, texts: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Analyze multiple texts and aggregate sentiment.
        
        Args:
            texts: List of dicts with 'text' and 'source_type'
            
        Returns:
            Aggregated sentiment analysis
        """
        if not texts:
            return {
                'count': 0,
                'composite_sentiment': 0,
                'market_signal': {'signal': 'neutral', 'confidence': 0}
            }
        
        sentiments = []
        market_signals = []
        
        for item in texts:
            result = self.analyze_text(
                item.get('text', ''),
                item.get('source_type', 'general')
            )
            sentiments.append(result['composite_sentiment'])
            market_signals.append(result['market_signal'])
        
        # Calculate aggregated metrics
        avg_sentiment = np.mean(sentiments)
        sentiment_std = np.std(sentiments)
        
        # Aggregate market signals
        signal_counts = {}
        total_confidence = 0
        
        for signal in market_signals:
            sig = signal['signal']
            signal_counts[sig] = signal_counts.get(sig, 0) + 1
            total_confidence += signal['confidence']
        
        # Determine dominant signal
        dominant_signal = max(signal_counts, key=signal_counts.get)
        signal_consensus = signal_counts[dominant_signal] / len(market_signals)
        
        return {
            'count': len(texts),
            'composite_sentiment': avg_sentiment,
            'sentiment_std': sentiment_std,
            'market_signal': {
                'signal': dominant_signal,
                'confidence': (total_confidence / len(market_signals)) * signal_consensus,
                'consensus': signal_consensus
            },
            'individual_results': [
                {
                    'text': t.get('text', '')[:100] + '...',
                    'sentiment': s,
                    'signal': ms['signal']
                }
                for t, s, ms in zip(texts, sentiments, market_signals)
            ]
        }


class CryptoSpecificSentiment:
    """
    Crypto-specific sentiment patterns and analysis.
    """
    
    # Crypto-specific sentiment indicators
    BULLISH_KEYWORDS = [
        'moon', 'lambo', 'hodl', 'buy the dip', 'diamond hands',
        'to the moon', 'bullish', 'pump', 'breakout', 'ath',
        'all time high', 'rocket', '🚀', '💎', '🙌', 'gm',
        'wagmi', 'lfg', 'ngmi'
    ]
    
    BEARISH_KEYWORDS = [
        'dump', 'crash', 'bear', 'sell', 'short', 'rekt',
        'bearish', 'bubble', 'scam', 'rug', 'pullback',
        'correction', 'capitulation', 'blood', 'red', '📉',
        'ngmi', 'cope', 'exit liquidity'
    ]
    
    FUD_KEYWORDS = [
        'fud', 'fear', 'uncertainty', 'doubt', 'panic',
        'worried', 'concerned', 'regulation', 'ban', 'illegal'
    ]
    
    FOMO_KEYWORDS = [
        'fomo', 'missing out', 'last chance', 'hurry',
        'explosive', 'parabolic', 'don\'t miss', 'opportunity'
    ]
    
    @staticmethod
    def analyze_crypto_sentiment(text: str) -> Dict[str, Any]:
        """
        Analyze text for crypto-specific sentiment patterns.
        """
        text_lower = text.lower()
        
        bullish = sum(1 for word in CryptoSpecificSentiment.BULLISH_KEYWORDS if word in text_lower)
        bearish = sum(1 for word in CryptoSpecificSentiment.BEARISH_KEYWORDS if word in text_lower)
        fud = sum(1 for word in CryptoSpecificSentiment.FUD_KEYWORDS if word in text_lower)
        fomo = sum(1 for word in CryptoSpecificSentiment.FOMO_KEYWORDS if word in text_lower)
        
        # Calculate sentiment score (-1 to 1)
        sentiment = (bullish - bearish) / (bullish + bearish + 1)
        
        result = {
            'crypto_sentiment': np.clip(sentiment, -1, 1),
            'bullish_keywords': bullish,
            'bearish_keywords': bearish,
            'fud_level': fud,
            'fomo_level': fomo
        }

        result['emotion'] = CryptoSpecificSentiment._get_market_emotion(bullish, bearish, fud, fomo)
        
        return result
    
    @staticmethod
    def _get_market_emotion(bullish: int, bearish: int, fud: int, fomo: int) -> str:
        """Determine overall market emotion."""
        emotions = {
            'euphoric': bullish + fomo,
            'fearful': bearish + fud,
            'neutral': 1  # Base level
        }
        
        dominant = max(emotions, key=emotions.get)
        
        if emotions[dominant] < 2:
            return 'neutral'
        
        if dominant == 'euphoric' and fomo > bullish:
            return 'fomo_driven'
        elif dominant == 'fearful' and fud > bearish:
            return 'fud_driven'
        
        return dominant 