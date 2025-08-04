"""
Conflict Resolver
=================

Resolves conflicts between System 1 and System 2 outputs.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ConflictResolver:
    """Resolves conflicts between dual-system outputs"""

    def __init__(self):
        self.resolution_history = []
        self.strategies = {
            'confidence': self._resolve_by_confidence,
            'consensus': self._resolve_by_consensus,
            'weighted': self._resolve_by_weighting
        }

    def resolve(self,
                system1_output: Dict[str, Any],
                system2_output: Dict[str, Any],
                strategy: str = 'weighted') -> Dict[str, Any]:
        """Resolve conflict between system outputs"""

        if strategy not in self.strategies:
            strategy = 'weighted'

        resolution = self.strategies[strategy](system1_output, system2_output)

        # Record resolution
        self.resolution_history.append({
            'strategy': strategy,
            'inputs': (system1_output, system2_output),
            'output': resolution,
            'timestamp': str(datetime.now())
        })

        return resolution

    def _resolve_by_confidence(self, s1: Dict[str, Any], s2: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve by selecting higher confidence output"""
        conf1 = s1.get('confidence', 0.0)
        conf2 = s2.get('confidence', 0.0)

        return s1 if conf1 >= conf2 else s2

    def _resolve_by_consensus(self, s1: Dict[str, Any], s2: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve by finding consensus elements"""
        # Simplified consensus - would need semantic comparison
        return {
            'type': 'consensus',
            'elements': [s1, s2],
            'confidence': (s1.get('confidence', 0) + s2.get('confidence', 0)) / 2
        }

    def _resolve_by_weighting(self, s1: Dict[str, Any], s2: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve by weighted combination"""
        w1 = s1.get('confidence', 0.5)
        w2 = s2.get('confidence', 0.5)

        # Normalize weights
        total = w1 + w2
        if total > 0:
            w1 /= total
            w2 /= total
        else:
            w1 = w2 = 0.5

        return {
            'type': 'weighted',
            'system1_contribution': s1,
            'system2_contribution': s2,
            'weights': {'system1': w1, 'system2': w2},
            'confidence': w1 * s1.get('confidence', 0) + w2 * s2.get('confidence', 0)
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get resolution statistics"""
        if not self.resolution_history:
            return {'total_resolutions': 0}

        strategy_counts = {}
        for res in self.resolution_history:
            strategy = res['strategy']
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

        return {
            'total_resolutions': len(self.resolution_history),
            'strategy_usage': strategy_counts
        }


# Missing import
from datetime import datetime
