"""
Advanced Rules Engine for KIMERA Trading
Implements business rule management, decision trees, and complex trading logic
"""

import logging
import json
import yaml
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class RuleType(Enum):
    """Types of trading rules"""
    ENTRY = "entry"
    EXIT = "exit"
    RISK_MANAGEMENT = "risk_management"
    POSITION_SIZING = "position_sizing"
    MARKET_CONDITION = "market_condition"
    PORTFOLIO = "portfolio"

class ConditionOperator(Enum):
    """Condition operators for rules"""
    EQUALS = "=="
    NOT_EQUALS = "!="
    GREATER_THAN = ">"
    LESS_THAN = "<"
    GREATER_EQUAL = ">="
    LESS_EQUAL = "<="
    IN = "in"
    NOT_IN = "not_in"
    CONTAINS = "contains"
    AND = "and"
    OR = "or"
    NOT = "not"

@dataclass
class RuleCondition:
    """Individual rule condition"""
    field: str
    operator: ConditionOperator
    value: Any
    weight: float = 1.0

@dataclass
class RuleAction:
    """Action to take when rule is triggered"""
    action_type: str
    parameters: Dict[str, Any]
    priority: int = 1

@dataclass
class TradingRule:
    """Complete trading rule definition"""
    rule_id: str
    name: str
    description: str
    rule_type: RuleType
    conditions: List[RuleCondition]
    actions: List[RuleAction]
    enabled: bool = True
    created_at: datetime = None
    updated_at: datetime = None

@dataclass
class RuleExecutionResult:
    """Result of rule execution"""
    rule_id: str
    triggered: bool
    confidence: float
    actions_taken: List[RuleAction]
    execution_time: datetime
    metadata: Dict[str, Any]

class RuleEvaluator(ABC):
    """Abstract base class for rule evaluators"""
    
    @abstractmethod
    def evaluate(self, condition: RuleCondition, context: Dict[str, Any]) -> bool:
        """Evaluate a single condition"""
        pass

class StandardRuleEvaluator(RuleEvaluator):
    """Standard rule evaluator for basic conditions"""
    
    def evaluate(self, condition: RuleCondition, context: Dict[str, Any]) -> bool:
        """Evaluate a single condition against context"""
        try:
            field_value = self._get_field_value(condition.field, context)
            
            if field_value is None:
                return False
            
            return self._apply_operator(field_value, condition.operator, condition.value)
            
        except Exception as e:
            logger.error(f"Error evaluating condition {condition.field}: {e}")
            return False
    
    def _get_field_value(self, field: str, context: Dict[str, Any]) -> Any:
        """Get field value from context, supporting nested fields"""
        try:
            # Support nested field access with dot notation
            if '.' in field:
                parts = field.split('.')
                value = context
                for part in parts:
                    value = value[part]
                return value
            else:
                return context.get(field)
        except (KeyError, TypeError):
            return None
    
    def _apply_operator(self, field_value: Any, operator: ConditionOperator, condition_value: Any) -> bool:
        """Apply operator to compare field value with condition value"""
        try:
            if operator == ConditionOperator.EQUALS:
                return field_value == condition_value
            elif operator == ConditionOperator.NOT_EQUALS:
                return field_value != condition_value
            elif operator == ConditionOperator.GREATER_THAN:
                return float(field_value) > float(condition_value)
            elif operator == ConditionOperator.LESS_THAN:
                return float(field_value) < float(condition_value)
            elif operator == ConditionOperator.GREATER_EQUAL:
                return float(field_value) >= float(condition_value)
            elif operator == ConditionOperator.LESS_EQUAL:
                return float(field_value) <= float(condition_value)
            elif operator == ConditionOperator.IN:
                return field_value in condition_value
            elif operator == ConditionOperator.NOT_IN:
                return field_value not in condition_value
            elif operator == ConditionOperator.CONTAINS:
                return condition_value in str(field_value)
            else:
                return False
        except (ValueError, TypeError):
            return False

class DecisionTree:
    """Decision tree for complex rule evaluation"""
    
    def __init__(self, name: str):
        self.name = name
        self.root = None
        self.nodes = {}
    
    def add_node(self, node_id: str, condition: RuleCondition, 
                 true_branch: str = None, false_branch: str = None,
                 action: RuleAction = None):
        """Add a node to the decision tree"""
        self.nodes[node_id] = {
            'condition': condition,
            'true_branch': true_branch,
            'false_branch': false_branch,
            'action': action
        }
        
        if self.root is None:
            self.root = node_id
    
    def evaluate(self, context: Dict[str, Any], evaluator: RuleEvaluator) -> List[RuleAction]:
        """Evaluate the decision tree and return actions"""
        actions = []
        
        if self.root is None:
            return actions
        
        current_node = self.root
        visited = set()
        
        while current_node and current_node not in visited:
            visited.add(current_node)
            node = self.nodes.get(current_node)
            
            if not node:
                break
            
            # Evaluate condition
            condition_result = evaluator.evaluate(node['condition'], context)
            
            # Add action if present
            if node['action']:
                actions.append(node['action'])
            
            # Move to next node
            if condition_result and node['true_branch']:
                current_node = node['true_branch']
            elif not condition_result and node['false_branch']:
                current_node = node['false_branch']
            else:
                break
        
        return actions

class AdvancedRulesEngine:
    """
    Advanced rules engine for trading decisions
    Supports:
    - Complex rule definitions
    - Decision trees
    - Rule priorities and weights
    - Dynamic rule loading
    - Performance monitoring
    """
    
    def __init__(self):
        self.rules: Dict[str, TradingRule] = {}
        self.decision_trees: Dict[str, DecisionTree] = {}
        self.evaluator = StandardRuleEvaluator()
        self.execution_history: List[RuleExecutionResult] = []
        self.rule_performance: Dict[str, Dict[str, float]] = {}
        
        # Load default rules
        self._load_default_rules()
    
    def add_rule(self, rule: TradingRule) -> bool:
        """Add a new trading rule"""
        try:
            if rule.created_at is None:
                rule.created_at = datetime.now()
            rule.updated_at = datetime.now()
            
            self.rules[rule.rule_id] = rule
            logger.info(f"Added rule: {rule.name} ({rule.rule_id})")
            return True
            
        except Exception as e:
            logger.error(f"Error adding rule {rule.rule_id}: {e}")
            return False
    
    def remove_rule(self, rule_id: str) -> bool:
        """Remove a trading rule"""
        try:
            if rule_id in self.rules:
                del self.rules[rule_id]
                logger.info(f"Removed rule: {rule_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error removing rule {rule_id}: {e}")
            return False
    
    def update_rule(self, rule_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing rule"""
        try:
            if rule_id not in self.rules:
                return False
            
            rule = self.rules[rule_id]
            
            for key, value in updates.items():
                if hasattr(rule, key):
                    setattr(rule, key, value)
            
            rule.updated_at = datetime.now()
            logger.info(f"Updated rule: {rule_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating rule {rule_id}: {e}")
            return False
    
    def evaluate_rules(self, context: Dict[str, Any], 
                      rule_types: List[RuleType] = None) -> List[RuleExecutionResult]:
        """
        Evaluate all applicable rules against the given context
        
        Args:
            context: Trading context (market data, portfolio, etc.)
            rule_types: Specific rule types to evaluate (optional)
            
        Returns:
            List of rule execution results
        """
        results = []
        
        try:
            # Filter rules by type if specified
            rules_to_evaluate = self.rules.values()
            if rule_types:
                rules_to_evaluate = [r for r in rules_to_evaluate if r.rule_type in rule_types]
            
            # Filter enabled rules
            rules_to_evaluate = [r for r in rules_to_evaluate if r.enabled]
            
            # Sort by priority (if actions have priority)
            for rule in rules_to_evaluate:
                result = self._evaluate_single_rule(rule, context)
                results.append(result)
                
                # Track performance
                self._update_rule_performance(rule.rule_id, result)
            
            # Sort results by confidence descending
            results.sort(key=lambda x: x.confidence, reverse=True)
            
            logger.info(f"Evaluated {len(rules_to_evaluate)} rules, {len([r for r in results if r.triggered])} triggered")
            
        except Exception as e:
            logger.error(f"Error evaluating rules: {e}")
        
        return results
    
    def _evaluate_single_rule(self, rule: TradingRule, context: Dict[str, Any]) -> RuleExecutionResult:
        """Evaluate a single trading rule"""
        try:
            start_time = datetime.now()
            
            # Evaluate all conditions
            condition_results = []
            total_weight = 0
            
            for condition in rule.conditions:
                result = self.evaluator.evaluate(condition, context)
                condition_results.append(result)
                total_weight += condition.weight
            
            # Calculate overall confidence based on weighted conditions
            if not condition_results:
                triggered = False
                confidence = 0.0
            else:
                weighted_score = sum(
                    result * condition.weight 
                    for result, condition in zip(condition_results, rule.conditions)
                )
                confidence = weighted_score / total_weight if total_weight > 0 else 0.0
                triggered = confidence > 0.5  # Majority of weighted conditions must be true
            
            # Determine actions to take
            actions_taken = []
            if triggered:
                actions_taken = rule.actions.copy()
            
            execution_time = datetime.now() - start_time
            
            result = RuleExecutionResult(
                rule_id=rule.rule_id,
                triggered=triggered,
                confidence=confidence,
                actions_taken=actions_taken,
                execution_time=start_time,
                metadata={
                    'rule_name': rule.name,
                    'rule_type': rule.rule_type.value,
                    'condition_results': condition_results,
                    'execution_duration_ms': execution_time.total_seconds() * 1000
                }
            )
            
            # Store in history
            self.execution_history.append(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error evaluating rule {rule.rule_id}: {e}")
            return RuleExecutionResult(
                rule_id=rule.rule_id,
                triggered=False,
                confidence=0.0,
                actions_taken=[],
                execution_time=datetime.now(),
                metadata={'error': str(e)}
            )
    
    def _update_rule_performance(self, rule_id: str, result: RuleExecutionResult):
        """Update rule performance metrics"""
        if rule_id not in self.rule_performance:
            self.rule_performance[rule_id] = {
                'total_evaluations': 0,
                'total_triggered': 0,
                'avg_confidence': 0.0,
                'success_rate': 0.0
            }
        
        perf = self.rule_performance[rule_id]
        perf['total_evaluations'] += 1
        
        if result.triggered:
            perf['total_triggered'] += 1
        
        # Update average confidence
        perf['avg_confidence'] = (
            (perf['avg_confidence'] * (perf['total_evaluations'] - 1) + result.confidence) /
            perf['total_evaluations']
        )
        
        # Update success rate
        perf['success_rate'] = perf['total_triggered'] / perf['total_evaluations']
    
    def get_rule_performance(self, rule_id: str = None) -> Dict[str, Any]:
        """Get performance metrics for rules"""
        if rule_id:
            return self.rule_performance.get(rule_id, {})
        return self.rule_performance.copy()
    
    def load_rules_from_file(self, file_path: str) -> bool:
        """Load rules from JSON or YAML file"""
        try:
            with open(file_path, 'r') as f:
                if file_path.endswith('.yaml') or file_path.endswith('.yml'):
                    data = yaml.safe_load(f)
                else:
                    data = json.load(f)
            
            rules_data = data.get('rules', [])
            loaded_count = 0
            
            for rule_data in rules_data:
                rule = self._dict_to_rule(rule_data)
                if rule and self.add_rule(rule):
                    loaded_count += 1
            
            logger.info(f"Loaded {loaded_count} rules from {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading rules from {file_path}: {e}")
            return False
    
    def save_rules_to_file(self, file_path: str) -> bool:
        """Save rules to JSON or YAML file"""
        try:
            rules_data = {
                'rules': [self._rule_to_dict(rule) for rule in self.rules.values()]
            }
            
            with open(file_path, 'w') as f:
                if file_path.endswith('.yaml') or file_path.endswith('.yml'):
                    yaml.dump(rules_data, f, default_flow_style=False)
                else:
                    json.dump(rules_data, f, indent=2, default=str)
            
            logger.info(f"Saved {len(self.rules)} rules to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving rules to {file_path}: {e}")
            return False
    
    def _dict_to_rule(self, data: Dict[str, Any]) -> Optional[TradingRule]:
        """Convert dictionary to TradingRule object"""
        try:
            conditions = []
            for cond_data in data.get('conditions', []):
                condition = RuleCondition(
                    field=cond_data['field'],
                    operator=ConditionOperator(cond_data['operator']),
                    value=cond_data['value'],
                    weight=cond_data.get('weight', 1.0)
                )
                conditions.append(condition)
            
            actions = []
            for action_data in data.get('actions', []):
                action = RuleAction(
                    action_type=action_data['action_type'],
                    parameters=action_data.get('parameters', {}),
                    priority=action_data.get('priority', 1)
                )
                actions.append(action)
            
            rule = TradingRule(
                rule_id=data['rule_id'],
                name=data['name'],
                description=data.get('description', ''),
                rule_type=RuleType(data['rule_type']),
                conditions=conditions,
                actions=actions,
                enabled=data.get('enabled', True),
                created_at=datetime.fromisoformat(data['created_at']) if data.get('created_at') else None,
                updated_at=datetime.fromisoformat(data['updated_at']) if data.get('updated_at') else None
            )
            
            return rule
            
        except Exception as e:
            logger.error(f"Error converting dict to rule: {e}")
            return None
    
    def _rule_to_dict(self, rule: TradingRule) -> Dict[str, Any]:
        """Convert TradingRule object to dictionary"""
        return {
            'rule_id': rule.rule_id,
            'name': rule.name,
            'description': rule.description,
            'rule_type': rule.rule_type.value,
            'conditions': [
                {
                    'field': cond.field,
                    'operator': cond.operator.value,
                    'value': cond.value,
                    'weight': cond.weight
                }
                for cond in rule.conditions
            ],
            'actions': [
                {
                    'action_type': action.action_type,
                    'parameters': action.parameters,
                    'priority': action.priority
                }
                for action in rule.actions
            ],
            'enabled': rule.enabled,
            'created_at': rule.created_at.isoformat() if rule.created_at else None,
            'updated_at': rule.updated_at.isoformat() if rule.updated_at else None
        }
    
    def _load_default_rules(self):
        """Load default trading rules"""
        try:
            # RSI Oversold Entry Rule
            rsi_oversold_rule = TradingRule(
                rule_id="rsi_oversold_entry",
                name="RSI Oversold Entry",
                description="Enter long position when RSI is oversold",
                rule_type=RuleType.ENTRY,
                conditions=[
                    RuleCondition("indicators.rsi", ConditionOperator.LESS_THAN, 30, 1.0),
                    RuleCondition("market_data.volume_ratio", ConditionOperator.GREATER_THAN, 1.2, 0.5)
                ],
                actions=[
                    RuleAction("enter_long", {"position_size": 0.02, "stop_loss": 0.02}, 1)
                ]
            )
            
            # RSI Overbought Exit Rule
            rsi_overbought_rule = TradingRule(
                rule_id="rsi_overbought_exit",
                name="RSI Overbought Exit",
                description="Exit long position when RSI is overbought",
                rule_type=RuleType.EXIT,
                conditions=[
                    RuleCondition("indicators.rsi", ConditionOperator.GREATER_THAN, 70, 1.0),
                    RuleCondition("portfolio.position_size", ConditionOperator.GREATER_THAN, 0, 1.0)
                ],
                actions=[
                    RuleAction("exit_long", {"percentage": 0.5}, 1)
                ]
            )
            
            # Risk Management Rule
            risk_management_rule = TradingRule(
                rule_id="max_drawdown_protection",
                name="Maximum Drawdown Protection",
                description="Exit all positions if portfolio drawdown exceeds threshold",
                rule_type=RuleType.RISK_MANAGEMENT,
                conditions=[
                    RuleCondition("portfolio.drawdown", ConditionOperator.GREATER_THAN, 0.05, 1.0)
                ],
                actions=[
                    RuleAction("exit_all_positions", {}, 1)
                ]
            )
            
            # Volatility Position Sizing Rule
            volatility_sizing_rule = TradingRule(
                rule_id="volatility_position_sizing",
                name="Volatility-Based Position Sizing",
                description="Adjust position size based on market volatility",
                rule_type=RuleType.POSITION_SIZING,
                conditions=[
                    RuleCondition("market_data.volatility", ConditionOperator.GREATER_THAN, 0.02, 1.0)
                ],
                actions=[
                    RuleAction("adjust_position_size", {"multiplier": 0.5}, 1)
                ]
            )
            
            # Add default rules
            default_rules = [rsi_oversold_rule, rsi_overbought_rule, risk_management_rule, volatility_sizing_rule]
            
            for rule in default_rules:
                self.add_rule(rule)
            
            logger.info(f"Loaded {len(default_rules)} default trading rules")
            
        except Exception as e:
            logger.error(f"Error loading default rules: {e}")
    
    def create_decision_tree(self, name: str) -> DecisionTree:
        """Create a new decision tree"""
        tree = DecisionTree(name)
        self.decision_trees[name] = tree
        return tree
    
    def evaluate_decision_tree(self, tree_name: str, context: Dict[str, Any]) -> List[RuleAction]:
        """Evaluate a decision tree"""
        if tree_name not in self.decision_trees:
            logger.error(f"Decision tree '{tree_name}' not found")
            return []
        
        tree = self.decision_trees[tree_name]
        return tree.evaluate(context, self.evaluator)
    
    def get_execution_history(self, limit: int = 100) -> List[RuleExecutionResult]:
        """Get recent rule execution history"""
        return self.execution_history[-limit:]
    
    def clear_execution_history(self):
        """Clear rule execution history"""
        self.execution_history.clear()
        logger.info("Cleared rule execution history")

# Factory function
def create_rules_engine() -> AdvancedRulesEngine:
    """Create and return a configured rules engine"""
    return AdvancedRulesEngine()