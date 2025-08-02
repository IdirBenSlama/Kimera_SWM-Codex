"""KIMERA Trading Execution Module

This module provides KIMERA with the ability to execute real-world trading actions.
It bridges the gap between cognitive analysis and market execution.
"""

from .kimera_action_interface import (
    KimeraActionInterface,
    ActionType,
    ExecutionStatus,
    ActionRequest,
    ActionResult,
    CognitiveFeedbackProcessor,
    create_kimera_action_interface
)

__all__ = [
    "KimeraActionInterface",
    "ActionType", 
    "ExecutionStatus",
    "ActionRequest",
    "ActionResult",
    "CognitiveFeedbackProcessor",
    "create_kimera_action_interface"
] 