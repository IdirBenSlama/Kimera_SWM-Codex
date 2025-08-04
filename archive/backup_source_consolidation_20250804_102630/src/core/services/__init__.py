"""
Services Module
==============

This module provides asynchronous background processing and essential services
to the Kimera core system, following aerospace-grade reliability standards.

Services include:
- Background job management with fault tolerance
- CLIP service for visual grounding capabilities
- Task scheduling and monitoring
- Service health checks and recovery

Design Principles:
- Non-blocking asynchronous operations
- Graceful degradation under failure
- Resource isolation and management
- Comprehensive monitoring and logging

Standards:
- DO-178C for software reliability
- ISO 26262 for functional safety
- NASA-STD-8719.13 for software safety
"""

from .background_job_manager import BackgroundJobManager, get_job_manager
from .clip_service_integration import CLIPServiceIntegration, get_clip_service

__all__ = [
    'BackgroundJobManager',
    'get_job_manager',
    'CLIPServiceIntegration', 
    'get_clip_service'
]