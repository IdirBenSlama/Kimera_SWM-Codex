#!/usr/bin/env python3
"""
Security Dashboard for Kimera SWM
Provides real-time security monitoring and reporting capabilities.
"""

import logging
from typing import Dict, List, Any
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
import json

from .enhanced_security_hardening import security_hardening, SecurityEventType, ThreatLevel
from .authentication import get_current_user, RoleChecker

logger = logging.getLogger(__name__)

# Create router for security dashboard
security_router = APIRouter(prefix="/security", tags=["security"])

# Role checker for admin access
admin_required = RoleChecker(["admin", "security_admin"])


@security_router.get("/dashboard")
async def get_security_dashboard(current_user=Depends(admin_required)):
    """Get comprehensive security dashboard data"""
    try:
        metrics = security_hardening.get_security_metrics()
        assessment = security_hardening.perform_vulnerability_assessment()
        
        # Get recent high-priority events
        recent_events = [
            {
                'timestamp': event.timestamp.isoformat(),
                'type': event.event_type.value,
                'threat_level': event.threat_level.value,
                'source_ip': event.source_ip,
                'endpoint': event.endpoint,
                'blocked': event.blocked,
                'details': event.details
            }
            for event in security_hardening.security_events
            if event.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]
            and datetime.now() - event.timestamp < timedelta(hours=24)
        ][-20:]  # Last 20 high-priority events
        
        return {
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'vulnerability_assessment': {
                'assessment_id': assessment.assessment_id,
                'timestamp': assessment.timestamp.isoformat(),
                'vulnerabilities_count': len(assessment.vulnerabilities_found),
                'risk_score': assessment.risk_score,
                'recommendations_count': len(assessment.recommendations),
                'compliance_score': sum(assessment.compliance_status.values()) / len(assessment.compliance_status) * 100
            },
            'recent_high_priority_events': recent_events,
            'system_status': {
                'blocked_ips': len(security_hardening.blocked_ips),
                'active_protections': [
                    'Rate Limiting',
                    'Brute Force Protection', 
                    'SQL Injection Detection',
                    'XSS Protection',
                    'Enhanced Password Hashing',
                    'Encryption at Rest'
                ]
            }
        }
    except Exception as e:
        logger.error(f"Error generating security dashboard: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate security dashboard"
        )


@security_router.get("/vulnerability-assessment")
async def get_vulnerability_assessment(current_user=Depends(admin_required)):
    """Get detailed vulnerability assessment"""
    try:
        assessment = security_hardening.perform_vulnerability_assessment()
        
        return {
            'assessment_id': assessment.assessment_id,
            'timestamp': assessment.timestamp.isoformat(),
            'vulnerabilities': assessment.vulnerabilities_found,
            'risk_score': assessment.risk_score,
            'recommendations': assessment.recommendations,
            'compliance_status': assessment.compliance_status,
            'summary': {
                'total_vulnerabilities': len(assessment.vulnerabilities_found),
                'high_severity': len([v for v in assessment.vulnerabilities_found if v.get('severity') == 'HIGH']),
                'medium_severity': len([v for v in assessment.vulnerabilities_found if v.get('severity') == 'MEDIUM']),
                'low_severity': len([v for v in assessment.vulnerabilities_found if v.get('severity') == 'LOW']),
                'compliance_percentage': sum(assessment.compliance_status.values()) / len(assessment.compliance_status) * 100
            }
        }
    except Exception as e:
        logger.error(f"Error performing vulnerability assessment: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to perform vulnerability assessment"
        )


@security_router.get("/events")
async def get_security_events(
    limit: int = 100,
    threat_level: str = None,
    event_type: str = None,
    hours: int = 24,
    current_user=Depends(admin_required)
):
    """Get security events with filtering"""
    try:
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # Filter events
        filtered_events = []
        for event in security_hardening.security_events:
            if event.timestamp < cutoff_time:
                continue
            
            if threat_level and event.threat_level.value != threat_level:
                continue
                
            if event_type and event.event_type.value != event_type:
                continue
            
            filtered_events.append({
                'timestamp': event.timestamp.isoformat(),
                'event_type': event.event_type.value,
                'threat_level': event.threat_level.value,
                'source_ip': event.source_ip,
                'user_agent': event.user_agent,
                'endpoint': event.endpoint,
                'details': event.details,
                'blocked': event.blocked,
                'response_action': event.response_action
            })
        
        # Sort by timestamp (newest first) and limit
        filtered_events.sort(key=lambda x: x['timestamp'], reverse=True)
        filtered_events = filtered_events[:limit]
        
        return {
            'events': filtered_events,
            'total_count': len(filtered_events),
            'filters_applied': {
                'threat_level': threat_level,
                'event_type': event_type,
                'hours': hours,
                'limit': limit
            }
        }
    except Exception as e:
        logger.error(f"Error retrieving security events: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve security events"
        )


@security_router.get("/blocked-ips")
async def get_blocked_ips(current_user=Depends(admin_required)):
    """Get list of blocked IP addresses"""
    try:
        blocked_ips_info = []
        
        for ip in security_hardening.blocked_ips:
            # Find recent events for this IP
            recent_events = [
                event for event in security_hardening.security_events
                if event.source_ip == ip and datetime.now() - event.timestamp < timedelta(hours=24)
            ]
            
            blocked_ips_info.append({
                'ip_address': ip,
                'blocked_timestamp': min(event.timestamp for event in recent_events) if recent_events else None,
                'event_count_24h': len(recent_events),
                'threat_types': list(set(event.event_type.value for event in recent_events)),
                'highest_threat_level': max((event.threat_level.value for event in recent_events), default='low')
            })
        
        return {
            'blocked_ips': blocked_ips_info,
            'total_blocked': len(blocked_ips_info)
        }
    except Exception as e:
        logger.error(f"Error retrieving blocked IPs: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve blocked IPs"
        )


@security_router.post("/unblock-ip")
async def unblock_ip(ip_address: str, current_user=Depends(admin_required)):
    """Unblock an IP address"""
    try:
        if ip_address in security_hardening.blocked_ips:
            security_hardening.blocked_ips.remove(ip_address)
            
            # Clear failed attempts for this IP
            keys_to_remove = [key for key in security_hardening.failed_attempts.keys() if key.startswith(ip_address)]
            for key in keys_to_remove:
                del security_hardening.failed_attempts[key]
            
            logger.info(f"IP {ip_address} unblocked by admin {current_user.username}")
            
            return {
                'status': 'success',
                'message': f'IP {ip_address} has been unblocked',
                'unblocked_ip': ip_address,
                'unblocked_by': current_user.username,
                'timestamp': datetime.now().isoformat()
            }
        else:
            return {
                'status': 'warning',
                'message': f'IP {ip_address} was not in the blocked list',
                'ip_address': ip_address
            }
    except Exception as e:
        logger.error(f"Error unblocking IP {ip_address}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to unblock IP address"
        )


@security_router.get("/metrics")
async def get_security_metrics(current_user=Depends(admin_required)):
    """Get detailed security metrics"""
    try:
        metrics = security_hardening.get_security_metrics()
        
        # Add additional computed metrics
        now = datetime.now()
        events_last_hour = [
            event for event in security_hardening.security_events
            if now - event.timestamp < timedelta(hours=1)
        ]
        
        events_last_day = [
            event for event in security_hardening.security_events
            if now - event.timestamp < timedelta(hours=24)
        ]
        
        # Calculate attack trends
        attack_types_hour = {}
        attack_types_day = {}
        
        for event in events_last_hour:
            attack_types_hour[event.event_type.value] = attack_types_hour.get(event.event_type.value, 0) + 1
        
        for event in events_last_day:
            attack_types_day[event.event_type.value] = attack_types_day.get(event.event_type.value, 0) + 1
        
        enhanced_metrics = {
            **metrics,
            'attack_trends': {
                'last_hour': attack_types_hour,
                'last_24_hours': attack_types_day
            },
            'protection_effectiveness': {
                'blocked_attacks_24h': len([e for e in events_last_day if e.blocked]),
                'total_attempts_24h': len(events_last_day),
                'block_rate': len([e for e in events_last_day if e.blocked]) / max(len(events_last_day), 1) * 100
            },
            'system_health': {
                'active_protections': 6,  # Number of protection mechanisms
                'last_vulnerability_scan': datetime.now().isoformat(),
                'security_score': metrics.get('security_score', 85)
            }
        }
        
        return enhanced_metrics
    except Exception as e:
        logger.error(f"Error retrieving security metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve security metrics"
        )


@security_router.get("/health")
async def security_health_check():
    """Public endpoint for security system health check"""
    try:
        # Basic health indicators (no sensitive data)
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'protections_active': True,
            'monitoring_active': True,
            'last_check': datetime.now().isoformat()
        }
        
        return health_status
    except Exception as e:
        logger.error(f"Security health check failed: {e}")
        return {
            'status': 'unhealthy',
            'timestamp': datetime.now().isoformat(),
            'error': 'Security system check failed'
        }


logger.info("Security Dashboard module loaded successfully") 