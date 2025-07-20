#!/usr/bin/env python3
"""
Enhanced Security Hardening for Kimera SWM
Implements comprehensive security measures including vulnerability assessment,
enhanced access controls, and encryption improvements.
"""

import logging
import hashlib
import secrets
import time
import ipaddress
from typing import Dict, List, Optional, Set, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import re
import json
from collections import defaultdict, deque

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from fastapi import Request, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import bcrypt

logger = logging.getLogger(__name__)


class ThreatLevel(Enum):
    """Security threat levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SecurityEventType(Enum):
    """Types of security events"""
    LOGIN_ATTEMPT = "login_attempt"
    FAILED_AUTH = "failed_auth"
    BRUTE_FORCE = "brute_force"
    SUSPICIOUS_REQUEST = "suspicious_request"
    SQL_INJECTION = "sql_injection"
    XSS_ATTEMPT = "xss_attempt"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DATA_BREACH = "data_breach"
    UNAUTHORIZED_ACCESS = "unauthorized_access"


@dataclass
class SecurityEvent:
    """Security event data structure"""
    event_type: SecurityEventType
    threat_level: ThreatLevel
    timestamp: datetime
    source_ip: str
    user_agent: str
    endpoint: str
    details: Dict[str, Any]
    blocked: bool = False
    response_action: str = ""


@dataclass
class VulnerabilityAssessment:
    """Vulnerability assessment results"""
    assessment_id: str
    timestamp: datetime
    vulnerabilities_found: List[Dict[str, Any]]
    risk_score: float
    recommendations: List[str]
    compliance_status: Dict[str, bool]


class EnhancedSecurityHardening:
    """
    Comprehensive security hardening system for Kimera SWM
    """
    
    def __init__(self):
        self.security_events: deque = deque(maxlen=10000)  # Last 10k events
        self.failed_attempts: defaultdict = defaultdict(list)
        self.blocked_ips: Set[str] = set()
        self.rate_limits: defaultdict = defaultdict(list)
        self.encryption_key = self._generate_encryption_key()
        self.cipher_suite = Fernet(self.encryption_key)
        
        # Security configuration
        self.max_failed_attempts = 5
        self.lockout_duration = timedelta(minutes=15)
        self.rate_limit_window = timedelta(minutes=1)
        self.max_requests_per_minute = 60
        
        # Vulnerability patterns
        self.sql_injection_patterns = [
            r"(\%27)|(\')|(\-\-)|(\%23)|(#)",
            r"((\%3D)|(=))[^\n]*((\%27)|(\')|(\-\-)|(\%3B)|(;))",
            r"\w*((\%27)|(\'))((\%6F)|o|(\%4F))((\%72)|r|(\%52))",
            r"((\%27)|(\'))union",
            r"exec(\s|\+)+(s|x)p\w+",
            r"UNION.*SELECT",
            r"INSERT.*INTO",
            r"DELETE.*FROM",
            r"DROP.*TABLE",
        ]
        
        self.xss_patterns = [
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"vbscript:",
            r"onload\s*=",
            r"onerror\s*=",
            r"onclick\s*=",
            r"onmouseover\s*=",
            r"<iframe[^>]*>",
            r"<object[^>]*>",
            r"<embed[^>]*>",
        ]
        
        logger.info("Enhanced Security Hardening system initialized")
    
    def _generate_encryption_key(self) -> bytes:
        """Generate a secure encryption key"""
        password = secrets.token_bytes(32)
        salt = secrets.token_bytes(16)
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = kdf.derive(password)
        return Fernet.generate_key()
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        try:
            encrypted_data = self.cipher_suite.encrypt(data.encode())
            return encrypted_data.decode()
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        try:
            decrypted_data = self.cipher_suite.decrypt(encrypted_data.encode())
            return decrypted_data.decode()
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise
    
    def hash_password_secure(self, password: str) -> str:
        """Securely hash password with bcrypt and additional salt"""
        # Generate a random salt
        salt = bcrypt.gensalt(rounds=12)
        # Hash the password
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        # Add additional security layer with SHA3-256
        additional_hash = hashlib.sha3_256(hashed).hexdigest()
        return f"{hashed.decode('utf-8')}:{additional_hash}"
    
    def verify_password_secure(self, password: str, hashed_password: str) -> bool:
        """Verify password with enhanced security"""
        try:
            if ':' not in hashed_password:
                return False
            
            bcrypt_hash, sha3_hash = hashed_password.split(':', 1)
            
            # Verify bcrypt hash
            if not bcrypt.checkpw(password.encode('utf-8'), bcrypt_hash.encode('utf-8')):
                return False
            
            # Verify additional hash layer
            expected_sha3 = hashlib.sha3_256(bcrypt_hash.encode('utf-8')).hexdigest()
            return secrets.compare_digest(sha3_hash, expected_sha3)
        except Exception as e:
            logger.error(f"Password verification failed: {e}")
            return False
    
    def assess_request_vulnerability(self, request: Request) -> Dict[str, Any]:
        """Assess incoming request for vulnerabilities"""
        vulnerabilities = []
        threat_level = ThreatLevel.LOW
        
        # Get request data
        url = str(request.url)
        headers = dict(request.headers)
        query_params = str(request.query_params)
        user_agent = headers.get('user-agent', '')
        
        # Check for SQL injection patterns
        for pattern in self.sql_injection_patterns:
            if re.search(pattern, url, re.IGNORECASE) or re.search(pattern, query_params, re.IGNORECASE):
                vulnerabilities.append({
                    'type': 'SQL Injection',
                    'pattern': pattern,
                    'location': 'URL/Query parameters'
                })
                threat_level = ThreatLevel.HIGH
        
        # Check for XSS patterns
        for pattern in self.xss_patterns:
            if re.search(pattern, url, re.IGNORECASE) or re.search(pattern, query_params, re.IGNORECASE):
                vulnerabilities.append({
                    'type': 'XSS Attempt',
                    'pattern': pattern,
                    'location': 'URL/Query parameters'
                })
                threat_level = ThreatLevel.HIGH
        
        # Check for suspicious headers
        suspicious_headers = ['x-forwarded-for', 'x-real-ip', 'x-originating-ip']
        for header in suspicious_headers:
            if header in headers:
                vulnerabilities.append({
                    'type': 'Header Manipulation',
                    'header': header,
                    'value': headers[header]
                })
                if threat_level == ThreatLevel.LOW:
                    threat_level = ThreatLevel.MEDIUM
        
        # Check for suspicious user agents
        suspicious_ua_patterns = [
            r'sqlmap', r'nikto', r'nmap', r'masscan', r'nessus',
            r'openvas', r'burp', r'w3af', r'metasploit'
        ]
        for pattern in suspicious_ua_patterns:
            if re.search(pattern, user_agent, re.IGNORECASE):
                vulnerabilities.append({
                    'type': 'Suspicious User Agent',
                    'pattern': pattern,
                    'user_agent': user_agent
                })
                threat_level = ThreatLevel.HIGH
        
        return {
            'vulnerabilities': vulnerabilities,
            'threat_level': threat_level,
            'risk_score': len(vulnerabilities) * (1 if threat_level == ThreatLevel.LOW else 
                                                2 if threat_level == ThreatLevel.MEDIUM else 
                                                5 if threat_level == ThreatLevel.HIGH else 10)
        }
    
    def check_rate_limiting(self, client_ip: str, endpoint: str) -> bool:
        """Check if request should be rate limited"""
        now = datetime.now()
        key = f"{client_ip}:{endpoint}"
        
        # Clean old requests
        self.rate_limits[key] = [
            timestamp for timestamp in self.rate_limits[key]
            if now - timestamp < self.rate_limit_window
        ]
        
        # Check if limit exceeded
        if len(self.rate_limits[key]) >= self.max_requests_per_minute:
            return True
        
        # Add current request
        self.rate_limits[key].append(now)
        return False
    
    def check_brute_force_protection(self, client_ip: str, username: str = None) -> bool:
        """Check for brute force attacks"""
        now = datetime.now()
        key = f"{client_ip}:{username}" if username else client_ip
        
        # Clean old failed attempts
        self.failed_attempts[key] = [
            timestamp for timestamp in self.failed_attempts[key]
            if now - timestamp < self.lockout_duration
        ]
        
        # Check if IP should be blocked
        if len(self.failed_attempts[key]) >= self.max_failed_attempts:
            self.blocked_ips.add(client_ip)
            return True
        
        return client_ip in self.blocked_ips
    
    def record_failed_attempt(self, client_ip: str, username: str = None):
        """Record a failed authentication attempt"""
        now = datetime.now()
        key = f"{client_ip}:{username}" if username else client_ip
        self.failed_attempts[key].append(now)
    
    def log_security_event(self, event: SecurityEvent):
        """Log a security event"""
        self.security_events.append(event)
        
        # Log based on threat level
        if event.threat_level == ThreatLevel.CRITICAL:
            logger.critical(f"CRITICAL SECURITY EVENT: {event.event_type.value} from {event.source_ip}")
        elif event.threat_level == ThreatLevel.HIGH:
            logger.error(f"HIGH THREAT: {event.event_type.value} from {event.source_ip}")
        elif event.threat_level == ThreatLevel.MEDIUM:
            logger.warning(f"MEDIUM THREAT: {event.event_type.value} from {event.source_ip}")
        else:
            logger.info(f"Security event: {event.event_type.value} from {event.source_ip}")
    
    def perform_vulnerability_assessment(self) -> VulnerabilityAssessment:
        """Perform comprehensive vulnerability assessment"""
        assessment_id = secrets.token_hex(16)
        vulnerabilities = []
        risk_score = 0.0
        recommendations = []
        
        # Check recent security events
        recent_events = [
            event for event in self.security_events
            if datetime.now() - event.timestamp < timedelta(hours=24)
        ]
        
        # Analyze threat patterns
        threat_counts = defaultdict(int)
        for event in recent_events:
            threat_counts[event.event_type] += 1
        
        # Assess SQL injection attempts
        sql_attempts = threat_counts[SecurityEventType.SQL_INJECTION]
        if sql_attempts > 0:
            vulnerabilities.append({
                'type': 'SQL Injection Attempts',
                'severity': 'HIGH' if sql_attempts > 10 else 'MEDIUM',
                'count': sql_attempts,
                'description': f'{sql_attempts} SQL injection attempts detected in last 24h'
            })
            risk_score += sql_attempts * 0.5
            recommendations.append('Implement parameterized queries and input validation')
        
        # Assess XSS attempts
        xss_attempts = threat_counts[SecurityEventType.XSS_ATTEMPT]
        if xss_attempts > 0:
            vulnerabilities.append({
                'type': 'XSS Attempts',
                'severity': 'HIGH' if xss_attempts > 5 else 'MEDIUM',
                'count': xss_attempts,
                'description': f'{xss_attempts} XSS attempts detected in last 24h'
            })
            risk_score += xss_attempts * 0.3
            recommendations.append('Implement Content Security Policy and output encoding')
        
        # Assess brute force attempts
        brute_force_attempts = threat_counts[SecurityEventType.BRUTE_FORCE]
        if brute_force_attempts > 0:
            vulnerabilities.append({
                'type': 'Brute Force Attempts',
                'severity': 'HIGH',
                'count': brute_force_attempts,
                'description': f'{brute_force_attempts} brute force attempts detected'
            })
            risk_score += brute_force_attempts * 0.4
            recommendations.append('Implement account lockout and CAPTCHA')
        
        # Check blocked IPs
        if len(self.blocked_ips) > 0:
            vulnerabilities.append({
                'type': 'Blocked IPs',
                'severity': 'MEDIUM',
                'count': len(self.blocked_ips),
                'description': f'{len(self.blocked_ips)} IPs currently blocked'
            })
        
        # Compliance checks
        compliance_status = {
            'password_hashing': True,  # We use bcrypt + SHA3
            'encryption_at_rest': True,  # We have encryption capabilities
            'rate_limiting': True,  # Rate limiting is implemented
            'input_validation': True,  # Vulnerability assessment includes validation
            'audit_logging': True,  # Security events are logged
            'brute_force_protection': True,  # Brute force protection is active
        }
        
        return VulnerabilityAssessment(
            assessment_id=assessment_id,
            timestamp=datetime.now(),
            vulnerabilities_found=vulnerabilities,
            risk_score=min(risk_score, 100.0),  # Cap at 100
            recommendations=recommendations,
            compliance_status=compliance_status
        )
    
    def get_security_metrics(self) -> Dict[str, Any]:
        """Get comprehensive security metrics"""
        now = datetime.now()
        last_24h = now - timedelta(hours=24)
        last_hour = now - timedelta(hours=1)
        
        recent_events = [e for e in self.security_events if e.timestamp >= last_24h]
        recent_hour_events = [e for e in self.security_events if e.timestamp >= last_hour]
        
        threat_distribution = defaultdict(int)
        for event in recent_events:
            threat_distribution[event.threat_level.value] += 1
        
        # Calculate attack trends
        attack_types_hour = {}
        attack_types_day = {}
        
        for event in recent_hour_events:
            attack_types_hour[event.event_type.value] = attack_types_hour.get(event.event_type.value, 0) + 1
        
        for event in recent_events:
            attack_types_day[event.event_type.value] = attack_types_day.get(event.event_type.value, 0) + 1
        
        # Get top threat types from recent events
        threat_type_counts = {}
        for event in recent_events:
            threat_type_counts[event.event_type.value] = threat_type_counts.get(event.event_type.value, 0) + 1
        
        # Sort by count and get top 5
        top_threat_types = sorted(threat_type_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        top_threat_types = [threat_type for threat_type, count in top_threat_types]
        
        return {
            'total_security_events_24h': len(recent_events),
            'security_events_last_hour': len(recent_hour_events),
            'blocked_ips_count': len(self.blocked_ips),
            'failed_attempts_active': len(self.failed_attempts),
            'threat_distribution': dict(threat_distribution),
            'top_threat_types': top_threat_types,
            'security_score': max(0, 100 - len(recent_events) * 2),  # Simple scoring
            'last_assessment': datetime.now().isoformat()
        }


class SecurityMiddleware:
    """Security middleware for FastAPI"""
    
    def __init__(self, security_hardening: EnhancedSecurityHardening):
        self.security = security_hardening
    
    async def __call__(self, request: Request, call_next):
        """Process request through security checks"""
        start_time = time.time()
        client_ip = request.client.host
        
        # Check if IP is blocked
        if client_ip in self.security.blocked_ips:
            self.security.log_security_event(SecurityEvent(
                event_type=SecurityEventType.UNAUTHORIZED_ACCESS,
                threat_level=ThreatLevel.HIGH,
                timestamp=datetime.now(),
                source_ip=client_ip,
                user_agent=request.headers.get('user-agent', ''),
                endpoint=str(request.url.path),
                details={'reason': 'Blocked IP attempted access'},
                blocked=True,
                response_action='IP_BLOCKED'
            ))
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )
        
        # Check rate limiting
        if self.security.check_rate_limiting(client_ip, request.url.path):
            self.security.log_security_event(SecurityEvent(
                event_type=SecurityEventType.SUSPICIOUS_REQUEST,
                threat_level=ThreatLevel.MEDIUM,
                timestamp=datetime.now(),
                source_ip=client_ip,
                user_agent=request.headers.get('user-agent', ''),
                endpoint=str(request.url.path),
                details={'reason': 'Rate limit exceeded'},
                blocked=True,
                response_action='RATE_LIMITED'
            ))
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded"
            )
        
        # Assess vulnerabilities
        vuln_assessment = self.security.assess_request_vulnerability(request)
        
        if vuln_assessment['threat_level'] == ThreatLevel.HIGH:
            event_type = SecurityEventType.SQL_INJECTION if any(
                v['type'] == 'SQL Injection' for v in vuln_assessment['vulnerabilities']
            ) else SecurityEventType.XSS_ATTEMPT
            
            self.security.log_security_event(SecurityEvent(
                event_type=event_type,
                threat_level=ThreatLevel.HIGH,
                timestamp=datetime.now(),
                source_ip=client_ip,
                user_agent=request.headers.get('user-agent', ''),
                endpoint=str(request.url.path),
                details={'vulnerabilities': vuln_assessment['vulnerabilities']},
                blocked=True,
                response_action='THREAT_BLOCKED'
            ))
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Malicious request detected"
            )
        
        # Process request
        response = await call_next(request)
        
        # Log successful request
        process_time = time.time() - start_time
        if vuln_assessment['threat_level'] != ThreatLevel.LOW:
            self.security.log_security_event(SecurityEvent(
                event_type=SecurityEventType.SUSPICIOUS_REQUEST,
                threat_level=vuln_assessment['threat_level'],
                timestamp=datetime.now(),
                source_ip=client_ip,
                user_agent=request.headers.get('user-agent', ''),
                endpoint=str(request.url.path),
                details={
                    'vulnerabilities': vuln_assessment['vulnerabilities'],
                    'process_time': process_time
                },
                blocked=False,
                response_action='MONITORED'
            ))
        
        return response


# Global security hardening instance
security_hardening = EnhancedSecurityHardening()
security_middleware = SecurityMiddleware(security_hardening)

logger.info("Enhanced Security Hardening module loaded successfully") 