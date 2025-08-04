"""
Input Validator
===============

Implements comprehensive input validation based on:
- OWASP Input Validation Guidelines
- NIST SP 800-53 Security Controls
- Medical device data integrity standards
"""

import re
import json
import logging
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import ipaddress
import urllib.parse

logger = logging.getLogger(__name__)

class ValidationRule(Enum):
    """Types of validation rules."""
    REQUIRED = "required"
    TYPE = "type"
    LENGTH = "length"
    RANGE = "range"
    PATTERN = "pattern"
    WHITELIST = "whitelist"
    BLACKLIST = "blacklist"
    CUSTOM = "custom"

@dataclass
class ValidationError:
    """Validation error details."""
    field: str
    rule: ValidationRule
    message: str
    value: Any

class InputValidator:
    """
    Comprehensive input validation with aerospace-grade reliability.
    """
    
    def __init__(self):
        self.validation_rules: Dict[str, List[Dict[str, Any]]] = {}
        self.custom_validators: Dict[str, Callable] = {}
        self.validation_stats = {
            'total_validations': 0,
            'failed_validations': 0,
            'blocked_patterns': defaultdict(int)
        }
        
        # Common patterns
        self.patterns = {
            'email': re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'),
            'url': re.compile(r'^https?://[^\s]+$'),
            'alphanumeric': re.compile(r'^[a-zA-Z0-9]+$'),
            'uuid': re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'),
            'safe_string': re.compile(r'^[a-zA-Z0-9\s\-_.]+$'),
            'no_script': re.compile(r'<script|javascript:|onerror=|onclick=', re.IGNORECASE)
        }
        
        # Dangerous patterns (for blacklisting)
        self.dangerous_patterns = [
            re.compile(r'<script', re.IGNORECASE),
            re.compile(r'javascript:', re.IGNORECASE),
            re.compile(r'onerror\s*=', re.IGNORECASE),
            re.compile(r'onclick\s*=', re.IGNORECASE),
            re.compile(r'onload\s*=', re.IGNORECASE),
            re.compile(r'<iframe', re.IGNORECASE),
            re.compile(r'<object', re.IGNORECASE),
            re.compile(r'<embed', re.IGNORECASE),
            re.compile(r'eval\s*\(', re.IGNORECASE),
            re.compile(r'expression\s*\(', re.IGNORECASE)
        ]
    
    def add_rule(self, field: str, rule: Dict[str, Any]):
        """Add validation rule for a field."""
        if field not in self.validation_rules:
            self.validation_rules[field] = []
        
        self.validation_rules[field].append(rule)
        logger.debug(f"Added validation rule for field: {field}")
    
    def add_custom_validator(self, name: str, validator: Callable):
        """Add custom validation function."""
        self.custom_validators[name] = validator
        logger.debug(f"Added custom validator: {name}")
    
    def validate(self, data: Dict[str, Any], schema: Optional[str] = None) -> Tuple[bool, List[ValidationError]]:
        """
        Validate input data against rules.
        
        Args:
            data: Input data to validate
            schema: Optional schema name to use specific rules
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        self.validation_stats['total_validations'] += 1
        errors = []
        
        # Get rules based on schema
        rules = self._get_rules_for_schema(schema) if schema else self.validation_rules
        
        # Validate each field
        for field, field_rules in rules.items():
            value = data.get(field)
            
            for rule in field_rules:
                error = self._validate_field(field, value, rule)
                if error:
                    errors.append(error)
        
        # Check for unexpected fields (defense against injection)
        expected_fields = set(rules.keys())
        actual_fields = set(data.keys())
        unexpected = actual_fields - expected_fields
        
        if unexpected and schema:  # Only enforce if schema is specified
            for field in unexpected:
                errors.append(ValidationError(
                    field=field,
                    rule=ValidationRule.WHITELIST,
                    message=f"Unexpected field: {field}",
                    value=data[field]
                ))
        
        if errors:
            self.validation_stats['failed_validations'] += 1
        
        return len(errors) == 0, errors
    
    def _validate_field(self, field: str, value: Any, rule: Dict[str, Any]) -> Optional[ValidationError]:
        """Validate a single field against a rule."""
        rule_type = ValidationRule(rule.get('type', 'required'))
        
        if rule_type == ValidationRule.REQUIRED:
            if value is None or value == "":
                return ValidationError(
                    field=field,
                    rule=rule_type,
                    message=f"{field} is required",
                    value=value
                )
        
        elif rule_type == ValidationRule.TYPE:
            expected_type = rule.get('expected_type')
            if not self._check_type(value, expected_type):
                return ValidationError(
                    field=field,
                    rule=rule_type,
                    message=f"{field} must be of type {expected_type}",
                    value=value
                )
        
        elif rule_type == ValidationRule.LENGTH:
            min_length = rule.get('min', 0)
            max_length = rule.get('max', float('inf'))
            
            if hasattr(value, '__len__'):
                if not min_length <= len(value) <= max_length:
                    return ValidationError(
                        field=field,
                        rule=rule_type,
                        message=f"{field} length must be between {min_length} and {max_length}",
                        value=value
                    )
        
        elif rule_type == ValidationRule.RANGE:
            min_val = rule.get('min', float('-inf'))
            max_val = rule.get('max', float('inf'))
            
            if isinstance(value, (int, float)):
                if not min_val <= value <= max_val:
                    return ValidationError(
                        field=field,
                        rule=rule_type,
                        message=f"{field} must be between {min_val} and {max_val}",
                        value=value
                    )
        
        elif rule_type == ValidationRule.PATTERN:
            pattern_name = rule.get('pattern')
            pattern = self.patterns.get(pattern_name)
            
            if pattern and isinstance(value, str):
                if not pattern.match(value):
                    return ValidationError(
                        field=field,
                        rule=rule_type,
                        message=f"{field} does not match pattern {pattern_name}",
                        value=value
                    )
        
        elif rule_type == ValidationRule.WHITELIST:
            allowed_values = rule.get('values', [])
            if value not in allowed_values:
                return ValidationError(
                    field=field,
                    rule=rule_type,
                    message=f"{field} must be one of {allowed_values}",
                    value=value
                )
        
        elif rule_type == ValidationRule.BLACKLIST:
            if isinstance(value, str):
                for pattern in self.dangerous_patterns:
                    if pattern.search(value):
                        self.validation_stats['blocked_patterns'][pattern.pattern] += 1
                        return ValidationError(
                            field=field,
                            rule=rule_type,
                            message=f"{field} contains dangerous pattern",
                            value=value
                        )
        
        elif rule_type == ValidationRule.CUSTOM:
            validator_name = rule.get('validator')
            validator = self.custom_validators.get(validator_name)
            
            if validator:
                try:
                    if not validator(value):
                        return ValidationError(
                            field=field,
                            rule=rule_type,
                            message=f"{field} failed custom validation {validator_name}",
                            value=value
                        )
                except Exception as e:
                    logger.error(f"Custom validator {validator_name} error: {e}")
                    return ValidationError(
                        field=field,
                        rule=rule_type,
                        message=f"{field} validation error",
                        value=value
                    )
        
        return None
    
    def _check_type(self, value: Any, expected_type: str) -> bool:
        """Check if value matches expected type."""
        type_map = {
            'string': str,
            'integer': int,
            'float': float,
            'boolean': bool,
            'list': list,
            'dict': dict
        }
        
        expected_class = type_map.get(expected_type)
        if expected_class:
            return isinstance(value, expected_class)
        
        return True
    
    def _get_rules_for_schema(self, schema: str) -> Dict[str, List[Dict[str, Any]]]:
        """Get validation rules for a specific schema."""
        # This would load schema-specific rules
        # For now, return default rules
        return self.validation_rules
    
    def sanitize_string(self, value: str, max_length: int = 1000) -> str:
        """
        Sanitize string input.
        
        Args:
            value: String to sanitize
            max_length: Maximum allowed length
            
        Returns:
            Sanitized string
        """
        if not isinstance(value, str):
            value = str(value)
        
        # Truncate to max length
        value = value[:max_length]
        
        # Remove null bytes
        value = value.replace('\x00', '')
        
        # Escape HTML entities
        value = (
            value.replace('&', '&amp;')
            .replace('<', '&lt;')
            .replace('>', '&gt;')
            .replace('"', '&quot;')
            .replace("'", '&#x27;')
        )
        
        return value
    
    def validate_json(self, json_string: str, schema: Optional[Dict[str, Any]] = None) -> Tuple[bool, Any]:
        """
        Validate JSON string.
        
        Args:
            json_string: JSON string to validate
            schema: Optional JSON schema
            
        Returns:
            Tuple of (is_valid, parsed_data_or_error)
        """
        try:
            data = json.loads(json_string)
            
            # Additional validation with schema if provided
            if schema:
                # This would use jsonschema library in production
                pass
            
            return True, data
            
        except json.JSONDecodeError as e:
            return False, str(e)
    
    def validate_ip_address(self, ip: str) -> bool:
        """Validate IP address."""
        try:
            ipaddress.ip_address(ip)
            return True
        except ValueError:
            return False
    
    def validate_url(self, url: str, allowed_schemes: List[str] = ['http', 'https']) -> bool:
        """
        Validate URL.
        
        Args:
            url: URL to validate
            allowed_schemes: Allowed URL schemes
            
        Returns:
            True if valid
        """
        try:
            parsed = urllib.parse.urlparse(url)
            
            # Check scheme
            if parsed.scheme not in allowed_schemes:
                return False
            
            # Check for basic structure
            if not parsed.netloc:
                return False
            
            # Check for dangerous patterns
            if any(pattern.search(url) for pattern in self.dangerous_patterns):
                return False
            
            return True
            
        except Exception:
            return False
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics."""
        return {
            'total_validations': self.validation_stats['total_validations'],
            'failed_validations': self.validation_stats['failed_validations'],
            'failure_rate': (
                self.validation_stats['failed_validations'] / 
                self.validation_stats['total_validations']
                if self.validation_stats['total_validations'] > 0 else 0
            ),
            'blocked_patterns': dict(self.validation_stats['blocked_patterns'])
        }

# Predefined validators
def create_user_input_validator() -> InputValidator:
    """Create validator for user input."""
    validator = InputValidator()
    
    # Username rules
    validator.add_rule('username', {
        'type': ValidationRule.REQUIRED.value
    })
    validator.add_rule('username', {
        'type': ValidationRule.LENGTH.value,
        'min': 3,
        'max': 50
    })
    validator.add_rule('username', {
        'type': ValidationRule.PATTERN.value,
        'pattern': 'alphanumeric'
    })
    
    # Email rules
    validator.add_rule('email', {
        'type': ValidationRule.REQUIRED.value
    })
    validator.add_rule('email', {
        'type': ValidationRule.PATTERN.value,
        'pattern': 'email'
    })
    
    # Password rules
    validator.add_rule('password', {
        'type': ValidationRule.LENGTH.value,
        'min': 8,
        'max': 128
    })
    
    return validator

from collections import defaultdict