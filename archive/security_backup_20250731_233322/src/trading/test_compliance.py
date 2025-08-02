"""Compliance tests for Kimera trading system security policies.

This module contains pytest tests verifying:
- Security policy enforcement
- Regulatory compliance checks
- Data protection mechanisms
- Audit logging requirements
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import MagicMock, patch
from cryptography.fernet import Fernet
from src.trading.security import SecurityManager
from src.utils.kimera_logger import get_trading_logger
logger = get_trading_logger(__name__)

@pytest.fixture
def mock_security_config():
    """Fixture providing mock security configuration."""
    return {
        'encryption_key': Fernet.generate_key().decode(),
        'audit_log_enabled': True,
        'data_retention_days': 90,
        'compliance_rules': {
            'gdpr': True,
            'pci_dss': False,
            'hipaa': False
        }
    }

@pytest.fixture
def security_manager(mock_security_config):
    """Fixture providing initialized SecurityManager."""
    return SecurityManager(mock_security_config)

@pytest.mark.asyncio
async def test_encryption_compliance(security_manager, mock_security_config):
    """Test data encryption meets compliance requirements."""
    logger.debug("Starting test_encryption_compliance")
    
    test_data = b"Sensitive trading data"
    encrypted = security_manager.encrypt_data(test_data)
    decrypted = security_manager.decrypt_data(encrypted)
    
    assert encrypted != test_data, "Data should be encrypted"
    assert decrypted == test_data, "Decrypted data should match original"
    logger.debug("Encryption/decryption cycle completed successfully")

@pytest.mark.asyncio
async def test_audit_logging(security_manager):
    """Test audit logging meets compliance requirements."""
    logger.debug("Starting test_audit_logging")
    
    test_event = {
        "timestamp": datetime.now().isoformat(),
        "user": "test_user",
        "action": "login",
        "status": "success"
    }
    
    with patch.object(security_manager, '_write_audit_log') as mock_log:
        await security_manager.log_audit_event(test_event)
        mock_log.assert_called_once_with(test_event)
        logger.debug("Audit logging verification passed")

@pytest.mark.asyncio
async def test_compliance_rule_validation(security_manager):
    """Test compliance rule validation logic."""
    logger.debug("Starting test_compliance_rule_validation")
    
    # Test GDPR compliance check
    assert security_manager.check_compliance('gdpr') == True
    
    # Test non-implemented compliance standard
    assert security_manager.check_compliance('pci_dss') == False
    logger.debug("Compliance rule validation tests passed")

@pytest.mark.asyncio
async def test_data_retention_policy(security_manager, mock_security_config):
    """Test data retention policy enforcement."""
    logger.debug("Starting test_data_retention_policy")
    
    test_data = {"key": "value"}
    retention_days = mock_security_config['data_retention_days']
    
    with patch('os.path.getmtime', return_value=datetime.now().timestamp()):
        assert security_manager.check_retention_compliance('test_file') == True
    
    with patch('os.path.getmtime', 
              return_value=(datetime.now() - timedelta(days=retention_days+1)).timestamp()):
        assert security_manager.check_retention_compliance('test_file') == False
    
    logger.debug("Data retention policy tests passed")