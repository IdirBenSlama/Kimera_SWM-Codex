import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.security.validator import InputValidator, create_user_input_validator


def test_validate_ip_address():
    validator = InputValidator()
    assert validator.validate_ip_address("127.0.0.1")
    assert not validator.validate_ip_address("999.999.999.999")


def test_validate_url():
    validator = InputValidator()
    assert validator.validate_url("https://example.com")
    assert not validator.validate_url("javascript:alert(1)")


def test_create_user_input_validator_rules():
    validator = create_user_input_validator()
    valid, errors = validator.validate(
        {
            "username": "user123",
            "email": "user@example.com",
            "password": "strongpass",
        }
    )
    assert valid
    valid, errors = validator.validate(
        {
            "username": "u",
            "email": "bad",
            "password": "short",
        }
    )
    assert not valid
    assert errors
