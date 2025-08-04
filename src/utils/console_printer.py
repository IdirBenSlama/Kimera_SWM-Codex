#!/usr/bin/env python3
"""
Canonical Console Printer for Kimera SWM
========================================

This module provides a centralized, single source of truth for all
user-facing console output across the Kimera codebase, especially for
high-level orchestration scripts, demos, and test runners.

It separates the concern of user-facing CLI presentation from system logging.
"""

import logging
import shutil

logger = logging.getLogger(__name__)


def get_terminal_width(default=80):
    """Gets the current terminal width."""
    return shutil.get_terminal_size((default, 20)).columns


def print_header(title: str, char: str = "=", width: int = None):
    """Prints a standardized header."""
    if width is None:
        width = get_terminal_width()
    logger.info("\n" + char * width)
    logger.info(title.center(width))
    logger.info(char * width)


def print_subheader(title: str, char: str = "-"):
    """Prints a standardized subheader."""
    logger.info(f"\n--- {title} ---")


def print_kv(key: str, value: str, indent: int = 2):
    """Prints a key-value pair."""
    logger.info(f"{' ' * indent}{key}: {value}")


def print_list(items: list, indent: int = 2):
    """Prints a list of items."""
    for item in items:
        logger.info(f"{' ' * indent}• {item}")


def print_major_section_header(title: str, char: str = "🌟"):
    """Prints a visually distinct major section header for server startup."""
    width = get_terminal_width()
    border = char * width
    logger.info("\n" + border)
    logger.info(char + " " * (width - 2) + char)
    logger.info(char + title.center(width - 2) + char)
    logger.info(char + " " * (width - 2) + char)
    logger.info(border)


def print_info(message: str):
    """Prints an informational message."""
    logger.info(message)


def print_success(message: str):
    """Prints a success message."""
    logger.info(f"✅ {message}")


def print_warning(message: str):
    """Prints a warning message."""
    logger.info(f"⚠️  {message}")


def print_error(message: str):
    """Prints an error message."""
    logger.info(f"❌ {message}")


def print_line(char: str = "=", width: int = None):
    """Prints a horizontal line."""
    if width is None:
        width = get_terminal_width()
    logger.info(char * width)
