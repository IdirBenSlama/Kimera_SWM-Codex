# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- Comprehensive docstrings for test_market_analyzer.py
- New fields to KimeraMarketData class in core/types.py

### Changed
- Resolved circular imports by consolidating types in core/types.py
- Updated test cases for contradiction detection in test_market_analyzer.py
- Standardized logger imports across test files

### Fixed
- Windows path handling in test execution
- Market data copy functionality using dataclasses.replace()
- Timestamp handling in contradiction detection tests