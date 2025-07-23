# KIMERA SWM Makefile
# Common development tasks and utilities

.PHONY: help install test lint format clean build docs run-dev

# Default target
help:
	@echo "KIMERA SWM Development Commands"
	@echo "================================"
	@echo "install     - Install dependencies"
	@echo "test        - Run all tests"
	@echo "test-unit   - Run unit tests only"
	@echo "test-cov    - Run tests with coverage"
	@echo "lint        - Run linting checks"
	@echo "format      - Format code"
	@echo "type-check  - Run type checking"
	@echo "security    - Run security scans"
	@echo "clean       - Clean temporary files"
	@echo "build       - Build the project"
	@echo "docs        - Generate documentation"
	@echo "run-dev     - Run development server"
	@echo "health      - Check system health"
	@echo "verify      - Verify project organization"

# Installation
install:
	@echo "Installing dependencies..."
	pip install poetry
	poetry install --with dev

install-prod:
	@echo "Installing production dependencies..."
	poetry install --only main

# Testing
test:
	@echo "Running all tests..."
	poetry run pytest

test-unit:
	@echo "Running unit tests..."
	poetry run pytest tests/unit/

test-integration:
	@echo "Running integration tests..."
	poetry run pytest tests/integration/

test-performance:
	@echo "Running performance tests..."
	poetry run pytest tests/performance/

test-cov:
	@echo "Running tests with coverage..."
	poetry run pytest --cov=src tests/ --cov-report=html --cov-report=term

# Code Quality
lint:
	@echo "Running linting checks..."
	poetry run ruff check src/ tests/
	poetry run black --check src/ tests/
	poetry run isort --check-only src/ tests/

format:
	@echo "Formatting code..."
	poetry run black src/ tests/
	poetry run isort src/ tests/

type-check:
	@echo "Running type checks..."
	poetry run mypy src/

security:
	@echo "Running security scans..."
	poetry run bandit -r src/
	poetry run safety check

# Quality Gates (all checks)
check-all: lint type-check security test-cov
	@echo "All quality checks passed!"

# Cleaning
clean:
	@echo "Cleaning temporary files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	rm -rf htmlcov/
	rm -rf dist/
	rm -rf build/

# Building
build:
	@echo "Building project..."
	poetry build

# Documentation
docs:
	@echo "Generating documentation..."
	cd docs && make html

docs-serve:
	@echo "Serving documentation locally..."
	cd docs && python -m http.server 8000

# Development
run-dev:
	@echo "Starting development server..."
	poetry run python src/main.py --dev

run-api:
	@echo "Starting API server..."
	poetry run uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

run-dashboard:
	@echo "Starting dashboard..."
	cd dashboard && python -m http.server 8080

# System Management
health:
	@echo "Checking system health..."
	poetry run python scripts/verify_reorganization.py

verify:
	@echo "Verifying project organization..."
	poetry run python scripts/verify_reorganization.py

migrate-imports:
	@echo "Migrating import paths..."
	poetry run python scripts/update_imports.py

# Database
db-init:
	@echo "Initializing database..."
	poetry run python src/vault/database.py --init

db-migrate:
	@echo "Running database migrations..."
	poetry run alembic upgrade head

db-reset:
	@echo "Resetting database..."
	poetry run python src/vault/database.py --reset

# Trading System
trading-test:
	@echo "Testing trading systems..."
	poetry run pytest tests/integration/test_trading_*.py

trading-paper:
	@echo "Starting paper trading..."
	poetry run python src/trading/paper_trading.py

# GPU & Performance
gpu-test:
	@echo "Testing GPU acceleration..."
	poetry run python scripts/gpu_test.py

benchmark:
	@echo "Running performance benchmarks..."
	poetry run python scripts/benchmark.py

# CI/CD Simulation
ci-check: install lint type-check security test-cov
	@echo "CI pipeline simulation complete!"

# Development Environment
setup-dev: install
	@echo "Setting up development environment..."
	cp .env.template .env
	poetry run pre-commit install
	@echo "Development environment ready!"

# Deployment
deploy-prep: clean check-all build
	@echo "Deployment preparation complete!"

# Docker
docker-build:
	@echo "Building Docker image..."
	docker build -t kimera-swm:latest .

docker-run:
	@echo "Running Docker container..."
	docker run -p 8000:8000 kimera-swm:latest

docker-dev:
	@echo "Running development Docker setup..."
	docker-compose -f config/docker/docker-compose.dev.yml up

# Monitoring
logs:
	@echo "Showing recent logs..."
	tail -f data/logs/*.log

monitor:
	@echo "Starting monitoring dashboard..."
	poetry run python scripts/monitor.py

# Utilities
count-lines:
	@echo "Counting lines of code..."
	find src/ -name "*.py" | xargs wc -l | tail -1

project-stats:
	@echo "Project statistics..."
	@echo "Python files: $$(find src/ -name '*.py' | wc -l)"
	@echo "Test files: $$(find tests/ -name '*.py' | wc -l)"
	@echo "Total lines: $$(find src/ -name '*.py' | xargs wc -l | tail -1)" 