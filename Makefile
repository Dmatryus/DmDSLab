# DmDSLab Project Makefile

.PHONY: help install install-dev install-all clean test test-cov lint format type-check build publish-test publish docs setup-dev init-db

# Default target
help:
	@echo "DmDSLab Development Commands"
	@echo "============================"
	@echo ""
	@echo "Setup and Installation:"
	@echo "  setup-dev     Setup development environment"
	@echo "  install       Install package"
	@echo "  install-dev   Install package with dev dependencies"
	@echo "  install-all   Install package with all dependencies"
	@echo ""
	@echo "Development:"
	@echo "  init-db       Initialize UCI database"
	@echo "  test          Run tests"
	@echo "  test-cov      Run tests with coverage report"
	@echo "  lint          Run linting (ruff)"
	@echo "  format        Format code (ruff + black)"
	@echo "  type-check    Run type checking (mypy)"
	@echo "  pre-commit    Run pre-commit hooks"
	@echo ""
	@echo "Build and Release:"
	@echo "  clean         Clean build artifacts"
	@echo "  build         Build package"
	@echo "  publish-test  Publish to TestPyPI"
	@echo "  publish       Publish to PyPI"
	@echo ""
	@echo "Documentation:"
	@echo "  docs          Generate documentation"

# Setup and Installation
setup-dev:
	@echo "🚀 Setting up development environment..."
	pip install --upgrade pip
	pip install -e .[dev,uci]
	pre-commit install
	@echo "✅ Development environment ready!"

install:
	pip install -e .

install-dev:
	pip install -e .[dev]

install-all:
	pip install -e .[all]

# Database
init-db:
	@echo "🗄️ Initializing UCI database..."
	python scripts/initialize_uci_database.py
	@echo "✅ Database initialized!"

# Development
test:
	@echo "🧪 Running tests..."
	pytest tests/ -v

test-cov:
	@echo "🧪 Running tests with coverage..."
	pytest tests/ -v --cov=dmdslab --cov-report=term-missing --cov-report=html

lint:
	@echo "🔍 Running linter..."
	ruff check dmdslab/ tests/

format:
	@echo "✨ Formatting code..."
	ruff check dmdslab/ tests/ --fix
	ruff format dmdslab/ tests/

type-check:
	@echo "🔍 Running type checker..."
	mypy dmdslab/

pre-commit:
	@echo "🔧 Running pre-commit hooks..."
	pre-commit run --all-files

# Build and Release
clean:
	@echo "🧹 Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete
	@echo "✅ Cleaned!"

build: clean
	@echo "📦 Building package..."
	python -m build
	twine check dist/*
	@echo "✅ Package built successfully!"

publish-test: build
	@echo "🚀 Publishing to TestPyPI..."
	@if [ -z "$(TEST_PYPI_TOKEN)" ]; then \
		echo "❌ TEST_PYPI_TOKEN not set"; \
		exit 1; \
	fi
	twine upload --repository testpypi --username __token__ --password $(TEST_PYPI_TOKEN) dist/*
	@echo "✅ Published to TestPyPI!"

publish: build
	@echo "🚀 Publishing to PyPI..."
	@if [ -z "$(PYPI_TOKEN)" ]; then \
		echo "❌ PYPI_TOKEN not set"; \
		exit 1; \
	fi
	twine upload --username __token__ --password $(PYPI_TOKEN) dist/*
	@echo "✅ Published to PyPI!"

# Documentation  
docs:
	@echo "📚 Generating documentation..."
	@echo "Documentation generation not implemented yet"
	@echo "Will be added in future versions"

# Quality checks (runs all quality tools)
check: lint type-check test
	@echo "✅ All quality checks passed!"

# Full development workflow
dev-check: format lint type-check test
	@echo "✅ Development workflow completed!"

# Install build tools
install-build-tools:
	pip install --upgrade pip build twine

# Version management
version:
	@python -c "import dmdslab; print(f'Current version: {dmdslab.__version__}')"

# Show package info
info:
	@echo "📋 Package Information:"
	@echo "Name: DmDSLab"
	@python -c "import dmdslab; print(f'Version: {dmdslab.__version__}')"
	@echo "Author: Dmatryus Detry"
	@echo "License: Apache-2.0"
	@echo "Repository: https://github.com/Dmatryus/DmDSLab"