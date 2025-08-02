#!/usr/bin/env python3
"""
Скрипт для создания структуры директорий проекта бенчмаркинга Pandas vs Polars.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict


def create_project_structure() -> bool:
    """
    Создает полную структуру директорий для проекта бенчмаркинга.

    Returns:
        bool: True если структура создана успешно, False в случае ошибки
    """
    # Определяем базовую директорию проекта
    base_dir = Path.cwd() / "pandas_polars_benchmark"

    # Структура проекта
    project_structure = {
        "src": {
            "core": [
                "__init__.py",
                "config.py",
                "runner.py",
                "checkpoint.py",
                "progress.py",
            ],
            "data": ["__init__.py", "generator.py", "loaders.py", "savers.py"],
            "profiling": [
                "__init__.py",
                "profiler.py",
                "memory_tracker.py",
                "timer.py",
            ],
            "operations": [
                "__init__.py",
                "base.py",
                "io_ops.py",
                "filter_ops.py",
                "groupby_ops.py",
                "sort_ops.py",
                "join_ops.py",
                "string_ops.py",
            ],
            "analysis": [
                "__init__.py",
                "statistical.py",
                "outliers.py",
                "comparisons.py",
            ],
            "reporting": [
                "__init__.py",
                "generator.py",
                "visualizations.py",
                "templates/",
            ],
            "utils": ["__init__.py", "logging.py", "validators.py", "helpers.py"],
        },
        "tests": {
            "unit": [
                "test_config.py",
                "test_generator.py",
                "test_profiler.py",
                "test_operations.py",
                "test_analysis.py",
            ],
            "integration": ["test_workflow.py", "test_checkpoints.py"],
            "fixtures": ["sample_config.yaml", "test_data/"],
        },
        "configs": ["default_config.yaml", "config_schema.yaml"],
        "data": {"generated": [], "metadata": []},
        "results": {"raw": [], "processed": [], "checkpoints": []},
        "reports": {"html": [], "assets": ["css/", "js/", "images/"]},
        "logs": [],
        "docs": ["README.md", "ARCHITECTURE.md", "USER_GUIDE.md"],
        "scripts": ["run_benchmark.py", "validate_config.py", "clean_data.py"],
    }

    try:
        # Создаем базовую директорию
        base_dir.mkdir(exist_ok=True)
        print(f"✓ Создана базовая директория: {base_dir}")

        # Рекурсивная функция для создания структуры
        def create_structure(current_path: Path, structure: Dict) -> None:
            for name, content in structure.items():
                path = current_path / name

                if isinstance(content, dict):
                    # Это поддиректория
                    path.mkdir(exist_ok=True)
                    print(f"  📁 {path.relative_to(base_dir)}")
                    create_structure(path, content)
                elif isinstance(content, list):
                    # Это директория с файлами
                    path.mkdir(exist_ok=True)
                    print(f"  📁 {path.relative_to(base_dir)}")

                    for file_name in content:
                        if file_name.endswith("/"):
                            # Это поддиректория
                            subdir = path / file_name.rstrip("/")
                            subdir.mkdir(exist_ok=True)
                            print(f"    📁 {subdir.relative_to(base_dir)}")
                        else:
                            # Это файл
                            file_path = path / file_name
                            if not file_path.exists():
                                file_path.touch()
                                print(f"    📄 {file_path.relative_to(base_dir)}")

        # Создаем структуру
        create_structure(base_dir, project_structure)

        # Создаем дополнительные файлы в корне проекта
        root_files = [
            ".gitignore",
            "requirements.txt",
            "requirements-dev.txt",
            "setup.py",
            "README.md",
            "LICENSE",
            ".env.example",
        ]

        print("\n📁 Корневые файлы:")
        for file_name in root_files:
            file_path = base_dir / file_name
            if not file_path.exists():
                file_path.touch()
                print(f"  📄 {file_name}")

        print(f"\n✅ Структура проекта успешно создана в: {base_dir}")
        return True

    except Exception as e:
        print(f"\n❌ Ошибка при создании структуры: {e}", file=sys.stderr)
        return False


def create_gitignore_content() -> str:
    """
    Создает содержимое для .gitignore файла.

    Returns:
        str: Содержимое .gitignore
    """
    return """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
.venv
pip-log.txt
pip-delete-this-directory.txt
.pytest_cache/
.coverage
htmlcov/
.tox/
.mypy_cache/
.dmypy.json
dmypy.json

# Project specific
data/generated/
results/raw/
results/checkpoints/
logs/*.log
reports/html/*.html
*.parquet
*.csv
!tests/fixtures/*.csv

# IDE
.idea/
.vscode/
*.swp
*.swo
*~
.DS_Store

# Environment
.env
.env.local

# Jupyter
.ipynb_checkpoints/
*.ipynb

# Distribution
dist/
build/
*.egg-info/
"""


def initialize_project_files() -> None:
    """
    Инициализирует основные файлы проекта с базовым содержимым.
    """
    base_dir = Path.cwd() / "pandas_polars_benchmark"

    # Создаем .gitignore
    gitignore_path = base_dir / ".gitignore"
    gitignore_path.write_text(create_gitignore_content(), encoding="utf-8")
    print(f"✓ Создан .gitignore")

    # Создаем базовый requirements.txt
    requirements = """# Core dependencies
pandas>=2.0.0
polars>=0.19.0
numpy>=1.24.0
pyyaml>=6.0
jsonschema>=4.17.0
psutil>=5.9.0
click>=8.1.0

# Analysis and reporting
scipy>=1.10.0
plotly>=5.14.0
jinja2>=3.1.0
tabulate>=0.9.0

# Development
pytest>=7.3.0
pytest-cov>=4.0.0
black>=23.3.0
flake8>=6.0.0
mypy>=1.3.0
pre-commit>=3.3.0

# Logging
colorlog>=6.7.0
tqdm>=4.65.0
"""

    requirements_path = base_dir / "requirements.txt"
    requirements_path.write_text(requirements, encoding="utf-8")
    print(f"✓ Создан requirements.txt")

    # Создаем базовый README.md
    readme_content = """# Pandas vs Polars Benchmark System

Comprehensive benchmarking system for comparing performance between Pandas and Polars libraries.

## Features

- Automated data generation with various types and sizes
- Isolated profiling with memory tracking
- Statistical analysis with outlier detection
- Interactive HTML reports with visualizations
- Checkpoint system for resumable runs
- Configurable operations and parameters

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```bash
# Run benchmark with default configuration
python scripts/run_benchmark.py --config configs/default_config.yaml

# Resume from checkpoint
python scripts/run_benchmark.py --resume

# Validate configuration only
python scripts/validate_config.py --config configs/default_config.yaml
```

## Project Structure

```
pandas_polars_benchmark/
├── src/               # Source code
├── tests/             # Test suite
├── configs/           # Configuration files
├── data/              # Generated datasets
├── results/           # Benchmark results
├── reports/           # Generated reports
├── logs/              # Execution logs
└── scripts/           # Executable scripts
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [User Guide](docs/USER_GUIDE.md)

## License

MIT License
"""

    readme_path = base_dir / "README.md"
    readme_path.write_text(readme_content, encoding="utf-8")
    print(f"✓ Создан README.md")


if __name__ == "__main__":
    print("🚀 Создание структуры проекта бенчмаркинга Pandas vs Polars\n")

    if create_project_structure():
        print("\n📝 Инициализация файлов проекта...")
        initialize_project_files()
        print("\n✨ Проект готов к разработке!")
        print("\nСледующие шаги:")
        print("1. cd pandas_polars_benchmark")
        print("2. python -m venv venv")
        print("3. source venv/bin/activate  # На Windows: venv\\Scripts\\activate")
        print("4. pip install -r requirements.txt")
    else:
        sys.exit(1)
