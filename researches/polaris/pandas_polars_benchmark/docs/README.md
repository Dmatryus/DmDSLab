# Pandas vs Polars Benchmark System

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
source venv/bin/activate  # On Windows: venv\Scripts\activate

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
