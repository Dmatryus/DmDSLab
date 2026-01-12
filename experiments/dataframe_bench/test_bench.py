"""Тестовый скрипт для проверки бенчмарка на 100MB данных."""

import logging
import os
import sys
import tempfile
from pathlib import Path

# Добавляем путь к synthdator
sys.path.insert(0, str(Path(__file__).parent.parent / "synthdator"))

from generator import GeneratorConfig, PipelineFactory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def generate_test_data(output_dir: Path) -> Path:
    """Генерирует тестовые данные 100MB."""
    logger.info("Generating 100MB test data...")

    config = GeneratorConfig(
        target_size="100MB",
        output_path=str(output_dir),
        seed=42,
        n_numeric=15,
        informative_ratio=0.6,
        n_categories=3,
        category_cardinality=10,
        task="regression",
        target_method="friedman1",
        target_noise=0.1,
        datetime_range=("2020-01-01", "2024-01-01"),
        with_date=True,
        n_booleans=2,
        nullable_ratio=0.02,
    )

    factory = PipelineFactory()
    pipeline = factory.create(config)
    meta = pipeline.run()

    file_size_mb = os.path.getsize(meta.file_path) / (1024 * 1024)
    logger.info(f"Generated: {meta.file_path}")
    logger.info(f"Rows: {meta.row_count:,}")
    logger.info(f"Columns: {len(meta.columns)}")
    logger.info(f"Size: {file_size_mb:.1f} MB")

    return Path(meta.file_path)


def run_quick_test(data_path: Path) -> None:
    """Запускает быстрый тест нескольких бенчмарков."""
    from polars_duckdb_bench import (
        BenchmarkConfig,
        BenchmarkRunner,
        ALL_BENCHMARKS,
        ALL_ML_PIPELINES,
    )

    config = BenchmarkConfig(
        n_runs=1,
        cold_runs=1,
        timeout_seconds=120,
    )

    runner = BenchmarkRunner(config=config, output_dir=data_path.parent / "results")

    # Тестируем несколько бенчмарков
    test_benchmarks = [
        "io_read_parquet",
        "filter_simple",
        "agg_single",
        "join_inner",
        "sort_single",
    ]

    logger.info("\n" + "=" * 60)
    logger.info("Running quick benchmark test")
    logger.info("=" * 60)

    benchmark_results = runner.run_all_benchmarks(
        data_path,
        data_size="100MB",
        benchmark_filter=test_benchmarks,
    )

    # Тестируем один ML пайплайн
    logger.info("\n" + "=" * 60)
    logger.info("Running ML pipeline test")
    logger.info("=" * 60)

    ml_results = runner.run_ml_pipeline(
        "regression",
        data_path,
        models=["ridge"],
        data_size="100MB",
    )

    # Сохраняем результаты
    output_path = runner.save_results(benchmark_results, ml_results, "test_results.json")

    # Итоги
    logger.info("\n" + "=" * 60)
    logger.info("Test Summary")
    logger.info("=" * 60)

    successful = [r for r in benchmark_results if r.success]
    failed = [r for r in benchmark_results if not r.success]
    logger.info(f"Benchmarks: {len(successful)} OK, {len(failed)} FAILED")

    if failed:
        for r in failed:
            logger.error(f"  FAILED: {r.operation}/{r.mode}: {r.error_message}")

    ml_ok = [r for r in ml_results if r.success]
    ml_fail = [r for r in ml_results if not r.success]
    logger.info(f"ML Pipelines: {len(ml_ok)} OK, {len(ml_fail)} FAILED")

    logger.info(f"\nResults saved to: {output_path}")


def main():
    """Главная функция."""
    # Создаём временную директорию для данных
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Генерируем данные
        data_path = generate_test_data(tmpdir)

        # Запускаем тесты
        run_quick_test(data_path)


if __name__ == "__main__":
    main()
