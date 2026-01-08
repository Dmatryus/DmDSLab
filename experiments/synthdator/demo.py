"""Demo script для тестирования Synthdator."""

import logging
import os
import tempfile

import duckdb

from generator import GeneratorConfig, PipelineFactory

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def demo_regression():
    """Демо: регрессия с числовыми фичами."""
    logger.info("=" * 60)
    logger.info("Demo: Regression task")
    logger.info("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        config = GeneratorConfig(
            target_size="1MB",
            output_path=tmpdir,
            seed=42,
            n_numeric=10,
            informative_ratio=0.5,
            task="regression",
            target_method="friedman1",
            target_noise=0.1,
        )

        factory = PipelineFactory()
        pipeline = factory.create(config)

        logger.info("Steps: %s", [type(s).__name__ for s in pipeline.steps])

        meta = pipeline.run()

        logger.info("Rows: %d", meta.row_count)
        logger.info("Columns: %s", list(meta.columns.keys()))
        logger.info("File size: %.1f KB", os.path.getsize(meta.file_path) / 1024)

        # Проверяем данные
        with duckdb.connect() as db:
            sample = db.execute(f"SELECT * FROM '{meta.file_path}' LIMIT 5").fetchdf()
            logger.info("Sample:\n%s", sample)


def demo_binary_classification():
    """Демо: бинарная классификация с категориями."""
    logger.info("\n" + "=" * 60)
    logger.info("Demo: Binary classification with categories")
    logger.info("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        config = GeneratorConfig(
            target_size="500KB",
            output_path=tmpdir,
            seed=123,
            n_numeric=5,
            n_categories=2,
            category_cardinality=5,
            task="binary",
            target_method="circles",
        )

        factory = PipelineFactory()
        pipeline = factory.create(config)

        logger.info("Steps: %s", [type(s).__name__ for s in pipeline.steps])

        meta = pipeline.run()

        logger.info("Rows: %d", meta.row_count)
        logger.info("Columns: %s", list(meta.columns.keys()))
        logger.info("File size: %.1f KB", os.path.getsize(meta.file_path) / 1024)

        # Проверяем распределение классов
        with duckdb.connect() as db:
            dist = db.execute(
                f"SELECT target_bin, COUNT(*) as cnt FROM '{meta.file_path}' GROUP BY 1"
            ).fetchdf()
            logger.info("Class distribution:\n%s", dist)


def demo_multiclass():
    """Демо: многоклассовая классификация."""
    logger.info("\n" + "=" * 60)
    logger.info("Demo: Multiclass classification")
    logger.info("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        config = GeneratorConfig(
            target_size="500KB",
            output_path=tmpdir,
            seed=456,
            n_numeric=8,
            task="multiclass",
            target_method="polynomial",
            n_classes=4,
        )

        factory = PipelineFactory()
        pipeline = factory.create(config)

        logger.info("Steps: %s", [type(s).__name__ for s in pipeline.steps])

        meta = pipeline.run()

        logger.info("Rows: %d", meta.row_count)
        logger.info("Columns: %s", list(meta.columns.keys()))

        # Проверяем распределение классов
        with duckdb.connect() as db:
            dist = db.execute(
                f"SELECT target_multi, COUNT(*) as cnt FROM '{meta.file_path}' GROUP BY 1 ORDER BY 1"
            ).fetchdf()
            logger.info("Class distribution:\n%s", dist)


def demo_ranking():
    """Демо: задача ранжирования."""
    logger.info("\n" + "=" * 60)
    logger.info("Demo: Ranking task")
    logger.info("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        config = GeneratorConfig(
            target_size="500KB",
            output_path=tmpdir,
            seed=789,
            n_numeric=6,
            n_categories=1,
            category_cardinality=10,
            task="ranking",
            target_method="linear",
        )

        factory = PipelineFactory()
        pipeline = factory.create(config)

        logger.info("Steps: %s", [type(s).__name__ for s in pipeline.steps])

        meta = pipeline.run()

        logger.info("Rows: %d", meta.row_count)
        logger.info("Columns: %s", list(meta.columns.keys()))

        # Проверяем распределение рангов
        with duckdb.connect() as db:
            dist = db.execute(
                f"SELECT target_rank, COUNT(*) as cnt FROM '{meta.file_path}' GROUP BY 1 ORDER BY 1"
            ).fetchdf()
            logger.info("Rank distribution:\n%s", dist)


def demo_full_featured():
    """Демо: все фичи включены."""
    logger.info("\n" + "=" * 60)
    logger.info("Demo: Full featured dataset")
    logger.info("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        config = GeneratorConfig(
            target_size="1MB",
            output_path=tmpdir,
            seed=42,
            n_numeric=10,
            informative_ratio=0.6,
            n_categories=2,
            category_cardinality=8,
            task="regression",
            target_method="nonlinear",
            with_datetime=True,
            datetime_start="2022-01-01",
            datetime_end="2024-01-01",
            with_date=True,
            n_booleans=2,
            nullable_ratio=0.05,
        )

        factory = PipelineFactory()
        pipeline = factory.create(config)

        logger.info("Steps: %s", [type(s).__name__ for s in pipeline.steps])

        meta = pipeline.run()

        logger.info("Rows: %d", meta.row_count)
        logger.info("Columns (%d): %s", len(meta.columns), list(meta.columns.keys()))
        logger.info("File size: %.1f KB", os.path.getsize(meta.file_path) / 1024)

        # Проверяем данные
        with duckdb.connect() as db:
            sample = db.execute(f"SELECT * FROM '{meta.file_path}' LIMIT 3").fetchdf()
            logger.info("Sample:\n%s", sample.to_string())

            # Проверяем NULL
            null_counts = db.execute(f"""
                SELECT
                    SUM(CASE WHEN numeric_0 IS NULL THEN 1 ELSE 0 END) as numeric_0_nulls,
                    SUM(CASE WHEN numeric_1 IS NULL THEN 1 ELSE 0 END) as numeric_1_nulls
                FROM '{meta.file_path}'
            """).fetchdf()
            logger.info("NULL counts:\n%s", null_counts)


if __name__ == "__main__":
    demo_regression()
    demo_binary_classification()
    demo_multiclass()
    demo_ranking()
    demo_full_featured()

    logger.info("\n" + "=" * 60)
    logger.info("All demos completed!")
    logger.info("=" * 60)
