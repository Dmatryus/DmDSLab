"""Demo script для генерации большого датасета (10GB)."""

import logging
import os
import time

import duckdb

from generator import GeneratorConfig, PipelineFactory

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def demo_10gb():
    """Генерация датасета 10GB."""
    logger.info("=" * 60)
    logger.info("Demo: 10GB dataset generation")
    logger.info("=" * 60)

    output_dir = "./output_10gb"
    os.makedirs(output_dir, exist_ok=True)

    config = GeneratorConfig(
        target_size="10GB",
        output_path=output_dir,
        seed=42,
        n_numeric=50,
        informative_ratio=0.6,
        n_categories=5,
        category_cardinality=20,
        task="regression",
        target_method="friedman1",
        target_noise=0.1,
        datetime_range=("2020-01-01", "2024-01-01"),
        with_date=True,
        n_booleans=3,
        nullable_ratio=0.02,
    )

    factory = PipelineFactory()
    pipeline = factory.create(config)

    logger.info("Steps: %s", [type(s).__name__ for s in pipeline.steps])
    logger.info("Starting generation...")

    start_time = time.time()
    meta = pipeline.run()
    elapsed = time.time() - start_time

    file_size_gb = os.path.getsize(meta.file_path) / (1024**3)

    logger.info("=" * 60)
    logger.info("Generation completed!")
    logger.info("Rows: %d", meta.row_count)
    logger.info("Columns: %d", len(meta.columns))
    logger.info("File size: %.2f GB", file_size_gb)
    logger.info("Time: %.1f seconds (%.1f minutes)", elapsed, elapsed / 60)
    logger.info("Throughput: %.2f GB/min", file_size_gb / (elapsed / 60))
    logger.info("File: %s", meta.file_path)
    logger.info("=" * 60)

    # Проверяем sample
    with duckdb.connect() as db:
        sample = db.execute(f"SELECT * FROM '{meta.file_path}' LIMIT 5").fetchdf()
        logger.info("Sample:\n%s", sample.to_string())

        # Статистика по колонкам
        null_stats = db.execute(
            f"""
            SELECT
                COUNT(*) as total_rows,
                SUM(CASE WHEN numeric_0 IS NULL THEN 1 ELSE 0 END) as numeric_0_nulls,
                COUNT(DISTINCT cat_category_0) as cat_0_unique
            FROM '{meta.file_path}'
        """
        ).fetchdf()
        logger.info("Stats:\n%s", null_stats)


if __name__ == "__main__":
    demo_10gb()
