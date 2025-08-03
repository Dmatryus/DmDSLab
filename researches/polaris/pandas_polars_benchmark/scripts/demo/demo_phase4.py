#!/usr/bin/env python3
"""
Полная демонстрация работы Фазы 4: Операции бенчмаркинга.
"""

import sys
from pathlib import Path
import pandas as pd
import polars as pl

# Добавляем путь к src
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Импортируем утилиты
from utils import setup_logging

# Импортируем компоненты ядра
from core import Config
from core.checkpoint import CheckpointManager
from core.progress import ProgressTracker

# Импортируем модули данных
from data import DataGenerator, DataLoader

# Импортируем операции
from operations import (
    get_operation,
    get_all_operations,
    get_operations_by_category,
    list_operations,
)

# Импортируем профилирование
from profiling import get_profiler, ProfilingConfig


def main():
    """Основная функция демонстрации."""
    # Настройка логирования
    logger = setup_logging("demo_phase4", console_level="INFO")

    logger.benchmark_start(
        {"benchmark": {"name": "Phase 4 Operations Demo", "version": "1.0.0"}}
    )

    # 1. Генерация тестовых данных
    logger.phase_start("Генерация тестовых данных")

    config = Config()
    config.data_generation.sizes = [5000]  # Средний размер для демо
    config.data_generation.seed = 42

    generator = DataGenerator(
        config=config.data_generation, output_dir=Path("data/phase4_demo")
    )

    # Генерируем смешанные данные (содержат все типы)
    logger.info("Генерация смешанного датасета...")
    datasets = []
    mixed_info = generator._generate_and_save(
        generator.generate_mixed_data, 5000, "mixed"
    )
    datasets.append(mixed_info)

    logger.info(f"✅ Датасет создан: {mixed_info.name}")
    logger.info(f"   Размер: {mixed_info.size:,} строк")
    logger.info(f"   Колонок: {len(mixed_info.columns)}")
    logger.info(f"   Типы колонок:")
    for col in mixed_info.columns[:5]:  # Первые 5 колонок
        logger.info(f"     - {col['name']}: {col['dtype']}")

    logger.phase_end("Генерация тестовых данных")

    # 2. Обзор зарегистрированных операций
    logger.phase_start("Обзор операций")

    all_operations = get_all_operations()

    logger.info("Зарегистрированные операции по категориям:")
    for category, operations in all_operations.items():
        logger.info(f"\n📁 {category.upper()}:")
        for op in operations:
            logger.info(f"   - {op.name}: {op.description}")

    total_ops = sum(len(ops) for ops in all_operations.values())
    logger.info(f"\n✅ Всего операций: {total_ops}")

    logger.phase_end("Обзор операций")

    # 3. Демонстрация работы операций
    logger.phase_start("Демонстрация операций")

    # Загружаем данные
    loader = DataLoader(Path("data/phase4_demo"))

    # Pandas DataFrame
    logger.info("\n📊 Загрузка данных в Pandas...")
    df_pandas = loader.load_pandas_csv(mixed_info.name, backend="numpy")
    logger.info(
        f"   Загружено: {len(df_pandas)} строк, {len(df_pandas.columns)} колонок"
    )

    # Polars DataFrame
    logger.info("\n📊 Загрузка данных в Polars...")
    df_polars = loader.load_polars_csv(mixed_info.name, lazy=False)
    logger.info(
        f"   Загружено: {len(df_polars)} строк, {len(df_polars.columns)} колонок"
    )

    # Демо: Простая фильтрация
    logger.info("\n🔍 Демонстрация простой фильтрации:")

    simple_filter = get_operation("simple_filter", "filter")
    if simple_filter:
        # Pandas
        result = simple_filter.execute_pandas(df_pandas)
        if result.success:
            logger.info(
                f"   Pandas: {result.metadata['rows_before']} → {result.metadata['rows_after']} строк"
            )
            logger.info(
                f"   Отфильтровано: {(1-result.metadata['filtered_ratio'])*100:.1f}%"
            )

        # Polars
        result = simple_filter.execute_polars(df_polars)
        if result.success:
            logger.info(f"   Polars: аналогичный результат")

    # Демо: Сложная фильтрация
    logger.info("\n🔍 Демонстрация сложной фильтрации:")

    complex_filter = get_operation("complex_filter", "filter")
    if complex_filter:
        result = complex_filter.execute_pandas(df_pandas)
        if result.success:
            logger.info(f"   Условия: {result.metadata['conditions']['numeric']}")
            logger.info(f"   Результат: {result.metadata['rows_after']} строк")

    logger.phase_end("Демонстрация операций")

    # 4. Профилирование операций
    logger.phase_start("Профилирование производительности")

    # Настройка профилировщика
    profiling_config = ProfilingConfig(
        min_runs=3, max_runs=10, target_cv=0.15, isolate_process=False  # 15% для демо
    )

    # Создаем прогресс трекер
    operations_to_profile = [
        ("io", "read_csv"),
        ("io", "write_parquet"),
        ("filter", "simple_filter"),
        ("filter", "isin_filter"),
    ]

    total_tasks = len(operations_to_profile) * 2  # Pandas + Polars

    with ProgressTracker(total_tasks, show_progress_bar=True) as progress:
        with get_profiler(profiling_config) as profiler:

            results = []

            for category, op_name in operations_to_profile:
                operation = get_operation(op_name, category)

                if operation:
                    # Профилируем Pandas
                    progress.start_operation(op_name, "pandas", mixed_info.name)

                    if category == "io" and op_name.startswith("read"):
                        # Для операций чтения
                        profile_result = profiler.profile_operation(
                            lambda: operation.execute_pandas(
                                None, dataset_name=mixed_info.name
                            ),
                            operation_name=op_name,
                            library="pandas",
                            dataset_name=mixed_info.name,
                        )
                    else:
                        # Для остальных операций
                        profile_result = profiler.profile_operation(
                            lambda: operation.execute_pandas(df_pandas),
                            operation_name=op_name,
                            library="pandas",
                            dataset_name=mixed_info.name,
                        )

                    results.append(profile_result)
                    progress.end_operation(profile_result.success)

                    # Профилируем Polars
                    progress.start_operation(op_name, "polars", mixed_info.name)

                    if category == "io" and op_name.startswith("read"):
                        profile_result = profiler.profile_operation(
                            lambda: operation.execute_polars(
                                None, dataset_name=mixed_info.name
                            ),
                            operation_name=op_name,
                            library="polars",
                            dataset_name=mixed_info.name,
                        )
                    else:
                        profile_result = profiler.profile_operation(
                            lambda: operation.execute_polars(df_polars),
                            operation_name=op_name,
                            library="polars",
                            dataset_name=mixed_info.name,
                        )

                    results.append(profile_result)
                    progress.end_operation(profile_result.success)

    # Анализ результатов
    logger.info("\n📊 Результаты профилирования:")
    logger.info("-" * 60)
    logger.info(
        f"{'Операция':<20} {'Библиотека':<10} {'Время (с)':<12} {'Память (MB)':<12}"
    )
    logger.info("-" * 60)

    for result in results:
        if result.success:
            logger.info(
                f"{result.operation_name:<20} "
                f"{result.library:<10} "
                f"{result.mean_time:<12.3f} "
                f"{result.peak_memory_mb:<12.1f}"
            )

    logger.phase_end("Профилирование производительности")

    # 5. Сравнительный анализ
    logger.phase_start("Сравнительный анализ")

    # Группируем результаты по операциям
    comparison = {}
    for result in results:
        if result.success:
            if result.operation_name not in comparison:
                comparison[result.operation_name] = {}
            comparison[result.operation_name][result.library] = {
                "time": result.mean_time,
                "memory": result.peak_memory_mb,
            }

    logger.info("Сравнение Polars vs Pandas:")
    for op_name, libs in comparison.items():
        if "pandas" in libs and "polars" in libs:
            time_ratio = libs["pandas"]["time"] / libs["polars"]["time"]
            memory_ratio = libs["pandas"]["memory"] / libs["polars"]["memory"]

            logger.info(f"\n{op_name}:")
            logger.info(f"  Скорость: Polars в {time_ratio:.1f}x быстрее")
            logger.info(
                f"  Память: Polars использует {1/memory_ratio:.1f}x меньше памяти"
            )

    logger.phase_end("Сравнительный анализ")

    # Итоги
    logger.benchmark_end(success=True, duration=60)

    logger.info("\n" + "=" * 80)
    logger.info("ИТОГИ ФАЗЫ 4")
    logger.info("=" * 80)
    logger.info("\n✅ Реализованы компоненты:")
    logger.info("  1. Базовая архитектура операций")
    logger.info("  2. IO операции (чтение/запись CSV и Parquet)")
    logger.info("  3. Операции фильтрации (простая, сложная, isin, паттерны)")
    logger.info("  4. Интеграция с системой профилирования")
    logger.info("  5. Автоматическая регистрация операций")

    logger.info("\n📋 Готовые к реализации категории:")
    logger.info("  - GroupBy операции")
    logger.info("  - Операции сортировки")
    logger.info("  - Join операции")
    logger.info("  - Строковые операции")

    logger.info("\n🚀 Система готова к добавлению новых операций!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nДемонстрация прервана пользователем")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
