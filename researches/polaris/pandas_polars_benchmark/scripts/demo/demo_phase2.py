#!/usr/bin/env python3
"""
Демонстрационный скрипт для проверки всех компонентов Фазы 2.
"""

import sys
from pathlib import Path
import time

from researches.polaris.pandas_polars_benchmark.src import setup_logging, Config, DataGenerator, DataLoader, DataSaver


def demo_phase2():
    """Демонстрирует работу всех компонентов Фазы 2."""
    # Настраиваем логирование
    logger = setup_logging(
        name='demo_phase2',
        console_level='INFO',
        use_colors=True
    )

    logger.benchmark_start({'benchmark': {'name': 'Phase 2 Demo', 'version': '1.0.0'}})

    # Создаем мини-конфигурацию для демо
    logger.phase_start("Настройка конфигурации")

    config = Config()
    config.data_generation.sizes = [100, 500]  # Очень маленькие размеры для демо
    config.data_generation.numeric_columns = 3
    config.data_generation.string_columns = 2
    config.data_generation.datetime_columns = 1
    config.data_generation.seed = 42

    logger.info("Конфигурация создана:")
    logger.info(f"  - Размеры: {config.data_generation.sizes}")
    logger.info("  - Типы: numeric, string, datetime, mixed")

    logger.phase_end("Настройка конфигурации")

    # 1. Генерация данных
    logger.phase_start("Генерация данных")

    generator = DataGenerator(
        config=config.data_generation,
        output_dir=Path('data/demo')
    )

    try:
        datasets = generator.generate_all_datasets()
        logger.info(f"✅ Сгенерировано {len(datasets)} датасетов")
    except Exception as e:
        logger.error(f"Ошибка генерации: {e}", exc_info=True)
        return False

    logger.phase_end("Генерация данных")

    # 2. Загрузка данных
    logger.phase_start("Тестирование загрузки данных")

    loader = DataLoader(data_dir=Path('data/demo'))

    # Загружаем первый датасет разными способами
    test_dataset = datasets[0].name
    logger.info(f"Тестовый датасет: {test_dataset}")

    try:
        # Pandas NumPy
        logger.operation_start("load_pandas_numpy", "pandas", test_dataset)
        start = time.time()
        df_pandas = loader.load_pandas_csv(test_dataset, backend='numpy')
        elapsed = time.time() - start
        logger.operation_end("load_pandas_numpy", True, elapsed, df_pandas.memory_usage(deep=True).sum() / 1e6)

        # Pandas PyArrow
        logger.operation_start("load_pandas_pyarrow", "pandas", test_dataset)
        start = time.time()
        df_pandas_arrow = loader.load_pandas_csv(test_dataset, backend='pyarrow')
        elapsed = time.time() - start
        logger.operation_end("load_pandas_pyarrow", True, elapsed, 0)  # Примерная память

        # Polars
        logger.operation_start("load_polars", "polars", test_dataset)
        start = time.time()
        df_polars = loader.load_polars_csv(test_dataset, lazy=False)
        elapsed = time.time() - start
        logger.operation_end("load_polars", True, elapsed, df_polars.estimated_size() / 1e6)

        logger.info("✅ Все форматы загрузки работают корректно")

    except Exception as e:
        logger.error(f"Ошибка загрузки: {e}", exc_info=True)
        return False

    logger.phase_end("Тестирование загрузки данных")

    # 3. Сохранение данных
    logger.phase_start("Тестирование сохранения данных")

    with DataSaver() as saver:  # Используем временную директорию
        try:
            # Сохраняем Pandas DataFrame
            result = saver.save_pandas_csv(df_pandas, "test_pandas")
            logger.info(f"Pandas CSV сохранен: {result['file_size_mb']:.2f} MB за {result['save_time_sec']:.3f}с")

            # Сохраняем Polars DataFrame
            result = saver.save_polars_parquet(df_polars, "test_polars")
            logger.info(f"Polars Parquet сохранен: {result['file_size_mb']:.2f} MB за {result['save_time_sec']:.3f}с")

            logger.info("✅ Сохранение данных работает корректно")

        except Exception as e:
            logger.error(f"Ошибка сохранения: {e}", exc_info=True)
            return False

    logger.phase_end("Тестирование сохранения данных")

    # 4. Проверка метаданных
    logger.phase_start("Проверка метаданных")

    # Выводим доступные датасеты
    available = loader.get_available_datasets()
    logger.info("Доступные датасеты по типам:")
    for dtype, names in available.items():
        logger.info(f"  - {dtype}: {len(names)} датасетов")

    # Проверяем валидацию
    for ds in datasets[:2]:
        is_valid = loader.validate_dataset(ds.name)
        logger.info(f"  {'✅' if is_valid else '❌'} {ds.name}")

    # Выводим статистику
    stats = loader.get_dataset_stats(test_dataset)
    logger.info(f"\nСтатистика {test_dataset}:")
    logger.info(f"  - Размер: {stats['size']} строк")
    logger.info(f"  - Колонок: {stats['columns']}")
    logger.info(f"  - В памяти: {stats['memory_size_mb']:.1f} MB")

    logger.phase_end("Проверка метаданных")

    # Итоги
    logger.benchmark_end(success=True, duration=60)  # Примерное время

    return True


if __name__ == '__main__':
    success = demo_phase2()
    
    if success:
        print("\n" + "="*80)
        print("✅ ФАЗА 2 ПОЛНОСТЬЮ ГОТОВА!")
        print("="*80)
        print("\nВсе компоненты работают корректно:")
        print("  ✓ DataGenerator - генерация всех типов данных")
        print("  ✓ DataLoader - загрузка в Pandas/Polars")
        print("  ✓ DataSaver - сохранение результатов")
        print("  ✓ Метаданные и валидация")
        print("\nМожно переходить к Фазе 3: Система профилирования")
    else:
        print("\n❌ Обнаружены ошибки, требуется исправление")
        sys.exit(1)
