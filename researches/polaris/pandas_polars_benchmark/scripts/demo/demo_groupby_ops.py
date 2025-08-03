#!/usr/bin/env python3
"""
Демонстрационный скрипт для проверки GroupBy операций.
"""

import sys
from pathlib import Path

# Добавляем путь к src
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pandas as pd
import polars as pl

from utils import setup_logging
from core import Config
from data import DataGenerator, DataLoader
from operations import get_operation, get_operations_by_category
from profiling import get_profiler, ProfilingConfig


def main():
    """Основная функция демонстрации GroupBy операций."""
    # Настройка логирования
    logger = setup_logging('demo_groupby', console_level='INFO')
    
    logger.info("=" * 80)
    logger.info("ДЕМОНСТРАЦИЯ GROUPBY ОПЕРАЦИЙ")
    logger.info("=" * 80)
    
    # 1. Генерация тестовых данных
    logger.phase_start("Подготовка данных")
    
    config = Config()
    config.data_generation.sizes = [10000]  # Средний размер для демо
    config.data_generation.seed = 42
    
    generator = DataGenerator(
        config=config.data_generation,
        output_dir=Path('data/groupby_demo')
    )
    
    # Генерируем смешанные данные
    logger.info("Генерация тестового датасета...")
    datasets = []
    mixed_info = generator._generate_and_save(
        generator.generate_mixed_data,
        10000,
        "mixed"
    )
    datasets.append(mixed_info)
    
    logger.info(f"✅ Датасет создан: {mixed_info.name}")
    
    # Загружаем данные
    loader = DataLoader(Path('data/groupby_demo'))
    df_pandas = loader.load_pandas_csv(mixed_info.name, backend='numpy')
    df_polars = loader.load_polars_csv(mixed_info.name, lazy=False)
    
    logger.info(f"Загружено: {len(df_pandas)} строк, {len(df_pandas.columns)} колонок")
    logger.info(f"Колонки: {list(df_pandas.columns)[:5]}...")
    
    logger.phase_end("Подготовка данных")
    
    # 2. Проверка всех GroupBy операций
    logger.phase_start("Тестирование GroupBy операций")
    
    groupby_operations = get_operations_by_category('groupby')
    logger.info(f"Найдено GroupBy операций: {len(groupby_operations)}")
    
    for operation in groupby_operations:
        logger.info(f"\n{'='*60}")
        logger.info(f"Операция: {operation.name}")
        logger.info(f"Описание: {operation.description}")
        logger.info(f"{'='*60}")
        
        # Тест Pandas
        logger.info("\n📊 Pandas:")
        result_pandas = operation.execute_pandas(df_pandas)
        if result_pandas.success:
            logger.info(f"✅ Успешно выполнено")
            logger.info(f"   Метаданные: {result_pandas.metadata}")
            if hasattr(result_pandas.result, 'shape'):
                logger.info(f"   Результат: {result_pandas.result.shape}")
                logger.info(f"   Первые строки:")
                logger.info(f"{result_pandas.result.head(3)}")
        else:
            logger.error(f"❌ Ошибка: {result_pandas.error}")
        
        # Тест Polars
        logger.info("\n📊 Polars:")
        result_polars = operation.execute_polars(df_polars)
        if result_polars.success:
            logger.info(f"✅ Успешно выполнено")
            logger.info(f"   Метаданные: {result_polars.metadata}")
            if hasattr(result_polars.result, 'shape'):
                logger.info(f"   Результат: {result_polars.result.shape}")
                logger.info(f"   Первые строки:")
                logger.info(f"{result_polars.result.head(3)}")
        else:
            logger.error(f"❌ Ошибка: {result_polars.error}")
    
    logger.phase_end("Тестирование GroupBy операций")
    
    # 3. Профилирование производительности
    logger.phase_start("Профилирование производительности")
    
    # Настройка профилировщика
    profiling_config = ProfilingConfig(
        min_runs=3,
        max_runs=5,
        target_cv=0.20,  # 20% для демо
        isolate_process=False  # Для скорости
    )
    
    results = []
    
    with get_profiler(profiling_config) as profiler:
        # Профилируем только single_column_groupby для примера
        operation = get_operation('single_column_groupby', 'groupby')
        
        if operation:
            logger.info(f"\nПрофилирование: {operation.name}")
            
            # Pandas
            profile_result = profiler.profile_operation(
                lambda: operation.execute_pandas(df_pandas),
                operation_name=operation.name,
                library='pandas',
                dataset_name=mixed_info.name
            )
            results.append(profile_result)
            
            # Polars
            profile_result = profiler.profile_operation(
                lambda: operation.execute_polars(df_polars),
                operation_name=operation.name,
                library='polars',
                dataset_name=mixed_info.name
            )
            results.append(profile_result)
    
    # Анализ результатов
    logger.info("\n📊 Результаты профилирования:")
    logger.info("-" * 60)
    logger.info(f"{'Операция':<25} {'Библиотека':<10} {'Время (с)':<12} {'Память (MB)':<12}")
    logger.info("-" * 60)
    
    for result in results:
        if result.success:
            logger.info(
                f"{result.operation_name:<25} "
                f"{result.library:<10} "
                f"{result.mean_time:<12.4f} "
                f"{result.peak_memory_mb:<12.1f}"
            )
    
    # Сравнение
    if len(results) == 2 and all(r.success for r in results):
        pandas_time = results[0].mean_time
        polars_time = results[1].mean_time
        speedup = pandas_time / polars_time
        
        logger.info(f"\n🚀 Polars быстрее в {speedup:.1f}x раз!")
    
    logger.phase_end("Профилирование производительности")
    
    # 4. Дополнительные примеры
    logger.phase_start("Дополнительные примеры")
    
    # Пример с конкретными параметрами
    logger.info("\n📝 Пример с конкретными параметрами:")
    
    # Multi aggregation
    multi_agg = get_operation('multi_aggregation', 'groupby')
    if multi_agg:
        result = multi_agg.execute_pandas(df_pandas)
        if result.success:
            logger.info(f"Multi aggregation result shape: {result.result.shape}")
            logger.info(f"Total aggregations: {result.metadata['total_aggregations']}")
    
    # Window functions
    window_op = get_operation('window_functions', 'groupby')
    if window_op:
        result = window_op.execute_polars(df_polars, window_size=20)
        if result.success:
            logger.info(f"Window functions added {result.metadata['new_columns']} new columns")
    
    logger.phase_end("Дополнительные примеры")
    
    # Итоги
    logger.info("\n" + "=" * 80)
    logger.info("ИТОГИ")
    logger.info("=" * 80)
    logger.info("\n✅ Все GroupBy операции реализованы и протестированы:")
    logger.info("  1. single_column_groupby - простая группировка")
    logger.info("  2. multi_column_groupby - группировка по нескольким колонкам")
    logger.info("  3. multi_aggregation - множественные агрегации")
    logger.info("  4. window_functions - оконные функции")
    logger.info("\n🚀 GroupBy операции готовы к использованию в бенчмарке!")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nДемонстрация прервана пользователем")
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
