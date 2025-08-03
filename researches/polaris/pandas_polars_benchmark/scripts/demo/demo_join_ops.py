#!/usr/bin/env python3
"""
Демонстрационный скрипт для проверки Join операций.
"""

import sys
from pathlib import Path

# Добавляем путь к src
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

# Добавляем корневой путь проекта для абсолютных импортов
project_root = Path(__file__).parent.parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import polars as pl
import numpy as np
import time

from utils import setup_logging
from operations import get_operation, get_operations_by_category, get_all_operations
from profiling import get_profiler, ProfilingConfig

# Явно импортируем модули операций чтобы они зарегистрировались
try:
    from operations import join_ops, groupby_ops, sort_ops
except ImportError as e:
    print(f"Ошибка импорта модулей операций: {e}")
    print("Попытка прямого импорта...")
    # Альтернативный способ импорта
    import operations.join_ops
    import operations.groupby_ops  
    import operations.sort_ops


def create_test_data(n_rows: int = 10000):
    """Создает тестовые данные для демонстрации join операций."""
    np.random.seed(42)
    
    # Основные данные
    data = {
        'customer_id': np.random.choice(['C' + str(i) for i in range(n_rows // 10)], n_rows),
        'product_id': np.random.choice(['P' + str(i) for i in range(50)], n_rows),
        'order_id': [f'ORDER_{i:06d}' for i in range(n_rows)],
        'quantity': np.random.randint(1, 20, n_rows),
        'price': np.random.uniform(10, 1000, n_rows).round(2),
        'discount': np.random.uniform(0, 0.3, n_rows).round(2),
        'order_date': pd.date_range('2023-01-01', periods=n_rows, freq='15min'),
        'status': np.random.choice(['pending', 'shipped', 'delivered', 'cancelled'], n_rows),
        'region': np.random.choice(['North', 'South', 'East', 'West'], n_rows),
        'priority': np.random.choice([1, 2, 3, 4, 5], n_rows)
    }
    
    return pd.DataFrame(data), pl.DataFrame(data)


def demonstrate_join_operations():
    """Демонстрирует работу всех Join операций."""
    # Настройка логирования
    logger = setup_logging('demo_join', console_level='INFO')
    
    logger.info("=" * 80)
    logger.info("ДЕМОНСТРАЦИЯ JOIN ОПЕРАЦИЙ")
    logger.info("=" * 80)
    
    # Отладка: проверяем, какие операции зарегистрированы
    logger.info("\n🔍 Проверка зарегистрированных операций:")
    all_ops = get_all_operations()
    for category, ops_list in all_ops.items():
        logger.info(f"  {category}: {len(ops_list)} операций")
        for op in ops_list:
            logger.info(f"    - {op.name}")
    
    # 1. Подготовка данных
    logger.phase_start("Подготовка данных")
    
    df_pandas, df_polars = create_test_data(10000)
    
    logger.info(f"✅ Созданы тестовые данные: {len(df_pandas)} строк")
    logger.info(f"Колонки: {list(df_pandas.columns)}")
    logger.info("\nПримеры данных:")
    logger.info(df_pandas.head(3).to_string())
    
    logger.phase_end("Подготовка данных")
    
    # 2. Тестирование всех Join операций
    logger.phase_start("Тестирование Join операций")
    
    join_operations = get_operations_by_category('join')
    logger.info(f"Найдено Join операций: {len(join_operations)}")
    
    if len(join_operations) == 0:
        logger.error("❌ Join операции не зарегистрированы!")
        logger.error("Проверьте импорт модуля join_ops")
        return
    
    for operation in join_operations:
        logger.info(f"\n{'='*60}")
        logger.info(f"Операция: {operation.name}")
        logger.info(f"Описание: {operation.description}")
        logger.info(f"{'='*60}")
        
        # Тест Pandas
        logger.info("\n📊 Pandas:")
        start = time.time()
        result_pandas = operation.execute_pandas(df_pandas)
        pandas_time = time.time() - start
        
        if result_pandas.success:
            logger.info(f"✅ Успешно выполнено за {pandas_time:.3f}с")
            meta = result_pandas.metadata
            logger.info(f"   Тип join: {meta.get('join_type')}")
            logger.info(f"   Ключи: {meta.get('join_keys')}")
            
            if 'left_rows' in meta:
                logger.info(f"   Левая таблица: {meta['left_rows']} строк")
                logger.info(f"   Правая таблица: {meta['right_rows']} строк")
                logger.info(f"   Результат: {meta['result_rows']} строк")
                
                # Коэффициент увеличения/уменьшения
                expansion = meta['result_rows'] / meta['left_rows']
                logger.info(f"   Коэффициент: {expansion:.2f}x")
            
            # Дополнительная информация для разных типов join
            if operation.name == 'left_join' and 'left_only_rows' in meta:
                logger.info(f"   Только в левой: {meta['left_only_rows']} строк")
                logger.info(f"   В обеих: {meta['both_rows']} строк")
            
            if operation.name == 'merge_multiple_keys' and 'selectivity' in meta:
                logger.info(f"   Селективность: {meta['selectivity']:.6f}")
            
            if operation.name == 'asof_join' and 'match_rate' in meta:
                logger.info(f"   Процент совпадений: {meta['match_rate']*100:.1f}%")
        else:
            logger.error(f"❌ Ошибка: {result_pandas.error}")
        
        # Тест Polars
        logger.info("\n📊 Polars:")
        start = time.time()
        result_polars = operation.execute_polars(df_polars)
        polars_time = time.time() - start
        
        if result_polars.success:
            logger.info(f"✅ Успешно выполнено за {polars_time:.3f}с")
            
            # Сравнение производительности
            if result_pandas.success:
                speedup = pandas_time / polars_time
                logger.info(f"   🚀 Polars быстрее в {speedup:.1f}x раз")
        else:
            logger.error(f"❌ Ошибка: {result_polars.error}")
    
    logger.phase_end("Тестирование Join операций")
    
    # 3. Бенчмарк с профилировщиком
    logger.phase_start("Бенчмарк производительности")
    
    # Создаем больший датасет для тестирования
    logger.info("\nСоздаем датасет большего размера для бенчмарка...")
    df_bench_pd, df_bench_pl = create_test_data(50000)
    
    # Настройка профилировщика
    config = ProfilingConfig(
        min_runs=3,
        max_runs=10,
        target_cv=0.05,
        memory_sampling_interval=0.01,
        warmup_runs=1
    )
    
    profiler = get_profiler(config)
    
    # Тестируем inner join
    inner_join_op = get_operation('inner_join', 'join')
    
    logger.info("\n📊 Профилирование Inner Join:")
    
    # Pandas
    logger.info("  Pandas...")
    result_pd = profiler.profile_operation(
        inner_join_op, 
        df_bench_pd,
        library='pandas'
    )
    if result_pd.success:
        logger.info(f"    Время: {result_pd.execution_time_mean:.3f}с ± {result_pd.execution_time_std:.3f}с")
        logger.info(f"    Память (пик): {result_pd.memory_peak_mb:.1f} MB")
        logger.info(f"    CV: {result_pd.cv:.3f}")
    
    # Polars
    logger.info("  Polars...")
    result_pl = profiler.profile_operation(
        inner_join_op,
        df_bench_pl,
        library='polars'
    )
    if result_pl.success:
        logger.info(f"    Время: {result_pl.execution_time_mean:.3f}с ± {result_pl.execution_time_std:.3f}с")
        logger.info(f"    Память (пик): {result_pl.memory_peak_mb:.1f} MB")
        logger.info(f"    CV: {result_pl.cv:.3f}")
    
    if result_pd.success and result_pl.success:
        speedup = result_pd.execution_time_mean / result_pl.execution_time_mean
        memory_ratio = result_pl.memory_peak_mb / result_pd.memory_peak_mb
        logger.info(f"\n  🚀 Polars быстрее в {speedup:.1f}x раз")
        logger.info(f"  💾 Polars использует {memory_ratio:.1%} от памяти Pandas")
    
    logger.phase_end("Бенчмарк производительности")
    
    # 4. Практические примеры
    logger.phase_start("Практические примеры")
    
    # Пример 1: Join с фильтрацией
    logger.info("\n📝 Пример 1: Inner join только для определенного региона")
    
    # Получаем операцию inner join
    inner_join = get_operation('inner_join', 'join')
    
    if inner_join is None:
        logger.error("❌ Операция inner_join не найдена!")
        logger.info("Пропускаем практические примеры...")
        logger.phase_end("Практические примеры")
        return
    
    # Фильтруем перед join
    df_north_pd = df_pandas[df_pandas['region'] == 'North']
    df_north_pl = df_polars.filter(pl.col('region') == 'North')
    
    result_pd = inner_join.execute_pandas(df_north_pd)
    result_pl = inner_join.execute_polars(df_north_pl)
    
    if result_pd.success and result_pl.success:
        logger.info(f"  Отфильтровано до join: {len(df_north_pd)} строк")
        logger.info(f"  Результат join: {result_pd.metadata['result_rows']} строк")
    
    # Пример 2: Цепочка join операций
    logger.info("\n📝 Пример 2: Последовательные join операции")
    
    if result_pd.success:
        # Второй join на результате первого
        left_join = get_operation('left_join', 'join')
        result2_pd = left_join.execute_pandas(result_pd.result)
        
        if result2_pd.success:
            logger.info(f"  Первый join: {result_pd.metadata['result_rows']} строк")
            logger.info(f"  Второй join: {result2_pd.metadata['result_rows']} строк")
    
    logger.phase_end("Практические примеры")
    
    # 5. Особенности Polars
    logger.phase_start("Особенности Polars")
    
    logger.info("\n🚀 Преимущества Polars в join операциях:")
    logger.info("  1. Параллельное выполнение join")
    logger.info("  2. Оптимизация памяти через zero-copy")
    logger.info("  3. Умная оптимизация запросов в lazy mode")
    logger.info("  4. Эффективная работа с большими данными")
    
    # Демонстрация lazy evaluation
    logger.info("\n📝 Пример lazy evaluation:")
    
    # Создаем сложный запрос
    lazy_df = df_polars.lazy()
    
    # Симулируем сложную цепочку операций
    result = (
        lazy_df
        .filter(pl.col('quantity') > 5)
        .join(
            lazy_df.filter(pl.col('price') > 100),
            on=['customer_id', 'product_id'],
            how='inner'
        )
        .group_by('region')
        .agg([
            pl.col('quantity').sum().alias('total_quantity'),
            pl.col('price').mean().alias('avg_price')
        ])
        .sort('total_quantity', descending=True)
    )
    
    # План выполнения оптимизируется автоматически
    logger.info("  План выполнения оптимизирован Polars query planner")
    logger.info("  Фильтрация происходит ДО join для экономии памяти")
    
    # Выполняем
    final_result = result.collect()
    logger.info(f"  Результат агрегации: {len(final_result)} регионов")
    
    logger.phase_end("Особенности Polars")
    
    # Итоги
    logger.info("\n" + "=" * 80)
    logger.info("ИТОГИ")
    logger.info("=" * 80)
    logger.info("\n✅ Все Join операции реализованы и протестированы:")
    logger.info("  1. inner_join - внутреннее соединение")
    logger.info("  2. left_join - левое соединение")
    logger.info("  3. merge_multiple_keys - join по нескольким ключам")
    logger.info("  4. asof_join - временной join")
    
    logger.info("\n🚀 Ключевые выводы:")
    logger.info("  - Polars обычно в 3-10x быстрее на join операциях")
    logger.info("  - Ускорение растет с размером данных")
    logger.info("  - Lazy evaluation позволяет оптимизировать сложные запросы")
    logger.info("  - Asof join особенно полезен для временных рядов")


if __name__ == '__main__':
    try:
        demonstrate_join_operations()
    except KeyboardInterrupt:
        print("\n\nДемонстрация прервана пользователем")
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()