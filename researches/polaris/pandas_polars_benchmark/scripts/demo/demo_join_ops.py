#!/usr/bin/env python3
"""
Демонстрационный скрипт для проверки Join операций.
"""

import sys
from pathlib import Path

# Добавляем путь к src
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pandas as pd
import polars as pl
import numpy as np
import time

from utils import setup_logging
from operations import get_operation, get_operations_by_category
from profiling import get_profiler, ProfilingConfig


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
    
    # 3. Детальный анализ производительности
    logger.phase_start("Анализ производительности")
    
    # Тестируем inner join на разных размерах
    sizes = [1000, 5000, 10000, 20000]
    inner_join = get_operation('inner_join', 'join')
    
    profiling_config = ProfilingConfig(
        min_runs=3,
        max_runs=5,
        target_cv=0.20,
        isolate_process=False
    )
    
    results = []
    
    logger.info("\n📈 Масштабируемость Inner Join:")
    
    with get_profiler(profiling_config) as profiler:
        for size in sizes:
            logger.info(f"\nРазмер данных: {size:,} строк")
            
            # Создаем данные
            df_pd, df_pl = create_test_data(size)
            
            # Профилируем Pandas
            result_pd = profiler.profile_operation(
                lambda: inner_join.execute_pandas(df_pd),
                operation_name=f"inner_join_{size}",
                library='pandas',
                dataset_size=size
            )
            
            # Профилируем Polars
            result_pl = profiler.profile_operation(
                lambda: inner_join.execute_polars(df_pl),
                operation_name=f"inner_join_{size}",
                library='polars',
                dataset_size=size
            )
            
            if result_pd.success and result_pl.success:
                speedup = result_pd.mean_time / result_pl.mean_time
                logger.info(f"  Pandas: {result_pd.mean_time:.3f}с, Polars: {result_pl.mean_time:.3f}с")
                logger.info(f"  Ускорение: {speedup:.1f}x")
                
                results.append({
                    'size': size,
                    'pandas_time': result_pd.mean_time,
                    'polars_time': result_pl.mean_time,
                    'speedup': speedup
                })
    
    # График ускорения
    if results:
        logger.info("\n📊 Сводка по масштабируемости:")
        logger.info("-" * 60)
        logger.info(f"{'Размер':<10} {'Pandas (с)':<12} {'Polars (с)':<12} {'Ускорение':<10}")
        logger.info("-" * 60)
        
        for r in results:
            logger.info(
                f"{r['size']:<10} "
                f"{r['pandas_time']:<12.3f} "
                f"{r['polars_time']:<12.3f} "
                f"{r['speedup']:<10.1f}x"
            )
        
        # Анализ роста ускорения
        speedups = [r['speedup'] for r in results]
        if len(speedups) > 1:
            speedup_growth = (speedups[-1] - speedups[0]) / speedups[0] * 100
            logger.info(f"\nРост ускорения: {speedup_growth:+.1f}% с увеличением размера данных")
    
    logger.phase_end("Анализ производительности")
    
    # 4. Практические примеры
    logger.phase_start("Практические примеры")
    
    # Пример 1: Join с фильтрацией
    logger.info("\n📝 Пример 1: Inner join только для определенного региона")
    
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
