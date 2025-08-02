#!/usr/bin/env python3
"""
Демонстрационный скрипт для проверки всех компонентов Фазы 3.
"""

import sys
import time
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path

# Добавляем путь к src
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from utils.logging import setup_logging
from data import DataLoader
from profiling import (
    MemoryTracker, Timer, ProfilingConfig, Profiler,
    measure_memory, measure_time, get_profiler
)


def demo_memory_tracker():
    """Демонстрация работы Memory Tracker."""
    print("\n" + "="*60)
    print("ДЕМО: Memory Tracker")
    print("="*60)
    
    # Создаем трекер памяти
    tracker = MemoryTracker(sampling_interval=0.05)
    
    print("Начинаем отслеживание памяти...")
    tracker.start_tracking()
    
    # Симулируем работу с памятью
    data = []
    for i in range(5):
        # Выделяем память
        chunk = np.random.rand(1000000)  # ~7.6 MB
        data.append(chunk)
        print(f"  Выделено блок {i+1}: {chunk.nbytes / 1024 / 1024:.1f} MB")
        time.sleep(0.2)
    
    # Останавливаем отслеживание
    stats = tracker.stop_tracking()
    
    print(f"\nРезультаты отслеживания памяти:")
    print(f"  - Пиковая память: {stats.peak_memory_mb:.1f} MB")
    print(f"  - Средняя память: {stats.average_memory_mb:.1f} MB")
    print(f"  - Мин/Макс: {stats.min_memory_mb:.1f} / {stats.max_memory_mb:.1f} MB")
    print(f"  - Количество замеров: {stats.sample_count}")
    print(f"  - Длительность: {stats.duration_seconds:.2f} сек")
    
    return True


def demo_timer():
    """Демонстрация работы Timer."""
    print("\n" + "="*60)
    print("ДЕМО: Timer с автоповтором")
    print("="*60)
    
    # Создаем таймер
    timer = Timer(min_runs=3, max_runs=20, target_cv=0.05)
    
    # Функция с переменным временем выполнения
    def variable_operation():
        # Симулируем операцию с небольшой вариацией
        base_time = 0.1
        variation = np.random.normal(0, 0.005)
        time.sleep(base_time + variation)
    
    print("Измеряем время выполнения до достижения CV < 5%...")
    result = timer.time_execution(variable_operation)
    
    print(f"\nРезультаты измерения времени:")
    print(f"  - Среднее время: {result.mean_time:.3f} сек")
    print(f"  - Медиана: {result.median_time:.3f} сек")
    print(f"  - Станд. отклонение: {result.std_dev:.3f} сек")
    print(f"  - Коэффициент вариации: {result.cv:.4f} ({result.cv*100:.2f}%)")
    print(f"  - Мин/Макс: {result.min_time:.3f} / {result.max_time:.3f} сек")
    print(f"  - Количество запусков: {result.runs_count}")
    print(f"  - Сходимость достигнута: {'Да' if result.converged else 'Нет'}")
    
    return result.converged


def demo_profiler_inline():
    """Демонстрация Profiler в inline режиме."""
    print("\n" + "="*60)
    print("ДЕМО: Profiler (inline mode)")
    print("="*60)
    
    # Создаем конфигурацию
    config = ProfilingConfig(
        min_runs=3,
        max_runs=10,
        target_cv=0.1,
        isolate_process=False  # Inline mode
    )
    
    # Создаем профайлер (автоматически настроится для текущей ОС)
    profiler = Profiler(config)
    
    # Операция для профилирования
    def pandas_groupby():
        df = pd.DataFrame({
            'category': np.random.choice(['A', 'B', 'C'], 10000),
            'value': np.random.randn(10000)
        })
        return df.groupby('category')['value'].mean()
    
    print("Профилируем операцию pandas groupby...")
    result = profiler.profile_operation(
        pandas_groupby,
        operation_name="groupby_mean",
        library="pandas",
        backend="numpy",
        dataset_name="demo_data",
        dataset_size=10000
    )
    
    print(f"\nРезультат профилирования:")
    print(f"  - Операция: {result.operation_name}")
    print(f"  - Библиотека: {result.library} ({result.backend})")
    print(f"  - Успешно: {'Да' if result.success else 'Нет'}")
    print(f"  - Время: {result.mean_time:.3f}±{result.std_time:.3f} сек")
    print(f"  - Память: {result.peak_memory_mb:.1f} MB (пик), {result.avg_memory_mb:.1f} MB (средн.)")
    print(f"  - Запусков: {result.runs_count}")
    
    profiler.cleanup()
    return result.success


def demo_profiler_isolated():
    """Демонстрация Profiler в isolated режиме."""
    print("\n" + "="*60)
    print("ДЕМО: Profiler (isolated process mode)")
    print("="*60)
    
    # Создаем конфигурацию
    config = ProfilingConfig(
        min_runs=3,
        max_runs=10,
        target_cv=0.1,
        isolate_process=True  # Isolated mode (автоматически отключится на Windows)
    )
    
    with get_profiler(config) as profiler:
        # На Windows будет использован inline режим
        actual_mode = "isolated" if profiler.config.isolate_process else "inline"
        print(f"Фактический режим: {actual_mode}")
        
        # Операция для профилирования
        def polars_filter():
            df = pl.DataFrame({
                'id': range(50000),
                'value': np.random.randn(50000),
                'category': np.random.choice(['X', 'Y', 'Z'], 50000)
            })
            return df.filter(pl.col('value') > 0)
        
        print(f"Профилируем операцию polars filter в {actual_mode} режиме...")
        result = profiler.profile_operation(
            polars_filter,
            operation_name="filter_positive",
            library="polars",
            dataset_name="demo_data",
            dataset_size=50000
        )
        
        print(f"\nРезультат профилирования:")
        print(f"  - Операция: {result.operation_name}")
        print(f"  - Библиотека: {result.library}")
        print(f"  - Успешно: {'Да' if result.success else 'Нет'}")
        
        if result.success:
            print(f"  - Время: {result.mean_time:.3f}±{result.std_time:.3f} сек")
            print(f"  - Память: {result.peak_memory_mb:.1f} MB (пик)")
            print(f"  - CV времени: {result.cv_time:.4f}")
            print(f"  - Сходимость: {'Да' if result.converged else 'Нет'}")
            
            if result.result_info:
                print(f"  - Результат: {result.result_info}")
    
    return result.success


def demo_decorators():
    """Демонстрация декораторов."""
    print("\n" + "="*60)
    print("ДЕМО: Декораторы @measure_memory и @measure_time")
    print("="*60)
    
    @measure_memory
    @measure_time(min_runs=5, target_cv=0.05)
    def memory_intensive_operation():
        # Выделяем и обрабатываем память
        data = np.random.rand(5000000)  # ~38 MB
        result = np.sort(data)
        return {'size': len(result)}
    
    print("Выполняем операцию с декораторами...")
    result = memory_intensive_operation()
    
    print(f"\nРезультат операции:")
    print(f"  - Размер данных: {result['size']:,}")
    
    if 'memory_stats' in result:
        ms = result['memory_stats']
        print(f"  - Память (пик): {ms['peak_memory_mb']:.1f} MB")
    
    if 'timing_stats' in result:
        ts = result['timing_stats']
        print(f"  - Время: {ts['mean_time']:.3f} сек")
        print(f"  - CV: {ts['cv']:.4f}")
    
    return True


def demo_real_operations():
    """Демонстрация профилирования реальных операций."""
    print("\n" + "="*60)
    print("ДЕМО: Профилирование реальных операций")
    print("="*60)
    
    # Загружаем небольшой датасет
    loader = DataLoader(Path('data/demo'))
    
    # Проверяем доступные датасеты
    available = loader.get_available_datasets()
    if not available:
        print("❌ Нет доступных датасетов. Сначала запустите demo_phase2.py")
        return False
    
    # Берем первый доступный датасет
    dataset_name = list(available.values())[0][0]
    print(f"Используем датасет: {dataset_name}")
    
    # Профилируем операции
    with get_profiler() as profiler:
        results = []
        
        # 1. Pandas read CSV
        def pandas_read():
            return loader.load_pandas_csv(dataset_name, backend='numpy')
        
        print("\n1. Профилируем pandas read_csv...")
        result = profiler.profile_operation(
            pandas_read,
            operation_name="read_csv",
            library="pandas",
            backend="numpy",
            dataset_name=dataset_name
        )
        results.append(result)
        
        # 2. Polars read CSV
        def polars_read():
            return loader.load_polars_csv(dataset_name, lazy=False)
        
        print("\n2. Профилируем polars read_csv...")
        result = profiler.profile_operation(
            polars_read,
            operation_name="read_csv", 
            library="polars",
            dataset_name=dataset_name
        )
        results.append(result)
    
    # Сравнение результатов
    print("\n" + "="*40)
    print("СРАВНЕНИЕ РЕЗУЛЬТАТОВ")
    print("="*40)
    
    for r in results:
        if r.success:
            print(f"\n{r.library} ({r.backend or 'default'}):")
            print(f"  Время: {r.mean_time:.3f} сек")
            print(f"  Память: {r.peak_memory_mb:.1f} MB")
            print(f"  Строк/колонок: {r.result_info.get('rows', '?')}/{r.result_info.get('columns', '?')}")
    
    return all(r.success for r in results)


def main():
    """Основная функция демонстрации."""
    # Настраиваем логирование
    logger = setup_logging(
        name='demo_phase3',
        console_level='INFO',
        use_colors=True
    )
    
    logger.benchmark_start({'benchmark': {'name': 'Phase 3 Demo', 'version': '1.0.0'}})
    
    print("\n" + "="*80)
    print("ДЕМОНСТРАЦИЯ КОМПОНЕНТОВ ФАЗЫ 3: СИСТЕМА ПРОФИЛИРОВАНИЯ")
    print("="*80)
    
    results = []
    
    # 1. Memory Tracker
    try:
        results.append(("Memory Tracker", demo_memory_tracker()))
    except Exception as e:
        print(f"❌ Ошибка в Memory Tracker: {e}")
        results.append(("Memory Tracker", False))
    
    # 2. Timer
    try:
        results.append(("Timer", demo_timer()))
    except Exception as e:
        print(f"❌ Ошибка в Timer: {e}")
        results.append(("Timer", False))
    
    # 3. Profiler Inline
    try:
        results.append(("Profiler Inline", demo_profiler_inline()))
    except Exception as e:
        print(f"❌ Ошибка в Profiler Inline: {e}")
        results.append(("Profiler Inline", False))
    
    # 4. Profiler Isolated
    try:
        results.append(("Profiler Isolated", demo_profiler_isolated()))
    except Exception as e:
        print(f"❌ Ошибка в Profiler Isolated: {e}")
        results.append(("Profiler Isolated", False))
    
    # 5. Decorators
    try:
        results.append(("Decorators", demo_decorators()))
    except Exception as e:
        print(f"❌ Ошибка в Decorators: {e}")
        results.append(("Decorators", False))
    
    # 6. Real Operations
    try:
        results.append(("Real Operations", demo_real_operations()))
    except Exception as e:
        print(f"❌ Ошибка в Real Operations: {e}")
        results.append(("Real Operations", False))
    
    # Итоги
    print("\n" + "="*80)
    print("ИТОГИ ТЕСТИРОВАНИЯ")
    print("="*80)
    
    for name, success in results:
        status = "✅" if success else "❌"
        print(f"{status} {name}")
    
    all_success = all(success for _, success in results)
    
    if all_success:
        print("\n✅ ВСЕ КОМПОНЕНТЫ ФАЗЫ 3 РАБОТАЮТ КОРРЕКТНО!")
        print("\nГотовые компоненты:")
        print("  ✓ MemoryTracker - отслеживание памяти с настраиваемым интервалом")
        print("  ✓ Timer - измерение времени с автоповтором до достижения CV")
        print("  ✓ Profiler - профилирование в изолированном процессе")
        print("  ✓ Декораторы для удобного использования")
        print("  ✓ Поддержка inline и isolated режимов")
        print("\nМожно переходить к реализации checkpoint системы!")
    else:
        print("\n❌ Обнаружены ошибки, требуется исправление")
    
    logger.benchmark_end(success=all_success, duration=60)
    
    return all_success


if __name__ == '__main__':
    # Важно для Windows!
    import multiprocessing
    multiprocessing.freeze_support()
    
    success = main()
    sys.exit(0 if success else 1)
