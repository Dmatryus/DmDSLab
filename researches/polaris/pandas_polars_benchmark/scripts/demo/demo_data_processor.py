#!/usr/bin/env python3

# Добавляем путь к src для импортов
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

"""
Демонстрация работы процессора данных для подготовки к визуализации.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

# Предполагаем, что модули находятся в правильной структуре
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.reporting.data_processor import (
    DataProcessor, 
    MetricType, 
    AggregationLevel,
    ProcessedData
)
from utils.logging import get_logger


def create_sample_results():
    """Создание примерных результатов бенчмаркинга."""
    np.random.seed(42)
    
    libraries = ['pandas', 'polars']
    operations = ['read_csv', 'filter', 'groupby', 'sort', 'join']
    dataset_sizes = [1000, 10000, 100000, 1000000]
    
    results = []
    
    for lib in libraries:
        for op in operations:
            for size in dataset_sizes:
                # Генерация реалистичных метрик
                base_time = {
                    'pandas': {'read_csv': 0.1, 'filter': 0.05, 'groupby': 0.2, 'sort': 0.15, 'join': 0.3},
                    'polars': {'read_csv': 0.03, 'filter': 0.01, 'groupby': 0.05, 'sort': 0.04, 'join': 0.08}
                }
                
                # Время выполнения растет с размером данных
                time_factor = np.log10(size) / 3
                base = base_time[lib][op]
                
                # Добавляем несколько измерений для каждой комбинации
                for _ in range(5):
                    exec_time = base * time_factor * np.random.normal(1.0, 0.1)
                    memory_usage = size * 8 / 1024 / 1024 * np.random.normal(1.0, 0.15)  # MB
                    memory_peak = memory_usage * np.random.uniform(1.1, 1.5)
                    
                    results.append({
                        'library': lib,
                        'operation': op,
                        'dataset_size': size,
                        'execution_time': max(0.001, exec_time),
                        'memory_usage': max(1.0, memory_usage),
                        'memory_peak': max(1.0, memory_peak),
                        'timestamp': pd.Timestamp.now()
                    })
    
    return pd.DataFrame(results)


def demonstrate_comparison_preparation():
    """Демонстрация подготовки данных для сравнения."""
    logger = get_logger(__name__)
    logger.info("="*60)
    logger.info("1. ПОДГОТОВКА ДАННЫХ ДЛЯ СРАВНИТЕЛЬНЫХ ГРАФИКОВ")
    logger.info("="*60)
    
    # Создание примерных данных
    df = create_sample_results()
    processor = DataProcessor()
    
    # Сравнение по операциям
    comparison_data = processor.prepare_comparison_data(
        df=df,
        metric=MetricType.EXECUTION_TIME,
        groupby=['operation'],
        pivot_column='library'
    )
    
    print("\nДанные для сравнения библиотек по операциям:")
    print(comparison_data.data)
    print(f"\nКолонки после обработки: {comparison_data.data.columns.tolist()}")
    print(f"Метаданные: {comparison_data.metadata}")
    
    # Сравнение с расчетом ускорения
    speedup_data = processor.prepare_comparison_data(
        df=df,
        metric=MetricType.SPEEDUP,
        groupby=['operation', 'dataset_size'],
        pivot_column='library'
    )
    
    print("\n\nДанные по ускорению (Pandas как baseline):")
    print(speedup_data.data.head(10))


def demonstrate_distribution_preparation():
    """Демонстрация подготовки данных для распределений."""
    logger = get_logger(__name__)
    logger.info("\n" + "="*60)
    logger.info("2. ПОДГОТОВКА ДАННЫХ ДЛЯ ГРАФИКОВ РАСПРЕДЕЛЕНИЯ")
    logger.info("="*60)
    
    df = create_sample_results()
    processor = DataProcessor()
    
    # Распределение времени выполнения для Polars
    dist_data = processor.prepare_distribution_data(
        df=df,
        metric=MetricType.EXECUTION_TIME,
        library='polars',
        operation='groupby'
    )
    
    print("\nДанные распределения для Polars (groupby):")
    print(f"Количество измерений: {len(dist_data.data)}")
    print(f"Статистики: {dist_data.metadata['statistics']}")
    print("\nПервые 5 записей:")
    print(dist_data.data.head())


def demonstrate_timeline_preparation():
    """Демонстрация подготовки данных для временных графиков."""
    logger = get_logger(__name__)
    logger.info("\n" + "="*60)
    logger.info("3. ПОДГОТОВКА ДАННЫХ ДЛЯ ГРАФИКОВ ЗАВИСИМОСТИ ОТ РАЗМЕРА")
    logger.info("="*60)
    
    df = create_sample_results()
    processor = DataProcessor()
    
    timeline_data = processor.prepare_timeline_data(
        df=df,
        metric=MetricType.EXECUTION_TIME,
        dataset_size_column='dataset_size'
    )
    
    print("\nДанные для графика зависимости от размера:")
    print(f"Размеры датасетов: {timeline_data.metadata['dataset_sizes']}")
    print(f"Операции: {timeline_data.metadata['operations']}")
    print("\nПример данных (filter операция):")
    print(timeline_data.data[timeline_data.data['operation'] == 'filter'])


def demonstrate_heatmap_preparation():
    """Демонстрация подготовки данных для тепловых карт."""
    logger = get_logger(__name__)
    logger.info("\n" + "="*60)
    logger.info("4. ПОДГОТОВКА ДАННЫХ ДЛЯ ТЕПЛОВЫХ КАРТ")
    logger.info("="*60)
    
    df = create_sample_results()
    processor = DataProcessor()
    
    heatmap_data = processor.prepare_heatmap_data(
        df=df,
        metric=MetricType.EXECUTION_TIME,
        row_column='operation',
        col_column='dataset_size'
    )
    
    print("\nДанные для тепловой карты:")
    print(f"Диапазон значений: {heatmap_data.metadata['value_range']}")
    print("\nДанные для Pandas:")
    print(heatmap_data.data[heatmap_data.data['library'] == 'pandas'])


def demonstrate_summary_table():
    """Демонстрация создания сводной таблицы."""
    logger = get_logger(__name__)
    logger.info("\n" + "="*60)
    logger.info("5. СОЗДАНИЕ СВОДНОЙ ТАБЛИЦЫ")
    logger.info("="*60)
    
    df = create_sample_results()
    processor = DataProcessor()
    
    summary = processor.create_summary_table(df)
    
    print("\nСводная таблица результатов:")
    print(summary.to_string(index=False, float_format='%.3f'))
    
    # Сохранение примера
    output_dir = Path("demo_output")
    output_dir.mkdir(exist_ok=True)
    
    summary.to_csv(output_dir / "summary_table.csv", index=False)
    print(f"\nСводная таблица сохранена в {output_dir / 'summary_table.csv'}")


def demonstrate_export():
    """Демонстрация экспорта данных."""
    logger = get_logger(__name__)
    logger.info("\n" + "="*60)
    logger.info("6. ЭКСПОРТ ОБРАБОТАННЫХ ДАННЫХ")
    logger.info("="*60)
    
    df = create_sample_results()
    processor = DataProcessor()
    
    # Подготовка данных
    processed = processor.prepare_comparison_data(
        df=df,
        metric=MetricType.EXECUTION_TIME,
        groupby=['operation'],
        pivot_column='library'
    )
    
    # Экспорт в JSON
    output_dir = Path("demo_output")
    output_dir.mkdir(exist_ok=True)
    
    json_path = output_dir / "comparison_data.json"
    processor.export_for_visualization(processed, json_path, format="json")
    
    # Экспорт в CSV
    csv_path = output_dir / "comparison_data.csv"
    processor.export_for_visualization(processed, csv_path, format="csv")
    
    print(f"Данные экспортированы в:")
    print(f"  - JSON: {json_path}")
    print(f"  - CSV: {csv_path}")
    
    # Показать содержимое JSON
    with open(json_path, 'r') as f:
        json_content = json.load(f)
    print(f"\nСтруктура JSON:")
    print(f"  - Ключи: {list(json_content.keys())}")
    print(f"  - Количество записей: {len(json_content['data'])}")
    print(f"  - Уровень агрегации: {json_content['aggregation_level']}")
    print(f"  - Тип метрики: {json_content['metric_type']}")


if __name__ == "__main__":
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🚀 Демонстрация работы процессора данных\n")
    
    # Запуск всех демонстраций
    demonstrate_comparison_preparation()
    demonstrate_distribution_preparation()
    demonstrate_timeline_preparation()
    demonstrate_heatmap_preparation()
    demonstrate_summary_table()
    demonstrate_export()
    
    print("\n✨ Демонстрация завершена!")
