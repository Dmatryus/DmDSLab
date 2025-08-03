#!/usr/bin/env python3
"""
Демонстрация работы модуля StatisticsCalculator.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List

# Добавляем путь к src в PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from analysis.statistics_calculator import (
    StatisticsCalculator, 
    DescriptiveStats,
    format_stats_report
)
from utils import setup_logging, get_logger


def generate_benchmark_data(size: int = 100, 
                          library: str = "pandas",
                          base_time: float = 1.0,
                          noise_level: float = 0.1) -> np.ndarray:
    """
    Генерация синтетических данных бенчмарка.
    
    Args:
        size: Количество измерений
        library: Название библиотеки (влияет на среднее время)
        base_time: Базовое время выполнения
        noise_level: Уровень шума (коэффициент вариации)
    
    Returns:
        np.ndarray: Массив времен выполнения
    """
    # Разное базовое время для разных библиотек
    if library == "polars":
        base_time *= 0.4  # Polars в среднем быстрее
    
    # Генерация с логнормальным распределением (типично для времени выполнения)
    mean_log = np.log(base_time)
    sigma = noise_level
    
    data = np.random.lognormal(mean_log, sigma, size)
    
    # Добавляем несколько выбросов
    n_outliers = int(size * 0.05)
    outlier_indices = np.random.choice(size, n_outliers, replace=False)
    data[outlier_indices] *= np.random.uniform(2, 4, n_outliers)
    
    return data


def demonstrate_basic_stats():
    """Демонстрация базовых статистик."""
    logger = get_logger(__name__)
    logger.info("\n" + "="*60)
    logger.info("1. БАЗОВЫЕ ОПИСАТЕЛЬНЫЕ СТАТИСТИКИ")
    logger.info("="*60)
    
    # Создание калькулятора
    calculator = StatisticsCalculator(confidence_level=0.95)
    
    # Генерация данных
    data = generate_benchmark_data(size=150, base_time=1.5, noise_level=0.15)
    
    # Расчет статистик
    stats = calculator.calculate_descriptive_stats(data)
    
    # Вывод отчета
    print(format_stats_report(stats, "Время выполнения операции"))
    
    return stats, data


def demonstrate_normality_tests():
    """Демонстрация тестов на нормальность."""
    logger = get_logger(__name__)
    logger.info("\n" + "="*60)
    logger.info("2. ТЕСТЫ НА НОРМАЛЬНОСТЬ РАСПРЕДЕЛЕНИЯ")
    logger.info("="*60)
    
    calculator = StatisticsCalculator()
    
    # Тест 1: Нормальное распределение
    normal_data = np.random.normal(10, 2, 200)
    normal_test = calculator.test_normality(normal_data)
    
    print("\nНормальное распределение (200 точек):")
    print(f"  Shapiro-Wilk: p-value = {normal_test.shapiro_pvalue:.4f}, "
          f"нормальное = {normal_test.shapiro_is_normal}")
    print(f"  D'Agostino: p-value = {normal_test.dagostino_pvalue:.4f}, "
          f"нормальное = {normal_test.dagostino_is_normal}")
    print(f"  Anderson-Darling: статистика = {normal_test.anderson_statistic:.4f}")
    
    # Тест 2: Логнормальное распределение (типично для времени)
    lognormal_data = np.random.lognormal(0, 0.5, 200)
    lognormal_test = calculator.test_normality(lognormal_data)
    
    print("\nЛогнормальное распределение (200 точек):")
    print(f"  Shapiro-Wilk: p-value = {lognormal_test.shapiro_pvalue:.4f}, "
          f"нормальное = {lognormal_test.shapiro_is_normal}")
    print(f"  D'Agostino: p-value = {lognormal_test.dagostino_pvalue:.4f}, "
          f"нормальное = {lognormal_test.dagostino_is_normal}")
    print(f"  Общий вывод: распределение {'нормальное' if lognormal_test.is_normal else 'НЕ нормальное'}")


def demonstrate_group_comparison():
    """Демонстрация сравнения групп."""
    logger = get_logger(__name__)
    logger.info("\n" + "="*60)
    logger.info("3. СРАВНЕНИЕ СТАТИСТИК МЕЖДУ БИБЛИОТЕКАМИ")
    logger.info("="*60)
    
    calculator = StatisticsCalculator()
    
    # Генерация данных для разных библиотек
    benchmark_results = {
        "pandas_read_csv": generate_benchmark_data(100, "pandas", 2.0, 0.1),
        "polars_read_csv": generate_benchmark_data(100, "polars", 2.0, 0.08),
        "pandas_groupby": generate_benchmark_data(100, "pandas", 5.0, 0.15),
        "polars_groupby": generate_benchmark_data(100, "polars", 5.0, 0.12),
    }
    
    # Расчет статистик для всех групп
    all_stats = calculator.calculate_stats_for_groups(benchmark_results)
    
    # Создание сводной таблицы
    summary_df = calculator.create_summary_table(all_stats)
    
    print("\nСводная таблица статистик:")
    print(summary_df.to_string(index=False))
    
    # Сравнение pandas vs polars для read_csv
    print("\n" + "-"*60)
    print("Сравнение pandas vs polars для операции read_csv:")
    
    relative_metrics = calculator.calculate_relative_metrics(
        all_stats["pandas_read_csv"],
        all_stats["polars_read_csv"]
    )
    
    print(f"  Ускорение (speedup): {relative_metrics['speedup']:.2f}x")
    print(f"  Разница в среднем времени: {relative_metrics['mean_difference_pct']:.1f}%")
    print(f"  Разница в CV: {relative_metrics['cv_difference']:.2f}%")
    print(f"  Доверительные интервалы пересекаются: {'Да' if relative_metrics['ci_overlap'] else 'Нет'}")
    
    return summary_df


def demonstrate_confidence_intervals():
    """Демонстрация доверительных интервалов."""
    logger = get_logger(__name__)
    logger.info("\n" + "="*60)
    logger.info("4. ДОВЕРИТЕЛЬНЫЕ ИНТЕРВАЛЫ")
    logger.info("="*60)
    
    calculator = StatisticsCalculator(confidence_level=0.95)
    
    # Разные размеры выборок
    sample_sizes = [10, 30, 100, 500]
    true_mean = 1.0
    
    print(f"\nИстинное среднее: {true_mean}")
    print("\nВлияние размера выборки на доверительный интервал:")
    print("-" * 50)
    print(f"{'Размер':<10} {'Среднее':<10} {'95% ДИ':<25} {'Ширина':<10}")
    print("-" * 50)
    
    for size in sample_sizes:
        data = np.random.normal(true_mean, 0.2, size)
        stats = calculator.calculate_descriptive_stats(data)
        
        ci_width = stats.confidence_interval[1] - stats.confidence_interval[0]
        ci_str = f"[{stats.confidence_interval[0]:.4f}, {stats.confidence_interval[1]:.4f}]"
        
        print(f"{size:<10} {stats.mean:<10.4f} {ci_str:<25} {ci_width:<10.4f}")


def main():
    """Основная функция демонстрации."""
    # Настройка логирования
    logger = setup_logging('demo_statistics', console_level='INFO')
    
    print("\n" + "="*80)
    print("ДЕМОНСТРАЦИЯ КАЛЬКУЛЯТОРА СТАТИСТИК")
    print("="*80)
    
    # 1. Базовые статистики
    stats, data = demonstrate_basic_stats()
    
    # 2. Тесты на нормальность
    demonstrate_normality_tests()
    
    # 3. Сравнение групп
    summary_df = demonstrate_group_comparison()
    
    # 4. Доверительные интервалы
    demonstrate_confidence_intervals()
    
    print("\n" + "="*80)
    print("Демонстрация завершена!")
    print("="*80)


if __name__ == "__main__":
    main()
