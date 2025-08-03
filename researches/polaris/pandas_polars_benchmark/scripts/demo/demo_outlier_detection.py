#!/usr/bin/env python3
"""
Демонстрация работы модуля обнаружения выбросов.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Добавляем путь к src
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from analysis.outlier_detector import (
    OutlierDetector, OutlierMethod, compare_outlier_methods
)
from utils import setup_logging


def generate_test_data(n_samples: int = 100, outlier_ratio: float = 0.1):
    """Генерация тестовых данных с выбросами."""
    np.random.seed(42)
    
    # Основные данные - нормальное распределение
    main_data = np.random.normal(100, 10, int(n_samples * (1 - outlier_ratio)))
    
    # Выбросы
    n_outliers = int(n_samples * outlier_ratio)
    outliers = np.concatenate([
        np.random.normal(150, 5, n_outliers // 2),  # Верхние выбросы
        np.random.normal(50, 5, n_outliers // 2)    # Нижние выбросы
    ])
    
    # Объединяем и перемешиваем
    data = np.concatenate([main_data, outliers])
    np.random.shuffle(data)
    
    return data


def visualize_outliers(data, result, title="Обнаружение выбросов"):
    """Визуализация результатов обнаружения выбросов."""
    plt.figure(figsize=(12, 6))
    
    # График 1: Гистограмма с выбросами
    plt.subplot(1, 2, 1)
    
    # Все данные
    plt.hist(data, bins=30, alpha=0.7, label='Все данные', color='blue')
    
    # Выбросы
    if result.outlier_values:
        plt.hist(result.outlier_values, bins=10, alpha=0.7, 
                label=f'Выбросы ({result.removed_count})', color='red')
    
    # Границы
    if result.lower_bound is not None:
        plt.axvline(result.lower_bound, color='green', linestyle='--', 
                   label=f'Нижняя граница: {result.lower_bound:.2f}')
    if result.upper_bound is not None:
        plt.axvline(result.upper_bound, color='green', linestyle='--',
                   label=f'Верхняя граница: {result.upper_bound:.2f}')
    
    plt.xlabel('Значение')
    plt.ylabel('Частота')
    plt.title(f'{title} - {result.method.upper()}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # График 2: Boxplot
    plt.subplot(1, 2, 2)
    
    # Данные до и после очистки
    clean_mask = np.ones(len(data), dtype=bool)
    clean_mask[result.outlier_indices] = False
    clean_data = data[clean_mask]
    
    box_data = [data, clean_data]
    plt.boxplot(box_data, labels=['До очистки', 'После очистки'])
    plt.ylabel('Значение')
    plt.title('Сравнение до и после удаления выбросов')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def demonstrate_outlier_detection():
    """Основная демонстрация."""
    # Настройка логирования
    logger = setup_logging('demo_outliers', console_level='INFO')
    
    logger.info("=" * 80)
    logger.info("ДЕМОНСТРАЦИЯ ОБНАРУЖЕНИЯ ВЫБРОСОВ")
    logger.info("=" * 80)
    
    # 1. Генерация тестовых данных
    logger.info("\n1. ГЕНЕРАЦИЯ ТЕСТОВЫХ ДАННЫХ")
    data = generate_test_data(n_samples=200, outlier_ratio=0.15)
    logger.info(f"Сгенерировано {len(data)} значений")
    logger.info(f"Среднее: {np.mean(data):.2f}, Std: {np.std(data):.2f}")
    
    # 2. Создание детектора
    detector = OutlierDetector(logger)
    
    # 3. Метод IQR
    logger.info("\n2. МЕТОД IQR (Межквартильный размах)")
    result_iqr = detector.detect_outliers(data, OutlierMethod.IQR, multiplier=1.5)
    logger.info(f"Обнаружено выбросов: {result_iqr.removed_count}")
    logger.info(f"Границы: [{result_iqr.lower_bound:.2f}, {result_iqr.upper_bound:.2f}]")
    
    # 4. Метод Z-score
    logger.info("\n3. МЕТОД Z-SCORE")
    result_zscore = detector.detect_outliers(data, OutlierMethod.ZSCORE, threshold=3.0)
    logger.info(f"Обнаружено выбросов: {result_zscore.removed_count}")
    logger.info(f"Границы: [{result_zscore.lower_bound:.2f}, {result_zscore.upper_bound:.2f}]")
    
    # 5. Метод процентилей
    logger.info("\n4. МЕТОД ПРОЦЕНТИЛЕЙ")
    result_percentile = detector.detect_outliers(
        data, OutlierMethod.PERCENTILE, 
        lower_percentile=5, upper_percentile=95
    )
    logger.info(f"Обнаружено выбросов: {result_percentile.removed_count}")
    logger.info(f"Границы: [{result_percentile.lower_bound:.2f}, {result_percentile.upper_bound:.2f}]")
    
    # 6. Сравнение методов
    logger.info("\n5. СРАВНЕНИЕ МЕТОДОВ")
    comparison_df = compare_outlier_methods(data)
    logger.info("\n" + comparison_df.to_string())
    
    # 7. Визуализация (если доступна)
    try:
        logger.info("\n6. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ")
        visualize_outliers(data, result_iqr, "Метод IQR")
        visualize_outliers(data, result_zscore, "Метод Z-score")
    except Exception as e:
        logger.warning(f"Визуализация недоступна: {e}")
    
    # 8. Пример работы с реальными данными бенчмарка
    logger.info("\n7. ПРИМЕР С ДАННЫМИ БЕНЧМАРКА")
    
    # Симулируем результаты времени выполнения
    benchmark_times = {
        "pandas_read_csv": np.random.gamma(2, 0.5, 50) + np.random.normal(0, 0.1, 50),
        "polars_read_csv": np.random.gamma(1.5, 0.3, 50) + np.random.normal(0, 0.05, 50),
    }
    
    # Добавляем несколько выбросов
    benchmark_times["pandas_read_csv"][5] = 10.0  # Выброс
    benchmark_times["pandas_read_csv"][15] = 0.1  # Выброс
    benchmark_times["polars_read_csv"][10] = 8.0  # Выброс
    
    # Анализируем
    results = detector.analyze_multiple_runs(benchmark_times, OutlierMethod.IQR)
    
    logger.info("\nОчищенные результаты:")
    for name, (clean_data, result) in results.items():
        logger.info(f"\n{name}:")
        logger.info(f"  Удалено выбросов: {result.removed_count} ({result.removed_percentage:.1f}%)")
        logger.info(f"  CV до очистки: {result.stats_before['cv']:.3f}")
        logger.info(f"  CV после очистки: {result.stats_after['cv']:.3f}")
    
    # 9. Рекомендации по выбору метода
    logger.info("\n8. РЕКОМЕНДАЦИИ ПО ВЫБОРУ МЕТОДА")
    logger.info("- IQR: Робастный метод, хорош для несимметричных распределений")
    logger.info("- Z-score: Предполагает нормальное распределение, чувствителен к выбросам")
    logger.info("- Процентили: Простой метод, хорош когда нужно удалить фиксированный процент")
    
    logger.info("\n✅ Модуль обнаружения выбросов работает корректно!")
    
    return detector, results


if __name__ == "__main__":
    try:
        demonstrate_outlier_detection()
    except KeyboardInterrupt:
        print("\n\nДемонстрация прервана пользователем")
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()