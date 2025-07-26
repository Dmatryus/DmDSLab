"""
Скрипт для запуска полного эксперимента по сравнению методов выбора порогов.

Использование:
    python run_threshold_experiment.py
"""

import sys
import warnings
from pathlib import Path
from datetime import datetime

from threshold_experiment import ThresholdExperiment
from threshold_visualization import (
    ThresholdExperimentVisualizer,
    generate_experiment_report,
)

warnings.filterwarnings("ignore")


def main():
    """Основная функция запуска эксперимента."""
    print("=" * 80)
    print("ЭКСПЕРИМЕНТ: СРАВНЕНИЕ МЕТОДОВ ВЫБОРА ПОРОГОВ ВЕРОЯТНОСТИ")
    print("=" * 80)
    print(f"Начало: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # 1. Создание и запуск эксперимента
    experiment = ThresholdExperiment(experiment_name="threshold_comparison")

    print("Этап 1: Запуск экспериментов")
    print("-" * 40)
    experiment.run_experiments()

    # 2. Сохранение результатов
    print("\nЭтап 2: Сохранение результатов")
    print("-" * 40)
    results_df = experiment.save_results()

    # 3. Создание визуализаций
    print("\nЭтап 3: Создание визуализаций")
    print("-" * 40)
    visualizer = ThresholdExperimentVisualizer(
        results_df, output_dir=f"results/{experiment.experiment_name}/plots"
    )
    visualizer.create_all_visualizations()

    # 4. Генерация текстового отчета
    print("\nЭтап 4: Генерация отчета")
    print("-" * 40)
    report_path = Path(f"results/{experiment.experiment_name}")
    report_text = generate_experiment_report(results_df, report_path)

    # 5. Вывод краткой сводки
    print("\nКРАТКАЯ СВОДКА РЕЗУЛЬТАТОВ")
    print("=" * 80)

    # Лучшие методы по F1
    best_methods = (
        results_df.groupby("threshold_method")["pseudo_f1"].mean().nlargest(5)
    )
    print("\nТоп-5 методов по среднему F1-Score:")
    for i, (method, f1) in enumerate(best_methods.items(), 1):
        print(f"{i}. {method:<30} F1: {f1:.3f}")

    # Лучшие модели
    best_models = results_df.groupby("model_name")["pseudo_f1"].mean().nlargest(5)
    print("\nТоп-5 моделей по среднему F1-Score:")
    for i, (model, f1) in enumerate(best_models.items(), 1):
        print(f"{i}. {model:<20} F1: {f1:.3f}")

    # Статистика покрытия
    coverage_stats = (
        results_df.groupby("threshold_method")["selection_rate"].mean().describe()
    )
    print(f"\nСтатистика покрытия (доля отобранных примеров):")
    print(f"  • Среднее: {coverage_stats['mean']:.1%}")
    print(f"  • Медиана: {coverage_stats['50%']:.1%}")
    print(f"  • Мин/Макс: {coverage_stats['min']:.1%} - {coverage_stats['max']:.1%}")

    print("\n" + "=" * 80)
    print(f"Завершено: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Результаты сохранены в: results/{experiment.experiment_name}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
