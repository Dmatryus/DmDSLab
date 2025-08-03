
# Добавляем путь к src для импортов
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

"""
Демонстрация работы движка сравнения результатов бенчмарков.
"""

import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.analysis.comparison_engine import (
    ComparisonEngine,
    ComparisonMetric,
    format_comparison_report
)


def generate_benchmark_data():
    """Генерация тестовых данных бенчмарка."""
    np.random.seed(42)
    
    # Симуляция результатов для разных операций
    results = {
        # Polars значительно быстрее
        "read_csv": {
            "pandas": np.random.normal(1.2, 0.1, 20),
            "polars": np.random.normal(0.3, 0.02, 20)
        },
        
        # Примерно одинаково
        "simple_filter": {
            "pandas": np.random.normal(0.5, 0.05, 15),
            "polars": np.random.normal(0.48, 0.04, 15)
        },
        
        # Pandas быстрее
        "string_operations": {
            "pandas": np.random.normal(0.8, 0.06, 18),
            "polars": np.random.normal(1.1, 0.08, 18)
        },
        
        # Polars умеренно быстрее
        "groupby_agg": {
            "pandas": np.random.normal(2.5, 0.2, 25),
            "polars": np.random.normal(1.8, 0.15, 25)
        },
        
        # Большая вариативность
        "complex_join": {
            "pandas": np.random.normal(5.0, 1.0, 30),
            "polars": np.random.normal(3.5, 0.8, 30)
        }
    }
    
    return results


def demo_single_comparison():
    """Демонстрация сравнения одной операции."""
    print("=== Демонстрация сравнения одной операции ===\n")
    
    # Создание движка
    engine = ComparisonEngine(confidence_level=0.95)
    
    # Генерация данных
    np.random.seed(42)
    pandas_times = np.random.normal(1.2, 0.1, 20)
    polars_times = np.random.normal(0.3, 0.02, 20)
    
    # Сравнение
    result = engine.compare_two_samples(
        baseline=pandas_times,
        comparison=polars_times,
        name="read_csv",
        baseline_library="pandas",
        comparison_library="polars",
        metric=ComparisonMetric.EXECUTION_TIME
    )
    
    # Вывод отчета
    print(format_comparison_report(result))


def demo_multiple_comparisons():
    """Демонстрация сравнения множества операций."""
    print("\n=== Демонстрация сравнения всех операций ===\n")
    
    # Создание движка
    engine = ComparisonEngine()
    
    # Генерация данных
    results = generate_benchmark_data()
    
    # Сравнение всех операций
    matrix = engine.compare_all_operations(
        results,
        baseline_library="pandas",
        comparison_library="polars"
    )
    
    # Общая статистика
    print(f"Общая статистика:")
    print(f"- Всего операций: {matrix.total_operations}")
    print(f"- Значимых различий: {matrix.significant_differences}")
    print(f"- Pandas быстрее: {matrix.baseline_wins}")
    print(f"- Polars быстрее: {matrix.comparison_wins}")
    print(f"- Нет разницы: {matrix.ties}")
    print(f"- Среднее улучшение: {matrix.mean_improvement:.1f}%")
    print(f"- Медианное улучшение: {matrix.median_improvement:.1f}%")
    
    # Детальная статистика
    summary = engine.get_summary_statistics(matrix)
    
    print("\nКатегории улучшений:")
    for category, operations in summary["improvement_categories"].items():
        if operations:
            print(f"- {category}: {operations}")
    
    print("\nТоп улучшений:")
    for name, improvement, speedup in summary["top_improvements"]:
        print(f"- {name}: {improvement:.1f}% ({speedup:.2f}x)")
    
    if summary["top_regressions"]:
        print("\nТоп регрессий:")
        for name, improvement, speedup in summary["top_regressions"]:
            print(f"- {name}: {improvement:.1f}% ({speedup:.2f}x)")


def demo_export_results():
    """Демонстрация экспорта результатов."""
    print("\n=== Демонстрация экспорта результатов ===\n")
    
    # Создание движка и генерация данных
    engine = ComparisonEngine()
    results = generate_benchmark_data()
    
    # Сравнение
    matrix = engine.compare_all_operations(results)
    
    # Экспорт в JSON
    output_dir = Path("demo_output")
    output_dir.mkdir(exist_ok=True)
    
    json_path = output_dir / "comparison_results.json"
    engine.export_results(matrix, json_path, format="json")
    print(f"Результаты экспортированы в JSON: {json_path}")
    
    # Экспорт в CSV
    csv_path = output_dir / "comparison_results.csv"
    engine.export_results(matrix, csv_path, format="csv")
    print(f"Результаты экспортированы в CSV: {csv_path}")
    
    # Показать пример содержимого JSON
    import json
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    print("\nПример структуры JSON:")
    print(f"- Метаданные: {list(data['metadata'].keys())}")
    print(f"- Сводка: {list(data['summary'].keys())}")
    print(f"- Детальные результаты для {len(data['detailed_results'])} операций")


def demo_statistical_tests():
    """Демонстрация различных статистических тестов."""
    print("\n=== Демонстрация статистических тестов ===\n")
    
    engine = ComparisonEngine()
    
    # Различные сценарии
    scenarios = {
        "Явное различие": {
            "baseline": np.random.normal(1.0, 0.1, 50),
            "comparison": np.random.normal(0.5, 0.05, 50)
        },
        "Нет различия": {
            "baseline": np.random.normal(1.0, 0.2, 30),
            "comparison": np.random.normal(1.0, 0.2, 30)
        },
        "Малый размер эффекта": {
            "baseline": np.random.normal(1.0, 0.1, 100),
            "comparison": np.random.normal(0.95, 0.1, 100)
        }
    }
    
    for scenario_name, data in scenarios.items():
        result = engine.compare_two_samples(
            baseline=data["baseline"],
            comparison=data["comparison"],
            name=scenario_name,
            baseline_library="A",
            comparison_library="B"
        )
        
        print(f"\n{scenario_name}:")
        print(f"- P-value (t-test): {result.p_value:.4f}")
        print(f"- P-value (Mann-Whitney): {result.mann_whitney_p:.4f}")
        print(f"- Cohen's d: {result.cohens_d:.3f}")
        print(f"- Значимость: {result.significance_level.value}")
        print(f"- Вывод: {result.winner}")


if __name__ == "__main__":
    # Запуск всех демонстраций
    demo_single_comparison()
    demo_multiple_comparisons()
    demo_export_results()
    demo_statistical_tests()
