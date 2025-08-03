"""
comparison_engine.py - Модуль для сравнения результатов бенчмарков между библиотеками.

Предоставляет инструменты для:
- Парного сравнения результатов
- Статистических тестов значимости
- Расчета относительного улучшения
- Построения матрицы сравнений
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import numpy as np
from scipy import stats
import pandas as pd
from pathlib import Path
import json


class ComparisonMetric(Enum):
    """Метрики для сравнения."""
    EXECUTION_TIME = "execution_time"
    MEMORY_PEAK = "memory_peak"
    MEMORY_MEAN = "memory_mean"
    CPU_USAGE = "cpu_usage"


class SignificanceLevel(Enum):
    """Уровни статистической значимости."""
    NOT_SIGNIFICANT = "not_significant"
    WEAKLY_SIGNIFICANT = "weakly_significant"  # p < 0.1
    SIGNIFICANT = "significant"                # p < 0.05
    HIGHLY_SIGNIFICANT = "highly_significant"  # p < 0.01


@dataclass
class ComparisonResult:
    """Результат сравнения двух серий измерений."""
    name: str
    baseline_library: str
    comparison_library: str
    metric: ComparisonMetric
    
    # Статистики для baseline
    baseline_mean: float
    baseline_std: float
    baseline_median: float
    baseline_count: int
    
    # Статистики для comparison
    comparison_mean: float
    comparison_std: float
    comparison_median: float
    comparison_count: int
    
    # Результаты сравнения
    relative_improvement: float  # Процент улучшения (отрицательный = ухудшение)
    speedup_factor: float       # Во сколько раз быстрее
    
    # Статистическая значимость
    t_statistic: float
    p_value: float
    significance_level: SignificanceLevel
    confidence_interval_lower: float
    confidence_interval_upper: float
    
    # Дополнительные тесты
    mann_whitney_u: Optional[float] = None
    mann_whitney_p: Optional[float] = None
    cohens_d: Optional[float] = None  # Размер эффекта
    
    # Выводы
    is_significant: bool = field(init=False)
    winner: str = field(init=False)
    
    def __post_init__(self):
        """Вычисление производных полей."""
        self.is_significant = self.significance_level in [
            SignificanceLevel.SIGNIFICANT,
            SignificanceLevel.HIGHLY_SIGNIFICANT
        ]
        
        # Определяем победителя (меньше время = лучше)
        if not self.is_significant:
            self.winner = "tie"
        elif self.comparison_mean < self.baseline_mean:
            self.winner = self.comparison_library
        else:
            self.winner = self.baseline_library


@dataclass
class ComparisonMatrix:
    """Матрица сравнений для всех операций."""
    baseline_library: str
    comparison_library: str
    metric: ComparisonMetric
    results: Dict[str, ComparisonResult]
    
    # Агрегированные статистики
    total_operations: int = field(init=False)
    significant_differences: int = field(init=False)
    baseline_wins: int = field(init=False)
    comparison_wins: int = field(init=False)
    ties: int = field(init=False)
    
    # Средние улучшения
    mean_improvement: float = field(init=False)
    median_improvement: float = field(init=False)
    
    def __post_init__(self):
        """Вычисление агрегированных статистик."""
        self.total_operations = len(self.results)
        self.significant_differences = sum(
            1 for r in self.results.values() if r.is_significant
        )
        self.baseline_wins = sum(
            1 for r in self.results.values() if r.winner == self.baseline_library
        )
        self.comparison_wins = sum(
            1 for r in self.results.values() if r.winner == self.comparison_library
        )
        self.ties = sum(
            1 for r in self.results.values() if r.winner == "tie"
        )
        
        improvements = [r.relative_improvement for r in self.results.values()]
        self.mean_improvement = np.mean(improvements) if improvements else 0
        self.median_improvement = np.median(improvements) if improvements else 0


class ComparisonEngine:
    """Движок для сравнения результатов бенчмарков."""
    
    def __init__(self,
                 confidence_level: float = 0.95,
                 min_samples: int = 5):
        """
        Инициализация движка сравнения.
        
        Args:
            confidence_level: Уровень доверия для интервалов
            min_samples: Минимальное количество измерений для сравнения
        """
        self.confidence_level = confidence_level
        self.min_samples = min_samples
        self.alpha = 1 - confidence_level
    
    def compare_two_samples(self,
                           baseline: np.ndarray,
                           comparison: np.ndarray,
                           name: str,
                           baseline_library: str,
                           comparison_library: str,
                           metric: ComparisonMetric = ComparisonMetric.EXECUTION_TIME
                           ) -> ComparisonResult:
        """
        Сравнение двух выборок измерений.
        
        Args:
            baseline: Базовые измерения
            comparison: Измерения для сравнения
            name: Название операции
            baseline_library: Название базовой библиотеки
            comparison_library: Название сравниваемой библиотеки
            metric: Метрика сравнения
        
        Returns:
            ComparisonResult: Результат сравнения
        """
        # Проверка минимального количества измерений
        if len(baseline) < self.min_samples or len(comparison) < self.min_samples:
            raise ValueError(
                f"Недостаточно измерений. Требуется минимум {self.min_samples}, "
                f"получено: baseline={len(baseline)}, comparison={len(comparison)}"
            )
        
        # Базовые статистики
        baseline_mean = np.mean(baseline)
        baseline_std = np.std(baseline, ddof=1)
        baseline_median = np.median(baseline)
        
        comparison_mean = np.mean(comparison)
        comparison_std = np.std(comparison, ddof=1)
        comparison_median = np.median(comparison)
        
        # Относительное улучшение (для времени: отрицательное = лучше)
        relative_improvement = ((baseline_mean - comparison_mean) / baseline_mean) * 100
        speedup_factor = baseline_mean / comparison_mean if comparison_mean > 0 else float('inf')
        
        # T-тест Уэлча (для выборок с разными дисперсиями)
        t_stat, p_value = stats.ttest_ind(baseline, comparison, equal_var=False)
        
        # Доверительный интервал для разности средних
        diff_mean = baseline_mean - comparison_mean
        diff_se = np.sqrt(
            (baseline_std**2 / len(baseline)) + 
            (comparison_std**2 / len(comparison))
        )
        t_critical = stats.t.ppf(1 - self.alpha/2, len(baseline) + len(comparison) - 2)
        ci_lower = diff_mean - t_critical * diff_se
        ci_upper = diff_mean + t_critical * diff_se
        
        # Непараметрический тест Манна-Уитни
        mann_u, mann_p = stats.mannwhitneyu(
            baseline, comparison, alternative='two-sided'
        )
        
        # Размер эффекта (Cohen's d)
        pooled_std = np.sqrt(
            ((len(baseline) - 1) * baseline_std**2 + 
             (len(comparison) - 1) * comparison_std**2) /
            (len(baseline) + len(comparison) - 2)
        )
        cohens_d = (baseline_mean - comparison_mean) / pooled_std if pooled_std > 0 else 0
        
        # Определение уровня значимости
        if p_value < 0.01:
            significance = SignificanceLevel.HIGHLY_SIGNIFICANT
        elif p_value < 0.05:
            significance = SignificanceLevel.SIGNIFICANT
        elif p_value < 0.1:
            significance = SignificanceLevel.WEAKLY_SIGNIFICANT
        else:
            significance = SignificanceLevel.NOT_SIGNIFICANT
        
        return ComparisonResult(
            name=name,
            baseline_library=baseline_library,
            comparison_library=comparison_library,
            metric=metric,
            baseline_mean=baseline_mean,
            baseline_std=baseline_std,
            baseline_median=baseline_median,
            baseline_count=len(baseline),
            comparison_mean=comparison_mean,
            comparison_std=comparison_std,
            comparison_median=comparison_median,
            comparison_count=len(comparison),
            relative_improvement=relative_improvement,
            speedup_factor=speedup_factor,
            t_statistic=t_stat,
            p_value=p_value,
            significance_level=significance,
            confidence_interval_lower=ci_lower,
            confidence_interval_upper=ci_upper,
            mann_whitney_u=mann_u,
            mann_whitney_p=mann_p,
            cohens_d=cohens_d
        )
    
    def compare_all_operations(self,
                              results: Dict[str, Dict[str, np.ndarray]],
                              baseline_library: str = "pandas",
                              comparison_library: str = "polars",
                              metric: ComparisonMetric = ComparisonMetric.EXECUTION_TIME
                              ) -> ComparisonMatrix:
        """
        Сравнение всех операций между двумя библиотеками.
        
        Args:
            results: Словарь {operation: {library: measurements}}
            baseline_library: Базовая библиотека
            comparison_library: Сравниваемая библиотека
            metric: Метрика для сравнения
        
        Returns:
            ComparisonMatrix: Матрица сравнений
        """
        comparison_results = {}
        
        for operation, libraries_data in results.items():
            if baseline_library not in libraries_data or comparison_library not in libraries_data:
                continue
            
            baseline_data = np.array(libraries_data[baseline_library])
            comparison_data = np.array(libraries_data[comparison_library])
            
            try:
                result = self.compare_two_samples(
                    baseline_data,
                    comparison_data,
                    operation,
                    baseline_library,
                    comparison_library,
                    metric
                )
                comparison_results[operation] = result
            except ValueError as e:
                print(f"Пропуск операции {operation}: {e}")
                continue
        
        return ComparisonMatrix(
            baseline_library=baseline_library,
            comparison_library=comparison_library,
            metric=metric,
            results=comparison_results
        )
    
    def get_summary_statistics(self, matrix: ComparisonMatrix) -> Dict[str, Any]:
        """
        Получение сводной статистики по матрице сравнений.
        
        Args:
            matrix: Матрица сравнений
        
        Returns:
            Dict: Сводная статистика
        """
        # Группировка по категориям улучшения
        improvements = {
            "major_improvement": [],     # > 50%
            "moderate_improvement": [],   # 20-50%
            "minor_improvement": [],      # 5-20%
            "negligible": [],            # -5 to 5%
            "minor_regression": [],      # -20 to -5%
            "moderate_regression": [],   # -50 to -20%
            "major_regression": []       # < -50%
        }
        
        for name, result in matrix.results.items():
            imp = result.relative_improvement
            
            if imp > 50:
                improvements["major_improvement"].append(name)
            elif imp > 20:
                improvements["moderate_improvement"].append(name)
            elif imp > 5:
                improvements["minor_improvement"].append(name)
            elif imp > -5:
                improvements["negligible"].append(name)
            elif imp > -20:
                improvements["minor_regression"].append(name)
            elif imp > -50:
                improvements["moderate_regression"].append(name)
            else:
                improvements["major_regression"].append(name)
        
        # Топ улучшений и регрессий
        sorted_results = sorted(
            matrix.results.items(),
            key=lambda x: x[1].relative_improvement,
            reverse=True
        )
        
        top_improvements = [
            (name, result.relative_improvement, result.speedup_factor)
            for name, result in sorted_results[:5]
            if result.relative_improvement > 0
        ]
        
        top_regressions = [
            (name, result.relative_improvement, result.speedup_factor)
            for name, result in sorted_results[-5:]
            if result.relative_improvement < 0
        ]
        
        return {
            "total_operations": matrix.total_operations,
            "significant_differences": matrix.significant_differences,
            "baseline_wins": matrix.baseline_wins,
            "comparison_wins": matrix.comparison_wins,
            "ties": matrix.ties,
            "mean_improvement": matrix.mean_improvement,
            "median_improvement": matrix.median_improvement,
            "improvement_categories": improvements,
            "top_improvements": top_improvements,
            "top_regressions": top_regressions,
            "significance_summary": {
                "highly_significant": sum(
                    1 for r in matrix.results.values()
                    if r.significance_level == SignificanceLevel.HIGHLY_SIGNIFICANT
                ),
                "significant": sum(
                    1 for r in matrix.results.values()
                    if r.significance_level == SignificanceLevel.SIGNIFICANT
                ),
                "weakly_significant": sum(
                    1 for r in matrix.results.values()
                    if r.significance_level == SignificanceLevel.WEAKLY_SIGNIFICANT
                ),
                "not_significant": sum(
                    1 for r in matrix.results.values()
                    if r.significance_level == SignificanceLevel.NOT_SIGNIFICANT
                )
            }
        }
    
    def export_results(self,
                      matrix: ComparisonMatrix,
                      output_path: Path,
                      format: str = "json") -> None:
        """
        Экспорт результатов сравнения.
        
        Args:
            matrix: Матрица сравнений
            output_path: Путь для сохранения
            format: Формат экспорта (json, csv)
        """
        if format == "json":
            # Подготовка данных для JSON
            data = {
                "metadata": {
                    "baseline_library": matrix.baseline_library,
                    "comparison_library": matrix.comparison_library,
                    "metric": matrix.metric.value,
                    "total_operations": matrix.total_operations,
                    "mean_improvement": matrix.mean_improvement,
                    "median_improvement": matrix.median_improvement
                },
                "summary": self.get_summary_statistics(matrix),
                "detailed_results": {
                    name: {
                        "baseline_mean": r.baseline_mean,
                        "comparison_mean": r.comparison_mean,
                        "relative_improvement": r.relative_improvement,
                        "speedup_factor": r.speedup_factor,
                        "p_value": r.p_value,
                        "significance": r.significance_level.value,
                        "cohens_d": r.cohens_d,
                        "winner": r.winner
                    }
                    for name, r in matrix.results.items()
                }
            }
            
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
        
        elif format == "csv":
            # Создание DataFrame для CSV
            rows = []
            for name, r in matrix.results.items():
                rows.append({
                    "operation": name,
                    "baseline_mean": r.baseline_mean,
                    "comparison_mean": r.comparison_mean,
                    "relative_improvement_%": r.relative_improvement,
                    "speedup_factor": r.speedup_factor,
                    "p_value": r.p_value,
                    "significance": r.significance_level.value,
                    "cohens_d": r.cohens_d,
                    "winner": r.winner
                })
            
            df = pd.DataFrame(rows)
            df.to_csv(output_path, index=False)


def format_comparison_report(result: ComparisonResult) -> str:
    """
    Форматирование отчета по одному сравнению.
    
    Args:
        result: Результат сравнения
    
    Returns:
        str: Отформатированный отчет
    """
    report = f"""
Сравнение: {result.name}
{'=' * 50}
Baseline: {result.baseline_library}
Comparison: {result.comparison_library}

Статистики производительности:
- {result.baseline_library}: {result.baseline_mean:.3f} ± {result.baseline_std:.3f} (медиана: {result.baseline_median:.3f})
- {result.comparison_library}: {result.comparison_mean:.3f} ± {result.comparison_std:.3f} (медиана: {result.comparison_median:.3f})

Улучшение производительности:
- Относительное улучшение: {result.relative_improvement:+.1f}%
- Фактор ускорения: {result.speedup_factor:.2f}x
- Доверительный интервал: [{result.confidence_interval_lower:.3f}, {result.confidence_interval_upper:.3f}]

Статистическая значимость:
- T-статистика: {result.t_statistic:.3f}
- P-value: {result.p_value:.4f}
- Уровень значимости: {result.significance_level.value}
- Cohen's d (размер эффекта): {result.cohens_d:.3f}

Непараметрический тест:
- Mann-Whitney U: {result.mann_whitney_u:.1f}
- P-value: {result.mann_whitney_p:.4f}

Вывод: {result.winner.upper()} {"значимо лучше" if result.is_significant else "нет значимой разницы"}
"""
    return report
