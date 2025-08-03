"""
Модуль для обнаружения выбросов в результатах бенчмаркинга.
"""

from typing import List, Dict, Tuple, Optional, Union
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum

from researches.polaris.pandas_polars_benchmark.src import get_logger


class OutlierMethod(Enum):
    """Методы обнаружения выбросов."""

    IQR = "iqr"
    ZSCORE = "zscore"
    ISOLATION_FOREST = "isolation_forest"
    PERCENTILE = "percentile"


@dataclass
class OutlierResult:
    """Результат обнаружения выбросов."""

    method: str
    outlier_indices: List[int]
    outlier_values: List[float]
    lower_bound: Optional[float]
    upper_bound: Optional[float]
    removed_count: int
    removed_percentage: float
    stats_before: Dict[str, float]
    stats_after: Dict[str, float]


class OutlierDetector:
    """Класс для обнаружения и удаления выбросов."""

    def __init__(self, logger=None):
        """
        Инициализация детектора выбросов.

        Args:
            logger: Логгер для вывода информации
        """
        self.logger = logger or get_logger(__name__)

    def detect_outliers(
        self,
        data: Union[List[float], np.ndarray, pd.Series],
        method: OutlierMethod = OutlierMethod.IQR,
        **kwargs,
    ) -> OutlierResult:
        """
        Обнаружение выбросов в данных.

        Args:
            data: Массив числовых данных
            method: Метод обнаружения выбросов
            **kwargs: Дополнительные параметры для методов

        Returns:
            OutlierResult: Результат обнаружения выбросов
        """
        # Преобразуем в numpy array
        if isinstance(data, list):
            data = np.array(data)
        elif isinstance(data, pd.Series):
            data = data.values

        # Удаляем NaN значения
        clean_data = data[~np.isnan(data)]

        # Считаем статистики до удаления
        stats_before = self._calculate_stats(clean_data)

        # Выбираем метод
        if method == OutlierMethod.IQR:
            result = self._detect_iqr(clean_data, **kwargs)
        elif method == OutlierMethod.ZSCORE:
            result = self._detect_zscore(clean_data, **kwargs)
        elif method == OutlierMethod.PERCENTILE:
            result = self._detect_percentile(clean_data, **kwargs)
        else:
            raise ValueError(f"Неподдерживаемый метод: {method}")

        # Обновляем результат
        result.method = method.value
        result.stats_before = stats_before

        # Считаем статистики после удаления
        clean_indices = [
            i for i in range(len(clean_data)) if i not in result.outlier_indices
        ]
        clean_values = clean_data[clean_indices]
        result.stats_after = self._calculate_stats(clean_values)

        self.logger.info(
            f"Обнаружено {result.removed_count} выбросов "
            f"({result.removed_percentage:.1f}%) методом {method.value}"
        )

        return result

    def _detect_iqr(self, data: np.ndarray, multiplier: float = 1.5) -> OutlierResult:
        """
        Обнаружение выбросов методом IQR (межквартильный размах).

        Args:
            data: Массив данных
            multiplier: Множитель для IQR (обычно 1.5)

        Returns:
            OutlierResult: Результат обнаружения
        """
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1

        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr

        # Находим индексы выбросов
        outlier_mask = (data < lower_bound) | (data > upper_bound)
        outlier_indices = np.where(outlier_mask)[0].tolist()
        outlier_values = data[outlier_mask].tolist()

        return OutlierResult(
            method="iqr",
            outlier_indices=outlier_indices,
            outlier_values=outlier_values,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            removed_count=len(outlier_indices),
            removed_percentage=len(outlier_indices) / len(data) * 100,
            stats_before={},  # Заполнится позже
            stats_after={},  # Заполнится позже
        )

    def _detect_zscore(self, data: np.ndarray, threshold: float = 3.0) -> OutlierResult:
        """
        Обнаружение выбросов методом Z-score.

        Args:
            data: Массив данных
            threshold: Пороговое значение Z-score (обычно 3)

        Returns:
            OutlierResult: Результат обнаружения
        """
        mean = np.mean(data)
        std = np.std(data)

        if std == 0:
            # Если стандартное отклонение 0, нет выбросов
            return OutlierResult(
                method="zscore",
                outlier_indices=[],
                outlier_values=[],
                lower_bound=mean,
                upper_bound=mean,
                removed_count=0,
                removed_percentage=0.0,
                stats_before={},
                stats_after={},
            )

        # Вычисляем Z-scores
        z_scores = np.abs((data - mean) / std)

        # Находим выбросы
        outlier_mask = z_scores > threshold
        outlier_indices = np.where(outlier_mask)[0].tolist()
        outlier_values = data[outlier_mask].tolist()

        # Границы на основе Z-score
        lower_bound = mean - threshold * std
        upper_bound = mean + threshold * std

        return OutlierResult(
            method="zscore",
            outlier_indices=outlier_indices,
            outlier_values=outlier_values,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            removed_count=len(outlier_indices),
            removed_percentage=len(outlier_indices) / len(data) * 100,
            stats_before={},
            stats_after={},
        )

    def _detect_percentile(
        self,
        data: np.ndarray,
        lower_percentile: float = 5.0,
        upper_percentile: float = 95.0,
    ) -> OutlierResult:
        """
        Обнаружение выбросов методом процентилей.

        Args:
            data: Массив данных
            lower_percentile: Нижний процентиль
            upper_percentile: Верхний процентиль

        Returns:
            OutlierResult: Результат обнаружения
        """
        lower_bound = np.percentile(data, lower_percentile)
        upper_bound = np.percentile(data, upper_percentile)

        # Находим выбросы
        outlier_mask = (data < lower_bound) | (data > upper_bound)
        outlier_indices = np.where(outlier_mask)[0].tolist()
        outlier_values = data[outlier_mask].tolist()

        return OutlierResult(
            method="percentile",
            outlier_indices=outlier_indices,
            outlier_values=outlier_values,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            removed_count=len(outlier_indices),
            removed_percentage=len(outlier_indices) / len(data) * 100,
            stats_before={},
            stats_after={},
        )

    def _calculate_stats(self, data: np.ndarray) -> Dict[str, float]:
        """
        Вычисление описательных статистик.

        Args:
            data: Массив данных

        Returns:
            Dict: Словарь со статистиками
        """
        if len(data) == 0:
            return {
                "count": 0,
                "mean": np.nan,
                "std": np.nan,
                "min": np.nan,
                "max": np.nan,
                "median": np.nan,
                "q1": np.nan,
                "q3": np.nan,
                "cv": np.nan,
            }

        mean = np.mean(data)
        std = np.std(data)

        return {
            "count": len(data),
            "mean": mean,
            "std": std,
            "min": np.min(data),
            "max": np.max(data),
            "median": np.median(data),
            "q1": np.percentile(data, 25),
            "q3": np.percentile(data, 75),
            "cv": std / mean if mean != 0 else np.nan,
        }

    def remove_outliers(
        self,
        data: Union[List[float], np.ndarray, pd.Series],
        method: OutlierMethod = OutlierMethod.IQR,
        **kwargs,
    ) -> Tuple[np.ndarray, OutlierResult]:
        """
        Удаление выбросов из данных.

        Args:
            data: Массив данных
            method: Метод обнаружения выбросов
            **kwargs: Дополнительные параметры

        Returns:
            Tuple[np.ndarray, OutlierResult]: Очищенные данные и результат
        """
        # Обнаруживаем выбросы
        result = self.detect_outliers(data, method, **kwargs)

        # Преобразуем в numpy array
        if isinstance(data, list):
            data = np.array(data)
        elif isinstance(data, pd.Series):
            data = data.values

        # Удаляем выбросы
        clean_mask = np.ones(len(data), dtype=bool)
        clean_mask[result.outlier_indices] = False
        clean_data = data[clean_mask]

        return clean_data, result

    def analyze_multiple_runs(
        self,
        runs_data: Dict[str, List[float]],
        method: OutlierMethod = OutlierMethod.IQR,
        **kwargs,
    ) -> Dict[str, Tuple[np.ndarray, OutlierResult]]:
        """
        Анализ выбросов для нескольких серий измерений.

        Args:
            runs_data: Словарь {название: список_измерений}
            method: Метод обнаружения выбросов
            **kwargs: Дополнительные параметры

        Returns:
            Dict: Словарь с очищенными данными и результатами
        """
        results = {}

        for name, data in runs_data.items():
            self.logger.info(f"Анализ выбросов для '{name}'...")
            clean_data, result = self.remove_outliers(data, method, **kwargs)
            results[name] = (clean_data, result)

            # Выводим краткую статистику
            self.logger.info(
                f"  До: mean={result.stats_before['mean']:.4f}, "
                f"std={result.stats_before['std']:.4f}, "
                f"cv={result.stats_before['cv']:.3f}"
            )
            self.logger.info(
                f"  После: mean={result.stats_after['mean']:.4f}, "
                f"std={result.stats_after['std']:.4f}, "
                f"cv={result.stats_after['cv']:.3f}"
            )

        return results


def compare_outlier_methods(
    data: Union[List[float], np.ndarray], methods: Optional[List[OutlierMethod]] = None
) -> pd.DataFrame:
    """
    Сравнение различных методов обнаружения выбросов.

    Args:
        data: Массив данных для анализа
        methods: Список методов для сравнения

    Returns:
        pd.DataFrame: Таблица сравнения методов
    """
    if methods is None:
        methods = [OutlierMethod.IQR, OutlierMethod.ZSCORE, OutlierMethod.PERCENTILE]

    detector = OutlierDetector()
    results = []

    for method in methods:
        result = detector.detect_outliers(data, method)
        results.append(
            {
                "method": method.value,
                "outliers_count": result.removed_count,
                "outliers_percentage": result.removed_percentage,
                "lower_bound": result.lower_bound,
                "upper_bound": result.upper_bound,
                "mean_before": result.stats_before["mean"],
                "std_before": result.stats_before["std"],
                "mean_after": result.stats_after["mean"],
                "std_after": result.stats_after["std"],
                "cv_before": result.stats_before["cv"],
                "cv_after": result.stats_after["cv"],
            }
        )

    return pd.DataFrame(results)
