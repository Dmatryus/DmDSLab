"""
Модуль с реализацией методов выбора порогов вероятности для псевдо-разметки.

Содержит реализации всех методов из руководства по выбору порогов.
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional, List, Union
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    f1_score,
    accuracy_score,
    confusion_matrix,
)
from dataclasses import dataclass, field
import warnings


@dataclass
class ThresholdResult:
    """Результат работы метода выбора порога."""

    threshold: float
    score: float
    method_name: str
    additional_info: Dict[str, Any] = field(default_factory=dict)
    confident_mask: Optional[np.ndarray] = None


class BaseThresholdSelector:
    """Базовый класс для методов выбора порога."""

    def __init__(self, name: str):
        self.name = name

    def select_threshold(
        self, y_true: Optional[np.ndarray], y_proba: np.ndarray, **kwargs
    ) -> ThresholdResult:
        """Выбор порога. Должен быть реализован в наследниках."""
        raise NotImplementedError


# ============== Методы для бинарной классификации ==============


class OptimalF1Threshold(BaseThresholdSelector):
    """Оптимизация порога по F1-мере."""

    def __init__(self):
        super().__init__("Optimal F1")

    def select_threshold(
        self, y_true: np.ndarray, y_proba: np.ndarray, **kwargs
    ) -> ThresholdResult:
        """
        Находит порог, максимизирующий F1-меру.

        Args:
            y_true: Истинные метки (требуется для этого метода)
            y_proba: Вероятности положительного класса

        Returns:
            ThresholdResult с оптимальным порогом
        """
        if y_true is None:
            raise ValueError("OptimalF1Threshold требует истинные метки")

        # Для бинарной классификации берем вероятность положительного класса
        if len(y_proba.shape) > 1 and y_proba.shape[1] == 2:
            y_proba = y_proba[:, 1]

        precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)

        # Вычисляем F1 для каждого порога
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)

        # Находим оптимальный порог
        optimal_idx = np.argmax(f1_scores[:-1])  # Последний элемент - всегда 1.0
        optimal_threshold = thresholds[optimal_idx]
        optimal_f1 = f1_scores[optimal_idx]

        # Создаем маску уверенных примеров
        confident_mask = y_proba >= optimal_threshold

        return ThresholdResult(
            threshold=optimal_threshold,
            score=optimal_f1,
            method_name=self.name,
            additional_info={
                "precision": precisions[optimal_idx],
                "recall": recalls[optimal_idx],
            },
            confident_mask=confident_mask,
        )


class YoudenJStatistic(BaseThresholdSelector):
    """Статистика Юдена (максимизация TPR + TNR - 1)."""

    def __init__(self):
        super().__init__("Youden J Statistic")

    def select_threshold(
        self, y_true: np.ndarray, y_proba: np.ndarray, **kwargs
    ) -> ThresholdResult:
        """
        Находит порог по статистике Юдена.

        Args:
            y_true: Истинные метки
            y_proba: Вероятности положительного класса

        Returns:
            ThresholdResult с оптимальным порогом
        """
        if y_true is None:
            raise ValueError("YoudenJStatistic требует истинные метки")

        # Для бинарной классификации
        if len(y_proba.shape) > 1 and y_proba.shape[1] == 2:
            y_proba = y_proba[:, 1]

        fpr, tpr, thresholds = roc_curve(y_true, y_proba)

        # J = Sensitivity + Specificity - 1 = TPR - FPR
        j_scores = tpr - fpr

        # Находим оптимальный порог
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]
        optimal_j = j_scores[optimal_idx]

        confident_mask = y_proba >= optimal_threshold

        return ThresholdResult(
            threshold=optimal_threshold,
            score=optimal_j,
            method_name=self.name,
            additional_info={
                "sensitivity": tpr[optimal_idx],
                "specificity": 1 - fpr[optimal_idx],
                "tpr": tpr[optimal_idx],
                "fpr": fpr[optimal_idx],
            },
            confident_mask=confident_mask,
        )


class CostSensitiveThreshold(BaseThresholdSelector):
    """Учет стоимости ошибок при выборе порога."""

    def __init__(self, cost_fp: float = 1.0, cost_fn: float = 1.0):
        super().__init__("Cost Sensitive")
        self.cost_fp = cost_fp
        self.cost_fn = cost_fn

    def select_threshold(
        self, y_true: np.ndarray, y_proba: np.ndarray, **kwargs
    ) -> ThresholdResult:
        """
        Находит порог, минимизирующий общую стоимость ошибок.

        Args:
            y_true: Истинные метки
            y_proba: Вероятности положительного класса

        Returns:
            ThresholdResult с оптимальным порогом
        """
        if y_true is None:
            raise ValueError("CostSensitiveThreshold требует истинные метки")

        # Для бинарной классификации
        if len(y_proba.shape) > 1 and y_proba.shape[1] == 2:
            y_proba = y_proba[:, 1]

        thresholds = np.unique(np.sort(y_proba))
        costs = []

        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)

            # Подсчет ошибок
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

            # Общая стоимость
            total_cost = fp * self.cost_fp + fn * self.cost_fn
            costs.append(total_cost)

        # Находим порог с минимальной стоимостью
        optimal_idx = np.argmin(costs)
        optimal_threshold = thresholds[optimal_idx]
        optimal_cost = costs[optimal_idx]

        confident_mask = y_proba >= optimal_threshold

        # Вычисляем экономию по сравнению с порогом 0.5
        y_pred_baseline = (y_proba >= 0.5).astype(int)
        tn_b, fp_b, fn_b, tp_b = confusion_matrix(y_true, y_pred_baseline).ravel()
        baseline_cost = fp_b * self.cost_fp + fn_b * self.cost_fn
        savings = baseline_cost - optimal_cost

        return ThresholdResult(
            threshold=optimal_threshold,
            score=-optimal_cost,  # Отрицательная стоимость для консистентности
            method_name=self.name,
            additional_info={
                "total_cost": optimal_cost,
                "baseline_cost": baseline_cost,
                "savings": savings,
                "savings_pct": (savings / baseline_cost) * 100 if baseline_cost > 0 else 0,
                "cost_fp": self.cost_fp,
                "cost_fn": self.cost_fn,
            },
            confident_mask=confident_mask,
        )


# ============== Эвристические методы ==============


class PercentileThreshold(BaseThresholdSelector):
    """Процентильный метод выбора порога."""

    def __init__(self, percentile: float = 80):
        super().__init__(f"Percentile {percentile}%")
        self.percentile = percentile

    def select_threshold(
        self, y_true: Optional[np.ndarray], y_proba: np.ndarray, **kwargs
    ) -> ThresholdResult:
        """
        Выбирает порог на заданном процентиле распределения вероятностей.

        Args:
            y_true: Не используется (метод не требует меток)
            y_proba: Вероятности (может быть матрицей для многоклассовой задачи)

        Returns:
            ThresholdResult с порогом на заданном процентиле
        """
        # Для многоклассовой задачи берем максимальную вероятность
        if len(y_proba.shape) > 1 and y_proba.shape[1] > 2:
            max_proba = np.max(y_proba, axis=1)
        elif len(y_proba.shape) > 1 and y_proba.shape[1] == 2:
            max_proba = y_proba[:, 1]
        else:
            max_proba = y_proba

        threshold = np.percentile(max_proba, self.percentile)
        confident_mask = max_proba >= threshold

        return ThresholdResult(
            threshold=threshold,
            score=np.sum(confident_mask) / len(confident_mask),  # Доля отобранных
            method_name=self.name,
            additional_info={
                "n_selected": np.sum(confident_mask),
                "n_total": len(confident_mask),
                "selection_rate": np.mean(confident_mask),
            },
            confident_mask=confident_mask,
        )


class FixedThreshold(BaseThresholdSelector):
    """Фиксированный порог."""

    def __init__(self, threshold: float = 0.9):
        super().__init__(f"Fixed {threshold}")
        self.threshold = threshold

    def select_threshold(
        self, y_true: Optional[np.ndarray], y_proba: np.ndarray, **kwargs
    ) -> ThresholdResult:
        """Возвращает фиксированный порог."""
        # Для многоклассовой задачи
        if len(y_proba.shape) > 1 and y_proba.shape[1] > 2:
            max_proba = np.max(y_proba, axis=1)
        elif len(y_proba.shape) > 1 and y_proba.shape[1] == 2:
            max_proba = y_proba[:, 1]
        else:
            max_proba = y_proba

        confident_mask = max_proba >= self.threshold

        return ThresholdResult(
            threshold=self.threshold,
            score=np.mean(confident_mask),
            method_name=self.name,
            additional_info={
                "n_selected": np.sum(confident_mask),
                "n_total": len(confident_mask),
            },
            confident_mask=confident_mask,
        )


# ============== Методы для многоклассовой классификации ==============


class EntropyThreshold(BaseThresholdSelector):
    """Энтропийный метод для многоклассовой классификации."""

    def __init__(self, max_entropy: float = 1.0):
        super().__init__(f"Entropy < {max_entropy}")
        self.max_entropy = max_entropy

    def calculate_entropy(self, proba_matrix: np.ndarray) -> np.ndarray:
        """Вычисление энтропии для каждого примера."""
        # Избегаем log(0)
        proba_matrix = np.clip(proba_matrix, 1e-10, 1)
        entropy = -np.sum(proba_matrix * np.log(proba_matrix), axis=1)
        return entropy

    def select_threshold(
        self, y_true: Optional[np.ndarray], y_proba: np.ndarray, **kwargs
    ) -> ThresholdResult:
        """
        Отбирает примеры с энтропией ниже порога.

        Args:
            y_true: Не используется
            y_proba: Матрица вероятностей для всех классов

        Returns:
            ThresholdResult
        """
        if len(y_proba.shape) == 1 or y_proba.shape[1] <= 2:
            warnings.warn(
                "EntropyThreshold предназначен для многоклассовой классификации"
            )

        entropy = self.calculate_entropy(y_proba)
        confident_mask = entropy < self.max_entropy

        return ThresholdResult(
            threshold=self.max_entropy,
            score=np.mean(confident_mask),
            method_name=self.name,
            additional_info={
                "mean_entropy": np.mean(entropy),
                "median_entropy": np.median(entropy),
                "n_selected": np.sum(confident_mask),
                "selection_rate": np.mean(confident_mask),
            },
            confident_mask=confident_mask,
        )


class MarginThreshold(BaseThresholdSelector):
    """Метод отрыва (margin) для многоклассовой классификации."""

    def __init__(self, min_margin: float = 0.2):
        super().__init__(f"Margin > {min_margin}")
        self.min_margin = min_margin

    def calculate_margins(
        self, proba_matrix: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Вычисление margin для каждого примера."""
        # Сортируем вероятности в убывающем порядке
        sorted_proba = np.sort(proba_matrix, axis=1)[:, ::-1]

        # Margin = разность между top-2 вероятностями
        margins = sorted_proba[:, 0] - sorted_proba[:, 1]

        top1_confidence = sorted_proba[:, 0]
        top2_confidence = sorted_proba[:, 1]

        return margins, top1_confidence, top2_confidence

    def select_threshold(
        self, y_true: Optional[np.ndarray], y_proba: np.ndarray, **kwargs
    ) -> ThresholdResult:
        """
        Отбирает примеры с margin выше порога.

        Args:
            y_true: Не используется
            y_proba: Матрица вероятностей для всех классов

        Returns:
            ThresholdResult
        """
        if len(y_proba.shape) == 1 or y_proba.shape[1] <= 2:
            warnings.warn("MarginThreshold предназначен для многоклассовой классификации")

        margins, top1_conf, top2_conf = self.calculate_margins(y_proba)
        confident_mask = margins >= self.min_margin

        return ThresholdResult(
            threshold=self.min_margin,
            score=np.mean(confident_mask),
            method_name=self.name,
            additional_info={
                "mean_margin": np.mean(margins),
                "median_margin": np.median(margins),
                "mean_top1_conf": np.mean(top1_conf),
                "mean_top2_conf": np.mean(top2_conf),
                "n_selected": np.sum(confident_mask),
                "selection_rate": np.mean(confident_mask),
            },
            confident_mask=confident_mask,
        )


# ============== Адаптивные методы ==============


class AdaptiveThreshold(BaseThresholdSelector):
    """Самоадаптивный порог с экспоненциальным скользящим средним."""

    def __init__(self, initial_threshold: float = 0.9, momentum: float = 0.999):
        super().__init__("Adaptive EMA")
        self.threshold = initial_threshold
        self.momentum = momentum
        self.history = []

    def select_threshold(
        self, y_true: Optional[np.ndarray], y_proba: np.ndarray, **kwargs
    ) -> ThresholdResult:
        """
        Адаптивно обновляет порог на основе текущих предсказаний.

        Args:
            y_true: Не используется
            y_proba: Вероятности

        Returns:
            ThresholdResult
        """
        # Для многоклассовой задачи
        if len(y_proba.shape) > 1 and y_proba.shape[1] > 2:
            max_proba = np.max(y_proba, axis=1)
        elif len(y_proba.shape) > 1 and y_proba.shape[1] == 2:
            max_proba = y_proba[:, 1]
        else:
            max_proba = y_proba

        # Обновляем порог через EMA
        current_mean = np.mean(max_proba)
        if len(self.history) == 0:
            self.threshold = current_mean
        else:
            self.threshold = self.momentum * self.threshold + (1 - self.momentum) * current_mean

        self.history.append(self.threshold)
        confident_mask = max_proba >= self.threshold

        return ThresholdResult(
            threshold=self.threshold,
            score=np.mean(confident_mask),
            method_name=self.name,
            additional_info={
                "current_mean_conf": current_mean,
                "n_selected": np.sum(confident_mask),
                "history_length": len(self.history),
                "threshold_trend": "increasing"
                if len(self.history) > 1 and self.history[-1] > self.history[0]
                else "decreasing",
            },
            confident_mask=confident_mask,
        )


# ============== Фабрика методов ==============


class ThresholdMethodFactory:
    """Фабрика для создания методов выбора порога."""

    @staticmethod
    def get_binary_methods() -> List[BaseThresholdSelector]:
        """Возвращает методы для бинарной классификации."""
        return [
            OptimalF1Threshold(),
            YoudenJStatistic(),
            CostSensitiveThreshold(cost_fp=1, cost_fn=2),
            PercentileThreshold(80),
            PercentileThreshold(90),
            FixedThreshold(0.7),
            FixedThreshold(0.9),
            FixedThreshold(0.95),
        ]

    @staticmethod
    def get_multiclass_methods() -> List[BaseThresholdSelector]:
        """Возвращает методы для многоклассовой классификации."""
        return [
            EntropyThreshold(0.5),
            EntropyThreshold(1.0),
            MarginThreshold(0.1),
            MarginThreshold(0.2),
            MarginThreshold(0.3),
            PercentileThreshold(70),
            PercentileThreshold(80),
            PercentileThreshold(90),
        ]

    @staticmethod
    def get_universal_methods() -> List[BaseThresholdSelector]:
        """Возвращает универсальные методы."""
        return [
            PercentileThreshold(percentile=p) for p in [70, 80, 85, 90, 95]
        ] + [
            FixedThreshold(threshold=t) for t in [0.6, 0.7, 0.8, 0.9, 0.95]
        ] + [
            AdaptiveThreshold(initial_threshold=0.9, momentum=0.999)
        ]

    @staticmethod
    def get_all_methods() -> Dict[str, List[BaseThresholdSelector]]:
        """Возвращает все методы по категориям."""
        return {
            "binary": ThresholdMethodFactory.get_binary_methods(),
            "multiclass": ThresholdMethodFactory.get_multiclass_methods(),
            "universal": ThresholdMethodFactory.get_universal_methods(),
        }
