"""
dmdslab/threshold_selection.py
==============================

Унифицированный модуль для выбора порогов в задачах псевдо-разметки.
Поддерживает как бинарную, так и многоклассовую классификацию.

Author: Dmatryus Detry
License: Apache 2.0
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import numpy as np
from scipy.stats import entropy
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)


class TaskType(Enum):
    """Тип задачи классификации"""

    BINARY = "binary"
    MULTICLASS = "multiclass"
    AUTO = "auto"


@dataclass
class ThresholdResult:
    """
    Универсальный результат применения порога.

    Attributes:
        method_name: Название метода
        task_type: Тип задачи (binary/multiclass)
        threshold: Значение порога (для бинарной) или dict с информацией (для многоклассовой)
        selected_indices: Индексы отобранных примеров
        selection_ratio: Доля отобранных примеров
        confidence_scores: Оценки уверенности для отобранных примеров
        predicted_labels: Предсказанные метки (классы)
        metrics: Словарь с метриками качества
        metadata: Дополнительная информация от метода
    """

    method_name: str
    task_type: TaskType
    threshold: Union[float, Dict[str, Any]]
    selected_indices: np.ndarray
    selection_ratio: float
    confidence_scores: np.ndarray
    predicted_labels: Optional[np.ndarray] = None
    metrics: Dict[str, float] = None
    metadata: Dict[str, Any] = None


class BaseThresholdSelector(ABC):
    """Базовый класс для всех методов выбора порога"""

    def __init__(self, name: str, supported_tasks: List[TaskType] = None):
        self.name = name
        self.supported_tasks = supported_tasks or [TaskType.BINARY, TaskType.MULTICLASS]

    def apply(
        self,
        probabilities: np.ndarray,
        y_true: Optional[np.ndarray] = None,
        task_type: TaskType = TaskType.AUTO,
        **kwargs,
    ) -> ThresholdResult:
        """
        Применение метода выбора порога.

        Args:
            probabilities: Вероятности. Для бинарной классификации - 1D массив P(y=1).
                          Для многоклассовой - 2D массив [n_samples, n_classes].
            y_true: Истинные метки (для методов, требующих валидацию)
            task_type: Тип задачи (auto определяет автоматически)
            **kwargs: Дополнительные параметры метода

        Returns:
            ThresholdResult с информацией об отборе
        """
        # Определяем тип задачи
        if task_type == TaskType.AUTO:
            task_type = self._detect_task_type(probabilities)

        # Проверяем поддержку
        if task_type not in self.supported_tasks:
            raise ValueError(
                f"Method '{self.name}' does not support {task_type.value} classification. "
                f"Supported types: {[t.value for t in self.supported_tasks]}"
            )

        # Вызываем соответствующий метод
        if task_type == TaskType.BINARY:
            return self._apply_binary(probabilities, y_true, **kwargs)
        else:
            return self._apply_multiclass(probabilities, y_true, **kwargs)

    def _detect_task_type(self, probabilities: np.ndarray) -> TaskType:
        """Автоматическое определение типа задачи по форме данных"""
        if probabilities.ndim == 1 or (
            probabilities.ndim == 2 and probabilities.shape[1] == 1
        ):
            return TaskType.BINARY
        elif probabilities.ndim == 2 and probabilities.shape[1] > 1:
            return TaskType.MULTICLASS
        else:
            raise ValueError(
                f"Cannot determine task type from shape {probabilities.shape}"
            )

    @abstractmethod
    def _apply_binary(
        self, probabilities: np.ndarray, y_true: Optional[np.ndarray], **kwargs
    ) -> ThresholdResult:
        """Применение для бинарной классификации"""
        pass

    @abstractmethod
    def _apply_multiclass(
        self, probabilities: np.ndarray, y_true: Optional[np.ndarray], **kwargs
    ) -> ThresholdResult:
        """Применение для многоклассовой классификации"""
        pass

    @property
    def requires_labels(self) -> bool:
        """Требует ли метод истинные метки"""
        return False

    def _compute_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, task_type: TaskType
    ) -> Dict[str, float]:
        """Вычисление метрик качества"""
        if len(y_true) == 0:
            return {}

        metrics = {"accuracy": accuracy_score(y_true, y_pred)}

        if task_type == TaskType.BINARY:
            metrics.update(
                {
                    "precision": precision_score(y_true, y_pred, zero_division=0),
                    "recall": recall_score(y_true, y_pred, zero_division=0),
                    "f1": f1_score(y_true, y_pred, zero_division=0),
                }
            )
        else:
            metrics.update(
                {
                    "macro_f1": f1_score(
                        y_true, y_pred, average="macro", zero_division=0
                    ),
                    "weighted_f1": f1_score(
                        y_true, y_pred, average="weighted", zero_division=0
                    ),
                }
            )

        return metrics


class MaxProbabilitySelector(BaseThresholdSelector):
    """
    Отбор по максимальной вероятности.
    Работает как для бинарной, так и для многоклассовой классификации.
    """

    def __init__(self, threshold: float = 0.5):
        super().__init__(f"MaxProbability (τ={threshold})")
        self.threshold = threshold

    def _apply_binary(self, probabilities, y_true=None, **kwargs):
        # Для бинарной классификации просто применяем порог
        selected_mask = probabilities >= self.threshold
        selected_indices = np.where(selected_mask)[0]

        result = ThresholdResult(
            method_name=self.name,
            task_type=TaskType.BINARY,
            threshold=self.threshold,
            selected_indices=selected_indices,
            selection_ratio=np.mean(selected_mask),
            confidence_scores=probabilities[selected_mask],
            predicted_labels=(probabilities[selected_mask] >= 0.5).astype(int),
            metadata={"fixed_threshold": self.threshold},
        )

        # Вычисляем метрики если есть истинные метки
        if y_true is not None and len(selected_indices) > 0:
            result.metrics = self._compute_metrics(
                y_true[selected_indices], result.predicted_labels, TaskType.BINARY
            )

        return result

    def _apply_multiclass(self, probabilities, y_true=None, **kwargs):
        # Для многоклассовой - отбираем по максимальной вероятности среди классов
        max_proba = np.max(probabilities, axis=1)
        selected_mask = max_proba >= self.threshold
        selected_indices = np.where(selected_mask)[0]

        predicted_classes = np.argmax(probabilities[selected_mask], axis=1)

        result = ThresholdResult(
            method_name=self.name,
            task_type=TaskType.MULTICLASS,
            threshold={"max_probability": self.threshold},
            selected_indices=selected_indices,
            selection_ratio=np.mean(selected_mask),
            confidence_scores=max_proba[selected_mask],
            predicted_labels=predicted_classes,
            metadata={
                "threshold": self.threshold,
                "mean_max_proba": np.mean(max_proba),
                "class_distribution": (
                    dict(zip(*np.unique(predicted_classes, return_counts=True)))
                    if len(predicted_classes) > 0
                    else {}
                ),
            },
        )

        if y_true is not None and len(selected_indices) > 0:
            result.metrics = self._compute_metrics(
                y_true[selected_indices], predicted_classes, TaskType.MULTICLASS
            )

        return result


class PercentileSelector(BaseThresholdSelector):
    """
    Процентильный метод.
    Адаптируется под тип задачи автоматически.
    """

    def __init__(self, percentile: float = 80):
        super().__init__(f"Percentile ({percentile}%)")
        self.percentile = percentile

    def _apply_binary(self, probabilities, y_true=None, **kwargs):
        threshold = np.percentile(probabilities, self.percentile)
        selected_mask = probabilities >= threshold
        selected_indices = np.where(selected_mask)[0]

        result = ThresholdResult(
            method_name=self.name,
            task_type=TaskType.BINARY,
            threshold=threshold,
            selected_indices=selected_indices,
            selection_ratio=np.mean(selected_mask),
            confidence_scores=probabilities[selected_mask],
            predicted_labels=(probabilities[selected_mask] >= 0.5).astype(int),
            metadata={"percentile": self.percentile, "computed_threshold": threshold},
        )

        if y_true is not None and len(selected_indices) > 0:
            result.metrics = self._compute_metrics(
                y_true[selected_indices], result.predicted_labels, TaskType.BINARY
            )

        return result

    def _apply_multiclass(self, probabilities, y_true=None, **kwargs):
        # Для многоклассовой применяем процентиль к максимальным вероятностям
        max_proba = np.max(probabilities, axis=1)
        threshold = np.percentile(max_proba, self.percentile)
        selected_mask = max_proba >= threshold
        selected_indices = np.where(selected_mask)[0]

        predicted_classes = np.argmax(probabilities[selected_mask], axis=1)

        result = ThresholdResult(
            method_name=self.name,
            task_type=TaskType.MULTICLASS,
            threshold={"percentile_on_max_prob": threshold},
            selected_indices=selected_indices,
            selection_ratio=np.mean(selected_mask),
            confidence_scores=max_proba[selected_mask],
            predicted_labels=predicted_classes,
            metadata={
                "percentile": self.percentile,
                "computed_threshold": threshold,
                "applied_to": "max_probability",
            },
        )

        if y_true is not None and len(selected_indices) > 0:
            result.metrics = self._compute_metrics(
                y_true[selected_indices], predicted_classes, TaskType.MULTICLASS
            )

        return result


class EntropySelector(BaseThresholdSelector):
    """
    Энтропийный метод.
    Автоматически адаптируется под количество классов.
    """

    def __init__(self, max_entropy: Optional[float] = None):
        super().__init__("Entropy")
        self.max_entropy = max_entropy

    def _apply_binary(self, probabilities, y_true=None, **kwargs):
        # Бинарная энтропия: H = -p*log(p) - (1-p)*log(1-p)
        p = np.clip(probabilities, 1e-10, 1 - 1e-10)
        binary_entropy = -p * np.log(p) - (1 - p) * np.log(1 - p)

        # Если порог не задан, используем медиану
        max_entropy = (
            self.max_entropy
            if self.max_entropy is not None
            else np.median(binary_entropy)
        )

        selected_mask = binary_entropy < max_entropy
        selected_indices = np.where(selected_mask)[0]

        result = ThresholdResult(
            method_name=self.name,
            task_type=TaskType.BINARY,
            threshold={"max_entropy": max_entropy},
            selected_indices=selected_indices,
            selection_ratio=np.mean(selected_mask),
            confidence_scores=probabilities[selected_mask],
            predicted_labels=(probabilities[selected_mask] >= 0.5).astype(int),
            metadata={
                "max_entropy": max_entropy,
                "mean_entropy": np.mean(binary_entropy),
                "entropy_type": "binary",
            },
        )

        if y_true is not None and len(selected_indices) > 0:
            result.metrics = self._compute_metrics(
                y_true[selected_indices], result.predicted_labels, TaskType.BINARY
            )

        return result

    def _apply_multiclass(self, probabilities, y_true=None, **kwargs):
        # Многоклассовая энтропия Шеннона
        sample_entropy = entropy(probabilities, axis=1)

        # Автоматическая адаптация порога под количество классов
        n_classes = probabilities.shape[1]
        max_possible_entropy = np.log(n_classes)

        if self.max_entropy is None:
            # Используем 50% от максимально возможной энтропии
            max_entropy = 0.5 * max_possible_entropy
        else:
            max_entropy = self.max_entropy

        selected_mask = sample_entropy < max_entropy
        selected_indices = np.where(selected_mask)[0]

        predicted_classes = np.argmax(probabilities[selected_mask], axis=1)

        result = ThresholdResult(
            method_name=self.name,
            task_type=TaskType.MULTICLASS,
            threshold={"max_entropy": max_entropy},
            selected_indices=selected_indices,
            selection_ratio=np.mean(selected_mask),
            confidence_scores=np.max(probabilities[selected_mask], axis=1),
            predicted_labels=predicted_classes,
            metadata={
                "max_entropy": max_entropy,
                "mean_entropy": np.mean(sample_entropy),
                "max_possible_entropy": max_possible_entropy,
                "normalized_threshold": max_entropy / max_possible_entropy,
                "n_classes": n_classes,
            },
        )

        if y_true is not None and len(selected_indices) > 0:
            result.metrics = self._compute_metrics(
                y_true[selected_indices], predicted_classes, TaskType.MULTICLASS
            )

        return result


class MarginSelector(BaseThresholdSelector):
    """
    Метод отрыва между классами.
    Для бинарной классификации - отрыв от 0.5.
    Для многоклассовой - разность между top-2 вероятностями.
    """

    def __init__(self, min_margin: float = 0.1):
        super().__init__(f"Margin (δ>{min_margin})")
        self.min_margin = min_margin

    def _apply_binary(self, probabilities, y_true=None, **kwargs):
        # Для бинарной классификации margin = |p - 0.5|
        margins = np.abs(probabilities - 0.5)
        selected_mask = margins > self.min_margin
        selected_indices = np.where(selected_mask)[0]

        result = ThresholdResult(
            method_name=self.name,
            task_type=TaskType.BINARY,
            threshold={"min_margin_from_0.5": self.min_margin},
            selected_indices=selected_indices,
            selection_ratio=np.mean(selected_mask),
            confidence_scores=probabilities[selected_mask],
            predicted_labels=(probabilities[selected_mask] >= 0.5).astype(int),
            metadata={
                "min_margin": self.min_margin,
                "mean_margin": np.mean(margins),
                "margin_type": "distance_from_0.5",
            },
        )

        if y_true is not None and len(selected_indices) > 0:
            result.metrics = self._compute_metrics(
                y_true[selected_indices], result.predicted_labels, TaskType.BINARY
            )

        return result

    def _apply_multiclass(self, probabilities, y_true=None, **kwargs):
        # Для многоклассовой - разность между двумя наибольшими вероятностями
        sorted_proba = np.sort(probabilities, axis=1)[:, ::-1]
        margins = sorted_proba[:, 0] - sorted_proba[:, 1]

        selected_mask = margins > self.min_margin
        selected_indices = np.where(selected_mask)[0]

        predicted_classes = np.argmax(probabilities[selected_mask], axis=1)

        # Анализ путаемых классов (где margin низкий)
        confused_mask = margins <= self.min_margin
        if np.any(confused_mask):
            confused_indices = np.where(confused_mask)[0]
            top2_classes = np.argsort(probabilities[confused_indices], axis=1)[:, -2:]

            confusion_pairs = {}
            for idx in range(len(top2_classes)):
                pair = tuple(sorted(top2_classes[idx]))
                confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1

            top_confused = sorted(
                confusion_pairs.items(), key=lambda x: x[1], reverse=True
            )[:5]
        else:
            top_confused = []

        result = ThresholdResult(
            method_name=self.name,
            task_type=TaskType.MULTICLASS,
            threshold={"min_margin": self.min_margin},
            selected_indices=selected_indices,
            selection_ratio=np.mean(selected_mask),
            confidence_scores=np.max(probabilities[selected_mask], axis=1),
            predicted_labels=predicted_classes,
            metadata={
                "min_margin": self.min_margin,
                "mean_margin": np.mean(margins),
                "margin_type": "top1_minus_top2",
                "confused_class_pairs": top_confused,
            },
        )

        if y_true is not None and len(selected_indices) > 0:
            result.metrics = self._compute_metrics(
                y_true[selected_indices], predicted_classes, TaskType.MULTICLASS
            )

        return result


class F1OptimizationSelector(BaseThresholdSelector):
    """Оптимизация по F1-мере (только для бинарной классификации)"""

    def __init__(self):
        super().__init__("F1 Optimization", supported_tasks=[TaskType.BINARY])

    @property
    def requires_labels(self) -> bool:
        return True

    def _apply_binary(self, probabilities, y_true=None, **kwargs):
        if y_true is None:
            raise ValueError("F1 optimization requires true labels (y_true)")

        # Вычисляем precision-recall кривую
        precisions, recalls, thresholds = precision_recall_curve(y_true, probabilities)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)

        # Находим оптимальный порог
        optimal_idx = np.argmax(f1_scores[:-1])
        optimal_threshold = thresholds[optimal_idx]

        # Применяем порог
        selected_mask = probabilities >= optimal_threshold
        selected_indices = np.where(selected_mask)[0]

        result = ThresholdResult(
            method_name=self.name,
            task_type=TaskType.BINARY,
            threshold=optimal_threshold,
            selected_indices=selected_indices,
            selection_ratio=np.mean(selected_mask),
            confidence_scores=probabilities[selected_mask],
            predicted_labels=(probabilities[selected_mask] >= 0.5).astype(int),
            metadata={
                "optimal_threshold": optimal_threshold,
                "max_f1": f1_scores[optimal_idx],
                "precision_at_threshold": precisions[optimal_idx],
                "recall_at_threshold": recalls[optimal_idx],
            },
        )

        if len(selected_indices) > 0:
            result.metrics = self._compute_metrics(
                y_true[selected_indices], result.predicted_labels, TaskType.BINARY
            )

        return result

    def _apply_multiclass(self, probabilities, y_true=None, **kwargs):
        raise NotImplementedError("F1 optimization is not implemented for multiclass")


class YoudenSelector(BaseThresholdSelector):
    """Статистика Юдена (только для бинарной классификации)"""

    def __init__(self):
        super().__init__("Youden's J", supported_tasks=[TaskType.BINARY])

    @property
    def requires_labels(self) -> bool:
        return True

    def _apply_binary(self, probabilities, y_true=None, **kwargs):
        if y_true is None:
            raise ValueError("Youden's J statistic requires true labels (y_true)")

        fpr, tpr, thresholds = roc_curve(y_true, probabilities)
        j_scores = tpr - fpr

        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]

        selected_mask = probabilities >= optimal_threshold
        selected_indices = np.where(selected_mask)[0]

        result = ThresholdResult(
            method_name=self.name,
            task_type=TaskType.BINARY,
            threshold=optimal_threshold,
            selected_indices=selected_indices,
            selection_ratio=np.mean(selected_mask),
            confidence_scores=probabilities[selected_mask],
            predicted_labels=(probabilities[selected_mask] >= 0.5).astype(int),
            metadata={
                "optimal_threshold": optimal_threshold,
                "j_score": j_scores[optimal_idx],
                "sensitivity": tpr[optimal_idx],
                "specificity": 1 - fpr[optimal_idx],
            },
        )

        if len(selected_indices) > 0:
            result.metrics = self._compute_metrics(
                y_true[selected_indices], result.predicted_labels, TaskType.BINARY
            )

        return result

    def _apply_multiclass(self, probabilities, y_true=None, **kwargs):
        raise NotImplementedError("Youden's J is not defined for multiclass")


class AdaptiveSelector(BaseThresholdSelector):
    """
    Адаптивный метод с обновлением порогов.
    Автоматически работает с обоими типами задач.
    """

    def __init__(self, initial_threshold: float = 0.9, momentum: float = 0.9):
        super().__init__(f"Adaptive (α={1-momentum})")
        self.initial_threshold = initial_threshold
        self.momentum = momentum
        self.history = []
        self.current_threshold = initial_threshold
        self.class_thresholds = None

    def _apply_binary(self, probabilities, y_true=None, **kwargs):
        # Обновляем порог на основе текущего распределения
        mean_confidence = np.mean(probabilities[probabilities > 0.5])

        # EMA обновление
        self.current_threshold = (
            self.momentum * self.current_threshold
            + (1 - self.momentum) * mean_confidence
        )

        selected_mask = probabilities >= self.current_threshold
        selected_indices = np.where(selected_mask)[0]

        self.history.append(
            {
                "iteration": len(self.history),
                "threshold": self.current_threshold,
                "mean_confidence": mean_confidence,
                "selection_ratio": np.mean(selected_mask),
            }
        )

        result = ThresholdResult(
            method_name=self.name,
            task_type=TaskType.BINARY,
            threshold=self.current_threshold,
            selected_indices=selected_indices,
            selection_ratio=np.mean(selected_mask),
            confidence_scores=probabilities[selected_mask],
            predicted_labels=(probabilities[selected_mask] >= 0.5).astype(int),
            metadata={
                "current_threshold": self.current_threshold,
                "iterations": len(self.history),
                "momentum": self.momentum,
            },
        )

        if y_true is not None and len(selected_indices) > 0:
            result.metrics = self._compute_metrics(
                y_true[selected_indices], result.predicted_labels, TaskType.BINARY
            )

        return result

    def _apply_multiclass(self, probabilities, y_true=None, **kwargs):
        n_classes = probabilities.shape[1]

        # Инициализация классо-специфичных порогов
        if self.class_thresholds is None:
            self.class_thresholds = np.full(n_classes, self.initial_threshold)

        # Предсказанные классы и максимальные вероятности
        predicted_classes = np.argmax(probabilities, axis=1)
        max_probas = np.max(probabilities, axis=1)

        # Применяем классо-специфичные пороги
        selected_mask = np.zeros(len(probabilities), dtype=bool)
        for i in range(len(probabilities)):
            class_idx = predicted_classes[i]
            selected_mask[i] = max_probas[i] >= self.class_thresholds[class_idx]

        selected_indices = np.where(selected_mask)[0]

        # Обновляем пороги для каждого класса
        for class_idx in range(n_classes):
            class_mask = predicted_classes == class_idx
            if np.any(class_mask):
                class_confidence = np.mean(max_probas[class_mask])
                self.class_thresholds[class_idx] = (
                    self.momentum * self.class_thresholds[class_idx]
                    + (1 - self.momentum) * class_confidence
                )

        result = ThresholdResult(
            method_name=self.name,
            task_type=TaskType.MULTICLASS,
            threshold={"class_thresholds": self.class_thresholds.copy()},
            selected_indices=selected_indices,
            selection_ratio=np.mean(selected_mask),
            confidence_scores=max_probas[selected_mask],
            predicted_labels=predicted_classes[selected_mask],
            metadata={
                "mean_threshold": np.mean(self.class_thresholds),
                "threshold_std": np.std(self.class_thresholds),
                "iterations": len(self.history),
                "momentum": self.momentum,
            },
        )

        if y_true is not None and len(selected_indices) > 0:
            result.metrics = self._compute_metrics(
                y_true[selected_indices], result.predicted_labels, TaskType.MULTICLASS
            )

        return result

    def reset(self):
        """Сброс адаптивных параметров"""
        self.current_threshold = self.initial_threshold
        self.class_thresholds = None
        self.history = []


# Удобные функции для быстрого использования


def select_confident_samples(
    probabilities: np.ndarray,
    method: str = "percentile",
    y_true: Optional[np.ndarray] = None,
    task_type: TaskType = TaskType.AUTO,
    return_result: bool = False,
    **kwargs,
) -> Union[np.ndarray, ThresholdResult]:
    """
    Удобная функция для отбора уверенных примеров.

    Args:
        probabilities: Вероятности предсказаний
        method: Название метода ('max_prob', 'percentile', 'entropy', 'margin', 'f1', 'youden', 'adaptive')
        y_true: Истинные метки (для методов оптимизации)
        task_type: Тип задачи (auto определяет автоматически)
        return_result: Возвращать полный результат вместо только индексов
        **kwargs: Параметры для конкретного метода

    Returns:
        selected_indices или ThresholdResult если return_result=True
    """
    # Создаем селектор
    selector = create_selector(method, **kwargs)

    # Применяем
    result = selector.apply(probabilities, y_true, task_type)

    if return_result:
        return result
    return result.selected_indices


def create_selector(method: str, **kwargs) -> BaseThresholdSelector:
    """
    Фабрика для создания селекторов.

    Args:
        method: Название метода
        **kwargs: Параметры метода
    """
    method = method.lower()

    if method in ["max_prob", "max_probability", "fixed"]:
        return MaxProbabilitySelector(threshold=kwargs.get("threshold", 0.5))

    elif method == "percentile":
        return PercentileSelector(percentile=kwargs.get("percentile", 80))

    elif method == "entropy":
        return EntropySelector(max_entropy=kwargs.get("max_entropy", None))

    elif method == "margin":
        return MarginSelector(min_margin=kwargs.get("min_margin", 0.1))

    elif method == "f1":
        return F1OptimizationSelector()

    elif method == "youden":
        return YoudenSelector()

    elif method == "adaptive":
        return AdaptiveSelector(
            initial_threshold=kwargs.get("initial_threshold", 0.9),
            momentum=kwargs.get("momentum", 0.9),
        )

    else:
        raise ValueError(f"Unknown method: {method}")


def get_available_methods(task_type: Optional[TaskType] = None) -> List[str]:
    """
    Получить список доступных методов.

    Args:
        task_type: Фильтр по типу задачи
    """
    all_methods = {
        "max_prob": [TaskType.BINARY, TaskType.MULTICLASS],
        "percentile": [TaskType.BINARY, TaskType.MULTICLASS],
        "entropy": [TaskType.BINARY, TaskType.MULTICLASS],
        "margin": [TaskType.BINARY, TaskType.MULTICLASS],
        "f1": [TaskType.BINARY],
        "youden": [TaskType.BINARY],
        "adaptive": [TaskType.BINARY, TaskType.MULTICLASS],
    }

    if task_type is None:
        return list(all_methods.keys())

    return [method for method, types in all_methods.items() if task_type in types]


# Пример использования и демонстрация

if __name__ == "__main__":
    print("Unified Threshold Selection Module Demo")
    print("=" * 50)

    # Пример 1: Бинарная классификация
    print("\n1. Binary Classification Example:")
    np.random.seed(42)
    binary_proba = np.random.beta(2, 5, 1000)

    # Автоматическое определение типа задачи
    result = select_confident_samples(
        binary_proba, method="percentile", percentile=80, return_result=True
    )
    print(f"   Task type detected: {result.task_type.value}")
    print(f"   Selected {result.selection_ratio:.1%} samples")
    print(f"   Threshold: {result.threshold:.3f}")

    # Пример 2: Многоклассовая классификация
    print("\n2. Multiclass Classification Example:")
    n_samples, n_classes = 1000, 5
    multiclass_proba = np.random.dirichlet(alpha=[2] * n_classes, size=n_samples)

    # Применяем тот же метод - он адаптируется автоматически
    result = select_confident_samples(
        multiclass_proba, method="entropy", return_result=True
    )
    print(f"   Task type detected: {result.task_type.value}")
    print(f"   Selected {result.selection_ratio:.1%} samples")
    print(f"   Entropy threshold: {result.threshold['max_entropy']:.3f}")

    # Пример 3: Margin метод работает для обоих типов
    print("\n3. Margin Method - Universal:")

    # Бинарная
    binary_result = select_confident_samples(
        binary_proba, method="margin", min_margin=0.2, return_result=True
    )
    print(f"   Binary: selected {binary_result.selection_ratio:.1%}")

    # Многоклассовая
    multi_result = select_confident_samples(
        multiclass_proba, method="margin", min_margin=0.2, return_result=True
    )
    print(f"   Multiclass: selected {multi_result.selection_ratio:.1%}")
    print(f"   Top confused pairs: {multi_result.metadata['confused_class_pairs'][:2]}")

    # Пример 4: Методы специфичные для типа задачи
    print("\n4. Task-specific methods:")
    print(f"   Binary-only methods: {get_available_methods(TaskType.BINARY)}")
    print(f"   All methods: {get_available_methods()}")
