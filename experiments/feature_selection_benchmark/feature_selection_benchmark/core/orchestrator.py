"""Оркестрация прогона: запуск FS-методов через кросс-валидацию.

Заглушка каркаса (E-001.4): сигнатуры зафиксированы по ARCHITECTURE §2/§3,
тела поднимают `NotImplementedError`.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..api import Dataset
from ..methods.base import FSMethod

__all__ = ["CVResult", "CVRunner"]


@dataclass
class CVResult:
    """Результат кросс-валидации одного FS-метода.

    Attributes:
        cv_score: Средняя оценка качества по фолдам.
        score_std: Стандартное отклонение оценки по фолдам.
        selected_features: Отобранные признаки.
        duration_sec: Длительность прогона метода в секундах.
    """

    cv_score: float
    score_std: float
    selected_features: list[str]
    duration_sec: float


class CVRunner:
    """Прогоняет FS-метод через кросс-валидацию на датасете."""

    def __init__(self, cv: int = 5, random_seed: int | None = None) -> None:
        """Инициализирует раннер.

        Args:
            cv: Число разбиений кросс-валидации.
            random_seed: Зерно случайности для воспроизводимости.
        """
        raise NotImplementedError

    def run_cv(
        self,
        method: FSMethod,
        dataset: Dataset,
        hyperparams: dict[str, object],
    ) -> CVResult:
        """Прогоняет один FS-метод через кросс-валидацию.

        Args:
            method: Метод отбора признаков.
            dataset: Входной датасет.
            hyperparams: Гиперпараметры метода.

        Returns:
            Агрегированный результат кросс-валидации.
        """
        raise NotImplementedError
