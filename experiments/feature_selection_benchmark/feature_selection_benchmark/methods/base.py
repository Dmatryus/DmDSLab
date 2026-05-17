"""Стабильные контракты интерфейса FS-методов.

Здесь зафиксированы базовый класс `FSMethod` (ABC — интерфейс, который
реализует каждый метод отбора признаков) и датакласс `MethodInfo`
(метаданные метода для `list_methods`). Сигнатуры выверены по
ARCHITECTURE §4 и являются точкой расширения для Персоны 2.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

__all__ = ["FSMethod", "MethodInfo"]


class FSMethod(ABC):
    """Базовый класс (интерфейс) метода отбора признаков.

    Каждый FS-метод — наследник `FSMethod`, объявляющий метаданные через
    атрибуты класса и реализующий абстрактный метод `fit_select`.

    Attributes:
        name: Уникальное имя метода.
        group: Группа метода — ``"filter"`` | ``"wrapper"`` |
            ``"embedded"`` | ``"shap"``.
        supported_tasks: Поддерживаемые типы задач — ``["classification"]`` |
            ``["regression"]`` | ``["classification", "regression"]``.
    """

    name: str
    group: str
    supported_tasks: list[str]

    @abstractmethod
    def fit_select(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        **hyperparams: Any,
    ) -> list[str]:
        """Отбирает признаки на обучающей выборке.

        Args:
            X_train: Обучающая выборка признаков.
            y_train: Обучающая целевая переменная.
            **hyperparams: Гиперпараметры конкретного метода.

        Returns:
            Список имён отобранных признаков.
        """
        ...


@dataclass
class MethodInfo:
    """Метаданные FS-метода (возвращается из `list_methods`).

    Attributes:
        name: Уникальное имя метода.
        description: Человекочитаемое описание метода.
        requires_target: Нужна ли методу целевая переменная.
        supports_multiclass: Поддерживает ли метод многоклассовую
            классификацию.
        output_type: Тип выхода метода — ``"ranking"`` (ранжирование
            признаков) | ``"subset"`` (подмножество признаков).
        hyperparameters: Словарь гиперпараметров метода — имя → значение
            по умолчанию / описание.
    """

    name: str
    description: str
    requires_target: bool
    supports_multiclass: bool
    output_type: Literal["ranking", "subset"]
    hyperparameters: dict[str, Any]
