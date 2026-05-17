"""Стабильные контракты интерфейса FS-методов.

Здесь зафиксированы:

- базовый класс `FSMethod` (ABC — интерфейс, который реализует каждый
  метод отбора признаков);
- датакласс `HyperParam` — declarative-дескриптор одного гиперпараметра
  (контракт пространства поиска по ADR 0002, OQ-1);
- датакласс `MethodInfo` — метаданные метода для `list_methods`.

Сигнатуры выверены по ARCHITECTURE §4 и ADR `0002` и являются точкой
расширения для Персоны 2.

Контракт пространства поиска (ADR 0002, OQ-1 → resolved)
--------------------------------------------------------

Каждый FS-метод объявляет пространство поиска своих гиперпараметров
**декларативно** — как обычные данные, без зависимости от Optuna. Слой
`methods/` не импортирует `optuna`; трансляцию declarative-дескриптора в
вызовы `trial.suggest_*` выполняет только `core/tuning.py` (E-006).

OQ-1 (форма контракта — расширение `MethodInfo.hyperparameters` против
отдельного типа `SearchSpace`) разрешён компромиссом: вводится отдельный
датакласс-дескриптор `HyperParam` на один гиперпараметр, а поле
`MethodInfo.hyperparameters` сохраняется (без добавления нового поля —
6 полей `MethodInfo` неизменны) и типизируется как
``dict[str, HyperParam]`` (имя гиперпараметра → его дескриптор).
Отдельный обёрточный тип `SearchSpace` не вводится: ``dict`` гиперпараметров
сам по себе уже является пространством поиска (PRINCIPLE 2 — Simplicity First).

Один дескриптор `HyperParam` описывает один из трёх видов параметра:

- ``kind="int"``    — целое из диапазона ``[low, high]`` (``log`` —
  логарифмическая шкала);
- ``kind="float"``  — вещественное из диапазона ``[low, high]`` (``log`` —
  логарифмическая шкала);
- ``kind="categorical"`` — значение из списка ``choices``.

`default` — значение для режима ``n_trials=0`` (прогон на дефолтах без
study). Метод без настраиваемых гиперпараметров объявляет
``MethodInfo.hyperparameters = {}``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd

__all__ = ["FSMethod", "HyperParam", "MethodInfo"]


@dataclass
class HyperParam:
    """Declarative-дескриптор одного гиперпараметра FS-метода.

    Описывает пространство поиска одного гиперпараметра как данные, без
    зависимости от Optuna (ADR `0002`, контракт E-005). `core/tuning.py`
    (E-006) транслирует `kind` в соответствующий вызов
    `trial.suggest_int` / `suggest_float` / `suggest_categorical`.

    Attributes:
        kind: Вид гиперпараметра — ``"int"`` | ``"float"`` |
            ``"categorical"``.
        default: Значение по умолчанию — используется в режиме
            ``n_trials=0`` (прогон без подбора).
        low: Нижняя граница диапазона. Обязательна для
            ``kind in {"int", "float"}``; игнорируется для
            ``"categorical"``.
        high: Верхняя граница диапазона. Обязательна для
            ``kind in {"int", "float"}``; игнорируется для
            ``"categorical"``.
        log: Логарифмическая шкала сэмплирования. Применима только для
            ``kind in {"int", "float"}``.
        choices: Список допустимых значений. Обязателен для
            ``kind="categorical"``; игнорируется для ``"int"`` / ``"float"``.

    Raises:
        ValueError: Если поля несовместимы с заявленным `kind` (например,
            ``"int"`` без границ или ``"categorical"`` без `choices`).
    """

    kind: Literal["int", "float", "categorical"]
    default: Any
    low: float | int | None = None
    high: float | int | None = None
    log: bool = False
    choices: list[Any] | None = None

    def __post_init__(self) -> None:
        """Проверяет согласованность полей с заявленным `kind`."""
        if self.kind in ("int", "float"):
            if self.low is None or self.high is None:
                raise ValueError(
                    f"HyperParam kind={self.kind!r} требует low и high"
                )
            if self.low > self.high:
                raise ValueError(
                    f"HyperParam: low ({self.low}) > high ({self.high})"
                )
        elif self.kind == "categorical":
            if not self.choices:
                raise ValueError(
                    "HyperParam kind='categorical' требует непустой choices"
                )
            if self.log:
                raise ValueError(
                    "HyperParam: log неприменим к kind='categorical'"
                )
        else:  # pragma: no cover - защищено Literal-типом
            raise ValueError(f"HyperParam: неизвестный kind {self.kind!r}")


class FSMethod(ABC):
    """Базовый класс (интерфейс) метода отбора признаков.

    Каждый FS-метод — наследник `FSMethod`, объявляющий метаданные через
    атрибуты класса и реализующий абстрактный метод `fit_select`.

    Пространство поиска гиперпараметров метод объявляет декларативно —
    словарём ``имя → HyperParam`` (см. модульный docstring и `HyperParam`).
    Этот словарь попадает в `MethodInfo.hyperparameters`.

    Опциональные зависимости (`shap`, `boruta`, `catboost` и др.) метод
    проверяет в `check_availability`: если зависимость не установлена,
    метод всё равно регистрируется, но помечается недоступным и
    исключается из дефолтного набора прогона (ADR `0002` / план E-005,
    OQ-2).

    Attributes:
        name: Уникальное имя метода.
        group: Группа метода — ``"filter"`` | ``"wrapper"`` |
            ``"embedded"`` | ``"shap"`` | ``"permutation"``.
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
            **hyperparams: Гиперпараметры конкретного метода. Имена и
                допустимые значения соответствуют declarative-пространству
                поиска метода (``MethodInfo.hyperparameters``).

        Returns:
            Список имён отобранных признаков.
        """
        ...

    @classmethod
    def check_availability(cls) -> str | None:
        """Проверяет доступность метода (наличие опц. зависимостей).

        Метод с опциональной зависимостью переопределяет этот метод и
        возвращает строку-причину, если зависимость не установлена.
        Базовая реализация считает метод всегда доступным.

        Returns:
            ``None``, если метод доступен; иначе — строка с причиной
            недоступности (например, ``"требуется пакет 'shap'"``).
        """
        return None


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
        hyperparameters: Declarative-пространство поиска — словарь
            ``имя гиперпараметра → HyperParam`` (контракт ADR `0002`,
            см. модульный docstring). Пустой словарь — у метода нет
            настраиваемых гиперпараметров.
        group: Группа метода — ``"filter"`` | ``"wrapper"`` |
            ``"embedded"`` | ``"shap"`` | ``"permutation"``.
        supported_tasks: Поддерживаемые типы задач.
        available: Доступен ли метод. ``False`` — опц. зависимость метода
            не установлена; такой метод исключается из дефолтного
            ``methods=None``-набора прогона (план E-005, OQ-2).
        unavailable_reason: Причина недоступности (``None``, если метод
            доступен).
    """

    name: str
    description: str
    requires_target: bool
    supports_multiclass: bool
    output_type: Literal["ranking", "subset"]
    hyperparameters: dict[str, HyperParam] = field(default_factory=dict)
    group: str = ""
    supported_tasks: list[str] = field(default_factory=list)
    available: bool = True
    unavailable_reason: str | None = None
