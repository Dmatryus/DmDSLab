"""Оркестрация прогона: запуск FS-методов через кросс-валидацию.

`CVRunner` прогоняет один FS-метод через k-fold кросс-валидацию: на каждом
фолде вызывает `method.fit_select`, обучает на отобранных признаках прокси-
модель-оценщик и измеряет качество на валидации. Прокси-модель — внутренний
оценщик ранжирования набора признаков, **не** финальная модель (CLAUDE
NEVER §1–2): обучение прокси нужно лишь чтобы сравнить наборы признаков
между собой.

Прокси-модель выбирается параметром конструктора `estimator`:

- ``"catboost"`` — CatBoost (ленивый импорт опц. пакета `catboost`); при
  отсутствии пакета — информативная ошибка;
- ``"random_forest"`` — `RandomForest*` из `scikit-learn` (core-зависимость);
- ``None`` — авто: CatBoost при наличии пакета, иначе тихий fallback на
  RandomForest.

Метрика возвращается в направлении «больше = лучше» (метрики-ошибки
инвертируются), чтобы Optuna (E-006.2) оптимизировала на ``maximize``
(план E-006 §5 R-5).
"""

from __future__ import annotations

import importlib.util
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import KFold, StratifiedKFold

from ..api import Dataset
from ..methods._utils import is_classification
from ..methods.base import FSMethod

__all__ = ["CVResult", "CVRunner"]

# Консервативные настройки прокси-CatBoost — это оценщик ранжирования
# набора признаков, а не финальная модель (план E-006 §5 R-2).
_CATBOOST_ITERATIONS = 100


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


def _catboost_available() -> bool:
    """Проверяет, установлен ли опц. пакет `catboost`.

    Returns:
        ``True``, если пакет `catboost` импортируется.
    """
    return importlib.util.find_spec("catboost") is not None


class CVRunner:
    """Прогоняет FS-метод через кросс-валидацию на датасете.

    Attributes:
        cv: Число разбиений кросс-валидации.
        random_seed: Зерно случайности фолд-сплиттера и прокси-модели.
        estimator: Выбранная прокси-модель — ``"catboost"`` |
            ``"random_forest"`` (после разрешения авто-режима).
    """

    def __init__(
        self,
        cv: int = 5,
        random_seed: int | None = None,
        estimator: str | None = None,
    ) -> None:
        """Инициализирует раннер.

        Args:
            cv: Число разбиений кросс-валидации.
            random_seed: Зерно случайности для воспроизводимости —
                сидирует фолд-сплиттер и прокси-модель (план E-006 §5 R-3).
            estimator: Выбор прокси-модели-оценщика — ``"catboost"`` |
                ``"random_forest"`` | ``None``. ``None`` — авто: CatBoost
                при наличии пакета, иначе RandomForest.

        Raises:
            ValueError: Если `cv` < 2 или `estimator` не из допустимого
                набора значений.
            ImportError: Если явно выбран ``"catboost"``, но опц. пакет
                `catboost` не установлен.
        """
        if cv < 2:
            raise ValueError(f"cv должно быть >= 2, получено {cv}")

        if estimator is None:
            resolved = "catboost" if _catboost_available() else "random_forest"
        elif estimator == "catboost":
            if not _catboost_available():
                raise ImportError(
                    "estimator='catboost': опц. пакет 'catboost' не "
                    "установлен. Установите его (extra `[methods]`: "
                    "pip install 'feature_selection_benchmark[methods]') "
                    "или используйте estimator='random_forest' / "
                    "estimator=None (авто-fallback на RandomForest)."
                )
            resolved = "catboost"
        elif estimator == "random_forest":
            resolved = "random_forest"
        else:
            raise ValueError(
                "estimator должно быть 'catboost', 'random_forest' или "
                f"None, получено {estimator!r}"
            )

        self.cv = cv
        self.random_seed = random_seed
        self.estimator = resolved

    def run_cv(
        self,
        method: FSMethod,
        dataset: Dataset,
        hyperparams: dict[str, object],
    ) -> CVResult:
        """Прогоняет один FS-метод через кросс-валидацию.

        На каждом фолде вызывает `method.fit_select` на обучающей части,
        обучает прокси-модель на отобранных признаках и измеряет метрику
        на валидационной части. `cv_score` — среднее по фолдам, `score_std`
        — стандартное отклонение. Метрика — в направлении «больше = лучше»
        (accuracy для классификации, R² для регрессии).

        Args:
            method: Метод отбора признаков.
            dataset: Входной датасет (признаки + целевая переменная).
            hyperparams: Гиперпараметры метода — передаются в `fit_select`.

        Returns:
            Агрегированный результат кросс-валидации.
        """
        start = time.perf_counter()

        X = dataset.data[dataset.features_list].reset_index(drop=True)
        y = dataset.data[dataset.target].reset_index(drop=True)
        classification = is_classification(y)

        splitter = self._make_splitter(classification)
        fold_scores: list[float] = []
        # Признаки последнего фолда — репрезентативный набор для отчёта.
        last_selected: list[str] = []

        for train_idx, val_idx in splitter.split(X, y):
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_val = X.iloc[val_idx]
            y_val = y.iloc[val_idx]

            selected = method.fit_select(X_train, y_train, **hyperparams)
            last_selected = list(selected)

            estimator = self._make_estimator(classification)
            estimator.fit(X_train[selected], y_train)
            predictions = estimator.predict(X_val[selected])
            fold_scores.append(
                self._score(y_val, predictions, classification)
            )

        scores = np.asarray(fold_scores, dtype=float)
        return CVResult(
            cv_score=float(scores.mean()),
            # ddof=1 — выборочное (несмещённое) СКО по cv фолдам:
            # конвенционально для отчётности std-по-фолдам (лидерборд E-007).
            score_std=float(scores.std(ddof=1)),
            selected_features=last_selected,
            duration_sec=time.perf_counter() - start,
        )

    def _make_splitter(
        self, classification: bool
    ) -> StratifiedKFold | KFold:
        """Создаёт сидированный фолд-сплиттер по типу задачи.

        Args:
            classification: ``True`` — задача классификации.

        Returns:
            `StratifiedKFold` для классификации, `KFold` для регрессии.
        """
        kwargs = {
            "n_splits": self.cv,
            "shuffle": True,
            "random_state": self.random_seed,
        }
        return (
            StratifiedKFold(**kwargs)
            if classification
            else KFold(**kwargs)
        )

    def _make_estimator(self, classification: bool):
        """Создаёт сидированную прокси-модель по типу задачи.

        Args:
            classification: ``True`` — задача классификации.

        Returns:
            Несбученный экземпляр прокси-модели (CatBoost или RandomForest).
        """
        if self.estimator == "catboost":
            from catboost import CatBoostClassifier, CatBoostRegressor

            cls = (
                CatBoostClassifier
                if classification
                else CatBoostRegressor
            )
            return cls(
                iterations=_CATBOOST_ITERATIONS,
                random_seed=self.random_seed,
                allow_writing_files=False,
                verbose=False,
            )

        from sklearn.ensemble import (
            RandomForestClassifier,
            RandomForestRegressor,
        )

        cls = (
            RandomForestClassifier
            if classification
            else RandomForestRegressor
        )
        return cls(random_state=self.random_seed)

    @staticmethod
    def _score(
        y_true: pd.Series,
        y_pred: np.ndarray,
        classification: bool,
    ) -> float:
        """Вычисляет метрику качества в направлении «больше = лучше».

        Для классификации — accuracy, для регрессии — R². Обе метрики
        растут с качеством, поэтому инверсия не нужна; контракт «больше =
        лучше» позволяет Optuna (E-006.2) оптимизировать на ``maximize``
        (план E-006 §5 R-5).

        Args:
            y_true: Истинные значения целевой переменной.
            y_pred: Предсказания прокси-модели.
            classification: ``True`` — задача классификации.

        Returns:
            Значение метрики (больше — лучше).
        """
        if classification:
            return float(accuracy_score(y_true, y_pred))
        return float(r2_score(y_true, y_pred))
