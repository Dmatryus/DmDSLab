"""FS-методы группы ``wrapper`` — отбор через перебор подмножеств с моделью.

Реализованы (E-005.3):

- ``rfe`` — рекурсивное исключение признаков (`sklearn.feature_selection.RFE`);
- ``sequential_feature_selector`` — последовательный жадный отбор
  (`sklearn.feature_selection.SequentialFeatureSelector`);
- ``stability_selection`` — отбор по частоте попадания признака в
  L1-модель на бутстрэп-подвыборках (рандомизированный Lasso/логрегрессия,
  поверх `scikit-learn`);
- ``boruta`` — алгоритм Boruta поверх случайного леса; опирается на
  опциональный пакет ``boruta``. Пакет импортируется **лениво** внутри
  метода; при его отсутствии метод регистрируется, но помечается
  недоступным (план E-005, OQ-2; R-2 — известная хрупкость совместимости
  `boruta`).

Wrapper-методы по природе ранжирующие/перебирающие: каждый объявляет
гиперпараметр, переводящий результат в подмножество признаков
(``fraction`` / ``selection_threshold``), и возвращает ``list[str]`` имён
(план E-005, R-6).

Тип задачи (классификация vs регрессия) методы определяют по `y_train`
автоматически: базовый эстиматор-обёртка подбирается под задачу. Зерно
рандома прокидывается гиперпараметром ``random_seed`` (план E-005, R-5).

Пространство поиска объявлено декларативно через `HyperParam` — без
зависимости от Optuna (ADR `0002`, контракт E-005.1).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ._utils import is_classification
from .base import FSMethod, HyperParam, MethodInfo
from .registry import register

__all__ = [
    "RFEMethod",
    "SequentialFeatureSelectorMethod",
    "BorutaMethod",
    "StabilitySelectionMethod",
]


def _resolve_n_features(fraction: float, n_total: int) -> int:
    """Переводит долю отбираемых признаков в абсолютное число.

    Args:
        fraction: Доля признаков для отбора в диапазоне ``(0, 1]``.
        n_total: Общее число признаков.

    Returns:
        Число отбираемых признаков — минимум 1, максимум ``n_total``.
    """
    n_selected = int(round(fraction * n_total))
    return max(1, min(n_selected, n_total))


def _make_estimator(
    is_classification: bool,
    random_seed: int | None,
    *,
    linear: bool = False,
):
    """Создаёт базовый эстиматор-обёртку для wrapper-метода.

    Args:
        is_classification: ``True`` — классификация, ``False`` — регрессия.
        random_seed: Зерно рандома эстиматора.
        linear: ``True`` — вернуть L1-линейную модель (для
            ``stability_selection``); ``False`` — ансамбль деревьев
            (для ``rfe`` / ``sequential_feature_selector`` / ``boruta``).

    Returns:
        Несбученный экземпляр эстиматора `scikit-learn`.
    """
    if linear:
        if is_classification:
            from sklearn.linear_model import LogisticRegression

            # L1-регуляризация: solver="saga" + l1_ratio=1.0 — актуальный
            # API scikit-learn (penalty="l1" объявлен устаревшим в 1.8).
            return LogisticRegression(
                solver="saga",
                l1_ratio=1.0,
                random_state=random_seed,
                max_iter=300,
            )
        from sklearn.linear_model import Lasso

        return Lasso(alpha=0.01, random_state=random_seed)

    if is_classification:
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=50, random_state=random_seed, n_jobs=-1
        )
    from sklearn.ensemble import RandomForestRegressor

    return RandomForestRegressor(
        n_estimators=50, random_state=random_seed, n_jobs=-1
    )


class RFEMethod(FSMethod):
    """Рекурсивное исключение признаков (RFE).

    Обёртка над `sklearn.feature_selection.RFE`: эстиматор обучается,
    наименее важный признак отбрасывается, процедура повторяется до
    достижения целевого числа признаков. Базовый эстиматор — случайный
    лес (под задачу, определённую по `y_train`).
    """

    name = "rfe"
    group = "wrapper"
    supported_tasks = ["classification", "regression"]

    @classmethod
    def method_info(cls) -> MethodInfo:
        """Возвращает метаданные метода `rfe`."""
        return MethodInfo(
            name=cls.name,
            description=(
                "Рекурсивное исключение признаков (RFE) поверх случайного "
                "леса — наименее важный признак отбрасывается итеративно."
            ),
            requires_target=True,
            supports_multiclass=True,
            output_type="subset",
            hyperparameters={
                "fraction": HyperParam(
                    kind="float", default=0.5, low=0.1, high=1.0
                ),
                "step": HyperParam(
                    kind="float", default=0.1, low=0.05, high=0.5
                ),
            },
        )

    def fit_select(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        **hyperparams: Any,
    ) -> list[str]:
        """Отбирает признаки рекурсивным исключением.

        Args:
            X_train: Обучающая выборка признаков.
            y_train: Обучающая целевая переменная.
            **hyperparams: ``fraction`` — доля отбираемых признаков;
                ``step`` — доля признаков, отбрасываемых за итерацию;
                ``random_seed`` — зерно рандома эстиматора.

        Returns:
            Список имён отобранных признаков.
        """
        from sklearn.feature_selection import RFE

        fraction = float(hyperparams.get("fraction", 0.5))
        step = float(hyperparams.get("step", 0.1))
        random_seed = hyperparams.get("random_seed")

        n_features = _resolve_n_features(fraction, X_train.shape[1])
        estimator = _make_estimator(is_classification(y_train), random_seed)
        selector = RFE(
            estimator=estimator,
            n_features_to_select=n_features,
            step=step,
        )
        selector.fit(X_train, y_train)
        mask = selector.get_support()
        return [
            c
            for c, keep in zip(X_train.columns, mask, strict=True)
            if keep
        ]


class SequentialFeatureSelectorMethod(FSMethod):
    """Последовательный отбор признаков (forward/backward).

    Обёртка над `sklearn.feature_selection.SequentialFeatureSelector`:
    жадно добавляет (или убирает) по одному признаку, оценивая прирост
    качества кросс-валидацией.
    """

    name = "sequential_feature_selector"
    group = "wrapper"
    supported_tasks = ["classification", "regression"]

    @classmethod
    def method_info(cls) -> MethodInfo:
        """Возвращает метаданные метода `sequential_feature_selector`."""
        return MethodInfo(
            name=cls.name,
            description=(
                "Последовательный жадный отбор признаков (forward/backward) "
                "с CV-оценкой прироста качества на каждом шаге."
            ),
            requires_target=True,
            supports_multiclass=True,
            output_type="subset",
            hyperparameters={
                "fraction": HyperParam(
                    kind="float", default=0.5, low=0.1, high=0.9
                ),
                "direction": HyperParam(
                    kind="categorical",
                    default="forward",
                    choices=["forward", "backward"],
                ),
            },
        )

    def fit_select(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        **hyperparams: Any,
    ) -> list[str]:
        """Отбирает признаки последовательным жадным перебором.

        Args:
            X_train: Обучающая выборка признаков.
            y_train: Обучающая целевая переменная.
            **hyperparams: ``fraction`` — доля отбираемых признаков;
                ``direction`` — ``"forward"`` | ``"backward"``;
                ``random_seed`` — зерно рандома эстиматора.

        Returns:
            Список имён отобранных признаков.
        """
        from sklearn.feature_selection import SequentialFeatureSelector

        fraction = float(hyperparams.get("fraction", 0.5))
        direction = str(hyperparams.get("direction", "forward"))
        random_seed = hyperparams.get("random_seed")

        n_total = X_train.shape[1]
        # SequentialFeatureSelector требует 1 <= n < n_total.
        n_features = _resolve_n_features(fraction, n_total)
        n_features = min(n_features, max(1, n_total - 1))

        estimator = _make_estimator(is_classification(y_train), random_seed)
        selector = SequentialFeatureSelector(
            estimator=estimator,
            n_features_to_select=n_features,
            direction=direction,
            cv=3,
        )
        selector.fit(X_train, y_train)
        mask = selector.get_support()
        return [
            c
            for c, keep in zip(X_train.columns, mask, strict=True)
            if keep
        ]


class BorutaMethod(FSMethod):
    """Отбор признаков методом Boruta.

    Boruta сравнивает важность реальных признаков с важностью их
    случайно перемешанных «теней»: признак подтверждается, если стабильно
    превосходит лучшую тень. Реализация опирается на опциональный пакет
    ``boruta`` (`BorutaPy` поверх случайного леса).

    Пакет ``boruta`` импортируется лениво внутри `fit_select`; его
    отсутствие не ломает импорт модуля. `check_availability` помечает
    метод недоступным, если пакет не установлен (план E-005, OQ-2; R-2).
    """

    name = "boruta"
    group = "wrapper"
    supported_tasks = ["classification", "regression"]

    @classmethod
    def check_availability(cls) -> str | None:
        """Проверяет наличие опционального пакета ``boruta``.

        Returns:
            ``None``, если пакет ``boruta`` установлен; иначе — строка с
            причиной недоступности.
        """
        import importlib.util

        if importlib.util.find_spec("boruta") is None:
            return (
                "требуется пакет 'boruta' (extra [methods]); "
                "установите: pip install boruta"
            )
        return None

    @classmethod
    def method_info(cls) -> MethodInfo:
        """Возвращает метаданные метода `boruta`."""
        return MethodInfo(
            name=cls.name,
            description=(
                "Алгоритм Boruta: признак подтверждается, если его важность "
                "стабильно превосходит важность случайных признаков-теней."
            ),
            requires_target=True,
            supports_multiclass=True,
            output_type="subset",
            hyperparameters={
                "max_iter": HyperParam(
                    kind="int", default=100, low=20, high=250
                ),
                "include_tentative": HyperParam(
                    kind="categorical",
                    default=False,
                    choices=[True, False],
                ),
            },
        )

    def fit_select(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        **hyperparams: Any,
    ) -> list[str]:
        """Отбирает признаки алгоритмом Boruta.

        Args:
            X_train: Обучающая выборка признаков.
            y_train: Обучающая целевая переменная.
            **hyperparams: ``max_iter`` — число итераций Boruta;
                ``include_tentative`` — включать ли «неопределённые»
                признаки; ``random_seed`` — зерно рандома.

        Returns:
            Список имён отобранных признаков. Если Boruta не подтвердил ни
            одного признака — возвращается исходный набор (пустой отбор
            бессмыслен для downstream-оценки).

        Raises:
            ImportError: Если опциональный пакет ``boruta`` не установлен.
        """
        from boruta import BorutaPy

        max_iter = int(hyperparams.get("max_iter", 100))
        include_tentative = bool(hyperparams.get("include_tentative", False))
        random_seed = hyperparams.get("random_seed")

        estimator = _make_estimator(is_classification(y_train), random_seed)
        selector = BorutaPy(
            estimator=estimator,
            n_estimators="auto",
            max_iter=max_iter,
            random_state=random_seed,
        )
        selector.fit(X_train.to_numpy(), y_train.to_numpy())

        columns = list(X_train.columns)
        confirmed = [
            c
            for c, keep in zip(columns, selector.support_, strict=True)
            if keep
        ]
        if include_tentative:
            confirmed += [
                c
                for c, keep in zip(
                    columns, selector.support_weak_, strict=True
                )
                if keep
            ]
        return confirmed if confirmed else columns


class StabilitySelectionMethod(FSMethod):
    """Отбор признаков методом Stability Selection.

    На множестве бутстрэп-подвыборок обучается L1-регуляризованная модель
    (Lasso / логистическая регрессия с L1); для каждого признака считается
    частота попадания в модель с ненулевым коэффициентом. Отбираются
    признаки с частотой не ниже порога ``selection_threshold``.

    Реализация опирается только на `scikit-learn` (опциональных
    зависимостей нет).
    """

    name = "stability_selection"
    group = "wrapper"
    supported_tasks = ["classification", "regression"]

    @classmethod
    def method_info(cls) -> MethodInfo:
        """Возвращает метаданные метода `stability_selection`."""
        return MethodInfo(
            name=cls.name,
            description=(
                "Stability Selection: частота попадания признака в "
                "L1-модель на бутстрэп-подвыборках; отбор по порогу частоты."
            ),
            requires_target=True,
            supports_multiclass=True,
            output_type="subset",
            hyperparameters={
                "n_bootstrap": HyperParam(
                    kind="int", default=50, low=20, high=200
                ),
                "sample_fraction": HyperParam(
                    kind="float", default=0.5, low=0.3, high=0.8
                ),
                "selection_threshold": HyperParam(
                    kind="float", default=0.6, low=0.3, high=0.9
                ),
            },
        )

    def fit_select(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        **hyperparams: Any,
    ) -> list[str]:
        """Отбирает признаки по стабильности L1-модели на подвыборках.

        Args:
            X_train: Обучающая выборка признаков.
            y_train: Обучающая целевая переменная.
            **hyperparams: ``n_bootstrap`` — число бутстрэп-подвыборок;
                ``sample_fraction`` — доля строк в подвыборке;
                ``selection_threshold`` — порог частоты отбора;
                ``random_seed`` — зерно рандома.

        Returns:
            Список имён признаков с частотой отбора не ниже порога. Если
            ни один признак не прошёл порог — возвращается признак с
            максимальной частотой (пустой отбор бессмыслен для оценки).
        """
        n_bootstrap = int(hyperparams.get("n_bootstrap", 50))
        sample_fraction = float(hyperparams.get("sample_fraction", 0.5))
        threshold = float(hyperparams.get("selection_threshold", 0.6))
        random_seed = hyperparams.get("random_seed")

        columns = list(X_train.columns)
        n_rows = X_train.shape[0]
        sample_size = max(2, int(round(sample_fraction * n_rows)))
        is_clf = is_classification(y_train)

        rng = np.random.default_rng(random_seed)
        X_values = X_train.to_numpy()
        y_values = y_train.to_numpy()
        counts = np.zeros(len(columns), dtype=float)

        for i in range(n_bootstrap):
            idx = rng.choice(n_rows, size=sample_size, replace=True)
            y_sub = y_values[idx]
            # Пропускаем вырожденную подвыборку (для классификации —
            # один класс): L1-модель на ней неинформативна.
            if is_clf and len(np.unique(y_sub)) < 2:
                continue
            # Зерно эстиматора детерминированно меняется по итерациям —
            # воспроизводимо при фиксированном random_seed (К2).
            seed = None if random_seed is None else int(random_seed) + i
            estimator = _make_estimator(is_clf, seed, linear=True)
            estimator.fit(X_values[idx], y_sub)
            coef = np.asarray(estimator.coef_).reshape(-1, len(columns))
            nonzero = np.any(np.abs(coef) > 1e-10, axis=0)
            counts += nonzero.astype(float)

        frequencies = counts / n_bootstrap
        selected = [
            c
            for c, freq in zip(columns, frequencies, strict=True)
            if freq >= threshold
        ]
        if selected:
            return selected
        return [columns[int(np.argmax(frequencies))]]


# --- Саморегистрация built-in wrapper-методов (план E-005, A-1) ------------
register("rfe", RFEMethod, "wrapper")
register(
    "sequential_feature_selector",
    SequentialFeatureSelectorMethod,
    "wrapper",
)
register("boruta", BorutaMethod, "wrapper")
register("stability_selection", StabilitySelectionMethod, "wrapper")
