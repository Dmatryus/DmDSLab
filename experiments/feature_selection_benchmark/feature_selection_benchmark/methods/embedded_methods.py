"""FS-методы группы ``embedded`` — отбор встроен в обучение модели.

Сюда входят Lasso/ElasticNet, CatBoost.select_features, важности по
приросту в деревьях (ARCHITECTURE §2).

Каждый метод:

- реализует `fit_select`, возвращающий список имён отобранных признаков;
- объявляет declarative-пространство поиска гиперпараметров через
  classmethod `method_info` (контракт E-005.1 / ADR `0002`);
- саморегистрируется в реестре строкой `register(...)` в конце модуля.

Опциональная зависимость `catboost` импортируется **лениво** внутри
методов `CatBoostSelectMethod` / `TreeGainImportanceMethod`; при её
отсутствии метод регистрируется, но помечается недоступным через
`check_availability` (OQ-2).
"""

from __future__ import annotations

import importlib.util
from typing import Any

import numpy as np
import pandas as pd

from ._utils import is_classification
from .base import FSMethod, HyperParam, MethodInfo
from .registry import register

__all__ = [
    "LassoMethod",
    "CatBoostSelectMethod",
    "TreeGainImportanceMethod",
]


def _catboost_unavailable_reason() -> str | None:
    """Возвращает причину недоступности `catboost` или ``None``.

    Returns:
        ``None``, если пакет `catboost` импортируем; иначе — строка с
        причиной для `MethodInfo.unavailable_reason`.
    """
    if importlib.util.find_spec("catboost") is None:
        return "требуется пакет 'catboost' (extra [methods])"
    return None


class LassoMethod(FSMethod):
    """Отбор признаков через L1-регуляризацию (Lasso / ElasticNet).

    Обучает линейную модель с L1-регуляризацией (для регрессии —
    `ElasticNet`, для классификации — `LogisticRegression` с
    `solver="saga"` и `l1_ratio`) и отбирает признаки с ненулевыми
    коэффициентами. Сила регуляризации и доля L1 (`l1_ratio`) —
    гиперпараметры.
    """

    name = "lasso"
    group = "embedded"
    supported_tasks = ["classification", "regression"]

    @classmethod
    def method_info(cls) -> MethodInfo:
        """Возвращает метаданные метода и пространство поиска."""
        return MethodInfo(
            name=cls.name,
            description=(
                "Отбор признаков через L1-регуляризацию (Lasso / ElasticNet "
                "/ L1-LogisticRegression): отбираются признаки с ненулевыми "
                "коэффициентами."
            ),
            requires_target=True,
            supports_multiclass=True,
            output_type="subset",
            hyperparameters={
                "alpha": HyperParam(
                    kind="float",
                    default=1.0,
                    low=1e-3,
                    high=1e2,
                    log=True,
                ),
                "l1_ratio": HyperParam(
                    kind="float",
                    default=1.0,
                    low=0.1,
                    high=1.0,
                ),
            },
        )

    def fit_select(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        **hyperparams: Any,
    ) -> list[str]:
        """Отбирает признаки с ненулевыми L1-коэффициентами.

        Args:
            X_train: Обучающая выборка признаков.
            y_train: Обучающая целевая переменная.
            **hyperparams: `alpha` — сила регуляризации; `l1_ratio` —
                доля L1 в ElasticNet (1.0 — чистый Lasso).

        Returns:
            Имена признаков с ненулевыми коэффициентами. Если все
            коэффициенты нулевые — fallback на признак с максимальным
            по модулю коэффициентом, чтобы не вернуть пустой набор.
        """
        from sklearn.linear_model import ElasticNet, LogisticRegression

        alpha = float(hyperparams.get("alpha", 1.0))
        l1_ratio = float(hyperparams.get("l1_ratio", 1.0))
        features = list(X_train.columns)

        if is_classification(y_train):
            # C — обратная сила регуляризации в LogisticRegression;
            # l1_ratio задаёт долю L1 (1.0 — чистый L1, как Lasso).
            model = LogisticRegression(
                C=1.0 / alpha,
                l1_ratio=l1_ratio,
                solver="saga",
                max_iter=2000,
            )
            model.fit(X_train, y_train)
            # coef_ — (n_classes, n_features) или (1, n_features).
            coef = np.abs(np.asarray(model.coef_)).sum(axis=0)
        else:
            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000)
            model.fit(X_train, y_train)
            coef = np.abs(np.asarray(model.coef_))

        selected = [
            feat
            for feat, c in zip(features, coef, strict=True)
            if c > 0.0
        ]
        if not selected:
            # Все коэффициенты обнулены — возвращаем самый сильный признак.
            selected = [features[int(np.argmax(coef))]]
        return selected


class CatBoostSelectMethod(FSMethod):
    """Отбор признаков через ``CatBoost.select_features``.

    Запускает встроенный в CatBoost рекурсивный алгоритм отбора
    признаков (`select_features`), который итеративно отбрасывает
    наименее важные признаки и оставляет лучшие `num_features_to_select`.

    Опциональная зависимость `catboost` импортируется лениво.
    """

    name = "catboost_select"
    group = "embedded"
    supported_tasks = ["classification", "regression"]

    @classmethod
    def check_availability(cls) -> str | None:
        """Проверяет наличие пакета `catboost`."""
        return _catboost_unavailable_reason()

    @classmethod
    def method_info(cls) -> MethodInfo:
        """Возвращает метаданные метода и пространство поиска."""
        return MethodInfo(
            name=cls.name,
            description=(
                "Отбор признаков встроенным алгоритмом "
                "CatBoost.select_features (рекурсивное отбрасывание "
                "наименее важных признаков)."
            ),
            requires_target=True,
            supports_multiclass=True,
            output_type="subset",
            hyperparameters={
                "select_fraction": HyperParam(
                    kind="float",
                    default=0.5,
                    low=0.1,
                    high=0.9,
                ),
                "iterations": HyperParam(
                    kind="int",
                    default=100,
                    low=50,
                    high=300,
                ),
            },
        )

    def fit_select(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        **hyperparams: Any,
    ) -> list[str]:
        """Отбирает признаки через `CatBoost.select_features`.

        Args:
            X_train: Обучающая выборка признаков.
            y_train: Обучающая целевая переменная.
            **hyperparams: `select_fraction` — доля признаков, которую
                нужно оставить; `iterations` — число итераций обучения
                CatBoost; `random_seed` — зерно для воспроизводимости.

        Returns:
            Имена отобранных признаков (минимум один).

        Raises:
            ImportError: Если пакет `catboost` не установлен.
        """
        from catboost import CatBoostClassifier, CatBoostRegressor, Pool

        select_fraction = float(hyperparams.get("select_fraction", 0.5))
        iterations = int(hyperparams.get("iterations", 100))
        random_seed = hyperparams.get("random_seed", None)
        features = list(X_train.columns)

        n_select = max(1, int(round(len(features) * select_fraction)))
        if n_select >= len(features):
            return features

        params: dict[str, Any] = {
            "iterations": iterations,
            "verbose": False,
            "allow_writing_files": False,
        }
        if random_seed is not None:
            params["random_seed"] = int(random_seed)

        if is_classification(y_train):
            model: Any = CatBoostClassifier(**params)
        else:
            model = CatBoostRegressor(**params)

        pool = Pool(data=X_train, label=y_train)
        summary = model.select_features(
            pool,
            features_for_select=list(range(len(features))),
            num_features_to_select=n_select,
            steps=3,
            train_final_model=False,
            verbose=False,
        )
        selected_names = summary.get("selected_features_names")
        if selected_names:
            return list(selected_names)
        # Fallback: трансляция индексов в имена.
        selected_idx = summary.get("selected_features", [])
        selected = [features[i] for i in selected_idx]
        return selected or [features[0]]


class TreeGainImportanceMethod(FSMethod):
    """Отбор признаков по важности прироста в деревьях.

    Обучает градиентный бустинг (CatBoost) и отбирает top-k признаков
    по важности типа `PredictionValuesChange` (вклад признака в
    предсказание). Ранжирующий метод — порог `top_k` переводит
    ранжирование в подмножество (план E-005, R-6).

    Опциональная зависимость `catboost` импортируется лениво.
    """

    name = "tree_gain_importance"
    group = "embedded"
    supported_tasks = ["classification", "regression"]

    @classmethod
    def check_availability(cls) -> str | None:
        """Проверяет наличие пакета `catboost`."""
        return _catboost_unavailable_reason()

    @classmethod
    def method_info(cls) -> MethodInfo:
        """Возвращает метаданные метода и пространство поиска."""
        return MethodInfo(
            name=cls.name,
            description=(
                "Отбор признаков по важности прироста в деревьях "
                "градиентного бустинга (CatBoost feature importance): "
                "top-k признаков по вкладу в предсказание."
            ),
            requires_target=True,
            supports_multiclass=True,
            output_type="ranking",
            hyperparameters={
                "top_k_fraction": HyperParam(
                    kind="float",
                    default=0.5,
                    low=0.1,
                    high=1.0,
                ),
                "iterations": HyperParam(
                    kind="int",
                    default=100,
                    low=50,
                    high=300,
                ),
            },
        )

    def fit_select(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        **hyperparams: Any,
    ) -> list[str]:
        """Отбирает top-k признаков по важности прироста в деревьях.

        Args:
            X_train: Обучающая выборка признаков.
            y_train: Обучающая целевая переменная.
            **hyperparams: `top_k_fraction` — доля топовых признаков по
                важности; `iterations` — число итераций CatBoost;
                `random_seed` — зерно для воспроизводимости.

        Returns:
            Имена top-k признаков, отсортированные по убыванию важности
            (минимум один).

        Raises:
            ImportError: Если пакет `catboost` не установлен.
        """
        from catboost import CatBoostClassifier, CatBoostRegressor

        top_k_fraction = float(hyperparams.get("top_k_fraction", 0.5))
        iterations = int(hyperparams.get("iterations", 100))
        random_seed = hyperparams.get("random_seed", None)
        features = list(X_train.columns)

        params: dict[str, Any] = {
            "iterations": iterations,
            "verbose": False,
            "allow_writing_files": False,
        }
        if random_seed is not None:
            params["random_seed"] = int(random_seed)

        if is_classification(y_train):
            model: Any = CatBoostClassifier(**params)
        else:
            model = CatBoostRegressor(**params)

        model.fit(X_train, y_train)
        importances = np.asarray(model.get_feature_importance())

        n_select = max(1, int(round(len(features) * top_k_fraction)))
        # Индексы признаков по убыванию важности (детерминированно).
        order = np.argsort(-importances, kind="stable")
        return [features[i] for i in order[:n_select]]


register("lasso", LassoMethod, "embedded")
register("catboost_select", CatBoostSelectMethod, "embedded")
register("tree_gain_importance", TreeGainImportanceMethod, "embedded")
