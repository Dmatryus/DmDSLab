"""FS-методы групп ``shap`` и ``permutation`` — отбор по важностям на
основе SHAP-значений и перестановок.

Сюда входят (ARCHITECTURE §2):

- :class:`ShapImportanceMethod` — отбор по ``mean(|SHAP|)`` (группа
  ``shap``); опц. зависимость — пакет ``shap``;
- :class:`PermutationImportanceMethod` — permutation importance Бреймана,
  ``sklearn.inspection.permutation_importance`` (группа ``permutation``);
- :class:`NullImportanceMethod` — null importance / target permutation
  (группа ``permutation``); опц. зависимость — пакет
  ``target-permutation-importances``.

SHAP используется здесь **только как метод отбора признаков**, не как
инструмент объяснимости модели (CLAUDE NEVER §4).

Пространство поиска гиперпараметров каждый метод объявляет декларативно
(словарь ``имя → HyperParam``) — слой ``methods/`` не импортирует
``optuna`` (ADR `0002`, контракт E-005.1).

Опциональные зависимости (``shap``,
``target-permutation-importances``) импортируются **лениво** — внутри
`fit_select` / `check_availability`. Если пакет не установлен, метод всё
равно регистрируется (см. конец файла), но помечается недоступным через
`FSMethod.check_availability` и исключается из дефолтного набора прогона
(план E-005, OQ-2).
"""

from __future__ import annotations

import importlib.util
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance

from .base import FSMethod, HyperParam, MethodInfo
from .registry import register

__all__ = [
    "ShapImportanceMethod",
    "PermutationImportanceMethod",
    "NullImportanceMethod",
]


def _is_classification(y_train: pd.Series) -> bool:
    """Грубое определение типа задачи по целевой переменной.

    Метод применяется, когда тип задачи не передан явно: целочисленная
    (или объектная / категориальная) `y` с небольшим числом уникальных
    значений трактуется как классификация, иначе — регрессия.

    Args:
        y_train: Целевая переменная.

    Returns:
        ``True`` для задачи классификации, ``False`` — для регрессии.
    """
    if y_train.dtype == object or isinstance(
        y_train.dtype, pd.CategoricalDtype
    ):
        return True
    n_unique = y_train.nunique()
    is_integer = pd.api.types.is_integer_dtype(y_train)
    return bool(is_integer and n_unique <= max(20, int(0.05 * len(y_train))))


def _resolve_task(y_train: pd.Series, task: str | None) -> str:
    """Возвращает тип задачи: явный аргумент или автоопределение.

    Args:
        y_train: Целевая переменная (для автоопределения).
        task: Явно переданный тип задачи или ``None``.

    Returns:
        ``"classification"`` или ``"regression"``.
    """
    if task in ("classification", "regression"):
        return task
    return "classification" if _is_classification(y_train) else "regression"


def _select_top_k(
    importances: pd.Series, top_k: int, feature_names: list[str]
) -> list[str]:
    """Отбирает ``top_k`` признаков с наибольшей важностью.

    Порядок детерминирован: сортировка по убыванию важности, ties
    разрешаются по исходному порядку признаков (критерий К2).

    Args:
        importances: Важности признаков (индекс — имена признаков).
        top_k: Число отбираемых признаков (обрезается до доступного
            количества; не меньше 1).
        feature_names: Имена признаков в исходном порядке.

    Returns:
        Список имён отобранных признаков.
    """
    k = max(1, min(int(top_k), len(feature_names)))
    order = {name: idx for idx, name in enumerate(feature_names)}
    ranked = sorted(
        feature_names,
        key=lambda name: (-float(importances.get(name, 0.0)), order[name]),
    )
    return ranked[:k]


class ShapImportanceMethod(FSMethod):
    """Отбор признаков по SHAP-важностям.

    Обучается простая tree-модель (`RandomForest`), для неё считаются
    SHAP-значения через `shap.TreeExplainer`; признаки ранжируются по
    ``mean(|SHAP|)`` и отбирается ``top_k`` (feature_selection_guide §4).

    Опциональная зависимость — пакет ``shap``; импортируется лениво.
    """

    name = "shap_importance"
    group = "shap"
    supported_tasks = ["classification", "regression"]

    @classmethod
    def check_availability(cls) -> str | None:
        """Проверяет наличие опц. пакета ``shap``.

        Returns:
            ``None``, если ``shap`` установлен; иначе строка-причина.
        """
        if importlib.util.find_spec("shap") is None:
            return "требуется пакет 'shap' (extra [methods])"
        return None

    @classmethod
    def method_info(cls) -> MethodInfo:
        """Метаданные метода и declarative-пространство поиска.

        Returns:
            `MethodInfo` метода (поля группы / доступности проставляет
            реестр).
        """
        return MethodInfo(
            name=cls.name,
            description=(
                "Отбор признаков по SHAP-важностям: tree-модель + "
                "ранжирование по mean(|SHAP|), top-k."
            ),
            requires_target=True,
            supports_multiclass=True,
            output_type="ranking",
            hyperparameters={
                "top_k": HyperParam(
                    kind="int", default=10, low=1, high=100
                ),
                "n_estimators": HyperParam(
                    kind="int", default=100, low=50, high=300
                ),
            },
        )

    def fit_select(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        **hyperparams: Any,
    ) -> list[str]:
        """Отбирает признаки по SHAP-важностям.

        Args:
            X_train: Обучающая выборка признаков.
            y_train: Обучающая целевая переменная.
            **hyperparams: ``top_k`` (число признаков), ``n_estimators``
                (размер леса), ``task`` (``None`` → автоопределение),
                ``random_seed``.

        Returns:
            Список имён отобранных признаков.

        Raises:
            ImportError: Если пакет ``shap`` не установлен.
        """
        import shap  # ленивый импорт опц. зависимости

        feature_names = list(X_train.columns)
        top_k = hyperparams.get("top_k", 10)
        n_estimators = int(hyperparams.get("n_estimators", 100))
        random_seed = hyperparams.get("random_seed")
        task = _resolve_task(y_train, hyperparams.get("task"))

        if task == "classification":
            model = RandomForestClassifier(
                n_estimators=n_estimators, random_state=random_seed
            )
        else:
            model = RandomForestRegressor(
                n_estimators=n_estimators, random_state=random_seed
            )
        model.fit(X_train, y_train)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train)
        # shap_values может быть списком (по классам) или ndarray;
        # сводим к важности одного скаляра на признак.
        sv = np.asarray(shap_values)
        # Усредняем |SHAP| по всем осям, кроме оси признаков.
        feature_axis = sv.shape.index(len(feature_names))
        abs_mean = np.abs(sv).mean(
            axis=tuple(i for i in range(sv.ndim) if i != feature_axis)
        )
        importances = pd.Series(abs_mean, index=feature_names)
        return _select_top_k(importances, top_k, feature_names)


class PermutationImportanceMethod(FSMethod):
    """Отбор признаков по важности перестановок (permutation importance).

    Обучается tree-модель (`RandomForest`), затем
    `sklearn.inspection.permutation_importance` измеряет падение качества
    при перемешивании каждого признака; отбирается ``top_k`` признаков с
    наибольшей важностью (feature_selection_guide §5).

    Опциональных зависимостей нет — метод доступен всегда.
    """

    name = "permutation_importance"
    group = "permutation"
    supported_tasks = ["classification", "regression"]

    @classmethod
    def method_info(cls) -> MethodInfo:
        """Метаданные метода и declarative-пространство поиска.

        Returns:
            `MethodInfo` метода (поля группы / доступности проставляет
            реестр).
        """
        return MethodInfo(
            name=cls.name,
            description=(
                "Permutation importance Бреймана: падение качества "
                "tree-модели при перемешивании признака, top-k."
            ),
            requires_target=True,
            supports_multiclass=True,
            output_type="ranking",
            hyperparameters={
                "top_k": HyperParam(
                    kind="int", default=10, low=1, high=100
                ),
                "n_repeats": HyperParam(
                    kind="int", default=10, low=3, high=30
                ),
                "n_estimators": HyperParam(
                    kind="int", default=100, low=50, high=300
                ),
            },
        )

    def fit_select(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        **hyperparams: Any,
    ) -> list[str]:
        """Отбирает признаки по permutation importance.

        Args:
            X_train: Обучающая выборка признаков.
            y_train: Обучающая целевая переменная.
            **hyperparams: ``top_k`` (число признаков), ``n_repeats``
                (число перестановок на признак), ``n_estimators``
                (размер леса), ``task`` (``None`` → автоопределение),
                ``random_seed``.

        Returns:
            Список имён отобранных признаков.
        """
        feature_names = list(X_train.columns)
        top_k = hyperparams.get("top_k", 10)
        n_repeats = int(hyperparams.get("n_repeats", 10))
        n_estimators = int(hyperparams.get("n_estimators", 100))
        random_seed = hyperparams.get("random_seed")
        task = _resolve_task(y_train, hyperparams.get("task"))

        if task == "classification":
            model = RandomForestClassifier(
                n_estimators=n_estimators, random_state=random_seed
            )
        else:
            model = RandomForestRegressor(
                n_estimators=n_estimators, random_state=random_seed
            )
        model.fit(X_train, y_train)

        result = permutation_importance(
            model,
            X_train,
            y_train,
            n_repeats=n_repeats,
            random_state=random_seed,
        )
        importances = pd.Series(
            result.importances_mean, index=feature_names
        )
        return _select_top_k(importances, top_k, feature_names)


class NullImportanceMethod(FSMethod):
    """Отбор признаков по null-важностям (target permutation).

    Идея (feature_selection_guide §6): сравнить «actual» важности с
    распределением важностей, полученных при перемешанной целевой
    переменной; признак значим, если actual важность стабильно выше
    null-распределения. Реализация — пакет
    ``target-permutation-importances``.

    Опциональная зависимость — пакет ``target-permutation-importances``;
    импортируется лениво.
    """

    name = "null_importance"
    group = "permutation"
    supported_tasks = ["classification", "regression"]

    @classmethod
    def check_availability(cls) -> str | None:
        """Проверяет наличие опц. пакета ``target-permutation-importances``.

        Returns:
            ``None``, если пакет установлен; иначе строка-причина.
        """
        if importlib.util.find_spec("target_permutation_importances") is None:
            return (
                "требуется пакет 'target-permutation-importances' "
                "(extra [methods])"
            )
        return None

    @classmethod
    def method_info(cls) -> MethodInfo:
        """Метаданные метода и declarative-пространство поиска.

        Returns:
            `MethodInfo` метода (поля группы / доступности проставляет
            реестр).
        """
        return MethodInfo(
            name=cls.name,
            description=(
                "Null importance / target permutation: сравнение actual "
                "важностей с распределением при перемешанной y, top-k."
            ),
            requires_target=True,
            supports_multiclass=True,
            output_type="ranking",
            hyperparameters={
                "top_k": HyperParam(
                    kind="int", default=10, low=1, high=100
                ),
                "num_iterations": HyperParam(
                    kind="int", default=20, low=5, high=100
                ),
                "n_estimators": HyperParam(
                    kind="int", default=100, low=50, high=300
                ),
            },
        )

    def fit_select(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        **hyperparams: Any,
    ) -> list[str]:
        """Отбирает признаки по null-важностям (target permutation).

        Args:
            X_train: Обучающая выборка признаков.
            y_train: Обучающая целевая переменная.
            **hyperparams: ``top_k`` (число признаков),
                ``num_iterations`` (число перестановок цели),
                ``n_estimators`` (размер леса), ``task`` (``None`` →
                автоопределение), ``random_seed``.

        Returns:
            Список имён отобранных признаков.

        Raises:
            ImportError: Если пакет ``target-permutation-importances``
                не установлен.
        """
        # ленивый импорт опц. зависимости
        import target_permutation_importances as tpi

        feature_names = list(X_train.columns)
        top_k = hyperparams.get("top_k", 10)
        num_iterations = int(hyperparams.get("num_iterations", 20))
        n_estimators = int(hyperparams.get("n_estimators", 100))
        random_seed = hyperparams.get("random_seed")
        task = _resolve_task(y_train, hyperparams.get("task"))

        if task == "classification":
            model_cls = RandomForestClassifier
        else:
            model_cls = RandomForestRegressor

        result = tpi.compute(
            model_cls=model_cls,
            model_cls_params={
                "n_estimators": n_estimators,
                "random_state": random_seed,
            },
            model_fit_params={},
            X=X_train,
            y=y_train,
            num_actual_runs=2,
            num_random_runs=num_iterations,
        )
        # result — DataFrame с колонкой feature и колонкой-важностью;
        # имя колонки важности зависит от версии пакета — берём
        # последнюю числовую колонку.
        result = result.set_index("feature")
        score_col = result.select_dtypes(include="number").columns[-1]
        importances = result[score_col].reindex(feature_names).fillna(0.0)
        return _select_top_k(importances, top_k, feature_names)


# --- Саморегистрация методов группы (план E-005, A-1) -----------------
# Группы РАЗНЫЕ: shap_importance → "shap"; permutation_importance и
# null_importance → "permutation".
register(ShapImportanceMethod.name, ShapImportanceMethod, group="shap")
register(
    PermutationImportanceMethod.name,
    PermutationImportanceMethod,
    group="permutation",
)
register(
    NullImportanceMethod.name, NullImportanceMethod, group="permutation"
)
