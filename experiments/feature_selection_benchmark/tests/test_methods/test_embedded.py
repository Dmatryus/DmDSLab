"""Тесты embedded-методов отбора признаков (E-005.4).

Покрывает три метода группы ``embedded``: `lasso`, `catboost_select`,
`tree_gain_importance` — реальный прогон `fit_select`, declarative-
пространство поиска, саморегистрацию в реестре.
"""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

from feature_selection_benchmark.methods import embedded_methods, registry
from feature_selection_benchmark.methods.base import HyperParam, MethodInfo
from feature_selection_benchmark.methods.embedded_methods import (
    CatBoostSelectMethod,
    LassoMethod,
    TreeGainImportanceMethod,
)

_CATBOOST_AVAILABLE = importlib.util.find_spec("catboost") is not None
_catboost_required = pytest.mark.skipif(
    not _CATBOOST_AVAILABLE, reason="catboost не установлен"
)


# --------------------------------------------------------------------------
# Фикстуры данных: информативные признаки + шум.
# --------------------------------------------------------------------------


@pytest.fixture
def classification_data() -> tuple[pd.DataFrame, pd.Series]:
    """Датасет классификации: 3 информативных признака + 4 шумовых."""
    rng = np.random.default_rng(42)
    n = 200
    informative = rng.normal(size=(n, 3))
    logits = informative @ np.array([2.5, -2.0, 1.5])
    y = pd.Series((logits + rng.normal(scale=0.3, size=n) > 0).astype(int))
    noise = rng.normal(size=(n, 4))
    data = np.hstack([informative, noise])
    cols = [f"inf_{i}" for i in range(3)] + [f"noise_{i}" for i in range(4)]
    return pd.DataFrame(data, columns=cols), y


@pytest.fixture
def regression_data() -> tuple[pd.DataFrame, pd.Series]:
    """Датасет регрессии: 3 информативных признака + 4 шумовых."""
    rng = np.random.default_rng(7)
    n = 200
    informative = rng.normal(size=(n, 3))
    target = informative @ np.array([3.0, -2.0, 1.0])
    y = pd.Series(target + rng.normal(scale=0.2, size=n))
    noise = rng.normal(size=(n, 4))
    data = np.hstack([informative, noise])
    cols = [f"inf_{i}" for i in range(3)] + [f"noise_{i}" for i in range(4)]
    return pd.DataFrame(data, columns=cols), y


# --------------------------------------------------------------------------
# Общие свойства метаданных и пространства поиска.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cls",
    [LassoMethod, CatBoostSelectMethod, TreeGainImportanceMethod],
)
def test_method_info_contract(cls) -> None:
    """`method_info` возвращает валидный `MethodInfo` с пространством поиска."""
    info = cls.method_info()
    assert isinstance(info, MethodInfo)
    assert info.name == cls.name
    assert info.description
    assert info.output_type in ("ranking", "subset")
    assert isinstance(info.hyperparameters, dict)
    assert info.hyperparameters, "embedded-метод объявляет гиперпараметры"
    for hp in info.hyperparameters.values():
        assert isinstance(hp, HyperParam)


def test_self_registration() -> None:
    """Импорт модуля саморегистрирует все три embedded-метода.

    Модуль `embedded_methods` уже импортирован на уровне модуля теста;
    его строки ``register(...)`` выполнились как побочный эффект импорта.
    """
    assert embedded_methods.__all__  # модуль импортирован
    for name in ("lasso", "catboost_select", "tree_gain_importance"):
        assert registry.is_registered(name)
        info = next(
            i for i in registry.list_registered() if i.name == name
        )
        assert info.group == "embedded"
        assert info.hyperparameters


# --------------------------------------------------------------------------
# lasso — scikit-learn, всегда доступен.
# --------------------------------------------------------------------------


def test_lasso_classification(classification_data) -> None:
    """`lasso` на классификации возвращает валидное непустое подмножество."""
    X, y = classification_data
    selected = LassoMethod().fit_select(X, y, alpha=0.05, l1_ratio=1.0)
    assert selected
    assert set(selected).issubset(set(X.columns))
    assert len(selected) == len(set(selected))


def test_lasso_regression(regression_data) -> None:
    """`lasso` на регрессии возвращает валидное непустое подмножество."""
    X, y = regression_data
    selected = LassoMethod().fit_select(X, y, alpha=0.1, l1_ratio=0.7)
    assert selected
    assert set(selected).issubset(set(X.columns))


def test_lasso_regression_picks_informative(regression_data) -> None:
    """`lasso` на регрессии удерживает информативные признаки."""
    X, y = regression_data
    selected = LassoMethod().fit_select(X, y, alpha=0.1, l1_ratio=1.0)
    informative = {"inf_0", "inf_1", "inf_2"}
    assert informative & set(selected), "хотя бы один информативный отобран"


def test_lasso_strong_regularization_nonempty(regression_data) -> None:
    """Даже при сильной регуляризации `lasso` не возвращает пустой набор."""
    X, y = regression_data
    selected = LassoMethod().fit_select(X, y, alpha=1e3, l1_ratio=1.0)
    assert len(selected) >= 1


def test_lasso_defaults(regression_data) -> None:
    """`lasso` работает на дефолтах (режим n_trials=0)."""
    X, y = regression_data
    defaults = {
        name: hp.default
        for name, hp in LassoMethod.method_info().hyperparameters.items()
    }
    selected = LassoMethod().fit_select(X, y, **defaults)
    assert set(selected).issubset(set(X.columns))


# --------------------------------------------------------------------------
# catboost_select / tree_gain_importance — опц. зависимость catboost.
# --------------------------------------------------------------------------


def test_catboost_methods_availability() -> None:
    """`check_availability` согласован с фактическим наличием catboost."""
    expected = None if _CATBOOST_AVAILABLE else str
    for cls in (CatBoostSelectMethod, TreeGainImportanceMethod):
        reason = cls.check_availability()
        if _CATBOOST_AVAILABLE:
            assert reason is None
        else:
            assert isinstance(reason, expected)
            assert "catboost" in reason


@_catboost_required
def test_catboost_select_classification(classification_data) -> None:
    """`catboost_select` на классификации даёт усечённое подмножество."""
    X, y = classification_data
    selected = CatBoostSelectMethod().fit_select(
        X, y, select_fraction=0.5, iterations=60, random_seed=42
    )
    assert selected
    assert set(selected).issubset(set(X.columns))
    assert len(selected) < len(X.columns)
    assert len(selected) == len(set(selected))


@_catboost_required
def test_catboost_select_regression(regression_data) -> None:
    """`catboost_select` на регрессии даёт валидное подмножество."""
    X, y = regression_data
    selected = CatBoostSelectMethod().fit_select(
        X, y, select_fraction=0.4, iterations=60, random_seed=7
    )
    assert selected
    assert set(selected).issubset(set(X.columns))
    assert len(selected) < len(X.columns)


@_catboost_required
def test_catboost_select_reproducible(regression_data) -> None:
    """`catboost_select` с одним `random_seed` детерминирован (К2)."""
    X, y = regression_data
    kwargs = dict(select_fraction=0.5, iterations=60, random_seed=123)
    first = CatBoostSelectMethod().fit_select(X, y, **kwargs)
    second = CatBoostSelectMethod().fit_select(X, y, **kwargs)
    assert first == second


@_catboost_required
def test_tree_gain_importance_classification(classification_data) -> None:
    """`tree_gain_importance` на классификации даёт top-k подмножество."""
    X, y = classification_data
    selected = TreeGainImportanceMethod().fit_select(
        X, y, top_k_fraction=0.5, iterations=60, random_seed=42
    )
    assert selected
    assert set(selected).issubset(set(X.columns))
    expected_k = max(1, round(len(X.columns) * 0.5))
    assert len(selected) == expected_k


@_catboost_required
def test_tree_gain_importance_regression(regression_data) -> None:
    """`tree_gain_importance` на регрессии ранжирует информативные выше."""
    X, y = regression_data
    selected = TreeGainImportanceMethod().fit_select(
        X, y, top_k_fraction=3 / 7, iterations=80, random_seed=7
    )
    informative = {"inf_0", "inf_1", "inf_2"}
    # Top-3 по важности должны включать хотя бы 2 информативных признака.
    assert len(informative & set(selected)) >= 2


@_catboost_required
def test_tree_gain_importance_reproducible(regression_data) -> None:
    """`tree_gain_importance` с одним `random_seed` детерминирован (К2)."""
    X, y = regression_data
    kwargs = dict(top_k_fraction=0.5, iterations=60, random_seed=99)
    first = TreeGainImportanceMethod().fit_select(X, y, **kwargs)
    second = TreeGainImportanceMethod().fit_select(X, y, **kwargs)
    assert first == second


@_catboost_required
def test_catboost_methods_defaults(regression_data) -> None:
    """catboost-методы работают на дефолтах (режим n_trials=0)."""
    X, y = regression_data
    for cls in (CatBoostSelectMethod, TreeGainImportanceMethod):
        defaults = {
            name: hp.default
            for name, hp in cls.method_info().hyperparameters.items()
        }
        selected = cls().fit_select(X, y, **defaults)
        assert selected
        assert set(selected).issubset(set(X.columns))
