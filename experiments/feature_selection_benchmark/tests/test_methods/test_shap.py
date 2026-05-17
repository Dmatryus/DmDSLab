"""Тесты SHAP/permutation-методов отбора признаков (E-005.5).

Покрывают три метода ``methods/shap_methods.py``:

- ``permutation_importance`` — опц. зависимостей нет, тесты реальные;
- ``shap_importance`` — опц. пакет ``shap``; тесты помечаются ``skip``,
  если пакет не установлен;
- ``null_importance`` — опц. пакет ``target-permutation-importances``;
  тесты помечаются ``skip`` при отсутствии пакета.

Проверяется: реализация `fit_select` возвращает валидное подмножество
признаков; саморегистрация в реестре с верной группой; declarative-
пространство поиска объявлено по контракту E-005.1; методы с опц.
зависимостями корректно деградируют (``available=False``) при отсутствии
пакета.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

from feature_selection_benchmark.methods import registry
from feature_selection_benchmark.methods.base import HyperParam, MethodInfo
from feature_selection_benchmark.methods.shap_methods import (
    NullImportanceMethod,
    PermutationImportanceMethod,
    ShapImportanceMethod,
)

_HAS_SHAP = importlib.util.find_spec("shap") is not None
_HAS_TPI = (
    importlib.util.find_spec("target_permutation_importances") is not None
)


@pytest.fixture
def classification_data() -> tuple[pd.DataFrame, pd.Series]:
    """Синтетический датасет классификации с информативными и шумовыми
    признаками.

    Признаки ``inf_0`` / ``inf_1`` коррелируют с целью, ``noise_*`` — шум.

    Returns:
        Пара ``(X, y)``.
    """
    rng = np.random.RandomState(42)
    n = 200
    inf_0 = rng.normal(size=n)
    inf_1 = rng.normal(size=n)
    y = ((inf_0 + inf_1) > 0).astype(int)
    X = pd.DataFrame(
        {
            "inf_0": inf_0,
            "inf_1": inf_1,
            "noise_0": rng.normal(size=n),
            "noise_1": rng.normal(size=n),
            "noise_2": rng.normal(size=n),
        }
    )
    return X, pd.Series(y, name="target")


@pytest.fixture
def regression_data() -> tuple[pd.DataFrame, pd.Series]:
    """Синтетический датасет регрессии с информативными и шумовыми
    признаками.

    Returns:
        Пара ``(X, y)``.
    """
    rng = np.random.RandomState(7)
    n = 200
    inf_0 = rng.normal(size=n)
    inf_1 = rng.normal(size=n)
    y = 3.0 * inf_0 - 2.0 * inf_1 + rng.normal(scale=0.1, size=n)
    X = pd.DataFrame(
        {
            "inf_0": inf_0,
            "inf_1": inf_1,
            "noise_0": rng.normal(size=n),
            "noise_1": rng.normal(size=n),
            "noise_2": rng.normal(size=n),
        }
    )
    return X, pd.Series(y, name="target")


# --- Регистрация и метаданные ----------------------------------------


def test_methods_self_register() -> None:
    """Импорт пакета регистрирует все три метода с верными группами."""
    # Импорт пакета методов наполняет реестр (саморегистрация).
    import feature_selection_benchmark.methods  # noqa: F401

    assert registry.is_registered("shap_importance")
    assert registry.is_registered("permutation_importance")
    assert registry.is_registered("null_importance")

    infos = {info.name: info for info in registry.list_registered()}
    assert infos["shap_importance"].group == "shap"
    assert infos["permutation_importance"].group == "permutation"
    assert infos["null_importance"].group == "permutation"


@pytest.mark.parametrize(
    "method_cls",
    [ShapImportanceMethod, PermutationImportanceMethod, NullImportanceMethod],
)
def test_method_info_declares_search_space(method_cls: type) -> None:
    """Каждый метод объявляет `MethodInfo` с declarative-пространством
    поиска по контракту E-005.1 (словарь ``имя → HyperParam``)."""
    info = method_cls.method_info()
    assert isinstance(info, MethodInfo)
    assert info.name == method_cls.name
    assert info.requires_target is True
    assert info.output_type == "ranking"
    assert info.hyperparameters, "ожидается непустое пространство поиска"
    for name, hp in info.hyperparameters.items():
        assert isinstance(hp, HyperParam), name
    # top_k присутствует у всех трёх ранжирующих методов.
    assert "top_k" in info.hyperparameters


# --- permutation_importance (реальные тесты) -------------------------


def test_permutation_importance_classification(
    classification_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """permutation_importance возвращает валидное подмножество признаков
    на задаче классификации."""
    X, y = classification_data
    method = PermutationImportanceMethod()
    selected = method.fit_select(
        X, y, top_k=2, n_repeats=5, n_estimators=50, random_seed=42
    )
    assert isinstance(selected, list)
    assert len(selected) == 2
    assert set(selected).issubset(set(X.columns))
    assert len(set(selected)) == len(selected)


def test_permutation_importance_regression(
    regression_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """permutation_importance возвращает валидное подмножество признаков
    на задаче регрессии."""
    X, y = regression_data
    method = PermutationImportanceMethod()
    selected = method.fit_select(
        X, y, top_k=3, n_repeats=5, n_estimators=50, random_seed=42
    )
    assert len(selected) == 3
    assert set(selected).issubset(set(X.columns))


def test_permutation_importance_picks_informative(
    regression_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """permutation_importance отбирает информативные признаки поверх
    шумовых."""
    X, y = regression_data
    method = PermutationImportanceMethod()
    selected = method.fit_select(
        X, y, top_k=2, n_repeats=10, n_estimators=100, random_seed=42
    )
    assert set(selected) == {"inf_0", "inf_1"}


def test_permutation_importance_deterministic(
    classification_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """При одинаковом ``random_seed`` отбор воспроизводим (К2)."""
    X, y = classification_data
    method = PermutationImportanceMethod()
    first = method.fit_select(
        X, y, top_k=3, n_repeats=5, n_estimators=50, random_seed=42
    )
    second = method.fit_select(
        X, y, top_k=3, n_repeats=5, n_estimators=50, random_seed=42
    )
    assert first == second


def test_permutation_importance_top_k_clamped(
    classification_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """``top_k`` больше числа признаков обрезается до доступного."""
    X, y = classification_data
    method = PermutationImportanceMethod()
    selected = method.fit_select(
        X, y, top_k=99, n_repeats=3, n_estimators=30, random_seed=0
    )
    assert len(selected) == len(X.columns)


def test_permutation_importance_always_available() -> None:
    """permutation_importance не имеет опц. зависимостей — доступен
    всегда."""
    assert PermutationImportanceMethod.check_availability() is None


# --- shap_importance (skip при отсутствии пакета) --------------------


def test_shap_importance_availability_reflects_package() -> None:
    """`check_availability` корректно отражает наличие пакета ``shap``."""
    reason = ShapImportanceMethod.check_availability()
    if _HAS_SHAP:
        assert reason is None
    else:
        assert reason is not None and "shap" in reason


@pytest.mark.skipif(not _HAS_SHAP, reason="пакет 'shap' не установлен")
def test_shap_importance_classification(
    classification_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """shap_importance возвращает валидное подмножество признаков."""
    X, y = classification_data
    method = ShapImportanceMethod()
    selected = method.fit_select(
        X, y, top_k=2, n_estimators=50, random_seed=42
    )
    assert len(selected) == 2
    assert set(selected).issubset(set(X.columns))


@pytest.mark.skipif(not _HAS_SHAP, reason="пакет 'shap' не установлен")
def test_shap_importance_regression(
    regression_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """shap_importance работает на задаче регрессии."""
    X, y = regression_data
    method = ShapImportanceMethod()
    selected = method.fit_select(
        X, y, top_k=3, n_estimators=50, random_seed=42
    )
    assert len(selected) == 3
    assert set(selected).issubset(set(X.columns))


# --- null_importance (skip при отсутствии пакета) --------------------


def test_null_importance_availability_reflects_package() -> None:
    """`check_availability` корректно отражает наличие пакета
    ``target-permutation-importances``."""
    reason = NullImportanceMethod.check_availability()
    if _HAS_TPI:
        assert reason is None
    else:
        assert reason is not None and "target-permutation" in reason


def test_null_importance_unavailable_excluded_from_default() -> None:
    """Недоступный метод (нет опц. пакета) исключён из дефолтного набора,
    но остаётся зарегистрирован (OQ-2)."""
    if _HAS_TPI:
        pytest.skip("пакет установлен — деградация не воспроизводима")
    assert registry.is_registered("null_importance")
    assert "null_importance" not in registry.default_method_names()
    infos = {info.name: info for info in registry.list_registered()}
    assert infos["null_importance"].available is False
    assert infos["null_importance"].unavailable_reason is not None


@pytest.mark.skipif(
    not _HAS_TPI,
    reason="пакет 'target-permutation-importances' не установлен",
)
def test_null_importance_classification(
    classification_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """null_importance возвращает валидное подмножество признаков."""
    X, y = classification_data
    method = NullImportanceMethod()
    selected = method.fit_select(
        X, y, top_k=2, num_iterations=5, n_estimators=50, random_seed=42
    )
    assert len(selected) == 2
    assert set(selected).issubset(set(X.columns))
