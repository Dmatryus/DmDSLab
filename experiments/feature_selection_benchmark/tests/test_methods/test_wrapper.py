"""Тесты wrapper-методов (E-005.3).

Проверяется реализация четырёх wrapper-методов: `rfe`,
`sequential_feature_selector`, `stability_selection`, `boruta`.

- Доступные методы (`rfe`, `sequential_feature_selector`,
  `stability_selection`) тестируются реальным прогоном на синтетических
  данных классификации и регрессии.
- `boruta` зависит от опционального пакета ``boruta``; тест его прогона
  помечается ``skip``, если пакет не установлен. Корректность деградации
  (метод регистрируется, помечается недоступным) проверяется всегда.

Изоляция реестра: built-in методы саморегистрируются при импорте
группа-модулей. Фикстура `wrapper_registry` чистит реестр и заново
наполняет его только wrapper-методами.
"""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

from feature_selection_benchmark.methods import registry, wrapper_methods
from feature_selection_benchmark.methods.base import HyperParam, MethodInfo

WRAPPER_NAMES = [
    "rfe",
    "sequential_feature_selector",
    "boruta",
    "stability_selection",
]

# Доступные без опц. зависимостей методы (boruta исключён).
AVAILABLE_NAMES = ["rfe", "sequential_feature_selector", "stability_selection"]


def _boruta_installed() -> bool:
    """Проверяет, установлен ли опциональный пакет ``boruta``."""
    return importlib.util.find_spec("boruta") is not None


@pytest.fixture
def wrapper_registry():
    """Очищает реестр и регистрирует только wrapper-методы.

    Yields:
        Модуль `registry` с наполнением из wrapper-методов.
    """
    registry.clear_registry()
    importlib.reload(wrapper_methods)
    yield registry
    registry.clear_registry()


@pytest.fixture
def clf_data() -> tuple[pd.DataFrame, pd.Series]:
    """Синтетический датасет классификации с информативными признаками."""
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=120,
        n_features=10,
        n_informative=4,
        n_redundant=2,
        random_state=42,
    )
    columns = [f"f{i}" for i in range(X.shape[1])]
    return pd.DataFrame(X, columns=columns), pd.Series(y, name="target")


@pytest.fixture
def reg_data() -> tuple[pd.DataFrame, pd.Series]:
    """Синтетический датасет регрессии с информативными признаками."""
    from sklearn.datasets import make_regression

    X, y = make_regression(
        n_samples=120,
        n_features=10,
        n_informative=4,
        noise=0.1,
        random_state=42,
    )
    columns = [f"f{i}" for i in range(X.shape[1])]
    return pd.DataFrame(X, columns=columns), pd.Series(y, name="target")


# --- Регистрация и метаданные ---------------------------------------------


def test_all_wrapper_methods_registered(wrapper_registry):
    """Все 4 wrapper-метода саморегистрируются при импорте модуля."""
    for name in WRAPPER_NAMES:
        assert wrapper_registry.is_registered(name)


def test_available_methods_in_default_set(wrapper_registry):
    """Методы без опц. зависимостей попадают в дефолтный набор прогона."""
    defaults = wrapper_registry.default_method_names()
    for name in AVAILABLE_NAMES:
        assert name in defaults


def test_method_info_declares_search_space(wrapper_registry):
    """Каждый wrapper-метод объявляет declarative-пространство поиска."""
    infos = {info.name: info for info in wrapper_registry.list_registered()}
    for name in WRAPPER_NAMES:
        info = infos[name]
        assert isinstance(info, MethodInfo)
        assert info.group == "wrapper"
        assert info.output_type == "subset"
        assert info.requires_target is True
        assert isinstance(info.hyperparameters, dict)
        assert info.hyperparameters, f"{name}: пустое пространство поиска"
        for hp in info.hyperparameters.values():
            assert isinstance(hp, HyperParam)


# --- boruta: корректная деградация опц. зависимости -----------------------


def test_boruta_availability_matches_package(wrapper_registry):
    """Статус доступности `boruta` соответствует наличию пакета ``boruta``."""
    infos = {info.name: info for info in wrapper_registry.list_registered()}
    boruta_info = infos["boruta"]
    if _boruta_installed():
        assert boruta_info.available is True
        assert boruta_info.unavailable_reason is None
    else:
        assert boruta_info.available is False
        assert boruta_info.unavailable_reason is not None
        assert "boruta" in boruta_info.unavailable_reason
        # Недоступный метод исключён из дефолтного набора прогона.
        assert "boruta" not in wrapper_registry.default_method_names()


def test_module_import_does_not_require_boruta():
    """Импорт модуля wrapper-методов не падает без пакета ``boruta``."""
    # Сам факт успешного импорта модуля на уровне теста — проверка.
    assert hasattr(wrapper_methods, "BorutaMethod")


# --- rfe -------------------------------------------------------------------


def test_rfe_classification_returns_valid_subset(wrapper_registry, clf_data):
    """`rfe` возвращает валидное непустое подмножество (классификация)."""
    X, y = clf_data
    method = wrapper_registry.get_method("rfe")
    selected = method.fit_select(X, y, fraction=0.5, random_seed=0)
    assert 0 < len(selected) <= X.shape[1]
    assert set(selected).issubset(set(X.columns))
    assert len(selected) == len(set(selected))


def test_rfe_regression_returns_valid_subset(wrapper_registry, reg_data):
    """`rfe` возвращает валидное подмножество для задачи регрессии."""
    X, y = reg_data
    method = wrapper_registry.get_method("rfe")
    selected = method.fit_select(X, y, fraction=0.3, random_seed=0)
    assert set(selected).issubset(set(X.columns))
    assert len(selected) == max(1, round(0.3 * X.shape[1]))


def test_rfe_deterministic_with_seed(wrapper_registry, clf_data):
    """`rfe` воспроизводим при фиксированном `random_seed` (К2)."""
    X, y = clf_data
    method = wrapper_registry.get_method("rfe")
    first = method.fit_select(X, y, fraction=0.5, random_seed=7)
    second = method.fit_select(X, y, fraction=0.5, random_seed=7)
    assert first == second


# --- sequential_feature_selector ------------------------------------------


def test_sfs_forward_returns_valid_subset(wrapper_registry, clf_data):
    """`sequential_feature_selector` (forward) даёт валидное подмножество."""
    X, y = clf_data
    method = wrapper_registry.get_method("sequential_feature_selector")
    selected = method.fit_select(
        X, y, fraction=0.4, direction="forward", random_seed=0
    )
    assert 0 < len(selected) < X.shape[1]
    assert set(selected).issubset(set(X.columns))


def test_sfs_backward_returns_valid_subset(wrapper_registry, reg_data):
    """`sequential_feature_selector` (backward) даёт валидное подмножество."""
    X, y = reg_data
    method = wrapper_registry.get_method("sequential_feature_selector")
    selected = method.fit_select(
        X, y, fraction=0.5, direction="backward", random_seed=0
    )
    assert 0 < len(selected) < X.shape[1]
    assert set(selected).issubset(set(X.columns))


# --- stability_selection ---------------------------------------------------


def test_stability_classification_returns_valid_subset(
    wrapper_registry, clf_data
):
    """`stability_selection` даёт валидное подмножество (классификация)."""
    X, y = clf_data
    method = wrapper_registry.get_method("stability_selection")
    selected = method.fit_select(
        X,
        y,
        n_bootstrap=30,
        sample_fraction=0.6,
        selection_threshold=0.5,
        random_seed=0,
    )
    assert 0 < len(selected) <= X.shape[1]
    assert set(selected).issubset(set(X.columns))


def test_stability_regression_returns_valid_subset(
    wrapper_registry, reg_data
):
    """`stability_selection` даёт валидное подмножество (регрессия)."""
    X, y = reg_data
    method = wrapper_registry.get_method("stability_selection")
    selected = method.fit_select(
        X,
        y,
        n_bootstrap=30,
        sample_fraction=0.6,
        selection_threshold=0.4,
        random_seed=0,
    )
    assert 0 < len(selected) <= X.shape[1]
    assert set(selected).issubset(set(X.columns))


def test_stability_high_threshold_falls_back_to_one(
    wrapper_registry, clf_data
):
    """При недостижимом пороге `stability_selection` возвращает 1 признак."""
    X, y = clf_data
    method = wrapper_registry.get_method("stability_selection")
    selected = method.fit_select(
        X,
        y,
        n_bootstrap=20,
        sample_fraction=0.5,
        selection_threshold=1.01,
        random_seed=0,
    )
    assert len(selected) == 1
    assert selected[0] in X.columns


def test_stability_deterministic_with_seed(wrapper_registry, clf_data):
    """`stability_selection` воспроизводим при фиксированном зерне (К2)."""
    X, y = clf_data
    method = wrapper_registry.get_method("stability_selection")
    kwargs = dict(
        n_bootstrap=25,
        sample_fraction=0.6,
        selection_threshold=0.5,
        random_seed=123,
    )
    first = method.fit_select(X, y, **kwargs)
    second = method.fit_select(X, y, **kwargs)
    assert first == second


# --- boruta: реальный прогон (skip без пакета) ----------------------------


@pytest.mark.skipif(
    not _boruta_installed(),
    reason="опциональный пакет 'boruta' не установлен",
)
def test_boruta_returns_valid_subset(wrapper_registry, clf_data):
    """`boruta` возвращает валидное непустое подмножество признаков."""
    X, y = clf_data
    method = wrapper_registry.get_method("boruta")
    selected = method.fit_select(X, y, max_iter=30, random_seed=0)
    assert 0 < len(selected) <= X.shape[1]
    assert set(selected).issubset(set(X.columns))
