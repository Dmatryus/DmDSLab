"""Тесты filter-методов отбора признаков (план E-005, E-005.2).

Проверяется, что каждый из четырёх filter-методов:

- реализует `fit_select` и возвращает валидное непустое подмножество
  имён признаков;
- объявляет `MethodInfo` с declarative-пространством поиска по контракту
  E-005.1 / ADR `0002`;
- саморегистрируется в реестре при импорте группа-модуля.

Методы на `scikit-learn` / `scipy` тестируются на реальных данных. Метод
`mrmr` зависит от опц. пакета `mrmr-selection`: соответствующие тесты
помечены `pytest.mark.skipif` и пропускаются, если пакет не установлен.
"""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

from feature_selection_benchmark.methods import filter_methods, registry
from feature_selection_benchmark.methods.base import HyperParam, MethodInfo
from feature_selection_benchmark.methods.filter_methods import (
    CorrelationMethod,
    MRMRMethod,
    MutualInformationMethod,
    VarianceThresholdMethod,
)

# --- Доступность опц. пакета mrmr -------------------------------------------

MRMR_AVAILABLE = importlib.util.find_spec("mrmr") is not None
mrmr_required = pytest.mark.skipif(
    not MRMR_AVAILABLE, reason="опц. пакет 'mrmr-selection' не установлен"
)


# --- Фикстуры данных --------------------------------------------------------


@pytest.fixture
def classification_data() -> tuple[pd.DataFrame, pd.Series]:
    """Синтетический датасет классификации: informative + noise + const.

    `informative_*` коррелируют с целью, `noise_*` — случайны, `const` —
    константа (нулевая дисперсия — должна отсеиваться).
    """
    rng = np.random.default_rng(42)
    n = 200
    y = rng.integers(0, 2, size=n)
    df = pd.DataFrame(
        {
            "informative_a": y + rng.normal(0, 0.3, n),
            "informative_b": y * 2.0 + rng.normal(0, 0.4, n),
            "noise_a": rng.normal(0, 1, n),
            "noise_b": rng.normal(0, 1, n),
            "noise_c": rng.normal(0, 1, n),
            "const": np.ones(n),
        }
    )
    return df, pd.Series(y, name="target")


@pytest.fixture
def regression_data() -> tuple[pd.DataFrame, pd.Series]:
    """Синтетический датасет регрессии: informative + noise + const."""
    rng = np.random.default_rng(7)
    n = 200
    base = rng.normal(0, 1, n)
    df = pd.DataFrame(
        {
            "informative_a": base + rng.normal(0, 0.2, n),
            "informative_b": 2.0 * base + rng.normal(0, 0.3, n),
            "noise_a": rng.normal(0, 1, n),
            "noise_b": rng.normal(0, 1, n),
            "const": np.full(n, 5.0),
        }
    )
    y = 3.0 * base + rng.normal(0, 0.1, n)
    return df, pd.Series(y, name="target")


# --- Хелпер проверки контракта fit_select -----------------------------------


def _assert_valid_subset(
    selected: list[str], X: pd.DataFrame
) -> None:
    """Проверяет, что `selected` — валидное непустое подмножество колонок."""
    assert isinstance(selected, list)
    assert len(selected) > 0, "fit_select вернул пустое подмножество"
    assert all(isinstance(name, str) for name in selected)
    assert set(selected).issubset(set(X.columns)), (
        "fit_select вернул имена вне набора признаков"
    )
    assert len(selected) == len(set(selected)), "дубликаты в подмножестве"


# --- VarianceThresholdMethod ------------------------------------------------


def test_variance_threshold_selects_subset(classification_data):
    X, y = classification_data
    selected = VarianceThresholdMethod().fit_select(X, y)
    _assert_valid_subset(selected, X)
    # Константа имеет нулевую дисперсию — должна быть отсеяна по дефолту.
    assert "const" not in selected


def test_variance_threshold_high_threshold_fallback(classification_data):
    X, y = classification_data
    # Завышенный порог отсекает все признаки — метод обязан вернуть
    # непустое подмножество (признак с макс. дисперсией).
    selected = VarianceThresholdMethod().fit_select(X, y, threshold=1e9)
    _assert_valid_subset(selected, X)
    assert len(selected) == 1


# --- CorrelationMethod ------------------------------------------------------


@pytest.mark.parametrize("method", ["pearson", "spearman"])
def test_correlation_selects_informative(classification_data, method):
    X, y = classification_data
    selected = CorrelationMethod().fit_select(X, y, method=method, k=2)
    _assert_valid_subset(selected, X)
    assert len(selected) == 2
    # Informative-признаки должны опередить шум по корреляции.
    assert "informative_a" in selected
    assert "informative_b" in selected


def test_correlation_regression(regression_data):
    X, y = regression_data
    selected = CorrelationMethod().fit_select(X, y, method="pearson", k=2)
    _assert_valid_subset(selected, X)
    assert set(selected) == {"informative_a", "informative_b"}


def test_correlation_k_clamped(classification_data):
    X, y = classification_data
    # k больше числа признаков — отбираются все.
    selected = CorrelationMethod().fit_select(X, y, k=999)
    assert len(selected) == X.shape[1]


# --- MutualInformationMethod ------------------------------------------------


def test_mutual_information_classification(classification_data):
    X, y = classification_data
    selected = MutualInformationMethod().fit_select(X, y, k=2)
    _assert_valid_subset(selected, X)
    assert len(selected) == 2
    assert "informative_a" in selected
    assert "informative_b" in selected


def test_mutual_information_regression(regression_data):
    X, y = regression_data
    selected = MutualInformationMethod().fit_select(X, y, k=2)
    _assert_valid_subset(selected, X)
    assert set(selected) == {"informative_a", "informative_b"}


def test_mutual_information_deterministic(classification_data):
    X, y = classification_data
    first = MutualInformationMethod().fit_select(X, y, k=3, random_state=0)
    second = MutualInformationMethod().fit_select(X, y, k=3, random_state=0)
    assert first == second


# --- MRMRMethod -------------------------------------------------------------


def test_mrmr_check_availability_matches_environment():
    reason = MRMRMethod.check_availability()
    if MRMR_AVAILABLE:
        assert reason is None
    else:
        assert isinstance(reason, str) and "mrmr" in reason.lower()


@mrmr_required
def test_mrmr_selects_subset(classification_data):
    X, y = classification_data
    selected = MRMRMethod().fit_select(X, y, k=3)
    _assert_valid_subset(selected, X)
    assert len(selected) == 3


# --- Метаданные / пространство поиска ----------------------------------------


@pytest.mark.parametrize(
    "cls",
    [
        VarianceThresholdMethod,
        CorrelationMethod,
        MutualInformationMethod,
        MRMRMethod,
    ],
)
def test_method_info_contract(cls):
    info = cls.method_info()
    assert isinstance(info, MethodInfo)
    assert info.name == cls.name
    assert info.description
    assert info.output_type in ("ranking", "subset")
    assert isinstance(info.hyperparameters, dict)
    # Каждый объявленный гиперпараметр — валидный declarative-дескриптор.
    for hp_name, hp in info.hyperparameters.items():
        assert isinstance(hp_name, str)
        assert isinstance(hp, HyperParam)
        if hp.kind in ("int", "float"):
            assert hp.low is not None and hp.high is not None
        else:
            assert hp.choices


# --- Саморегистрация --------------------------------------------------------


def test_filter_methods_self_register():
    # Импорт группа-модуля наполняет реестр (план E-005, A-1) — он уже
    # импортирован пакетом `methods`, реестр наполнен на момент теста.
    assert filter_methods.__name__.endswith("filter_methods")
    for name in (
        "variance_threshold",
        "correlation",
        "mutual_information",
        "mrmr",
    ):
        assert registry.is_registered(name), f"{name} не зарегистрирован"
        info = next(
            i for i in registry.list_registered() if i.name == name
        )
        assert info.group == "filter"


def test_mrmr_registered_with_availability_status():
    # mrmr регистрируется всегда; статус доступности отражает среду (OQ-2).
    assert registry.is_registered("mrmr")
    info = next(i for i in registry.list_registered() if i.name == "mrmr")
    if MRMR_AVAILABLE:
        assert info.available is True
    else:
        assert info.available is False
        assert info.unavailable_reason is not None
        assert "mrmr" not in registry.default_method_names()
