"""Тесты ядра кросс-валидации `CVRunner` (план E-006, E-006.1).

Проверяется, что `CVRunner.run_cv`:

- возвращает корректный `CVResult` для классификации и регрессии;
- детерминирован при фиксированном `random_seed` (фолды + прокси-модель);
- работает с обеими прокси-моделями (`catboost` / `random_forest`);
- авто-режим (`estimator=None`) разрешает прокси-модель и при явном
  `estimator="catboost"` без пакета поднимает информативную ошибку.

Прокси-CatBoost зависит от опц. пакета `catboost`: соответствующие тесты
помечены `pytest.mark.skipif` и пропускаются, если пакет не установлен.
"""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

from feature_selection_benchmark.api import Dataset
from feature_selection_benchmark.core.orchestrator import CVResult, CVRunner
from feature_selection_benchmark.methods.filter_methods import (
    VarianceThresholdMethod,
)

# --- Доступность опц. пакета catboost ---------------------------------------

CATBOOST_AVAILABLE = importlib.util.find_spec("catboost") is not None
catboost_required = pytest.mark.skipif(
    not CATBOOST_AVAILABLE, reason="опц. пакет 'catboost' не установлен"
)


# --- Фикстуры данных --------------------------------------------------------


@pytest.fixture
def classification_dataset() -> Dataset:
    """Синтетический датасет классификации: informative + noise + const."""
    rng = np.random.default_rng(42)
    n = 200
    y = rng.integers(0, 2, size=n)
    df = pd.DataFrame(
        {
            "informative_a": y + rng.normal(0, 0.3, n),
            "informative_b": y * 2.0 + rng.normal(0, 0.4, n),
            "noise_a": rng.normal(0, 1, n),
            "noise_b": rng.normal(0, 1, n),
            "const": np.zeros(n),
            "target": y,
        }
    )
    features = ["informative_a", "informative_b", "noise_a", "noise_b", "const"]
    return Dataset(data=df, features_list=features, target="target")


@pytest.fixture
def regression_dataset() -> Dataset:
    """Синтетический датасет регрессии: informative + noise."""
    rng = np.random.default_rng(7)
    n = 200
    x_a = rng.normal(0, 1, n)
    x_b = rng.normal(0, 1, n)
    y = 3.0 * x_a - 2.0 * x_b + rng.normal(0, 0.2, n)
    df = pd.DataFrame(
        {
            "informative_a": x_a,
            "informative_b": x_b,
            "noise_a": rng.normal(0, 1, n),
            "noise_b": rng.normal(0, 1, n),
            "target": y,
        }
    )
    features = ["informative_a", "informative_b", "noise_a", "noise_b"]
    return Dataset(data=df, features_list=features, target="target")


# --- Конструктор: разрешение estimator --------------------------------------


def test_init_rejects_small_cv() -> None:
    """`cv` < 2 поднимает `ValueError`."""
    with pytest.raises(ValueError, match="cv"):
        CVRunner(cv=1)


def test_init_rejects_unknown_estimator() -> None:
    """Неизвестное значение `estimator` поднимает `ValueError`."""
    with pytest.raises(ValueError, match="estimator"):
        CVRunner(estimator="xgboost")


def test_init_random_forest_explicit() -> None:
    """Явный `estimator='random_forest'` разрешается без CatBoost."""
    runner = CVRunner(cv=3, random_seed=0, estimator="random_forest")
    assert runner.estimator == "random_forest"


def test_init_auto_resolves_estimator() -> None:
    """Авто-режим (`estimator=None`) разрешается в одну из прокси-моделей."""
    runner = CVRunner(cv=3, random_seed=0, estimator=None)
    expected = "catboost" if CATBOOST_AVAILABLE else "random_forest"
    assert runner.estimator == expected


@pytest.mark.skipif(
    CATBOOST_AVAILABLE, reason="catboost установлен — ошибка не возникает"
)
def test_init_explicit_catboost_without_package_raises() -> None:
    """Явный `estimator='catboost'` без пакета поднимает `ImportError`."""
    with pytest.raises(ImportError, match="catboost"):
        CVRunner(estimator="catboost")


@catboost_required
def test_init_catboost_explicit() -> None:
    """Явный `estimator='catboost'` разрешается при наличии пакета."""
    runner = CVRunner(cv=3, random_seed=0, estimator="catboost")
    assert runner.estimator == "catboost"


# --- run_cv: корректный CVResult --------------------------------------------


def test_run_cv_classification_returns_valid_result(
    classification_dataset: Dataset,
) -> None:
    """`run_cv` на классификации возвращает корректный `CVResult`."""
    runner = CVRunner(cv=5, random_seed=42, estimator="random_forest")
    result = runner.run_cv(
        VarianceThresholdMethod(), classification_dataset, {}
    )

    assert isinstance(result, CVResult)
    assert isinstance(result.cv_score, float)
    assert isinstance(result.score_std, float)
    assert result.score_std >= 0.0
    assert result.duration_sec >= 0.0
    assert len(result.selected_features) > 0
    assert set(result.selected_features).issubset(
        set(classification_dataset.features_list)
    )
    # Accuracy ограничена [0, 1].
    assert 0.0 <= result.cv_score <= 1.0


def test_run_cv_regression_returns_valid_result(
    regression_dataset: Dataset,
) -> None:
    """`run_cv` на регрессии возвращает корректный `CVResult`."""
    runner = CVRunner(cv=5, random_seed=42, estimator="random_forest")
    result = runner.run_cv(VarianceThresholdMethod(), regression_dataset, {})

    assert isinstance(result, CVResult)
    assert result.score_std >= 0.0
    assert result.duration_sec >= 0.0
    assert len(result.selected_features) > 0
    # R² «больше = лучше»; на informative-датасете прокси-модель должна
    # объяснять заметную долю дисперсии.
    assert result.cv_score > 0.5


def test_run_cv_passes_hyperparams(
    classification_dataset: Dataset,
) -> None:
    """`run_cv` пробрасывает `hyperparams` в `fit_select` метода."""
    runner = CVRunner(cv=3, random_seed=0, estimator="random_forest")
    # Высокий порог дисперсии — const-признак гарантированно отсеивается.
    result = runner.run_cv(
        VarianceThresholdMethod(),
        classification_dataset,
        {"threshold": 0.5},
    )
    assert "const" not in result.selected_features


# --- Детерминизм ------------------------------------------------------------


def test_run_cv_deterministic_with_fixed_seed(
    classification_dataset: Dataset,
) -> None:
    """Два прогона с одним `random_seed` дают идентичный `CVResult`."""
    method = VarianceThresholdMethod()
    runner_a = CVRunner(cv=5, random_seed=123, estimator="random_forest")
    runner_b = CVRunner(cv=5, random_seed=123, estimator="random_forest")

    result_a = runner_a.run_cv(method, classification_dataset, {})
    result_b = runner_b.run_cv(method, classification_dataset, {})

    assert result_a.cv_score == result_b.cv_score
    assert result_a.score_std == result_b.score_std
    assert result_a.selected_features == result_b.selected_features


def test_run_cv_regression_deterministic_with_fixed_seed(
    regression_dataset: Dataset,
) -> None:
    """Детерминизм `run_cv` на регрессии при фиксированном seed."""
    method = VarianceThresholdMethod()
    runner_a = CVRunner(cv=4, random_seed=99, estimator="random_forest")
    runner_b = CVRunner(cv=4, random_seed=99, estimator="random_forest")

    result_a = runner_a.run_cv(method, regression_dataset, {})
    result_b = runner_b.run_cv(method, regression_dataset, {})

    assert result_a.cv_score == result_b.cv_score
    assert result_a.score_std == result_b.score_std


# --- Прокси-модель CatBoost -------------------------------------------------


@catboost_required
def test_run_cv_catboost_classification(
    classification_dataset: Dataset,
) -> None:
    """`run_cv` с прокси-CatBoost возвращает корректный результат."""
    runner = CVRunner(cv=3, random_seed=42, estimator="catboost")
    result = runner.run_cv(
        VarianceThresholdMethod(), classification_dataset, {}
    )
    assert isinstance(result, CVResult)
    assert 0.0 <= result.cv_score <= 1.0
    assert len(result.selected_features) > 0


@catboost_required
def test_run_cv_catboost_deterministic(
    regression_dataset: Dataset,
) -> None:
    """Прокси-CatBoost детерминирован при фиксированном `random_seed`."""
    method = VarianceThresholdMethod()
    runner_a = CVRunner(cv=3, random_seed=5, estimator="catboost")
    runner_b = CVRunner(cv=3, random_seed=5, estimator="catboost")

    result_a = runner_a.run_cv(method, regression_dataset, {})
    result_b = runner_b.run_cv(method, regression_dataset, {})

    assert result_a.cv_score == result_b.cv_score
    assert result_a.score_std == result_b.score_std
