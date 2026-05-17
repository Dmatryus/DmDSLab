"""Тесты подбора гиперпараметров `tune_hyperparams` (план E-006, E-006.2).

Проверяется, что `tune_hyperparams`:

- подбирает гиперпараметры FS-метода через Optuna TPE (ADR `0002`);
- в режиме ``n_trials=0`` возвращает дефолты метода без запуска study;
- транслирует declarative-дескриптор `HyperParam` в нужный вызов
  `trial.suggest_*` — покрыты все три вида: ``int`` / ``float`` /
  ``categorical``;
- детерминирован при фиксированном `random_seed` (критерий К2);
- возвращает пустой dict у метода без настраиваемых гиперпараметров.

Используются реальные FS-методы группы `filter` с непустым declarative-
пространством поиска: `MutualInformationMethod` (один ``int``-параметр),
`CorrelationMethod` (``categorical`` + ``int``), `VarianceThresholdMethod`
(один ``float``-параметр).
"""

from __future__ import annotations

import numpy as np
import optuna
import pandas as pd
import pytest

from feature_selection_benchmark.api import Dataset
from feature_selection_benchmark.core.tuning import tune_hyperparams
from feature_selection_benchmark.methods.base import FSMethod, MethodInfo
from feature_selection_benchmark.methods.filter_methods import (
    CorrelationMethod,
    MutualInformationMethod,
    VarianceThresholdMethod,
)

# Optuna-логи (INFO по trial'у) — лишний шум в выводе тестов.
optuna.logging.set_verbosity(optuna.logging.WARNING)


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


# --- n_trials=0 → дефолты ---------------------------------------------------


def test_zero_trials_returns_defaults(
    classification_dataset: Dataset,
) -> None:
    """`n_trials=0` возвращает дефолты метода без запуска study."""
    best = tune_hyperparams(
        MutualInformationMethod(),
        classification_dataset,
        n_trials=0,
        random_seed=42,
    )
    assert best == {"k": 10}


def test_zero_trials_returns_all_defaults(
    classification_dataset: Dataset,
) -> None:
    """`n_trials=0` возвращает дефолты по всем гиперпараметрам метода."""
    best = tune_hyperparams(
        CorrelationMethod(),
        classification_dataset,
        n_trials=0,
        random_seed=42,
    )
    assert best == {"method": "pearson", "k": 10}


# --- Метод без гиперпараметров ----------------------------------------------


class _NoHyperParamsMethod(FSMethod):
    """FS-метод без настраиваемых гиперпараметров — для проверки контракта."""

    name = "no_hyperparams"
    group = "filter"
    supported_tasks = ["classification", "regression"]

    @classmethod
    def method_info(cls) -> MethodInfo:
        """Возвращает метаданные с пустым пространством поиска."""
        return MethodInfo(
            name=cls.name,
            description="Метод без настраиваемых гиперпараметров.",
            requires_target=False,
            supports_multiclass=True,
            output_type="subset",
            hyperparameters={},
        )

    def fit_select(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        **hyperparams: object,
    ) -> list[str]:
        """Возвращает первый признак — реализация не важна для теста."""
        return [X_train.columns[0]]


def test_no_hyperparams_returns_empty_dict(
    classification_dataset: Dataset,
) -> None:
    """Метод без настраиваемых гиперпараметров — возвращается пустой dict."""
    best = tune_hyperparams(
        _NoHyperParamsMethod(),
        classification_dataset,
        n_trials=20,
        random_seed=42,
    )
    assert best == {}


# --- Трансляция HyperParam → trial.suggest_* --------------------------------


def test_tunes_int_param(classification_dataset: Dataset) -> None:
    """`int`-гиперпараметр подбирается в объявленных границах."""
    best = tune_hyperparams(
        MutualInformationMethod(),
        classification_dataset,
        n_trials=5,
        random_seed=42,
    )
    assert set(best) == {"k"}
    assert isinstance(best["k"], int)
    assert 1 <= best["k"] <= 50


def test_tunes_float_param(classification_dataset: Dataset) -> None:
    """`float`-гиперпараметр подбирается в объявленных границах."""
    best = tune_hyperparams(
        VarianceThresholdMethod(),
        classification_dataset,
        n_trials=5,
        random_seed=42,
    )
    assert set(best) == {"threshold"}
    assert isinstance(best["threshold"], float)
    assert 0.0 <= best["threshold"] <= 1.0


def test_tunes_categorical_and_int_params(
    classification_dataset: Dataset,
) -> None:
    """`categorical` + `int` гиперпараметры подбираются совместно."""
    best = tune_hyperparams(
        CorrelationMethod(),
        classification_dataset,
        n_trials=5,
        random_seed=42,
    )
    assert set(best) == {"method", "k"}
    assert best["method"] in ("pearson", "spearman")
    assert isinstance(best["k"], int)
    assert 1 <= best["k"] <= 50


# --- Детерминизм (К2) -------------------------------------------------------


def test_same_seed_same_result(classification_dataset: Dataset) -> None:
    """Одинаковый `random_seed` → одинаковый `best_params`."""
    first = tune_hyperparams(
        CorrelationMethod(),
        classification_dataset,
        n_trials=8,
        random_seed=123,
    )
    second = tune_hyperparams(
        CorrelationMethod(),
        classification_dataset,
        n_trials=8,
        random_seed=123,
    )
    assert first == second


def test_different_seed_may_differ(
    classification_dataset: Dataset,
) -> None:
    """Подбор реально исследует пространство — результат корректен по типу."""
    best = tune_hyperparams(
        CorrelationMethod(),
        classification_dataset,
        n_trials=8,
        random_seed=999,
    )
    assert set(best) == {"method", "k"}
