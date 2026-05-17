"""Общие pytest-фикстуры тестового пакета.

Синтетические датасеты бенчмарка вынесены сюда из `test_orchestrator.py`
и `test_tuning.py` — pytest обнаруживает фикстуры `conftest.py`
автоматически, без явного импорта в тест-файлах.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from feature_selection_benchmark.api import Dataset


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
