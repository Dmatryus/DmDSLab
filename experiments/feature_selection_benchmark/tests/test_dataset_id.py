"""Тесты функции генерации ``dataset_id`` (подзадача E-002.1).

Покрывают критерии DoD: формат id, детерминизм (в т.ч. между процессами —
через сверку с зафиксированным литералом), различение схем и dtype,
стабильность при изменении числа строк, влияние имени, валидацию.
"""

import pandas as pd
import pytest

from feature_selection_benchmark.storage.dataset_id import compute_dataset_id


def _frame() -> pd.DataFrame:
    """Эталонная таблица с явно заданными dtype — id платформонезависим."""
    return pd.DataFrame(
        {
            "f1": pd.Series([1, 2, 3], dtype="int64"),
            "f2": pd.Series([0.1, 0.2, 0.3], dtype="float64"),
            "target": pd.Series([0, 1, 0], dtype="int64"),
        }
    )


def test_format() -> None:
    ds_id = compute_dataset_id("titanic", _frame())
    name, sep, fingerprint = ds_id.partition("::")
    assert sep == "::"
    assert name == "titanic"
    assert len(fingerprint) == 12
    assert all(ch in "0123456789abcdef" for ch in fingerprint)


def test_deterministic() -> None:
    assert compute_dataset_id("d", _frame()) == compute_dataset_id("d", _frame())


def test_cross_process_stable() -> None:
    # Сверка с зафиксированным литералом: ловит недетерминированный
    # (солёный по процессу) хэш, если им случайно заменят SHA-256.
    assert compute_dataset_id("titanic", _frame()) == "titanic::3484c0d4e0a3"


def test_row_count_does_not_affect_id() -> None:
    full = _frame()
    single_row = full.iloc[:1].copy()
    assert compute_dataset_id("d", full) == compute_dataset_id("d", single_row)


def test_column_rename_changes_id() -> None:
    base = _frame()
    renamed = base.rename(columns={"f1": "f1_renamed"})
    assert compute_dataset_id("d", base) != compute_dataset_id("d", renamed)


def test_dtype_change_changes_id() -> None:
    base = _frame()
    retyped = base.astype({"f1": "float64"})
    assert compute_dataset_id("d", base) != compute_dataset_id("d", retyped)


def test_name_change_changes_id() -> None:
    assert compute_dataset_id("a", _frame()) != compute_dataset_id("b", _frame())


@pytest.mark.parametrize("bad_name", ["", "   "])
def test_empty_name_rejected(bad_name: str) -> None:
    with pytest.raises(ValueError):
        compute_dataset_id(bad_name, _frame())
