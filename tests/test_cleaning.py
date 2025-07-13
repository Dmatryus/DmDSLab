import pytest
import pandas as pd
from dmdslab.cleaning import (
    drop_almost_const_columns,
    drop_almost_empty_rows,
    drop_duplicates,
)


@pytest.fixture
def sample_df():
    """Фикстура с тестовыми данными"""
    return pd.DataFrame({
        "A": [1, 2, None, 4],
        "B": [1, 1, 1, 1],
        "C": [None, None, None, None],
        "D": [1, 2, 3, 4],
    })


def test_drop_almost_empty_rows(sample_df):
    # Тестирование удаления почти пустых строк
    result = drop_almost_empty_rows(sample_df, threshold=0.2)
    assert len(result) == 3  # Ожидаем удаление одной строки

    result = drop_almost_empty_rows(sample_df, threshold=0.9)
    assert len(result) == 4  # Ни одна строка не должна быть удалена


def test_drop_almost_const_columns(sample_df):
    # Тестирование удаления почти константных столбцов
    result = drop_almost_const_columns(sample_df, threshold=0.8)
    assert "B" not in result.columns  # Столбец B должен быть удален
    assert "C" not in result.columns  # Столбец C должен быть удален


@pytest.mark.parametrize("mode,expected_rows,expected_cols", [
    ("rows", 2, 3),
    ("columns", 3, 2),
    ("all", 2, 2),
])
def test_drop_duplicates(mode, expected_rows, expected_cols):
    # Тестирование удаления дубликатов
    df_duplicates = pd.DataFrame({
        "A": [1, 1, 2],
        "B": [3, 3, 4],
        "C": [1, 1, 2]
    })

    result = drop_duplicates(df_duplicates, mode=mode)
    assert len(result) == expected_rows
    assert len(result.columns) == expected_cols
