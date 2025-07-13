from typing import Literal

import pandas as pd


def drop_almost_empty_rows(df: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
    """
    Drop rows that have more than threshold percentage of empty values.
    """
    f_threshold = 1 - threshold
    return df.dropna(thresh=int(round(f_threshold * len(df.columns))), axis=0)


def drop_almost_const_columns(
        df: pd.DataFrame, threshold: float = 0.95
) -> pd.DataFrame:
    """
    Drop columns that have more than threshold percentage of constant values.
    """
    drop_columns = []
    for col in df.columns:
        max_count = df[col].value_counts(dropna=False).max()
        if max_count / len(df) > threshold:
            drop_columns.append(col)
    return df.drop(columns=drop_columns)


def drop_duplicates(
        df: pd.DataFrame, mode: Literal["columns", "rows", "all"] = "all"
) -> pd.DataFrame:
    """
    Drop columns that have the same values.
    """
    result = df.copy()
    if mode in ("rows", "all"):
        result = result.drop_duplicates()
    if mode in ("columns", "all"):
        # Преобразуем DataFrame в массив numpy
        arr = result.to_numpy()

        # Находим уникальные столбцы
        unique_cols = []
        seen = set()

        for i in range(arr.shape[1]):
            col = tuple(arr[:, i])
            if col not in seen:
                unique_cols.append(result.columns[i])
                seen.add(col)

        result = result.loc[:,unique_cols]

    return result
