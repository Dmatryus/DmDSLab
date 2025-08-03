"""
Операции для работы со строковыми данными для бенчмаркинга.
"""

from typing import Optional, Union, List, Dict, Any
import pandas as pd
import polars as pl
import re

from .base import DataAwareOperation, OperationResult, register_operation


class StringContainsOperation(DataAwareOperation):
    """Проверка содержания подстроки в строковых данных."""

    def __init__(self):
        super().__init__(
            name="string_contains",
            category="string",
            description="Проверка содержания подстроки в строковых колонках",
        )

    def execute_pandas(
        self,
        df: pd.DataFrame,
        backend: str = "numpy",
        column: Optional[str] = None,
        substring: str = "str_5",
        case_sensitive: bool = True,
        **kwargs,
    ) -> OperationResult:
        """Проверка содержания подстроки в Pandas."""
        try:
            # Выбираем строковую колонку
            if column is None:
                string_cols = self.get_string_columns(df)
                if not string_cols:
                    return OperationResult(
                        success=False,
                        error=ValueError("Нет строковых колонок в датафрейме"),
                    )
                column = string_cols[0]

            # Создаем новую колонку с результатом проверки
            result_df = df.copy()
            result_df[f"{column}_contains"] = df[column].str.contains(
                substring, case=case_sensitive, na=False
            )

            # Считаем статистику
            contains_count = result_df[f"{column}_contains"].sum()
            contains_ratio = contains_count / len(df) if len(df) > 0 else 0

            metadata = {
                "column": column,
                "substring": substring,
                "case_sensitive": case_sensitive,
                "contains_count": int(contains_count),
                "contains_ratio": contains_ratio,
                "backend": backend,
            }

            return OperationResult(success=True, result=result_df, metadata=metadata)

        except Exception as e:
            return OperationResult(success=False, error=e)

    def execute_polars(
        self,
        df: Union[pl.DataFrame, pl.LazyFrame],
        lazy: bool = False,
        column: Optional[str] = None,
        substring: str = "str_5",
        case_sensitive: bool = True,
        **kwargs,
    ) -> OperationResult:
        """Проверка содержания подстроки в Polars."""
        try:
            # Выбираем колонку
            if column is None:
                if lazy:
                    # Для lazy mode предполагаем первую строковую колонку
                    column = next((col for col in df.columns if "str" in col), None)
                    if column is None:
                        return OperationResult(
                            success=False, error=ValueError("Нет строковых колонок")
                        )
                else:
                    string_cols = self.get_string_columns(df)
                    if not string_cols:
                        return OperationResult(
                            success=False, error=ValueError("Нет строковых колонок")
                        )
                    column = string_cols[0]

            # Создаем новую колонку
            if case_sensitive:
                result_df = df.with_columns(
                    pl.col(column).str.contains(substring).alias(f"{column}_contains")
                )
            else:
                result_df = df.with_columns(
                    pl.col(column)
                    .str.to_lowercase()
                    .str.contains(substring.lower())
                    .alias(f"{column}_contains")
                )

            metadata = {
                "column": column,
                "substring": substring,
                "case_sensitive": case_sensitive,
                "lazy": lazy,
            }

            if not lazy:
                contains_count = result_df[f"{column}_contains"].sum()
                metadata.update(
                    {
                        "contains_count": int(contains_count),
                        "contains_ratio": (
                            contains_count / len(df) if len(df) > 0 else 0
                        ),
                    }
                )

            return OperationResult(success=True, result=result_df, metadata=metadata)

        except Exception as e:
            return OperationResult(success=False, error=e)


class StringReplaceOperation(DataAwareOperation):
    """Замена подстроки в строковых данных."""

    def __init__(self):
        super().__init__(
            name="string_replace",
            category="string",
            description="Замена подстрок в строковых колонках",
        )

    def execute_pandas(
        self,
        df: pd.DataFrame,
        backend: str = "numpy",
        column: Optional[str] = None,
        pattern: str = "_",
        replacement: str = "-",
        regex: bool = False,
        **kwargs,
    ) -> OperationResult:
        """Замена подстроки в Pandas."""
        try:
            # Выбираем колонку
            if column is None:
                string_cols = self.get_string_columns(df)
                if not string_cols:
                    return OperationResult(
                        success=False, error=ValueError("Нет строковых колонок")
                    )
                column = string_cols[0]

            # Выполняем замену
            result_df = df.copy()
            result_df[f"{column}_replaced"] = df[column].str.replace(
                pattern, replacement, regex=regex
            )

            # Считаем количество изменений
            changed_rows = (result_df[f"{column}_replaced"] != df[column]).sum()
            changed_ratio = changed_rows / len(df) if len(df) > 0 else 0

            metadata = {
                "column": column,
                "pattern": pattern,
                "replacement": replacement,
                "regex": regex,
                "changed_rows": int(changed_rows),
                "changed_ratio": changed_ratio,
                "backend": backend,
            }

            return OperationResult(success=True, result=result_df, metadata=metadata)

        except Exception as e:
            return OperationResult(success=False, error=e)

    def execute_polars(
        self,
        df: Union[pl.DataFrame, pl.LazyFrame],
        lazy: bool = False,
        column: Optional[str] = None,
        pattern: str = "_",
        replacement: str = "-",
        regex: bool = False,
        **kwargs,
    ) -> OperationResult:
        """Замена подстроки в Polars."""
        try:
            # Выбираем колонку
            if column is None:
                if lazy:
                    column = next((col for col in df.columns if "str" in col), None)
                    if column is None:
                        return OperationResult(
                            success=False, error=ValueError("Нет строковых колонок")
                        )
                else:
                    string_cols = self.get_string_columns(df)
                    if not string_cols:
                        return OperationResult(
                            success=False, error=ValueError("Нет строковых колонок")
                        )
                    column = string_cols[0]

            # Выполняем замену
            if regex:
                result_df = df.with_columns(
                    pl.col(column)
                    .str.replace_all(pattern, replacement)
                    .alias(f"{column}_replaced")
                )
            else:
                result_df = df.with_columns(
                    pl.col(column)
                    .str.replace_all(pattern, replacement, literal=True)
                    .alias(f"{column}_replaced")
                )

            metadata = {
                "column": column,
                "pattern": pattern,
                "replacement": replacement,
                "regex": regex,
                "lazy": lazy,
            }

            if not lazy:
                # Подсчет изменений требует материализации
                temp_df = result_df.select(
                    [pl.col(column), pl.col(f"{column}_replaced")]
                )
                changed_rows = (temp_df[column] != temp_df[f"{column}_replaced"]).sum()
                metadata.update(
                    {
                        "changed_rows": int(changed_rows),
                        "changed_ratio": changed_rows / len(df) if len(df) > 0 else 0,
                    }
                )

            return OperationResult(success=True, result=result_df, metadata=metadata)

        except Exception as e:
            return OperationResult(success=False, error=e)


class StringExtractOperation(DataAwareOperation):
    """Извлечение данных по паттерну из строк."""

    def __init__(self):
        super().__init__(
            name="string_extract",
            category="string",
            description="Извлечение данных по регулярному выражению",
        )

    def execute_pandas(
        self,
        df: pd.DataFrame,
        backend: str = "numpy",
        column: Optional[str] = None,
        pattern: str = r"str_(\d+)",
        **kwargs,
    ) -> OperationResult:
        """Извлечение по паттерну в Pandas."""
        try:
            # Выбираем колонку
            if column is None:
                string_cols = self.get_string_columns(df)
                if not string_cols:
                    return OperationResult(
                        success=False, error=ValueError("Нет строковых колонок")
                    )
                column = string_cols[0]

            # Извлекаем данные
            result_df = df.copy()
            extracted = df[column].str.extract(pattern, expand=False)
            result_df[f"{column}_extracted"] = extracted

            # Статистика
            successful_extracts = extracted.notna().sum()
            extract_ratio = successful_extracts / len(df) if len(df) > 0 else 0

            metadata = {
                "column": column,
                "pattern": pattern,
                "successful_extracts": int(successful_extracts),
                "extract_ratio": extract_ratio,
                "backend": backend,
            }

            return OperationResult(success=True, result=result_df, metadata=metadata)

        except Exception as e:
            return OperationResult(success=False, error=e)

    def execute_polars(
        self,
        df: Union[pl.DataFrame, pl.LazyFrame],
        lazy: bool = False,
        column: Optional[str] = None,
        pattern: str = r"str_(\d+)",
        **kwargs,
    ) -> OperationResult:
        """Извлечение по паттерну в Polars."""
        try:
            # Выбираем колонку
            if column is None:
                if lazy:
                    column = next((col for col in df.columns if "str" in col), None)
                    if column is None:
                        return OperationResult(
                            success=False, error=ValueError("Нет строковых колонок")
                        )
                else:
                    string_cols = self.get_string_columns(df)
                    if not string_cols:
                        return OperationResult(
                            success=False, error=ValueError("Нет строковых колонок")
                        )
                    column = string_cols[0]

            # Извлекаем данные
            result_df = df.with_columns(
                pl.col(column)
                .str.extract(pattern, group_index=1)
                .alias(f"{column}_extracted")
            )

            metadata = {"column": column, "pattern": pattern, "lazy": lazy}

            if not lazy:
                successful_extracts = (
                    result_df[f"{column}_extracted"].is_not_null().sum()
                )
                metadata.update(
                    {
                        "successful_extracts": int(successful_extracts),
                        "extract_ratio": (
                            successful_extracts / len(df) if len(df) > 0 else 0
                        ),
                    }
                )

            return OperationResult(success=True, result=result_df, metadata=metadata)

        except Exception as e:
            return OperationResult(success=False, error=e)


class StringConcatOperation(DataAwareOperation):
    """Конкатенация строковых колонок."""

    def __init__(self):
        super().__init__(
            name="string_concat",
            category="string",
            description="Конкатенация нескольких строковых колонок",
        )

    def execute_pandas(
        self,
        df: pd.DataFrame,
        backend: str = "numpy",
        columns: Optional[List[str]] = None,
        separator: str = " | ",
        **kwargs,
    ) -> OperationResult:
        """Конкатенация строк в Pandas."""
        try:
            # Выбираем колонки
            if columns is None:
                string_cols = self.get_string_columns(df)
                if len(string_cols) < 2:
                    # Если мало строковых колонок, добавим числовые как строки
                    numeric_cols = self.get_numeric_columns(df)
                    if numeric_cols:
                        columns = string_cols + [numeric_cols[0]]
                    else:
                        return OperationResult(
                            success=False,
                            error=ValueError("Недостаточно колонок для конкатенации"),
                        )
                else:
                    columns = string_cols[:3]  # Максимум 3 колонки

            # Выполняем конкатенацию
            result_df = df.copy()

            # Преобразуем все колонки в строки
            concat_cols = []
            for col in columns:
                if col in df.columns:
                    concat_cols.append(df[col].astype(str))

            if len(concat_cols) >= 2:
                result_df["concatenated"] = concat_cols[0]
                for col in concat_cols[1:]:
                    result_df["concatenated"] = (
                        result_df["concatenated"] + separator + col
                    )

            # Статистика
            avg_length = result_df["concatenated"].str.len().mean()
            max_length = result_df["concatenated"].str.len().max()

            metadata = {
                "columns": columns,
                "separator": separator,
                "avg_length": float(avg_length),
                "max_length": int(max_length),
                "backend": backend,
            }

            return OperationResult(success=True, result=result_df, metadata=metadata)

        except Exception as e:
            return OperationResult(success=False, error=e)

    def execute_polars(
        self,
        df: Union[pl.DataFrame, pl.LazyFrame],
        lazy: bool = False,
        columns: Optional[List[str]] = None,
        separator: str = " | ",
        **kwargs,
    ) -> OperationResult:
        """Конкатенация строк в Polars."""
        try:
            # Выбираем колонки
            if columns is None:
                if lazy:
                    # Предполагаем структуру
                    string_cols = [col for col in df.columns if "str" in col]
                    numeric_cols = [col for col in df.columns if "num" in col]
                else:
                    string_cols = self.get_string_columns(df)
                    numeric_cols = self.get_numeric_columns(df)

                if len(string_cols) < 2 and numeric_cols:
                    columns = string_cols + [numeric_cols[0]]
                elif len(string_cols) >= 2:
                    columns = string_cols[:3]
                else:
                    return OperationResult(
                        success=False, error=ValueError("Недостаточно колонок")
                    )

            # Преобразуем колонки в строки и конкатенируем
            concat_exprs = []
            for col in columns:
                if col in df.columns:
                    concat_exprs.append(pl.col(col).cast(pl.Utf8))

            if len(concat_exprs) >= 2:
                # Polars concat_str для эффективной конкатенации
                result_df = df.with_columns(
                    pl.concat_str(concat_exprs, separator=separator).alias(
                        "concatenated"
                    )
                )
            else:
                return OperationResult(
                    success=False, error=ValueError("Недостаточно валидных колонок")
                )

            metadata = {"columns": columns, "separator": separator, "lazy": lazy}

            if not lazy:
                lengths = result_df.select(pl.col("concatenated").str.len_chars())[
                    "concatenated"
                ]
                metadata.update(
                    {
                        "avg_length": float(lengths.mean()),
                        "max_length": int(lengths.max()),
                    }
                )

            return OperationResult(success=True, result=result_df, metadata=metadata)

        except Exception as e:
            return OperationResult(success=False, error=e)


class StringLengthOperation(DataAwareOperation):
    """Вычисление длины строк."""

    def __init__(self):
        super().__init__(
            name="string_length",
            category="string",
            description="Вычисление длины строк и статистики",
        )

    def execute_pandas(
        self,
        df: pd.DataFrame,
        backend: str = "numpy",
        column: Optional[str] = None,
        **kwargs,
    ) -> OperationResult:
        """Вычисление длины строк в Pandas."""
        try:
            # Выбираем колонку
            if column is None:
                string_cols = self.get_string_columns(df)
                if not string_cols:
                    return OperationResult(
                        success=False, error=ValueError("Нет строковых колонок")
                    )
                column = string_cols[0]

            # Вычисляем длины
            result_df = df.copy()
            result_df[f"{column}_length"] = df[column].str.len()

            # Добавляем категории по длине
            lengths = result_df[f"{column}_length"]
            result_df[f"{column}_length_category"] = pd.cut(
                lengths,
                bins=[0, 5, 10, 20, float("inf")],
                labels=["short", "medium", "long", "very_long"],
            )

            # Статистика
            length_stats = {
                "min": int(lengths.min()),
                "max": int(lengths.max()),
                "mean": float(lengths.mean()),
                "median": float(lengths.median()),
                "std": float(lengths.std()),
            }

            # Распределение по категориям
            category_counts = (
                result_df[f"{column}_length_category"].value_counts().to_dict()
            )

            metadata = {
                "column": column,
                "length_stats": length_stats,
                "category_distribution": {
                    str(k): int(v) for k, v in category_counts.items()
                },
                "backend": backend,
            }

            return OperationResult(success=True, result=result_df, metadata=metadata)

        except Exception as e:
            return OperationResult(success=False, error=e)

    def execute_polars(
        self,
        df: Union[pl.DataFrame, pl.LazyFrame],
        lazy: bool = False,
        column: Optional[str] = None,
        **kwargs,
    ) -> OperationResult:
        """Вычисление длины строк в Polars."""
        try:
            # Выбираем колонку
            if column is None:
                if lazy:
                    column = next((col for col in df.columns if "str" in col), None)
                    if column is None:
                        return OperationResult(
                            success=False, error=ValueError("Нет строковых колонок")
                        )
                else:
                    string_cols = self.get_string_columns(df)
                    if not string_cols:
                        return OperationResult(
                            success=False, error=ValueError("Нет строковых колонок")
                        )
                    column = string_cols[0]

            # Вычисляем длины и категории
            result_df = df.with_columns(
                [
                    pl.col(column).str.len_chars().alias(f"{column}_length"),
                    pl.when(pl.col(column).str.len_chars() <= 5)
                    .then(pl.lit("short"))
                    .when(pl.col(column).str.len_chars() <= 10)
                    .then(pl.lit("medium"))
                    .when(pl.col(column).str.len_chars() <= 20)
                    .then(pl.lit("long"))
                    .otherwise(pl.lit("very_long"))
                    .alias(f"{column}_length_category"),
                ]
            )

            metadata = {"column": column, "lazy": lazy}

            if not lazy:
                # Собираем статистику
                stats = result_df.select(
                    [
                        pl.col(f"{column}_length").min().alias("min"),
                        pl.col(f"{column}_length").max().alias("max"),
                        pl.col(f"{column}_length").mean().alias("mean"),
                        pl.col(f"{column}_length").median().alias("median"),
                        pl.col(f"{column}_length").std().alias("std"),
                    ]
                ).to_dicts()[0]

                # Распределение по категориям
                categories = result_df.group_by(f"{column}_length_category").count()
                category_dict = {
                    row[f"{column}_length_category"]: row["count"]
                    for row in categories.to_dicts()
                }

                metadata.update(
                    {
                        "length_stats": {
                            k: float(v) if k != "min" and k != "max" else int(v)
                            for k, v in stats.items()
                        },
                        "category_distribution": category_dict,
                    }
                )

            return OperationResult(success=True, result=result_df, metadata=metadata)

        except Exception as e:
            return OperationResult(success=False, error=e)


# Регистрируем все строковые операции
register_operation(StringContainsOperation())
register_operation(StringReplaceOperation())
register_operation(StringExtractOperation())
register_operation(StringConcatOperation())
register_operation(StringLengthOperation())
