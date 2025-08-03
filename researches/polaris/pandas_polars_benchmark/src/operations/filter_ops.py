"""
Операции фильтрации данных для бенчмаркинга.
"""

from typing import Optional, Union, List, Any
import numpy as np
import pandas as pd
import polars as pl

from researches.polaris.pandas_polars_benchmark.src.operations.base import (
    DataAwareOperation, OperationResult, register_operation
)


class SimpleFilterOperation(DataAwareOperation):
    """Простая фильтрация по одному условию."""
    
    def __init__(self):
        super().__init__(
            name="simple_filter",
            category="filter",
            description="Фильтрация по одному числовому условию"
        )
    
    def execute_pandas(self,
                      df: pd.DataFrame,
                      backend: str = "numpy",
                      column: Optional[str] = None,
                      threshold: Optional[float] = None,
                      **kwargs) -> OperationResult:
        """Простая фильтрация в Pandas."""
        try:
            # Выбираем колонку для фильтрации
            if column is None:
                numeric_cols = self.get_numeric_columns(df)
                if not numeric_cols:
                    return OperationResult(
                        success=False,
                        error=ValueError("Нет числовых колонок для фильтрации")
                    )
                column = numeric_cols[0]
            
            # Определяем порог (медиана если не указан)
            if threshold is None:
                threshold = df[column].median()
            
            # Выполняем фильтрацию
            result_df = df[df[column] > threshold]
            
            metadata = {
                "column": column,
                "threshold": float(threshold),
                "rows_before": len(df),
                "rows_after": len(result_df),
                "filtered_ratio": len(result_df) / len(df) if len(df) > 0 else 0
            }
            
            return OperationResult(
                success=True,
                result=result_df,
                metadata=metadata
            )
            
        except Exception as e:
            return OperationResult(success=False, error=e)
    
    def execute_polars(self,
                      df: Union[pl.DataFrame, pl.LazyFrame],
                      lazy: bool = False,
                      column: Optional[str] = None,
                      threshold: Optional[float] = None,
                      **kwargs) -> OperationResult:
        """Простая фильтрация в Polars."""
        try:
            # Для lazy нужно сначала получить статистику
            if lazy and threshold is None:
                stats_df = df.select(pl.col(column).median()).collect()
                threshold = stats_df[0, 0]
            
            # Выбираем колонку
            if column is None:
                if lazy:
                    # Для lazy берем первую колонку (предполагаем числовую)
                    column = df.columns[0]
                else:
                    numeric_cols = self.get_numeric_columns(df)
                    if not numeric_cols:
                        return OperationResult(
                            success=False,
                            error=ValueError("Нет числовых колонок для фильтрации")
                        )
                    column = numeric_cols[0]
            
            # Определяем порог
            if threshold is None and not lazy:
                threshold = df[column].median()
            
            # Выполняем фильтрацию
            result_df = df.filter(pl.col(column) > threshold)
            
            # Для lazy собираем результат для метаданных
            if lazy:
                collected = result_df.collect()
                rows_after = len(collected)
                # Возвращаем обратно lazy версию
                result_df = result_df
            else:
                rows_after = len(result_df)
            
            metadata = {
                "column": column,
                "threshold": float(threshold) if threshold is not None else None,
                "lazy": lazy
            }
            
            if not lazy:
                metadata.update({
                    "rows_before": len(df),
                    "rows_after": rows_after,
                    "filtered_ratio": rows_after / len(df) if len(df) > 0 else 0
                })
            
            return OperationResult(
                success=True,
                result=result_df,
                metadata=metadata
            )
            
        except Exception as e:
            return OperationResult(success=False, error=e)


class ComplexFilterOperation(DataAwareOperation):
    """Сложная фильтрация по нескольким условиям."""
    
    def __init__(self):
        super().__init__(
            name="complex_filter",
            category="filter",
            description="Фильтрация по нескольким условиям с AND/OR"
        )
    
    def execute_pandas(self,
                      df: pd.DataFrame,
                      backend: str = "numpy",
                      **kwargs) -> OperationResult:
        """Сложная фильтрация в Pandas."""
        try:
            numeric_cols = self.get_numeric_columns(df)
            string_cols = self.get_string_columns(df)
            
            if len(numeric_cols) < 2:
                return OperationResult(
                    success=False,
                    error=ValueError("Нужно минимум 2 числовые колонки")
                )
            
            # Создаем сложное условие
            col1, col2 = numeric_cols[0], numeric_cols[1]
            threshold1 = df[col1].quantile(0.25)
            threshold2 = df[col2].quantile(0.75)
            
            # Основное условие: (col1 > 25th percentile) AND (col2 < 75th percentile)
            condition = (df[col1] > threshold1) & (df[col2] < threshold2)
            
            # Добавляем условие по строкам если есть
            if string_cols:
                str_col = string_cols[0]
                # Ищем строки, начинающиеся с определенных символов
                str_condition = df[str_col].str.startswith(('str_1', 'str_2', 'str_3'), na=False)
                condition = condition | str_condition
            
            result_df = df[condition]
            
            metadata = {
                "conditions": {
                    "numeric": f"({col1} > {threshold1:.2f}) AND ({col2} < {threshold2:.2f})",
                    "string": f"{string_cols[0]} starts with str_1/2/3" if string_cols else None
                },
                "rows_before": len(df),
                "rows_after": len(result_df),
                "filtered_ratio": len(result_df) / len(df) if len(df) > 0 else 0
            }
            
            return OperationResult(
                success=True,
                result=result_df,
                metadata=metadata
            )
            
        except Exception as e:
            return OperationResult(success=False, error=e)
    
    def execute_polars(self,
                      df: Union[pl.DataFrame, pl.LazyFrame],
                      lazy: bool = False,
                      **kwargs) -> OperationResult:
        """Сложная фильтрация в Polars."""
        try:
            # Получаем колонки
            if lazy:
                # Для lazy предполагаем структуру
                numeric_cols = [col for col in df.columns if 'num' in col or 'mixed_num' in col]
                string_cols = [col for col in df.columns if 'str' in col or 'mixed_str' in col]
            else:
                numeric_cols = self.get_numeric_columns(df)
                string_cols = self.get_string_columns(df)
            
            if len(numeric_cols) < 2:
                return OperationResult(
                    success=False,
                    error=ValueError("Нужно минимум 2 числовые колонки")
                )
            
            col1, col2 = numeric_cols[0], numeric_cols[1]
            
            # Для lazy нужно собрать квантили
            if lazy:
                stats = df.select([
                    pl.col(col1).quantile(0.25).alias("q1"),
                    pl.col(col2).quantile(0.75).alias("q2")
                ]).collect()
                threshold1 = stats["q1"][0]
                threshold2 = stats["q2"][0]
            else:
                threshold1 = df[col1].quantile(0.25)
                threshold2 = df[col2].quantile(0.75)
            
            # Создаем условия
            condition = (pl.col(col1) > threshold1) & (pl.col(col2) < threshold2)
            
            # Добавляем строковое условие
            if string_cols:
                str_col = string_cols[0]
                str_condition = pl.col(str_col).str.starts_with("str_1") | \
                               pl.col(str_col).str.starts_with("str_2") | \
                               pl.col(str_col).str.starts_with("str_3")
                condition = condition | str_condition
            
            result_df = df.filter(condition)
            
            metadata = {
                "lazy": lazy,
                "columns_used": {
                    "numeric": [col1, col2],
                    "string": string_cols[:1] if string_cols else []
                }
            }
            
            if not lazy:
                metadata.update({
                    "rows_before": len(df),
                    "rows_after": len(result_df),
                    "filtered_ratio": len(result_df) / len(df) if len(df) > 0 else 0
                })
            
            return OperationResult(
                success=True,
                result=result_df,
                metadata=metadata
            )
            
        except Exception as e:
            return OperationResult(success=False, error=e)


class IsInFilterOperation(DataAwareOperation):
    """Фильтрация с использованием isin()."""
    
    def __init__(self):
        super().__init__(
            name="isin_filter",
            category="filter",
            description="Фильтрация по списку значений (isin)"
        )
    
    def execute_pandas(self,
                      df: pd.DataFrame,
                      backend: str = "numpy",
                      column: Optional[str] = None,
                      values: Optional[List[Any]] = None,
                      **kwargs) -> OperationResult:
        """Фильтрация isin в Pandas."""
        try:
            # Выбираем колонку
            if column is None:
                # Предпочитаем строковые колонки для isin
                string_cols = self.get_string_columns(df)
                if string_cols:
                    column = string_cols[0]
                else:
                    numeric_cols = self.get_numeric_columns(df)
                    if numeric_cols:
                        column = numeric_cols[0]
                    else:
                        return OperationResult(
                            success=False,
                            error=ValueError("Нет подходящих колонок")
                        )
            
            # Генерируем список значений если не указан
            if values is None:
                unique_values = df[column].unique()
                # Берем 10% уникальных значений
                n_values = max(1, len(unique_values) // 10)
                values = np.random.choice(unique_values, size=n_values, replace=False).tolist()
            
            # Фильтруем
            result_df = df[df[column].isin(values)]
            
            metadata = {
                "column": column,
                "n_values": len(values),
                "rows_before": len(df),
                "rows_after": len(result_df),
                "filtered_ratio": len(result_df) / len(df) if len(df) > 0 else 0
            }
            
            return OperationResult(
                success=True,
                result=result_df,
                metadata=metadata
            )
            
        except Exception as e:
            return OperationResult(success=False, error=e)
    
    def execute_polars(self,
                      df: Union[pl.DataFrame, pl.LazyFrame],
                      lazy: bool = False,
                      column: Optional[str] = None,
                      values: Optional[List[Any]] = None,
                      **kwargs) -> OperationResult:
        """Фильтрация isin в Polars."""
        try:
            # Выбираем колонку
            if column is None:
                if lazy:
                    # Для lazy предполагаем первую строковую колонку
                    column = next((col for col in df.columns if 'str' in col), df.columns[0])
                else:
                    string_cols = self.get_string_columns(df)
                    if string_cols:
                        column = string_cols[0]
                    else:
                        numeric_cols = self.get_numeric_columns(df)
                        column = numeric_cols[0] if numeric_cols else df.columns[0]
            
            # Для lazy и отсутствующих values нужно сначала получить уникальные значения
            if values is None:
                if lazy:
                    unique_df = df.select(pl.col(column).unique()).collect()
                    unique_values = unique_df[column].to_list()
                else:
                    unique_values = df[column].unique().to_list()
                
                n_values = max(1, len(unique_values) // 10)
                values = np.random.choice(unique_values, size=n_values, replace=False).tolist()
            
            # Фильтруем
            result_df = df.filter(pl.col(column).is_in(values))
            
            metadata = {
                "column": column,
                "n_values": len(values),
                "lazy": lazy
            }
            
            if not lazy:
                metadata.update({
                    "rows_before": len(df),
                    "rows_after": len(result_df),
                    "filtered_ratio": len(result_df) / len(df) if len(df) > 0 else 0
                })
            
            return OperationResult(
                success=True,
                result=result_df,
                metadata=metadata
            )
            
        except Exception as e:
            return OperationResult(success=False, error=e)


class PatternFilterOperation(DataAwareOperation):
    """Фильтрация по строковому паттерну."""
    
    def __init__(self):
        super().__init__(
            name="pattern_filter",
            category="filter",
            description="Фильтрация по строковому паттерну (contains/regex)"
        )
    
    def execute_pandas(self,
                      df: pd.DataFrame,
                      backend: str = "numpy",
                      column: Optional[str] = None,
                      pattern: str = "str_[0-9]{2}",
                      regex: bool = True,
                      **kwargs) -> OperationResult:
        """Фильтрация по паттерну в Pandas."""
        try:
            # Выбираем строковую колонку
            if column is None:
                string_cols = self.get_string_columns(df)
                if not string_cols:
                    return OperationResult(
                        success=False,
                        error=ValueError("Нет строковых колонок")
                    )
                column = string_cols[0]
            
            # Фильтруем
            if regex:
                result_df = df[df[column].str.contains(pattern, na=False, regex=True)]
            else:
                result_df = df[df[column].str.contains(pattern, na=False, regex=False)]
            
            metadata = {
                "column": column,
                "pattern": pattern,
                "regex": regex,
                "rows_before": len(df),
                "rows_after": len(result_df),
                "filtered_ratio": len(result_df) / len(df) if len(df) > 0 else 0
            }
            
            return OperationResult(
                success=True,
                result=result_df,
                metadata=metadata
            )
            
        except Exception as e:
            return OperationResult(success=False, error=e)
    
    def execute_polars(self,
                      df: Union[pl.DataFrame, pl.LazyFrame],
                      lazy: bool = False,
                      column: Optional[str] = None,
                      pattern: str = "str_[0-9]{2}",
                      regex: bool = True,
                      **kwargs) -> OperationResult:
        """Фильтрация по паттерну в Polars."""
        try:
            # Выбираем колонку
            if column is None:
                if lazy:
                    # Предполагаем первую строковую колонку
                    column = next((col for col in df.columns if 'str' in col), None)
                    if column is None:
                        return OperationResult(
                            success=False,
                            error=ValueError("Нет строковых колонок")
                        )
                else:
                    string_cols = self.get_string_columns(df)
                    if not string_cols:
                        return OperationResult(
                            success=False,
                            error=ValueError("Нет строковых колонок")
                        )
                    column = string_cols[0]
            
            # Фильтруем
            if regex:
                result_df = df.filter(pl.col(column).str.contains(pattern))
            else:
                result_df = df.filter(pl.col(column).str.contains_literal(pattern))
            
            metadata = {
                "column": column,
                "pattern": pattern,
                "regex": regex,
                "lazy": lazy
            }
            
            if not lazy:
                metadata.update({
                    "rows_before": len(df),
                    "rows_after": len(result_df),
                    "filtered_ratio": len(result_df) / len(df) if len(df) > 0 else 0
                })
            
            return OperationResult(
                success=True,
                result=result_df,
                metadata=metadata
            )
            
        except Exception as e:
            return OperationResult(success=False, error=e)


# Регистрируем все операции фильтрации
register_operation(SimpleFilterOperation())
register_operation(ComplexFilterOperation())
register_operation(IsInFilterOperation())
register_operation(PatternFilterOperation())
