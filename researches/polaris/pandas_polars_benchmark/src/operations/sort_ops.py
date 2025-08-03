"""
Операции сортировки данных для бенчмаркинга.
"""

from typing import Optional, Union, List, Dict, Any, Tuple
import pandas as pd
import polars as pl
import numpy as np

from researches.polaris.pandas_polars_benchmark.src.operations.base import (
    DataAwareOperation, OperationResult, register_operation
)


class SingleColumnSortOperation(DataAwareOperation):
    """Сортировка по одной колонке."""
    
    def __init__(self):
        super().__init__(
            name="single_column_sort",
            category="sort",
            description="Сортировка данных по одной колонке"
        )
    
    def execute_pandas(self,
                      df: pd.DataFrame,
                      backend: str = "numpy",
                      column: Optional[str] = None,
                      ascending: bool = True,
                      na_position: str = "last",
                      **kwargs) -> OperationResult:
        """Сортировка по одной колонке в Pandas."""
        try:
            # Выбираем колонку для сортировки
            if column is None:
                # Приоритеты: числовые -> datetime -> строковые
                numeric_cols = self.get_numeric_columns(df)
                datetime_cols = self.get_datetime_columns(df)
                string_cols = self.get_string_columns(df)
                
                if numeric_cols:
                    column = numeric_cols[0]
                elif datetime_cols:
                    column = datetime_cols[0]
                elif string_cols:
                    column = string_cols[0]
                else:
                    column = df.columns[0]
            
            # Засекаем количество null значений для метрик
            null_count = df[column].isna().sum()
            
            # Выполняем сортировку
            result_df = df.sort_values(
                by=column,
                ascending=ascending,
                na_position=na_position,
                ignore_index=True  # Сбрасываем индекс
            )
            
            # Проверяем корректность сортировки (первые и последние значения)
            if len(result_df) > 0:
                first_value = result_df[column].iloc[0]
                last_value = result_df[column].iloc[-1]
                
                # Для строк и других типов
                if pd.api.types.is_string_dtype(df[column]):
                    first_value = str(first_value) if pd.notna(first_value) else None
                    last_value = str(last_value) if pd.notna(last_value) else None
            else:
                first_value = None
                last_value = None
            
            metadata = {
                "column": column,
                "column_dtype": str(df[column].dtype),
                "ascending": ascending,
                "na_position": na_position,
                "null_count": int(null_count),
                "rows": len(result_df),
                "first_value": first_value,
                "last_value": last_value,
                "backend": backend
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
                      ascending: bool = True,
                      nulls_last: bool = True,
                      **kwargs) -> OperationResult:
        """Сортировка по одной колонке в Polars."""
        try:
            # Выбираем колонку
            if column is None:
                if lazy:
                    # Для lazy выбираем первую подходящую колонку
                    numeric_cols = [col for col in df.columns if 'num' in col or 'mixed_num' in col]
                    datetime_cols = [col for col in df.columns if 'datetime' in col]
                    string_cols = [col for col in df.columns if 'str' in col or 'mixed_str' in col]
                else:
                    numeric_cols = self.get_numeric_columns(df)
                    datetime_cols = self.get_datetime_columns(df)
                    string_cols = self.get_string_columns(df)
                
                if numeric_cols:
                    column = numeric_cols[0]
                elif datetime_cols:
                    column = datetime_cols[0]
                elif string_cols:
                    column = string_cols[0]
                else:
                    column = df.columns[0]
            
            # Для eager mode считаем null значения
            null_count = None
            if not lazy:
                null_count = df[column].null_count()
            
            # Выполняем сортировку
            result_df = df.sort(
                column,
                descending=not ascending,
                nulls_last=nulls_last
            )
            
            metadata = {
                "column": column,
                "ascending": ascending,
                "nulls_last": nulls_last,
                "lazy": lazy
            }
            
            if not lazy:
                metadata["null_count"] = null_count
                metadata["rows"] = len(result_df)
                
                # Получаем первое и последнее значения
                if len(result_df) > 0:
                    first_value = result_df[column][0]
                    last_value = result_df[column][-1]
                    
                    # Конвертируем в Python типы для сериализации
                    if first_value is not None:
                        first_value = first_value.item() if hasattr(first_value, 'item') else first_value
                    if last_value is not None:
                        last_value = last_value.item() if hasattr(last_value, 'item') else last_value
                    
                    metadata["first_value"] = first_value
                    metadata["last_value"] = last_value
            
            return OperationResult(
                success=True,
                result=result_df,
                metadata=metadata
            )
            
        except Exception as e:
            return OperationResult(success=False, error=e)


class MultiColumnSortOperation(DataAwareOperation):
    """Сортировка по нескольким колонкам."""
    
    def __init__(self):
        super().__init__(
            name="multi_column_sort",
            category="sort",
            description="Сортировка данных по нескольким колонкам с разными направлениями"
        )
    
    def _select_sort_columns(self, df: Union[pd.DataFrame, pl.DataFrame], 
                           is_lazy: bool = False) -> List[Tuple[str, bool]]:
        """
        Выбирает колонки для сортировки с направлениями.
        
        Returns:
            List[(column_name, ascending)]
        """
        sort_columns = []
        
        if is_lazy:
            # Для lazy режима предполагаем структуру
            # Первая строковая колонка - по возрастанию
            string_cols = [col for col in df.columns if 'str' in col]
            if string_cols:
                sort_columns.append((string_cols[0], True))
            
            # Первая числовая колонка - по убыванию
            numeric_cols = [col for col in df.columns if 'num' in col]
            if numeric_cols:
                sort_columns.append((numeric_cols[0], False))
            
            # Datetime колонка - по возрастанию
            datetime_cols = [col for col in df.columns if 'datetime' in col]
            if datetime_cols:
                sort_columns.append((datetime_cols[0], True))
        else:
            # Для eager mode анализируем типы
            if isinstance(df, pd.DataFrame):
                string_cols = self.get_string_columns(df)
                numeric_cols = self.get_numeric_columns(df)
                datetime_cols = self.get_datetime_columns(df)
            else:  # Polars
                string_cols = self.get_string_columns(df)
                numeric_cols = self.get_numeric_columns(df)
                datetime_cols = self.get_datetime_columns(df)
            
            # Строковая колонка - ascending
            if string_cols:
                sort_columns.append((string_cols[0], True))
            
            # Числовая колонка - descending (для разнообразия)
            if numeric_cols:
                # Берем колонку, которая еще не используется
                for col in numeric_cols:
                    if not any(col == sc[0] for sc in sort_columns):
                        sort_columns.append((col, False))
                        break
            
            # Datetime колонка - ascending
            if datetime_cols:
                for col in datetime_cols:
                    if not any(col == sc[0] for sc in sort_columns):
                        sort_columns.append((col, True))
                        break
        
        # Если не хватает колонок, добавляем любые оставшиеся
        if len(sort_columns) < 2:
            used_cols = {sc[0] for sc in sort_columns}
            for i, col in enumerate(df.columns):
                if col not in used_cols:
                    # Чередуем направления
                    ascending = (i % 2 == 0)
                    sort_columns.append((col, ascending))
                    if len(sort_columns) >= 3:  # Максимум 3 колонки
                        break
        
        return sort_columns[:3]  # Ограничиваем 3 колонками
    
    def execute_pandas(self,
                      df: pd.DataFrame,
                      backend: str = "numpy",
                      columns: Optional[List[str]] = None,
                      ascending: Optional[List[bool]] = None,
                      **kwargs) -> OperationResult:
        """Сортировка по нескольким колонкам в Pandas."""
        try:
            # Автоматический выбор колонок если не указаны
            if columns is None:
                sort_specs = self._select_sort_columns(df)
                columns = [spec[0] for spec in sort_specs]
                ascending = [spec[1] for spec in sort_specs]
            elif ascending is None:
                # Если указаны колонки, но не направления
                ascending = [True] * len(columns)
            
            # Проверяем, что количество колонок и направлений совпадает
            if len(columns) != len(ascending):
                ascending = ascending[:len(columns)] if len(ascending) > len(columns) else \
                           ascending + [True] * (len(columns) - len(ascending))
            
            # Считаем null значения для каждой колонки
            null_counts = {col: int(df[col].isna().sum()) for col in columns}
            
            # Выполняем сортировку
            result_df = df.sort_values(
                by=columns,
                ascending=ascending,
                na_position='last',
                ignore_index=True
            )
            
            # Получаем информацию о первых значениях после сортировки
            first_row_values = {}
            if len(result_df) > 0:
                for col in columns:
                    value = result_df[col].iloc[0]
                    if pd.notna(value):
                        if pd.api.types.is_string_dtype(df[col]):
                            value = str(value)
                        elif pd.api.types.is_numeric_dtype(df[col]):
                            value = float(value)
                        first_row_values[col] = value
            
            metadata = {
                "columns": columns,
                "ascending": ascending,
                "null_counts": null_counts,
                "rows": len(result_df),
                "sort_key_count": len(columns),
                "first_row_values": first_row_values,
                "backend": backend
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
                      columns: Optional[List[str]] = None,
                      descending: Optional[List[bool]] = None,
                      **kwargs) -> OperationResult:
        """Сортировка по нескольким колонкам в Polars."""
        try:
            # Автоматический выбор колонок
            if columns is None:
                sort_specs = self._select_sort_columns(df, is_lazy=lazy)
                columns = [spec[0] for spec in sort_specs]
                # В Polars используется descending, а не ascending
                descending = [not spec[1] for spec in sort_specs]
            elif descending is None:
                descending = [False] * len(columns)
            
            # Проверяем соответствие длин
            if len(columns) != len(descending):
                descending = descending[:len(columns)] if len(descending) > len(columns) else \
                           descending + [False] * (len(columns) - len(descending))
            
            # Выполняем сортировку
            result_df = df.sort(
                columns,
                descending=descending,
                nulls_last=True
            )
            
            metadata = {
                "columns": columns,
                "descending": descending,
                "sort_key_count": len(columns),
                "lazy": lazy
            }
            
            if not lazy:
                metadata["rows"] = len(result_df)
                
                # Считаем null значения
                null_counts = {}
                for col in columns:
                    null_counts[col] = result_df[col].null_count()
                metadata["null_counts"] = null_counts
                
                # Первая строка значений
                first_row_values = {}
                if len(result_df) > 0:
                    for col in columns:
                        value = result_df[col][0]
                        if value is not None:
                            # Конвертируем в Python тип
                            if hasattr(value, 'item'):
                                value = value.item()
                            first_row_values[col] = value
                metadata["first_row_values"] = first_row_values
            
            return OperationResult(
                success=True,
                result=result_df,
                metadata=metadata
            )
            
        except Exception as e:
            return OperationResult(success=False, error=e)


class StableSortOperation(DataAwareOperation):
    """Стабильная сортировка с сохранением порядка равных элементов."""
    
    def __init__(self):
        super().__init__(
            name="stable_sort",
            category="sort",
            description="Стабильная сортировка с сохранением порядка равных элементов"
        )
    
    def execute_pandas(self,
                      df: pd.DataFrame,
                      backend: str = "numpy",
                      **kwargs) -> OperationResult:
        """Стабильная сортировка в Pandas."""
        try:
            # Выбираем колонку с низкой кардинальностью для демонстрации стабильности
            string_cols = self.get_string_columns(df)
            column = None
            
            if string_cols:
                # Ищем колонку с небольшим количеством уникальных значений
                for col in string_cols:
                    unique_count = df[col].nunique()
                    if unique_count < len(df) * 0.1:  # Менее 10% уникальных
                        column = col
                        break
            
            if column is None:
                # Берем первую строковую или первую колонку
                column = string_cols[0] if string_cols else df.columns[0]
            
            # Добавляем индексную колонку для проверки стабильности
            df_with_index = df.copy()
            df_with_index['_original_order'] = range(len(df))
            
            # Выполняем стабильную сортировку (по умолчанию в pandas)
            result_df = df_with_index.sort_values(
                by=column,
                kind='stable',  # Явно указываем стабильную сортировку
                ignore_index=False  # Сохраняем индекс для проверки
            )
            
            # Проверяем стабильность - для одинаковых значений порядок должен сохраняться
            unique_values = df[column].unique()[:5]  # Проверяем первые 5 уникальных значений
            stability_check = {}
            
            for value in unique_values:
                if pd.notna(value):
                    mask = result_df[column] == value
                    if mask.sum() > 1:  # Есть дубликаты
                        orders = result_df.loc[mask, '_original_order'].tolist()
                        is_stable = orders == sorted(orders)
                        stability_check[str(value)] = is_stable
            
            # Убираем служебную колонку из результата
            result_df = result_df.drop('_original_order', axis=1)
            
            metadata = {
                "column": column,
                "unique_count": int(df[column].nunique()),
                "total_rows": len(df),
                "sort_algorithm": "stable",
                "stability_verified": all(stability_check.values()) if stability_check else True,
                "stability_check_sample": stability_check,
                "backend": backend
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
        """Стабильная сортировка в Polars."""
        try:
            # Выбираем колонку
            if lazy:
                string_cols = [col for col in df.columns if 'str' in col]
            else:
                string_cols = self.get_string_columns(df)
            
            column = string_cols[0] if string_cols else df.columns[0]
            
            # Добавляем индекс для проверки стабильности
            if lazy:
                df_with_index = df.with_row_count(name="_original_order")
            else:
                df_with_index = df.with_row_count(name="_original_order")
            
            # Polars sort всегда стабильна
            result_df = df_with_index.sort(column)
            
            # Для eager mode проверяем стабильность
            stability_info = {}
            if not lazy:
                unique_count = df[column].n_unique()
                
                # Проверка стабильности для нескольких значений
                unique_values = df[column].unique().head(5)
                stability_check = {}
                
                for value in unique_values:
                    if value is not None:
                        mask = result_df[column] == value
                        filtered = result_df.filter(mask)
                        if len(filtered) > 1:
                            orders = filtered["_original_order"].to_list()
                            is_stable = orders == sorted(orders)
                            stability_check[str(value)] = is_stable
                
                stability_info = {
                    "unique_count": unique_count,
                    "stability_verified": all(stability_check.values()) if stability_check else True,
                    "stability_check_sample": stability_check
                }
            
            # Убираем служебную колонку
            result_df = result_df.drop("_original_order")
            
            metadata = {
                "column": column,
                "sort_algorithm": "stable (default in polars)",
                "lazy": lazy,
                **stability_info
            }
            
            if not lazy:
                metadata["total_rows"] = len(df)
            
            return OperationResult(
                success=True,
                result=result_df,
                metadata=metadata
            )
            
        except Exception as e:
            return OperationResult(success=False, error=e)


# Регистрируем все операции сортировки
register_operation(SingleColumnSortOperation())
register_operation(MultiColumnSortOperation())
register_operation(StableSortOperation())
