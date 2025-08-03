"""
Операции группировки и агрегации данных для бенчмаркинга.
"""

from typing import Optional, Union, List, Dict, Any
import pandas as pd
import polars as pl

from operations.base import (
    DataAwareOperation, OperationResult, register_operation
)


class SingleColumnGroupByOperation(DataAwareOperation):
    """Группировка по одной колонке с простой агрегацией."""
    
    def __init__(self):
        super().__init__(
            name="single_column_groupby",
            category="groupby",
            description="Группировка по одной колонке с агрегацией sum/mean/count"
        )
    
    def execute_pandas(self,
                      df: pd.DataFrame,
                      backend: str = "numpy",
                      group_column: Optional[str] = None,
                      agg_column: Optional[str] = None,
                      agg_func: str = "mean",
                      **kwargs) -> OperationResult:
        """Группировка в Pandas."""
        try:
            # Выбираем колонку для группировки
            if group_column is None:
                string_cols = self.get_string_columns(df)
                if string_cols:
                    group_column = string_cols[0]
                else:
                    # Если нет строковых, берем первую колонку
                    group_column = df.columns[0]
            
            # Выбираем колонку для агрегации
            if agg_column is None:
                numeric_cols = self.get_numeric_columns(df)
                if numeric_cols and numeric_cols[0] != group_column:
                    agg_column = numeric_cols[0]
                else:
                    # Берем вторую колонку если первая используется для группировки
                    for col in df.columns:
                        if col != group_column and pd.api.types.is_numeric_dtype(df[col]):
                            agg_column = col
                            break
            
            if agg_column is None:
                return OperationResult(
                    success=False,
                    error=ValueError("Нет подходящей колонки для агрегации")
                )
            
            # Выполняем группировку
            if agg_func == "mean":
                result_df = df.groupby(group_column)[agg_column].mean().reset_index()
            elif agg_func == "sum":
                result_df = df.groupby(group_column)[agg_column].sum().reset_index()
            elif agg_func == "count":
                result_df = df.groupby(group_column).size().reset_index(name='count')
            else:
                result_df = df.groupby(group_column)[agg_column].agg(agg_func).reset_index()
            
            metadata = {
                "group_column": group_column,
                "agg_column": agg_column,
                "agg_func": agg_func,
                "groups_count": len(result_df),
                "rows_before": len(df),
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
                      group_column: Optional[str] = None,
                      agg_column: Optional[str] = None,
                      agg_func: str = "mean",
                      **kwargs) -> OperationResult:
        """Группировка в Polars."""
        try:
            # Выбираем колонки
            if group_column is None:
                if lazy:
                    # Для lazy предполагаем первую строковую колонку
                    group_column = next((col for col in df.columns if 'str' in col), df.columns[0])
                else:
                    string_cols = self.get_string_columns(df)
                    group_column = string_cols[0] if string_cols else df.columns[0]
            
            if agg_column is None:
                if lazy:
                    # Предполагаем первую числовую колонку
                    agg_column = next((col for col in df.columns 
                                     if 'num' in col and col != group_column), None)
                else:
                    numeric_cols = self.get_numeric_columns(df)
                    agg_column = next((col for col in numeric_cols if col != group_column), None)
            
            if agg_column is None and agg_func != "count":
                return OperationResult(
                    success=False,
                    error=ValueError("Нет подходящей колонки для агрегации")
                )
            
            # Выполняем группировку
            if agg_func == "mean":
                result_df = df.group_by(group_column).agg(pl.col(agg_column).mean())
            elif agg_func == "sum":
                result_df = df.group_by(group_column).agg(pl.col(agg_column).sum())
            elif agg_func == "count":
                result_df = df.group_by(group_column).count()
            else:
                # Общий случай
                result_df = df.group_by(group_column).agg(
                    getattr(pl.col(agg_column), agg_func)()
                )
            
            metadata = {
                "group_column": group_column,
                "agg_column": agg_column,
                "agg_func": agg_func,
                "lazy": lazy
            }
            
            if not lazy:
                metadata["groups_count"] = len(result_df)
                metadata["rows_before"] = len(df)
            
            return OperationResult(
                success=True,
                result=result_df,
                metadata=metadata
            )
            
        except Exception as e:
            return OperationResult(success=False, error=e)


class MultiColumnGroupByOperation(DataAwareOperation):
    """Группировка по нескольким колонкам."""
    
    def __init__(self):
        super().__init__(
            name="multi_column_groupby",
            category="groupby",
            description="Группировка по нескольким колонкам с агрегацией"
        )
    
    def execute_pandas(self,
                      df: pd.DataFrame,
                      backend: str = "numpy",
                      **kwargs) -> OperationResult:
        """Группировка по нескольким колонкам в Pandas."""
        try:
            # Выбираем колонки для группировки
            string_cols = self.get_string_columns(df)
            numeric_cols = self.get_numeric_columns(df)
            
            # Берем первые две строковые колонки или комбинацию
            if len(string_cols) >= 2:
                group_columns = string_cols[:2]
            elif len(string_cols) == 1 and len(numeric_cols) >= 1:
                # Используем одну строковую и одну числовую
                group_columns = [string_cols[0], numeric_cols[0]]
                numeric_cols = numeric_cols[1:]  # Убираем использованную
            else:
                # Используем первые две колонки
                group_columns = list(df.columns[:2])
            
            # Выбираем колонки для агрегации
            agg_columns = []
            for col in numeric_cols:
                if col not in group_columns:
                    agg_columns.append(col)
                    if len(agg_columns) >= 2:
                        break
            
            if not agg_columns:
                return OperationResult(
                    success=False,
                    error=ValueError("Недостаточно числовых колонок для агрегации")
                )
            
            # Выполняем группировку с несколькими агрегациями
            agg_dict = {}
            for col in agg_columns:
                agg_dict[col] = ['mean', 'sum', 'count']
            
            result_df = df.groupby(group_columns).agg(agg_dict)
            
            # Flatten column names
            result_df.columns = ['_'.join(col).strip() for col in result_df.columns.values]
            result_df = result_df.reset_index()
            
            metadata = {
                "group_columns": group_columns,
                "agg_columns": agg_columns,
                "groups_count": len(result_df),
                "rows_before": len(df),
                "aggregations_count": len(agg_columns) * 3  # 3 функции на колонку
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
        """Группировка по нескольким колонкам в Polars."""
        try:
            # Определяем колонки
            if lazy:
                # Для lazy режима предполагаем структуру
                string_cols = [col for col in df.columns if 'str' in col]
                numeric_cols = [col for col in df.columns if 'num' in col or 'mixed_num' in col]
            else:
                string_cols = self.get_string_columns(df)
                numeric_cols = self.get_numeric_columns(df)
            
            # Выбираем колонки для группировки
            if len(string_cols) >= 2:
                group_columns = string_cols[:2]
            elif len(string_cols) == 1 and len(numeric_cols) >= 1:
                group_columns = [string_cols[0], numeric_cols[0]]
                numeric_cols = numeric_cols[1:]
            else:
                group_columns = df.columns[:2]
            
            # Выбираем колонки для агрегации
            agg_columns = []
            for col in numeric_cols:
                if col not in group_columns:
                    agg_columns.append(col)
                    if len(agg_columns) >= 2:
                        break
            
            if not agg_columns:
                return OperationResult(
                    success=False,
                    error=ValueError("Недостаточно числовых колонок для агрегации")
                )
            
            # Создаем выражения агрегации
            agg_exprs = []
            for col in agg_columns:
                agg_exprs.extend([
                    pl.col(col).mean().alias(f"{col}_mean"),
                    pl.col(col).sum().alias(f"{col}_sum"),
                    pl.col(col).count().alias(f"{col}_count")
                ])
            
            # Выполняем группировку
            result_df = df.group_by(group_columns).agg(agg_exprs)
            
            metadata = {
                "group_columns": group_columns,
                "agg_columns": agg_columns,
                "lazy": lazy,
                "aggregations_count": len(agg_exprs)
            }
            
            if not lazy:
                metadata["groups_count"] = len(result_df)
                metadata["rows_before"] = len(df)
            
            return OperationResult(
                success=True,
                result=result_df,
                metadata=metadata
            )
            
        except Exception as e:
            return OperationResult(success=False, error=e)


class MultiAggregationOperation(DataAwareOperation):
    """Множественные агрегации с различными функциями."""
    
    def __init__(self):
        super().__init__(
            name="multi_aggregation",
            category="groupby",
            description="Группировка с множественными агрегатными функциями"
        )
    
    def execute_pandas(self,
                      df: pd.DataFrame,
                      backend: str = "numpy",
                      **kwargs) -> OperationResult:
        """Множественные агрегации в Pandas."""
        try:
            # Выбираем колонку для группировки
            string_cols = self.get_string_columns(df)
            group_column = string_cols[0] if string_cols else df.columns[0]
            
            # Выбираем числовые колонки для агрегации
            numeric_cols = self.get_numeric_columns(df)
            agg_columns = [col for col in numeric_cols if col != group_column][:3]
            
            if not agg_columns:
                return OperationResult(
                    success=False,
                    error=ValueError("Нет числовых колонок для агрегации")
                )
            
            # Создаем словарь агрегаций
            agg_dict = {}
            for i, col in enumerate(agg_columns):
                if i == 0:
                    # Первая колонка - все основные агрегации
                    agg_dict[col] = ['sum', 'mean', 'std', 'min', 'max', 'count']
                elif i == 1:
                    # Вторая колонка - подмножество
                    agg_dict[col] = ['mean', 'std', 'median']
                else:
                    # Остальные - базовые
                    agg_dict[col] = ['sum', 'mean']
            
            # Выполняем группировку
            result_df = df.groupby(group_column).agg(agg_dict)
            
            # Flatten column names
            result_df.columns = ['_'.join(col).strip() for col in result_df.columns.values]
            result_df = result_df.reset_index()
            
            # Добавляем дополнительные вычисления
            if len(result_df) > 0 and f"{agg_columns[0]}_sum" in result_df.columns:
                # Добавляем процент от общей суммы
                total_sum = result_df[f"{agg_columns[0]}_sum"].sum()
                if total_sum > 0:
                    result_df[f"{agg_columns[0]}_pct"] = (
                        result_df[f"{agg_columns[0]}_sum"] / total_sum * 100
                    )
            
            metadata = {
                "group_column": group_column,
                "agg_columns": agg_columns,
                "groups_count": len(result_df),
                "rows_before": len(df),
                "total_aggregations": sum(len(funcs) for funcs in agg_dict.values())
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
        """Множественные агрегации в Polars."""
        try:
            # Определяем колонки
            if lazy:
                string_cols = [col for col in df.columns if 'str' in col]
                numeric_cols = [col for col in df.columns if 'num' in col or 'mixed_num' in col]
            else:
                string_cols = self.get_string_columns(df)
                numeric_cols = self.get_numeric_columns(df)
            
            group_column = string_cols[0] if string_cols else df.columns[0]
            agg_columns = [col for col in numeric_cols if col != group_column][:3]
            
            if not agg_columns:
                return OperationResult(
                    success=False,
                    error=ValueError("Нет числовых колонок для агрегации")
                )
            
            # Создаем выражения агрегации
            agg_exprs = []
            
            # Первая колонка - все агрегации
            if len(agg_columns) > 0:
                col = agg_columns[0]
                agg_exprs.extend([
                    pl.col(col).sum().alias(f"{col}_sum"),
                    pl.col(col).mean().alias(f"{col}_mean"),
                    pl.col(col).std().alias(f"{col}_std"),
                    pl.col(col).min().alias(f"{col}_min"),
                    pl.col(col).max().alias(f"{col}_max"),
                    pl.col(col).count().alias(f"{col}_count")
                ])
            
            # Вторая колонка - подмножество
            if len(agg_columns) > 1:
                col = agg_columns[1]
                agg_exprs.extend([
                    pl.col(col).mean().alias(f"{col}_mean"),
                    pl.col(col).std().alias(f"{col}_std"),
                    pl.col(col).median().alias(f"{col}_median")
                ])
            
            # Остальные - базовые
            for col in agg_columns[2:]:
                agg_exprs.extend([
                    pl.col(col).sum().alias(f"{col}_sum"),
                    pl.col(col).mean().alias(f"{col}_mean")
                ])
            
            # Выполняем группировку
            result_df = df.group_by(group_column).agg(agg_exprs)
            
            # Добавляем процент от общей суммы (только для eager mode)
            if not lazy and len(agg_columns) > 0:
                sum_col = f"{agg_columns[0]}_sum"
                if sum_col in result_df.columns:
                    total_sum = result_df[sum_col].sum()
                    if total_sum > 0:
                        result_df = result_df.with_columns(
                            (pl.col(sum_col) / total_sum * 100).alias(f"{agg_columns[0]}_pct")
                        )
            
            metadata = {
                "group_column": group_column,
                "agg_columns": agg_columns,
                "lazy": lazy,
                "total_aggregations": len(agg_exprs)
            }
            
            if not lazy:
                metadata["groups_count"] = len(result_df)
                metadata["rows_before"] = len(df)
            
            return OperationResult(
                success=True,
                result=result_df,
                metadata=metadata
            )
            
        except Exception as e:
            return OperationResult(success=False, error=e)


class WindowFunctionOperation(DataAwareOperation):
    """Оконные функции (rolling/expanding)."""
    
    def __init__(self):
        super().__init__(
            name="window_functions",
            category="groupby",
            description="Оконные функции с группировкой"
        )
    
    def execute_pandas(self,
                      df: pd.DataFrame,
                      backend: str = "numpy",
                      window_size: int = 10,
                      **kwargs) -> OperationResult:
        """Оконные функции в Pandas."""
        try:
            # Берем первую числовую колонку
            numeric_cols = self.get_numeric_columns(df)
            if not numeric_cols:
                return OperationResult(
                    success=False,
                    error=ValueError("Нет числовых колонок для оконных функций")
                )
            
            value_column = numeric_cols[0]
            
            # Проверяем наличие datetime колонки
            datetime_cols = self.get_datetime_columns(df)
            
            if datetime_cols:
                # Если есть datetime, используем её для сортировки
                df_sorted = df.sort_values(datetime_cols[0])
                
                # Добавляем rolling статистики
                result_df = df_sorted.copy()
                result_df[f'{value_column}_rolling_mean'] = (
                    df_sorted[value_column].rolling(window=window_size, min_periods=1).mean()
                )
                result_df[f'{value_column}_rolling_std'] = (
                    df_sorted[value_column].rolling(window=window_size, min_periods=1).std()
                )
                result_df[f'{value_column}_expanding_mean'] = (
                    df_sorted[value_column].expanding(min_periods=1).mean()
                )
            else:
                # Без datetime просто применяем rolling
                result_df = df.copy()
                result_df[f'{value_column}_rolling_mean'] = (
                    df[value_column].rolling(window=window_size, min_periods=1).mean()
                )
                result_df[f'{value_column}_rolling_std'] = (
                    df[value_column].rolling(window=window_size, min_periods=1).std()
                )
                result_df[f'{value_column}_expanding_mean'] = (
                    df[value_column].expanding(min_periods=1).mean()
                )
            
            # Добавляем rank
            result_df[f'{value_column}_rank'] = df[value_column].rank(method='average')
            result_df[f'{value_column}_pct_rank'] = df[value_column].rank(pct=True)
            
            metadata = {
                "value_column": value_column,
                "window_size": window_size,
                "rows": len(result_df),
                "new_columns": 5,  # rolling_mean, rolling_std, expanding_mean, rank, pct_rank
                "has_datetime": len(datetime_cols) > 0
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
                      window_size: int = 10,
                      **kwargs) -> OperationResult:
        """Оконные функции в Polars."""
        try:
            # Определяем колонки
            if lazy:
                numeric_cols = [col for col in df.columns if 'num' in col or 'mixed_num' in col]
                datetime_cols = [col for col in df.columns if 'datetime' in col]
            else:
                numeric_cols = self.get_numeric_columns(df)
                datetime_cols = self.get_datetime_columns(df)
            
            if not numeric_cols:
                return OperationResult(
                    success=False,
                    error=ValueError("Нет числовых колонок для оконных функций")
                )
            
            value_column = numeric_cols[0]
            
            # Создаем выражения для оконных функций
            window_exprs = [
                pl.col(value_column).rolling_mean(window_size=window_size)
                    .alias(f'{value_column}_rolling_mean'),
                pl.col(value_column).rolling_std(window_size=window_size)
                    .alias(f'{value_column}_rolling_std'),
                pl.col(value_column).cum_mean()
                    .alias(f'{value_column}_expanding_mean'),
                pl.col(value_column).rank()
                    .alias(f'{value_column}_rank'),
                pl.col(value_column).rank() / pl.col(value_column).count()
                    .alias(f'{value_column}_pct_rank')
            ]
            
            # Применяем оконные функции
            if datetime_cols and not lazy:
                # Сортируем по datetime если есть
                result_df = df.sort(datetime_cols[0]).with_columns(window_exprs)
            else:
                # Без сортировки
                result_df = df.with_columns(window_exprs)
            
            metadata = {
                "value_column": value_column,
                "window_size": window_size,
                "lazy": lazy,
                "new_columns": 5,
                "has_datetime": len(datetime_cols) > 0
            }
            
            if not lazy:
                metadata["rows"] = len(result_df)
            
            return OperationResult(
                success=True,
                result=result_df,
                metadata=metadata
            )
            
        except Exception as e:
            return OperationResult(success=False, error=e)


# Регистрируем все операции группировки
register_operation(SingleColumnGroupByOperation())
register_operation(MultiColumnGroupByOperation())
register_operation(MultiAggregationOperation())
register_operation(WindowFunctionOperation())
