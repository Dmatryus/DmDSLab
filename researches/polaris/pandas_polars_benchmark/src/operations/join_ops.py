"""
Операции соединения данных (join) для бенчмаркинга.
"""

from typing import Optional, Union, List, Dict, Any, Tuple
import pandas as pd
import polars as pl
import numpy as np

from operations.base import (
    Operation, OperationResult, register_operation
)


class JoinOperationBase(Operation):
    """Базовый класс для операций соединения."""
    
    def _prepare_join_data(self, df: Union[pd.DataFrame, pl.DataFrame], 
                          is_polars: bool = False) -> Tuple[Any, Any]:
        """
        Подготавливает два датафрейма для соединения из одного.
        
        Returns:
            Tuple[left_df, right_df]
        """
        n_rows = len(df)
        split_point = n_rows // 2
        
        if is_polars:
            # Для Polars
            # Создаем левую часть - первая половина + некоторые изменения
            left_df = df.head(split_point + split_point // 2)
            
            # Создаем правую часть - вторая половина + некоторые дубликаты
            right_df = df.tail(split_point + split_point // 4)
            
            # Добавляем суффиксы к некоторым колонкам чтобы различать их после join
            numeric_cols = [col for col in df.columns if df[col].dtype in [pl.Int32, pl.Int64, pl.Float32, pl.Float64]]
            
            # Переименовываем некоторые колонки в правом датафрейме
            rename_dict = {}
            for i, col in enumerate(numeric_cols[1:], 1):  # Пропускаем первую колонку (ключ)
                if i <= 2:  # Переименовываем только первые 2 колонки
                    rename_dict[col] = f"{col}_right"
            
            if rename_dict:
                right_df = right_df.rename(rename_dict)
                
        else:
            # Для Pandas
            # Левая часть
            left_df = df.iloc[:split_point + split_point // 2].copy()
            
            # Правая часть
            right_df = df.iloc[n_rows - split_point - split_point // 4:].copy()
            
            # Переименовываем некоторые колонки
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            rename_dict = {}
            for i, col in enumerate(numeric_cols[1:], 1):
                if i <= 2:
                    rename_dict[col] = f"{col}_right"
            
            if rename_dict:
                right_df = right_df.rename(columns=rename_dict)
        
        return left_df, right_df
    
    def _select_join_keys(self, df: Union[pd.DataFrame, pl.DataFrame],
                         is_polars: bool = False) -> List[str]:
        """Выбирает подходящие колонки для использования в качестве ключей соединения."""
        if is_polars:
            # Для Polars
            # Приоритет: строковые колонки с умеренной кардинальностью
            string_cols = [col for col in df.columns if df[col].dtype == pl.Utf8]
            int_cols = [col for col in df.columns if df[col].dtype in [pl.Int32, pl.Int64]]
            
            # Берем первую строковую колонку если есть
            if string_cols:
                return [string_cols[0]]
            # Иначе первую целочисленную
            elif int_cols:
                return [int_cols[0]]
            else:
                return [df.columns[0]]
        else:
            # Для Pandas
            string_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
            int_cols = df.select_dtypes(include=['int32', 'int64']).columns.tolist()
            
            if string_cols:
                return [string_cols[0]]
            elif int_cols:
                return [int_cols[0]]
            else:
                return [df.columns[0]]


class InnerJoinOperation(JoinOperationBase):
    """Внутреннее соединение (inner join)."""
    
    def __init__(self):
        super().__init__(
            name="inner_join",
            category="join",
            description="Внутреннее соединение двух датафреймов"
        )
    
    def execute_pandas(self,
                      df: pd.DataFrame,
                      backend: str = "numpy",
                      join_keys: Optional[List[str]] = None,
                      **kwargs) -> OperationResult:
        """Inner join в Pandas."""
        try:
            # Подготавливаем данные для join
            left_df, right_df = self._prepare_join_data(df, is_polars=False)
            
            # Выбираем ключи соединения
            if join_keys is None:
                join_keys = self._select_join_keys(left_df, is_polars=False)
            
            # Считаем статистику до join
            left_key_unique = left_df[join_keys[0]].nunique()
            right_key_unique = right_df[join_keys[0]].nunique()
            
            # Выполняем inner join
            result_df = pd.merge(
                left_df,
                right_df,
                on=join_keys,
                how='inner',
                suffixes=('_left', '_right')
            )
            
            # Считаем статистику результата
            result_key_unique = result_df[join_keys[0]].nunique()
            join_ratio = len(result_df) / (len(left_df) + len(right_df))
            
            metadata = {
                "join_type": "inner",
                "join_keys": join_keys,
                "left_rows": len(left_df),
                "right_rows": len(right_df),
                "result_rows": len(result_df),
                "left_key_unique": int(left_key_unique),
                "right_key_unique": int(right_key_unique),
                "result_key_unique": int(result_key_unique),
                "join_ratio": join_ratio,
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
                      join_keys: Optional[List[str]] = None,
                      **kwargs) -> OperationResult:
        """Inner join в Polars."""
        try:
            # Подготавливаем данные
            left_df, right_df = self._prepare_join_data(df, is_polars=True)
            
            # Выбираем ключи
            if join_keys is None:
                join_keys = self._select_join_keys(left_df, is_polars=True)
            
            # Для lazy mode конвертируем в lazy
            if lazy:
                if not isinstance(left_df, pl.LazyFrame):
                    left_df = left_df.lazy()
                if not isinstance(right_df, pl.LazyFrame):
                    right_df = right_df.lazy()
            
            # Статистика для eager mode
            stats = {}
            if not lazy:
                stats["left_key_unique"] = left_df[join_keys[0]].n_unique()
                stats["right_key_unique"] = right_df[join_keys[0]].n_unique()
            
            # Выполняем join
            result_df = left_df.join(
                right_df,
                on=join_keys,
                how='inner',
                suffix='_right'
            )
            
            metadata = {
                "join_type": "inner",
                "join_keys": join_keys,
                "lazy": lazy
            }
            
            if not lazy:
                metadata.update({
                    "left_rows": len(left_df),
                    "right_rows": len(right_df),
                    "result_rows": len(result_df),
                    "result_key_unique": result_df[join_keys[0]].n_unique(),
                    **stats
                })
            
            return OperationResult(
                success=True,
                result=result_df,
                metadata=metadata
            )
            
        except Exception as e:
            return OperationResult(success=False, error=e)


class LeftJoinOperation(JoinOperationBase):
    """Левое соединение (left join)."""
    
    def __init__(self):
        super().__init__(
            name="left_join",
            category="join",
            description="Левое соединение двух датафреймов"
        )
    
    def execute_pandas(self,
                      df: pd.DataFrame,
                      backend: str = "numpy",
                      join_keys: Optional[List[str]] = None,
                      **kwargs) -> OperationResult:
        """Left join в Pandas."""
        try:
            # Подготавливаем данные
            left_df, right_df = self._prepare_join_data(df, is_polars=False)
            
            # Модифицируем правый датафрейм - удаляем некоторые строки
            # чтобы продемонстрировать особенности left join
            right_df = right_df.iloc[::2]  # Берем каждую вторую строку
            
            # Выбираем ключи
            if join_keys is None:
                join_keys = self._select_join_keys(left_df, is_polars=False)
            
            # Статистика до join
            left_keys_set = set(left_df[join_keys[0]].dropna())
            right_keys_set = set(right_df[join_keys[0]].dropna())
            matched_keys = left_keys_set.intersection(right_keys_set)
            
            # Выполняем left join
            result_df = pd.merge(
                left_df,
                right_df,
                on=join_keys,
                how='left',
                suffixes=('', '_right'),
                indicator=True  # Добавляем индикатор для анализа
            )
            
            # Анализ результатов join
            merge_stats = result_df['_merge'].value_counts().to_dict()
            null_count = result_df.iloc[:, -2].isna().sum()  # Null в последней колонке из right
            
            # Убираем служебную колонку _merge
            result_df = result_df.drop('_merge', axis=1)
            
            metadata = {
                "join_type": "left",
                "join_keys": join_keys,
                "left_rows": len(left_df),
                "right_rows": len(right_df),
                "result_rows": len(result_df),
                "matched_keys_count": len(matched_keys),
                "left_only_rows": merge_stats.get('left_only', 0),
                "both_rows": merge_stats.get('both', 0),
                "null_count_from_right": int(null_count),
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
                      join_keys: Optional[List[str]] = None,
                      **kwargs) -> OperationResult:
        """Left join в Polars."""
        try:
            # Подготавливаем данные
            left_df, right_df = self._prepare_join_data(df, is_polars=True)
            
            # Модифицируем правый датафрейм
            right_df = right_df.filter(pl.arange(0, len(right_df)) % 2 == 0)
            
            # Выбираем ключи
            if join_keys is None:
                join_keys = self._select_join_keys(left_df, is_polars=True)
            
            # Для lazy mode
            if lazy:
                if not isinstance(left_df, pl.LazyFrame):
                    left_df = left_df.lazy()
                if not isinstance(right_df, pl.LazyFrame):
                    right_df = right_df.lazy()
            
            # Выполняем join
            result_df = left_df.join(
                right_df,
                on=join_keys,
                how='left',
                suffix='_right'
            )
            
            metadata = {
                "join_type": "left",
                "join_keys": join_keys,
                "lazy": lazy
            }
            
            if not lazy:
                # Анализ результатов
                # Находим колонку из right датафрейма
                right_cols = [col for col in result_df.columns if col.endswith('_right')]
                if right_cols:
                    null_count = result_df[right_cols[0]].null_count()
                else:
                    null_count = 0
                
                metadata.update({
                    "left_rows": len(left_df),
                    "right_rows": len(right_df),
                    "result_rows": len(result_df),
                    "null_count_from_right": null_count
                })
            
            return OperationResult(
                success=True,
                result=result_df,
                metadata=metadata
            )
            
        except Exception as e:
            return OperationResult(success=False, error=e)


class MultiKeyJoinOperation(JoinOperationBase):
    """Соединение по нескольким ключам."""
    
    def __init__(self):
        super().__init__(
            name="merge_multiple_keys",
            category="join",
            description="Соединение датафреймов по нескольким ключам"
        )
    
    def _select_multiple_join_keys(self, df: Union[pd.DataFrame, pl.DataFrame],
                                  is_polars: bool = False) -> List[str]:
        """Выбирает несколько колонок для join."""
        keys = []
        
        if is_polars:
            # Строковые колонки
            string_cols = [col for col in df.columns if df[col].dtype == pl.Utf8]
            if string_cols:
                keys.extend(string_cols[:1])
            
            # Целочисленные колонки
            int_cols = [col for col in df.columns if df[col].dtype in [pl.Int32, pl.Int64]]
            for col in int_cols:
                if col not in keys:
                    keys.append(col)
                    if len(keys) >= 2:
                        break
        else:
            # Pandas
            string_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
            if string_cols:
                keys.extend(string_cols[:1])
            
            int_cols = df.select_dtypes(include=['int32', 'int64']).columns.tolist()
            for col in int_cols:
                if col not in keys:
                    keys.append(col)
                    if len(keys) >= 2:
                        break
        
        # Если не хватает ключей, добавляем любые колонки
        if len(keys) < 2:
            for col in df.columns:
                if col not in keys:
                    keys.append(col)
                    if len(keys) >= 2:
                        break
        
        return keys[:2]  # Максимум 2 ключа
    
    def execute_pandas(self,
                      df: pd.DataFrame,
                      backend: str = "numpy",
                      join_keys: Optional[List[str]] = None,
                      **kwargs) -> OperationResult:
        """Multi-key join в Pandas."""
        try:
            # Подготавливаем данные
            left_df, right_df = self._prepare_join_data(df, is_polars=False)
            
            # Выбираем ключи
            if join_keys is None:
                join_keys = self._select_multiple_join_keys(left_df, is_polars=False)
            
            # Добавляем небольшой шум во второй ключ правого датафрейма
            # чтобы уменьшить количество совпадений
            if len(join_keys) >= 2 and pd.api.types.is_numeric_dtype(right_df[join_keys[1]]):
                right_df[join_keys[1]] = right_df[join_keys[1]] + np.random.choice([-1, 0, 1], len(right_df))
            
            # Считаем уникальные комбинации ключей
            left_key_combinations = len(left_df[join_keys].drop_duplicates())
            right_key_combinations = len(right_df[join_keys].drop_duplicates())
            
            # Выполняем join
            result_df = pd.merge(
                left_df,
                right_df,
                on=join_keys,
                how='inner',
                suffixes=('_left', '_right')
            )
            
            # Анализ результата
            result_key_combinations = len(result_df[join_keys].drop_duplicates())
            selectivity = len(result_df) / (len(left_df) * len(right_df)) if len(left_df) * len(right_df) > 0 else 0
            
            metadata = {
                "join_type": "inner",
                "join_keys": join_keys,
                "join_key_count": len(join_keys),
                "left_rows": len(left_df),
                "right_rows": len(right_df),
                "result_rows": len(result_df),
                "left_key_combinations": int(left_key_combinations),
                "right_key_combinations": int(right_key_combinations),
                "result_key_combinations": int(result_key_combinations),
                "selectivity": selectivity,
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
                      join_keys: Optional[List[str]] = None,
                      **kwargs) -> OperationResult:
        """Multi-key join в Polars."""
        try:
            # Подготавливаем данные
            left_df, right_df = self._prepare_join_data(df, is_polars=True)
            
            # Выбираем ключи
            if join_keys is None:
                join_keys = self._select_multiple_join_keys(left_df, is_polars=True)
            
            # Модифицируем второй ключ
            if len(join_keys) >= 2:
                col_type = right_df[join_keys[1]].dtype
                if col_type in [pl.Int32, pl.Int64, pl.Float32, pl.Float64]:
                    noise = pl.Series(np.random.choice([-1, 0, 1], len(right_df)))
                    right_df = right_df.with_columns(
                        (pl.col(join_keys[1]) + noise).alias(join_keys[1])
                    )
            
            # Для lazy mode
            if lazy:
                if not isinstance(left_df, pl.LazyFrame):
                    left_df = left_df.lazy()
                if not isinstance(right_df, pl.LazyFrame):
                    right_df = right_df.lazy()
            
            # Выполняем join
            result_df = left_df.join(
                right_df,
                on=join_keys,
                how='inner',
                suffix='_right'
            )
            
            metadata = {
                "join_type": "inner",
                "join_keys": join_keys,
                "join_key_count": len(join_keys),
                "lazy": lazy
            }
            
            if not lazy:
                # Считаем статистику
                left_combinations = left_df.select(join_keys).unique().shape[0]
                right_combinations = right_df.select(join_keys).unique().shape[0]
                result_combinations = result_df.select(join_keys).unique().shape[0]
                
                metadata.update({
                    "left_rows": len(left_df),
                    "right_rows": len(right_df),
                    "result_rows": len(result_df),
                    "left_key_combinations": left_combinations,
                    "right_key_combinations": right_combinations,
                    "result_key_combinations": result_combinations
                })
            
            return OperationResult(
                success=True,
                result=result_df,
                metadata=metadata
            )
            
        except Exception as e:
            return OperationResult(success=False, error=e)


class AsofJoinOperation(JoinOperationBase):
    """As-of join (для временных рядов)."""
    
    def __init__(self):
        super().__init__(
            name="asof_join",
            category="join",
            description="As-of join для временных рядов"
        )
    
    def execute_pandas(self,
                      df: pd.DataFrame,
                      backend: str = "numpy",
                      **kwargs) -> OperationResult:
        """Asof join в Pandas."""
        try:
            # Ищем datetime колонку
            datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
            if not datetime_cols:
                return OperationResult(
                    success=False,
                    error=ValueError("Нет datetime колонок для asof join")
                )
            
            time_col = datetime_cols[0]
            
            # Создаем два временных ряда с разными частотами
            n_rows = len(df)
            
            # Левый - более частый
            left_df = df.iloc[::2].copy()  # Каждая вторая строка
            left_df = left_df.sort_values(time_col)
            
            # Правый - более редкий с небольшим сдвигом
            right_df = df.iloc[::5].copy()  # Каждая пятая строка
            # Сдвигаем время на несколько минут
            right_df[time_col] = right_df[time_col] + pd.Timedelta(minutes=30)
            right_df = right_df.sort_values(time_col)
            
            # Выбираем колонку для join (первая числовая)
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            value_col = numeric_cols[0] if numeric_cols else df.columns[1]
            
            # Переименовываем колонки в правом датафрейме
            right_cols_rename = {col: f"{col}_right" for col in right_df.columns if col != time_col}
            right_df = right_df.rename(columns=right_cols_rename)
            
            # Выполняем asof join
            result_df = pd.merge_asof(
                left_df,
                right_df,
                on=time_col,
                direction='backward'  # Ближайшее значение в прошлом
            )
            
            # Считаем статистику
            matched_count = result_df[f"{value_col}_right"].notna().sum()
            
            metadata = {
                "join_type": "asof",
                "time_column": time_col,
                "direction": "backward",
                "left_rows": len(left_df),
                "right_rows": len(right_df),
                "result_rows": len(result_df),
                "matched_rows": int(matched_count),
                "match_rate": matched_count / len(result_df),
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
        """Asof join в Polars."""
        try:
            # Ищем datetime колонку
            datetime_cols = [col for col in df.columns 
                           if df[col].dtype in [pl.Datetime, pl.Date]]
            
            if not datetime_cols:
                return OperationResult(
                    success=False,
                    error=ValueError("Нет datetime колонок для asof join")
                )
            
            time_col = datetime_cols[0]
            
            # Создаем временные ряды
            # Левый
            left_df = df.filter(pl.arange(0, len(df)) % 2 == 0)
            left_df = left_df.sort(time_col)
            
            # Правый со сдвигом
            right_df = df.filter(pl.arange(0, len(df)) % 5 == 0)
            # Добавляем 30 минут
            right_df = right_df.with_columns(
                (pl.col(time_col) + pl.duration(minutes=30)).alias(time_col)
            )
            right_df = right_df.sort(time_col)
            
            # Переименовываем колонки
            for col in right_df.columns:
                if col != time_col:
                    right_df = right_df.rename({col: f"{col}_right"})
            
            # Для lazy mode
            if lazy:
                if not isinstance(left_df, pl.LazyFrame):
                    left_df = left_df.lazy()
                if not isinstance(right_df, pl.LazyFrame):
                    right_df = right_df.lazy()
            
            # Выполняем asof join
            result_df = left_df.join_asof(
                right_df,
                on=time_col,
                strategy='backward'
            )
            
            metadata = {
                "join_type": "asof",
                "time_column": time_col,
                "strategy": "backward",
                "lazy": lazy
            }
            
            if not lazy:
                # Находим первую колонку из правого датафрейма
                right_cols = [col for col in result_df.columns if col.endswith('_right')]
                if right_cols:
                    matched_count = result_df[right_cols[0]].is_not_null().sum()
                else:
                    matched_count = 0
                
                metadata.update({
                    "left_rows": len(left_df),
                    "right_rows": len(right_df),
                    "result_rows": len(result_df),
                    "matched_rows": matched_count,
                    "match_rate": matched_count / len(result_df) if len(result_df) > 0 else 0
                })
            
            return OperationResult(
                success=True,
                result=result_df,
                metadata=metadata
            )
            
        except Exception as e:
            return OperationResult(success=False, error=e)


# Регистрируем все операции соединения
register_operation(InnerJoinOperation())
register_operation(LeftJoinOperation())
register_operation(MultiKeyJoinOperation())
register_operation(AsofJoinOperation())
