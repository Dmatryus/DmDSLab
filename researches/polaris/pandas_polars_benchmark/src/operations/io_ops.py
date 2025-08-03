"""
Операции ввода/вывода данных для бенчмаркинга.
"""

import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, Union

import pandas as pd
import polars as pl

from researches.polaris.pandas_polars_benchmark.src.operations.base import (
    Operation, OperationResult, register_operation
)
from researches.polaris.pandas_polars_benchmark.src.data import DataLoader, DataSaver


class ReadCSVOperation(Operation):
    """Операция чтения CSV файла."""
    
    def __init__(self):
        super().__init__(
            name="read_csv",
            category="io",
            description="Чтение CSV файла в DataFrame"
        )
        self.loader = DataLoader()
    
    def execute_pandas(self, 
                      df: pd.DataFrame,  # Игнорируется для операций чтения
                      backend: str = "numpy",
                      dataset_name: str = "",
                      **kwargs) -> OperationResult:
        """Чтение CSV в Pandas DataFrame."""
        try:
            if not dataset_name:
                return OperationResult(
                    success=False,
                    error=ValueError("dataset_name обязателен для операции чтения")
                )
            
            result_df = self.loader.load_pandas_csv(dataset_name, backend)
            
            metadata = {
                "rows": len(result_df),
                "columns": len(result_df.columns),
                "backend": backend,
                "memory_usage_mb": result_df.memory_usage(deep=True).sum() / (1024 * 1024)
            }
            
            return OperationResult(
                success=True,
                result=result_df,
                metadata=metadata
            )
            
        except Exception as e:
            return OperationResult(success=False, error=e)
    
    def execute_polars(self,
                      df: Union[pl.DataFrame, pl.LazyFrame],  # Игнорируется
                      lazy: bool = False,
                      dataset_name: str = "",
                      **kwargs) -> OperationResult:
        """Чтение CSV в Polars DataFrame."""
        try:
            if not dataset_name:
                return OperationResult(
                    success=False,
                    error=ValueError("dataset_name обязателен для операции чтения")
                )
            
            result_df = self.loader.load_polars_csv(dataset_name, lazy)
            
            metadata = {
                "lazy": lazy,
                "format": "csv"
            }
            
            # Для eager mode добавляем статистику
            if not lazy:
                metadata.update({
                    "rows": len(result_df),
                    "columns": len(result_df.columns),
                    "estimated_size_mb": result_df.estimated_size() / (1024 * 1024)
                })
            
            return OperationResult(
                success=True,
                result=result_df,
                metadata=metadata
            )
            
        except Exception as e:
            return OperationResult(success=False, error=e)


class ReadParquetOperation(Operation):
    """Операция чтения Parquet файла."""
    
    def __init__(self):
        super().__init__(
            name="read_parquet",
            category="io",
            description="Чтение Parquet файла в DataFrame"
        )
        self.loader = DataLoader()
    
    def execute_pandas(self,
                      df: pd.DataFrame,
                      backend: str = "numpy",
                      dataset_name: str = "",
                      **kwargs) -> OperationResult:
        """Чтение Parquet в Pandas DataFrame."""
        try:
            if not dataset_name:
                return OperationResult(
                    success=False,
                    error=ValueError("dataset_name обязателен для операции чтения")
                )
            
            result_df = self.loader.load_pandas_parquet(dataset_name, backend)
            
            metadata = {
                "rows": len(result_df),
                "columns": len(result_df.columns),
                "backend": backend,
                "memory_usage_mb": result_df.memory_usage(deep=True).sum() / (1024 * 1024)
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
                      dataset_name: str = "",
                      **kwargs) -> OperationResult:
        """Чтение Parquet в Polars DataFrame."""
        try:
            if not dataset_name:
                return OperationResult(
                    success=False,
                    error=ValueError("dataset_name обязателен для операции чтения")
                )
            
            result_df = self.loader.load_polars_parquet(dataset_name, lazy)
            
            metadata = {
                "lazy": lazy,
                "format": "parquet"
            }
            
            if not lazy:
                metadata.update({
                    "rows": len(result_df),
                    "columns": len(result_df.columns),
                    "estimated_size_mb": result_df.estimated_size() / (1024 * 1024)
                })
            
            return OperationResult(
                success=True,
                result=result_df,
                metadata=metadata
            )
            
        except Exception as e:
            return OperationResult(success=False, error=e)


class WriteCSVOperation(Operation):
    """Операция записи в CSV файл."""
    
    def __init__(self):
        super().__init__(
            name="write_csv",
            category="io",
            description="Запись DataFrame в CSV файл"
        )
    
    def execute_pandas(self,
                      df: pd.DataFrame,
                      backend: str = "numpy",
                      **kwargs) -> OperationResult:
        """Запись Pandas DataFrame в CSV."""
        try:
            with DataSaver() as saver:
                result = saver.save_pandas_csv(
                    df, 
                    f"test_write_{id(df)}",
                    **kwargs
                )
                
                metadata = {
                    "file_size_mb": result["file_size_mb"],
                    "save_time_sec": result["save_time_sec"],
                    "rows_written": result["rows"],
                    "columns_written": result["columns"]
                }
                
                return OperationResult(
                    success=True,
                    result=None,  # Операция записи не возвращает DataFrame
                    metadata=metadata
                )
                
        except Exception as e:
            return OperationResult(success=False, error=e)
    
    def execute_polars(self,
                      df: Union[pl.DataFrame, pl.LazyFrame],
                      lazy: bool = False,
                      **kwargs) -> OperationResult:
        """Запись Polars DataFrame в CSV."""
        try:
            with DataSaver() as saver:
                result = saver.save_polars_csv(
                    df,
                    f"test_write_{id(df)}",
                    **kwargs
                )
                
                metadata = {
                    "file_size_mb": result["file_size_mb"],
                    "save_time_sec": result["save_time_sec"],
                    "lazy_mode": result.get("lazy_mode", False)
                }
                
                if result.get("rows") is not None:
                    metadata["rows_written"] = result["rows"]
                    metadata["columns_written"] = result["columns"]
                
                return OperationResult(
                    success=True,
                    result=None,
                    metadata=metadata
                )
                
        except Exception as e:
            return OperationResult(success=False, error=e)


class WriteParquetOperation(Operation):
    """Операция записи в Parquet файл."""
    
    def __init__(self):
        super().__init__(
            name="write_parquet",
            category="io",
            description="Запись DataFrame в Parquet файл"
        )
    
    def execute_pandas(self,
                      df: pd.DataFrame,
                      backend: str = "numpy",
                      compression: str = "snappy",
                      **kwargs) -> OperationResult:
        """Запись Pandas DataFrame в Parquet."""
        try:
            with DataSaver() as saver:
                result = saver.save_pandas_parquet(
                    df,
                    f"test_write_{id(df)}",
                    compression=compression,
                    **kwargs
                )
                
                metadata = {
                    "file_size_mb": result["file_size_mb"],
                    "save_time_sec": result["save_time_sec"],
                    "rows_written": result["rows"],
                    "columns_written": result["columns"],
                    "compression": compression
                }
                
                return OperationResult(
                    success=True,
                    result=None,
                    metadata=metadata
                )
                
        except Exception as e:
            return OperationResult(success=False, error=e)
    
    def execute_polars(self,
                      df: Union[pl.DataFrame, pl.LazyFrame],
                      lazy: bool = False,
                      compression: str = "snappy",
                      **kwargs) -> OperationResult:
        """Запись Polars DataFrame в Parquet."""
        try:
            with DataSaver() as saver:
                result = saver.save_polars_parquet(
                    df,
                    f"test_write_{id(df)}",
                    compression=compression,
                    **kwargs
                )
                
                metadata = {
                    "file_size_mb": result["file_size_mb"],
                    "save_time_sec": result["save_time_sec"],
                    "lazy_mode": result.get("lazy_mode", False),
                    "compression": compression
                }
                
                if result.get("rows") is not None:
                    metadata["rows_written"] = result["rows"]
                    metadata["columns_written"] = result["columns"]
                
                return OperationResult(
                    success=True,
                    result=None,
                    metadata=metadata
                )
                
        except Exception as e:
            return OperationResult(success=False, error=e)


# Регистрируем все IO операции
register_operation(ReadCSVOperation())
register_operation(ReadParquetOperation())
register_operation(WriteCSVOperation())
register_operation(WriteParquetOperation())


# Функции-обертки для удобного использования
def create_io_operations() -> Dict[str, Operation]:
    """
    Создает и возвращает все IO операции.
    
    Returns:
        Dict[str, Operation]: Словарь имя -> операция
    """
    operations = {
        "read_csv": ReadCSVOperation(),
        "read_parquet": ReadParquetOperation(), 
        "write_csv": WriteCSVOperation(),
        "write_parquet": WriteParquetOperation()
    }
    
    return operations


def profile_io_operation(operation_name: str,
                        df: Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame],
                        library: str,
                        backend: Optional[str] = None,
                        **kwargs) -> OperationResult:
    """
    Профилирует IO операцию.
    
    Args:
        operation_name: Имя операции
        df: DataFrame
        library: Библиотека
        backend: Backend для pandas
        **kwargs: Дополнительные параметры
        
    Returns:
        OperationResult: Результат выполнения
    """
    operations = create_io_operations()
    
    if operation_name not in operations:
        return OperationResult(
            success=False,
            error=ValueError(f"Неизвестная IO операция: {operation_name}")
        )
    
    operation = operations[operation_name]
    return operation.execute(df, library, backend, **kwargs)
