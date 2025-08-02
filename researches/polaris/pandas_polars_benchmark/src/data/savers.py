"""
Модуль для сохранения данных в различных форматах для операций записи.
"""

from pathlib import Path
from typing import Union, Optional, Dict, Any
import time
import tempfile

import pandas as pd
import polars as pl

from researches.polaris.pandas_polars_benchmark.src.utils.logging import get_logger


class DataSaver:
    """Класс для сохранения данных в различных форматах."""

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Инициализация сохранителя данных.

        Args:
            output_dir: Директория для сохранения (если None, используется временная)
        """
        if output_dir:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.use_temp = False
        else:
            # Используем временную директорию для операций записи
            self.temp_dir = tempfile.TemporaryDirectory()
            self.output_dir = Path(self.temp_dir.name)
            self.use_temp = True

        self.logger = get_logger("data_saver")

    def save_pandas_csv(
        self, df: pd.DataFrame, filename: str, **kwargs
    ) -> Dict[str, Any]:
        """
        Сохраняет Pandas DataFrame в CSV.

        Args:
            df: DataFrame для сохранения
            filename: Имя файла (без расширения)
            **kwargs: Дополнительные параметры для to_csv

        Returns:
            Dict с информацией о сохранении
        """
        output_path = self.output_dir / f"{filename}.csv"

        self.logger.debug(f"Сохранение Pandas DataFrame в CSV: {output_path}")

        start_time = time.time()
        df.to_csv(output_path, index=False, **kwargs)
        save_time = time.time() - start_time

        file_size_mb = output_path.stat().st_size / (1024 * 1024)

        result = {
            "path": output_path,
            "format": "csv",
            "library": "pandas",
            "save_time_sec": save_time,
            "file_size_mb": file_size_mb,
            "rows": len(df),
            "columns": len(df.columns),
        }

        self.logger.debug(f"CSV сохранен: {file_size_mb:.2f} MB за {save_time:.3f}с")

        return result

    def save_pandas_parquet(
        self, df: pd.DataFrame, filename: str, **kwargs
    ) -> Dict[str, Any]:
        """
        Сохраняет Pandas DataFrame в Parquet.

        Args:
            df: DataFrame для сохранения
            filename: Имя файла (без расширения)
            **kwargs: Дополнительные параметры для to_parquet

        Returns:
            Dict с информацией о сохранении
        """
        output_path = self.output_dir / f"{filename}.parquet"

        self.logger.debug(f"Сохранение Pandas DataFrame в Parquet: {output_path}")

        start_time = time.time()
        df.to_parquet(output_path, index=False, **kwargs)
        save_time = time.time() - start_time

        file_size_mb = output_path.stat().st_size / (1024 * 1024)

        result = {
            "path": output_path,
            "format": "parquet",
            "library": "pandas",
            "save_time_sec": save_time,
            "file_size_mb": file_size_mb,
            "rows": len(df),
            "columns": len(df.columns),
        }

        self.logger.debug(
            f"Parquet сохранен: {file_size_mb:.2f} MB за {save_time:.3f}с"
        )

        return result

    def save_polars_csv(
        self, df: Union[pl.DataFrame, pl.LazyFrame], filename: str, **kwargs
    ) -> Dict[str, Any]:
        """
        Сохраняет Polars DataFrame в CSV.

        Args:
            df: DataFrame для сохранения
            filename: Имя файла (без расширения)
            **kwargs: Дополнительные параметры для write_csv

        Returns:
            Dict с информацией о сохранении
        """
        output_path = self.output_dir / f"{filename}.csv"

        self.logger.debug(f"Сохранение Polars DataFrame в CSV: {output_path}")

        # Если это LazyFrame, нужно сначала собрать
        if isinstance(df, pl.LazyFrame):
            self.logger.debug("Сбор LazyFrame перед сохранением")
            start_collect = time.time()
            df = df.collect()
            collect_time = time.time() - start_collect
            self.logger.debug(f"LazyFrame собран за {collect_time:.3f}с")
        else:
            collect_time = 0

        start_time = time.time()
        df.write_csv(output_path, **kwargs)
        save_time = time.time() - start_time

        file_size_mb = output_path.stat().st_size / (1024 * 1024)

        result = {
            "path": output_path,
            "format": "csv",
            "library": "polars",
            "save_time_sec": save_time,
            "collect_time_sec": collect_time,
            "total_time_sec": save_time + collect_time,
            "file_size_mb": file_size_mb,
            "rows": len(df),
            "columns": len(df.columns),
        }

        self.logger.debug(
            f"CSV сохранен: {file_size_mb:.2f} MB за {save_time:.3f}с "
            f"(total: {save_time + collect_time:.3f}с)"
        )

        return result

    def save_polars_parquet(
        self, df: Union[pl.DataFrame, pl.LazyFrame], filename: str, **kwargs
    ) -> Dict[str, Any]:
        """
        Сохраняет Polars DataFrame в Parquet.

        Args:
            df: DataFrame для сохранения
            filename: Имя файла (без расширения)
            **kwargs: Дополнительные параметры для write_parquet

        Returns:
            Dict с информацией о сохранении
        """
        output_path = self.output_dir / f"{filename}.parquet"

        self.logger.debug(f"Сохранение Polars DataFrame в Parquet: {output_path}")

        # Если это LazyFrame, используем sink_parquet для эффективности
        if isinstance(df, pl.LazyFrame):
            start_time = time.time()
            df.sink_parquet(output_path, **kwargs)
            save_time = time.time() - start_time

            # Для LazyFrame не можем получить количество строк без сбора
            rows = None
            columns = None
        else:
            start_time = time.time()
            df.write_parquet(output_path, **kwargs)
            save_time = time.time() - start_time

            rows = len(df)
            columns = len(df.columns)

        file_size_mb = output_path.stat().st_size / (1024 * 1024)

        result = {
            "path": output_path,
            "format": "parquet",
            "library": "polars",
            "save_time_sec": save_time,
            "file_size_mb": file_size_mb,
            "rows": rows,
            "columns": columns,
            "lazy_mode": isinstance(df, pl.LazyFrame),
        }

        self.logger.debug(
            f"Parquet сохранен: {file_size_mb:.2f} MB за {save_time:.3f}с"
        )

        return result

    def save_dataframe(
        self,
        df: Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame],
        filename: str,
        format: str = "csv",
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Универсальный метод сохранения DataFrame.

        Args:
            df: DataFrame для сохранения
            filename: Имя файла (без расширения)
            format: Формат файла ('csv' или 'parquet')
            **kwargs: Дополнительные параметры для сохранения

        Returns:
            Dict с информацией о сохранении
        """
        # Определяем библиотеку
        if isinstance(df, pd.DataFrame):
            library = "pandas"
        elif isinstance(df, (pl.DataFrame, pl.LazyFrame)):
            library = "polars"
        else:
            raise TypeError(f"Неподдерживаемый тип DataFrame: {type(df)}")

        # Выбираем метод сохранения
        if library == "pandas":
            if format == "csv":
                return self.save_pandas_csv(df, filename, **kwargs)
            elif format == "parquet":
                return self.save_pandas_parquet(df, filename, **kwargs)
            else:
                raise ValueError(f"Неподдерживаемый формат для pandas: {format}")

        elif library == "polars":
            if format == "csv":
                return self.save_polars_csv(df, filename, **kwargs)
            elif format == "parquet":
                return self.save_polars_parquet(df, filename, **kwargs)
            else:
                raise ValueError(f"Неподдерживаемый формат для polars: {format}")

    def cleanup(self) -> None:
        """Очистка временных файлов."""
        if self.use_temp and hasattr(self, "temp_dir"):
            self.logger.debug("Очистка временной директории")
            self.temp_dir.cleanup()

    def __enter__(self):
        """Вход в контекстный менеджер."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Выход из контекстного менеджера с очисткой."""
        self.cleanup()
