"""
Модуль для генерации синтетических данных для бенчмаркинга.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field, asdict
import json
from datetime import datetime, timedelta
import string
import random

from researches.polaris.pandas_polars_benchmark.src.utils.logging import get_logger
from researches.polaris.pandas_polars_benchmark.src.core.config import (
    DataGenerationConfig,
)


@dataclass
class DatasetInfo:
    """Информация о сгенерированном датасете."""

    name: str
    size: int  # Количество строк
    type: str  # 'numeric', 'string', 'datetime', 'mixed'
    columns: List[Dict[str, Any]]  # Информация о каждой колонке
    file_paths: Dict[str, Path] = field(
        default_factory=dict
    )  # {'csv': Path, 'parquet': Path}
    metadata: Dict[str, Any] = field(default_factory=dict)  # Дополнительные метаданные

    # Статистика о датасете
    memory_size_mb: Optional[float] = None
    generation_time_sec: Optional[float] = None
    null_count: Optional[Dict[str, int]] = None
    cardinality: Optional[Dict[str, int]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Конвертирует в словарь для сохранения в JSON."""
        data = asdict(self)
        # Конвертируем Path объекты в строки
        if "file_paths" in data:
            data["file_paths"] = {k: str(v) for k, v in data["file_paths"].items()}
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DatasetInfo:
        """Создает экземпляр из словаря."""
        # Конвертируем строки обратно в Path объекты
        if "file_paths" in data:
            data["file_paths"] = {k: Path(v) for k, v in data["file_paths"].items()}
        return cls(**data)


class DataGenerator:
    """Основной класс для генерации синтетических данных."""

    def __init__(self, config: DataGenerationConfig, output_dir: Optional[Path] = None):
        """
        Инициализация генератора данных.

        Args:
            config: Конфигурация для генерации данных
            output_dir: Директория для сохранения данных
        """
        self.config = config
        self.output_dir = output_dir or Path("data/generated")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Создаем поддиректории для разных форматов
        self.csv_dir = self.output_dir / "csv"
        self.parquet_dir = self.output_dir / "parquet"
        self.metadata_dir = self.output_dir / "metadata"

        for dir_path in [self.csv_dir, self.parquet_dir, self.metadata_dir]:
            dir_path.mkdir(exist_ok=True)

        # Настраиваем генератор случайных чисел для воспроизводимости
        self.random_seed = config.seed
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)

        self.logger = get_logger("data_generator")
        self.generated_datasets: List[DatasetInfo] = []

    def generate_all_datasets(self) -> List[DatasetInfo]:
        """
        Генерирует все датасеты согласно конфигурации.

        Returns:
            List[DatasetInfo]: Список информации о сгенерированных датасетах
        """
        self.logger.phase_start("Генерация данных")

        total_datasets = len(self.config.sizes) * 4  # 4 типа данных
        current = 0

        for size in self.config.sizes:
            self.logger.info(f"Генерация датасетов размера {size:,} строк")

            # Генерируем числовые данные
            current += 1
            self.logger.progress(
                current, total_datasets, f"Числовые данные ({size:,} строк)"
            )
            numeric_info = self._generate_and_save(
                data_func=self.generate_numeric_data, size=size, data_type="numeric"
            )
            self.generated_datasets.append(numeric_info)

            # Генерируем строковые данные
            current += 1
            self.logger.progress(
                current, total_datasets, f"Строковые данные ({size:,} строк)"
            )
            string_info = self._generate_and_save(
                data_func=self.generate_string_data, size=size, data_type="string"
            )
            self.generated_datasets.append(string_info)

            # Генерируем временные ряды
            current += 1
            self.logger.progress(
                current, total_datasets, f"Временные ряды ({size:,} строк)"
            )
            datetime_info = self._generate_and_save(
                data_func=self.generate_datetime_data, size=size, data_type="datetime"
            )
            self.generated_datasets.append(datetime_info)

            # Генерируем смешанные данные
            current += 1
            self.logger.progress(
                current, total_datasets, f"Смешанные данные ({size:,} строк)"
            )
            mixed_info = self._generate_and_save(
                data_func=self.generate_mixed_data, size=size, data_type="mixed"
            )
            self.generated_datasets.append(mixed_info)

        # Сохраняем общую метаинформацию
        self._save_metadata_summary()

        self.logger.phase_end("Генерация данных")
        self.logger.info(
            f"Всего сгенерировано датасетов: {len(self.generated_datasets)}"
        )

        return self.generated_datasets

    def generate_numeric_data(self, size: int) -> pd.DataFrame:
        """
        Генерирует датасет с числовыми данными.

        Args:
            size: Количество строк

        Returns:
            pd.DataFrame: Сгенерированный датасет
        """
        self.logger.debug(
            f"Генерация числовых данных: {size} строк, {self.config.numeric_columns} колонок"
        )

        data = {}
        columns_info = []

        # Генерируем колонки с разными распределениями и типами
        for i in range(self.config.numeric_columns):
            # Выбираем тип данных
            dtype = np.random.choice(self.config.numeric_dtypes)
            # Выбираем распределение
            distribution = np.random.choice(self.config.numeric_distributions)

            # Генерируем данные согласно распределению
            if distribution == "normal":
                values = np.random.normal(loc=100, scale=20, size=size)
            elif distribution == "uniform":
                values = np.random.uniform(low=0, high=1000, size=size)
            elif distribution == "exponential":
                values = np.random.exponential(scale=50, size=size)
            else:
                values = np.random.randn(size) * 100

            # Конвертируем в нужный тип
            if "int" in dtype:
                values = values.astype(dtype)
            else:
                values = values.astype(dtype)

            # Добавляем пропущенные значения
            if self.config.numeric_null_ratio > 0:
                null_mask = np.random.random(size) < self.config.numeric_null_ratio
                values = pd.Series(values)
                values[null_mask] = np.nan
                values = values.to_numpy()

            col_name = f"num_col_{i}"
            data[col_name] = values

            columns_info.append(
                {
                    "name": col_name,
                    "dtype": dtype,
                    "distribution": distribution,
                    "null_ratio": self.config.numeric_null_ratio,
                }
            )

        df = pd.DataFrame(data)
        return df

    def generate_string_data(self, size: int) -> pd.DataFrame:
        """
        Генерирует датасет со строковыми данными.

        Args:
            size: Количество строк

        Returns:
            pd.DataFrame: Сгенерированный датасет
        """
        self.logger.debug(
            f"Генерация строковых данных: {size} строк, {self.config.string_columns} колонок"
        )

        data = {}
        columns_info = []

        for i in range(self.config.string_columns):
            # Выбираем кардинальность для этой колонки
            cardinality = np.random.choice(self.config.string_cardinality)

            # Генерируем уникальные значения
            unique_values = []
            for j in range(cardinality):
                # Генерируем строку случайной длины
                length = np.random.randint(
                    self.config.string_length_range[0],
                    self.config.string_length_range[1] + 1,
                )
                value = "".join(
                    random.choices(string.ascii_letters + string.digits, k=length)
                )
                unique_values.append(f"str_{value}")

            # Генерируем колонку путем случайного выбора из уникальных значений
            values = np.random.choice(unique_values, size=size)

            # Добавляем пропущенные значения
            if self.config.string_null_ratio > 0:
                null_mask = np.random.random(size) < self.config.string_null_ratio
                values = pd.Series(values)
                values[null_mask] = None
                values = values.to_numpy()

            col_name = f"str_col_{i}"
            data[col_name] = values

            columns_info.append(
                {
                    "name": col_name,
                    "dtype": "string",
                    "cardinality": cardinality,
                    "null_ratio": self.config.string_null_ratio,
                }
            )

        df = pd.DataFrame(data)
        return df

    def generate_datetime_data(self, size: int) -> pd.DataFrame:
        """
        Генерирует датасет с временными рядами.

        Args:
            size: Количество строк

        Returns:
            pd.DataFrame: Сгенерированный датасет
        """
        self.logger.debug(
            f"Генерация временных рядов: {size} строк, {self.config.datetime_columns} колонок"
        )

        data = {}
        columns_info = []

        # Парсим начальную дату
        start_date = pd.to_datetime(self.config.datetime_start)

        for i in range(self.config.datetime_columns):
            # Генерируем временной ряд с заданной частотой
            if i == 0:
                # Первая колонка - регулярный временной ряд
                dates = pd.date_range(
                    start=start_date, periods=size, freq=self.config.datetime_frequency
                )
            else:
                # Остальные колонки - со случайными смещениями
                base_dates = pd.date_range(
                    start=start_date, periods=size, freq=self.config.datetime_frequency
                )
                # Добавляем случайный шум
                random_hours = np.random.randint(-24, 24, size=size)
                random_minutes = np.random.randint(-60, 60, size=size)
                dates = (
                    base_dates
                    + pd.to_timedelta(random_hours, unit="h")
                    + pd.to_timedelta(random_minutes, unit="m")
                )

            # Применяем timezone если указан
            if self.config.datetime_tz:
                dates = dates.tz_localize(self.config.datetime_tz)

            col_name = f"datetime_col_{i}"
            data[col_name] = dates

            columns_info.append(
                {
                    "name": col_name,
                    "dtype": "datetime64[ns]",
                    "timezone": self.config.datetime_tz,
                    "frequency": (
                        self.config.datetime_frequency if i == 0 else "irregular"
                    ),
                }
            )

        df = pd.DataFrame(data)
        return df

    def generate_mixed_data(self, size: int) -> pd.DataFrame:
        """
        Генерирует датасет со смешанными типами данных.

        Args:
            size: Количество строк

        Returns:
            pd.DataFrame: Сгенерированный датасет
        """
        self.logger.debug(
            f"Генерация смешанных данных: {size} строк, "
            f"{self.config.mixed_numeric_columns} числовых, "
            f"{self.config.mixed_string_columns} строковых, "
            f"{self.config.mixed_datetime_columns} datetime колонок"
        )

        # Временно изменяем конфигурацию для генерации нужного количества колонок
        original_numeric = self.config.numeric_columns
        original_string = self.config.string_columns
        original_datetime = self.config.datetime_columns

        self.config.numeric_columns = self.config.mixed_numeric_columns
        self.config.string_columns = self.config.mixed_string_columns
        self.config.datetime_columns = self.config.mixed_datetime_columns

        # Генерируем части данных
        dfs = []

        if self.config.mixed_numeric_columns > 0:
            numeric_df = self.generate_numeric_data(size)
            # Переименовываем колонки для mixed
            numeric_df.columns = [f"mixed_{col}" for col in numeric_df.columns]
            dfs.append(numeric_df)

        if self.config.mixed_string_columns > 0:
            string_df = self.generate_string_data(size)
            # Переименовываем колонки для mixed
            string_df.columns = [f"mixed_{col}" for col in string_df.columns]
            dfs.append(string_df)

        if self.config.mixed_datetime_columns > 0:
            datetime_df = self.generate_datetime_data(size)
            # Переименовываем колонки для mixed
            datetime_df.columns = [f"mixed_{col}" for col in datetime_df.columns]
            dfs.append(datetime_df)

        # Восстанавливаем оригинальную конфигурацию
        self.config.numeric_columns = original_numeric
        self.config.string_columns = original_string
        self.config.datetime_columns = original_datetime

        # Объединяем все части
        if dfs:
            df = pd.concat(dfs, axis=1)
        else:
            df = pd.DataFrame(index=range(size))

        return df

    def _generate_and_save(self, data_func, size: int, data_type: str) -> DatasetInfo:
        """
        Генерирует данные и сохраняет в различных форматах.

        Args:
            data_func: Функция генерации данных
            size: Размер датасета
            data_type: Тип данных

        Returns:
            DatasetInfo: Информация о сгенерированном датасете
        """
        import time

        # Засекаем время генерации
        start_time = time.time()

        # Генерируем данные
        df = data_func(size)

        generation_time = time.time() - start_time

        # Создаем имя датасета
        dataset_name = f"{data_type}_{size}"

        # Сохраняем в разных форматах
        file_paths = {}

        # CSV
        csv_path = self.csv_dir / f"{dataset_name}.csv"
        df.to_csv(csv_path, index=False)
        file_paths["csv"] = csv_path
        self.logger.debug(f"Сохранен CSV: {csv_path}")

        # Parquet
        parquet_path = self.parquet_dir / f"{dataset_name}.parquet"
        df.to_parquet(parquet_path, index=False)
        file_paths["parquet"] = parquet_path
        self.logger.debug(f"Сохранен Parquet: {parquet_path}")

        # Собираем информацию о колонках
        columns_info = []
        for col in df.columns:
            col_info = {
                "name": col,
                "dtype": str(df[col].dtype),
                "null_count": int(df[col].isna().sum()),
                "unique_count": int(df[col].nunique()),
            }
            columns_info.append(col_info)

        # Считаем статистику
        memory_size_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        null_counts = {col: int(df[col].isna().sum()) for col in df.columns}
        cardinality = {col: int(df[col].nunique()) for col in df.columns}

        # Создаем информацию о датасете
        dataset_info = DatasetInfo(
            name=dataset_name,
            size=size,
            type=data_type,
            columns=columns_info,
            file_paths=file_paths,
            metadata={
                "generation_seed": self.random_seed,
                "generation_timestamp": datetime.now().isoformat(),
            },
            memory_size_mb=memory_size_mb,
            generation_time_sec=generation_time,
            null_count=null_counts,
            cardinality=cardinality,
        )

        # Сохраняем метаданные
        metadata_path = self.metadata_dir / f"{dataset_name}_info.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(dataset_info.to_dict(), f, indent=2)

        self.logger.info(
            f"Датасет '{dataset_name}' сгенерирован: "
            f"{size:,} строк, {len(df.columns)} колонок, "
            f"{memory_size_mb:.1f} MB, время: {generation_time:.2f}с"
        )

        return dataset_info

    def _save_metadata_summary(self) -> None:
        """Сохраняет сводную информацию о всех сгенерированных датасетах."""
        summary = {
            "generation_timestamp": datetime.now().isoformat(),
            "total_datasets": len(self.generated_datasets),
            "config": {"sizes": self.config.sizes, "seed": self.config.seed},
            "datasets": [ds.to_dict() for ds in self.generated_datasets],
        }

        summary_path = self.metadata_dir / "generation_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        self.logger.info(f"Сводная информация сохранена: {summary_path}")
