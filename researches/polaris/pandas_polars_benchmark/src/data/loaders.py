"""
Модуль для загрузки датасетов в различных форматах и библиотеках.
"""

from pathlib import Path
from typing import Union, Optional, Dict, Any, List
import json
import time

import pandas as pd
import polars as pl

from researches.polaris.pandas_polars_benchmark.src import get_logger, DatasetInfo


class DataLoader:
    """Класс для загрузки датасетов в различных форматах."""

    def __init__(self, data_dir: Optional[Path] = None):
        """
        Инициализация загрузчика данных.

        Args:
            data_dir: Базовая директория с данными
        """
        self.data_dir = data_dir or Path("data/generated")
        self.metadata_dir = self.data_dir / "metadata"
        self.logger = get_logger("data_loader")

        # Кэш для метаданных
        self._datasets_info: Optional[List[DatasetInfo]] = None

    def load_datasets_info(self) -> List[DatasetInfo]:
        """
        Загружает информацию о всех доступных датасетах.

        Returns:
            List[DatasetInfo]: Список информации о датасетах
        """
        if self._datasets_info is not None:
            return self._datasets_info

        summary_path = self.metadata_dir / "generation_summary.json"

        if not summary_path.exists():
            self.logger.warning(f"Файл сводной информации не найден: {summary_path}")
            # Пытаемся загрузить отдельные файлы метаданных
            self._datasets_info = self._load_individual_metadata()
        else:
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)

            self._datasets_info = [
                DatasetInfo.from_dict(ds_dict) for ds_dict in summary["datasets"]
            ]

            self.logger.info(
                f"Загружена информация о {len(self._datasets_info)} датасетах"
            )

        return self._datasets_info

    def _load_individual_metadata(self) -> List[DatasetInfo]:
        """Загружает метаданные из отдельных файлов."""
        datasets = []

        for metadata_file in self.metadata_dir.glob("*_info.json"):
            if metadata_file.name == "generation_summary.json":
                continue

            try:
                with open(metadata_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                datasets.append(DatasetInfo.from_dict(data))
            except Exception as e:
                self.logger.error(f"Ошибка загрузки метаданных из {metadata_file}: {e}")

        return datasets

    def get_dataset_info(self, name: str) -> Optional[DatasetInfo]:
        """
        Получает информацию о конкретном датасете.

        Args:
            name: Имя датасета

        Returns:
            DatasetInfo или None если не найден
        """
        datasets = self.load_datasets_info()
        for ds in datasets:
            if ds.name == name:
                return ds
        return None

    def load_pandas_csv(
        self, dataset_name: str, backend: str = "numpy"
    ) -> pd.DataFrame:
        """
        Загружает CSV файл в Pandas DataFrame.

        Args:
            dataset_name: Имя датасета
            backend: Backend для pandas ('numpy' или 'pyarrow')

        Returns:
            pd.DataFrame: Загруженный датасет
        """
        dataset_info = self.get_dataset_info(dataset_name)
        if not dataset_info:
            raise ValueError(f"Датасет '{dataset_name}' не найден")

        csv_path = dataset_info.file_paths.get("csv")
        if not csv_path or not csv_path.exists():
            raise FileNotFoundError(f"CSV файл не найден: {csv_path}")

        self.logger.debug(f"Загрузка CSV в Pandas ({backend}): {csv_path}")

        start_time = time.time()

        if backend == "pyarrow":
            # Используем PyArrow backend
            df = pd.read_csv(csv_path, engine="pyarrow", dtype_backend="pyarrow")
        else:
            # Стандартный NumPy backend
            df = pd.read_csv(csv_path)

        load_time = time.time() - start_time
        self.logger.debug(f"CSV загружен за {load_time:.3f}с")

        return df

    def load_pandas_parquet(
        self, dataset_name: str, backend: str = "numpy"
    ) -> pd.DataFrame:
        """
        Загружает Parquet файл в Pandas DataFrame.

        Args:
            dataset_name: Имя датасета
            backend: Backend для pandas ('numpy' или 'pyarrow')

        Returns:
            pd.DataFrame: Загруженный датасет
        """
        dataset_info = self.get_dataset_info(dataset_name)
        if not dataset_info:
            raise ValueError(f"Датасет '{dataset_name}' не найден")

        parquet_path = dataset_info.file_paths.get("parquet")
        if not parquet_path or not parquet_path.exists():
            raise FileNotFoundError(f"Parquet файл не найден: {parquet_path}")

        self.logger.debug(f"Загрузка Parquet в Pandas ({backend}): {parquet_path}")

        start_time = time.time()

        if backend == "pyarrow":
            # Используем PyArrow backend
            df = pd.read_parquet(
                parquet_path, engine="pyarrow", dtype_backend="pyarrow"
            )
        else:
            # Стандартный backend с NumPy типами
            df = pd.read_parquet(parquet_path, engine="pyarrow")

        load_time = time.time() - start_time
        self.logger.debug(f"Parquet загружен за {load_time:.3f}с")

        return df

    def load_polars_csv(
        self, dataset_name: str, lazy: bool = False
    ) -> Union[pl.DataFrame, pl.LazyFrame]:
        """
        Загружает CSV файл в Polars DataFrame.

        Args:
            dataset_name: Имя датасета
            lazy: Использовать lazy loading

        Returns:
            pl.DataFrame или pl.LazyFrame: Загруженный датасет
        """
        dataset_info = self.get_dataset_info(dataset_name)
        if not dataset_info:
            raise ValueError(f"Датасет '{dataset_name}' не найден")

        csv_path = dataset_info.file_paths.get("csv")
        if not csv_path or not csv_path.exists():
            raise FileNotFoundError(f"CSV файл не найден: {csv_path}")

        self.logger.debug(
            f"Загрузка CSV в Polars ({'lazy' if lazy else 'eager'}): {csv_path}"
        )

        start_time = time.time()

        if lazy:
            df = pl.scan_csv(csv_path)
        else:
            df = pl.read_csv(csv_path)

        load_time = time.time() - start_time
        self.logger.debug(f"CSV загружен за {load_time:.3f}с")

        return df

    def load_polars_parquet(
        self, dataset_name: str, lazy: bool = False
    ) -> Union[pl.DataFrame, pl.LazyFrame]:
        """
        Загружает Parquet файл в Polars DataFrame.

        Args:
            dataset_name: Имя датасета
            lazy: Использовать lazy loading

        Returns:
            pl.DataFrame или pl.LazyFrame: Загруженный датасет
        """
        dataset_info = self.get_dataset_info(dataset_name)
        if not dataset_info:
            raise ValueError(f"Датасет '{dataset_name}' не найден")

        parquet_path = dataset_info.file_paths.get("parquet")
        if not parquet_path or not parquet_path.exists():
            raise FileNotFoundError(f"Parquet файл не найден: {parquet_path}")

        self.logger.debug(
            f"Загрузка Parquet в Polars ({'lazy' if lazy else 'eager'}): {parquet_path}"
        )

        start_time = time.time()

        if lazy:
            df = pl.scan_parquet(parquet_path)
        else:
            df = pl.read_parquet(parquet_path)

        load_time = time.time() - start_time
        self.logger.debug(f"Parquet загружен за {load_time:.3f}с")

        return df

    def load_dataset(
        self,
        dataset_name: str,
        library: str,
        format: str = "csv",
        backend: Optional[str] = None,
        lazy: bool = False,
    ) -> Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame]:
        """
        Универсальный метод загрузки датасета.

        Args:
            dataset_name: Имя датасета
            library: Библиотека ('pandas' или 'polars')
            format: Формат файла ('csv' или 'parquet')
            backend: Backend для pandas ('numpy' или 'pyarrow')
            lazy: Lazy loading для polars

        Returns:
            DataFrame соответствующей библиотеки
        """
        if library == "pandas":
            if format == "csv":
                return self.load_pandas_csv(dataset_name, backend or "numpy")
            elif format == "parquet":
                return self.load_pandas_parquet(dataset_name, backend or "numpy")
            else:
                raise ValueError(f"Неподдерживаемый формат: {format}")

        elif library == "polars":
            if format == "csv":
                return self.load_polars_csv(dataset_name, lazy)
            elif format == "parquet":
                return self.load_polars_parquet(dataset_name, lazy)
            else:
                raise ValueError(f"Неподдерживаемый формат: {format}")

        else:
            raise ValueError(f"Неподдерживаемая библиотека: {library}")

    def get_available_datasets(self) -> Dict[str, List[str]]:
        """
        Возвращает список доступных датасетов, сгруппированных по типу.

        Returns:
            Dict[str, List[str]]: Словарь {тип: [имена датасетов]}
        """
        datasets = self.load_datasets_info()

        grouped = {}
        for ds in datasets:
            if ds.type not in grouped:
                grouped[ds.type] = []
            grouped[ds.type].append(ds.name)

        # Сортируем для удобства
        for data_type in grouped:
            grouped[data_type].sort()

        return grouped

    def validate_dataset(self, dataset_name: str) -> bool:
        """
        Проверяет доступность и корректность датасета.

        Args:
            dataset_name: Имя датасета

        Returns:
            bool: True если датасет валиден
        """
        dataset_info = self.get_dataset_info(dataset_name)
        if not dataset_info:
            self.logger.error(f"Метаданные для датасета '{dataset_name}' не найдены")
            return False

        # Проверяем существование файлов
        for format_name, path in dataset_info.file_paths.items():
            if not path.exists():
                self.logger.error(f"Файл {format_name} не найден: {path}")
                return False

            # Проверяем, что файл не пустой
            if path.stat().st_size == 0:
                self.logger.error(f"Файл {format_name} пустой: {path}")
                return False

        self.logger.debug(f"Датасет '{dataset_name}' валиден")
        return True

    def get_dataset_stats(self, dataset_name: str) -> Dict[str, Any]:
        """
        Возвращает статистику о датасете.

        Args:
            dataset_name: Имя датасета

        Returns:
            Dict со статистикой
        """
        dataset_info = self.get_dataset_info(dataset_name)
        if not dataset_info:
            raise ValueError(f"Датасет '{dataset_name}' не найден")

        stats = {
            "name": dataset_info.name,
            "type": dataset_info.type,
            "size": dataset_info.size,
            "columns": len(dataset_info.columns),
            "memory_size_mb": dataset_info.memory_size_mb,
            "generation_time_sec": dataset_info.generation_time_sec,
            "file_sizes": {},
        }

        # Добавляем размеры файлов
        for format_name, path in dataset_info.file_paths.items():
            if path.exists():
                size_mb = path.stat().st_size / (1024 * 1024)
                stats["file_sizes"][format_name] = round(size_mb, 2)

        return stats
