"""
Метаданные и извлечение информации для UCI Dataset Loader.

Этот модуль содержит классы и функции для работы с метаданными датасетов,
включая определение типов задач, категориальных признаков и статистики.
"""


import contextlib
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple, Union
import numpy as np
import pandas as pd
from collections import Counter

from dmdslab.datasets.uci.uci_types import (
    TaskType,
    Domain,
    DatasetID,
    FeatureMatrix,
    TargetVector,
    CategoricalIndices,
    FeatureNames,
    MetadataDict,
    NUMERIC_DTYPES,
    CATEGORICAL_DTYPES,
)
from dmdslab.datasets.uci.uci_exceptions import ValidationError, DataFormatError


@dataclass
class DatasetInfo:
    """Информация о датасете UCI.

    Attributes:
        dataset_id: Идентификатор датасета
        name: Название датасета
        description: Описание датасета
        task_type: Тип задачи машинного обучения
        domain: Домен/область применения
        n_instances: Количество примеров
        n_features: Количество признаков
        feature_types: Типы признаков (numeric, categorical, etc.)
        has_missing_values: Наличие пропущенных значений
        additional_info: Дополнительная информация
        cached_at: Время добавления в кеш
        cache_version: Версия формата кеша
    """

    dataset_id: DatasetID
    name: str
    description: str = ""
    task_type: TaskType = TaskType.UNKNOWN
    domain: Domain = Domain.OTHER
    n_instances: int = 0
    n_features: int = 0
    feature_types: List[str] = field(default_factory=list)
    has_missing_values: bool = False
    additional_info: Dict[str, Any] = field(default_factory=dict)
    cached_at: Optional[datetime] = None
    cache_version: str = "1.0"

    @classmethod
    def from_uci_data(
        cls,
        dataset_id: DatasetID,
        uci_data: Any,
        features: Optional[FeatureMatrix] = None,
        target: Optional[TargetVector] = None,
    ) -> "DatasetInfo":
        """Создание DatasetInfo из данных UCI.

        Args:
            dataset_id: ID датасета
            uci_data: Объект данных из ucimlrepo
            features: Матрица признаков (опционально)
            target: Целевая переменная (опционально)

        Returns:
            Экземпляр DatasetInfo
        """
        # Извлекаем базовую информацию
        metadata = getattr(uci_data, "metadata", {})

        # Имя датасета
        name = metadata.get("name", f"Dataset_{dataset_id}")

        # Описание
        description = metadata.get("abstract", "") or metadata.get("description", "")

        # Размерность
        if features is not None:
            n_instances, n_features = (
                features.shape if features.ndim > 1 else (features.shape[0], 1)
            )
        else:
            n_instances = metadata.get("num_instances", 0)
            n_features = metadata.get("num_features", 0)

        # Проверка пропущенных значений
        has_missing = False
        if features is None:
            has_missing = metadata.get("has_missing_values", False)

        elif isinstance(features, pd.DataFrame):
            has_missing = features.isnull().any().any()
        else:
            has_missing = np.isnan(features).any()
        # Определение типа задачи
        task_type_str = metadata.get("task", "Unknown")
        try:
            task_type = TaskType.from_string(task_type_str)
        except ValueError:
            task_type = TaskType.UNKNOWN

        # Определение домена
        area = metadata.get("area", "Other")
        try:
            domain = Domain.from_string(area)
        except ValueError:
            domain = Domain.OTHER

        # Дополнительная информация
        additional_info = {
            "uci_id": metadata.get("uci_id", dataset_id),
            "doi": metadata.get("doi", ""),
            "creators": metadata.get("creators", []),
            "year": metadata.get("year", None),
            "characteristics": metadata.get("characteristics", []),
            "source": metadata.get("source", ""),
        }

        # Удаляем пустые значения
        additional_info = {k: v for k, v in additional_info.items() if v}

        return cls(
            dataset_id=dataset_id,
            name=name,
            description=description,
            task_type=task_type,
            domain=domain,
            n_instances=n_instances,
            n_features=n_features,
            has_missing_values=has_missing,
            additional_info=additional_info,
            cached_at=datetime.now(),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь для сериализации.

        Returns:
            Словарь с данными
        """
        data = asdict(self)

        # Преобразуем enums в строки
        data["task_type"] = self.task_type.value
        data["domain"] = self.domain.value

        # Преобразуем datetime в строку
        if self.cached_at:
            data["cached_at"] = self.cached_at.isoformat()

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetInfo":
        """Создание из словаря после десериализации.

        Args:
            data: Словарь с данными

        Returns:
            Экземпляр DatasetInfo
        """
        # Копируем данные
        data = data.copy()

        # Преобразуем строки обратно в enums
        if "task_type" in data:
            data["task_type"] = TaskType.from_string(data["task_type"])

        if "domain" in data:
            data["domain"] = Domain.from_string(data["domain"])

        # Преобразуем строку обратно в datetime
        if "cached_at" in data and data["cached_at"]:
            data["cached_at"] = datetime.fromisoformat(data["cached_at"])

        return cls(**data)


class MetadataExtractor:
    """Класс для извлечения метаданных из датасетов."""

    def __init__(self, categorical_threshold=0.05, max_unique_for_categorical=20):
        """Инициализация экстрактора метаданных."""
        self.categorical_threshold = (
            categorical_threshold  # 5% от общего числа примеров
        )
        self.max_unique_for_categorical = (
            max_unique_for_categorical  # Максимум уникальных значений
        )

    def extract_from_uci(self, raw_data: Any) -> MetadataDict:
        """Извлечение метаданных из сырых данных UCI.

        Args:
            raw_data: Объект данных из ucimlrepo.fetch()

        Returns:
            Словарь с метаданными
        """
        metadata = {}

        # Базовые метаданные
        if hasattr(raw_data, "metadata"):
            metadata.update(raw_data.metadata)

        # Извлекаем данные
        if hasattr(raw_data, "data"):
            data = raw_data.data

            # Признаки и целевая переменная
            features = data.get("features")
            targets = data.get("targets")

            if features is not None:
                metadata["feature_names"] = (
                    list(features.columns) if hasattr(features, "columns") else None
                )
                metadata["n_features"] = features.shape[1] if features.ndim > 1 else 1
                metadata["n_instances"] = features.shape[0]

            if targets is not None:
                if hasattr(targets, "columns"):
                    metadata["target_names"] = list(targets.columns)
                metadata["n_targets"] = targets.shape[1] if targets.ndim > 1 else 1

        return metadata

    def infer_feature_types(self, features: FeatureMatrix) -> List[str]:
        """Определение типов признаков.

        Args:
            features: Матрица признаков

        Returns:
            Список типов признаков
        """
        feature_types = []

        if isinstance(features, pd.DataFrame):
            for col in features.columns:
                dtype = features[col].dtype
                if pd.api.types.is_numeric_dtype(dtype):
                    # Проверяем, может ли быть категориальным
                    n_unique = features[col].nunique()
                    if n_unique <= self.max_unique_for_categorical:
                        feature_types.append("categorical")
                    else:
                        feature_types.append("numeric")
                elif pd.api.types.is_object_dtype(dtype):
                    feature_types.append("categorical")
                elif pd.api.types.is_bool_dtype(dtype):
                    feature_types.append("binary")
                else:
                    feature_types.append("unknown")
        else:
            # Для numpy arrays
            n_features = features.shape[1] if features.ndim > 1 else 1

            for i in range(n_features):
                col = features[:, i] if features.ndim > 1 else features

                # Проверяем тип данных
                if np.issubdtype(col.dtype, np.number):
                    n_unique = len(np.unique(col[~np.isnan(col)]))
                    if n_unique <= self.max_unique_for_categorical:
                        feature_types.append("categorical")
                    else:
                        feature_types.append("numeric")
                else:
                    feature_types.append("categorical")

        return feature_types

    def detect_categorical(
        self, features: FeatureMatrix, feature_names: Optional[FeatureNames] = None
    ) -> CategoricalIndices:
        """Автоматическое определение категориальных признаков.

        Args:
            features: Матрица признаков
            feature_names: Имена признаков (опционально)

        Returns:
            Список индексов категориальных признаков
        """
        categorical_indices = []

        if isinstance(features, pd.DataFrame):
            categorical_indices.extend(
                i
                for i, col in enumerate(features.columns)
                if self._is_categorical_column(features[col])
            )
        else:
            # Для numpy arrays
            n_features = features.shape[1] if features.ndim > 1 else 1

            for i in range(n_features):
                col = features[:, i] if features.ndim > 1 else features
                if self._is_categorical_array(col):
                    categorical_indices.append(i)

        return categorical_indices

    def _is_categorical_column(self, series: pd.Series) -> bool:
        """Проверка, является ли pandas Series категориальной.

        Args:
            series: Столбец данных

        Returns:
            True если категориальный
        """

        # Если объектный тип
        if pd.api.types.is_object_dtype(series):
            return True

        # Если булевый тип
        if pd.api.types.is_bool_dtype(series):
            return True

        # Для числовых типов проверяем количество уникальных значений
        if pd.api.types.is_numeric_dtype(series):
            n_unique = series.nunique()
            n_total = len(series)

            # Критерии для категориальности:
            # 1. Мало уникальных значений
            # 2. Соотношение уникальных к общему числу
            if n_unique <= self.max_unique_for_categorical:
                return True

            if n_unique / n_total <= self.categorical_threshold:
                # Дополнительная проверка: все ли значения целые
                if series.dtype in ["int64", "int32", "int16", "int8"]:
                    return True

                # Проверяем, являются ли float значения целыми
                if all(series.dropna() == series.dropna().astype(int)):
                    return True

        return False

    def _is_categorical_array(self, array: np.ndarray) -> bool:
        """Проверка, является ли numpy array категориальным.

        Args:
            array: Массив данных

        Returns:
            True если категориальный
        """
        # Для объектных типов
        if array.dtype == np.object_:
            return True

        # Для числовых типов
        if np.issubdtype(array.dtype, np.number):
            # Убираем NaN для подсчета
            clean_array = array[~np.isnan(array)]

            if len(clean_array) == 0:
                return False

            n_unique = len(np.unique(clean_array))
            n_total = len(clean_array)

            # Те же критерии
            if n_unique <= self.max_unique_for_categorical:
                return True

            if n_unique / n_total <= self.categorical_threshold and np.all(clean_array == clean_array.astype(int)):
                return True

        return False

    def calculate_statistics(
        self, features: FeatureMatrix, target: Optional[TargetVector] = None
    ) -> Dict[str, Any]:
        """Вычисление статистики по датасету.

        Args:
            features: Матрица признаков
            target: Целевая переменная (опционально)

        Returns:
            Словарь со статистикой
        """
        stats = {
            "features": (
                {
                    "shape": features.shape,
                    "dtypes": features.dtypes.value_counts().to_dict(),
                    "missing_values": features.isnull().sum().to_dict(),
                    "numeric_features": features.select_dtypes(
                        include=[np.number]
                    ).columns.tolist(),
                    "categorical_features": features.select_dtypes(
                        exclude=[np.number]
                    ).columns.tolist(),
                }
                if isinstance(features, pd.DataFrame)
                else {
                    "shape": features.shape,
                    "dtype": str(features.dtype),
                    "missing_values": int(np.isnan(features).sum()),
                    "min": float(np.nanmin(features)),
                    "max": float(np.nanmax(features)),
                    "mean": float(np.nanmean(features)),
                }
            )
        }

        # Статистика по целевой переменной
        if target is not None:
            if isinstance(target, (pd.Series, pd.DataFrame)):
                unique_values = target.nunique()
                stats["target"] = {
                    "shape": target.shape,
                    "unique_values": int(unique_values),
                    "dtype": str(target.dtype),
                    "distribution": (
                        target.value_counts().to_dict() if unique_values <= 20 else None
                    ),
                }
            else:
                unique_values = len(np.unique(target[~np.isnan(target)]))
                stats["target"] = {
                    "shape": target.shape,
                    "unique_values": unique_values,
                    "dtype": str(target.dtype),
                }

                if unique_values <= 20:
                    values, counts = np.unique(
                        target[~np.isnan(target)], return_counts=True
                    )
                    stats["target"]["distribution"] = dict(
                        zip(values.tolist(), counts.tolist())
                    )

        return stats

    def determine_task_type(
        self,
        features: FeatureMatrix,
        target: Optional[TargetVector] = None,
        metadata: Optional[MetadataDict] = None,
    ) -> TaskType:
        """Определение типа задачи машинного обучения.

        Args:
            features: Матрица признаков
            target: Целевая переменная (опционально)
            metadata: Метаданные датасета (опционально)

        Returns:
            Тип задачи
        """
        # Сначала проверяем метаданные
        if metadata and "task" in metadata:
            with contextlib.suppress(ValueError):
                return TaskType.from_string(metadata["task"])
        # Если нет целевой переменной - кластеризация
        if target is None:
            return TaskType.CLUSTERING

        # Анализируем целевую переменную
        if isinstance(target, (pd.Series, pd.DataFrame)):
            n_unique = target.nunique()
            dtype = target.dtype

            # Бинарная или мультиклассовая классификация
            if n_unique <= 20 or pd.api.types.is_object_dtype(dtype):
                return TaskType.CLASSIFICATION

            # Регрессия для непрерывных значений
            if pd.api.types.is_numeric_dtype(dtype):
                return TaskType.REGRESSION
        else:
            # Для numpy arrays
            unique_values = np.unique(target[~np.isnan(target)])
            n_unique = len(unique_values)

            # Классификация
            if n_unique <= 20:
                return TaskType.CLASSIFICATION

            # Проверяем, являются ли значения целыми
            if np.all(unique_values == unique_values.astype(int)) and n_unique <= 50:
                return TaskType.CLASSIFICATION

            # По умолчанию - регрессия для числовых данных
            if np.issubdtype(target.dtype, np.number):
                return TaskType.REGRESSION

        return TaskType.UNKNOWN
