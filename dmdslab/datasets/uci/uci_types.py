"""
Типы данных и перечисления для UCI Dataset Loader.

Этот модуль содержит все определения типов, перечисления и подсказки типов,
используемые в функциональности загрузки датасетов UCI.
"""

from enum import Enum, auto
from typing import TypeVar, Union, List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np
import pandas as pd


class TaskType(Enum):
    """Перечисление типов задач машинного обучения."""
    
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    UNKNOWN = "unknown"
    
    @classmethod
    def from_string(cls, value: str) -> "TaskType":
        """Создать TaskType из строкового значения.
        
        Args:
            value: Строковое представление типа задачи
            
        Returns:
            Значение перечисления TaskType
            
        Raises:
            ValueError: Если значение не является допустимым типом задачи
        """
        value_lower = value.lower().strip()
        for task_type in cls:
            if task_type.value == value_lower:
                return task_type
        raise ValueError(f"Неизвестный тип задачи: {value}")
    
    def __str__(self) -> str:
        return self.value


class Domain(Enum):
    """Перечисление доменов датасетов."""
    
    BUSINESS = "business"
    COMPUTER = "computer"
    ENGINEERING = "engineering"
    GAMES = "games"
    LIFE = "life"
    PHYSICAL = "physical"
    SOCIAL = "social"
    OTHER = "other"
    
    @classmethod
    def from_string(cls, value: str) -> "Domain":
        """Создать Domain из строкового значения.
        
        Args:
            value: Строковое представление домена
            
        Returns:
            Значение перечисления Domain
            
        Raises:
            ValueError: Если значение не является допустимым доменом
        """
        value_lower = value.lower().strip()
        return next(
            (domain for domain in cls if domain.value == value_lower), cls.OTHER
        )
    
    def __str__(self) -> str:
        return self.value


class CacheStatus(Enum):
    """Перечисление статусов кеша."""
    
    HIT = "hit"  # Данные найдены в кеше
    MISS = "miss"  # Данные отсутствуют в кеше
    STALE = "stale"  # Данные в кеше устарели
    CORRUPTED = "corrupted"  # Файл кеша поврежден
    
    def __str__(self) -> str:
        return self.value


class LogLevel(Enum):
    """Перечисление уровней логирования."""
    
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    
    @property
    def numeric_level(self) -> int:
        """Получить числовой уровень логирования для модуля logging Python."""
        import logging
        return getattr(logging, self.value)
    
    def __str__(self) -> str:
        return self.value


# Псевдонимы типов для улучшения читаемости кода
DatasetID = Union[int, str]  # ID датасета UCI может быть int или string
FeatureMatrix = Union[np.ndarray, pd.DataFrame]  # Данные признаков
TargetVector = Union[np.ndarray, pd.Series]  # Целевые данные
CategoricalIndices = List[int]  # Индексы категориальных признаков
FeatureNames = List[str]  # Имена признаков
MetadataDict = Dict[str, Any]  # Словарь метаданных

# Переменные типов для обобщенных типов
T = TypeVar('T')
DataType = TypeVar('DataType', np.ndarray, pd.DataFrame, pd.Series)

# Структурированные типы для сложных данных
CacheEntry = Dict[str, Any]  # Структура: {'data': Any, 'metadata': Dict, 'timestamp': float}
ValidationResult = Tuple[bool, List[str]]  # (is_valid, error_messages)
ProgressCallback = Optional[callable]  # Обратный вызов для обновления прогресса

# Пути к файлам
CachePath = Union[str, Path]
DatasetPath = Union[str, Path]

# Типы структуры датасета
class DatasetStructure:
    """Подсказки типов для компонентов структуры датасета."""
    
    Features = FeatureMatrix
    Target = Optional[TargetVector]
    FeatureNames = Optional[FeatureNames]
    CategoricalIndices = Optional[CategoricalIndices]
    Metadata = MetadataDict


# Константы для проверки типов
NUMERIC_DTYPES = (np.integer, np.floating, np.complexfloating)
CATEGORICAL_DTYPES = (np.object_, np.unicode_, np.string_)
SUPPORTED_PICKLE_PROTOCOLS = (4, 5)  # Поддерживаемые протоколы pickle
DEFAULT_PICKLE_PROTOCOL = 4  # По умолчанию для совместимости