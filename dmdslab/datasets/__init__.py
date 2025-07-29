"""
Dataset management utilities for DmDSLab.

Includes:
- UCI Dataset Manager for accessing UCI ML Repository
- Integration with ML data containers
"""

from .ml_data_container import (
    DataInfo,
    DataSplit,
    ModelData,
    create_data_split,
    create_kfold_data,
)

# Импорты из UCI модуля - если доступен
from .uci import POPULAR_DATASETS as UCI_POPULAR_DATASETS
from .uci import (
    CacheManager,
    DatasetInfo,
    Domain,  # Дополнительные классы
    TaskType,
    UCIDatasetManager,
)
from .uci import clear_cache as clear_uci_cache
from .uci import get_cache_info as get_uci_cache_info
from .uci import load_by_name as load_uci_by_name
from .uci import load_dataset as load_uci_dataset
from .uci import load_datasets as load_uci_datasets

_UCI_AVAILABLE = True


__all__ = [
    # ML data container классы
    "ModelData",
    "DataSplit",
    "DataInfo",
    "create_data_split",
    "create_kfold_data",
    # UCI классы и функции
    "UCIDatasetManager",
    "load_uci_dataset",
    "load_uci_datasets",
    "load_uci_by_name",
    "clear_uci_cache",
    "get_uci_cache_info",
    "UCI_POPULAR_DATASETS",
    # Дополнительные классы из UCI
    "DatasetInfo",
    "CacheManager",
    "TaskType",
    "Domain",
]
