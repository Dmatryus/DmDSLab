"""
Модуль datasets для DmDSLab.

Предоставляет функциональность для работы с различными датасетами,
включая загрузку из UCI репозитория и управление данными.

Основные компоненты:
    - ModelData: Контейнер для хранения данных ML моделей
    - UCIDatasetManager: Загрузчик датасетов из UCI репозитория

Подмодули:
    - ml_data_container: Основной контейнер данных ModelData
    - uci: Загрузчик датасетов UCI Machine Learning Repository
"""

# Импорт основного контейнера данных
from dmdslab.datasets.ml_data_container import ModelData

# Импорт UCI загрузчика
from dmdslab.datasets.uci import (
    UCIDatasetManager,
    load_dataset as load_uci_dataset,
    load_datasets as load_uci_datasets,
    load_by_name as load_uci_by_name,
    clear_cache as clear_uci_cache,
    get_cache_info as get_uci_cache_info,
    POPULAR_DATASETS as UCI_POPULAR_DATASETS,
)

# Реэкспорт основных классов для удобства
from dmdslab.datasets.uci import (
    DatasetInfo,
    CacheManager,
    TaskType,
    Domain,
)

# Версия модуля datasets
__version__ = "2.0.0"

# Публичный API
__all__ = [
    # Основные классы
    "ModelData",
    "UCIDatasetManager",
    # Функции быстрой загрузки
    "load_uci_dataset",
    "load_uci_datasets",
    "load_uci_by_name",
    # Управление кешем
    "clear_uci_cache",
    "get_uci_cache_info",
    # Дополнительные классы
    "DatasetInfo",
    "CacheManager",
    "TaskType",
    "Domain",
    # Константы
    "UCI_POPULAR_DATASETS",
]


# Удобные функции-обертки на уровне модуля datasets
def quick_load_iris():
    """Быстрая загрузка датасета Iris.

    Returns:
        ModelData: Датасет Iris

    Example:
        >>> from dmdslab.datasets import quick_load_iris
        >>> iris = quick_load_iris()
    """
    return load_uci_dataset(53)


def quick_load_wine():
    """Быстрая загрузка датасета Wine.

    Returns:
        ModelData: Датасет Wine

    Example:
        >>> from dmdslab.datasets import quick_load_wine
        >>> wine = quick_load_wine()
    """
    return load_uci_dataset(19)


def quick_load_breast_cancer():
    """Быстрая загрузка датасета Breast Cancer Wisconsin.

    Returns:
        ModelData: Датасет Breast Cancer

    Example:
        >>> from dmdslab.datasets import quick_load_breast_cancer
        >>> bc = quick_load_breast_cancer()
    """
    return load_uci_dataset(17)


# Словарь быстрых загрузчиков
QUICK_LOADERS = {
    "iris": quick_load_iris,
    "wine": quick_load_wine,
    "breast_cancer": quick_load_breast_cancer,
}


def list_available_loaders():
    """Список доступных быстрых загрузчиков.

    Returns:
        dict: Словарь с информацией о загрузчиках
    """
    return {
        "quick_loaders": list(QUICK_LOADERS.keys()),
        "uci_popular": list(UCI_POPULAR_DATASETS.keys()),
        "total_quick": len(QUICK_LOADERS),
        "total_uci_popular": len(UCI_POPULAR_DATASETS),
    }


def get_dataset_info():
    """Получение общей информации о модуле datasets.

    Returns:
        dict: Информация о модуле
    """
    return {
        "version": __version__,
        "submodules": ["ml_data_container", "uci"],
        "features": [
            "UCI dataset loading",
            "Automatic caching",
            "Categorical feature detection",
            "Progress bars",
            "ModelData integration",
        ],
        "quick_loaders": len(QUICK_LOADERS),
        "popular_uci_datasets": len(UCI_POPULAR_DATASETS),
    }
