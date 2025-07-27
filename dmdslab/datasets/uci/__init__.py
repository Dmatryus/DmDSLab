"""
UCI Dataset Loader для DmDSLab.

Модуль для загрузки и управления датасетами из репозитория UCI Machine Learning.

Основные компоненты:
    - UCIDatasetManager: Главный класс для загрузки датасетов
    - CacheManager: Управление кешированием
    - DatasetInfo: Информация о датасете
    - Исключения: UCIDatasetError и производные

Примеры использования:

    # Базовое использование
    >>> from dmdslab.datasets.uci import UCIDatasetManager
    >>> manager = UCIDatasetManager()
    >>> iris = manager.load_dataset(53)  # Загрузка Iris dataset

    # Загрузка без кеша
    >>> manager = UCIDatasetManager(use_cache=False)
    >>> data = manager.load_dataset(17)

    # Загрузка нескольких датасетов
    >>> datasets = manager.load_datasets([53, 17, 45])

    # Работа с кешем напрямую
    >>> from dmdslab.datasets.uci import CacheManager
    >>> cache = CacheManager("~/.uci_datasets")
    >>> cache.get_statistics()

Поддерживаемые параметры:
    - cache_dir: Директория для кеша
    - use_cache: Использовать ли кеширование
    - log_level: Уровень логирования (DEBUG, INFO, WARNING, ERROR)
    - show_progress: Показывать ли progress bar
    - raise_on_missing: Вызывать ли исключение при отсутствии датасета
"""

# Версия модуля
__version__ = "1.0.0"

# Импорт основных классов
from dmdslab.datasets.uci.uci_manager import UCIDatasetManager
from dmdslab.datasets.uci.uci_cache import (
    CacheManager,
    safe_pickle_dump,
    safe_pickle_load,
)
from dmdslab.datasets.uci.uci_metadata import DatasetInfo, MetadataExtractor

# Импорт типов
from dmdslab.datasets.uci.uci_types import (
    TaskType,
    Domain,
    CacheStatus,
    LogLevel,
    DatasetID,
    FeatureMatrix,
    TargetVector,
    CategoricalIndices,
    FeatureNames,
    MetadataDict,
)

# Импорт исключений
from dmdslab.datasets.uci.uci_exceptions import (
    UCIDatasetError,
    CacheError,
    DatasetNotFoundError,
    NetworkError,
    DataFormatError,
    ValidationError,
    ConfigurationError,
)

# Импорт утилит
from dmdslab.datasets.uci.uci_utils import (
    setup_logger,
    validate_dataset_id,
    format_dataset_info,
    print_dataset_summary,
    create_progress_bar,
    get_popular_datasets,
    format_cache_size,
)

# Определяем публичный API модуля
__all__ = [
    # Основные классы
    "UCIDatasetManager",
    "CacheManager",
    "DatasetInfo",
    "MetadataExtractor",
    # Типы
    "TaskType",
    "Domain",
    "CacheStatus",
    "LogLevel",
    "DatasetID",
    # Исключения
    "UCIDatasetError",
    "CacheError",
    "DatasetNotFoundError",
    "NetworkError",
    "DataFormatError",
    "ValidationError",
    "ConfigurationError",
    # Утилиты
    "setup_logger",
    "validate_dataset_id",
    "format_dataset_info",
    "print_dataset_summary",
    "get_popular_datasets",
    "format_cache_size",
    # Функции кеша
    "safe_pickle_dump",
    "safe_pickle_load",
]


# Удобные алиасы для обратной совместимости
UCILoader = UCIDatasetManager  # Альтернативное имя


def load_dataset(
    dataset_id: DatasetID,
    cache_dir: str = None,
    use_cache: bool = True,
    force_reload: bool = False,
    show_progress: bool = True,
) -> "ModelData":
    """Быстрая функция для загрузки одного датасета.

    Удобная обертка для одноразовой загрузки без создания менеджера.

    Args:
        dataset_id: ID датасета UCI
        cache_dir: Директория кеша (по умолчанию ~/.uci_datasets)
        use_cache: Использовать кеширование
        force_reload: Принудительная перезагрузка
        show_progress: Показывать progress bar

    Returns:
        Объект ModelData с загруженными данными

    Examples:
        >>> from dmdslab.datasets.uci import load_dataset
        >>> iris = load_dataset(53)
    """
    manager = UCIDatasetManager(
        cache_dir=cache_dir,
        use_cache=use_cache,
        show_progress=show_progress,
        log_level="WARNING",  # Минимум логов для быстрой функции
    )
    return manager.load_dataset(dataset_id, force_reload=force_reload)


def load_datasets(
    dataset_ids: list,
    cache_dir: str = None,
    use_cache: bool = True,
    force_reload: bool = False,
    show_progress: bool = True,
    stop_on_error: bool = False,
) -> list:
    """Быстрая функция для загрузки нескольких датасетов.

    Args:
        dataset_ids: Список ID датасетов UCI
        cache_dir: Директория кеша
        use_cache: Использовать кеширование
        force_reload: Принудительная перезагрузка
        show_progress: Показывать progress bar
        stop_on_error: Остановиться при первой ошибке

    Returns:
        Список объектов ModelData или исключений

    Examples:
        >>> from dmdslab.datasets.uci import load_datasets
        >>> datasets = load_datasets([53, 17, 45])
    """
    manager = UCIDatasetManager(
        cache_dir=cache_dir,
        use_cache=use_cache,
        show_progress=show_progress,
        log_level="INFO",
    )
    return manager.load_datasets(
        dataset_ids, force_reload=force_reload, stop_on_error=stop_on_error
    )


def clear_cache(dataset_id: DatasetID = None, cache_dir: str = None) -> bool:
    """Очистка кеша датасетов.

    Args:
        dataset_id: ID датасета для очистки (None - очистить весь кеш)
        cache_dir: Директория кеша

    Returns:
        True если успешно очищено
    """
    manager = UCIDatasetManager(cache_dir=cache_dir, log_level="INFO")
    return manager.clear_cache(dataset_id)


def get_cache_info(cache_dir: str = None) -> dict:
    """Получение информации о кеше.

    Args:
        cache_dir: Директория кеша

    Returns:
        Словарь с информацией о кеше
    """
    manager = UCIDatasetManager(cache_dir=cache_dir, log_level="WARNING")
    return manager.get_cache_info()


# Словарь с популярными датасетами для быстрого доступа
POPULAR_DATASETS = {
    "iris": 53,
    "wine": 19,
    "breast_cancer": 17,
    "heart_disease": 45,
    "adult": 2,
    "car_evaluation": 15,
    "mushroom": 73,
    "abalone": 1,
    "digits": 80,
    "glass": 42,
    "hepatitis": 46,
    "letter": 59,
    "dry_bean": 697,
}


def load_by_name(name: str, **kwargs) -> "ModelData":
    """Загрузка популярного датасета по имени.

    Args:
        name: Имя датасета (например, 'iris', 'wine')
        **kwargs: Дополнительные параметры для load_dataset

    Returns:
        Объект ModelData

    Raises:
        ValueError: Если имя не найдено в списке популярных

    Examples:
        >>> from dmdslab.datasets.uci import load_by_name
        >>> iris = load_by_name('iris')
    """
    name_lower = name.lower().replace(" ", "_").replace("-", "_")

    if name_lower not in POPULAR_DATASETS:
        available = ", ".join(sorted(POPULAR_DATASETS.keys()))
        raise ValueError(
            f"Датасет '{name}' не найден в списке популярных. "
            f"Доступные: {available}"
        )

    dataset_id = POPULAR_DATASETS[name_lower]
    return load_dataset(dataset_id, **kwargs)


# Информация о модуле
def _get_module_info() -> dict:
    """Получение информации о модуле."""
    return {
        "version": __version__,
        "author": "DmDSLab Team",
        "description": "UCI Machine Learning Repository Dataset Loader",
        "supported_formats": ["numpy", "pandas"],
        "cache_format": "pickle",
        "popular_datasets": list(POPULAR_DATASETS.keys()),
        "total_popular": len(POPULAR_DATASETS),
    }


# При импорте модуля можно вывести краткую информацию
if __name__ == "__main__":
    import pprint

    print("UCI Dataset Loader Module")
    print("=" * 50)
    pprint.pprint(_get_module_info())
