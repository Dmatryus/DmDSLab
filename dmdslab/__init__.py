"""
DmDSLab: Data Science Laboratory Toolkit

Библиотека инструментов для автоматизации рутинных задач в Data Science проектах.

Основные модули:
- datasets: Работа с датасетами и контейнеры данных
"""

__version__ = "1.0.1"
__author__ = "Dmatryus Detry"
__email__ = "dmatryus.sqrt49@yandex.ru"
__license__ = "Apache-2.0"

# Импорт основных компонентов для удобства использования
from .datasets import (
    DataInfo,
    DataSplit,
    ModelData,
    create_data_split,
    create_kfold_data,
)

# Условный импорт UCI компонентов (требует ucimlrepo)
try:
    from .datasets.uci_dataset_manager import (
        DatasetInfo,
        Domain,
        TaskType,
        UCIDatasetManager,
        print_dataset_summary,
    )

    _has_uci = True
except ImportError:
    _has_uci = False

    # Создаем placeholder для документации
    class _UCINotAvailable:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "UCI functionality requires ucimlrepo package. "
                "Install it with: pip install DmDSLab[uci]"
            )

    UCIDatasetManager = _UCINotAvailable
    DatasetInfo = _UCINotAvailable
    TaskType = None
    Domain = None
    print_dataset_summary = _UCINotAvailable

__all__ = [
    # Версия и метаданные
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    # Основные контейнеры данных
    "ModelData",
    "DataSplit",
    "DataInfo",
    # Функции создания разбиений
    "create_data_split",
    "create_kfold_data",
    # UCI компоненты (если доступны)
    "UCIDatasetManager",
    "DatasetInfo",
    "TaskType",
    "Domain",
    "print_dataset_summary",
]


def get_version():
    """Получить версию библиотеки."""
    return __version__


def has_uci_support():
    """Проверить доступность UCI функциональности."""
    return _has_uci
