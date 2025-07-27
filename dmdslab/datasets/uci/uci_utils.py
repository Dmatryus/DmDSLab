"""
Утилиты и вспомогательные функции для UCI Dataset Loader.

Этот модуль содержит различные утилиты, включая:
- Progress bar функции для отображения прогресса загрузки
- Функции логирования и настройки логгера
- Вспомогательные функции для форматирования и валидации
"""

import logging
import functools
import time
from pathlib import Path
from typing import Optional, Union, Dict, Any, List, Callable, TypeVar, Tuple
from contextlib import contextmanager
from datetime import datetime

try:
    from tqdm.auto import tqdm
except ImportError:
    from tqdm import tqdm

import numpy as np
import pandas as pd

from .types import DatasetID, LogLevel, TaskType, Domain, MetadataDict


# Type variable для декораторов
F = TypeVar('F', bound=Callable[..., Any])


# ============================================================================
# Логирование
# ============================================================================

def setup_logger(
    name: str = "uci_loader",
    level: Union[str, LogLevel] = LogLevel.INFO,
    log_file: Optional[Path] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """Настройка логгера для UCI Dataset Loader.
    
    Args:
        name: Имя логгера
        level: Уровень логирования (строка или LogLevel enum)
        log_file: Путь к файлу для записи логов (опционально)
        format_string: Формат сообщений (опционально)
        
    Returns:
        Настроенный объект логгера
    """
    # Преобразуем уровень логирования
    if isinstance(level, LogLevel):
        numeric_level = level.numeric_level
    else:
        numeric_level = getattr(logging, level.upper())
    
    # Создаем или получаем логгер
    logger = logging.getLogger(name)
    logger.setLevel(numeric_level)
    
    # Удаляем существующие обработчики
    logger.handlers = []
    
    # Формат по умолчанию
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    formatter = logging.Formatter(format_string)
    
    # Консольный обработчик
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Файловый обработчик (если указан)
    if log_file is not None:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def log_execution_time(logger: Optional[logging.Logger] = None) -> Callable[[F], F]:
    """Декоратор для логирования времени выполнения функции.
    
    Args:
        logger: Логгер для записи (если None, создается новый)
        
    Returns:
        Декорированная функция
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = logging.getLogger(func.__module__)
            
            start_time = time.time()
            logger.debug(f"Начало выполнения {func.__name__}")
            
            try:
                result = func(*args, **kwargs)
                elapsed_time = time.time() - start_time
                logger.debug(
                    f"Завершено {func.__name__} за {elapsed_time:.2f} сек"
                )
                return result
            except Exception as e:
                elapsed_time = time.time() - start_time
                logger.error(
                    f"Ошибка в {func.__name__} после {elapsed_time:.2f} сек: {e}"
                )
                raise
        
        return wrapper
    return decorator


# ============================================================================
# Progress Bar
# ============================================================================

def create_progress_bar(
    total: Optional[int] = None,
    desc: str = "Прогресс",
    unit: str = "it",
    leave: bool = True,
    disable: bool = False,
    **kwargs
) -> tqdm:
    """Создание progress bar с настройками по умолчанию.
    
    Args:
        total: Общее количество итераций
        desc: Описание прогресса
        unit: Единица измерения
        leave: Оставить progress bar после завершения
        disable: Отключить progress bar
        **kwargs: Дополнительные параметры для tqdm
        
    Returns:
        Объект tqdm progress bar
    """
    default_kwargs = {
        'total': total,
        'desc': desc,
        'unit': unit,
        'leave': leave,
        'disable': disable,
        'ncols': 100,
        'bar_format': '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
    }
    default_kwargs.update(kwargs)
    
    return tqdm(**default_kwargs)


@contextmanager
def progress_context(
    desc: str = "Обработка",
    total: Optional[int] = None,
    disable: bool = False
):
    """Контекстный менеджер для progress bar.
    
    Args:
        desc: Описание процесса
        total: Общее количество шагов
        disable: Отключить отображение
        
    Yields:
        Объект progress bar
    """
    pbar = create_progress_bar(total=total, desc=desc, disable=disable)
    try:
        yield pbar
    finally:
        pbar.close()


def download_with_progress(
    download_func: Callable,
    total_size: Optional[int] = None,
    desc: str = "Загрузка",
    chunk_size: int = 8192
) -> Any:
    """Обертка для функции загрузки с отображением прогресса.
    
    Args:
        download_func: Функция загрузки
        total_size: Общий размер в байтах (если известен)
        desc: Описание загрузки
        chunk_size: Размер чанка для обновления
        
    Returns:
        Результат функции загрузки
    """
    with progress_context(desc=desc, total=total_size, unit='B', unit_scale=True) as pbar:
        def update_callback(chunk_size: int):
            pbar.update(chunk_size)
        
        return download_func(progress_callback=update_callback)


# ============================================================================
# Валидация
# ============================================================================

def validate_dataset_id(dataset_id: Any) -> DatasetID:
    """Валидация ID датасета.
    
    Args:
        dataset_id: ID датасета для проверки
        
    Returns:
        Валидный ID датасета
        
    Raises:
        ValueError: Если ID невалидный
    """
    if isinstance(dataset_id, (int, str)):
        if isinstance(dataset_id, str):
            # Проверяем, что строка не пустая
            if not dataset_id.strip():
                raise ValueError("ID датасета не может быть пустой строкой")
            # Пытаемся преобразовать в int, если это число
            try:
                return int(dataset_id)
            except ValueError:
                return dataset_id
        elif isinstance(dataset_id, int):
            if dataset_id <= 0:
                raise ValueError("ID датасета должен быть положительным числом")
            return dataset_id
    
    raise ValueError(
        f"ID датасета должен быть int или str, получен {type(dataset_id).__name__}"
    )


# ============================================================================
# Форматирование
# ============================================================================

def format_dataset_info(
    dataset_id: DatasetID,
    name: str,
    task_type: TaskType,
    n_instances: int,
    n_features: int,
    domain: Optional[Domain] = None,
    has_missing: bool = False,
    cached: Optional[bool] = False,
    cache_size: Optional[float] = None
) -> str:
    """Форматирование информации о датасете для вывода.
    
    Args:
        dataset_id: ID датасета
        name: Название датасета
        task_type: Тип задачи
        n_instances: Количество примеров
        n_features: Количество признаков
        domain: Домен датасета
        has_missing: Есть ли пропущенные значения
        cached: Находится ли в кеше (None если кеш отключен)
        cache_size: Размер в кеше (в МБ)
        
    Returns:
        Отформатированная строка с информацией
    """
    lines = [
        f"{'='*60}",
        f"Датасет: {name} (ID: {dataset_id})",
        f"{'='*60}",
        f"Тип задачи: {task_type.value}",
        f"Размерность: {n_instances} × {n_features}",
    ]
    
    if domain:
        lines.append(f"Домен: {domain.value}")
    
    if has_missing:
        lines.append("⚠️  Содержит пропущенные значения")
    
    if cached:
        cache_info = "✅ В кеше"
        if cache_size is not None:
            cache_info += f" ({cache_size:.2f} МБ)"
        lines.append(cache_info)
    elif cached is None:
        lines.append("⚪ Кеш отключен")
    else:
        lines.append("❌ Не в кеше")
    
    lines.append(f"{'='*60}")
    
    return "\n".join(lines)


def print_dataset_summary(
    features: Union[np.ndarray, pd.DataFrame],
    target: Optional[Union[np.ndarray, pd.Series]] = None,
    feature_names: Optional[List[str]] = None,
    categorical_indices: Optional[List[int]] = None,
    max_features: int = 10
) -> None:
    """Вывод краткой сводки по датасету.
    
    Args:
        features: Матрица признаков
        target: Целевая переменная
        feature_names: Имена признаков
        categorical_indices: Индексы категориальных признаков
        max_features: Максимальное количество признаков для вывода
    """
    print("\n📊 Сводка по датасету:")
    print("-" * 40)
    
    # Информация о признаках
    if isinstance(features, pd.DataFrame):
        n_samples, n_features = features.shape
        print(f"Тип данных: pandas.DataFrame")
    else:
        n_samples, n_features = features.shape if features.ndim > 1 else (features.shape[0], 1)
        print(f"Тип данных: numpy.ndarray")
    
    print(f"Размерность: {n_samples} × {n_features}")
    
    # Имена признаков
    if feature_names:
        print(f"\nПризнаки ({min(max_features, len(feature_names))} из {len(feature_names)}):")
        for i, name in enumerate(feature_names[:max_features]):
            cat_marker = " [категориальный]" if categorical_indices and i in categorical_indices else ""
            print(f"  {i+1}. {name}{cat_marker}")
        if len(feature_names) > max_features:
            print(f"  ... и еще {len(feature_names) - max_features} признаков")
    
    # Информация о целевой переменной
    if target is not None:
        print(f"\nЦелевая переменная:")
        if isinstance(target, (pd.Series, np.ndarray)):
            unique_values = np.unique(target)
            if len(unique_values) <= 10:
                print(f"  Уникальные значения: {unique_values}")
            else:
                print(f"  Количество уникальных значений: {len(unique_values)}")
                print(f"  Диапазон: [{np.min(target)}, {np.max(target)}]")
    
    print("-" * 40)


def format_cache_size(size_bytes: int) -> str:
    """Форматирование размера в байтах в читаемый вид.
    
    Args:
        size_bytes: Размер в байтах
        
    Returns:
        Отформатированная строка
    """
    for unit in ['Б', 'КБ', 'МБ', 'ГБ', 'ТБ']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} ПБ"


def estimate_download_size(
    n_instances: int,
    n_features: int,
    dtype_size: int = 8
) -> Tuple[int, str]:
    """Оценка размера датасета для загрузки.
    
    Args:
        n_instances: Количество примеров
        n_features: Количество признаков
        dtype_size: Размер одного элемента в байтах
        
    Returns:
        Кортеж (размер в байтах, отформатированная строка)
    """
    # Оценка: матрица признаков + вектор целей + метаданные
    estimated_bytes = (
        n_instances * n_features * dtype_size +  # Признаки
        n_instances * dtype_size +                # Целевая переменная
        n_features * 100                          # Метаданные (имена и т.д.)
    )
    
    # Добавляем 20% на накладные расходы
    estimated_bytes = int(estimated_bytes * 1.2)
    
    return estimated_bytes, format_cache_size(estimated_bytes)


def get_timestamp() -> str:
    """Получение текущей временной метки в читаемом формате.
    
    Returns:
        Строка с временной меткой
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def safe_dict_update(
    base_dict: Dict[str, Any],
    update_dict: Dict[str, Any],
    overwrite: bool = False
) -> Dict[str, Any]:
    """Безопасное обновление словаря.
    
    Args:
        base_dict: Базовый словарь
        update_dict: Словарь с обновлениями
        overwrite: Перезаписывать существующие ключи
        
    Returns:
        Обновленный словарь
    """
    result = base_dict.copy()
    
    for key, value in update_dict.items():
        if overwrite or key not in result:
            result[key] = value
    
    return result


def get_popular_datasets() -> List[Dict[str, Any]]:
    """Получение списка популярных датасетов UCI.
    
    Returns:
        Список словарей с информацией о популярных датасетах
    """
    return [
        {
            'id': 17,
            'name': 'Breast Cancer Wisconsin (Diagnostic)',
            'task_type': 'classification',
            'instances': 569,
            'features': 30
        },
        {
            'id': 53,
            'name': 'Iris',
            'task_type': 'classification',
            'instances': 150,
            'features': 4
        },
        {
            'id': 14,
            'name': 'Breast Cancer',
            'task_type': 'classification',
            'instances': 286,
            'features': 9
        },
        {
            'id': 15,
            'name': 'Car Evaluation',
            'task_type': 'classification',
            'instances': 1728,
            'features': 6
        },
        {
            'id': 19,
            'name': 'Wine',
            'task_type': 'classification',
            'instances': 178,
            'features': 13
        },
        {
            'id': 45,
            'name': 'Heart Disease',
            'task_type': 'classification',
            'instances': 303,
            'features': 13
        },
        {
            'id': 80,
            'name': 'Optical Recognition of Handwritten Digits',
            'task_type': 'classification',
            'instances': 5620,
            'features': 64
        },
        {
            'id': 697,
            'name': 'Dry Bean Dataset',
            'task_type': 'classification',
            'instances': 13611,
            'features': 16
        }
    ]


def create_download_report(
    dataset_ids: List[DatasetID],
    results: Dict[DatasetID, Union[str, Exception]],
    start_time: float,
    cache_dir: Path
) -> str:
    """Создание отчета о загрузке датасетов.
    
    Args:
        dataset_ids: Список ID датасетов
        results: Результаты загрузки (успех или исключение)
        start_time: Время начала загрузки
        cache_dir: Директория кеша
        
    Returns:
        Отформатированный отчет
    """
    elapsed_time = time.time() - start_time
    successful = sum(1 for r in results.values() if not isinstance(r, Exception))
    failed = len(results) - successful
    
    lines = [
        "\n" + "="*60,
        "📋 ОТЧЕТ О ЗАГРУЗКЕ ДАТАСЕТОВ",
        "="*60,
        f"Время начала: {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}",
        f"Время выполнения: {elapsed_time:.2f} сек",
        f"Директория кеша: {cache_dir}",
        "",
        f"Всего датасетов: {len(dataset_ids)}",
        f"✅ Успешно загружено: {successful}",
        f"❌ Ошибки загрузки: {failed}",
        "",
        "ДЕТАЛИ:",
        "-"*60
    ]
    
    # Детали по каждому датасету
    for dataset_id in dataset_ids:
        result = results.get(dataset_id)
        if isinstance(result, Exception):
            status = f"❌ Ошибка: {type(result).__name__}: {str(result)}"
        else:
            status = f"✅ Успешно загружен"
        lines.append(f"ID {dataset_id}: {status}")
    
    lines.extend([
        "-"*60,
        f"Размер кеша после загрузки: {_get_cache_size(cache_dir)}",
        "="*60
    ])
    
    return "\n".join(lines)


def _get_cache_size(cache_dir: Path) -> str:
    """Получение размера директории кеша.
    
    Args:
        cache_dir: Путь к директории кеша
        
    Returns:
        Отформатированный размер
    """
    # Специальный случай для отключенного кеша
    if cache_dir.name == "no_cache":
        return "Кеш отключен"
    
    if not cache_dir.exists():
        return "0 Б"
    
    total_size = sum(
        f.stat().st_size 
        for f in cache_dir.rglob('*') 
        if f.is_file()
    )
    
    return format_cache_size(total_size)