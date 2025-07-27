"""
Модуль кеширования для UCI Dataset Loader.

Этот модуль содержит класс CacheManager для управления кешированием датасетов,
включая сохранение, загрузку, валидацию и управление размером кеша.

Classes:
    CacheManager: Основной класс для управления кешем

Functions:
    safe_pickle_dump: Безопасное сохранение в pickle
    safe_pickle_load: Безопасная загрузка из pickle
"""

import json
import pickle
import shutil
import tempfile
import hashlib
import zipfile
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union
from datetime import datetime
import logging

from dmdslab.datasets.uci.uci_types import (
    DatasetID,
    CacheStatus,
    CachePath,
    DEFAULT_PICKLE_PROTOCOL,
    SUPPORTED_PICKLE_PROTOCOLS,
)
from dmdslab.datasets.uci.uci_exceptions import CacheError
from dmdslab.datasets.uci.uci_utils import format_cache_size, get_timestamp


def safe_pickle_dump(
    obj: Any, file_path: Path, protocol: int = DEFAULT_PICKLE_PROTOCOL
) -> None:
    """Безопасное сохранение объекта в pickle с обработкой ошибок.

    Args:
        obj: Объект для сериализации
        file_path: Путь к файлу
        protocol: Версия протокола pickle

    Raises:
        CacheError: При ошибке сериализации
    """
    temp_path = None
    try:
        # Используем временный файл для атомарности
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=file_path.parent, delete=False
        ) as tmp_file:
            pickle.dump(obj, tmp_file, protocol=protocol)
            temp_path = Path(tmp_file.name)

        # Атомарно перемещаем на место
        temp_path.replace(file_path)

    except Exception as e:
        # Удаляем временный файл при ошибке
        if temp_path and temp_path.exists():
            temp_path.unlink()

        raise CacheError(
            f"Ошибка сохранения в pickle: {e}",
            cache_path=str(file_path),
            details={"error": str(e), "error_type": type(e).__name__},
        )


def safe_pickle_load(file_path: Path) -> Any:
    """Безопасная загрузка объекта из pickle с проверками.

    Args:
        file_path: Путь к файлу

    Returns:
        Десериализованный объект

    Raises:
        CacheError: При ошибке десериализации
    """
    if not file_path.exists():
        raise CacheError(
            f"Файл не найден: {file_path}",
            cache_path=str(file_path),
            cache_status=CacheStatus.MISS,
        )

    try:
        with open(file_path, "rb") as f:
            # Проверяем размер файла
            f.seek(0, 2)  # Перейти в конец
            file_size = f.tell()
            f.seek(0)  # Вернуться в начало

            if file_size == 0:
                raise CacheError(
                    "Файл кеша пуст",
                    cache_path=str(file_path),
                    cache_status=CacheStatus.CORRUPTED,
                )
            return pickle.load(f)

    except pickle.UnpicklingError as e:
        raise CacheError(
            f"Ошибка десериализации pickle: {e}",
            cache_path=str(file_path),
            cache_status=CacheStatus.CORRUPTED,
            details={"error": str(e)},
        )
    except Exception as e:
        raise CacheError(
            f"Неожиданная ошибка при загрузке из pickle: {e}",
            cache_path=str(file_path),
            details={"error": str(e), "error_type": type(e).__name__},
        )


class CacheManager:
    """Менеджер кеша для датасетов UCI.

    Обеспечивает управление кешированными датасетами с поддержкой
    версионирования, валидации и контроля размера.

    Attributes:
        cache_dir: Директория для хранения файлов кеша
        index: Индекс кешированных датасетов
        index_path: Путь к файлу индекса
        protocol_version: Версия протокола pickle
        max_cache_size: Максимальный размер кеша в байтах (None - без ограничений)
        logger: Логгер для записи операций

    Constants:
        CACHE_INDEX_FILE: Имя файла индекса кеша
        CACHE_VERSION: Версия формата кеша
    """

    CACHE_INDEX_FILE = "cache_index.json"
    CACHE_VERSION = "1.0"

    cache_dir: Path
    index: Dict[str, Any]
    index_path: Path
    protocol_version: int
    max_cache_size: Optional[int]
    logger: logging.Logger

    def _check_cache_dir_writable(self) -> bool:
        """Проверка доступности директории кеша для записи.

        Returns:
            True если директория доступна для записи
        """
        try:
            # Пытаемся создать временный файл
            test_file = self.cache_dir / ".test_write_access"
            test_file.touch()
            test_file.unlink()
            return True
        except Exception:
            return False

    def __init__(
        self,
        cache_dir: Union[str, Path],
        protocol_version: int = DEFAULT_PICKLE_PROTOCOL,
        max_cache_size: Optional[int] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """Инициализация менеджера кеша.

        Args:
            cache_dir: Путь к директории кеша
            protocol_version: Версия протокола pickle для сериализации
            max_cache_size: Максимальный размер кеша в байтах
            logger: Логгер (если None, создается новый)
        """
        self.cache_dir = Path(cache_dir)
        self.protocol_version = protocol_version
        self.max_cache_size = max_cache_size

        # Проверка версии протокола
        if protocol_version not in SUPPORTED_PICKLE_PROTOCOLS:
            raise ValueError(
                f"Неподдерживаемая версия протокола pickle: {protocol_version}. "
                f"Поддерживаются: {SUPPORTED_PICKLE_PROTOCOLS}"
            )

        # Настройка логгера
        self.logger = logger or logging.getLogger(__name__)

        # Создаем директорию если не существует
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise CacheError(
                f"Не удалось создать директорию кеша: {e}",
                cache_path=str(cache_dir),
                details={"error": str(e)},
            )

        # Проверяем доступность для записи
        if not self._check_cache_dir_writable():
            raise CacheError(
                f"Директория кеша недоступна для записи: {cache_dir}",
                cache_path=str(cache_dir),
            )

        # Загружаем индекс
        self.index_path = self.cache_dir / self.CACHE_INDEX_FILE
        self.index = self.load_index()

        self.logger.debug(f"CacheManager инициализирован. Директория: {self.cache_dir}")

    def load_index(self) -> Dict[str, Any]:
        """Загрузка индекса кеша из файла.

        Returns:
            Словарь с индексом кеша
        """
        if not self.index_path.exists():
            self.logger.debug("Индекс кеша не найден, создаем новый")
            return {}

        try:
            with open(self.index_path, "r", encoding="utf-8") as f:
                index = json.load(f)

            # Валидация версии
            cache_version = index.get("_version", "0.0")
            if cache_version != self.CACHE_VERSION:
                self.logger.warning(
                    f"Версия кеша ({cache_version}) не совпадает с текущей "
                    f"({self.CACHE_VERSION}). Индекс будет обновлен."
                )

            return index

        except Exception as e:
            self.logger.error(f"Ошибка загрузки индекса кеша: {e}")
            return {}

    def save_index(self, index: Optional[Dict[str, Any]] = None) -> None:
        """Сохранение индекса кеша в файл.

        Args:
            index: Индекс для сохранения (если None, используется self.index)
        """
        if index is None:
            index = self.index

        # Добавляем метаинформацию
        index["_version"] = self.CACHE_VERSION
        index["_updated_at"] = get_timestamp()

        try:
            # Атомарная запись через временный файл
            with tempfile.NamedTemporaryFile(
                mode="w", dir=self.cache_dir, delete=False, encoding="utf-8"
            ) as tmp_file:
                json.dump(index, tmp_file, indent=2, ensure_ascii=False)
                tmp_path = Path(tmp_file.name)

            # Перемещаем временный файл на место индекса
            tmp_path.replace(self.index_path)

            self.logger.debug("Индекс кеша сохранен")

        except Exception as e:
            self.logger.error(f"Ошибка сохранения индекса кеша: {e}")
            # Удаляем временный файл если остался
            if "tmp_path" in locals() and tmp_path.exists():
                tmp_path.unlink()

    def get_cache_path(self, dataset_id: DatasetID) -> Path:
        """Получение пути к файлу кеша для датасета.

        Args:
            dataset_id: ID датасета

        Returns:
            Путь к файлу кеша
        """
        # Создаем уникальное имя файла
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"dataset_{dataset_id}_{timestamp}.pkl"
        return self.cache_dir / filename

    def exists(self, dataset_id: DatasetID) -> bool:
        """Проверка наличия датасета в кеше.

        Args:
            dataset_id: ID датасета

        Returns:
            True если датасет есть в кеше
        """
        str_id = str(dataset_id)

        if str_id not in self.index:
            return False

        # Проверяем физическое наличие файла
        cache_info = self.index[str_id]
        cache_file = self.cache_dir / cache_info.get("filename", "")

        return cache_file.exists()

    def save_dataset(
        self,
        dataset_id: DatasetID,
        data: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Сохранение датасета в кеш.

        Args:
            dataset_id: ID датасета
            data: Данные для сохранения
            metadata: Метаданные датасета

        Raises:
            CacheError: При ошибке сохранения
        """
        str_id = str(dataset_id)

        # Проверяем размер кеша перед сохранением
        if self.max_cache_size is not None:
            self._check_cache_size()

        # Получаем путь для нового файла
        cache_path = self.get_cache_path(dataset_id)

        try:
            # Сохраняем через безопасную функцию
            safe_pickle_dump(data, cache_path, protocol=self.protocol_version)

            # Получаем информацию о файле
            file_size = cache_path.stat().st_size
            file_hash = self._calculate_file_hash(cache_path)

            # Удаляем старую версию если есть
            if str_id in self.index:
                old_file = self.cache_dir / self.index[str_id].get("filename", "")
                if old_file.exists() and old_file != cache_path:
                    old_file.unlink()
                    self.logger.debug(f"Удалена старая версия кеша для {dataset_id}")

            # Обновляем индекс
            self.index[str_id] = {
                "dataset_id": dataset_id,
                "filename": cache_path.name,
                "size_bytes": file_size,
                "size_human": format_cache_size(file_size),
                "hash": file_hash,
                "protocol_version": self.protocol_version,
                "cached_at": get_timestamp(),
                "metadata": metadata or {},
            }

            # Сохраняем индекс
            self.save_index()

            self.logger.info(
                f"Датасет {dataset_id} сохранен в кеш "
                f"({format_cache_size(file_size)})"
            )

        except Exception as e:
            # Удаляем целевой файл если был создан
            if cache_path.exists():
                cache_path.unlink()

            raise CacheError(
                f"Ошибка сохранения датасета {dataset_id} в кеш",
                dataset_id=dataset_id,
                cache_path=str(cache_path),
                details={"error": str(e)},
            )

    def load_dataset(self, dataset_id: DatasetID) -> Tuple[Any, Dict[str, Any]]:
        """Загрузка датасета из кеша.

        Args:
            dataset_id: ID датасета

        Returns:
            Кортеж (данные, метаданные)

        Raises:
            CacheError: При ошибке загрузки или если датасет не найден
        """
        str_id = str(dataset_id)

        if str_id not in self.index:
            raise CacheError(
                f"Датасет {dataset_id} не найден в кеше",
                dataset_id=dataset_id,
                cache_status=CacheStatus.MISS,
            )

        cache_info = self.index[str_id]
        cache_file = self.cache_dir / cache_info["filename"]

        if not cache_file.exists():
            raise CacheError(
                f"Файл кеша для датасета {dataset_id} не найден",
                dataset_id=dataset_id,
                cache_status=CacheStatus.CORRUPTED,
                cache_path=str(cache_file),
            )

        try:
            return self._load_dataset_process(cache_file, cache_info, dataset_id)
        except Exception as e:
            raise CacheError(
                f"Ошибка загрузки датасета {dataset_id} из кеша",
                dataset_id=dataset_id,
                cache_path=str(cache_file),
                details={"error": str(e)},
            )

    def _load_dataset_process(self, cache_file, cache_info, dataset_id):
        # Проверяем хеш файла
        current_hash = self._calculate_file_hash(cache_file)
        if current_hash != cache_info.get("hash"):
            self.logger.warning(
                f"Хеш файла кеша для {dataset_id} не совпадает. "
                "Файл мог быть поврежден."
            )

        # Загружаем данные через безопасную функцию
        data = safe_pickle_load(cache_file)

        # Извлекаем метаданные
        metadata = cache_info.get("metadata", {})

        self.logger.debug(f"Датасет {dataset_id} загружен из кеша")

        return data, metadata

    def invalidate(self, dataset_id: DatasetID) -> None:
        """Инвалидация (удаление) датасета из кеша.

        Args:
            dataset_id: ID датасета для удаления
        """
        str_id = str(dataset_id)

        if str_id not in self.index:
            self.logger.warning(f"Датасет {dataset_id} не найден в кеше")
            return

        cache_info = self.index[str_id]
        cache_file = self.cache_dir / cache_info["filename"]

        # Удаляем файл
        if cache_file.exists():
            try:
                cache_file.unlink()
                self.logger.debug(f"Файл кеша для {dataset_id} удален")
            except Exception as e:
                self.logger.error(f"Ошибка удаления файла кеша: {e}")

        # Удаляем из индекса
        del self.index[str_id]
        self.save_index()

        self.logger.info(f"Датасет {dataset_id} удален из кеша")

    def calculate_cache_size(self) -> int:
        """Вычисление общего размера кеша.

        Returns:
            Размер кеша в байтах
        """
        return sum(
            cache_info.get("size_bytes", 0)
            for str_id, cache_info in self.index.items()
            if not str_id.startswith("_")  # Исключаем метаданные
        )

    def clear_all(self) -> None:
        """Полная очистка кеша."""
        self.logger.warning("Начата полная очистка кеша")

        # Удаляем все файлы кеша
        for str_id, cache_info in list(self.index.items()):
            if not str_id.startswith("_"):  # Исключаем метаданные
                cache_file = self.cache_dir / cache_info.get("filename", "")
                if cache_file.exists():
                    try:
                        cache_file.unlink()
                    except Exception as e:
                        self.logger.error(f"Ошибка удаления {cache_file}: {e}")

        # Очищаем индекс (сохраняем только метаданные)
        self.index = {k: v for k, v in self.index.items() if k.startswith("_")}
        self.save_index()

        self.logger.info("Кеш полностью очищен")

    def get_cached_datasets(self) -> List[DatasetID]:
        """Получение списка ID кешированных датасетов.

        Returns:
            Список ID датасетов в кеше
        """
        datasets = []

        for str_id, cache_info in self.index.items():
            if not str_id.startswith("_"):  # Пропускаем метаданные
                dataset_id = cache_info.get("dataset_id", str_id)
                if self.exists(dataset_id):
                    datasets.append(dataset_id)

        return datasets

    def get_cache_info(self, dataset_id: Optional[DatasetID] = None) -> Dict[str, Any]:
        """Получение информации о кеше.

        Args:
            dataset_id: ID датасета (если None, возвращает общую информацию)

        Returns:
            Словарь с информацией о кеше
        """
        if dataset_id is not None:
            return self._get_dataset_info(dataset_id)

        # Общая информация о кеше
        total_size = self.calculate_cache_size()
        dataset_count = len(self.get_cached_datasets())

        return {
            "cache_directory": str(self.cache_dir),
            "total_datasets": dataset_count,
            "total_size_bytes": total_size,
            "total_size": format_cache_size(total_size),
            "max_cache_size": (
                format_cache_size(self.max_cache_size)
                if self.max_cache_size
                else "Не ограничен"
            ),
            "protocol_version": self.protocol_version,
            "cache_version": self.CACHE_VERSION,
            "index_updated": self.index.get("_updated_at", "Неизвестно"),
        }

    def _get_dataset_info(self, dataset_id):
        # Информация о конкретном датасете
        str_id = str(dataset_id)

        if str_id not in self.index:
            return {"status": "not_found", "dataset_id": dataset_id}

        cache_info = self.index[str_id].copy()
        cache_info["status"] = "cached"

        # Проверяем актуальность
        cache_file = self.cache_dir / cache_info.get("filename", "")
        if not cache_file.exists():
            cache_info["status"] = "corrupted"

        return cache_info

    def cleanup_old_versions(self, keep_latest: int = 1) -> int:
        """Очистка старых версий датасетов.

        Args:
            keep_latest: Количество последних версий для сохранения

        Returns:
            Количество удаленных файлов
        """
        # Группируем файлы по dataset_id
        dataset_files = {}

        for cache_file in self.cache_dir.glob("dataset_*.pkl"):
            # Извлекаем ID из имени файла
            parts = cache_file.stem.split("_")
            if len(parts) >= 3:
                dataset_id = parts[1]
                if dataset_id not in dataset_files:
                    dataset_files[dataset_id] = []
                dataset_files[dataset_id].append(cache_file)

        removed_count = 0

        # Удаляем старые версии
        for files in dataset_files.values():
            if len(files) > keep_latest:
                # Сортируем по времени модификации
                files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

                # Удаляем старые
                for old_file in files[keep_latest:]:
                    try:
                        old_file.unlink()
                        removed_count += 1
                        self.logger.debug(f"Удален старый файл: {old_file.name}")
                    except Exception as e:
                        self.logger.error(f"Ошибка удаления {old_file}: {e}")

        if removed_count > 0:
            self.logger.info(f"Удалено старых версий: {removed_count}")

        return removed_count

    def _check_cache_size(self) -> None:
        """Проверка и управление размером кеша."""
        if self.max_cache_size is None:
            return

        current_size = self.calculate_cache_size()

        if current_size > self.max_cache_size:
            self.logger.warning(
                f"Размер кеша ({format_cache_size(current_size)}) "
                f"превышает лимит ({format_cache_size(self.max_cache_size)})"
            )

            # Собираем только датасеты (исключая метаданные)
            cached_datasets = [
                {
                    "id": str_id,
                    "size": cache_info.get("size_bytes", 0),
                    "cached_at": cache_info.get("cached_at", ""),
                }
                for str_id, cache_info in self.index.items()
                if not str_id.startswith("_")
            ]
            cached_datasets.sort(key=lambda x: x["cached_at"])

            # Удаляем старые датасеты пока не уложимся в лимит
            for dataset in cached_datasets:
                if current_size <= self.max_cache_size * 0.9:  # Оставляем 10% запаса
                    break

                self.invalidate(dataset["id"])
                current_size -= dataset["size"]

                self.logger.info(
                    f"Удален датасет {dataset['id']} для освобождения места"
                )

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Вычисление хеша файла для проверки целостности.

        Args:
            file_path: Путь к файлу

        Returns:
            Хеш файла (MD5)
        """
        hash_md5 = hashlib.md5()

        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)

        return hash_md5.hexdigest()

    def validate_cache(self) -> Dict[str, List[str]]:
        """Валидация всего кеша на целостность.

        Returns:
            Словарь с результатами валидации:
            - 'valid': список валидных датасетов
            - 'corrupted': список поврежденных датасетов
            - 'missing': список отсутствующих файлов
        """
        results = {"valid": [], "corrupted": [], "missing": []}

        for str_id, cache_info in self.index.items():
            if str_id.startswith("_"):  # Пропускаем метаданные
                continue

            dataset_id = cache_info.get("dataset_id", str_id)
            cache_file = self.cache_dir / cache_info.get("filename", "")

            if not cache_file.exists():
                results["missing"].append(dataset_id)
                self.logger.warning(f"Файл кеша для {dataset_id} не найден")
                continue

            # Проверяем хеш
            try:
                current_hash = self._calculate_file_hash(cache_file)
                expected_hash = cache_info.get("hash")

                if current_hash != expected_hash:
                    results["corrupted"].append(dataset_id)
                    self.logger.warning(
                        f"Хеш файла для {dataset_id} не совпадает "
                        f"(ожидается: {expected_hash}, получен: {current_hash})"
                    )
                else:
                    results["valid"].append(dataset_id)

            except Exception as e:
                results["corrupted"].append(dataset_id)
                self.logger.error(f"Ошибка проверки {dataset_id}: {e}")

        self.logger.info(
            f"Результаты валидации кеша: "
            f"валидных: {len(results['valid'])}, "
            f"поврежденных: {len(results['corrupted'])}, "
            f"отсутствующих: {len(results['missing'])}"
        )

        return results

    def repair_cache(self, remove_corrupted: bool = True) -> int:
        """Восстановление кеша путем удаления поврежденных записей.

        Args:
            remove_corrupted: Удалять ли поврежденные файлы

        Returns:
            Количество исправленных записей
        """
        validation_results = self.validate_cache()
        repaired_count = 0

        # Удаляем записи об отсутствующих файлах
        for dataset_id in validation_results["missing"]:
            str_id = str(dataset_id)
            if str_id in self.index:
                del self.index[str_id]
                repaired_count += 1
                self.logger.info(
                    f"Удалена запись об отсутствующем датасете {dataset_id}"
                )

        # Обрабатываем поврежденные файлы
        if remove_corrupted:
            for dataset_id in validation_results["corrupted"]:
                self.invalidate(dataset_id)
                repaired_count += 1
                self.logger.info(f"Удален поврежденный датасет {dataset_id}")

        # Сохраняем обновленный индекс
        if repaired_count > 0:
            self.save_index()
            self.logger.info(f"Кеш восстановлен. Исправлено записей: {repaired_count}")

        return repaired_count

    def _calc_stat(self, stats, datasets):
        # Средний размер
        stats["average_dataset_size"] = (
            stats["total_size_bytes"] / len(datasets) if datasets else 0
        )

        if not datasets:
            return

        # Сортировка по размеру
        datasets_by_size = sorted(datasets, key=lambda x: x["size"])
        stats["smallest_dataset"] = {
            "id": datasets_by_size[0]["id"],
            "size": format_cache_size(datasets_by_size[0]["size"]),
        }
        stats["largest_dataset"] = {
            "id": datasets_by_size[-1]["id"],
            "size": format_cache_size(datasets_by_size[-1]["size"]),
        }

        # Сортировка по времени
        datasets_by_time = sorted(datasets, key=lambda x: x["cached_at"])
        stats["oldest_dataset"] = {
            "id": datasets_by_time[0]["id"],
            "cached_at": datasets_by_time[0]["cached_at"],
        }
        stats["newest_dataset"] = {
            "id": datasets_by_time[-1]["id"],
            "cached_at": datasets_by_time[-1]["cached_at"],
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Получение статистики использования кеша.

        Returns:
            Словарь со статистикой
        """
        stats = {
            "total_datasets": 0,
            "total_size_bytes": 0,
            "average_dataset_size": 0,
            "largest_dataset": None,
            "smallest_dataset": None,
            "oldest_dataset": None,
            "newest_dataset": None,
            "protocol_versions": {},
            "cache_efficiency": 0.0,
        }

        datasets = []

        for str_id, cache_info in self.index.items():
            if str_id.startswith("_"):  # Исключаем метаданные
                continue

            dataset_id = cache_info.get("dataset_id", str_id)
            size_bytes = cache_info.get("size_bytes", 0)
            cached_at = cache_info.get("cached_at", "")
            protocol_version = cache_info.get("protocol_version", "unknown")

            datasets.append(
                {"id": dataset_id, "size": size_bytes, "cached_at": cached_at}
            )

            stats["total_size_bytes"] += size_bytes

            # Подсчет версий протокола
            if protocol_version not in stats["protocol_versions"]:
                stats["protocol_versions"][protocol_version] = 0
            stats["protocol_versions"][protocol_version] += 1

        stats["total_datasets"] = len(datasets)

        if datasets:
            self._calc_stat(stats, datasets)

        # Форматирование размеров
        stats["total_size"] = format_cache_size(stats["total_size_bytes"])
        stats["average_dataset_size_formatted"] = format_cache_size(
            stats["average_dataset_size"]
        )

        return stats

    def export_cache(self, export_path: Path) -> None:
        """Экспорт всего кеша в архив.

        Args:
            export_path: Путь для сохранения архива

        Raises:
            CacheError: При ошибке экспорта
        """
        try:
            with zipfile.ZipFile(export_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                # Добавляем индекс
                zipf.write(self.index_path, "cache_index.json")

                # Добавляем все файлы датасетов
                for str_id, cache_info in self.index.items():
                    if not str_id.startswith("_"):  # Исключаем метаданные
                        cache_file = self.cache_dir / cache_info.get("filename", "")
                        if cache_file.exists():
                            zipf.write(cache_file, cache_file.name)

            self.logger.info(f"Кеш экспортирован в {export_path}")

        except Exception as e:
            raise CacheError(
                f"Ошибка экспорта кеша: {e}",
                details={"export_path": str(export_path), "error": str(e)},
            )

    def import_cache(self, import_path: Path, merge: bool = False) -> int:
        """Импорт кеша из архива.

        Args:
            import_path: Путь к архиву
            merge: Объединить с существующим кешем (иначе заменить)

        Returns:
            Количество импортированных датасетов

        Raises:
            CacheError: При ошибке импорта
        """
        try:
            if not merge:
                # Очищаем существующий кеш
                self.clear_all()

            imported_count = 0

            with zipfile.ZipFile(import_path, "r") as zipf:
                # Извлекаем во временную директорию
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_path = Path(temp_dir)
                    zipf.extractall(temp_path)

                    # Загружаем индекс
                    temp_index_path = temp_path / "cache_index.json"
                    if temp_index_path.exists():
                        with open(temp_index_path, "r", encoding="utf-8") as f:
                            imported_index = json.load(f)

                        # Копируем файлы и обновляем индекс
                        for str_id, cache_info in imported_index.items():
                            if str_id.startswith("_"):  # Пропускаем метаданные
                                continue

                            filename = cache_info.get("filename", "")
                            temp_file = temp_path / filename

                            if temp_file.exists():
                                # Копируем файл
                                target_file = self.cache_dir / filename
                                shutil.copy2(temp_file, target_file)

                                # Обновляем индекс
                                if merge and str_id in self.index:
                                    self.logger.info(
                                        f"Датасет {str_id} обновлен из импорта"
                                    )
                                else:
                                    self.logger.info(f"Датасет {str_id} импортирован")

                                self.index[str_id] = cache_info
                                imported_count += 1

            # Сохраняем обновленный индекс
            self.save_index()

            self.logger.info(f"Импортировано датасетов: {imported_count}")
            return imported_count

        except Exception as e:
            raise CacheError(
                f"Ошибка импорта кеша: {e}",
                details={"import_path": str(import_path), "error": str(e)},
            )
