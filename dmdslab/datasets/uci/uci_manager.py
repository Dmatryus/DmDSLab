"""
Основной менеджер для загрузки датасетов UCI.

Этот модуль содержит главный класс UCIDatasetManager, который координирует
весь процесс загрузки, кеширования и обработки датасетов из репозитория UCI.
"""

import logging
import pickle
import json
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Union
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

try:
    import ucimlrepo
except ImportError:
    raise ImportError(
        "Пакет ucimlrepo не установлен. "
        "Установите его командой: pip install ucimlrepo"
    )

from .types import (
    DatasetID, LogLevel, TaskType, FeatureMatrix, TargetVector,
    CategoricalIndices, FeatureNames, MetadataDict, CacheStatus,
    DEFAULT_PICKLE_PROTOCOL
)
from .exceptions import (
    UCIDatasetError, CacheError, DatasetNotFoundError,
    NetworkError, DataFormatError, ValidationError
)
from .metadata import DatasetInfo, MetadataExtractor
from .utils import (
    setup_logger, validate_dataset_id, format_dataset_info,
    print_dataset_summary, create_progress_bar, progress_context,
    get_timestamp, format_cache_size, create_download_report
)

# Импортируем ModelData из основного пакета
try:
    from ...ml_data_container import ModelData
except ImportError:
    # Fallback для случая, если структура пакета отличается
    from dmdslab.datasets.ml_data_container import ModelData


class UCIDatasetManager:
    """Менеджер для загрузки и управления датасетами UCI.
    
    Основной класс для работы с датасетами из репозитория UCI Machine Learning.
    Поддерживает загрузку, кеширование, автоматическое определение типов признаков
    и интеграцию с классом ModelData.
    
    Attributes:
        cache_dir: Optional[Path]  # Директория для хранения кеша (None если кеш отключен)
        cache_index_path: Optional[Path]  # Путь к файлу индекса кеша
        cache_index: Dict[str, Any]  # Индекс кешированных датасетов
        metadata_extractor: MetadataExtractor  # Экстрактор метаданных
        raise_on_missing: Вызывать исключение при отсутствии датасета
        logger: Объект логгера
        show_progress: Показывать progress bar при загрузке
        use_cache: Использовать ли кеширование
    """
    
    def __init__(
        self,
        cache_dir: Optional[Union[str, Path]] = None,
        raise_on_missing: bool = True,
        log_level: Union[str, LogLevel] = LogLevel.INFO,
        show_progress: bool = True,
        use_cache: bool = True
    ):
        """Инициализация менеджера датасетов UCI.
        
        Args:
            cache_dir: Директория для кеша (по умолчанию ~/.uci_datasets)
            raise_on_missing: Вызывать исключение если датасет не найден
            log_level: Уровень логирования
            show_progress: Показывать progress bar при загрузке
            use_cache: Использовать кеширование (если False, данные не сохраняются)
        """
        # Параметры поведения
        self.raise_on_missing = raise_on_missing
        self.show_progress = show_progress
        self.use_cache = use_cache
        
        # Настройка директории кеша только если кеш используется
        if self.use_cache:
            if cache_dir is None:
                cache_dir = Path.home() / '.uci_datasets'
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = None
        
        # Настройка логирования
        self._setup_logging(log_level)
        
        # Инициализация кеша только если используется
        if self.use_cache:
            self._setup_cache()
            self.logger.info(f"UCIDatasetManager инициализирован. Кеш: {self.cache_dir}")
        else:
            self.cache_index = {}
            self.cache_index_path = None
            self.logger.info("UCIDatasetManager инициализирован. Кеширование отключено")
        
        # Экстрактор метаданных
        self.metadata_extractor = MetadataExtractor()
    
    def _setup_logging(self, log_level: Union[str, LogLevel]) -> None:
        """Настройка логирования.
        
        Args:
            log_level: Уровень логирования
        """
        self.logger = setup_logger(
            name="uci_loader",
            level=log_level,
            format_string='%(asctime)s - UCI Loader - %(levelname)s - %(message)s'
        )
    
    def _setup_cache(self) -> None:
        """Инициализация системы кеширования."""
        # Создаем директорию кеша если не существует
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Путь к индексу кеша
        self.cache_index_path = self.cache_dir / 'cache_index.json'
        
        # Загружаем или создаем индекс
        self._load_cache_index()
        
        self.logger.debug(f"Кеш инициализирован. Записей в индексе: {len(self.cache_index)}")
    
    def _load_cache_index(self) -> None:
        """Загрузка индекса кеша."""
        # Если кеш отключен, создаем пустой индекс
        if not self.use_cache:
            self.cache_index = {}
            return
        
        if self.cache_index_path.exists():
            try:
                with open(self.cache_index_path, 'r', encoding='utf-8') as f:
                    self.cache_index = json.load(f)
                self.logger.debug("Индекс кеша успешно загружен")
            except Exception as e:
                self.logger.warning(f"Ошибка загрузки индекса кеша: {e}. Создаем новый.")
                self.cache_index = {}
        else:
            self.cache_index = {}
            self.logger.debug("Создан новый индекс кеша")
    
    def _save_cache_index(self) -> None:
        """Сохранение индекса кеша."""
        # Если кеш отключен, ничего не сохраняем
        if not self.use_cache:
            return
        
        try:
            with open(self.cache_index_path, 'w', encoding='utf-8') as f:
                json.dump(self.cache_index, f, indent=2, ensure_ascii=False)
            self.logger.debug("Индекс кеша сохранен")
        except Exception as e:
            self.logger.error(f"Ошибка сохранения индекса кеша: {e}")
    
    def _check_cache(self, dataset_id: DatasetID) -> Tuple[CacheStatus, Optional[Dict[str, Any]]]:
        """Проверка наличия датасета в кеше.
        
        Args:
            dataset_id: ID датасета
            
        Returns:
            Кортеж (статус кеша, метаданные кеша или None)
        """
        # Если кеш отключен, всегда возвращаем MISS
        if not self.use_cache:
            return CacheStatus.MISS, None
        
        str_id = str(dataset_id)
        
        if str_id not in self.cache_index:
            return CacheStatus.MISS, None
        
        cache_info = self.cache_index[str_id]
        cache_file = self.cache_dir / cache_info['filename']
        
        if not cache_file.exists():
            self.logger.warning(f"Файл кеша для датасета {dataset_id} не найден")
            return CacheStatus.CORRUPTED, cache_info
        
        return CacheStatus.HIT, cache_info
    
    def _load_from_cache(self, dataset_id: DatasetID) -> ModelData:
        """Загрузка датасета из кеша.
        
        Args:
            dataset_id: ID датасета
            
        Returns:
            Объект ModelData
            
        Raises:
            CacheError: При ошибке загрузки из кеша
        """
        # Проверка на случай некорректного вызова
        if not self.use_cache:
            raise CacheError(
                "Попытка загрузки из кеша при отключенном кешировании",
                dataset_id=dataset_id
            )
        
        str_id = str(dataset_id)
        cache_info = self.cache_index[str_id]
        cache_file = self.cache_dir / cache_info['filename']
        
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            
            self.logger.info(f"Датасет {dataset_id} загружен из кеша")
            
            # Воссоздаем ModelData из кешированных данных
            return self._create_model_data(
                features=cached_data['features'],
                target=cached_data.get('target'),
                feature_names=cached_data.get('feature_names'),
                metadata=cached_data.get('metadata', {})
            )
            
        except Exception as e:
            raise CacheError(
                f"Ошибка загрузки датасета {dataset_id} из кеша",
                dataset_id=dataset_id,
                cache_path=str(cache_file),
                details={'error': str(e)}
            )
    
    def _save_to_cache(
        self,
        dataset_id: DatasetID,
        data: Dict[str, Any],
        metadata: MetadataDict
    ) -> bool:
        """Сохранение датасета в кеш.
        
        Args:
            dataset_id: ID датасета
            data: Данные для сохранения
            metadata: Метаданные датасета
            
        Returns:
            True если успешно сохранено или кеш отключен
        """
        # Если кеш отключен, просто возвращаем успех
        if not self.use_cache:
            self.logger.debug(f"Кеширование отключено, пропускаем сохранение датасета {dataset_id}")
            return True
        
        str_id = str(dataset_id)
        timestamp = get_timestamp()
        filename = f"dataset_{str_id}_{timestamp.replace(':', '-').replace(' ', '_')}.pkl"
        cache_file = self.cache_dir / filename
        
        try:
            # Сохраняем данные
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f, protocol=DEFAULT_PICKLE_PROTOCOL)
            
            # Обновляем индекс
            file_size = cache_file.stat().st_size
            self.cache_index[str_id] = {
                'filename': filename,
                'dataset_id': dataset_id,
                'cached_at': timestamp,
                'size_bytes': file_size,
                'size_human': format_cache_size(file_size),
                'metadata': metadata
            }
            
            # Сохраняем индекс
            self._save_cache_index()
            
            self.logger.info(
                f"Датасет {dataset_id} сохранен в кеш "
                f"({format_cache_size(file_size)})"
            )
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка сохранения в кеш: {e}")
            # Удаляем файл если он был частично создан
            if cache_file.exists():
                cache_file.unlink()
            return False
    
    def _fetch_from_uci(
        self,
        dataset_id: DatasetID,
        show_progress: Optional[bool] = None
    ) -> Any:
        """Загрузка датасета из репозитория UCI.
        
        Args:
            dataset_id: ID датасета
            show_progress: Показывать progress bar
            
        Returns:
            Объект данных из ucimlrepo
            
        Raises:
            DatasetNotFoundError: Если датасет не найден
            NetworkError: При сетевых ошибках
        """
        if show_progress is None:
            show_progress = self.show_progress
        
        self.logger.info(f"Начинаем загрузку датасета {dataset_id} из UCI")
        
        try:
            # Создаем progress bar если нужно
            if show_progress:
                pbar = create_progress_bar(
                    total=None,  # Неизвестный размер
                    desc=f"Загрузка датасета {dataset_id}",
                    unit='B',
                    unit_scale=True
                )
            
            # Загружаем датасет
            dataset = ucimlrepo.fetch(ucid=int(dataset_id))
            
            if show_progress:
                pbar.close()
            
            self.logger.info(f"Датасет {dataset_id} успешно загружен")
            return dataset
            
        except ValueError as e:
            if "Dataset not found" in str(e):
                raise DatasetNotFoundError(
                    dataset_id=dataset_id,
                    searched_locations=["UCI ML Repository"]
                )
            raise NetworkError(
                f"Ошибка загрузки датасета {dataset_id}: {e}",
                dataset_id=dataset_id
            )
        except Exception as e:
            raise NetworkError(
                f"Неожиданная ошибка при загрузке датасета {dataset_id}: {e}",
                dataset_id=dataset_id,
                details={'error_type': type(e).__name__}
            )
    
    def _analyze_structure(self, raw_data: Any) -> Tuple[FeatureMatrix, Optional[TargetVector], Optional[FeatureNames]]:
        """Анализ структуры загруженных данных.
        
        Args:
            raw_data: Сырые данные из UCI
            
        Returns:
            Кортеж (features, target, feature_names)
            
        Raises:
            DataFormatError: При неправильном формате данных
        """
        try:
            # Извлекаем данные
            data = raw_data.data
            features = data.get('features')
            targets = data.get('targets')
            
            if features is None:
                raise DataFormatError(
                    "Отсутствуют признаки в загруженных данных",
                    expected_format="data.features",
                    actual_format="None"
                )
            
            # Извлекаем имена признаков
            feature_names = None
            if hasattr(features, 'columns'):
                feature_names = list(features.columns)
            
            # Обработка целевой переменной
            target = None
            if targets is not None:
                if isinstance(targets, pd.DataFrame) and targets.shape[1] == 1:
                    target = targets.iloc[:, 0]
                elif isinstance(targets, pd.Series):
                    target = targets
                elif isinstance(targets, np.ndarray):
                    if targets.ndim == 1:
                        target = targets
                    elif targets.shape[1] == 1:
                        target = targets[:, 0]
                    else:
                        # Множественные целевые переменные
                        self.logger.warning(
                            f"Обнаружено {targets.shape[1]} целевых переменных. "
                            "Используется первая."
                        )
                        target = targets[:, 0] if isinstance(targets, np.ndarray) else targets.iloc[:, 0]
            
            self.logger.debug(
                f"Структура данных: features {features.shape}, "
                f"target {target.shape if target is not None else 'None'}"
            )
            
            return features, target, feature_names
            
        except Exception as e:
            raise DataFormatError(
                f"Ошибка анализа структуры данных: {e}",
                details={'error': str(e)}
            )
    
    def _detect_task_type(
        self,
        features: FeatureMatrix,
        target: Optional[TargetVector],
        metadata: Optional[MetadataDict] = None
    ) -> TaskType:
        """Определение типа задачи машинного обучения.
        
        Args:
            features: Матрица признаков
            target: Целевая переменная
            metadata: Метаданные датасета
            
        Returns:
            Тип задачи
        """
        return self.metadata_extractor.determine_task_type(features, target, metadata)
    
    def _detect_categorical_features(
        self,
        features: FeatureMatrix,
        feature_names: Optional[FeatureNames] = None
    ) -> CategoricalIndices:
        """Определение категориальных признаков.
        
        Args:
            features: Матрица признаков
            feature_names: Имена признаков
            
        Returns:
            Список индексов категориальных признаков
        """
        categorical_indices = self.metadata_extractor.detect_categorical(features, feature_names)
        
        if categorical_indices:
            self.logger.info(
                f"Обнаружено {len(categorical_indices)} категориальных признаков: "
                f"{categorical_indices[:5]}{'...' if len(categorical_indices) > 5 else ''}"
            )
        
        return categorical_indices
    
    def _create_model_data(
        self,
        features: FeatureMatrix,
        target: Optional[TargetVector],
        feature_names: Optional[FeatureNames],
        metadata: MetadataDict
    ) -> ModelData:
        """Создание объекта ModelData.
        
        Args:
            features: Матрица признаков
            target: Целевая переменная
            feature_names: Имена признаков
            metadata: Метаданные
            
        Returns:
            Объект ModelData
        """
        # Определяем тип задачи
        task_type = self._detect_task_type(features, target, metadata)
        
        # Определяем категориальные признаки
        categorical_features = self._detect_categorical_features(features, feature_names)
        
        # Подготавливаем данные для ModelData
        if isinstance(features, pd.DataFrame):
            X = features.values
            feature_names = list(features.columns)
        else:
            X = features
        
        if target is not None:
            if isinstance(target, pd.Series):
                y = target.values
            else:
                y = target
        else:
            y = None
        
        # Создаем объект ModelData
        model_data = ModelData(
            X=X,
            y=y,
            feature_names=feature_names,
            categorical_features=categorical_features
        )
        
        # Добавляем метаданные
        model_data.metadata = metadata
        model_data.task_type = task_type.value
        
        return model_data
    
    def load_dataset(
        self,
        dataset_id: DatasetID,
        force_reload: bool = False
    ) -> ModelData:
        """Загрузка одного датасета.
        
        Args:
            dataset_id: ID датасета для загрузки
            force_reload: Принудительная перезагрузка игнорируя кеш
                         (игнорируется если use_cache=False)
            
        Returns:
            Объект ModelData с загруженными данными
            
        Raises:
            DatasetNotFoundError: Если датасет не найден
            UCIDatasetError: При других ошибках загрузки
        """
        # Валидация ID
        dataset_id = validate_dataset_id(dataset_id)
        
        self.logger.info(f"Запрос на загрузку датасета {dataset_id}")
        
        # Проверяем кеш если не требуется принудительная перезагрузка и кеш включен
        if not force_reload and self.use_cache:
            cache_status, cache_info = self._check_cache(dataset_id)
            
            if cache_status == CacheStatus.HIT:
                self.logger.debug(f"Датасет {dataset_id} найден в кеше")
                try:
                    return self._load_from_cache(dataset_id)
                except CacheError as e:
                    self.logger.warning(f"Ошибка загрузки из кеша: {e}. Загружаем заново.")
                    # Продолжаем с загрузкой из UCI
            else:
                self.logger.debug(f"Датасет {dataset_id} не найден в кеше")
        elif force_reload and self.use_cache:
            self.logger.info(f"Принудительная перезагрузка датасета {dataset_id}")
        elif not self.use_cache:
            self.logger.debug("Кеширование отключено, загружаем напрямую из UCI")
        
        # Загружаем из UCI
        try:
            raw_data = self._fetch_from_uci(dataset_id)
            
            # Анализируем структуру
            features, target, feature_names = self._analyze_structure(raw_data)
            
            # Извлекаем метаданные
            metadata = self.metadata_extractor.extract_from_uci(raw_data)
            
            # Создаем DatasetInfo
            dataset_info = DatasetInfo.from_uci_data(
                dataset_id=dataset_id,
                uci_data=raw_data,
                features=features,
                target=target
            )
            
            # Подготавливаем данные для кеширования
            cache_data = {
                'features': features,
                'target': target,
                'feature_names': feature_names,
                'metadata': metadata,
                'dataset_info': dataset_info.to_dict()
            }
            
            # Сохраняем в кеш
            self._save_to_cache(dataset_id, cache_data, metadata)
            
            # Создаем ModelData
            model_data = self._create_model_data(features, target, feature_names, metadata)
            
            # Выводим информацию о датасете
            if self.logger.level <= logging.INFO:
                print(format_dataset_info(
                    dataset_id=dataset_id,
                    name=dataset_info.name,
                    task_type=dataset_info.task_type,
                    n_instances=dataset_info.n_instances,
                    n_features=dataset_info.n_features,
                    domain=dataset_info.domain,
                    has_missing=dataset_info.has_missing_values,
                    cached=True if self.use_cache else None,  # True если кеш включен, None если отключен
                    cache_size=None
                ))
            
            return model_data
            
        except UCIDatasetError:
            # Пробрасываем наши исключения дальше
            raise
        except Exception as e:
            # Оборачиваем неожиданные исключения
            raise UCIDatasetError(
                f"Неожиданная ошибка при загрузке датасета {dataset_id}: {e}",
                dataset_id=dataset_id,
                details={'error_type': type(e).__name__, 'error': str(e)}
            )
    
    def load_datasets(
        self,
        dataset_ids: List[DatasetID],
        force_reload: bool = False,
        stop_on_error: bool = False
    ) -> List[Union[ModelData, Exception]]:
        """Загрузка нескольких датасетов последовательно.
        
        Args:
            dataset_ids: Список ID датасетов для загрузки
            force_reload: Принудительная перезагрузка для всех
                         (игнорируется если use_cache=False)
            stop_on_error: Остановиться при первой ошибке
            
        Returns:
            Список результатов (ModelData или Exception для каждого датасета)
            
        Raises:
            UCIDatasetError: Если stop_on_error=True и произошла ошибка
        """
        self.logger.info(f"Начинаем загрузку {len(dataset_ids)} датасетов")
        
        results = []
        start_time = datetime.now().timestamp()
        
        # Progress bar для общего прогресса
        with progress_context(
            desc="Загрузка датасетов",
            total=len(dataset_ids),
            disable=not self.show_progress
        ) as pbar:
            
            for i, dataset_id in enumerate(dataset_ids):
                pbar.set_description(f"Загрузка датасета {dataset_id}")
                
                try:
                    model_data = self.load_dataset(dataset_id, force_reload)
                    results.append(model_data)
                    
                except Exception as e:
                    self.logger.error(f"Ошибка загрузки датасета {dataset_id}: {e}")
                    
                    if stop_on_error:
                        raise
                    
                    results.append(e)
                
                pbar.update(1)
        
        # Создаем отчет
        results_dict = {
            dataset_ids[i]: results[i] for i in range(len(dataset_ids))
        }
        
        # Используем cache_dir только если кеш включен
        cache_dir_for_report = self.cache_dir if self.use_cache else Path("no_cache")
        
        report = create_download_report(
            dataset_ids=dataset_ids,
            results=results_dict,
            start_time=start_time,
            cache_dir=cache_dir_for_report
        )
        
        if self.logger.level <= logging.INFO:
            print(report)
        
        return results
    
    def clear_cache(self, dataset_id: Optional[DatasetID] = None) -> bool:
        """Очистка кеша.
        
        Args:
            dataset_id: ID датасета для очистки (None - очистить весь кеш)
            
        Returns:
            True если успешно очищено или кеш отключен
        """
        # Если кеш отключен, нечего очищать
        if not self.use_cache:
            self.logger.info("Кеширование отключено, очистка не требуется")
            return True
        
        if dataset_id is None:
            # Очистка всего кеша
            self.logger.warning("Очистка всего кеша")
            
            try:
                # Удаляем все файлы
                for file_path in self.cache_dir.glob("*.pkl"):
                    file_path.unlink()
                
                # Очищаем индекс
                self.cache_index = {}
                self._save_cache_index()
                
                self.logger.info("Кеш полностью очищен")
                return True
                
            except Exception as e:
                self.logger.error(f"Ошибка очистки кеша: {e}")
                return False
        else:
            # Очистка конкретного датасета
            dataset_id = validate_dataset_id(dataset_id)
            str_id = str(dataset_id)
            
            if str_id not in self.cache_index:
                self.logger.warning(f"Датасет {dataset_id} не найден в кеше")
                return False
            
            try:
                # Удаляем файл
                cache_info = self.cache_index[str_id]
                cache_file = self.cache_dir / cache_info['filename']
                
                if cache_file.exists():
                    cache_file.unlink()
                
                # Удаляем из индекса
                del self.cache_index[str_id]
                self._save_cache_index()
                
                self.logger.info(f"Датасет {dataset_id} удален из кеша")
                return True
                
            except Exception as e:
                self.logger.error(f"Ошибка удаления датасета {dataset_id} из кеша: {e}")
                return False
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Получение информации о кеше.
        
        Returns:
            Словарь с информацией о кеше
        """
        # Если кеш отключен, возвращаем соответствующую информацию
        if not self.use_cache:
            return {
                'cache_enabled': False,
                'message': 'Кеширование отключено'
            }
        
        total_size = 0
        dataset_count = len(self.cache_index)
        
        # Подсчитываем общий размер
        for cache_info in self.cache_index.values():
            total_size += cache_info.get('size_bytes', 0)
        
        # Информация о датасетах
        datasets_info = []
        for dataset_id, cache_info in self.cache_index.items():
            datasets_info.append({
                'dataset_id': dataset_id,
                'cached_at': cache_info.get('cached_at'),
                'size': cache_info.get('size_human', 'N/A'),
                'name': cache_info.get('metadata', {}).get('name', f'Dataset {dataset_id}')
            })
        
        # Сортируем по времени кеширования
        datasets_info.sort(key=lambda x: x.get('cached_at', ''), reverse=True)
        
        return {
            'cache_enabled': True,
            'cache_directory': str(self.cache_dir),
            'total_datasets': dataset_count,
            'total_size': format_cache_size(total_size),
            'total_size_bytes': total_size,
            'datasets': datasets_info[:10],  # Топ 10 последних
            'index_file': str(self.cache_index_path)
        }