"""
Исключения для UCI Dataset Loader.

Этот модуль содержит все специализированные исключения,
используемые в функциональности загрузки датасетов UCI.
"""

from typing import Optional, Any, Dict
from .types import DatasetID, CacheStatus


class UCIDatasetError(Exception):
    """Базовое исключение для всех ошибок UCI Dataset Loader.
    
    Attributes:
        message: Сообщение об ошибке
        dataset_id: ID датасета, связанного с ошибкой (если применимо)
        error_type: Тип ошибки для категоризации
        details: Дополнительные детали об ошибке
    """
    
    def __init__(
        self,
        message: str,
        dataset_id: Optional[DatasetID] = None,
        error_type: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Инициализация исключения UCI Dataset.
        
        Args:
            message: Основное сообщение об ошибке
            dataset_id: ID датасета (опционально)
            error_type: Тип ошибки для категоризации (опционально)
            details: Дополнительная информация об ошибке (опционально)
        """
        super().__init__(message)
        self.message = message
        self.dataset_id = dataset_id
        self.error_type = error_type or self.__class__.__name__
        self.details = details or {}
    
    def __str__(self) -> str:
        """Строковое представление исключения."""
        parts = [self.message]
        
        if self.dataset_id is not None:
            parts.append(f"Dataset ID: {self.dataset_id}")
        
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            parts.append(f"Details: {details_str}")
        
        return " | ".join(parts)
    
    def __repr__(self) -> str:
        """Подробное представление исключения для отладки."""
        return (
            f"{self.__class__.__name__}("
            f"message={self.message!r}, "
            f"dataset_id={self.dataset_id!r}, "
            f"error_type={self.error_type!r}, "
            f"details={self.details!r})"
        )


class CacheError(UCIDatasetError):
    """Исключение для ошибок, связанных с кешем.
    
    Возникает при проблемах с чтением, записью или управлением кешем.
    """
    
    def __init__(
        self,
        message: str,
        dataset_id: Optional[DatasetID] = None,
        cache_status: Optional[CacheStatus] = None,
        cache_path: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Инициализация исключения кеша.
        
        Args:
            message: Сообщение об ошибке
            dataset_id: ID датасета (опционально)
            cache_status: Статус кеша при возникновении ошибки (опционально)
            cache_path: Путь к файлу кеша (опционально)
            details: Дополнительная информация (опционально)
        """
        super().__init__(message, dataset_id, "CacheError", details)
        self.cache_status = cache_status
        self.cache_path = cache_path
        
        # Добавляем информацию о кеше в детали
        if cache_status:
            self.details['cache_status'] = str(cache_status)
        if cache_path:
            self.details['cache_path'] = cache_path


class ValidationError(UCIDatasetError):
    """Исключение для ошибок валидации данных.
    
    Возникает при обнаружении некорректных данных, несоответствии формата
    или других проблемах валидации.
    """
    
    def __init__(
        self,
        message: str,
        dataset_id: Optional[DatasetID] = None,
        field_name: Optional[str] = None,
        invalid_value: Optional[Any] = None,
        expected_type: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Инициализация исключения валидации.
        
        Args:
            message: Сообщение об ошибке
            dataset_id: ID датасета (опционально)
            field_name: Имя поля с ошибкой (опционально)
            invalid_value: Некорректное значение (опционально)
            expected_type: Ожидаемый тип данных (опционально)
            details: Дополнительная информация (опционально)
        """
        super().__init__(message, dataset_id, "ValidationError", details)
        self.field_name = field_name
        self.invalid_value = invalid_value
        self.expected_type = expected_type
        
        # Добавляем информацию о валидации в детали
        if field_name:
            self.details['field_name'] = field_name
        if invalid_value is not None:
            self.details['invalid_value'] = repr(invalid_value)
        if expected_type:
            self.details['expected_type'] = expected_type


class DatasetNotFoundError(UCIDatasetError):
    """Исключение для случаев, когда датасет не найден.
    
    Возникает при попытке загрузить несуществующий датасет
    или когда датасет недоступен на сервере UCI.
    """
    
    def __init__(
        self,
        dataset_id: DatasetID,
        message: Optional[str] = None,
        searched_locations: Optional[list] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Инициализация исключения отсутствующего датасета.
        
        Args:
            dataset_id: ID датасета, который не был найден
            message: Сообщение об ошибке (опционально)
            searched_locations: Места, где производился поиск (опционально)
            details: Дополнительная информация (опционально)
        """
        if message is None:
            message = f"Датасет с ID '{dataset_id}' не найден"
        
        super().__init__(message, dataset_id, "DatasetNotFoundError", details)
        self.searched_locations = searched_locations or []
        
        # Добавляем информацию о поиске в детали
        if searched_locations:
            self.details['searched_locations'] = searched_locations


class NetworkError(UCIDatasetError):
    """Исключение для сетевых ошибок.
    
    Возникает при проблемах с сетевым соединением,
    недоступности сервера или тайм-аутах.
    """
    
    def __init__(
        self,
        message: str,
        dataset_id: Optional[DatasetID] = None,
        url: Optional[str] = None,
        status_code: Optional[int] = None,
        retry_count: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Инициализация сетевого исключения.
        
        Args:
            message: Сообщение об ошибке
            dataset_id: ID датасета (опционально)
            url: URL, при обращении к которому произошла ошибка (опционально)
            status_code: HTTP код ответа (опционально)
            retry_count: Количество попыток (опционально)
            details: Дополнительная информация (опционально)
        """
        super().__init__(message, dataset_id, "NetworkError", details)
        self.url = url
        self.status_code = status_code
        self.retry_count = retry_count
        
        # Добавляем сетевую информацию в детали
        if url:
            self.details['url'] = url
        if status_code is not None:
            self.details['status_code'] = status_code
        if retry_count is not None:
            self.details['retry_count'] = retry_count


class DataFormatError(UCIDatasetError):
    """Исключение для ошибок формата данных.
    
    Возникает при получении данных в неожиданном формате
    или при невозможности их корректной обработки.
    """
    
    def __init__(
        self,
        message: str,
        dataset_id: Optional[DatasetID] = None,
        expected_format: Optional[str] = None,
        actual_format: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Инициализация исключения формата данных.
        
        Args:
            message: Сообщение об ошибке
            dataset_id: ID датасета (опционально)
            expected_format: Ожидаемый формат данных (опционально)
            actual_format: Фактический формат данных (опционально)
            details: Дополнительная информация (опционально)
        """
        super().__init__(message, dataset_id, "DataFormatError", details)
        self.expected_format = expected_format
        self.actual_format = actual_format
        
        # Добавляем информацию о формате в детали
        if expected_format:
            self.details['expected_format'] = expected_format
        if actual_format:
            self.details['actual_format'] = actual_format


class ConfigurationError(UCIDatasetError):
    """Исключение для ошибок конфигурации.
    
    Возникает при неправильной настройке загрузчика
    или несовместимых параметрах.
    """
    
    def __init__(
        self,
        message: str,
        parameter_name: Optional[str] = None,
        parameter_value: Optional[Any] = None,
        valid_values: Optional[list] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Инициализация исключения конфигурации.
        
        Args:
            message: Сообщение об ошибке
            parameter_name: Имя параметра с ошибкой (опционально)
            parameter_value: Значение параметра (опционально)
            valid_values: Список допустимых значений (опционально)
            details: Дополнительная информация (опционально)
        """
        super().__init__(message, None, "ConfigurationError", details)
        self.parameter_name = parameter_name
        self.parameter_value = parameter_value
        self.valid_values = valid_values
        
        # Добавляем информацию о конфигурации в детали
        if parameter_name:
            self.details['parameter_name'] = parameter_name
        if parameter_value is not None:
            self.details['parameter_value'] = repr(parameter_value)
        if valid_values:
            self.details['valid_values'] = valid_values