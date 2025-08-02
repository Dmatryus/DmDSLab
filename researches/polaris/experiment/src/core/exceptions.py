"""
Пользовательские исключения для проекта бенчмарков
"""
from typing import Optional, Dict, Any


class BenchmarkException(Exception):
    """Базовое исключение для всех ошибок проекта"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ConfigurationError(BenchmarkException):
    """Ошибки конфигурации"""
    pass


class DataGenerationError(BenchmarkException):
    """Ошибки при генерации данных"""
    
    def __init__(self, message: str, dataset_name: Optional[str] = None, 
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details)
        self.dataset_name = dataset_name
        if dataset_name:
            self.details['dataset_name'] = dataset_name


class BenchmarkExecutionError(BenchmarkException):
    """Ошибки при выполнении бенчмарков"""
    
    def __init__(self, message: str, framework: Optional[str] = None,
                 operation: Optional[str] = None, dataset: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details)
        self.framework = framework
        self.operation = operation
        self.dataset = dataset
        
        # Добавляем контекст в details
        if framework:
            self.details['framework'] = framework
        if operation:
            self.details['operation'] = operation
        if dataset:
            self.details['dataset'] = dataset


class InsufficientMemoryError(BenchmarkExecutionError):
    """Недостаточно памяти для выполнения операции"""
    
    def __init__(self, required_mb: float, available_mb: float, **kwargs):
        message = f"Insufficient memory: required {required_mb:.1f}MB, available {available_mb:.1f}MB"
        super().__init__(message, **kwargs)
        self.required_mb = required_mb
        self.available_mb = available_mb
        self.details.update({
            'required_mb': required_mb,
            'available_mb': available_mb
        })


class CheckpointError(BenchmarkException):
    """Ошибки работы с checkpoint'ами"""
    
    def __init__(self, message: str, checkpoint_path: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details)
        self.checkpoint_path = checkpoint_path
        if checkpoint_path:
            self.details['checkpoint_path'] = checkpoint_path


class ReportGenerationError(BenchmarkException):
    """Ошибки при генерации отчетов"""
    pass


class ValidationError(BenchmarkException):
    """Ошибки валидации данных"""
    
    def __init__(self, message: str, field: Optional[str] = None,
                 value: Any = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details)
        self.field = field
        self.value = value
        if field:
            self.details['field'] = field
        if value is not None:
            self.details['value'] = value


class FileOperationError(BenchmarkException):
    """Ошибки при работе с файлами"""
    
    def __init__(self, message: str, file_path: Optional[str] = None,
                 operation: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details)
        self.file_path = file_path
        self.operation = operation
        if file_path:
            self.details['file_path'] = file_path
        if operation:
            self.details['operation'] = operation


class MonitoringError(BenchmarkException):
    """Ошибки мониторинга ресурсов"""
    pass


class FrameworkNotSupportedError(BenchmarkException):
    """Фреймворк не поддерживается для данной операции"""
    
    def __init__(self, framework: str, operation: str, reason: Optional[str] = None):
        message = f"Framework '{framework}' does not support operation '{operation}'"
        if reason:
            message += f": {reason}"
        super().__init__(message)
        self.framework = framework
        self.operation = operation
        self.reason = reason
        self.details = {
            'framework': framework,
            'operation': operation,
            'reason': reason
        }