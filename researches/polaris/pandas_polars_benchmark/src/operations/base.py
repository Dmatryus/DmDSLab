"""
Базовые классы и интерфейсы для операций бенчмаркинга.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass
import pandas as pd
import polars as pl

from researches.polaris.pandas_polars_benchmark.src.utils.logging import get_logger


@dataclass
class OperationResult:
    """Результат выполнения операции."""
    success: bool
    result: Optional[Any] = None
    error: Optional[Exception] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class Operation(ABC):
    """Абстрактный базовый класс для всех операций."""
    
    def __init__(self, name: str, category: str, description: str = ""):
        """
        Инициализация операции.
        
        Args:
            name: Уникальное имя операции
            category: Категория операции (io, filter, groupby, etc.)
            description: Описание операции
        """
        self.name = name
        self.category = category
        self.description = description
        self.logger = get_logger(f'operations.{category}')
        
    @abstractmethod
    def execute_pandas(self, 
                      df: pd.DataFrame, 
                      backend: str = "numpy",
                      **kwargs) -> OperationResult:
        """
        Выполнение операции с Pandas DataFrame.
        
        Args:
            df: Pandas DataFrame
            backend: Backend pandas ('numpy' или 'pyarrow')
            **kwargs: Дополнительные параметры операции
            
        Returns:
            OperationResult: Результат выполнения
        """
        pass
    
    @abstractmethod
    def execute_polars(self,
                      df: Union[pl.DataFrame, pl.LazyFrame],
                      lazy: bool = False,
                      **kwargs) -> OperationResult:
        """
        Выполнение операции с Polars DataFrame.
        
        Args:
            df: Polars DataFrame или LazyFrame
            lazy: Использовать lazy evaluation
            **kwargs: Дополнительные параметры операции
            
        Returns:
            OperationResult: Результат выполнения
        """
        pass
    
    def execute(self,
               df: Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame],
               library: str,
               backend: Optional[str] = None,
               **kwargs) -> OperationResult:
        """
        Универсальный метод выполнения операции.
        
        Args:
            df: DataFrame любой поддерживаемой библиотеки
            library: Название библиотеки ('pandas' или 'polars')
            backend: Backend для pandas
            **kwargs: Дополнительные параметры
            
        Returns:
            OperationResult: Результат выполнения
        """
        try:
            if library == "pandas":
                if not isinstance(df, pd.DataFrame):
                    raise TypeError(f"Ожидался pandas.DataFrame, получен {type(df)}")
                return self.execute_pandas(df, backend or "numpy", **kwargs)
                
            elif library == "polars":
                if not isinstance(df, (pl.DataFrame, pl.LazyFrame)):
                    raise TypeError(f"Ожидался polars.DataFrame/LazyFrame, получен {type(df)}")
                lazy = isinstance(df, pl.LazyFrame)
                return self.execute_polars(df, lazy=lazy, **kwargs)
                
            else:
                raise ValueError(f"Неподдерживаемая библиотека: {library}")
                
        except Exception as e:
            self.logger.error(f"Ошибка при выполнении операции {self.name}: {e}")
            return OperationResult(success=False, error=e)
    
    def get_params(self) -> Dict[str, Any]:
        """
        Возвращает параметры операции для логирования.
        
        Returns:
            Dict: Параметры операции
        """
        return {
            "name": self.name,
            "category": self.category,
            "description": self.description
        }
    
    def validate_input(self, df: Any) -> bool:
        """
        Валидация входных данных.
        
        Args:
            df: DataFrame для валидации
            
        Returns:
            bool: True если данные валидны
        """
        if df is None:
            return False
            
        if isinstance(df, pd.DataFrame):
            return len(df) > 0
        elif isinstance(df, (pl.DataFrame, pl.LazyFrame)):
            if isinstance(df, pl.LazyFrame):
                return True  # LazyFrame валиден по определению
            return len(df) > 0
            
        return False
    
    def __str__(self) -> str:
        """Строковое представление операции."""
        return f"{self.category}.{self.name}"
    
    def __repr__(self) -> str:
        """Представление для отладки."""
        return f"Operation(name='{self.name}', category='{self.category}')"


class DataAwareOperation(Operation):
    """Базовый класс для операций, требующих знания о структуре данных."""
    
    def __init__(self, name: str, category: str, description: str = ""):
        super().__init__(name, category, description)
        self._column_cache = {}
    
    def get_numeric_columns(self, df: Union[pd.DataFrame, pl.DataFrame]) -> List[str]:
        """Получает список числовых колонок."""
        cache_key = id(df)
        
        if cache_key in self._column_cache:
            return self._column_cache[cache_key]['numeric']
        
        if isinstance(df, pd.DataFrame):
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        else:  # Polars
            numeric_cols = [col for col in df.columns 
                           if df[col].dtype in [pl.Int32, pl.Int64, pl.Float32, pl.Float64]]
        
        self._column_cache[cache_key] = {'numeric': numeric_cols}
        return numeric_cols
    
    def get_string_columns(self, df: Union[pd.DataFrame, pl.DataFrame]) -> List[str]:
        """Получает список строковых колонок."""
        cache_key = id(df)
        
        if cache_key in self._column_cache and 'string' in self._column_cache[cache_key]:
            return self._column_cache[cache_key]['string']
        
        if isinstance(df, pd.DataFrame):
            string_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
        else:  # Polars
            string_cols = [col for col in df.columns if df[col].dtype == pl.Utf8]
        
        if cache_key not in self._column_cache:
            self._column_cache[cache_key] = {}
        self._column_cache[cache_key]['string'] = string_cols
        
        return string_cols
    
    def get_datetime_columns(self, df: Union[pd.DataFrame, pl.DataFrame]) -> List[str]:
        """Получает список datetime колонок."""
        cache_key = id(df)
        
        if cache_key in self._column_cache and 'datetime' in self._column_cache[cache_key]:
            return self._column_cache[cache_key]['datetime']
        
        if isinstance(df, pd.DataFrame):
            datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        else:  # Polars
            datetime_cols = [col for col in df.columns 
                           if df[col].dtype in [pl.Datetime, pl.Date]]
        
        if cache_key not in self._column_cache:
            self._column_cache[cache_key] = {}
        self._column_cache[cache_key]['datetime'] = datetime_cols
        
        return datetime_cols


class OperationRegistry:
    """Реестр всех доступных операций."""
    
    def __init__(self):
        self._operations: Dict[str, Operation] = {}
        self._categories: Dict[str, List[str]] = {}
        self.logger = get_logger('operation_registry')
    
    def register(self, operation: Operation) -> None:
        """
        Регистрирует операцию в реестре.
        
        Args:
            operation: Экземпляр операции
        """
        key = f"{operation.category}.{operation.name}"
        
        if key in self._operations:
            self.logger.warning(f"Операция {key} уже зарегистрирована, перезаписываем")
        
        self._operations[key] = operation
        
        # Добавляем в категории
        if operation.category not in self._categories:
            self._categories[operation.category] = []
        
        if operation.name not in self._categories[operation.category]:
            self._categories[operation.category].append(operation.name)
        
        self.logger.debug(f"Зарегистрирована операция: {key}")
    
    def get(self, name: str, category: Optional[str] = None) -> Optional[Operation]:
        """
        Получает операцию по имени.
        
        Args:
            name: Имя операции
            category: Категория (опционально)
            
        Returns:
            Operation или None
        """
        if category:
            key = f"{category}.{name}"
        else:
            # Пытаемся найти в любой категории
            for cat_name, ops in self._categories.items():
                if name in ops:
                    key = f"{cat_name}.{name}"
                    break
            else:
                return None
        
        return self._operations.get(key)
    
    def get_by_category(self, category: str) -> List[Operation]:
        """
        Получает все операции в категории.
        
        Args:
            category: Название категории
            
        Returns:
            List[Operation]: Список операций
        """
        operations = []
        
        if category in self._categories:
            for op_name in self._categories[category]:
                key = f"{category}.{op_name}"
                if key in self._operations:
                    operations.append(self._operations[key])
        
        return operations
    
    def get_all_operations(self) -> Dict[str, List[Operation]]:
        """
        Получает все операции, сгруппированные по категориям.
        
        Returns:
            Dict[str, List[Operation]]: Словарь категория -> список операций
        """
        result = {}
        
        for category in self._categories:
            result[category] = self.get_by_category(category)
        
        return result
    
    def list_operations(self) -> List[str]:
        """
        Возвращает список всех зарегистрированных операций.
        
        Returns:
            List[str]: Список в формате "category.operation"
        """
        return list(self._operations.keys())
    
    def clear(self) -> None:
        """Очищает реестр."""
        self._operations.clear()
        self._categories.clear()


# Глобальный реестр операций
_registry = OperationRegistry()


def register_operation(operation: Operation) -> None:
    """
    Регистрирует операцию в глобальном реестре.
    
    Args:
        operation: Экземпляр операции
    """
    _registry.register(operation)


def get_operation(name: str, category: Optional[str] = None) -> Optional[Operation]:
    """
    Получает операцию из глобального реестра.
    
    Args:
        name: Имя операции
        category: Категория
        
    Returns:
        Operation или None
    """
    return _registry.get(name, category)


def get_operations_by_category(category: str) -> List[Operation]:
    """
    Получает все операции в категории.
    
    Args:
        category: Название категории
        
    Returns:
        List[Operation]: Список операций
    """
    return _registry.get_by_category(category)


def get_all_operations() -> Dict[str, List[Operation]]:
    """
    Получает все зарегистрированные операции.
    
    Returns:
        Dict[str, List[Operation]]: Словарь категория -> операции
    """
    return _registry.get_all_operations()


# Декоратор для автоматической регистрации
def operation(name: str, category: str, description: str = ""):
    """
    Декоратор для автоматической регистрации операций.
    
    Пример:
        @operation("simple_filter", "filter", "Простая фильтрация")
        class SimpleFilterOperation(Operation):
            ...
    """
    def decorator(cls):
        # Создаем экземпляр и регистрируем
        instance = cls(name, category, description)
        register_operation(instance)
        return cls
    
    return decorator
