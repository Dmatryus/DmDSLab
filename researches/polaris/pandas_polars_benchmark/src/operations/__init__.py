"""
Модуль операций для бенчмаркинга Pandas vs Polars.
"""

# Базовые классы и функции
from .base import (
    Operation,
    DataAwareOperation,
    OperationResult,
    OperationRegistry,
    register_operation,
    get_operation,
    get_operations_by_category,
    get_all_operations,
    operation
)

# IO операции
from .io_ops import (
    ReadCSVOperation,
    ReadParquetOperation,
    WriteCSVOperation,
    WriteParquetOperation,
    create_io_operations,
    profile_io_operation
)

# Filter операции
from .filter_ops import (
    SimpleFilterOperation,
    ComplexFilterOperation,
    IsInFilterOperation,
    PatternFilterOperation
)

# Экспортируемые имена
__all__ = [
    # Базовые классы
    'Operation',
    'DataAwareOperation',
    'OperationResult',
    'OperationRegistry',
    
    # Функции реестра
    'register_operation',
    'get_operation',
    'get_operations_by_category',
    'get_all_operations',
    'operation',
    
    # IO операции
    'ReadCSVOperation',
    'ReadParquetOperation',
    'WriteCSVOperation',
    'WriteParquetOperation',
    'create_io_operations',
    'profile_io_operation',
    
    # Filter операции
    'SimpleFilterOperation',
    'ComplexFilterOperation',
    'IsInFilterOperation',
    'PatternFilterOperation',
]


# Вспомогательная функция для быстрого доступа к операциям
def get_available_operations() -> dict:
    """
    Возвращает словарь всех доступных операций по категориям.
    
    Returns:
        dict: Словарь вида {категория: {имя_операции: операция}}
    """
    result = {}
    all_ops = get_all_operations()
    
    for category, operations in all_ops.items():
        result[category] = {op.name: op for op in operations}
    
    return result


def list_operations() -> None:
    """Выводит список всех доступных операций."""
    print("Доступные операции:")
    print("-" * 50)
    
    for category, operations in get_all_operations().items():
        print(f"\n{category.upper()}:")
        for op in operations:
            print(f"  - {op.name}: {op.description}")
