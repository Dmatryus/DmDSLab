"""
Валидаторы для конфигурации эксперимента
"""
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
import re

from ..constants import (
    MIN_ROWS, MAX_ROWS, MAX_COLUMNS, MAX_MEMORY_GB,
    DatasetSize, DatasetShape, DataTypeMix, Framework,
    OperationType, Distribution, FileFormat
)
from ..exceptions import ValidationError


class ConfigValidator:
    """Класс для валидации конфигурации"""
    
    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def validate_experiment_config(self, config: Dict[str, Any]) -> bool:
        """
        Полная валидация конфигурации эксперимента
        
        Args:
            config: Словарь с конфигурацией
            
        Returns:
            True если конфигурация валидна
            
        Raises:
            ValidationError: При критических ошибках валидации
        """
        self.errors = []
        self.warnings = []
        
        # Проверка обязательных секций
        required_sections = ['experiment', 'data_generation', 'benchmarks', 'output']
        for section in required_sections:
            if section not in config:
                self.errors.append(f"Missing required section: '{section}'")
        
        if self.errors:
            raise ValidationError(
                f"Configuration missing required sections",
                details={'errors': self.errors}
            )
        
        # Валидация каждой секции
        self._validate_experiment_section(config.get('experiment', {}))
        self._validate_data_generation_section(config.get('data_generation', {}))
        self._validate_benchmarks_section(config.get('benchmarks', {}))
        self._validate_output_section(config.get('output', {}))
        
        if config.get('logging'):
            self._validate_logging_section(config['logging'])
        
        # Проверка на критические ошибки
        if self.errors:
            raise ValidationError(
                "Configuration validation failed",
                details={'errors': self.errors, 'warnings': self.warnings}
            )
        
        # Вывод предупреждений
        if self.warnings:
            import logging
            logger = logging.getLogger(__name__)
            for warning in self.warnings:
                logger.warning(f"Configuration warning: {warning}")
        
        return True
    
    def _validate_experiment_section(self, experiment: Dict[str, Any]):
        """Валидация секции experiment"""
        if not experiment.get('name'):
            self.errors.append("Experiment name is required")
        elif not isinstance(experiment['name'], str):
            self.errors.append("Experiment name must be a string")
        elif not re.match(r'^[\w\-]+$', experiment['name']):
            self.warnings.append(
                "Experiment name should contain only letters, numbers, "
                "underscores and hyphens"
            )
    
    def _validate_data_generation_section(self, data_gen: Dict[str, Any]):
        """Валидация секции data_generation"""
        # Проверка seed
        if 'seed' not in data_gen:
            self.warnings.append("No seed specified, using default")
        elif not isinstance(data_gen['seed'], int):
            self.errors.append("Seed must be an integer")
        
        # Проверка размеров
        sizes = data_gen.get('sizes', [])
        if not sizes:
            self.errors.append("At least one dataset size must be defined")
        else:
            self._validate_sizes(sizes)
        
        # Проверка форм
        shapes = data_gen.get('shapes', [])
        if not shapes:
            self.errors.append("At least one dataset shape must be defined")
        else:
            self._validate_shapes(shapes)
        
        # Проверка type mixes
        type_mixes = data_gen.get('type_mixes', [])
        if not type_mixes:
            self.errors.append("At least one type mix must be defined")
        else:
            self._validate_type_mixes(type_mixes)
        
        # Проверка column types
        if 'column_types' in data_gen:
            self._validate_column_types(data_gen['column_types'])
    
    def _validate_sizes(self, sizes: List[Dict[str, Any]]):
        """Валидация размеров датасетов"""
        seen_names = set()
        
        for size in sizes:
            name = size.get('name')
            if not name:
                self.errors.append("Size configuration missing 'name'")
                continue
            
            if name in seen_names:
                self.errors.append(f"Duplicate size name: '{name}'")
            seen_names.add(name)
            
            # Проверка что имя соответствует enum
            try:
                DatasetSize(name)
            except ValueError:
                self.errors.append(
                    f"Invalid size name '{name}'. "
                    f"Must be one of: {[e.value for e in DatasetSize]}"
                )
            
            # Проверка target_size_mb
            target_mb = size.get('target_size_mb')
            if target_mb is None:
                self.errors.append(f"Size '{name}' missing 'target_size_mb'")
            elif not isinstance(target_mb, (int, float)) or target_mb <= 0:
                self.errors.append(
                    f"Size '{name}' target_size_mb must be positive number"
                )
            elif target_mb > MAX_MEMORY_GB * 1024:
                self.warnings.append(
                    f"Size '{name}' target_size_mb ({target_mb}MB) "
                    f"exceeds max memory limit ({MAX_MEMORY_GB}GB)"
                )
    
    def _validate_shapes(self, shapes: List[Dict[str, Any]]):
        """Валидация форм датасетов"""
        seen_names = set()
        
        for shape in shapes:
            name = shape.get('name')
            if not name:
                self.errors.append("Shape configuration missing 'name'")
                continue
            
            if name in seen_names:
                self.errors.append(f"Duplicate shape name: '{name}'")
            seen_names.add(name)
            
            # Проверка что имя соответствует enum
            try:
                DatasetShape(name)
            except ValueError:
                self.errors.append(
                    f"Invalid shape name '{name}'. "
                    f"Must be one of: {[e.value for e in DatasetShape]}"
                )
            
            # Проверка columns
            columns = shape.get('columns')
            if columns is None:
                self.errors.append(f"Shape '{name}' missing 'columns'")
            elif not isinstance(columns, int) or columns <= 0:
                self.errors.append(
                    f"Shape '{name}' columns must be positive integer"
                )
            elif columns > MAX_COLUMNS:
                self.errors.append(
                    f"Shape '{name}' columns ({columns}) "
                    f"exceeds maximum ({MAX_COLUMNS})"
                )
            
            # Проверка description
            if not shape.get('description'):
                self.warnings.append(f"Shape '{name}' missing description")
    
    def _validate_type_mixes(self, type_mixes: List[Dict[str, Any]]):
        """Валидация распределений типов"""
        seen_names = set()
        
        for mix in type_mixes:
            name = mix.get('name')
            if not name:
                self.errors.append("Type mix configuration missing 'name'")
                continue
            
            if name in seen_names:
                self.errors.append(f"Duplicate type mix name: '{name}'")
            seen_names.add(name)
            
            # Проверка что имя соответствует enum
            try:
                DataTypeMix(name)
            except ValueError:
                self.errors.append(
                    f"Invalid type mix name '{name}'. "
                    f"Must be one of: {[e.value for e in DataTypeMix]}"
                )
            
            # Проверка composition
            composition = mix.get('composition', {})
            if not composition:
                self.errors.append(f"Type mix '{name}' missing composition")
                continue
            
            # Проверка процентов
            total_percentage = 0.0
            for dtype, percentage_str in composition.items():
                if not isinstance(percentage_str, str) or not percentage_str.endswith('%'):
                    self.errors.append(
                        f"Type mix '{name}': percentage for '{dtype}' "
                        f"must be string ending with '%'"
                    )
                    continue
                
                try:
                    percentage = float(percentage_str.rstrip('%'))
                    if percentage < 0 or percentage > 100:
                        self.errors.append(
                            f"Type mix '{name}': percentage for '{dtype}' "
                            f"must be between 0 and 100"
                        )
                    total_percentage += percentage
                except ValueError:
                    self.errors.append(
                        f"Type mix '{name}': invalid percentage format "
                        f"for '{dtype}': {percentage_str}"
                    )
            
            # Проверка суммы процентов
            if abs(total_percentage - 100.0) > 0.01:
                self.errors.append(
                    f"Type mix '{name}': percentages sum to {total_percentage}%, "
                    f"expected 100%"
                )
    
    def _validate_column_types(self, column_types: Dict[str, Any]):
        """Валидация конфигурации типов колонок"""
        # Валидация числовых типов
        if 'numeric' in column_types:
            for dtype, config in column_types['numeric'].items():
                self._validate_numeric_type(dtype, config)
        
        # Валидация строковых типов
        if 'string' in column_types:
            for dtype, config in column_types['string'].items():
                self._validate_string_type(dtype, config)
        
        # Валидация категориальных типов
        if 'categorical' in column_types:
            for dtype, config in column_types['categorical'].items():
                self._validate_categorical_type(dtype, config)
        
        # Валидация временных типов
        if 'temporal' in column_types:
            for dtype, config in column_types['temporal'].items():
                self._validate_temporal_type(dtype, config)
        
        # Валидация булевых типов
        if 'boolean' in column_types:
            for dtype, config in column_types['boolean'].items():
                self._validate_boolean_type(dtype, config)
    
    def _validate_numeric_type(self, dtype: str, config: Dict[str, Any]):
        """Валидация конфигурации числового типа"""
        valid_types = ['int8', 'int16', 'int32', 'int64', 'float32', 'float64']
        if dtype not in valid_types:
            self.errors.append(f"Invalid numeric type: '{dtype}'")
            return
        
        # Проверка min/max
        if 'min' in config and 'max' in config:
            if config['min'] > config['max']:
                self.errors.append(
                    f"Numeric type '{dtype}': min > max"
                )
        
        # Проверка distribution
        if 'distribution' in config:
            try:
                Distribution(config['distribution'])
            except ValueError:
                self.errors.append(
                    f"Numeric type '{dtype}': invalid distribution "
                    f"'{config['distribution']}'"
                )
        
        # Проверка null_percentage
        if 'null_percentage' in config:
            null_pct = config['null_percentage']
            if not isinstance(null_pct, (int, float)) or null_pct < 0 or null_pct > 100:
                self.errors.append(
                    f"Numeric type '{dtype}': null_percentage must be 0-100"
                )
    
    def _validate_string_type(self, dtype: str, config: Dict[str, Any]):
        """Валидация конфигурации строкового типа"""
        if not dtype.startswith('string_'):
            self.errors.append(f"Invalid string type: '{dtype}'")
            return
        
        # Проверка длин
        min_len = config.get('min_length', 0)
        max_len = config.get('max_length', 0)
        
        if not isinstance(min_len, int) or min_len < 0:
            self.errors.append(f"String type '{dtype}': min_length must be non-negative integer")
        
        if not isinstance(max_len, int) or max_len < min_len:
            self.errors.append(f"String type '{dtype}': max_length must be >= min_length")
        
        # Проверка charset
        valid_charsets = ['alphanumeric', 'ascii', 'unicode']
        if config.get('charset') not in valid_charsets:
            self.errors.append(
                f"String type '{dtype}': charset must be one of {valid_charsets}"
            )
    
    def _validate_categorical_type(self, dtype: str, config: Dict[str, Any]):
        """Валидация конфигурации категориального типа"""
        cardinality = config.get('cardinality')
        if not isinstance(cardinality, int) or cardinality <= 0:
            self.errors.append(
                f"Categorical type '{dtype}': cardinality must be positive integer"
            )
    
    def _validate_temporal_type(self, dtype: str, config: Dict[str, Any]):
        """Валидация конфигурации временного типа"""
        if dtype == 'datetime' or dtype == 'date':
            if not config.get('start') or not config.get('end'):
                self.errors.append(
                    f"Temporal type '{dtype}': 'start' and 'end' are required"
                )
            # TODO: Проверка формата дат
    
    def _validate_boolean_type(self, dtype: str, config: Dict[str, Any]):
        """Валидация конфигурации булевого типа"""
        true_ratio = config.get('true_ratio', 0.5)
        if not isinstance(true_ratio, (int, float)) or true_ratio < 0 or true_ratio > 1:
            self.errors.append(
                f"Boolean type '{dtype}': true_ratio must be between 0 and 1"
            )
    
    def _validate_benchmarks_section(self, benchmarks: Dict[str, Any]):
        """Валидация секции benchmarks"""
        # Проверка iterations
        iterations = benchmarks.get('iterations', 1)
        if not isinstance(iterations, int) or iterations < 1:
            self.errors.append("Benchmarks iterations must be positive integer")
        
        # Проверка операций
        operations = benchmarks.get('operations', {})
        if not operations:
            self.errors.append("No operations defined for benchmarks")
        else:
            self._validate_operations(operations)
        
        # Проверка фреймворков
        frameworks = benchmarks.get('frameworks', {})
        if not frameworks:
            self.errors.append("No frameworks defined for benchmarks")
        else:
            self._validate_frameworks(frameworks)
    
    def _validate_operations(self, operations: Dict[str, List[str]]):
        """Валидация операций"""
        # Проверка что хотя бы одна операция определена
        all_operations = []
        for op_type, op_list in operations.items():
            if isinstance(op_list, list):
                all_operations.extend(op_list)
        
        if not all_operations:
            self.errors.append("At least one operation must be defined")
        
        # TODO: Проверка валидности имен операций
    
    def _validate_frameworks(self, frameworks: Dict[str, List[Dict]]):
        """Валидация фреймворков"""
        if not frameworks.get('pandas') and not frameworks.get('polars'):
            self.errors.append("At least one framework must be defined")
        
        # Валидация конфигураций pandas
        for fw in frameworks.get('pandas', []):
            if not fw.get('name'):
                self.errors.append("Pandas framework missing 'name'")
            if fw.get('backend') not in ['numpy', 'pyarrow']:
                self.errors.append(
                    f"Invalid pandas backend: '{fw.get('backend')}'"
                )
        
        # Валидация конфигураций polars
        for fw in frameworks.get('polars', []):
            if not fw.get('name'):
                self.errors.append("Polars framework missing 'name'")
            if fw.get('api') not in ['eager', 'lazy']:
                self.errors.append(
                    f"Invalid polars api: '{fw.get('api')}'"
                )
    
    def _validate_output_section(self, output: Dict[str, Any]):
        """Валидация секции output"""
        # Проверка results_dir
        results_dir = output.get('results_dir')
        if not results_dir:
            self.errors.append("Output results_dir is required")
        
        # Проверка report_name
        report_name = output.get('report_name')
        if not report_name:
            self.errors.append("Output report_name is required")
        elif not report_name.endswith('.html'):
            self.warnings.append("Report name should end with '.html'")
    
    def _validate_logging_section(self, logging: Dict[str, Any]):
        """Валидация секции logging"""
        level = logging.get('level', 'INFO')
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR']
        if level not in valid_levels:
            self.errors.append(
                f"Invalid logging level '{level}'. "
                f"Must be one of: {valid_levels}"
            )


def validate_path(path: Path, must_exist: bool = False, 
                 must_be_dir: bool = False) -> bool:
    """
    Валидация пути к файлу или директории
    
    Args:
        path: Путь для проверки
        must_exist: Должен ли путь существовать
        must_be_dir: Должен ли путь быть директорией
        
    Returns:
        True если путь валиден
        
    Raises:
        ValidationError: При ошибках валидации
    """
    if must_exist and not path.exists():
        raise ValidationError(f"Path does not exist: {path}")
    
    if must_be_dir and path.exists() and not path.is_dir():
        raise ValidationError(f"Path is not a directory: {path}")
    
    # Проверка доступности для записи
    if path.exists():
        if not os.access(path, os.W_OK):
            raise ValidationError(f"Path is not writable: {path}")
    else:
        # Проверяем родительскую директорию
        parent = path.parent
        if not parent.exists():
            raise ValidationError(f"Parent directory does not exist: {parent}")
        if not os.access(parent, os.W_OK):
            raise ValidationError(f"Parent directory is not writable: {parent}")
    
    return True