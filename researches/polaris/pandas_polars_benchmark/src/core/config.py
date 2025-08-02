"""
Модуль для работы с конфигурацией системы бенчмаркинга.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
import jsonschema
from jsonschema import validate, ValidationError


@dataclass
class DataGenerationConfig:
    """Конфигурация для генерации данных."""
    sizes: List[int] = field(default_factory=lambda: [10000, 100000, 1000000])
    seed: int = 42
    
    # Numeric data config
    numeric_columns: int = 10
    numeric_dtypes: List[str] = field(default_factory=lambda: ["int64", "float64"])
    numeric_null_ratio: float = 0.05
    numeric_distributions: List[str] = field(default_factory=lambda: ["normal", "uniform"])
    
    # String data config
    string_columns: int = 5
    string_cardinality: List[int] = field(default_factory=lambda: [10, 100, 1000])
    string_null_ratio: float = 0.1
    string_length_range: tuple = (5, 50)
    
    # Datetime data config
    datetime_columns: int = 3
    datetime_frequency: str = "1min"
    datetime_start: str = "2020-01-01"
    datetime_tz: Optional[str] = None
    
    # Mixed data config
    mixed_numeric_columns: int = 5
    mixed_string_columns: int = 3
    mixed_datetime_columns: int = 2


@dataclass
class ProfilingConfig:
    """Конфигурация для профилирования."""
    min_runs: int = 3
    max_runs: int = 100
    target_cv: float = 0.05
    timeout_seconds: int = 300
    memory_sampling_interval: float = 0.1
    isolate_process: bool = True
    warmup_runs: int = 1


@dataclass
class ReportingConfig:
    """Конфигурация для генерации отчетов."""
    output_format: str = "html"
    include_raw_data: bool = True
    statistical_tests: List[str] = field(default_factory=lambda: ["normality_test", "paired_comparison"])
    confidence_level: float = 0.95
    chart_theme: str = "plotly"
    include_system_info: bool = True


@dataclass
class LibraryConfig:
    """Конфигурация для тестируемых библиотек."""
    name: str
    version: Optional[str] = None
    backends: List[str] = field(default_factory=list)
    enabled: bool = True


class Config:
    """Основной класс для управления конфигурацией."""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Инициализация конфигурации.
        
        Args:
            config_path: Путь к файлу конфигурации. Если не указан,
                        используется конфигурация по умолчанию.
        """
        self.config_path = Path(config_path) if config_path else None
        self._config_data: Dict[str, Any] = {}
        self._schema: Optional[Dict[str, Any]] = None
        
        # Подконфигурации
        self.data_generation: DataGenerationConfig = DataGenerationConfig()
        self.profiling: ProfilingConfig = ProfilingConfig()
        self.reporting: ReportingConfig = ReportingConfig()
        self.libraries: List[LibraryConfig] = []
        self.operations: Dict[str, List[str]] = {}
        
        # Загружаем конфигурацию
        if self.config_path:
            self.load_from_file(self.config_path)
        else:
            self._load_defaults()
    
    def load_from_file(self, path: Union[str, Path]) -> bool:
        """
        Загружает конфигурацию из YAML файла.
        
        Args:
            path: Путь к файлу конфигурации
            
        Returns:
            bool: True если загрузка успешна, False в случае ошибки
        """
        try:
            path = Path(path)
            if not path.exists():
                raise FileNotFoundError(f"Файл конфигурации не найден: {path}")
            
            with open(path, 'r', encoding='utf-8') as f:
                self._config_data = yaml.safe_load(f)
            
            # Парсим конфигурацию в структурированные объекты
            self._parse_config()
            
            # Валидируем если есть схема
            if self._schema:
                self.validate()
            
            return True
            
        except Exception as e:
            print(f"Ошибка при загрузке конфигурации: {e}")
            return False
    
    def load_schema(self, schema_path: Union[str, Path]) -> None:
        """
        Загружает JSON схему для валидации.
        
        Args:
            schema_path: Путь к файлу схемы
        """
        with open(schema_path, 'r', encoding='utf-8') as f:
            self._schema = yaml.safe_load(f)
    
    def validate(self) -> List[ValidationError]:
        """
        Валидирует конфигурацию согласно схеме.
        
        Returns:
            List[ValidationError]: Список ошибок валидации
        """
        errors = []
        
        if not self._schema:
            errors.append(ValidationError("Схема не загружена"))
            return errors
        
        try:
            validate(instance=self._config_data, schema=self._schema)
        except ValidationError as e:
            errors.append(e)
        
        # Дополнительная валидация значений
        errors.extend(self._validate_values())
        
        return errors
    
    def get_section(self, name: str) -> Dict[str, Any]:
        """
        Получает секцию конфигурации по имени.
        
        Args:
            name: Имя секции
            
        Returns:
            Dict: Данные секции
        """
        return self._config_data.get(name, {})
    
    def _parse_config(self) -> None:
        """Парсит raw конфигурацию в структурированные объекты."""
        # Data generation
        data_gen = self._config_data.get('data_generation', {})
        self.data_generation = DataGenerationConfig(
            sizes=data_gen.get('sizes', [10000, 100000, 1000000]),
            seed=data_gen.get('seed', 42),
            numeric_columns=data_gen.get('numeric', {}).get('columns', 10),
            numeric_dtypes=data_gen.get('numeric', {}).get('dtypes', ["int64", "float64"]),
            numeric_null_ratio=data_gen.get('numeric', {}).get('null_ratio', 0.05),
            string_columns=data_gen.get('string', {}).get('columns', 5),
            string_cardinality=data_gen.get('string', {}).get('cardinality', [10, 100, 1000]),
            string_null_ratio=data_gen.get('string', {}).get('null_ratio', 0.1),
            datetime_columns=data_gen.get('datetime', {}).get('columns', 3),
            datetime_frequency=data_gen.get('datetime', {}).get('frequency', '1min'),
            mixed_numeric_columns=data_gen.get('mixed', {}).get('numeric_columns', 5),
            mixed_string_columns=data_gen.get('mixed', {}).get('string_columns', 3),
            mixed_datetime_columns=data_gen.get('mixed', {}).get('datetime_columns', 2)
        )
        
        # Profiling
        prof = self._config_data.get('profiling', {})
        self.profiling = ProfilingConfig(
            min_runs=prof.get('min_runs', 3),
            max_runs=prof.get('max_runs', 100),
            target_cv=prof.get('target_cv', 0.05),
            timeout_seconds=prof.get('timeout_seconds', 300),
            memory_sampling_interval=prof.get('memory_sampling_interval', 0.1),
            isolate_process=prof.get('isolate_process', True),
            warmup_runs=prof.get('warmup_runs', 1)
        )
        
        # Reporting
        rep = self._config_data.get('reporting', {})
        self.reporting = ReportingConfig(
            output_format=rep.get('output_format', 'html'),
            include_raw_data=rep.get('include_raw_data', True),
            statistical_tests=rep.get('statistical_tests', ['normality_test', 'paired_comparison']),
            confidence_level=rep.get('confidence_level', 0.95),
            chart_theme=rep.get('chart_theme', 'plotly'),
            include_system_info=rep.get('include_system_info', True)
        )
        
        # Libraries
        libs = self._config_data.get('environment', {}).get('libraries', {})
        self.libraries = []
        
        # Pandas
        if 'pandas' in libs:
            pandas_config = libs['pandas']
            self.libraries.append(LibraryConfig(
                name='pandas',
                version=pandas_config.get('version'),
                backends=pandas_config.get('backends', ['numpy']),
                enabled=pandas_config.get('enabled', True)
            ))
        
        # Polars
        if 'polars' in libs:
            polars_config = libs['polars']
            self.libraries.append(LibraryConfig(
                name='polars',
                version=polars_config.get('version'),
                backends=[],  # Polars не имеет backends
                enabled=polars_config.get('enabled', True)
            ))
        
        # Operations
        self.operations = self._config_data.get('operations', {})
    
    def _validate_values(self) -> List[ValidationError]:
        """Дополнительная валидация значений конфигурации."""
        errors = []
        
        # Проверяем размеры данных
        if any(size <= 0 for size in self.data_generation.sizes):
            errors.append(ValidationError("Размеры данных должны быть положительными"))
        
        # Проверяем параметры профилирования
        if self.profiling.min_runs > self.profiling.max_runs:
            errors.append(ValidationError("min_runs не может быть больше max_runs"))
        
        if not 0 < self.profiling.target_cv < 1:
            errors.append(ValidationError("target_cv должен быть между 0 и 1"))
        
        # Проверяем null ratios
        if not 0 <= self.data_generation.numeric_null_ratio <= 1:
            errors.append(ValidationError("numeric_null_ratio должен быть между 0 и 1"))
            
        if not 0 <= self.data_generation.string_null_ratio <= 1:
            errors.append(ValidationError("string_null_ratio должен быть между 0 и 1"))
        
        # Проверяем, что хотя бы одна библиотека включена
        if not any(lib.enabled for lib in self.libraries):
            errors.append(ValidationError("Хотя бы одна библиотека должна быть включена"))
        
        # Проверяем операции
        if not self.operations:
            errors.append(ValidationError("Должна быть определена хотя бы одна операция"))
        
        return errors
    
    def _load_defaults(self) -> None:
        """Загружает конфигурацию по умолчанию."""
        self._config_data = {
            'benchmark': {
                'name': 'Pandas vs Polars Benchmark',
                'version': '1.0.0'
            },
            'environment': {
                'libraries': {
                    'pandas': {
                        'backends': ['numpy', 'pyarrow'],
                        'enabled': True
                    },
                    'polars': {
                        'enabled': True
                    }
                }
            },
            'data_generation': {
                'sizes': [10000, 100000, 1000000],
                'seed': 42
            },
            'operations': {
                'io': ['read_csv', 'read_parquet'],
                'filter': ['simple_filter', 'complex_filter'],
                'groupby': ['single_column_groupby', 'multi_column_groupby'],
                'sort': ['single_column_sort'],
                'join': ['inner_join']
            }
        }
        self._parse_config()
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертирует конфигурацию обратно в словарь."""
        return self._config_data
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Сохраняет конфигурацию в файл.
        
        Args:
            path: Путь для сохранения
        """
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(self._config_data, f, default_flow_style=False, sort_keys=False)
