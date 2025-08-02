"""
Модуль для валидации конфигурации с использованием JSON Schema.
"""

from typing import Dict, List, Any, Optional
from pathlib import Path
import yaml
import jsonschema
from jsonschema import Draft7Validator, ValidationError


class ConfigSchema:
    """Класс для валидации конфигурации согласно схеме."""
    
    # JSON Schema для валидации структуры конфигурации
    DEFAULT_SCHEMA = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "required": ["benchmark", "environment", "data_generation", "operations"],
        "properties": {
            "benchmark": {
                "type": "object",
                "required": ["name", "version"],
                "properties": {
                    "name": {"type": "string"},
                    "version": {"type": "string", "pattern": "^\\d+\\.\\d+\\.\\d+$"}
                }
            },
            "environment": {
                "type": "object",
                "required": ["libraries"],
                "properties": {
                    "python_version": {"type": "string"},
                    "libraries": {
                        "type": "object",
                        "properties": {
                            "pandas": {
                                "type": "object",
                                "properties": {
                                    "version": {"type": "string"},
                                    "backends": {
                                        "type": "array",
                                        "items": {"type": "string", "enum": ["numpy", "pyarrow"]},
                                        "minItems": 1
                                    },
                                    "enabled": {"type": "boolean", "default": True}
                                }
                            },
                            "polars": {
                                "type": "object",
                                "properties": {
                                    "version": {"type": "string"},
                                    "enabled": {"type": "boolean", "default": True}
                                }
                            }
                        },
                        "minProperties": 1
                    }
                }
            },
            "data_generation": {
                "type": "object",
                "required": ["sizes"],
                "properties": {
                    "sizes": {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 1},
                        "minItems": 1
                    },
                    "seed": {"type": "integer", "default": 42},
                    "types": {
                        "type": "object",
                        "properties": {
                            "numeric": {
                                "type": "object",
                                "properties": {
                                    "columns": {"type": "integer", "minimum": 1},
                                    "dtypes": {
                                        "type": "array",
                                        "items": {"type": "string", "enum": ["int32", "int64", "float32", "float64"]}
                                    },
                                    "null_ratio": {"type": "number", "minimum": 0, "maximum": 1},
                                    "distributions": {
                                        "type": "array",
                                        "items": {"type": "string", "enum": ["normal", "uniform", "exponential"]}
                                    }
                                }
                            },
                            "string": {
                                "type": "object",
                                "properties": {
                                    "columns": {"type": "integer", "minimum": 1},
                                    "cardinality": {
                                        "type": "array",
                                        "items": {"type": "integer", "minimum": 1}
                                    },
                                    "null_ratio": {"type": "number", "minimum": 0, "maximum": 1},
                                    "length_range": {
                                        "type": "array",
                                        "items": {"type": "integer", "minimum": 1},
                                        "minItems": 2,
                                        "maxItems": 2
                                    }
                                }
                            },
                            "datetime": {
                                "type": "object",
                                "properties": {
                                    "columns": {"type": "integer", "minimum": 1},
                                    "frequency": {"type": "string"},
                                    "start": {"type": "string", "format": "date"},
                                    "timezone": {"type": ["string", "null"]}
                                }
                            },
                            "mixed": {
                                "type": "object",
                                "properties": {
                                    "numeric_columns": {"type": "integer", "minimum": 0},
                                    "string_columns": {"type": "integer", "minimum": 0},
                                    "datetime_columns": {"type": "integer", "minimum": 0}
                                }
                            }
                        }
                    }
                }
            },
            "operations": {
                "type": "object",
                "minProperties": 1,
                "additionalProperties": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1
                }
            },
            "profiling": {
                "type": "object",
                "properties": {
                    "min_runs": {"type": "integer", "minimum": 1},
                    "max_runs": {"type": "integer", "minimum": 1},
                    "target_cv": {"type": "number", "minimum": 0, "maximum": 1},
                    "timeout_seconds": {"type": "integer", "minimum": 1},
                    "memory_sampling_interval": {"type": "number", "minimum": 0.01},
                    "isolate_process": {"type": "boolean", "default": True},
                    "warmup_runs": {"type": "integer", "minimum": 0}
                }
            },
            "reporting": {
                "type": "object",
                "properties": {
                    "output_format": {"type": "string", "enum": ["html", "json", "csv"]},
                    "include_raw_data": {"type": "boolean"},
                    "statistical_tests": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["normality_test", "paired_comparison", "mann_whitney", "wilcoxon"]
                        }
                    },
                    "confidence_level": {"type": "number", "minimum": 0, "maximum": 1},
                    "chart_theme": {"type": "string"},
                    "include_system_info": {"type": "boolean"}
                }
            }
        }
    }
    
    def __init__(self, schema: Optional[Dict[str, Any]] = None):
        """
        Инициализация валидатора схемы.
        
        Args:
            schema: JSON схема для валидации. Если не указана, используется схема по умолчанию.
        """
        self.schema = schema or self.DEFAULT_SCHEMA
        self.validator = Draft7Validator(self.schema)
    
    def validate_structure(self, data: Dict[str, Any]) -> List[ValidationError]:
        """
        Валидирует структуру конфигурации согласно схеме.
        
        Args:
            data: Данные конфигурации для валидации
            
        Returns:
            List[ValidationError]: Список ошибок валидации структуры
        """
        errors = []
        for error in self.validator.iter_errors(data):
            errors.append(error)
        return errors
    
    def validate_values(self, data: Dict[str, Any]) -> List[ValidationError]:
        """
        Валидирует значения конфигурации (дополнительные проверки).
        
        Args:
            data: Данные конфигурации для валидации
            
        Returns:
            List[ValidationError]: Список ошибок валидации значений
        """
        errors = []
        
        # Проверка согласованности параметров профилирования
        profiling = data.get('profiling', {})
        min_runs = profiling.get('min_runs', 1)
        max_runs = profiling.get('max_runs', 100)
        
        if min_runs > max_runs:
            errors.append(ValidationError(
                f"profiling.min_runs ({min_runs}) не может быть больше "
                f"profiling.max_runs ({max_runs})"
            ))
        
        # Проверка, что хотя бы одна библиотека включена
        libraries = data.get('environment', {}).get('libraries', {})
        enabled_libs = []
        
        for lib_name, lib_config in libraries.items():
            if lib_config.get('enabled', True):
                enabled_libs.append(lib_name)
        
        if not enabled_libs:
            errors.append(ValidationError(
                "Хотя бы одна библиотека должна быть включена (enabled: true)"
            ))
        
        # Проверка размеров данных
        sizes = data.get('data_generation', {}).get('sizes', [])
        if sizes and len(set(sizes)) != len(sizes):
            errors.append(ValidationError(
                "data_generation.sizes содержит дублирующиеся значения"
            ))
        
        # Проверка операций
        operations = data.get('operations', {})
        total_operations = sum(len(ops) for ops in operations.values())
        
        if total_operations == 0:
            errors.append(ValidationError(
                "Должна быть определена хотя бы одна операция"
            ))
        
        # Проверка уникальности операций внутри категорий
        for category, ops in operations.items():
            if len(set(ops)) != len(ops):
                errors.append(ValidationError(
                    f"operations.{category} содержит дублирующиеся операции"
                ))
        
        # Проверка типов данных для генерации
        types = data.get('data_generation', {}).get('types', {})
        if types:
            # Проверка, что для mixed типа сумма колонок > 0
            mixed = types.get('mixed', {})
            if mixed:
                total_cols = (mixed.get('numeric_columns', 0) + 
                            mixed.get('string_columns', 0) + 
                            mixed.get('datetime_columns', 0))
                if total_cols == 0:
                    errors.append(ValidationError(
                        "Для mixed типа данных должна быть хотя бы одна колонка"
                    ))
            
            # Проверка length_range для строк
            string_config = types.get('string', {})
            length_range = string_config.get('length_range', [])
            if length_range and len(length_range) == 2:
                if length_range[0] > length_range[1]:
                    errors.append(ValidationError(
                        f"string.length_range: минимальная длина ({length_range[0]}) "
                        f"не может быть больше максимальной ({length_range[1]})"
                    ))
        
        return errors
    
    def check_required_fields(self, data: Dict[str, Any]) -> bool:
        """
        Проверяет наличие всех обязательных полей.
        
        Args:
            data: Данные конфигурации
            
        Returns:
            bool: True если все обязательные поля присутствуют
        """
        return len(list(self.validator.iter_errors(data))) == 0
    
    def check_data_types(self, data: Dict[str, Any]) -> bool:
        """
        Проверяет корректность типов данных.
        
        Args:
            data: Данные конфигурации
            
        Returns:
            bool: True если все типы данных корректны
        """
        try:
            self.validator.validate(data)
            return True
        except ValidationError:
            return False
    
    def get_defaults(self) -> Dict[str, Any]:
        """
        Возвращает значения по умолчанию из схемы.
        
        Returns:
            Dict: Словарь с значениями по умолчанию
        """
        defaults = {}
        
        def extract_defaults(schema_part: Dict[str, Any], path: str = "") -> None:
            if isinstance(schema_part, dict):
                if 'default' in schema_part:
                    defaults[path] = schema_part['default']
                
                if 'properties' in schema_part:
                    for prop, prop_schema in schema_part['properties'].items():
                        new_path = f"{path}.{prop}" if path else prop
                        extract_defaults(prop_schema, new_path)
        
        extract_defaults(self.schema)
        return defaults
    
    @classmethod
    def create_example_config(cls) -> Dict[str, Any]:
        """
        Создает пример валидной конфигурации.
        
        Returns:
            Dict: Пример конфигурации
        """
        return {
            "benchmark": {
                "name": "Pandas vs Polars Performance Comparison",
                "version": "1.0.0"
            },
            "environment": {
                "python_version": "3.11+",
                "libraries": {
                    "pandas": {
                        "version": "latest",
                        "backends": ["numpy", "pyarrow"],
                        "enabled": True
                    },
                    "polars": {
                        "version": "latest",
                        "enabled": True
                    }
                }
            },
            "data_generation": {
                "sizes": [10000, 100000, 1000000],
                "seed": 42,
                "types": {
                    "numeric": {
                        "columns": 10,
                        "dtypes": ["int64", "float64"],
                        "null_ratio": 0.05,
                        "distributions": ["normal", "uniform"]
                    },
                    "string": {
                        "columns": 5,
                        "cardinality": [10, 100, 1000],
                        "null_ratio": 0.1,
                        "length_range": [5, 50]
                    },
                    "datetime": {
                        "columns": 3,
                        "frequency": "1min",
                        "start": "2020-01-01",
                        "timezone": None
                    },
                    "mixed": {
                        "numeric_columns": 5,
                        "string_columns": 3,
                        "datetime_columns": 2
                    }
                }
            },
            "operations": {
                "io": ["read_csv", "read_parquet", "write_csv", "write_parquet"],
                "filter": ["simple_filter", "complex_filter", "isin_filter"],
                "groupby": ["single_column_groupby", "multi_column_groupby"],
                "sort": ["single_column_sort", "multi_column_sort"],
                "join": ["inner_join", "left_join"],
                "string": ["concatenation", "contains", "regex_extract"]
            },
            "profiling": {
                "min_runs": 3,
                "max_runs": 100,
                "target_cv": 0.05,
                "timeout_seconds": 300,
                "memory_sampling_interval": 0.1,
                "isolate_process": True,
                "warmup_runs": 1
            },
            "reporting": {
                "output_format": "html",
                "include_raw_data": True,
                "statistical_tests": ["normality_test", "paired_comparison"],
                "confidence_level": 0.95,
                "chart_theme": "plotly",
                "include_system_info": True
            }
        }
    
    def save_schema(self, path: Path) -> None:
        """
        Сохраняет схему в файл.
        
        Args:
            path: Путь для сохранения схемы
        """
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(self.schema, f, default_flow_style=False, sort_keys=False)
    
    @classmethod
    def load_schema(cls, path: Path) -> 'ConfigSchema':
        """
        Загружает схему из файла.
        
        Args:
            path: Путь к файлу схемы
            
        Returns:
            ConfigSchema: Экземпляр валидатора с загруженной схемой
        """
        with open(path, 'r', encoding='utf-8') as f:
            schema = yaml.safe_load(f)
        return cls(schema)
