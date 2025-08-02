"""
Dataclasses для конфигурации эксперимента
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union
from datetime import datetime
from pathlib import Path

from researches.polaris.experiment.src.core.constants import (
    DatasetSize,
    DatasetShape,
    DataTypeMix,
    Distribution,
    FileFormat,
    Framework,
    LogLevel,
    DEFAULT_SEED,
    DEFAULT_ITERATIONS,
    DEFAULT_WARMUP_ITERATIONS,
)


@dataclass
class NumericTypeConfig:
    """Конфигурация для числовых типов данных"""

    min: Union[int, float] = 0
    max: Union[int, float] = 100
    distribution: Distribution = Distribution.UNIFORM
    null_percentage: float = 0.0


@dataclass
class StringTypeConfig:
    """Конфигурация для строковых типов данных"""

    min_length: int
    max_length: int
    charset: str = "alphanumeric"


@dataclass
class CategoricalTypeConfig:
    """Конфигурация для категориальных типов данных"""

    cardinality: int


@dataclass
class TemporalTypeConfig:
    """Конфигурация для временных типов данных"""

    start: str
    end: str
    freq: Optional[str] = None
    format: Optional[str] = None


@dataclass
class BooleanTypeConfig:
    """Конфигурация для булевых типов данных"""

    true_ratio: float = 0.5
    null_percentage: float = 0.0


@dataclass
class ColumnTypesConfig:
    """Конфигурация всех типов колонок"""

    numeric: Dict[str, NumericTypeConfig] = field(default_factory=dict)
    string: Dict[str, StringTypeConfig] = field(default_factory=dict)
    categorical: Dict[str, CategoricalTypeConfig] = field(default_factory=dict)
    temporal: Dict[str, TemporalTypeConfig] = field(default_factory=dict)
    boolean: Dict[str, BooleanTypeConfig] = field(default_factory=dict)


@dataclass
class SizeConfig:
    """Конфигурация размера датасета"""

    name: str
    target_size_mb: int


@dataclass
class ShapeConfig:
    """Конфигурация формы датасета"""

    name: str
    columns: int
    description: str


@dataclass
class TypeMixConfig:
    """Конфигурация распределения типов данных"""

    name: str
    description: str
    composition: Dict[str, str]  # тип -> процент (например, "30%")


@dataclass
class DataGenerationConfig:
    """Конфигурация генерации данных"""

    seed: int = DEFAULT_SEED
    sizes: List[SizeConfig] = field(default_factory=list)
    shapes: List[ShapeConfig] = field(default_factory=list)
    type_mixes: List[TypeMixConfig] = field(default_factory=list)
    column_types: ColumnTypesConfig = field(default_factory=ColumnTypesConfig)


@dataclass
class OperationsConfig:
    """Конфигурация операций для бенчмарков"""

    read_operations: List[str] = field(default_factory=list)
    transform_operations: List[str] = field(default_factory=list)
    aggregation_operations: List[str] = field(default_factory=list)
    join_operations: List[str] = field(default_factory=list)
    sort_operations: List[str] = field(default_factory=list)
    string_operations: List[str] = field(default_factory=list)
    column_operations: List[str] = field(default_factory=list)


@dataclass
class FrameworkConfig:
    """Конфигурация фреймворка"""

    name: str
    backend: Optional[str] = None
    dtype_backend: Optional[str] = None
    api: Optional[str] = None
    streaming: bool = False


@dataclass
class FrameworksConfig:
    """Конфигурация всех фреймворков"""

    pandas: List[FrameworkConfig] = field(default_factory=list)
    polars: List[FrameworkConfig] = field(default_factory=list)


@dataclass
class BenchmarkConfig:
    """Конфигурация бенчмарков"""

    iterations: int = DEFAULT_ITERATIONS
    warmup_iterations: int = DEFAULT_WARMUP_ITERATIONS
    operations: OperationsConfig = field(default_factory=OperationsConfig)
    frameworks: FrameworksConfig = field(default_factory=FrameworksConfig)


@dataclass
class OutputConfig:
    """Конфигурация вывода результатов"""

    results_dir: str = "results"
    report_name: str = "benchmark_report.html"
    save_intermediate: bool = True
    include_memory_profiles: bool = True


@dataclass
class LoggingConfig:
    """Конфигурация логирования"""

    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    @property
    def log_level(self) -> LogLevel:
        """Преобразование строки в LogLevel enum"""
        return LogLevel[self.level.upper()]


@dataclass
class ExperimentInfo:
    """Информация об эксперименте"""

    name: str
    date: Union[str, datetime] = "auto"

    def __post_init__(self):
        """Автоматическая установка даты если указано 'auto'"""
        if self.date == "auto":
            self.date = datetime.now()


@dataclass
class ExperimentConfig:
    """Полная конфигурация эксперимента"""

    experiment: ExperimentInfo
    data_generation: DataGenerationConfig
    benchmarks: BenchmarkConfig
    output: OutputConfig
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def from_dict(cls, data: Dict) -> "ExperimentConfig":
        """Создание конфигурации из словаря"""
        # Преобразование вложенных словарей в dataclasses
        experiment = ExperimentInfo(**data.get("experiment", {}))

        # Data generation config
        dg_data = data.get("data_generation", {})
        sizes = [SizeConfig(**s) for s in dg_data.get("sizes", [])]
        shapes = [ShapeConfig(**s) for s in dg_data.get("shapes", [])]
        type_mixes = [TypeMixConfig(**tm) for tm in dg_data.get("type_mixes", [])]

        # Column types
        ct_data = dg_data.get("column_types", {})
        column_types = ColumnTypesConfig()

        # Numeric types
        for name, config in ct_data.get("numeric", {}).items():
            column_types.numeric[name] = NumericTypeConfig(**config)

        # String types
        for name, config in ct_data.get("string", {}).items():
            column_types.string[name] = StringTypeConfig(**config)

        # Categorical types
        for name, config in ct_data.get("categorical", {}).items():
            column_types.categorical[name] = CategoricalTypeConfig(**config)

        # Temporal types
        for name, config in ct_data.get("temporal", {}).items():
            column_types.temporal[name] = TemporalTypeConfig(**config)

        # Boolean types
        for name, config in ct_data.get("boolean", {}).items():
            column_types.boolean[name] = BooleanTypeConfig(**config)

        data_generation = DataGenerationConfig(
            seed=dg_data.get("seed", DEFAULT_SEED),
            sizes=sizes,
            shapes=shapes,
            type_mixes=type_mixes,
            column_types=column_types,
        )

        # Benchmarks config
        bench_data = data.get("benchmarks", {})
        operations = OperationsConfig(**bench_data.get("operations", {}))

        fw_data = bench_data.get("frameworks", {})
        frameworks = FrameworksConfig(
            pandas=[FrameworkConfig(**f) for f in fw_data.get("pandas", [])],
            polars=[FrameworkConfig(**f) for f in fw_data.get("polars", [])],
        )

        benchmarks = BenchmarkConfig(
            iterations=bench_data.get("iterations", DEFAULT_ITERATIONS),
            warmup_iterations=bench_data.get(
                "warmup_iterations", DEFAULT_WARMUP_ITERATIONS
            ),
            operations=operations,
            frameworks=frameworks,
        )

        # Output config
        output = OutputConfig(**data.get("output", {}))

        # Logging config
        logging = LoggingConfig(**data.get("logging", {}))

        return cls(
            experiment=experiment,
            data_generation=data_generation,
            benchmarks=benchmarks,
            output=output,
            logging=logging,
        )

    def validate(self) -> List[str]:
        """Валидация конфигурации, возвращает список ошибок"""
        errors = []

        # Проверка наличия размеров
        if not self.data_generation.sizes:
            errors.append("No dataset sizes defined")

        # Проверка наличия форм
        if not self.data_generation.shapes:
            errors.append("No dataset shapes defined")

        # Проверка наличия type mixes
        if not self.data_generation.type_mixes:
            errors.append("No type mixes defined")

        # Проверка операций
        all_ops = (
            self.benchmarks.operations.read_operations
            + self.benchmarks.operations.transform_operations
            + self.benchmarks.operations.aggregation_operations
            + self.benchmarks.operations.join_operations
            + self.benchmarks.operations.sort_operations
            + self.benchmarks.operations.string_operations
            + self.benchmarks.operations.column_operations
        )
        if not all_ops:
            errors.append("No operations defined for benchmarks")

        # Проверка фреймворков
        if (
            not self.benchmarks.frameworks.pandas
            and not self.benchmarks.frameworks.polars
        ):
            errors.append("No frameworks defined for benchmarks")

        # Проверка итераций
        if self.benchmarks.iterations < 1:
            errors.append("Iterations must be at least 1")

        # Проверка процентов в type mixes
        for tm in self.data_generation.type_mixes:
            total = 0
            for percentage in tm.composition.values():
                try:
                    value = float(percentage.rstrip("%"))
                    total += value
                except ValueError:
                    errors.append(
                        f"Invalid percentage format in type mix '{tm.name}': {percentage}"
                    )

            if abs(total - 100.0) > 0.01:  # Допуск на ошибки округления
                errors.append(
                    f"Type mix '{tm.name}' percentages sum to {total}%, expected 100%"
                )

        return errors
