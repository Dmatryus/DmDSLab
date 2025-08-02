"""
Загрузчик и валидатор конфигурации
"""

import yaml
import json
import jsonschema
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging
from copy import deepcopy

from researches.polaris.experiment.src.core.config.models import ExperimentConfig
from ..constants import CONFIG_DIR, ERROR_MESSAGES
from ..exceptions import ConfigurationError


logger = logging.getLogger(__name__)


class ConfigLoader:
    """Загрузчик конфигурации из YAML файлов с валидацией"""

    def __init__(self, schema_path: Optional[Path] = None):
        """
        Инициализация загрузчика

        Args:
            schema_path: Путь к JSON schema для валидации
        """
        self.schema_path = schema_path or CONFIG_DIR / "schema.json"
        self._schema = None

    @property
    def schema(self) -> Dict:
        """Ленивая загрузка JSON schema"""
        if self._schema is None and self.schema_path.exists():
            with open(self.schema_path, "r") as f:
                self._schema = json.load(f)
        return self._schema

    def load_yaml(self, path: Union[str, Path]) -> ExperimentConfig:
        """
        Загрузка конфигурации из YAML файла

        Args:
            path: Путь к YAML файлу

        Returns:
            ExperimentConfig объект

        Raises:
            ConfigurationError: При ошибках загрузки или валидации
        """
        path = Path(path)

        if not path.exists():
            raise ConfigurationError(
                ERROR_MESSAGES["CONFIG_NOT_FOUND"].format(path=path)
            )

        logger.info(f"Loading configuration from {path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Failed to parse YAML: {e}")

        # Валидация против JSON schema если доступна
        if self.schema:
            self._validate_against_schema(data)

        # Создание объекта конфигурации
        try:
            config = ExperimentConfig.from_dict(data)
        except (TypeError, ValueError) as e:
            raise ConfigurationError(f"Invalid configuration structure: {e}")

        # Дополнительная валидация через методы dataclass
        errors = config.validate()
        if errors:
            raise ConfigurationError(
                ERROR_MESSAGES["INVALID_CONFIG"].format(error="; ".join(errors))
            )

        logger.info("Configuration loaded successfully")
        return config

    def _validate_against_schema(self, data: Dict):
        """
        Валидация данных против JSON schema

        Args:
            data: Данные для валидации

        Raises:
            ConfigurationError: При несоответствии schema
        """
        try:
            jsonschema.validate(instance=data, schema=self.schema)
        except jsonschema.ValidationError as e:
            raise ConfigurationError(f"Schema validation failed: {e.message}")
        except jsonschema.SchemaError as e:
            logger.warning(f"Schema itself is invalid: {e}")

    def validate_config(self, config: ExperimentConfig) -> bool:
        """
        Валидация объекта конфигурации

        Args:
            config: Объект конфигурации для валидации

        Returns:
            True если конфигурация валидна

        Raises:
            ConfigurationError: При ошибках валидации
        """
        errors = config.validate()
        if errors:
            raise ConfigurationError(
                ERROR_MESSAGES["INVALID_CONFIG"].format(error="; ".join(errors))
            )
        return True

    def merge_configs(
        self, configs: List[Union[Dict, ExperimentConfig]]
    ) -> ExperimentConfig:
        """
        Объединение нескольких конфигураций
        Последующие конфигурации перезаписывают предыдущие

        Args:
            configs: Список конфигураций для объединения

        Returns:
            Объединенная конфигурация
        """
        if not configs:
            raise ConfigurationError("No configurations provided for merging")

        # Начинаем с пустого словаря
        merged_data = {}

        for config in configs:
            if isinstance(config, ExperimentConfig):
                # Преобразуем обратно в словарь для мержинга
                config_data = self._config_to_dict(config)
            else:
                config_data = deepcopy(config)

            # Рекурсивное объединение
            merged_data = self._deep_merge(merged_data, config_data)

        # Создаем объект конфигурации из объединенных данных
        return ExperimentConfig.from_dict(merged_data)

    def _deep_merge(self, dict1: Dict, dict2: Dict) -> Dict:
        """
        Рекурсивное объединение двух словарей

        Args:
            dict1: Первый словарь (будет изменен)
            dict2: Второй словарь (приоритетный)

        Returns:
            Объединенный словарь
        """
        result = deepcopy(dict1)

        for key, value in dict2.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = deepcopy(value)

        return result

    def _config_to_dict(self, config: ExperimentConfig) -> Dict:
        """
        Преобразование ExperimentConfig обратно в словарь

        Args:
            config: Объект конфигурации

        Returns:
            Словарь с данными конфигурации
        """

        # Простая реализация через __dict__,
        # в реальном проекте может потребоваться более сложная логика
        def _dataclass_to_dict(obj):
            if hasattr(obj, "__dataclass_fields__"):
                result = {}
                for field_name in obj.__dataclass_fields__:
                    value = getattr(obj, field_name)
                    if hasattr(value, "__dataclass_fields__"):
                        result[field_name] = _dataclass_to_dict(value)
                    elif isinstance(value, list):
                        result[field_name] = [
                            (
                                _dataclass_to_dict(item)
                                if hasattr(item, "__dataclass_fields__")
                                else item
                            )
                            for item in value
                        ]
                    elif isinstance(value, dict):
                        result[field_name] = {
                            k: (
                                _dataclass_to_dict(v)
                                if hasattr(v, "__dataclass_fields__")
                                else v
                            )
                            for k, v in value.items()
                        }
                    else:
                        result[field_name] = value
                return result
            return obj

        return _dataclass_to_dict(config)

    def save_config(self, config: ExperimentConfig, path: Union[str, Path]):
        """
        Сохранение конфигурации в YAML файл

        Args:
            config: Объект конфигурации
            path: Путь для сохранения
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = self._config_to_dict(config)

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Configuration saved to {path}")

    @classmethod
    def create_default_config(cls) -> ExperimentConfig:
        """
        Создание конфигурации по умолчанию

        Returns:
            Конфигурация с базовыми настройками
        """
        from researches.polaris.experiment.src.core.config.models import (
            ExperimentInfo,
            DataGenerationConfig,
            BenchmarkConfig,
            OutputConfig,
            LoggingConfig,
            SizeConfig,
            ShapeConfig,
            TypeMixConfig,
            OperationsConfig,
            FrameworksConfig,
            FrameworkConfig,
        )

        return ExperimentConfig(
            experiment=ExperimentInfo(name="default_experiment"),
            data_generation=DataGenerationConfig(
                sizes=[
                    SizeConfig(name="small", target_size_mb=10),
                    SizeConfig(name="medium", target_size_mb=100),
                ],
                shapes=[
                    ShapeConfig(
                        name="narrow_long",
                        columns=10,
                        description="Few columns, many rows",
                    ),
                    ShapeConfig(
                        name="balanced", columns=50, description="Balanced shape"
                    ),
                ],
                type_mixes=[
                    TypeMixConfig(
                        name="numeric_heavy",
                        description="Mostly numeric columns",
                        composition={"int32": "40%", "float64": "40%", "string": "20%"},
                    )
                ],
            ),
            benchmarks=BenchmarkConfig(
                operations=OperationsConfig(
                    read_operations=["read_csv"],
                    transform_operations=["filter_numeric"],
                    aggregation_operations=["group_by_single"],
                ),
                frameworks=FrameworksConfig(
                    pandas=[FrameworkConfig(name="pandas_numpy", backend="numpy")],
                    polars=[FrameworkConfig(name="polars_eager", api="eager")],
                ),
            ),
            output=OutputConfig(),
            logging=LoggingConfig(),
        )
