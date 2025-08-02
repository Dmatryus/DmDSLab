#!/usr/bin/env python3
"""
Скрипт для валидации конфигурационного файла.
"""

import sys
import click
import yaml
from pathlib import Path
from typing import Optional

# Добавляем путь к src в PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from core.config import Config
from core.config_schema import ConfigSchema
from utils.logging import setup_logging, get_logger


@click.command()
@click.option(
    '--config', '-c',
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help='Путь к файлу конфигурации для валидации'
)
@click.option(
    '--schema', '-s',
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help='Путь к файлу схемы (опционально)'
)
@click.option(
    '--verbose', '-v',
    is_flag=True,
    help='Подробный вывод'
)
@click.option(
    '--create-example',
    is_flag=True,
    help='Создать пример конфигурации и выйти'
)
def validate_config(config: Path, 
                   schema: Optional[Path], 
                   verbose: bool,
                   create_example: bool) -> None:
    """
    Валидирует конфигурационный файл для системы бенчмаркинга.
    
    Проверяет структуру, типы данных и логическую согласованность параметров.
    """
    # Настраиваем логирование
    log_level = 'DEBUG' if verbose else 'INFO'
    setup_logging(console_level=log_level, use_colors=True)
    logger = get_logger()
    
    # Если запрошен пример конфигурации
    if create_example:
        logger.info("Создание примера конфигурации...")
        example_config = ConfigSchema.create_example_config()
        
        output_path = Path('example_config.yaml')
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(example_config, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"✅ Пример конфигурации создан: {output_path}")
        return
    
    logger.info(f"Валидация конфигурации: {config}")
    
    # Загружаем схему
    if schema:
        logger.info(f"Используется пользовательская схема: {schema}")
        config_schema = ConfigSchema.load_schema(schema)
    else:
        logger.info("Используется схема по умолчанию")
        config_schema = ConfigSchema()
    
    # Загружаем конфигурацию для валидации
    try:
        with open(config, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        logger.error(f"❌ Ошибка парсинга YAML: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Ошибка чтения файла: {e}")
        sys.exit(1)
    
    # Валидация структуры
    logger.info("Проверка структуры конфигурации...")
    structure_errors = config_schema.validate_structure(config_data)
    
    if structure_errors:
        logger.error("❌ Найдены ошибки структуры:")
        for error in structure_errors:
            logger.error(f"  - {error.message}")
            if verbose and error.path:
                logger.debug(f"    Путь: {' -> '.join(str(p) for p in error.path)}")
    else:
        logger.info("✅ Структура конфигурации корректна")
    
    # Валидация значений
    logger.info("Проверка значений конфигурации...")
    value_errors = config_schema.validate_values(config_data)
    
    if value_errors:
        logger.error("❌ Найдены ошибки значений:")
        for error in value_errors:
            logger.error(f"  - {error}")
    else:
        logger.info("✅ Значения конфигурации корректны")
    
    # Пробуем загрузить конфигурацию через основной класс
    logger.info("Загрузка конфигурации...")
    try:
        cfg = Config(config)
        logger.info("✅ Конфигурация успешно загружена")
        
        if verbose:
            # Выводим сводку по конфигурации
            logger.debug("\nСводка по конфигурации:")
            logger.debug(f"  Название: {cfg._config_data['benchmark']['name']}")
            logger.debug(f"  Версия: {cfg._config_data['benchmark']['version']}")
            logger.debug(f"  Размеры данных: {cfg.data_generation.sizes}")
            logger.debug(f"  Библиотеки:")
            for lib in cfg.libraries:
                if lib.enabled:
                    backends = f" (backends: {', '.join(lib.backends)})" if lib.backends else ""
                    logger.debug(f"    - {lib.name}{backends}")
            logger.debug(f"  Операции:")
            for category, ops in cfg.operations.items():
                logger.debug(f"    - {category}: {len(ops)} операций")
            logger.debug(f"  Профилирование:")
            logger.debug(f"    - Запуски: {cfg.profiling.min_runs}-{cfg.profiling.max_runs}")
            logger.debug(f"    - Целевой CV: {cfg.profiling.target_cv}")
            logger.debug(f"    - Таймаут: {cfg.profiling.timeout_seconds}с")
    
    except Exception as e:
        logger.error(f"❌ Ошибка загрузки конфигурации: {e}")
        if verbose:
            logger.exception("Детали ошибки:")
        sys.exit(1)
    
    # Итоговый результат
    total_errors = len(structure_errors) + len(value_errors)
    if total_errors == 0:
        logger.info("\n✅ Конфигурация полностью валидна и готова к использованию!")
        sys.exit(0)
    else:
        logger.error(f"\n❌ Найдено ошибок: {total_errors}")
        logger.error("Исправьте ошибки перед запуском бенчмарка.")
        sys.exit(1)


if __name__ == '__main__':
    validate_config()
