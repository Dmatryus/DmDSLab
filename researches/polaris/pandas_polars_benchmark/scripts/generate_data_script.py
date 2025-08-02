#!/usr/bin/env python3
"""
Скрипт для генерации синтетических данных согласно конфигурации.
"""

import sys
import click
from pathlib import Path
import time

# Добавляем путь к src в PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from core.config import Config
from data import DataGenerator
from utils.logging import setup_logging, get_logger


@click.command()
@click.option(
    '--config', '-c',
    type=click.Path(exists=True, path_type=Path),
    default='configs/default_config.yaml',
    help='Путь к файлу конфигурации'
)
@click.option(
    '--output-dir', '-o',
    type=click.Path(path_type=Path),
    default=None,
    help='Директория для сохранения данных (по умолчанию из конфигурации)'
)
@click.option(
    '--force', '-f',
    is_flag=True,
    help='Перезаписать существующие данные'
)
@click.option(
    '--sizes',
    type=str,
    default=None,
    help='Размеры датасетов через запятую (например: 1000,10000,100000)'
)
@click.option(
    '--verbose', '-v',
    is_flag=True,
    help='Подробный вывод'
)
def generate_data(config: Path, output_dir: Path, force: bool, sizes: str, verbose: bool):
    """
    Генерирует синтетические данные для бенчмаркинга Pandas vs Polars.
    
    Создает датасеты различных типов (numeric, string, datetime, mixed) 
    и размеров согласно конфигурации.
    """
    # Настраиваем логирование
    log_level = 'DEBUG' if verbose else 'INFO'
    logger = setup_logging(
        name='data_generation',
        console_level=log_level,
        use_colors=True
    )
    
    start_time = time.time()
    
    logger.info("=" * 80)
    logger.info("ГЕНЕРАЦИЯ ДАННЫХ ДЛЯ БЕНЧМАРКА")
    logger.info("=" * 80)
    
    # Загружаем конфигурацию
    logger.info(f"Загрузка конфигурации: {config}")
    cfg = Config(config)
    
    # Если указан output_dir, используем его
    if output_dir:
        data_dir = output_dir
    else:
        data_dir = Path(cfg.get_section('paths').get('data_dir', 'data/generated'))
    
    logger.info(f"Директория для данных: {data_dir}")
    
    # Переопределяем размеры если указаны
    if sizes:
        try:
            custom_sizes = [int(s.strip()) for s in sizes.split(',')]
            cfg.data_generation.sizes = custom_sizes
            logger.info(f"Используются пользовательские размеры: {custom_sizes}")
        except ValueError as e:
            logger.error(f"Ошибка парсинга размеров: {e}")
            sys.exit(1)
    
    # Проверяем существующие данные
    if data_dir.exists() and not force:
        existing_files = list(data_dir.rglob('*.csv')) + list(data_dir.rglob('*.parquet'))
        if existing_files:
            logger.warning(f"В директории {data_dir} уже есть {len(existing_files)} файлов")
            if not click.confirm("Продолжить и перезаписать существующие данные?"):
                logger.info("Генерация отменена")
                return
    
    # Выводим параметры генерации
    logger.info("\nПараметры генерации:")
    logger.info(f"  Размеры: {cfg.data_generation.sizes}")
    logger.info(f"  Seed: {cfg.data_generation.seed}")
    logger.info(f"  Типы данных:")
    logger.info(f"    - Numeric: {cfg.data_generation.numeric_columns} колонок")
    logger.info(f"    - String: {cfg.data_generation.string_columns} колонок")
    logger.info(f"    - Datetime: {cfg.data_generation.datetime_columns} колонок")
    logger.info(f"    - Mixed: {cfg.data_generation.mixed_numeric_columns} numeric, "
                f"{cfg.data_generation.mixed_string_columns} string, "
                f"{cfg.data_generation.mixed_datetime_columns} datetime")
    
    # Создаем генератор
    generator = DataGenerator(
        config=cfg.data_generation,
        output_dir=data_dir
    )
    
    try:
        # Генерируем все датасеты
        datasets = generator.generate_all_datasets()
        
        # Выводим сводку
        logger.info("\n" + "=" * 80)
        logger.info("СВОДКА ГЕНЕРАЦИИ")
        logger.info("=" * 80)
        
        total_size_mb = 0
        total_files = 0
        
        for ds in datasets:
            csv_size = ds.file_paths['csv'].stat().st_size / (1024 * 1024)
            parquet_size = ds.file_paths['parquet'].stat().st_size / (1024 * 1024)
            total_size_mb += csv_size + parquet_size
            total_files += 2
            
            logger.info(f"\n📊 {ds.name}:")
            logger.info(f"   Строк: {ds.size:,}")
            logger.info(f"   Колонок: {len(ds.columns)}")
            logger.info(f"   В памяти: {ds.memory_size_mb:.1f} MB")
            logger.info(f"   CSV: {csv_size:.1f} MB")
            logger.info(f"   Parquet: {parquet_size:.1f} MB")
            logger.info(f"   Время генерации: {ds.generation_time_sec:.2f}с")
        
        elapsed_time = time.time() - start_time
        
        logger.info(f"\n{'=' * 40}")
        logger.info(f"✅ ГЕНЕРАЦИЯ ЗАВЕРШЕНА УСПЕШНО!")
        logger.info(f"{'=' * 40}")
        logger.info(f"Всего датасетов: {len(datasets)}")
        logger.info(f"Всего файлов: {total_files}")
        logger.info(f"Общий размер: {total_size_mb:.1f} MB")
        logger.info(f"Время выполнения: {elapsed_time:.1f}с")
        logger.info(f"Данные сохранены в: {data_dir.absolute()}")
        
    except Exception as e:
        logger.error(f"❌ Ошибка при генерации данных: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    generate_data()
