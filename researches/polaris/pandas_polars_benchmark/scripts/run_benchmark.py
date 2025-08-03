#!/usr/bin/env python3
"""
Основной скрипт для запуска системы бенчмаркинга Pandas vs Polars.

Использование:
    python run_benchmark.py --config configs/default_config.yaml
    python run_benchmark.py --resume
    python run_benchmark.py --config configs/small_test.yaml --dry-run
"""

import sys
import click
from pathlib import Path
from datetime import datetime

# Добавляем путь к src в PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from core.benchmark_runner import BenchmarkRunner, create_benchmark_runner
from utils.logging import setup_logging, get_logger


@click.command()
@click.option(
    '--config', '-c',
    type=click.Path(exists=True, path_type=Path),
    help='Путь к файлу конфигурации'
)
@click.option(
    '--resume', '-r',
    is_flag=True,
    help='Возобновить выполнение с последнего чекпоинта'
)
@click.option(
    '--output-dir', '-o',
    type=click.Path(path_type=Path),
    default='results',
    help='Директория для сохранения результатов'
)
@click.option(
    '--dry-run',
    is_flag=True,
    help='Только валидация без выполнения'
)
@click.option(
    '--validate-only',
    is_flag=True,
    help='Только валидация конфигурации'
)
@click.option(
    '--verbose', '-v',
    is_flag=True,
    help='Подробный вывод'
)
@click.option(
    '--quiet', '-q',
    is_flag=True,
    help='Минимальный вывод'
)
@click.option(
    '--force',
    is_flag=True,
    help='Принудительный запуск (игнорировать предупреждения)'
)
def main(config: Path,
         resume: bool,
         output_dir: Path,
         dry_run: bool,
         validate_only: bool,
         verbose: bool,
         quiet: bool,
         force: bool) -> None:
    """
    Запуск системы бенчмаркинга Pandas vs Polars.
    
    Система выполняет комплексное сравнение производительности двух библиотек
    на различных операциях и размерах данных.
    """
    # Определение уровня логирования
    if quiet:
        log_level = 'WARNING'
    elif verbose:
        log_level = 'DEBUG'
    else:
        log_level = 'INFO'
    
    # Настройка логирования для CLI
    setup_logging(console_level=log_level, use_colors=True)
    logger = get_logger('cli')
    
    # ASCII арт заголовок
    if not quiet:
        print("""
╔══════════════════════════════════════════════════════════════╗
║        PANDAS vs POLARS BENCHMARK SYSTEM v1.0                ║
╚══════════════════════════════════════════════════════════════╝
        """)
    
    # Обработка режима возобновления
    if resume:
        # Ищем последний чекпоинт
        checkpoint_dir = output_dir / "checkpoints"
        if not checkpoint_dir.exists():
            logger.error("Директория чекпоинтов не найдена")
            sys.exit(1)
        
        checkpoints = list(checkpoint_dir.glob("checkpoint_*.json"))
        if not checkpoints:
            logger.error("Чекпоинты не найдены")
            sys.exit(1)
        
        # Берем последний чекпоинт
        latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
        logger.info(f"Найден чекпоинт: {latest_checkpoint.name}")
        
        # Загружаем конфигурацию из чекпоинта
        import json
        with open(latest_checkpoint, 'r') as f:
            checkpoint_data = json.load(f)
        
        # Создаем временный файл конфигурации
        temp_config = output_dir / "temp_resume_config.yaml"
        # Здесь нужно восстановить конфигурацию из чекпоинта
        # Для простоты предполагаем, что config путь сохранен в чекпоинте
        
        if not config:
            logger.error("При возобновлении необходимо указать файл конфигурации")
            sys.exit(1)
    
    # Проверка обязательных параметров
    if not resume and not config:
        logger.error("Необходимо указать файл конфигурации (--config) или флаг возобновления (--resume)")
        sys.exit(1)
    
    # Только валидация
    if validate_only:
        logger.info("Режим валидации конфигурации")
        try:
            runner = create_benchmark_runner(
                config_path=config,
                output_dir=output_dir,
                dry_run=True
            )
            if runner._validate_environment():
                logger.info("✅ Валидация успешна")
                sys.exit(0)
            else:
                logger.error("❌ Валидация не пройдена")
                sys.exit(1)
        except Exception as e:
            logger.error(f"Ошибка валидации: {e}")
            sys.exit(1)
    
    # Проверка существующих результатов
    if not resume and output_dir.exists() and not force:
        existing_files = list(output_dir.glob("**/*"))
        if existing_files:
            logger.warning(f"Директория {output_dir} уже содержит файлы")
            if not dry_run:
                response = input("Продолжить и перезаписать? (y/n): ")
                if response.lower() != 'y':
                    logger.info("Отменено пользователем")
                    sys.exit(0)
    
    # Создание и запуск бенчмарка
    try:
        logger.info(f"Конфигурация: {config}")
        logger.info(f"Директория результатов: {output_dir}")
        logger.info(f"Режим: {'Возобновление' if resume else 'Новый запуск'}")
        
        if dry_run:
            logger.info("🔍 Режим DRY RUN - только проверка")
        
        # Создание runner
        runner = create_benchmark_runner(
            config_path=config,
            resume=resume,
            dry_run=dry_run,
            output_dir=output_dir
        )
        
        # Запуск
        start_time = datetime.now()
        success = runner.run()
        end_time = datetime.now()
        
        # Результаты
        if success:
            duration = end_time - start_time
            logger.info(f"\n✅ Бенчмарк завершен успешно!")
            logger.info(f"Общее время: {duration}")
            
            if not dry_run:
                logger.info(f"\n📊 Результаты сохранены в: {output_dir}")
                logger.info(f"📈 Отчет: {output_dir}/reports/")
                logger.info(f"📉 Анализ: {output_dir}/analysis/")
                logger.info(f"💾 Данные: {output_dir}/data/")
        else:
            logger.error("\n❌ Бенчмарк завершен с ошибками")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Прервано пользователем")
        logger.info("Состояние сохранено. Используйте --resume для продолжения")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n💥 Критическая ошибка: {e}", exc_info=verbose)
        sys.exit(1)


if __name__ == '__main__':
    main()
