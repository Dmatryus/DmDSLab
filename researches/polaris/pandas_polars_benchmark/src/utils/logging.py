"""
Модуль для настройки многоуровневого логирования с поддержкой цветного вывода и ротации.
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

# Проверяем наличие colorlog
try:
    import colorlog
    HAS_COLORLOG = True
except ImportError:
    HAS_COLORLOG = False
    print("Warning: colorlog not installed. Install it with: pip install colorlog")


class LoggerSetup:
    """Класс для настройки и управления логированием."""
    
    # Цветовая схема для разных уровней логирования
    COLOR_SCHEME = {
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    }
    
    # Форматы сообщений
    CONSOLE_FORMAT = '%(log_color)s%(asctime)s - %(name)-15s - %(levelname)-8s - %(message)s%(reset)s'
    FILE_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    SIMPLE_FORMAT = '%(log_color)s%(levelname)-8s%(reset)s %(message)s'
    
    def __init__(self, 
                 name: str = 'benchmark',
                 log_dir: Optional[Path] = None,
                 console_level: str = 'INFO',
                 file_level: str = 'DEBUG',
                 max_bytes: int = 10 * 1024 * 1024,  # 10MB
                 backup_count: int = 5,
                 use_colors: bool = True):
        """
        Инициализация системы логирования.
        
        Args:
            name: Имя логгера
            log_dir: Директория для сохранения логов
            console_level: Уровень логирования для консоли
            file_level: Уровень логирования для файла
            max_bytes: Максимальный размер файла лога перед ротацией
            backup_count: Количество файлов для хранения при ротации
            use_colors: Использовать цветной вывод в консоль
        """
        self.name = name
        self.log_dir = log_dir or Path('logs')
        self.console_level = getattr(logging, console_level.upper())
        self.file_level = getattr(logging, file_level.upper())
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.use_colors = use_colors and sys.stdout.isatty() and HAS_COLORLOG
        
        # Создаем директорию для логов
        self.log_dir.mkdir(exist_ok=True)
        
        # Настраиваем логгер
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """
        Настраивает и возвращает логгер.
        
        Returns:
            logging.Logger: Настроенный логгер
        """
        # Получаем или создаем логгер
        logger = logging.getLogger(self.name)
        logger.setLevel(logging.DEBUG)  # Устанавливаем минимальный уровень
        
        # Удаляем существующие обработчики (избегаем дублирования)
        logger.handlers.clear()
        
        # Добавляем консольный обработчик
        console_handler = self._create_console_handler()
        logger.addHandler(console_handler)
        
        # Добавляем файловый обработчик с ротацией
        file_handler = self._create_file_handler()
        logger.addHandler(file_handler)
        
        # Предотвращаем распространение логов к родительскому логгеру
        logger.propagate = False
        
        return logger
    
    def _create_console_handler(self) -> logging.Handler:
        """
        Создает обработчик для вывода в консоль с цветной подсветкой.
        
        Returns:
            logging.Handler: Консольный обработчик
        """
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.console_level)
        
        if self.use_colors and HAS_COLORLOG:
            # Используем colorlog для цветного вывода
            formatter = colorlog.ColoredFormatter(
                self.CONSOLE_FORMAT,
                datefmt='%Y-%m-%d %H:%M:%S',
                log_colors=self.COLOR_SCHEME,
                secondary_log_colors={},
                style='%'
            )
        else:
            # Обычный форматтер без цветов
            formatter = logging.Formatter(
                self.FILE_FORMAT.replace('%(log_color)s', '').replace('%(reset)s', ''),
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        
        console_handler.setFormatter(formatter)
        return console_handler
    
    def _create_file_handler(self) -> logging.Handler:
        """
        Создает обработчик для записи в файл с ротацией.
        
        Returns:
            logging.Handler: Файловый обработчик с ротацией
        """
        # Имя файла с текущей датой
        log_filename = self.log_dir / f"{self.name}_{datetime.now().strftime('%Y%m%d')}.log"
        
        # Создаем обработчик с ротацией по размеру
        file_handler = logging.handlers.RotatingFileHandler(
            filename=str(log_filename),  # Конвертируем Path в строку
            maxBytes=self.max_bytes,
            backupCount=self.backup_count,
            encoding='utf-8'
        )
        
        file_handler.setLevel(self.file_level)
        
        # Форматтер для файла (без цветов)
        formatter = logging.Formatter(
            self.FILE_FORMAT,
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_handler.setFormatter(formatter)
        return file_handler
    
    def get_logger(self, name: Optional[str] = None) -> logging.Logger:
        """
        Возвращает логгер (основной или дочерний).
        
        Args:
            name: Имя дочернего логгера. Если None, возвращает основной логгер.
            
        Returns:
            logging.Logger: Логгер
        """
        if name:
            return logging.getLogger(f"{self.name}.{name}")
        return self.logger
    
    def set_level(self, level: str, handler_type: str = 'both') -> None:
        """
        Изменяет уровень логирования.
        
        Args:
            level: Новый уровень логирования
            handler_type: Тип обработчика ('console', 'file', 'both')
        """
        level_value = getattr(logging, level.upper())
        
        for handler in self.logger.handlers:
            if handler_type == 'both':
                handler.setLevel(level_value)
            elif handler_type == 'console' and isinstance(handler, logging.StreamHandler):
                handler.setLevel(level_value)
            elif handler_type == 'file' and isinstance(handler, logging.handlers.RotatingFileHandler):
                handler.setLevel(level_value)
    
    def add_file_handler(self, filename: str, level: Optional[str] = None) -> None:
        """
        Добавляет дополнительный файловый обработчик.
        
        Args:
            filename: Имя файла для логов
            level: Уровень логирования для этого файла
        """
        filepath = self.log_dir / filename
        handler = logging.FileHandler(str(filepath), encoding='utf-8')
        
        if level:
            handler.setLevel(getattr(logging, level.upper()))
        else:
            handler.setLevel(self.file_level)
        
        formatter = logging.Formatter(
            self.FILE_FORMAT,
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        
        self.logger.addHandler(handler)


class BenchmarkLogger:
    """Специализированный логгер для бенчмарка с дополнительными методами."""
    
    def __init__(self, logger: logging.Logger):
        """
        Инициализация бенчмарк логгера.
        
        Args:
            logger: Базовый логгер
        """
        self.logger = logger
    
    def debug(self, msg: str, **kwargs) -> None:
        """Логирование отладочной информации."""
        self.logger.debug(msg, **kwargs)
    
    def info(self, msg: str, **kwargs) -> None:
        """Логирование информационных сообщений."""
        self.logger.info(msg, **kwargs)
    
    def warning(self, msg: str, **kwargs) -> None:
        """Логирование предупреждений."""
        self.logger.warning(msg, **kwargs)
    
    def error(self, msg: str, exc_info: bool = False, **kwargs) -> None:
        """Логирование ошибок."""
        self.logger.error(msg, exc_info=exc_info, **kwargs)
    
    def critical(self, msg: str, exc_info: bool = True, **kwargs) -> None:
        """Логирование критических ошибок."""
        self.logger.critical(msg, exc_info=exc_info, **kwargs)
    
    def benchmark_start(self, config: Dict[str, Any]) -> None:
        """Логирование начала бенчмарка."""
        self.info("=" * 80)
        self.info("НАЧАЛО БЕНЧМАРКА")
        self.info(f"Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.info(f"Конфигурация: {config.get('benchmark', {}).get('name', 'Unknown')}")
        self.info(f"Версия: {config.get('benchmark', {}).get('version', 'Unknown')}")
        self.info("=" * 80)
    
    def benchmark_end(self, success: bool, duration: float) -> None:
        """Логирование окончания бенчмарка."""
        self.info("=" * 80)
        status = "УСПЕШНО" if success else "С ОШИБКАМИ"
        self.info(f"БЕНЧМАРК ЗАВЕРШЕН {status}")
        self.info(f"Время выполнения: {duration:.2f} секунд")
        self.info(f"Время окончания: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.info("=" * 80)
    
    def operation_start(self, operation: str, library: str, dataset: str) -> None:
        """Логирование начала операции."""
        self.info(f"Начало операции: {operation} | Библиотека: {library} | Датасет: {dataset}")
    
    def operation_end(self, operation: str, success: bool, time: float, memory: float) -> None:
        """Логирование окончания операции."""
        if success:
            self.info(
                f"Операция завершена: {operation} | "
                f"Время: {time:.3f}с | Память: {memory:.1f}MB"
            )
        else:
            self.error(f"Операция завершилась с ошибкой: {operation}")
    
    def phase_start(self, phase: str) -> None:
        """Логирование начала фазы."""
        self.info(f"\n{'=' * 40}")
        self.info(f"НАЧАЛО ФАЗЫ: {phase}")
        self.info(f"{'=' * 40}")
    
    def phase_end(self, phase: str, success: bool = True) -> None:
        """Логирование окончания фазы."""
        status = "успешно" if success else "с ошибками"
        self.info(f"Фаза '{phase}' завершена {status}")
        self.info(f"{'=' * 40}\n")
    
    def progress(self, current: int, total: int, description: str = "") -> None:
        """Логирование прогресса выполнения."""
        percentage = (current / total) * 100
        msg = f"Прогресс: {current}/{total} ({percentage:.1f}%)"
        if description:
            msg += f" - {description}"
        self.info(msg)


# Глобальный экземпляр логгера
_logger_setup: Optional[LoggerSetup] = None
_benchmark_logger: Optional[BenchmarkLogger] = None


def setup_logging(name: str = 'benchmark',
                  log_dir: Optional[Path] = None,
                  console_level: str = 'INFO',
                  file_level: str = 'DEBUG',
                  use_colors: bool = True) -> BenchmarkLogger:
    """
    Настраивает и возвращает глобальный логгер для бенчмарка.
    
    Args:
        name: Имя логгера
        log_dir: Директория для логов
        console_level: Уровень для консоли
        file_level: Уровень для файла
        use_colors: Использовать цвета
        
    Returns:
        BenchmarkLogger: Настроенный логгер
    """
    global _logger_setup, _benchmark_logger
    
    _logger_setup = LoggerSetup(
        name=name,
        log_dir=log_dir,
        console_level=console_level,
        file_level=file_level,
        use_colors=use_colors
    )
    
    _benchmark_logger = BenchmarkLogger(_logger_setup.get_logger())
    return _benchmark_logger


def get_logger(name: Optional[str] = None) -> BenchmarkLogger:
    """
    Возвращает логгер (основной или дочерний).
    
    Args:
        name: Имя дочернего логгера
        
    Returns:
        BenchmarkLogger: Логгер
    """
    global _logger_setup, _benchmark_logger
    
    if _logger_setup is None:
        # Если логирование не настроено, настраиваем с параметрами по умолчанию
        setup_logging()
    
    if name:
        return BenchmarkLogger(_logger_setup.get_logger(name))
    
    return _benchmark_logger


def set_log_level(level: str, handler_type: str = 'both') -> None:
    """
    Изменяет уровень логирования.
    
    Args:
        level: Новый уровень
        handler_type: Тип обработчика
    """
    global _logger_setup
    
    if _logger_setup:
        _logger_setup.set_level(level, handler_type)


# Вспомогательные функции для прямого использования
def debug(msg: str, **kwargs) -> None:
    """Быстрый доступ к debug логированию."""
    get_logger().debug(msg, **kwargs)


def info(msg: str, **kwargs) -> None:
    """Быстрый доступ к info логированию."""
    get_logger().info(msg, **kwargs)


def warning(msg: str, **kwargs) -> None:
    """Быстрый доступ к warning логированию."""
    get_logger().warning(msg, **kwargs)


def error(msg: str, exc_info: bool = False, **kwargs) -> None:
    """Быстрый доступ к error логированию."""
    get_logger().error(msg, exc_info=exc_info, **kwargs)


def critical(msg: str, exc_info: bool = True, **kwargs) -> None:
    """Быстрый доступ к critical логированию."""
    get_logger().critical(msg, exc_info=exc_info, **kwargs)
