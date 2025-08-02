"""
Модуль для управления чекпоинтами и восстановления состояния бенчмарка.
"""

import json
import pickle
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Set, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
import hashlib
import os
import sys

from researches.polaris.pandas_polars_benchmark.src.utils.logging import get_logger
from researches.polaris.pandas_polars_benchmark.src.profiling import ProfileResult


@dataclass
class BenchmarkState:
    """Состояние выполнения бенчмарка."""
    # Идентификация запуска
    run_id: str
    start_time: str
    config_hash: str
    
    # Прогресс выполнения
    total_operations: int = 0
    completed_operations: int = 0
    failed_operations: int = 0
    skipped_operations: int = 0
    
    # Детали выполнения
    completed_tasks: Set[str] = field(default_factory=set)
    failed_tasks: Dict[str, str] = field(default_factory=dict)  # task_id -> error
    results: Dict[str, ProfileResult] = field(default_factory=dict)  # task_id -> result
    
    # Метаданные
    last_update: str = field(default_factory=lambda: datetime.now().isoformat())
    checkpoint_version: str = "1.0.0"
    system_info: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def progress_percentage(self) -> float:
        """Процент выполнения."""
        if self.total_operations == 0:
            return 0.0
        return (self.completed_operations / self.total_operations) * 100
    
    @property
    def is_complete(self) -> bool:
        """Проверка завершенности бенчмарка."""
        return self.completed_operations >= self.total_operations
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертирует в словарь для сериализации."""
        data = asdict(self)
        # Конвертируем set в list для JSON
        data['completed_tasks'] = list(self.completed_tasks)
        # Конвертируем ProfileResult объекты в словари
        data['results'] = {k: v.to_dict() for k, v in self.results.items()}
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BenchmarkState':
        """Создает из словаря."""
        # Конвертируем list обратно в set
        if 'completed_tasks' in data:
            data['completed_tasks'] = set(data['completed_tasks'])
        # Конвертируем словари обратно в ProfileResult
        if 'results' in data:
            results = {}
            for k, v in data['results'].items():
                # Создаем ProfileResult из словаря
                result = ProfileResult(**v)
                results[k] = result
            data['results'] = results
        return cls(**data)


class CheckpointManager:
    """Менеджер для управления чекпоинтами."""
    
    def __init__(self, 
                 checkpoint_dir: Optional[Path] = None,
                 checkpoint_interval: int = 10,
                 max_checkpoints: int = 5):
        """
        Инициализация менеджера чекпоинтов.
        
        Args:
            checkpoint_dir: Директория для сохранения чекпоинтов
            checkpoint_interval: Интервал автосохранения (количество операций)
            max_checkpoints: Максимальное количество хранимых чекпоинтов
        """
        self.checkpoint_dir = checkpoint_dir or Path("results/checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_interval = checkpoint_interval
        self.max_checkpoints = max_checkpoints
        
        self.logger = get_logger('checkpoint_manager')
        
        # Текущее состояние
        self.current_state: Optional[BenchmarkState] = None
        self.operations_since_checkpoint = 0
        
        # Файл блокировки для предотвращения одновременного доступа
        self.lock_file = self.checkpoint_dir / '.checkpoint.lock'
    
    def initialize_state(self, 
                        run_id: str,
                        config: Dict[str, Any],
                        total_operations: int) -> BenchmarkState:
        """
        Инициализирует новое состояние бенчмарка.
        
        Args:
            run_id: Уникальный идентификатор запуска
            config: Конфигурация бенчмарка
            total_operations: Общее количество операций
            
        Returns:
            BenchmarkState: Инициализированное состояние
        """
        # Создаем хеш конфигурации для проверки совместимости
        config_str = json.dumps(config, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()
        
        # Собираем информацию о системе
        import platform
        system_info = {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'processor': platform.processor(),
            'cpu_count': os.cpu_count()
        }
        
        self.current_state = BenchmarkState(
            run_id=run_id,
            start_time=datetime.now().isoformat(),
            config_hash=config_hash,
            total_operations=total_operations,
            system_info=system_info
        )
        
        self.logger.info(
            f"Инициализировано состояние: run_id={run_id}, "
            f"total_operations={total_operations}"
        )
        
        # Сохраняем начальный чекпоинт
        self.save_checkpoint()
        
        return self.current_state
    
    def load_checkpoint(self, checkpoint_file: Optional[Path] = None) -> Optional[BenchmarkState]:
        """
        Загружает чекпоинт из файла.
        
        Args:
            checkpoint_file: Путь к файлу чекпоинта (если None, ищет последний)
            
        Returns:
            BenchmarkState или None если не найден
        """
        try:
            if checkpoint_file is None:
                # Ищем последний чекпоинт
                checkpoint_file = self._find_latest_checkpoint()
                if checkpoint_file is None:
                    self.logger.info("Чекпоинты не найдены")
                    return None
            
            self.logger.info(f"Загрузка чекпоинта: {checkpoint_file}")
            
            # Загружаем с блокировкой
            with self._file_lock(checkpoint_file, 'r') as f:
                data = json.load(f)
            
            # Восстанавливаем состояние
            self.current_state = BenchmarkState.from_dict(data)
            
            self.logger.info(
                f"Чекпоинт загружен: run_id={self.current_state.run_id}, "
                f"progress={self.current_state.progress_percentage:.1f}%"
            )
            
            return self.current_state
            
        except Exception as e:
            self.logger.error(f"Ошибка загрузки чекпоинта: {e}")
            return None
    
    def save_checkpoint(self, force: bool = False) -> bool:
        """
        Сохраняет текущее состояние в чекпоинт.
        
        Args:
            force: Принудительное сохранение независимо от интервала
            
        Returns:
            bool: True если сохранено успешно
        """
        if self.current_state is None:
            self.logger.warning("Нет состояния для сохранения")
            return False
        
        # Проверяем интервал
        if not force and self.operations_since_checkpoint < self.checkpoint_interval:
            return False
        
        try:
            # Обновляем время последнего обновления
            self.current_state.last_update = datetime.now().isoformat()
            
            # Формируем имя файла
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            checkpoint_file = self.checkpoint_dir / f"checkpoint_{self.current_state.run_id}_{timestamp}.json"
            
            # Сохраняем с блокировкой
            with self._file_lock(checkpoint_file, 'w') as f:
                json.dump(
                    self.current_state.to_dict(),
                    f,
                    indent=2,
                    ensure_ascii=False
                )
            
            self.logger.info(
                f"Чекпоинт сохранен: {checkpoint_file.name} "
                f"(прогресс: {self.current_state.progress_percentage:.1f}%)"
            )
            
            # Сбрасываем счетчик
            self.operations_since_checkpoint = 0
            
            # Очищаем старые чекпоинты
            self._cleanup_old_checkpoints()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка сохранения чекпоинта: {e}")
            return False
    
    def update_progress(self,
                       task_id: str,
                       result: Optional[ProfileResult] = None,
                       error: Optional[str] = None) -> None:
        """
        Обновляет прогресс выполнения.
        
        Args:
            task_id: Идентификатор задачи
            result: Результат выполнения (если успешно)
            error: Сообщение об ошибке (если неуспешно)
        """
        if self.current_state is None:
            self.logger.warning("Состояние не инициализировано")
            return
        
        # Проверяем, не была ли задача уже выполнена
        if task_id in self.current_state.completed_tasks:
            self.logger.debug(f"Задача {task_id} уже выполнена, пропускаем")
            return
        
        # Обновляем состояние
        if error:
            self.current_state.failed_operations += 1
            self.current_state.failed_tasks[task_id] = error
            self.logger.warning(f"Задача {task_id} завершилась с ошибкой: {error}")
        else:
            self.current_state.completed_operations += 1
            self.current_state.completed_tasks.add(task_id)
            if result:
                self.current_state.results[task_id] = result
            self.logger.debug(f"Задача {task_id} выполнена успешно")
        
        # Увеличиваем счетчик
        self.operations_since_checkpoint += 1
        
        # Автосохранение по интервалу
        if self.operations_since_checkpoint >= self.checkpoint_interval:
            self.save_checkpoint()
    
    def get_pending_tasks(self, all_tasks: List[str]) -> List[str]:
        """
        Возвращает список задач, которые еще не выполнены.
        
        Args:
            all_tasks: Полный список всех задач
            
        Returns:
            List[str]: Список невыполненных задач
        """
        if self.current_state is None:
            return all_tasks
        
        completed = self.current_state.completed_tasks
        failed = set(self.current_state.failed_tasks.keys())
        
        # Возвращаем задачи, которые не выполнены и не провалены
        # (или можно включить провальные для повторной попытки)
        pending = [task for task in all_tasks 
                  if task not in completed and task not in failed]
        
        self.logger.info(
            f"Задач к выполнению: {len(pending)} из {len(all_tasks)} "
            f"(выполнено: {len(completed)}, ошибок: {len(failed)})"
        )
        
        return pending
    
    def get_task_result(self, task_id: str) -> Optional[ProfileResult]:
        """
        Получает результат выполненной задачи.
        
        Args:
            task_id: Идентификатор задачи
            
        Returns:
            ProfileResult или None
        """
        if self.current_state and task_id in self.current_state.results:
            return self.current_state.results[task_id]
        return None
    
    def is_task_completed(self, task_id: str) -> bool:
        """Проверяет, выполнена ли задача."""
        return (self.current_state is not None and 
                task_id in self.current_state.completed_tasks)
    
    def clear_checkpoint(self) -> None:
        """Очищает текущий чекпоинт и все файлы."""
        self.current_state = None
        self.operations_since_checkpoint = 0
        
        # Удаляем все файлы чекпоинтов
        for checkpoint_file in self.checkpoint_dir.glob("checkpoint_*.json"):
            try:
                checkpoint_file.unlink()
                self.logger.debug(f"Удален чекпоинт: {checkpoint_file.name}")
            except Exception as e:
                self.logger.error(f"Ошибка удаления чекпоинта: {e}")
    
    def _find_latest_checkpoint(self) -> Optional[Path]:
        """Находит последний чекпоинт по времени модификации."""
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_*.json"))
        if not checkpoints:
            return None
        
        # Сортируем по времени модификации
        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        return latest
    
    def _cleanup_old_checkpoints(self) -> None:
        """Удаляет старые чекпоинты, оставляя только последние N."""
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_*.json"))
        
        if len(checkpoints) <= self.max_checkpoints:
            return
        
        # Сортируем по времени модификации
        checkpoints.sort(key=lambda p: p.stat().st_mtime)
        
        # Удаляем самые старые
        to_remove = checkpoints[:-self.max_checkpoints]
        for checkpoint in to_remove:
            try:
                checkpoint.unlink()
                self.logger.debug(f"Удален старый чекпоинт: {checkpoint.name}")
            except Exception as e:
                self.logger.error(f"Ошибка удаления старого чекпоинта: {e}")
    
    def _file_lock(self, file_path: Path, mode: str):
        """
        Контекстный менеджер для блокировки файла (только для Unix).
        На Windows просто возвращает открытый файл без блокировки.
        """
        class FileLock:
            def __init__(self, path, mode):
                self.path = path
                self.mode = mode
                self.file = None
                self.locked = False
            
            def __enter__(self):
                self.file = open(self.path, self.mode, encoding='utf-8')
                
                # Блокировка только на Unix системах
                if sys.platform not in ('win32', 'cygwin'):
                    try:
                        import fcntl
                        fcntl.flock(self.file.fileno(), fcntl.LOCK_EX)
                        self.locked = True
                    except ImportError:
                        # fcntl недоступен, продолжаем без блокировки
                        pass
                    except Exception as e:
                        # Игнорируем ошибки блокировки
                        import warnings
                        warnings.warn(f"Не удалось заблокировать файл: {e}")
                
                return self.file
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                if self.locked and sys.platform not in ('win32', 'cygwin'):
                    try:
                        import fcntl
                        fcntl.flock(self.file.fileno(), fcntl.LOCK_UN)
                    except:
                        pass
                
                if self.file:
                    self.file.close()
        
        return FileLock(file_path, mode)
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """Генерирует сводный отчет о текущем состоянии."""
        if self.current_state is None:
            return {"status": "no_state"}
        
        state = self.current_state
        
        # Подсчет времени выполнения
        start_time = datetime.fromisoformat(state.start_time)
        current_time = datetime.now()
        duration = current_time - start_time
        
        # Статистика по результатам
        time_stats = {}
        memory_stats = {}
        
        for task_id, result in state.results.items():
            if result.success:
                key = f"{result.library}_{result.backend or 'default'}"
                
                if key not in time_stats:
                    time_stats[key] = []
                    memory_stats[key] = []
                
                time_stats[key].append(result.mean_time)
                memory_stats[key].append(result.peak_memory_mb)
        
        return {
            "run_id": state.run_id,
            "start_time": state.start_time,
            "duration_seconds": duration.total_seconds(),
            "progress": {
                "percentage": state.progress_percentage,
                "completed": state.completed_operations,
                "failed": state.failed_operations,
                "total": state.total_operations,
                "is_complete": state.is_complete
            },
            "statistics": {
                "time": {k: {"mean": sum(v)/len(v), "count": len(v)} 
                        for k, v in time_stats.items()},
                "memory": {k: {"mean": sum(v)/len(v), "max": max(v)} 
                          for k, v in memory_stats.items()}
            },
            "errors": len(state.failed_tasks),
            "last_update": state.last_update
        }


class TaskIdentifier:
    """Генератор уникальных идентификаторов задач."""
    
    @staticmethod
    def generate(operation_name: str,
                library: str,
                backend: Optional[str],
                dataset_name: str) -> str:
        """
        Генерирует уникальный идентификатор задачи.
        
        Args:
            operation_name: Название операции
            library: Библиотека
            backend: Backend (может быть None)
            dataset_name: Имя датасета
            
        Returns:
            str: Уникальный идентификатор
        """
        parts = [operation_name, library]
        if backend:
            parts.append(backend)
        parts.append(dataset_name)
        
        return "_".join(parts)
    
    @staticmethod
    def parse(task_id: str) -> Tuple[str, str, Optional[str], str]:
        """
        Парсит идентификатор задачи обратно в компоненты.
        
        Args:
            task_id: Идентификатор задачи
            
        Returns:
            Tuple: (operation_name, library, backend, dataset_name)
        """
        parts = task_id.split("_")
        
        if len(parts) == 3:
            # Без backend
            return parts[0], parts[1], None, parts[2]
        elif len(parts) == 4:
            # С backend
            return parts[0], parts[1], parts[2], parts[3]
        else:
            raise ValueError(f"Неверный формат task_id: {task_id}")
