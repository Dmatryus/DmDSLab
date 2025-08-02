"""
Модуль для отслеживания и отображения прогресса выполнения бенчмарка.
"""

import time
import sys
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
import threading
from collections import deque

# Пробуем импортировать tqdm
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("Warning: tqdm not installed. Install it with: pip install tqdm")

from researches.polaris.pandas_polars_benchmark.src.utils.logging import get_logger


@dataclass
class OperationTiming:
    """Информация о времени выполнения операции."""
    operation_name: str
    library: str
    dataset_name: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    success: bool = True
    
    def complete(self, success: bool = True) -> None:
        """Отмечает операцию как завершенную."""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.success = success


class ProgressTracker:
    """Класс для отслеживания прогресса выполнения бенчмарка."""
    
    def __init__(self,
                 total_operations: int,
                 show_progress_bar: bool = True,
                 update_interval: float = 0.5):
        """
        Инициализация трекера прогресса.
        
        Args:
            total_operations: Общее количество операций
            show_progress_bar: Показывать progress bar
            update_interval: Интервал обновления отображения (сек)
        """
        self.total_operations = total_operations
        self.show_progress_bar = show_progress_bar and HAS_TQDM
        self.update_interval = update_interval
        
        self.logger = get_logger('progress_tracker')
        
        # Счетчики
        self.completed_operations = 0
        self.failed_operations = 0
        self.skipped_operations = 0
        
        # Время выполнения
        self.start_time = time.time()
        self.operation_timings: List[OperationTiming] = []
        self.current_operation: Optional[OperationTiming] = None
        
        # История для расчета ETA
        self.recent_durations = deque(maxlen=10)  # Последние 10 операций
        
        # Progress bar
        self.pbar: Optional[tqdm] = None
        if self.show_progress_bar:
            self._init_progress_bar()
        
        # Поток для обновления дисплея
        self._update_thread: Optional[threading.Thread] = None
        self._stop_update = threading.Event()
    
    def _init_progress_bar(self) -> None:
        """Инициализирует progress bar."""
        self.pbar = tqdm(
            total=self.total_operations,
            desc="Benchmark Progress",
            unit="op",
            unit_scale=False,
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            ncols=100,
            colour='green'
        )
    
    def start_operation(self,
                       operation_name: str,
                       library: str,
                       dataset_name: str) -> None:
        """
        Отмечает начало операции.
        
        Args:
            operation_name: Название операции
            library: Библиотека
            dataset_name: Имя датасета
        """
        # Завершаем предыдущую операцию если она не завершена
        if self.current_operation and self.current_operation.end_time is None:
            self.current_operation.complete(success=False)
            self.failed_operations += 1
        
        # Создаем новую операцию
        self.current_operation = OperationTiming(
            operation_name=operation_name,
            library=library,
            dataset_name=dataset_name,
            start_time=time.time()
        )
        
        # Обновляем описание progress bar
        if self.pbar:
            desc = f"{operation_name} | {library} | {dataset_name}"
            # Обрезаем если слишком длинное
            if len(desc) > 50:
                desc = desc[:47] + "..."
            self.pbar.set_description(desc)
    
    def end_operation(self, success: bool = True) -> None:
        """
        Отмечает конец текущей операции.
        
        Args:
            success: Успешно ли завершилась операция
        """
        if not self.current_operation:
            self.logger.warning("Попытка завершить операцию, которая не была начата")
            return
        
        # Завершаем операцию
        self.current_operation.complete(success)
        self.operation_timings.append(self.current_operation)
        
        # Обновляем счетчики
        if success:
            self.completed_operations += 1
            self.recent_durations.append(self.current_operation.duration)
        else:
            self.failed_operations += 1
        
        # Обновляем progress bar
        if self.pbar:
            self.pbar.update(1)
            self._update_postfix()
        
        self.current_operation = None
    
    def skip_operation(self, reason: str = "") -> None:
        """
        Отмечает операцию как пропущенную.
        
        Args:
            reason: Причина пропуска
        """
        self.skipped_operations += 1
        
        if self.pbar:
            self.pbar.update(1)
            self._update_postfix()
        
        if reason:
            self.logger.info(f"Операция пропущена: {reason}")
    
    def update_progress(self, current: int, description: str = "") -> None:
        """
        Обновляет прогресс до указанного значения.
        
        Args:
            current: Текущее количество выполненных операций
            description: Описание текущей операции
        """
        if self.pbar:
            # Обновляем до нужного значения
            self.pbar.n = min(current, self.total_operations)
            self.pbar.refresh()
            
            if description:
                self.pbar.set_description(description)
            
            self._update_postfix()
    
    def _update_postfix(self) -> None:
        """Обновляет дополнительную информацию в progress bar."""
        if not self.pbar:
            return
        
        postfix_dict = {
            "✓": self.completed_operations,
            "✗": self.failed_operations,
            "→": self.skipped_operations
        }
        
        # Добавляем ETA
        eta = self.get_eta()
        if eta:
            postfix_dict["ETA"] = eta
        
        self.pbar.set_postfix(postfix_dict)
    
    def get_eta(self) -> Optional[str]:
        """
        Рассчитывает примерное время до завершения.
        
        Returns:
            str: Форматированное время или None
        """
        if not self.recent_durations:
            return None
        
        remaining_ops = self.total_operations - (
            self.completed_operations + 
            self.failed_operations + 
            self.skipped_operations
        )
        
        if remaining_ops <= 0:
            return "00:00:00"
        
        # Среднее время последних операций
        avg_duration = sum(self.recent_durations) / len(self.recent_durations)
        
        # Оценка оставшегося времени
        eta_seconds = remaining_ops * avg_duration
        
        # Форматируем
        eta_td = timedelta(seconds=int(eta_seconds))
        return str(eta_td)
    
    def get_elapsed_time(self) -> str:
        """Возвращает прошедшее время."""
        elapsed = time.time() - self.start_time
        return str(timedelta(seconds=int(elapsed)))
    
    def get_progress_info(self) -> Dict[str, Any]:
        """
        Возвращает полную информацию о прогрессе.
        
        Returns:
            Dict с информацией о прогрессе
        """
        current_total = (self.completed_operations + 
                        self.failed_operations + 
                        self.skipped_operations)
        
        progress_pct = (current_total / self.total_operations * 100) if self.total_operations > 0 else 0
        
        # Расчет средней скорости
        elapsed = time.time() - self.start_time
        ops_per_second = current_total / elapsed if elapsed > 0 else 0
        
        # Статистика по времени операций
        if self.operation_timings:
            durations = [op.duration for op in self.operation_timings if op.duration]
            avg_duration = sum(durations) / len(durations) if durations else 0
            min_duration = min(durations) if durations else 0
            max_duration = max(durations) if durations else 0
        else:
            avg_duration = min_duration = max_duration = 0
        
        return {
            "progress_percentage": progress_pct,
            "completed": self.completed_operations,
            "failed": self.failed_operations,
            "skipped": self.skipped_operations,
            "total": self.total_operations,
            "remaining": self.total_operations - current_total,
            "elapsed_time": self.get_elapsed_time(),
            "eta": self.get_eta(),
            "operations_per_second": ops_per_second,
            "timing_stats": {
                "average": avg_duration,
                "min": min_duration,
                "max": max_duration
            }
        }
    
    def display_summary(self) -> None:
        """Отображает итоговую сводку."""
        info = self.get_progress_info()
        
        print("\n" + "="*60)
        print("ИТОГИ ВЫПОЛНЕНИЯ БЕНЧМАРКА")
        print("="*60)
        print(f"Общее время: {info['elapsed_time']}")
        print(f"Всего операций: {info['total']}")
        print(f"  ✓ Успешно: {info['completed']} ({info['completed']/info['total']*100:.1f}%)")
        print(f"  ✗ С ошибками: {info['failed']} ({info['failed']/info['total']*100:.1f}%)")
        print(f"  → Пропущено: {info['skipped']} ({info['skipped']/info['total']*100:.1f}%)")
        print(f"Средняя скорость: {info['operations_per_second']:.2f} оп/сек")
        
        if info['timing_stats']['average'] > 0:
            print(f"\nВремя операций:")
            print(f"  Среднее: {info['timing_stats']['average']:.2f} сек")
            print(f"  Мин: {info['timing_stats']['min']:.2f} сек")
            print(f"  Макс: {info['timing_stats']['max']:.2f} сек")
        
        print("="*60)
    
    def start_live_update(self) -> None:
        """Запускает поток для live обновления дисплея."""
        if self._update_thread is not None:
            return
        
        self._stop_update.clear()
        self._update_thread = threading.Thread(
            target=self._live_update_loop,
            daemon=True
        )
        self._update_thread.start()
    
    def stop_live_update(self) -> None:
        """Останавливает live обновление."""
        if self._update_thread is None:
            return
        
        self._stop_update.set()
        self._update_thread.join(timeout=1.0)
        self._update_thread = None
    
    def _live_update_loop(self) -> None:
        """Цикл обновления дисплея в отдельном потоке."""
        while not self._stop_update.is_set():
            if self.pbar:
                self._update_postfix()
            time.sleep(self.update_interval)
    
    def close(self) -> None:
        """Закрывает progress bar и освобождает ресурсы."""
        self.stop_live_update()
        
        if self.pbar:
            self.pbar.close()
            self.pbar = None
    
    def __enter__(self):
        """Вход в контекстный менеджер."""
        self.start_live_update()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Выход из контекстного менеджера."""
        self.close()
        if exc_type is None:
            self.display_summary()


class SimpleProgressPrinter:
    """Простой принтер прогресса для случаев когда tqdm недоступен."""
    
    def __init__(self, total: int, width: int = 50):
        """
        Инициализация простого принтера.
        
        Args:
            total: Общее количество операций
            width: Ширина progress bar
        """
        self.total = total
        self.width = width
        self.current = 0
        self.start_time = time.time()
        self.last_update = 0
    
    def update(self, current: int, description: str = "") -> None:
        """Обновляет отображение прогресса."""
        self.current = current
        
        # Обновляем не чаще раза в секунду
        if time.time() - self.last_update < 1.0:
            return
        
        self.last_update = time.time()
        
        # Рассчитываем прогресс
        if self.total > 0:
            progress = self.current / self.total
            filled = int(self.width * progress)
            bar = "█" * filled + "░" * (self.width - filled)
            
            # Время
            elapsed = time.time() - self.start_time
            if self.current > 0:
                eta = (elapsed / self.current) * (self.total - self.current)
                eta_str = str(timedelta(seconds=int(eta)))
            else:
                eta_str = "??:??:??"
            
            # Очищаем строку и печатаем
            sys.stdout.write('\r')
            sys.stdout.write(
                f"{description[:30]:30} [{bar}] "
                f"{self.current}/{self.total} "
                f"({progress*100:.1f}%) "
                f"ETA: {eta_str}"
            )
            sys.stdout.flush()
    
    def finish(self) -> None:
        """Завершает отображение."""
        sys.stdout.write('\n')
        sys.stdout.flush()


def create_progress_tracker(total_operations: int,
                          show_progress: bool = True) -> ProgressTracker:
    """
    Фабричная функция для создания progress tracker.
    
    Args:
        total_operations: Общее количество операций
        show_progress: Показывать ли прогресс
        
    Returns:
        ProgressTracker: Настроенный трекер
    """
    # Проверяем, в интерактивном режиме ли мы
    is_interactive = sys.stdout.isatty()
    
    return ProgressTracker(
        total_operations=total_operations,
        show_progress_bar=show_progress and is_interactive
    )
