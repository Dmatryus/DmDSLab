"""
Основной класс для запуска бенчмарка с интеграцией всех модулей.
"""

import time
import sys
import uuid
import json
import signal
import platform
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass

from researches.polaris.pandas_polars_benchmark.src.core.config import Config
from researches.polaris.pandas_polars_benchmark.src.core.checkpoint import (
    CheckpointManager,
    BenchmarkState,
)
from researches.polaris.pandas_polars_benchmark.src.core.progress import (
    ProgressTracker,
    create_progress_tracker,
)
from researches.polaris.pandas_polars_benchmark.src.data import (
    DataGenerator,
    DatasetInfo,
)
from researches.polaris.pandas_polars_benchmark.src.profiling import (
    Profiler,
    ProfileResult,
)
from researches.polaris.pandas_polars_benchmark.src.operations import OperationRegistry
from researches.polaris.pandas_polars_benchmark.src.analysis import (
    OutlierDetector,
    StatisticsCalculator,
    ComparisonEngine,
    OutlierMethod,
    ComparisonMatrix,
)
from researches.polaris.pandas_polars_benchmark.src.reporting import (
    DataProcessor,
    VisualizationEngine,
    HTMLRenderer,
    MetricType,
    ReportConfig,
)
from researches.polaris.pandas_polars_benchmark.src.utils.logging import (
    setup_logging,
    get_logger,
)

from researches.polaris.pandas_polars_benchmark.src.analysis.comparison_engine import (
    ComparisonMetric,
)


@dataclass
class BenchmarkTask:
    """Представляет одну задачу бенчмаркинга."""

    task_id: str
    library: str
    backend: Optional[str]
    operation_name: str
    operation_category: str
    dataset: DatasetInfo

    def __hash__(self):
        return hash(self.task_id)


class BenchmarkRunner:
    """Основной класс для запуска и управления бенчмарком."""

    def __init__(
        self,
        config_path: Path,
        resume: bool = False,
        dry_run: bool = False,
        output_dir: Optional[Path] = None,
    ):
        """
        Инициализация BenchmarkRunner.

        Args:
            config_path: Путь к файлу конфигурации
            resume: Возобновить с последнего чекпоинта
            dry_run: Только валидация без выполнения
            output_dir: Директория для результатов
        """
        self.config_path = config_path
        self.resume = resume
        self.dry_run = dry_run
        self.output_dir = output_dir or Path("results")

        # Настройка логирования
        setup_logging(
            log_dir=self.output_dir,
            console_level="INFO",
        )
        self.logger = get_logger("benchmark_runner")

        # Загрузка конфигурации
        self.logger.info(f"Загрузка конфигурации из {config_path}")
        self.config = Config(config_path)

        # Инициализация компонентов
        self.run_id = str(uuid.uuid4())
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=self.output_dir / "checkpoints"
        )
        self.progress_tracker: Optional[ProgressTracker] = None
        self.state: Optional[BenchmarkState] = None

        # Регистрация обработчика сигналов
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Компоненты для работы
        self.data_generator = DataGenerator(
            config=self.config.data_generation,
            output_dir=self.output_dir / "data",
        )
        self.profiler = Profiler(config=self.config.profiling)
        self.operation_registry = OperationRegistry()

        # Анализ и отчеты
        self.outlier_detector = OutlierDetector()
        self.statistics_calculator = StatisticsCalculator()
        self.comparison_engine = ComparisonEngine()

    def run(self) -> bool:
        """
        Основной метод запуска бенчмарка.

        Returns:
            bool: True если успешно завершено
        """
        try:
            self.logger.info("=" * 80)
            self.logger.info(
                f"ЗАПУСК БЕНЧМАРКА: {self.config._config_data['benchmark']['name']}"
            )
            self.logger.info(
                f"Версия: {self.config._config_data['benchmark']['version']}"
            )
            self.logger.info(f"Run ID: {self.run_id}")
            self.logger.info("=" * 80)

            # Валидация окружения
            if not self._validate_environment():
                return False

            # Подготовка задач
            tasks = self._prepare_tasks()

            if self.dry_run:
                self.logger.info(f"Dry run: будет выполнено {len(tasks)} задач")
                return True

            # Инициализация или восстановление состояния
            if self.resume:
                if not self._resume_from_checkpoint(tasks):
                    return False
            else:
                self._initialize_state(tasks)

            # Выполнение бенчмарка
            success = self._execute_benchmark(tasks)

            if success:
                # Анализ результатов
                self._analyze_results()

                # Генерация отчета
                self._generate_report()

                self.logger.info("✅ Бенчмарк успешно завершен!")

            return success

        except Exception as e:
            self.logger.error(f"Критическая ошибка: {e}", exc_info=True)
            return False
        finally:
            if self.progress_tracker:
                self.progress_tracker.close()

    def _validate_environment(self) -> bool:
        """Валидация окружения перед запуском."""
        self.logger.info("Проверка окружения...")

        # Информация о системе
        self.logger.info(f"Python: {platform.python_version()}")
        self.logger.info(f"Платформа: {platform.platform()}")
        self.logger.info(f"Процессор: {platform.processor()}")
        self.logger.info(f"CPU: {platform.machine()}")

        # Проверка библиотек
        issues = []

        for lib_config in self.config.libraries:
            if not lib_config.enabled:
                continue

            try:
                if lib_config.name == "pandas":
                    import pandas as pd

                    self.logger.info(f"Pandas версия: {pd.__version__}")
                elif lib_config.name == "polars":
                    import polars as pl

                    self.logger.info(f"Polars версия: {pl.__version__}")
            except ImportError:
                issues.append(f"Библиотека {lib_config.name} не установлена")

        # Проверка директорий
        required_dirs = [
            self.output_dir,
            self.output_dir / "data",
            self.output_dir / "logs",
            self.output_dir / "checkpoints",
            self.output_dir / "analysis",
            self.output_dir / "reports",
        ]

        for dir_path in required_dirs:
            dir_path.mkdir(parents=True, exist_ok=True)

        if issues:
            for issue in issues:
                self.logger.error(f"❌ {issue}")
            return False

        self.logger.info("✅ Окружение валидно")
        return True

    def _prepare_tasks(self) -> List[BenchmarkTask]:
        """Подготовка списка задач для выполнения."""
        self.logger.info("Подготовка задач...")

        tasks = []

        # Генерация или загрузка датасетов
        datasets = self.data_generator.generate_all_datasets()

        # Создание задач для каждой комбинации
        for dataset in datasets:
            for lib_config in self.config.libraries:
                if not lib_config.enabled:
                    continue

                # Для каждого backend (или None если не указаны)
                backends = lib_config.backends if lib_config.backends else [None]

                for backend in backends:
                    for category, operations in self.config.operations.items():
                        for op_name in operations:
                            task_id = self._generate_task_id(
                                lib_config.name, backend, op_name, dataset.name
                            )

                            task = BenchmarkTask(
                                task_id=task_id,
                                library=lib_config.name,
                                backend=backend,
                                operation_name=op_name,
                                operation_category=category,
                                dataset=dataset,
                            )
                            tasks.append(task)

        self.logger.info(f"Подготовлено задач: {len(tasks)}")
        return tasks

    def _generate_task_id(
        self, library: str, backend: Optional[str], operation: str, dataset: str
    ) -> str:
        """Генерирует уникальный ID для задачи."""
        parts = [library]
        if backend:
            parts.append(backend)
        parts.extend([operation, dataset])
        return "_".join(parts)

    def _initialize_state(self, tasks: List[BenchmarkTask]) -> None:
        """Инициализация нового состояния."""
        self.state = self.checkpoint_manager.initialize_state(
            run_id=self.run_id,
            config=self.config._config_data,
            total_operations=len(tasks),
        )

        self.progress_tracker = create_progress_tracker(
            total_operations=len(tasks), show_progress=True
        )

    def _resume_from_checkpoint(self, tasks: List[BenchmarkTask]) -> bool:
        """Восстановление из чекпоинта."""
        self.logger.info("Попытка восстановления из чекпоинта...")

        state = self.checkpoint_manager.load_checkpoint()
        if not state:
            self.logger.error("Чекпоинт не найден")
            return False

        # Проверка совместимости конфигурации
        current_config_str = json.dumps(self.config._config_data, sort_keys=True)
        current_hash = self.checkpoint_manager._calculate_config_hash(
            current_config_str
        )

        if state.config_hash != current_hash:
            self.logger.warning("Конфигурация изменилась с момента создания чекпоинта")
            response = input("Продолжить с новой конфигурацией? (y/n): ")
            if response.lower() != "y":
                return False

        self.state = state
        self.run_id = state.run_id

        # Восстановление прогресса
        completed_count = len(state.completed_tasks)
        self.progress_tracker = create_progress_tracker(
            total_operations=len(tasks), show_progress=True
        )
        self.progress_tracker.update_progress(
            completed_count, f"Восстановлено: {completed_count} из {len(tasks)}"
        )

        self.logger.info(
            f"✅ Восстановлено из чекпоинта: {completed_count} задач выполнено"
        )
        return True

    def _execute_benchmark(self, tasks: List[BenchmarkTask]) -> bool:
        """Выполнение всех задач бенчмарка."""
        self.logger.info("Начало выполнения бенчмарка...")

        try:
            for task in tasks:
                # Проверка, выполнена ли задача
                if task.task_id in self.state.completed_tasks:
                    continue

                # Проверка на прерывание
                if self._interrupted:
                    self.logger.warning("Получен сигнал прерывания")
                    self._save_checkpoint()
                    return False

                # Выполнение задачи
                success = self._execute_task(task)

                # Обновление состояния
                if success:
                    self.state.completed_operations += 1
                else:
                    self.state.failed_operations += 1

                # Автосохранение чекпоинта
                if (
                    self.state.completed_operations + self.state.failed_operations
                ) % 10 == 0:
                    self._save_checkpoint()

            # Финальное сохранение
            self._save_checkpoint()

            # Проверка результатов
            if self.state.failed_operations > 0:
                self.logger.warning(
                    f"Завершено с ошибками: {self.state.failed_operations} задач не выполнено"
                )

            return True

        except Exception as e:
            self.logger.error(f"Ошибка выполнения: {e}", exc_info=True)
            self._save_checkpoint()
            return False

    def _execute_task(self, task: BenchmarkTask) -> bool:
        """Выполнение одной задачи профилирования."""
        self.progress_tracker.start_operation(
            operation_name=task.operation_name,
            library=task.library,
            dataset_name=task.dataset.name,
        )

        try:
            # Получение операции из реестра
            operation = self.operation_registry.get_operation(
                task.operation_category, task.operation_name
            )

            if not operation:
                raise ValueError(
                    f"Операция не найдена: {task.operation_category}.{task.operation_name}"
                )

            # Профилирование операции
            result = self.profiler.profile_operation(
                operation=operation,
                dataset_path=task.dataset.file_paths["csv"],
                library=task.library,
                backend=task.backend,
            )

            # Сохранение результата
            self.state.results[task.task_id] = result
            self.state.completed_tasks.add(task.task_id)

            self.progress_tracker.end_operation(success=True)
            return True

        except Exception as e:
            self.logger.error(f"Ошибка выполнения задачи {task.task_id}: {e}")
            self.state.failed_tasks[task.task_id] = str(e)
            self.progress_tracker.end_operation(success=False)
            return False

    def _analyze_results(self) -> None:
        """Анализ результатов бенчмарка."""
        self.logger.info("Анализ результатов...")

        # Группировка результатов по операциям
        results_by_operation: Dict[str, Dict[str, List[float]]] = {}

        for task_id, result in self.state.results.items():
            if not result.success:
                continue

            op_key = f"{result.operation_category}.{result.operation_name}"
            if op_key not in results_by_operation:
                results_by_operation[op_key] = {}

            lib_key = result.library
            if result.backend:
                lib_key = f"{result.library}_{result.backend}"

            if lib_key not in results_by_operation[op_key]:
                results_by_operation[op_key][lib_key] = []

            results_by_operation[op_key][lib_key].extend(result.execution_times)

        # Удаление выбросов и статистический анализ
        self.analysis_results = {}

        for operation, lib_results in results_by_operation.items():
            self.analysis_results[operation] = {}

            for library, times in lib_results.items():
                # Удаление выбросов
                clean_times, outlier_result = self.outlier_detector.remove_outliers(
                    times, method=OutlierMethod.IQR
                )

                # Расчет статистик
                stats = self.statistics_calculator.calculate_descriptive_stats(
                    clean_times
                )

                self.analysis_results[operation][library] = {
                    "times": clean_times,
                    "stats": stats,
                    "outliers_removed": outlier_result.removed_count,
                }

        # Сравнение библиотек
        comparison_results = {}
        for operation, lib_data in self.analysis_results.items():
            if "pandas" in lib_data and "polars" in lib_data:
                result = self.comparison_engine.compare_two_samples(
                    baseline=lib_data["pandas"]["times"],
                    comparison=lib_data["polars"]["times"],
                    name=operation,
                    baseline_library="pandas",
                    comparison_library="polars",
                )
                comparison_results[operation] = result

        # Создание матрицы сравнения
        self.comparison_matrix = ComparisonMatrix(
            baseline_library="pandas",
            comparison_library="polars",
            metric=ComparisonMetric.EXECUTION_TIME,
            results=comparison_results,
        )

        # Сохранение результатов анализа
        analysis_path = self.output_dir / "analysis" / f"analysis_{self.run_id}.json"
        self.comparison_engine.export_results(
            self.comparison_matrix, analysis_path, format="json"
        )

        self.logger.info(f"✅ Анализ завершен. Результаты сохранены в {analysis_path}")

    def _generate_report(self) -> None:
        """Генерация финального отчета."""
        self.logger.info("Генерация отчета...")

        # Подготовка данных для визуализации
        processor = DataProcessor()
        viz_engine = VisualizationEngine()
        html_renderer = HTMLRenderer()

        # Преобразование результатов в DataFrame
        results_data = []
        for task_id, result in self.state.results.items():
            if result.success:
                for time in result.execution_times:
                    results_data.append(
                        {
                            "library": result.library,
                            "backend": result.backend or "default",
                            "operation": result.operation_name,
                            "category": result.operation_category,
                            "dataset": result.dataset_name,
                            "dataset_size": result.dataset_size,
                            "execution_time": time,
                            "memory_peak": result.peak_memory_mb,
                            "memory_mean": result.avg_memory_mb,
                        }
                    )

        import pandas as pd

        results_df = pd.DataFrame(results_data)

        # Сохранение сырых данных
        raw_data_path = self.output_dir / "analysis" / f"raw_results_{self.run_id}.csv"
        results_df.to_csv(raw_data_path, index=False)

        # Подготовка данных для различных графиков
        comparison_data = processor.prepare_comparison_data(
            df=results_df,
            groupby=["operation"],
            pivot_column="library",
        )

        timeline_data = processor.prepare_timeline_data(
            df=results_df,
            metric=MetricType.EXECUTION_TIME,
            dataset_size_column="dataset_size",
        )

        # Создание визуализаций
        figures = {
            "comparison": viz_engine.create_comparison_bar_chart(comparison_data),
            "timeline": viz_engine.create_line_chart(timeline_data),
            "speedup": viz_engine.create_speedup_chart(
                processor.prepare_comparison_data(
                    df=results_df,
                    metric=MetricType.SPEEDUP,
                    groupby=["operation"],
                    pivot_column="library",
                )
            ),
            "distribution": viz_engine.create_box_plot(
                processor.prepare_distribution_data(
                    df=results_df, metric=MetricType.EXECUTION_TIME
                )
            ),
        }

        # Сводная таблица
        summary_df = processor.create_summary_table(results_df)

        # Генерация HTML отчета
        report_config = ReportConfig(
            title=self.config._config_data["benchmark"]["name"],
            subtitle=f"Run ID: {self.run_id}",
            author="Benchmark System",
            description=self.config._config_data["benchmark"].get("description", ""),
        )

        report_path = self.output_dir / "reports" / f"report_{self.run_id}.html"

        html_renderer.render_report(
            figures=figures,
            summary_data=summary_df,
            config=report_config,
            output_path=report_path,
            comparison_matrix=self.comparison_matrix,
        )

        self.logger.info(f"✅ Отчет сохранен: {report_path}")

    def _save_checkpoint(self) -> None:
        """Сохранение текущего состояния."""
        if self.state:
            self.state.last_update = datetime.now().isoformat()
            self.checkpoint_manager.save_checkpoint(self.state)

    def _signal_handler(self, signum, frame):
        """Обработчик сигналов для корректного завершения."""
        self.logger.warning(f"Получен сигнал {signum}")
        self._interrupted = True

    _interrupted = False


def create_benchmark_runner(config_path: Path, **kwargs) -> BenchmarkRunner:
    """
    Фабричная функция для создания BenchmarkRunner.

    Args:
        config_path: Путь к конфигурации
        **kwargs: Дополнительные параметры для BenchmarkRunner

    Returns:
        BenchmarkRunner: Настроенный экземпляр
    """
    return BenchmarkRunner(config_path, **kwargs)
