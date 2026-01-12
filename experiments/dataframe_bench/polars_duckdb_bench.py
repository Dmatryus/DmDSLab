"""
Polars vs DuckDB Benchmark
Сравнение производительности операций и ML пайплайнов.
"""

import argparse
import gc
import tempfile
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import duckdb
import polars as pl
import psutil

# =============================================================================
# Конфигурация
# =============================================================================


@dataclass
class DataGenerationConfig:
    """Конфигурация генерации данных для бенчмарка.

    Используется для автоматической генерации данных через Synthdator.
    Если указан data_path в BenchmarkConfig, эти параметры игнорируются.

    Генерируется один большой файл с max_size, затем для разных data_sizes
    читается нужное количество строк (через n_rows).

    Attributes:
        max_size: Максимальный размер генерируемого файла (например, "10GB").
            Должен быть >= max(data_sizes). Если None — берётся max из data_sizes.
        n_numeric: Количество числовых фич.
        informative_ratio: Доля информативных фич (0-1).
        n_categories: Количество категориальных колонок.
        category_cardinality: Количество уникальных значений в категории.
        tasks: Список ML-задач для генерации таргетов.
            Каждая задача создаёт свой таргет: target_regression, target_binary и т.д.
        target_method: Метод генерации целевой переменной.
        target_noise: Уровень шума в таргете (0-1).
        n_classes: Количество классов для multiclass.
        datetime_range: Диапазон дат для timestamp как tuple (start, end).
        with_date: Генерировать ли колонку date из timestamp.
        n_booleans: Количество boolean колонок.
        nullable_ratio: Доля NULL значений (0-1).
    """

    # Размер файла
    max_size: str | None = None  # None = берётся max из data_sizes

    # Числовые фичи
    n_numeric: int = 20
    informative_ratio: float = 0.5

    # Категориальные фичи
    n_categories: int = 5
    category_cardinality: int = 10

    # Таргеты - список задач для генерации
    tasks: list[str] = field(
        default_factory=lambda: ["regression", "binary", "multiclass"]
    )
    target_method: str = "linear"
    target_noise: float = 0.1
    n_classes: int = 5

    # Временные данные
    datetime_range: tuple[str, str] | None = None
    with_date: bool = False

    # Boolean
    n_booleans: int = 0

    # NULL
    nullable_ratio: float = 0.0


@dataclass
class BenchmarkConfig:
    """Конфигурация бенчмарка.

    Attributes:
        data_path: Путь к parquet-файлу с данными. Если указан — используется
            существующий файл. Если None — данные генерируются автоматически.
        data_sizes: Список размеров данных для тестирования (например, ["100MB", "1GB"]).
            Бенчмарк читает нужное количество строк из одного большого файла.
        data_generation: Параметры генерации данных. Используется если data_path=None.
        output_path: Директория для сохранения результатов и сгенерированных данных.
        seed: Seed для воспроизводимости. None для случайного порядка.
        n_runs: Количество прогонов каждого бенчмарка.
        cold_runs: Количество холодных прогонов (без кеша).
        timeout_seconds: Таймаут для одного бенчмарка в секундах.
        metrics_interval_ms: Интервал сбора метрик (память, CPU) в миллисекундах.
        frameworks: Фреймворки для тестирования ("polars", "duckdb").
        polars_modes: Режимы Polars ("eager", "lazy", "streaming").
        benchmark_filter: Список бенчмарков для запуска. None = все.
        ml_pipelines: Список ML пайплайнов для запуска. None = все.
        meta: Метаданные датасета из synthdator (загружается автоматически).
    """

    # Данные
    data_path: Path | None = None
    data_sizes: list[str] = field(default_factory=lambda: ["100MB"])
    data_generation: DataGenerationConfig | None = field(
        default_factory=DataGenerationConfig
    )
    output_path: Path = field(default_factory=lambda: Path("results"))

    # Метаданные датасета (загружается автоматически из meta.pkl)
    meta: Any = None

    # Воспроизводимость
    seed: int | None = None

    # Параметры запуска
    n_runs: int = 3
    cold_runs: int = 1
    timeout_seconds: int = 300
    metrics_interval_ms: int = 100

    # Фреймворки
    frameworks: list[str] = field(default_factory=lambda: ["polars", "duckdb"])
    polars_modes: list[str] = field(
        default_factory=lambda: ["eager", "lazy", "streaming"]
    )

    # Фильтры
    benchmark_filter: list[str] | None = None  # None = все бенчмарки
    ml_pipelines: list[str] | None = None  # None = все ML пайплайны

    def __post_init__(self) -> None:
        """Валидация параметров конфига."""
        # Конвертация строк в Path
        if isinstance(self.data_path, str):
            self.data_path = Path(self.data_path)
        if isinstance(self.output_path, str):
            self.output_path = Path(self.output_path)

        # Валидация параметров запуска
        if self.n_runs < 1:
            raise ValueError("n_runs должен быть >= 1")
        if self.cold_runs < 0:
            raise ValueError("cold_runs должен быть >= 0")
        if self.cold_runs > self.n_runs:
            raise ValueError("cold_runs не может быть больше n_runs")
        if self.timeout_seconds < 1:
            raise ValueError("timeout_seconds должен быть >= 1")
        if self.metrics_interval_ms < 10:
            raise ValueError("metrics_interval_ms должен быть >= 10")

        # Валидация фреймворков
        valid_frameworks = {"polars", "duckdb"}
        for fw in self.frameworks:
            if fw not in valid_frameworks:
                raise ValueError(f"Неизвестный фреймворк: {fw}")

        valid_modes = {"eager", "lazy", "streaming"}
        for mode in self.polars_modes:
            if mode not in valid_modes:
                raise ValueError(f"Неизвестный режим Polars: {mode}")


# =============================================================================
# Сбор метрик
# =============================================================================


@dataclass
class MetricsSnapshot:
    timestamp: float
    memory_mb: float
    cpu_percent: float


@dataclass
class CollectedMetrics:
    peak_memory_mb: float = 0.0
    avg_cpu_percent: float = 0.0
    snapshots: list[MetricsSnapshot] = field(default_factory=list)


class MetricsCollector:
    """Фоновый сборщик метрик памяти и CPU."""

    def __init__(self, interval_ms: int = 100):
        self.interval_ms = interval_ms
        self._process = psutil.Process()
        self._snapshots: list[MetricsSnapshot] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def _collect_loop(self):
        self._process.cpu_percent()
        while not self._stop_event.is_set():
            try:
                memory_info = self._process.memory_info()
                memory_mb = memory_info.rss / (1024 * 1024)
                cpu_percent = self._process.cpu_percent()
                snapshot = MetricsSnapshot(
                    timestamp=time.perf_counter(),
                    memory_mb=memory_mb,
                    cpu_percent=cpu_percent,
                )
                with self._lock:
                    self._snapshots.append(snapshot)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            self._stop_event.wait(self.interval_ms / 1000)

    def start(self):
        self._snapshots = []
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._collect_loop, daemon=True)
        self._thread.start()

    def stop(self) -> CollectedMetrics:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        with self._lock:
            snapshots = self._snapshots.copy()
        if not snapshots:
            return CollectedMetrics()
        peak_memory_mb = max(s.memory_mb for s in snapshots)
        avg_cpu_percent = sum(s.cpu_percent for s in snapshots) / len(snapshots)
        return CollectedMetrics(
            peak_memory_mb=peak_memory_mb,
            avg_cpu_percent=avg_cpu_percent,
            snapshots=snapshots,
        )

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False


# =============================================================================
# Декоратор бенчмарка
# =============================================================================


def benchmark(
    interval_ms: int = 100,
) -> Callable[[Callable[..., Any]], Callable[..., tuple[Any, float, CollectedMetrics]]]:
    """
    Декоратор для измерения производительности.
    Выполняет gc.collect(), замеряет время и метрики.
    """

    def decorator(
        func: Callable[..., Any],
    ) -> Callable[..., tuple[Any, float, CollectedMetrics]]:
        def wrapper(*args: Any, **kwargs: Any) -> tuple[Any, float, CollectedMetrics]:
            gc.collect()
            collector = MetricsCollector(interval_ms=interval_ms)
            collector.start()
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
            finally:
                end_time = time.perf_counter()
                metrics = collector.stop()
            return result, end_time - start_time, metrics

        return wrapper

    return decorator


# =============================================================================
# Результаты бенчмарка
# =============================================================================


@dataclass
class BenchmarkResult:
    operation: str
    framework: str  # "polars" | "duckdb"
    mode: str  # "eager" | "lazy" | "streaming" | "sql"
    data_size: str
    run_number: int
    is_cold: bool
    wall_time_seconds: float
    peak_memory_mb: float
    cpu_percent: float
    success: bool
    error_message: str | None = None


@dataclass
class MLPipelineResult:
    task_type: str
    model_name: str
    framework: str
    data_size: str
    feature_set: str
    preprocessing: str
    validation: str
    time_loading: float
    time_cleaning: float
    time_feature_engineering: float
    time_preprocessing: float
    time_splitting: float
    time_training: float
    time_validation: float
    time_total: float
    peak_memory_mb: float
    metrics: dict[str, float] = field(default_factory=dict)
    success: bool = True
    error_message: str | None = None


@dataclass
class BenchmarkResults:
    """Агрегированные результаты выполнения бенчмарков (аналог Meta в synthdator)."""

    benchmarks: list[BenchmarkResult] = field(default_factory=list)
    ml_pipelines: list[MLPipelineResult] = field(default_factory=list)

    @property
    def success_count(self) -> int:
        """Количество успешных бенчмарков."""
        return sum(1 for r in self.benchmarks if r.success)

    @property
    def failed_count(self) -> int:
        """Количество неуспешных бенчмарков."""
        return sum(1 for r in self.benchmarks if not r.success)

    @property
    def ml_success_count(self) -> int:
        """Количество успешных ML пайплайнов."""
        return sum(1 for r in self.ml_pipelines if r.success)

    @property
    def ml_failed_count(self) -> int:
        """Количество неуспешных ML пайплайнов."""
        return sum(1 for r in self.ml_pipelines if not r.success)

    def save(self, path: Path) -> None:
        """Сохраняет результаты в JSON."""
        import json
        from dataclasses import asdict
        from datetime import datetime

        data = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "benchmarks_total": len(self.benchmarks),
                "benchmarks_success": self.success_count,
                "benchmarks_failed": self.failed_count,
                "ml_pipelines_total": len(self.ml_pipelines),
                "ml_pipelines_success": self.ml_success_count,
                "ml_pipelines_failed": self.ml_failed_count,
            },
            "benchmarks": [asdict(r) for r in self.benchmarks],
            "ml_pipelines": [asdict(r) for r in self.ml_pipelines],
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    def summary(self) -> str:
        """Возвращает текстовую сводку результатов."""
        lines = [
            "=" * 60,
            "Benchmark Results Summary",
            "=" * 60,
            f"Benchmarks: {self.success_count} OK, {self.failed_count} FAILED",
            f"ML Pipelines: {self.ml_success_count} OK, {self.ml_failed_count} FAILED",
        ]

        if self.failed_count > 0:
            lines.append("\nFailed benchmarks:")
            for r in self.benchmarks:
                if not r.success:
                    lines.append(f"  - {r.operation}/{r.mode}: {r.error_message}")

        if self.ml_failed_count > 0:
            lines.append("\nFailed ML pipelines:")
            for r in self.ml_pipelines:
                if not r.success:
                    lines.append(f"  - {r.task_type}/{r.model_name}: {r.error_message}")

        lines.append("=" * 60)
        return "\n".join(lines)


# =============================================================================
# Базовый класс бенчмарка
# =============================================================================


class BaseBenchmark(ABC):
    """Базовый класс для бенчмарков операций."""

    operation_id: str = ""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.n_rows: int | None = None  # Количество строк для чтения (None = все)

    @abstractmethod
    def run_polars_eager(self, data_path: Path) -> Any:
        pass

    @abstractmethod
    def run_polars_lazy(self, data_path: Path) -> Any:
        pass

    @abstractmethod
    def run_polars_streaming(self, data_path: Path) -> Any:
        pass

    @abstractmethod
    def run_duckdb(self, data_path: Path) -> Any:
        pass

    def _limit_df(self, df: pl.DataFrame) -> pl.DataFrame:
        """Ограничивает DataFrame до n_rows строк."""
        if self.n_rows is not None and len(df) > self.n_rows:
            return df.head(self.n_rows)
        return df

    def _limit_lazy(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """Ограничивает LazyFrame до n_rows строк."""
        if self.n_rows is not None:
            return lf.head(self.n_rows)
        return lf

    def _limit_sql(self) -> str:
        """Возвращает SQL LIMIT clause."""
        if self.n_rows is not None:
            return f" LIMIT {self.n_rows}"
        return ""

    def _run_single(
        self,
        func: Callable,
        data_path: Path,
        framework: str,
        mode: str,
        data_size: str,
        run_number: int,
        is_cold: bool,
    ) -> BenchmarkResult:
        gc.collect()
        collector = MetricsCollector(interval_ms=self.config.metrics_interval_ms)
        collector.start()
        start_time = time.perf_counter()
        success = True
        error_message = None
        try:
            func(data_path)
        except Exception as e:
            success = False
            error_message = str(e)
        finally:
            end_time = time.perf_counter()
            metrics = collector.stop()
        return BenchmarkResult(
            operation=self.operation_id,
            framework=framework,
            mode=mode,
            data_size=data_size,
            run_number=run_number,
            is_cold=is_cold,
            wall_time_seconds=end_time - start_time,
            peak_memory_mb=metrics.peak_memory_mb,
            cpu_percent=metrics.avg_cpu_percent,
            success=success,
            error_message=error_message,
        )

    def run_all(
        self, data_path: Path, data_size: str, n_rows: int | None = None
    ) -> list[BenchmarkResult]:
        """Запускает бенчмарк для всех фреймворков и режимов.

        Args:
            data_path: Путь к parquet-файлу.
            data_size: Метка размера данных (например, "1GB").
            n_rows: Количество строк для чтения. None = весь файл.

        Returns:
            Список результатов бенчмарков.
        """
        # Устанавливаем n_rows для использования в run_* методах
        self.n_rows = n_rows

        # Вызов setup если определён
        if hasattr(self, "setup"):
            self.setup(data_path)

        results: list[BenchmarkResult] = []
        runners = [
            (self.run_polars_eager, "polars", "eager"),
            (self.run_polars_lazy, "polars", "lazy"),
            (self.run_polars_streaming, "polars", "streaming"),
            (self.run_duckdb, "duckdb", "sql"),
        ]
        for func, framework, mode in runners:
            if framework not in self.config.frameworks:
                continue
            if framework == "polars" and mode not in self.config.polars_modes:
                continue
            for run_number in range(1, self.config.n_runs + 1):
                is_cold = run_number <= self.config.cold_runs
                result = self._run_single(
                    func=func,
                    data_path=data_path,
                    framework=framework,
                    mode=mode,
                    data_size=data_size,
                    run_number=run_number,
                    is_cold=is_cold,
                )
                results.append(result)
        return results


# =============================================================================
# I/O Benchmarks
# =============================================================================


class ReadParquetBenchmark(BaseBenchmark):
    operation_id = "io_read_parquet"

    def run_polars_eager(self, data_path: Path) -> Any:
        df = pl.read_parquet(data_path)
        return self._limit_df(df)

    def run_polars_lazy(self, data_path: Path) -> Any:
        lf = pl.scan_parquet(data_path)
        return self._limit_lazy(lf).collect()

    def run_polars_streaming(self, data_path: Path) -> Any:
        lf = pl.scan_parquet(data_path)
        return self._limit_lazy(lf).collect(engine="streaming")

    def run_duckdb(self, data_path: Path) -> Any:
        return duckdb.sql(f"SELECT * FROM '{data_path}'{self._limit_sql()}").fetchall()


class ReadCSVBenchmark(BaseBenchmark):
    operation_id = "io_read_csv"

    def run_polars_eager(self, data_path: Path) -> Any:
        df = pl.read_csv(data_path)
        return self._limit_df(df)

    def run_polars_lazy(self, data_path: Path) -> Any:
        lf = pl.scan_csv(data_path)
        return self._limit_lazy(lf).collect()

    def run_polars_streaming(self, data_path: Path) -> Any:
        lf = pl.scan_csv(data_path)
        return self._limit_lazy(lf).collect(engine="streaming")

    def run_duckdb(self, data_path: Path) -> Any:
        return duckdb.sql(f"SELECT * FROM '{data_path}'{self._limit_sql()}").fetchall()


class WriteParquetBenchmark(BaseBenchmark):
    operation_id = "io_write_parquet"

    def __init__(self, config: BenchmarkConfig):
        super().__init__(config)
        self._df_polars: pl.DataFrame | None = None
        self._temp_dir = tempfile.mkdtemp()

    def setup(self, data_path: Path):
        df = pl.read_parquet(data_path)
        self._df_polars = self._limit_df(df)

    def run_polars_eager(self, data_path: Path) -> Any:
        out_path = Path(self._temp_dir) / "out_eager.parquet"
        self._df_polars.write_parquet(out_path)

    def run_polars_lazy(self, data_path: Path) -> Any:
        out_path = Path(self._temp_dir) / "out_lazy.parquet"
        self._df_polars.lazy().sink_parquet(out_path)

    def run_polars_streaming(self, data_path: Path) -> Any:
        out_path = Path(self._temp_dir) / "out_streaming.parquet"
        self._df_polars.lazy().sink_parquet(out_path, engine="streaming")

    def run_duckdb(self, data_path: Path) -> Any:
        out_path = Path(self._temp_dir) / "out_duckdb.parquet"
        duckdb.sql(f"COPY (SELECT * FROM '{data_path}'{self._limit_sql()}) TO '{out_path}' (FORMAT PARQUET)")


class WriteCSVBenchmark(BaseBenchmark):
    operation_id = "io_write_csv"

    def __init__(self, config: BenchmarkConfig):
        super().__init__(config)
        self._df_polars: pl.DataFrame | None = None
        self._temp_dir = tempfile.mkdtemp()

    def setup(self, data_path: Path):
        df = pl.read_parquet(data_path)
        self._df_polars = self._limit_df(df)

    def run_polars_eager(self, data_path: Path) -> Any:
        out_path = Path(self._temp_dir) / "out_eager.csv"
        self._df_polars.write_csv(out_path)

    def run_polars_lazy(self, data_path: Path) -> Any:
        out_path = Path(self._temp_dir) / "out_lazy.csv"
        self._df_polars.lazy().sink_csv(out_path)

    def run_polars_streaming(self, data_path: Path) -> Any:
        out_path = Path(self._temp_dir) / "out_streaming.csv"
        self._df_polars.lazy().sink_csv(out_path, engine="streaming")

    def run_duckdb(self, data_path: Path) -> Any:
        out_path = Path(self._temp_dir) / "out_duckdb.csv"
        duckdb.sql(f"COPY (SELECT * FROM '{data_path}'{self._limit_sql()}) TO '{out_path}' (FORMAT CSV)")


# =============================================================================
# Filter Benchmarks
# =============================================================================


class FilterSimpleBenchmark(BaseBenchmark):
    operation_id = "filter_simple"

    def run_polars_eager(self, data_path: Path) -> Any:
        df = pl.read_parquet(data_path)
        df = self._limit_df(df)
        return df.filter(pl.col("numeric_0") > 0.5)

    def run_polars_lazy(self, data_path: Path) -> Any:
        lf = pl.scan_parquet(data_path)
        lf = self._limit_lazy(lf)
        return lf.filter(pl.col("numeric_0") > 0.5).collect()

    def run_polars_streaming(self, data_path: Path) -> Any:
        lf = pl.scan_parquet(data_path)
        lf = self._limit_lazy(lf)
        return lf.filter(pl.col("numeric_0") > 0.5).collect(engine="streaming")

    def run_duckdb(self, data_path: Path) -> Any:
        limit = self._limit_sql()
        subq = f"SELECT * FROM '{data_path}'{limit}" if limit else f"'{data_path}'"
        return duckdb.sql(f"SELECT * FROM ({subq}) WHERE numeric_0 > 0.5").fetchall()


class FilterComplexBenchmark(BaseBenchmark):
    operation_id = "filter_complex"

    def run_polars_eager(self, data_path: Path) -> Any:
        df = pl.read_parquet(data_path)
        df = self._limit_df(df)
        return df.filter(
            ((pl.col("numeric_0") > 0.3) & (pl.col("numeric_1") < 0.7))
            | (pl.col("numeric_2") > 0.9)
        )

    def run_polars_lazy(self, data_path: Path) -> Any:
        lf = pl.scan_parquet(data_path)
        lf = self._limit_lazy(lf)
        return lf.filter(
            ((pl.col("numeric_0") > 0.3) & (pl.col("numeric_1") < 0.7))
            | (pl.col("numeric_2") > 0.9)
        ).collect()

    def run_polars_streaming(self, data_path: Path) -> Any:
        lf = pl.scan_parquet(data_path)
        lf = self._limit_lazy(lf)
        return lf.filter(
            ((pl.col("numeric_0") > 0.3) & (pl.col("numeric_1") < 0.7))
            | (pl.col("numeric_2") > 0.9)
        ).collect(engine="streaming")

    def run_duckdb(self, data_path: Path) -> Any:
        limit = self._limit_sql()
        subq = f"SELECT * FROM '{data_path}'{limit}" if limit else f"'{data_path}'"
        return duckdb.sql(
            f"""SELECT * FROM ({subq})
            WHERE (numeric_0 > 0.3 AND numeric_1 < 0.7) OR numeric_2 > 0.9"""
        ).fetchall()


# =============================================================================
# Aggregation Benchmarks
# =============================================================================


class AggSingleBenchmark(BaseBenchmark):
    operation_id = "agg_single"

    def run_polars_eager(self, data_path: Path) -> Any:
        df = pl.read_parquet(data_path)
        df = self._limit_df(df)
        return df.group_by("cat_category_0").agg(pl.col("numeric_0").sum())

    def run_polars_lazy(self, data_path: Path) -> Any:
        lf = pl.scan_parquet(data_path)
        lf = self._limit_lazy(lf)
        return lf.group_by("cat_category_0").agg(pl.col("numeric_0").sum()).collect()

    def run_polars_streaming(self, data_path: Path) -> Any:
        lf = pl.scan_parquet(data_path)
        lf = self._limit_lazy(lf)
        return lf.group_by("cat_category_0").agg(pl.col("numeric_0").sum()).collect(engine="streaming")

    def run_duckdb(self, data_path: Path) -> Any:
        limit = self._limit_sql()
        subq = f"SELECT * FROM '{data_path}'{limit}" if limit else f"'{data_path}'"
        return duckdb.sql(
            f"SELECT cat_category_0, SUM(numeric_0) FROM ({subq}) GROUP BY cat_category_0"
        ).fetchall()


class AggMultiBenchmark(BaseBenchmark):
    operation_id = "agg_multi"

    def run_polars_eager(self, data_path: Path) -> Any:
        df = pl.read_parquet(data_path)
        df = self._limit_df(df)
        return df.group_by("cat_category_0").agg(
            pl.col("numeric_0").sum(),
            pl.col("numeric_0").mean(),
            pl.col("numeric_0").count(),
            pl.col("numeric_0").std(),
            pl.col("numeric_0").min(),
            pl.col("numeric_0").max(),
        )

    def run_polars_lazy(self, data_path: Path) -> Any:
        lf = pl.scan_parquet(data_path)
        lf = self._limit_lazy(lf)
        return lf.group_by("cat_category_0").agg(
            pl.col("numeric_0").sum(),
            pl.col("numeric_0").mean(),
            pl.col("numeric_0").count(),
            pl.col("numeric_0").std(),
            pl.col("numeric_0").min(),
            pl.col("numeric_0").max(),
        ).collect()

    def run_polars_streaming(self, data_path: Path) -> Any:
        lf = pl.scan_parquet(data_path)
        lf = self._limit_lazy(lf)
        return lf.group_by("cat_category_0").agg(
            pl.col("numeric_0").sum(),
            pl.col("numeric_0").mean(),
            pl.col("numeric_0").count(),
            pl.col("numeric_0").std(),
            pl.col("numeric_0").min(),
            pl.col("numeric_0").max(),
        ).collect(engine="streaming")

    def run_duckdb(self, data_path: Path) -> Any:
        limit = self._limit_sql()
        subq = f"SELECT * FROM '{data_path}'{limit}" if limit else f"'{data_path}'"
        return duckdb.sql(
            f"""SELECT cat_category_0, SUM(numeric_0), AVG(numeric_0), COUNT(numeric_0),
            STDDEV(numeric_0), MIN(numeric_0), MAX(numeric_0)
            FROM ({subq}) GROUP BY cat_category_0"""
        ).fetchall()


class AggNuniqueBenchmark(BaseBenchmark):
    operation_id = "agg_nunique"

    def run_polars_eager(self, data_path: Path) -> Any:
        df = pl.read_parquet(data_path)
        df = self._limit_df(df)
        return df.group_by("cat_category_0").agg(pl.col("cat_category_1").n_unique())

    def run_polars_lazy(self, data_path: Path) -> Any:
        lf = pl.scan_parquet(data_path)
        lf = self._limit_lazy(lf)
        return lf.group_by("cat_category_0").agg(pl.col("cat_category_1").n_unique()).collect()

    def run_polars_streaming(self, data_path: Path) -> Any:
        lf = pl.scan_parquet(data_path)
        lf = self._limit_lazy(lf)
        return lf.group_by("cat_category_0").agg(pl.col("cat_category_1").n_unique()).collect(engine="streaming")

    def run_duckdb(self, data_path: Path) -> Any:
        limit = self._limit_sql()
        subq = f"SELECT * FROM '{data_path}'{limit}" if limit else f"'{data_path}'"
        return duckdb.sql(
            f"SELECT cat_category_0, COUNT(DISTINCT cat_category_1) FROM ({subq}) GROUP BY cat_category_0"
        ).fetchall()


class AggMultiKeysBenchmark(BaseBenchmark):
    operation_id = "agg_multiple_keys"

    def run_polars_eager(self, data_path: Path) -> Any:
        df = pl.read_parquet(data_path)
        df = self._limit_df(df)
        return df.group_by("cat_category_0", "cat_category_1").agg(pl.col("numeric_0").sum())

    def run_polars_lazy(self, data_path: Path) -> Any:
        lf = pl.scan_parquet(data_path)
        lf = self._limit_lazy(lf)
        return lf.group_by("cat_category_0", "cat_category_1").agg(pl.col("numeric_0").sum()).collect()

    def run_polars_streaming(self, data_path: Path) -> Any:
        lf = pl.scan_parquet(data_path)
        lf = self._limit_lazy(lf)
        return lf.group_by("cat_category_0", "cat_category_1").agg(pl.col("numeric_0").sum()).collect(engine="streaming")

    def run_duckdb(self, data_path: Path) -> Any:
        limit = self._limit_sql()
        subq = f"SELECT * FROM '{data_path}'{limit}" if limit else f"'{data_path}'"
        return duckdb.sql(
            f"SELECT cat_category_0, cat_category_1, SUM(numeric_0) FROM ({subq}) GROUP BY cat_category_0, cat_category_1"
        ).fetchall()


# =============================================================================
# Join Benchmarks
# =============================================================================


class JoinBenchmarkBase(BaseBenchmark):
    """Базовый класс для join бенчмарков."""

    def __init__(self, config: BenchmarkConfig):
        super().__init__(config)
        self._right_df: pl.DataFrame | None = None
        self._right_path: Path | None = None

    def setup(self, data_path: Path):
        df = pl.read_parquet(data_path)
        df = self._limit_df(df)
        self._right_df = df.select("cat_category_0", "numeric_0").unique(subset=["cat_category_0"])
        self._right_path = data_path.parent / "right_table.parquet"
        self._right_df.write_parquet(self._right_path)


class JoinInnerBenchmark(JoinBenchmarkBase):
    operation_id = "join_inner"

    def run_polars_eager(self, data_path: Path) -> Any:
        df = pl.read_parquet(data_path)
        df = self._limit_df(df)
        return df.join(self._right_df, on="cat_category_0", how="inner")

    def run_polars_lazy(self, data_path: Path) -> Any:
        lf = pl.scan_parquet(data_path)
        lf = self._limit_lazy(lf)
        return lf.join(self._right_df.lazy(), on="cat_category_0", how="inner").collect()

    def run_polars_streaming(self, data_path: Path) -> Any:
        lf = pl.scan_parquet(data_path)
        lf = self._limit_lazy(lf)
        return lf.join(self._right_df.lazy(), on="cat_category_0", how="inner").collect(engine="streaming")

    def run_duckdb(self, data_path: Path) -> Any:
        limit = self._limit_sql()
        subq = f"(SELECT * FROM '{data_path}'{limit})" if limit else f"'{data_path}'"
        return duckdb.sql(
            f"""SELECT * FROM {subq} AS l
            INNER JOIN '{self._right_path}' AS r ON l.cat_category_0 = r.cat_category_0"""
        ).fetchall()


class JoinLeftBenchmark(JoinBenchmarkBase):
    operation_id = "join_left"

    def run_polars_eager(self, data_path: Path) -> Any:
        df = pl.read_parquet(data_path)
        df = self._limit_df(df)
        return df.join(self._right_df, on="cat_category_0", how="left")

    def run_polars_lazy(self, data_path: Path) -> Any:
        lf = pl.scan_parquet(data_path)
        lf = self._limit_lazy(lf)
        return lf.join(self._right_df.lazy(), on="cat_category_0", how="left").collect()

    def run_polars_streaming(self, data_path: Path) -> Any:
        lf = pl.scan_parquet(data_path)
        lf = self._limit_lazy(lf)
        return lf.join(self._right_df.lazy(), on="cat_category_0", how="left").collect(engine="streaming")

    def run_duckdb(self, data_path: Path) -> Any:
        limit = self._limit_sql()
        subq = f"(SELECT * FROM '{data_path}'{limit})" if limit else f"'{data_path}'"
        return duckdb.sql(
            f"""SELECT * FROM {subq} AS l
            LEFT JOIN '{self._right_path}' AS r ON l.cat_category_0 = r.cat_category_0"""
        ).fetchall()


class JoinOuterBenchmark(JoinBenchmarkBase):
    operation_id = "join_outer"

    def run_polars_eager(self, data_path: Path) -> Any:
        df = pl.read_parquet(data_path)
        df = self._limit_df(df)
        return df.join(self._right_df, on="cat_category_0", how="full")

    def run_polars_lazy(self, data_path: Path) -> Any:
        lf = pl.scan_parquet(data_path)
        lf = self._limit_lazy(lf)
        return lf.join(self._right_df.lazy(), on="cat_category_0", how="full").collect()

    def run_polars_streaming(self, data_path: Path) -> Any:
        lf = pl.scan_parquet(data_path)
        lf = self._limit_lazy(lf)
        return lf.join(self._right_df.lazy(), on="cat_category_0", how="full").collect(engine="streaming")

    def run_duckdb(self, data_path: Path) -> Any:
        limit = self._limit_sql()
        subq = f"(SELECT * FROM '{data_path}'{limit})" if limit else f"'{data_path}'"
        return duckdb.sql(
            f"""SELECT * FROM {subq} AS l
            FULL OUTER JOIN '{self._right_path}' AS r ON l.cat_category_0 = r.cat_category_0"""
        ).fetchall()


class JoinAntiBenchmark(JoinBenchmarkBase):
    operation_id = "join_anti"

    def run_polars_eager(self, data_path: Path) -> Any:
        df = pl.read_parquet(data_path)
        df = self._limit_df(df)
        return df.join(self._right_df, on="cat_category_0", how="anti")

    def run_polars_lazy(self, data_path: Path) -> Any:
        lf = pl.scan_parquet(data_path)
        lf = self._limit_lazy(lf)
        return lf.join(self._right_df.lazy(), on="cat_category_0", how="anti").collect()

    def run_polars_streaming(self, data_path: Path) -> Any:
        lf = pl.scan_parquet(data_path)
        lf = self._limit_lazy(lf)
        return lf.join(self._right_df.lazy(), on="cat_category_0", how="anti").collect(engine="streaming")

    def run_duckdb(self, data_path: Path) -> Any:
        limit = self._limit_sql()
        subq = f"(SELECT * FROM '{data_path}'{limit})" if limit else f"'{data_path}'"
        return duckdb.sql(
            f"""SELECT * FROM {subq} AS l
            WHERE NOT EXISTS (SELECT 1 FROM '{self._right_path}' AS r WHERE l.cat_category_0 = r.cat_category_0)"""
        ).fetchall()


class JoinSemiBenchmark(JoinBenchmarkBase):
    operation_id = "join_semi"

    def run_polars_eager(self, data_path: Path) -> Any:
        df = pl.read_parquet(data_path)
        df = self._limit_df(df)
        return df.join(self._right_df, on="cat_category_0", how="semi")

    def run_polars_lazy(self, data_path: Path) -> Any:
        lf = pl.scan_parquet(data_path)
        lf = self._limit_lazy(lf)
        return lf.join(self._right_df.lazy(), on="cat_category_0", how="semi").collect()

    def run_polars_streaming(self, data_path: Path) -> Any:
        lf = pl.scan_parquet(data_path)
        lf = self._limit_lazy(lf)
        return lf.join(self._right_df.lazy(), on="cat_category_0", how="semi").collect(engine="streaming")

    def run_duckdb(self, data_path: Path) -> Any:
        limit = self._limit_sql()
        subq = f"(SELECT * FROM '{data_path}'{limit})" if limit else f"'{data_path}'"
        return duckdb.sql(
            f"""SELECT * FROM {subq} AS l
            WHERE EXISTS (SELECT 1 FROM '{self._right_path}' AS r WHERE l.cat_category_0 = r.cat_category_0)"""
        ).fetchall()


# =============================================================================
# Sort Benchmarks
# =============================================================================


class SortSingleBenchmark(BaseBenchmark):
    operation_id = "sort_single"

    def run_polars_eager(self, data_path: Path) -> Any:
        df = pl.read_parquet(data_path)
        df = self._limit_df(df)
        return df.sort("numeric_0")

    def run_polars_lazy(self, data_path: Path) -> Any:
        lf = pl.scan_parquet(data_path)
        lf = self._limit_lazy(lf)
        return lf.sort("numeric_0").collect()

    def run_polars_streaming(self, data_path: Path) -> Any:
        lf = pl.scan_parquet(data_path)
        lf = self._limit_lazy(lf)
        return lf.sort("numeric_0").collect(engine="streaming")

    def run_duckdb(self, data_path: Path) -> Any:
        limit = self._limit_sql()
        subq = f"SELECT * FROM '{data_path}'{limit}" if limit else f"'{data_path}'"
        return duckdb.sql(f"SELECT * FROM ({subq}) ORDER BY numeric_0").fetchall()


class SortMultiBenchmark(BaseBenchmark):
    operation_id = "sort_multi"

    def run_polars_eager(self, data_path: Path) -> Any:
        df = pl.read_parquet(data_path)
        df = self._limit_df(df)
        return df.sort("cat_category_0", "numeric_0", "numeric_1")

    def run_polars_lazy(self, data_path: Path) -> Any:
        lf = pl.scan_parquet(data_path)
        lf = self._limit_lazy(lf)
        return lf.sort("cat_category_0", "numeric_0", "numeric_1").collect()

    def run_polars_streaming(self, data_path: Path) -> Any:
        lf = pl.scan_parquet(data_path)
        lf = self._limit_lazy(lf)
        return lf.sort("cat_category_0", "numeric_0", "numeric_1").collect(engine="streaming")

    def run_duckdb(self, data_path: Path) -> Any:
        limit = self._limit_sql()
        subq = f"SELECT * FROM '{data_path}'{limit}" if limit else f"'{data_path}'"
        return duckdb.sql(
            f"SELECT * FROM ({subq}) ORDER BY cat_category_0, numeric_0, numeric_1"
        ).fetchall()


class SortTopKBenchmark(BaseBenchmark):
    operation_id = "sort_topk"

    def run_polars_eager(self, data_path: Path) -> Any:
        df = pl.read_parquet(data_path)
        df = self._limit_df(df)
        return df.top_k(1000, by="numeric_0")

    def run_polars_lazy(self, data_path: Path) -> Any:
        lf = pl.scan_parquet(data_path)
        lf = self._limit_lazy(lf)
        return lf.top_k(1000, by="numeric_0").collect()

    def run_polars_streaming(self, data_path: Path) -> Any:
        lf = pl.scan_parquet(data_path)
        lf = self._limit_lazy(lf)
        return lf.top_k(1000, by="numeric_0").collect(engine="streaming")

    def run_duckdb(self, data_path: Path) -> Any:
        limit = self._limit_sql()
        subq = f"SELECT * FROM '{data_path}'{limit}" if limit else f"'{data_path}'"
        return duckdb.sql(
            f"SELECT * FROM ({subq}) ORDER BY numeric_0 DESC LIMIT 1000"
        ).fetchall()


# =============================================================================
# Window Benchmarks
# =============================================================================


class WindowRankBenchmark(BaseBenchmark):
    operation_id = "window_rank"

    def run_polars_eager(self, data_path: Path) -> Any:
        df = pl.read_parquet(data_path)
        df = self._limit_df(df)
        return df.with_columns(pl.col("numeric_0").rank().over("cat_category_0").alias("rank"))

    def run_polars_lazy(self, data_path: Path) -> Any:
        lf = pl.scan_parquet(data_path)
        lf = self._limit_lazy(lf)
        return lf.with_columns(pl.col("numeric_0").rank().over("cat_category_0").alias("rank")).collect()

    def run_polars_streaming(self, data_path: Path) -> Any:
        lf = pl.scan_parquet(data_path)
        lf = self._limit_lazy(lf)
        return lf.with_columns(pl.col("numeric_0").rank().over("cat_category_0").alias("rank")).collect(engine="streaming")

    def run_duckdb(self, data_path: Path) -> Any:
        limit = self._limit_sql()
        subq = f"SELECT * FROM '{data_path}'{limit}" if limit else f"'{data_path}'"
        return duckdb.sql(
            f"""SELECT *, RANK() OVER (PARTITION BY cat_category_0 ORDER BY numeric_0) as rank
            FROM ({subq})"""
        ).fetchall()


class WindowRollingMeanBenchmark(BaseBenchmark):
    operation_id = "window_rolling_mean"

    def run_polars_eager(self, data_path: Path) -> Any:
        df = pl.read_parquet(data_path)
        df = self._limit_df(df).sort("id")
        return df.with_columns(pl.col("numeric_0").rolling_mean(window_size=7).alias("rolling_mean"))

    def run_polars_lazy(self, data_path: Path) -> Any:
        lf = pl.scan_parquet(data_path)
        lf = self._limit_lazy(lf).sort("id")
        return lf.with_columns(pl.col("numeric_0").rolling_mean(window_size=7).alias("rolling_mean")).collect()

    def run_polars_streaming(self, data_path: Path) -> Any:
        lf = pl.scan_parquet(data_path)
        lf = self._limit_lazy(lf).sort("id")
        return lf.with_columns(pl.col("numeric_0").rolling_mean(window_size=7).alias("rolling_mean")).collect(engine="streaming")

    def run_duckdb(self, data_path: Path) -> Any:
        limit = self._limit_sql()
        subq = f"SELECT * FROM '{data_path}'{limit}" if limit else f"'{data_path}'"
        return duckdb.sql(
            f"""SELECT *, AVG(numeric_0) OVER (ORDER BY id ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) as rolling_mean
            FROM ({subq})"""
        ).fetchall()


class WindowLagBenchmark(BaseBenchmark):
    operation_id = "window_lag"

    def run_polars_eager(self, data_path: Path) -> Any:
        df = pl.read_parquet(data_path)
        df = self._limit_df(df).sort("id")
        return df.with_columns(pl.col("numeric_0").shift(1).alias("lag_1"))

    def run_polars_lazy(self, data_path: Path) -> Any:
        lf = pl.scan_parquet(data_path)
        lf = self._limit_lazy(lf).sort("id")
        return lf.with_columns(pl.col("numeric_0").shift(1).alias("lag_1")).collect()

    def run_polars_streaming(self, data_path: Path) -> Any:
        lf = pl.scan_parquet(data_path)
        lf = self._limit_lazy(lf).sort("id")
        return lf.with_columns(pl.col("numeric_0").shift(1).alias("lag_1")).collect(engine="streaming")

    def run_duckdb(self, data_path: Path) -> Any:
        limit = self._limit_sql()
        subq = f"SELECT * FROM '{data_path}'{limit}" if limit else f"'{data_path}'"
        return duckdb.sql(
            f"""SELECT *, LAG(numeric_0, 1) OVER (ORDER BY id) as lag_1
            FROM ({subq})"""
        ).fetchall()


class WindowCumsumBenchmark(BaseBenchmark):
    operation_id = "window_cumsum"

    def run_polars_eager(self, data_path: Path) -> Any:
        df = pl.read_parquet(data_path)
        df = self._limit_df(df).sort("id")
        return df.with_columns(pl.col("numeric_0").cum_sum().alias("cumsum"))

    def run_polars_lazy(self, data_path: Path) -> Any:
        lf = pl.scan_parquet(data_path)
        lf = self._limit_lazy(lf).sort("id")
        return lf.with_columns(pl.col("numeric_0").cum_sum().alias("cumsum")).collect()

    def run_polars_streaming(self, data_path: Path) -> Any:
        lf = pl.scan_parquet(data_path)
        lf = self._limit_lazy(lf).sort("id")
        return lf.with_columns(pl.col("numeric_0").cum_sum().alias("cumsum")).collect(engine="streaming")

    def run_duckdb(self, data_path: Path) -> Any:
        limit = self._limit_sql()
        subq = f"SELECT * FROM '{data_path}'{limit}" if limit else f"'{data_path}'"
        return duckdb.sql(
            f"""SELECT *, SUM(numeric_0) OVER (ORDER BY id ROWS UNBOUNDED PRECEDING) as cumsum
            FROM ({subq})"""
        ).fetchall()


# =============================================================================
# Vector Operations Benchmarks
# =============================================================================


class VecAddBenchmark(BaseBenchmark):
    operation_id = "vec_add"

    def run_polars_eager(self, data_path: Path) -> Any:
        df = pl.read_parquet(data_path)
        df = self._limit_df(df)
        return df.with_columns((pl.col("numeric_0") + pl.col("numeric_1")).alias("sum"))

    def run_polars_lazy(self, data_path: Path) -> Any:
        lf = pl.scan_parquet(data_path)
        lf = self._limit_lazy(lf)
        return lf.with_columns((pl.col("numeric_0") + pl.col("numeric_1")).alias("sum")).collect()

    def run_polars_streaming(self, data_path: Path) -> Any:
        lf = pl.scan_parquet(data_path)
        lf = self._limit_lazy(lf)
        return lf.with_columns((pl.col("numeric_0") + pl.col("numeric_1")).alias("sum")).collect(engine="streaming")

    def run_duckdb(self, data_path: Path) -> Any:
        limit = self._limit_sql()
        subq = f"SELECT * FROM '{data_path}'{limit}" if limit else f"'{data_path}'"
        return duckdb.sql(f"SELECT *, numeric_0 + numeric_1 as sum FROM ({subq})").fetchall()


class VecCompoundBenchmark(BaseBenchmark):
    operation_id = "vec_compound"

    def run_polars_eager(self, data_path: Path) -> Any:
        df = pl.read_parquet(data_path)
        df = self._limit_df(df)
        return df.with_columns(
            ((pl.col("numeric_0") + pl.col("numeric_1")) * pl.col("numeric_2") - pl.col("numeric_3")).alias(
                "compound"
            )
        )

    def run_polars_lazy(self, data_path: Path) -> Any:
        lf = pl.scan_parquet(data_path)
        lf = self._limit_lazy(lf)
        return lf.with_columns(
            ((pl.col("numeric_0") + pl.col("numeric_1")) * pl.col("numeric_2") - pl.col("numeric_3")).alias(
                "compound"
            )
        ).collect()

    def run_polars_streaming(self, data_path: Path) -> Any:
        lf = pl.scan_parquet(data_path)
        lf = self._limit_lazy(lf)
        return lf.with_columns(
            ((pl.col("numeric_0") + pl.col("numeric_1")) * pl.col("numeric_2") - pl.col("numeric_3")).alias(
                "compound"
            )
        ).collect(engine="streaming")

    def run_duckdb(self, data_path: Path) -> Any:
        limit = self._limit_sql()
        subq = f"SELECT * FROM '{data_path}'{limit}" if limit else f"'{data_path}'"
        return duckdb.sql(
            f"SELECT *, (numeric_0 + numeric_1) * numeric_2 - numeric_3 as compound FROM ({subq})"
        ).fetchall()


# =============================================================================
# Transform Benchmarks
# =============================================================================


class TransformConcatVBenchmark(BaseBenchmark):
    operation_id = "transform_concat_v"

    def run_polars_eager(self, data_path: Path) -> Any:
        df = pl.read_parquet(data_path)
        df = self._limit_df(df)
        df1 = df.head(len(df) // 2)
        df2 = df.tail(len(df) // 2)
        return pl.concat([df1, df2])

    def run_polars_lazy(self, data_path: Path) -> Any:
        lf = pl.scan_parquet(data_path)
        lf = self._limit_lazy(lf)
        n = lf.select(pl.len()).collect().item()
        lf = pl.scan_parquet(data_path)
        lf = self._limit_lazy(lf)
        lf1 = lf.head(n // 2)
        lf2 = lf.tail(n // 2)
        return pl.concat([lf1, lf2]).collect()

    def run_polars_streaming(self, data_path: Path) -> Any:
        lf = pl.scan_parquet(data_path)
        lf = self._limit_lazy(lf)
        n = lf.select(pl.len()).collect().item()
        lf = pl.scan_parquet(data_path)
        lf = self._limit_lazy(lf)
        lf1 = lf.head(n // 2)
        lf2 = lf.tail(n // 2)
        return pl.concat([lf1, lf2]).collect(engine="streaming")

    def run_duckdb(self, data_path: Path) -> Any:
        limit = self._limit_sql()
        subq = f"SELECT * FROM '{data_path}'{limit}" if limit else f"'{data_path}'"
        return duckdb.sql(
            f"""SELECT * FROM ({subq}) LIMIT (SELECT COUNT(*)/2 FROM ({subq}))
            UNION ALL
            SELECT * FROM ({subq}) OFFSET (SELECT COUNT(*)/2 FROM ({subq}))"""
        ).fetchall()


class TransformConcatHBenchmark(BaseBenchmark):
    operation_id = "transform_concat_h"

    def run_polars_eager(self, data_path: Path) -> Any:
        df = pl.read_parquet(data_path)
        df = self._limit_df(df)
        df1 = df.select("id", "numeric_0", "numeric_1")
        df2 = df.select("numeric_2", "numeric_3")
        return pl.concat([df1, df2], how="horizontal")

    def run_polars_lazy(self, data_path: Path) -> Any:
        lf = pl.scan_parquet(data_path)
        lf = self._limit_lazy(lf)
        lf1 = lf.select("id", "numeric_0", "numeric_1")
        lf2 = lf.select("numeric_2", "numeric_3")
        return pl.concat([lf1, lf2], how="horizontal").collect()

    def run_polars_streaming(self, data_path: Path) -> Any:
        lf = pl.scan_parquet(data_path)
        lf = self._limit_lazy(lf)
        lf1 = lf.select("id", "numeric_0", "numeric_1")
        lf2 = lf.select("numeric_2", "numeric_3")
        return pl.concat([lf1, lf2], how="horizontal").collect(engine="streaming")

    def run_duckdb(self, data_path: Path) -> Any:
        limit = self._limit_sql()
        subq = f"SELECT * FROM '{data_path}'{limit}" if limit else f"'{data_path}'"
        return duckdb.sql(
            f"SELECT id, numeric_0, numeric_1, numeric_2, numeric_3 FROM ({subq})"
        ).fetchall()


class TransformUnpivotBenchmark(BaseBenchmark):
    operation_id = "transform_unpivot"

    def run_polars_eager(self, data_path: Path) -> Any:
        df = pl.read_parquet(data_path)
        df = self._limit_df(df)
        return df.unpivot(
            index=["id", "cat_category_0"],
            on=["numeric_0", "numeric_1", "numeric_2"],
        )

    def run_polars_lazy(self, data_path: Path) -> Any:
        lf = pl.scan_parquet(data_path)
        lf = self._limit_lazy(lf)
        return lf.unpivot(
            index=["id", "cat_category_0"],
            on=["numeric_0", "numeric_1", "numeric_2"],
        ).collect()

    def run_polars_streaming(self, data_path: Path) -> Any:
        lf = pl.scan_parquet(data_path)
        lf = self._limit_lazy(lf)
        return lf.unpivot(
            index=["id", "cat_category_0"],
            on=["numeric_0", "numeric_1", "numeric_2"],
        ).collect(engine="streaming")

    def run_duckdb(self, data_path: Path) -> Any:
        limit = self._limit_sql()
        subq = f"SELECT * FROM '{data_path}'{limit}" if limit else f"'{data_path}'"
        return duckdb.sql(
            f"""UNPIVOT ({subq})
            ON numeric_0, numeric_1, numeric_2
            INTO NAME variable VALUE value"""
        ).fetchall()


# =============================================================================
# String Benchmarks
# =============================================================================


class StringLowerBenchmark(BaseBenchmark):
    operation_id = "str_lower"

    def run_polars_eager(self, data_path: Path) -> Any:
        df = pl.read_parquet(data_path)
        df = self._limit_df(df)
        return df.with_columns(pl.col("cat_category_0").str.to_lowercase().alias("lower"))

    def run_polars_lazy(self, data_path: Path) -> Any:
        lf = pl.scan_parquet(data_path)
        lf = self._limit_lazy(lf)
        return lf.with_columns(pl.col("cat_category_0").str.to_lowercase().alias("lower")).collect()

    def run_polars_streaming(self, data_path: Path) -> Any:
        lf = pl.scan_parquet(data_path)
        lf = self._limit_lazy(lf)
        return lf.with_columns(pl.col("cat_category_0").str.to_lowercase().alias("lower")).collect(engine="streaming")

    def run_duckdb(self, data_path: Path) -> Any:
        limit = self._limit_sql()
        subq = f"SELECT * FROM '{data_path}'{limit}" if limit else f"'{data_path}'"
        return duckdb.sql(f"SELECT *, lower(cat_category_0) as lower FROM ({subq})").fetchall()


class StringLengthBenchmark(BaseBenchmark):
    operation_id = "str_length"

    def run_polars_eager(self, data_path: Path) -> Any:
        df = pl.read_parquet(data_path)
        df = self._limit_df(df)
        return df.with_columns(pl.col("cat_category_0").str.len_chars().alias("length"))

    def run_polars_lazy(self, data_path: Path) -> Any:
        lf = pl.scan_parquet(data_path)
        lf = self._limit_lazy(lf)
        return lf.with_columns(pl.col("cat_category_0").str.len_chars().alias("length")).collect()

    def run_polars_streaming(self, data_path: Path) -> Any:
        lf = pl.scan_parquet(data_path)
        lf = self._limit_lazy(lf)
        return lf.with_columns(pl.col("cat_category_0").str.len_chars().alias("length")).collect(engine="streaming")

    def run_duckdb(self, data_path: Path) -> Any:
        limit = self._limit_sql()
        subq = f"SELECT * FROM '{data_path}'{limit}" if limit else f"'{data_path}'"
        return duckdb.sql(f"SELECT *, length(cat_category_0) as length FROM ({subq})").fetchall()


# =============================================================================
# Datetime Benchmarks
# =============================================================================


class DatetimeExtractYearBenchmark(BaseBenchmark):
    operation_id = "dt_extract_year"

    def run_polars_eager(self, data_path: Path) -> Any:
        df = pl.read_parquet(data_path)
        df = self._limit_df(df)
        return df.with_columns(pl.col("timestamp").dt.year().alias("year"))

    def run_polars_lazy(self, data_path: Path) -> Any:
        lf = pl.scan_parquet(data_path)
        lf = self._limit_lazy(lf)
        return lf.with_columns(pl.col("timestamp").dt.year().alias("year")).collect()

    def run_polars_streaming(self, data_path: Path) -> Any:
        lf = pl.scan_parquet(data_path)
        lf = self._limit_lazy(lf)
        return lf.with_columns(pl.col("timestamp").dt.year().alias("year")).collect(engine="streaming")

    def run_duckdb(self, data_path: Path) -> Any:
        limit = self._limit_sql()
        subq = f"SELECT * FROM '{data_path}'{limit}" if limit else f"'{data_path}'"
        return duckdb.sql(f"SELECT *, year(timestamp) as year FROM ({subq})").fetchall()


class DatetimeTruncateBenchmark(BaseBenchmark):
    operation_id = "dt_truncate"

    def run_polars_eager(self, data_path: Path) -> Any:
        df = pl.read_parquet(data_path)
        df = self._limit_df(df)
        return df.with_columns(pl.col("timestamp").dt.truncate("1d").alias("truncated"))

    def run_polars_lazy(self, data_path: Path) -> Any:
        lf = pl.scan_parquet(data_path)
        lf = self._limit_lazy(lf)
        return lf.with_columns(pl.col("timestamp").dt.truncate("1d").alias("truncated")).collect()

    def run_polars_streaming(self, data_path: Path) -> Any:
        lf = pl.scan_parquet(data_path)
        lf = self._limit_lazy(lf)
        return lf.with_columns(pl.col("timestamp").dt.truncate("1d").alias("truncated")).collect(engine="streaming")

    def run_duckdb(self, data_path: Path) -> Any:
        limit = self._limit_sql()
        subq = f"SELECT * FROM '{data_path}'{limit}" if limit else f"'{data_path}'"
        return duckdb.sql(
            f"SELECT *, date_trunc('day', timestamp) as truncated FROM ({subq})"
        ).fetchall()


# =============================================================================
# Missing Values Benchmarks
# =============================================================================


class MissingFillnaBenchmark(BaseBenchmark):
    operation_id = "missing_fillna"

    def run_polars_eager(self, data_path: Path) -> Any:
        df = pl.read_parquet(data_path)
        df = self._limit_df(df)
        return df.with_columns(pl.col("numeric_0").fill_null(0).alias("filled"))

    def run_polars_lazy(self, data_path: Path) -> Any:
        lf = pl.scan_parquet(data_path)
        lf = self._limit_lazy(lf)
        return lf.with_columns(pl.col("numeric_0").fill_null(0).alias("filled")).collect()

    def run_polars_streaming(self, data_path: Path) -> Any:
        lf = pl.scan_parquet(data_path)
        lf = self._limit_lazy(lf)
        return lf.with_columns(pl.col("numeric_0").fill_null(0).alias("filled")).collect(engine="streaming")

    def run_duckdb(self, data_path: Path) -> Any:
        limit = self._limit_sql()
        subq = f"SELECT * FROM '{data_path}'{limit}" if limit else f"'{data_path}'"
        return duckdb.sql(
            f"SELECT *, COALESCE(numeric_0, 0) as filled FROM ({subq})"
        ).fetchall()


class MissingDropnaBenchmark(BaseBenchmark):
    operation_id = "missing_dropna"

    def run_polars_eager(self, data_path: Path) -> Any:
        df = pl.read_parquet(data_path)
        df = self._limit_df(df)
        return df.drop_nulls()

    def run_polars_lazy(self, data_path: Path) -> Any:
        lf = pl.scan_parquet(data_path)
        lf = self._limit_lazy(lf)
        return lf.drop_nulls().collect()

    def run_polars_streaming(self, data_path: Path) -> Any:
        lf = pl.scan_parquet(data_path)
        lf = self._limit_lazy(lf)
        return lf.drop_nulls().collect(engine="streaming")

    def run_duckdb(self, data_path: Path) -> Any:
        limit = self._limit_sql()
        subq = f"SELECT * FROM '{data_path}'{limit}" if limit else f"'{data_path}'"
        df = duckdb.sql(f"SELECT * FROM ({subq})").pl()
        cols = df.columns
        where_clause = " AND ".join(f"{c} IS NOT NULL" for c in cols)
        return duckdb.sql(f"SELECT * FROM ({subq}) WHERE {where_clause}").fetchall()


# =============================================================================
# Stats Benchmarks
# =============================================================================


class StatsQuantileBenchmark(BaseBenchmark):
    operation_id = "stats_quantile"

    def run_polars_eager(self, data_path: Path) -> Any:
        df = pl.read_parquet(data_path)
        df = self._limit_df(df)
        return df.select(pl.col("numeric_0").quantile(0.95))

    def run_polars_lazy(self, data_path: Path) -> Any:
        lf = pl.scan_parquet(data_path)
        lf = self._limit_lazy(lf)
        return lf.select(pl.col("numeric_0").quantile(0.95)).collect()

    def run_polars_streaming(self, data_path: Path) -> Any:
        lf = pl.scan_parquet(data_path)
        lf = self._limit_lazy(lf)
        return lf.select(pl.col("numeric_0").quantile(0.95)).collect(engine="streaming")

    def run_duckdb(self, data_path: Path) -> Any:
        limit = self._limit_sql()
        subq = f"SELECT * FROM '{data_path}'{limit}" if limit else f"'{data_path}'"
        return duckdb.sql(
            f"SELECT quantile_cont(numeric_0, 0.95) FROM ({subq})"
        ).fetchall()


class StatsCorrBenchmark(BaseBenchmark):
    operation_id = "stats_corr"

    def run_polars_eager(self, data_path: Path) -> Any:
        df = pl.read_parquet(data_path)
        df = self._limit_df(df)
        return df.select(pl.corr("numeric_0", "numeric_1"))

    def run_polars_lazy(self, data_path: Path) -> Any:
        lf = pl.scan_parquet(data_path)
        lf = self._limit_lazy(lf)
        return lf.select(pl.corr("numeric_0", "numeric_1")).collect()

    def run_polars_streaming(self, data_path: Path) -> Any:
        lf = pl.scan_parquet(data_path)
        lf = self._limit_lazy(lf)
        return lf.select(pl.corr("numeric_0", "numeric_1")).collect(engine="streaming")

    def run_duckdb(self, data_path: Path) -> Any:
        limit = self._limit_sql()
        subq = f"SELECT * FROM '{data_path}'{limit}" if limit else f"'{data_path}'"
        return duckdb.sql(f"SELECT corr(numeric_0, numeric_1) FROM ({subq})").fetchall()


class StatsValueCountsBenchmark(BaseBenchmark):
    operation_id = "stats_value_counts"

    def run_polars_eager(self, data_path: Path) -> Any:
        df = pl.read_parquet(data_path)
        df = self._limit_df(df)
        return df.select(pl.col("cat_category_0").value_counts())

    def run_polars_lazy(self, data_path: Path) -> Any:
        lf = pl.scan_parquet(data_path)
        lf = self._limit_lazy(lf)
        return lf.select(pl.col("cat_category_0").value_counts()).collect()

    def run_polars_streaming(self, data_path: Path) -> Any:
        lf = pl.scan_parquet(data_path)
        lf = self._limit_lazy(lf)
        return lf.select(pl.col("cat_category_0").value_counts()).collect(engine="streaming")

    def run_duckdb(self, data_path: Path) -> Any:
        limit = self._limit_sql()
        subq = f"SELECT * FROM '{data_path}'{limit}" if limit else f"'{data_path}'"
        return duckdb.sql(
            f"SELECT cat_category_0, COUNT(*) as count FROM ({subq}) GROUP BY cat_category_0"
        ).fetchall()


# =============================================================================
# Unique Benchmarks
# =============================================================================


class UniqueDistinctBenchmark(BaseBenchmark):
    operation_id = "unique_distinct"

    def run_polars_eager(self, data_path: Path) -> Any:
        df = pl.read_parquet(data_path)
        df = self._limit_df(df)
        return df.unique()

    def run_polars_lazy(self, data_path: Path) -> Any:
        lf = pl.scan_parquet(data_path)
        lf = self._limit_lazy(lf)
        return lf.unique().collect()

    def run_polars_streaming(self, data_path: Path) -> Any:
        lf = pl.scan_parquet(data_path)
        lf = self._limit_lazy(lf)
        return lf.unique().collect(engine="streaming")

    def run_duckdb(self, data_path: Path) -> Any:
        limit = self._limit_sql()
        subq = f"SELECT * FROM '{data_path}'{limit}" if limit else f"'{data_path}'"
        return duckdb.sql(f"SELECT DISTINCT * FROM ({subq})").fetchall()


class UniqueIsInBenchmark(BaseBenchmark):
    operation_id = "unique_is_in"

    def __init__(self, config: BenchmarkConfig):
        super().__init__(config)
        self._values: list[str] = []

    def setup(self, data_path: Path):
        df = pl.read_parquet(data_path)
        df = self._limit_df(df)
        self._values = df.select("cat_category_0").unique().head(5).to_series().to_list()

    def run_polars_eager(self, data_path: Path) -> Any:
        df = pl.read_parquet(data_path)
        df = self._limit_df(df)
        return df.filter(pl.col("cat_category_0").is_in(self._values))

    def run_polars_lazy(self, data_path: Path) -> Any:
        lf = pl.scan_parquet(data_path)
        lf = self._limit_lazy(lf)
        return lf.filter(pl.col("cat_category_0").is_in(self._values)).collect()

    def run_polars_streaming(self, data_path: Path) -> Any:
        lf = pl.scan_parquet(data_path)
        lf = self._limit_lazy(lf)
        return lf.filter(pl.col("cat_category_0").is_in(self._values)).collect(engine="streaming")

    def run_duckdb(self, data_path: Path) -> Any:
        limit = self._limit_sql()
        subq = f"SELECT * FROM '{data_path}'{limit}" if limit else f"'{data_path}'"
        values_str = ", ".join(f"'{v}'" for v in self._values)
        return duckdb.sql(
            f"SELECT * FROM ({subq}) WHERE cat_category_0 IN ({values_str})"
        ).fetchall()


# =============================================================================
# Cast Benchmarks
# =============================================================================


class CastIntToFloatBenchmark(BaseBenchmark):
    operation_id = "cast_int_to_float"

    def run_polars_eager(self, data_path: Path) -> Any:
        df = pl.read_parquet(data_path)
        df = self._limit_df(df)
        return df.with_columns(pl.col("id").cast(pl.Float64).alias("id_float"))

    def run_polars_lazy(self, data_path: Path) -> Any:
        lf = pl.scan_parquet(data_path)
        lf = self._limit_lazy(lf)
        return lf.with_columns(pl.col("id").cast(pl.Float64).alias("id_float")).collect()

    def run_polars_streaming(self, data_path: Path) -> Any:
        lf = pl.scan_parquet(data_path)
        lf = self._limit_lazy(lf)
        return lf.with_columns(pl.col("id").cast(pl.Float64).alias("id_float")).collect(engine="streaming")

    def run_duckdb(self, data_path: Path) -> Any:
        limit = self._limit_sql()
        subq = f"SELECT * FROM '{data_path}'{limit}" if limit else f"'{data_path}'"
        return duckdb.sql(
            f"SELECT *, CAST(id AS DOUBLE) as id_float FROM ({subq})"
        ).fetchall()


class CastToStringBenchmark(BaseBenchmark):
    operation_id = "cast_to_string"

    def run_polars_eager(self, data_path: Path) -> Any:
        df = pl.read_parquet(data_path)
        df = self._limit_df(df)
        return df.with_columns(pl.col("numeric_0").cast(pl.String).alias("num_str"))

    def run_polars_lazy(self, data_path: Path) -> Any:
        lf = pl.scan_parquet(data_path)
        lf = self._limit_lazy(lf)
        return lf.with_columns(pl.col("numeric_0").cast(pl.String).alias("num_str")).collect()

    def run_polars_streaming(self, data_path: Path) -> Any:
        lf = pl.scan_parquet(data_path)
        lf = self._limit_lazy(lf)
        return lf.with_columns(pl.col("numeric_0").cast(pl.String).alias("num_str")).collect(engine="streaming")

    def run_duckdb(self, data_path: Path) -> Any:
        limit = self._limit_sql()
        subq = f"SELECT * FROM '{data_path}'{limit}" if limit else f"'{data_path}'"
        return duckdb.sql(
            f"SELECT *, CAST(numeric_0 AS VARCHAR) as num_str FROM ({subq})"
        ).fetchall()


# =============================================================================
# Реестр бенчмарков
# =============================================================================

ALL_BENCHMARKS: dict[str, type[BaseBenchmark]] = {
    # I/O
    "io_read_parquet": ReadParquetBenchmark,
    "io_read_csv": ReadCSVBenchmark,
    "io_write_parquet": WriteParquetBenchmark,
    "io_write_csv": WriteCSVBenchmark,
    # Filter
    "filter_simple": FilterSimpleBenchmark,
    "filter_complex": FilterComplexBenchmark,
    # Aggregation
    "agg_single": AggSingleBenchmark,
    "agg_multi": AggMultiBenchmark,
    "agg_nunique": AggNuniqueBenchmark,
    "agg_multiple_keys": AggMultiKeysBenchmark,
    # Join
    "join_inner": JoinInnerBenchmark,
    "join_left": JoinLeftBenchmark,
    "join_outer": JoinOuterBenchmark,
    "join_anti": JoinAntiBenchmark,
    "join_semi": JoinSemiBenchmark,
    # Sort
    "sort_single": SortSingleBenchmark,
    "sort_multi": SortMultiBenchmark,
    "sort_topk": SortTopKBenchmark,
    # Window
    "window_rank": WindowRankBenchmark,
    "window_rolling_mean": WindowRollingMeanBenchmark,
    "window_lag": WindowLagBenchmark,
    "window_cumsum": WindowCumsumBenchmark,
    # Vector ops
    "vec_add": VecAddBenchmark,
    "vec_compound": VecCompoundBenchmark,
    # Transform
    "transform_concat_v": TransformConcatVBenchmark,
    "transform_concat_h": TransformConcatHBenchmark,
    "transform_unpivot": TransformUnpivotBenchmark,
    # String
    "str_lower": StringLowerBenchmark,
    "str_length": StringLengthBenchmark,
    # Datetime
    "dt_extract_year": DatetimeExtractYearBenchmark,
    "dt_truncate": DatetimeTruncateBenchmark,
    # Missing
    "missing_fillna": MissingFillnaBenchmark,
    "missing_dropna": MissingDropnaBenchmark,
    # Stats
    "stats_quantile": StatsQuantileBenchmark,
    "stats_corr": StatsCorrBenchmark,
    "stats_value_counts": StatsValueCountsBenchmark,
    # Unique
    "unique_distinct": UniqueDistinctBenchmark,
    "unique_is_in": UniqueIsInBenchmark,
    # Cast
    "cast_int_to_float": CastIntToFloatBenchmark,
    "cast_to_string": CastToStringBenchmark,
}


# =============================================================================
# ML Pipeline Base
# =============================================================================


class MLPipelineBase(ABC):
    """Базовый класс для ML пайплайнов."""

    task_type: str = ""

    def __init__(self, config: BenchmarkConfig, framework: str = "polars"):
        self.config = config
        self.framework = framework  # "polars" | "duckdb"
        self._collector: MetricsCollector | None = None
        self.n_rows: int | None = None  # Количество строк для чтения (None = все)

    def _read_data(self, data_path: Path) -> pl.DataFrame:
        """Загрузка данных."""
        if self.framework == "polars":
            df = pl.read_parquet(data_path)
        else:
            df = duckdb.sql(f"SELECT * FROM '{data_path}'").pl()
        # Ограничиваем количество строк если задано
        if self.n_rows is not None and len(df) > self.n_rows:
            df = df.head(self.n_rows)
        return df

    def _get_target_col(self, df: pl.DataFrame) -> str | None:
        """Определить колонку target из доступных в датафрейме."""
        # Приоритетный порядок поиска
        target_candidates = [
            "target",  # общий
            "target_reg",  # synthdator regression
            "target_bin",  # synthdator binary
            "target_multi",  # synthdator multiclass
            "target_rank",  # synthdator ranking
        ]
        for col in target_candidates:
            if col in df.columns:
                return col
        return None

    def _time_stage(self, func: Callable, *args, **kwargs) -> tuple[Any, float]:
        """Замерить время выполнения этапа."""
        gc.collect()
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        return result, elapsed

    @abstractmethod
    def load_data(self, data_path: Path) -> pl.DataFrame:
        pass

    @abstractmethod
    def clean_data(self, df: pl.DataFrame) -> pl.DataFrame:
        pass

    @abstractmethod
    def engineer_features(self, df: pl.DataFrame) -> pl.DataFrame:
        pass

    @abstractmethod
    def preprocess(
        self, df: pl.DataFrame
    ) -> tuple[Any, Any]:  # (features, target)
        pass

    @abstractmethod
    def split_data(
        self, features: Any, target: Any
    ) -> tuple[Any, Any, Any, Any]:  # (train_f, test_f, train_t, test_t)
        pass

    @abstractmethod
    def train(self, features_train: Any, target_train: Any, model_name: str) -> Any:
        pass

    @abstractmethod
    def validate(
        self, model: Any, features_test: Any, target_test: Any
    ) -> dict[str, float]:
        pass

    def run(
        self,
        data_path: Path,
        data_size: str,
        model_name: str,
        feature_set: str = "baseline",
        preprocessing_strategy: str = "normalized",
        validation_strategy: str = "kfold",
        n_rows: int | None = None,
    ) -> MLPipelineResult:
        """Запустить полный пайплайн.

        Args:
            data_path: Путь к parquet-файлу.
            data_size: Метка размера данных (например, "1GB").
            model_name: Название модели для обучения.
            feature_set: Набор фич ("baseline" и т.д.).
            preprocessing_strategy: Стратегия препроцессинга.
            validation_strategy: Стратегия валидации.
            n_rows: Количество строк для чтения. None = весь файл.

        Returns:
            MLPipelineResult с результатами.
        """
        # Устанавливаем n_rows для использования в load_data
        self.n_rows = n_rows

        gc.collect()
        self._collector = MetricsCollector(interval_ms=self.config.metrics_interval_ms)
        self._collector.start()

        times = {}
        success = True
        error_message = None
        ml_metrics: dict[str, float] = {}

        try:
            # 1. Loading
            df, times["loading"] = self._time_stage(self.load_data, data_path)

            # 2. Cleaning
            df, times["cleaning"] = self._time_stage(self.clean_data, df)

            # 3. Feature Engineering
            df, times["feature_engineering"] = self._time_stage(
                self.engineer_features, df
            )

            # 4. Preprocessing
            (features, target), times["preprocessing"] = self._time_stage(
                self.preprocess, df
            )

            # 5. Split
            splits, times["splitting"] = self._time_stage(
                self.split_data, features, target
            )
            features_train, features_test, target_train, target_test = splits

            # 6. Training
            model, times["training"] = self._time_stage(
                self.train, features_train, target_train, model_name
            )

            # 7. Validation
            ml_metrics, times["validation"] = self._time_stage(
                self.validate, model, features_test, target_test
            )

        except Exception as e:
            success = False
            error_message = str(e)
            for key in ["loading", "cleaning", "feature_engineering",
                        "preprocessing", "splitting", "training", "validation"]:
                times.setdefault(key, 0.0)

        collected = self._collector.stop()
        time_total = sum(times.values())

        return MLPipelineResult(
            task_type=self.task_type,
            model_name=model_name,
            framework=self.framework,
            data_size=data_size,
            feature_set=feature_set,
            preprocessing=preprocessing_strategy,
            validation=validation_strategy,
            time_loading=times.get("loading", 0.0),
            time_cleaning=times.get("cleaning", 0.0),
            time_feature_engineering=times.get("feature_engineering", 0.0),
            time_preprocessing=times.get("preprocessing", 0.0),
            time_splitting=times.get("splitting", 0.0),
            time_training=times.get("training", 0.0),
            time_validation=times.get("validation", 0.0),
            time_total=time_total,
            peak_memory_mb=collected.peak_memory_mb,
            metrics=ml_metrics,
            success=success,
            error_message=error_message,
        )


# =============================================================================
# Binary Classification Pipeline
# =============================================================================


class BinaryClassificationPipeline(MLPipelineBase):
    """Пайплайн бинарной классификации."""

    task_type = "binary_classification"

    MODELS = {
        "xgboost": lambda: __import__("xgboost").XGBClassifier(
            n_estimators=100, max_depth=6, n_jobs=-1, verbosity=0
        ),
        "lightgbm": lambda: __import__("lightgbm").LGBMClassifier(
            n_estimators=100, max_depth=6, n_jobs=-1, verbose=-1
        ),
        "catboost": lambda: __import__("catboost").CatBoostClassifier(
            n_estimators=100, max_depth=6, thread_count=-1, verbose=0
        ),
        "logistic_regression": lambda: __import__(
            "sklearn.linear_model", fromlist=["LogisticRegression"]
        ).LogisticRegression(max_iter=1000, n_jobs=-1),
        "random_forest": lambda: __import__(
            "sklearn.ensemble", fromlist=["RandomForestClassifier"]
        ).RandomForestClassifier(n_estimators=100, max_depth=6, n_jobs=-1),
    }

    def load_data(self, data_path: Path) -> pl.DataFrame:
        return self._read_data(data_path)

    def clean_data(self, df: pl.DataFrame) -> pl.DataFrame:
        # Удалить дубликаты и строки с пропусками в target
        df = df.unique()
        target_col = self._get_target_col(df)
        if target_col:
            df = df.drop_nulls(subset=[target_col])
        return df

    def engineer_features(self, df: pl.DataFrame) -> pl.DataFrame:
        # Базовые агрегации по категориям
        numeric_cols = [c for c in df.columns if c.startswith("numeric_")]
        if numeric_cols and "cat_category_0" in df.columns:
            for col in numeric_cols[:3]:  # Ограничим первыми 3
                new_col = f"{col}_cat0_mean"
                if new_col not in df.columns:
                    df = df.with_columns(
                        pl.col(col).mean().over("cat_category_0").alias(new_col)
                    )
        return df

    def preprocess(self, df: pl.DataFrame) -> tuple[Any, Any]:
        from sklearn.preprocessing import StandardScaler

        # Выбираем числовые колонки для features
        feature_cols = [c for c in df.columns if c.startswith("numeric_")]
        feature_cols += [c for c in df.columns if c.endswith("_cat0_mean")]

        features = df.select(feature_cols).to_numpy()

        target_col = self._get_target_col(df)
        if not target_col:
            raise ValueError("No target column found in data")
        target = df.select(target_col).to_numpy().ravel()

        # Нормализация
        scaler = StandardScaler()
        features = scaler.fit_transform(features)

        return features, target

    def split_data(
        self, features: Any, target: Any
    ) -> tuple[Any, Any, Any, Any]:
        from sklearn.model_selection import train_test_split

        return train_test_split(features, target, test_size=0.2, random_state=42)

    def train(self, features_train: Any, target_train: Any, model_name: str) -> Any:
        model = self.MODELS[model_name]()
        model.fit(features_train, target_train)
        return model

    def validate(
        self, model: Any, features_test: Any, target_test: Any
    ) -> dict[str, float]:
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        predictions = model.predict(features_test)
        proba = (
            model.predict_proba(features_test)[:, 1]
            if hasattr(model, "predict_proba")
            else predictions
        )

        return {
            "accuracy": accuracy_score(target_test, predictions),
            "f1": f1_score(target_test, predictions, average="binary"),
            "precision": precision_score(target_test, predictions, average="binary"),
            "recall": recall_score(target_test, predictions, average="binary"),
            "roc_auc": roc_auc_score(target_test, proba),
        }


# =============================================================================
# Regression Pipeline
# =============================================================================


class RegressionPipeline(MLPipelineBase):
    """Пайплайн регрессии."""

    task_type = "regression"

    MODELS = {
        "xgboost": lambda: __import__("xgboost").XGBRegressor(
            n_estimators=100, max_depth=6, n_jobs=-1, verbosity=0
        ),
        "lightgbm": lambda: __import__("lightgbm").LGBMRegressor(
            n_estimators=100, max_depth=6, n_jobs=-1, verbose=-1
        ),
        "catboost": lambda: __import__("catboost").CatBoostRegressor(
            n_estimators=100, max_depth=6, thread_count=-1, verbose=0
        ),
        "ridge": lambda: __import__(
            "sklearn.linear_model", fromlist=["Ridge"]
        ).Ridge(alpha=1.0),
        "lasso": lambda: __import__(
            "sklearn.linear_model", fromlist=["Lasso"]
        ).Lasso(alpha=1.0),
        "elastic_net": lambda: __import__(
            "sklearn.linear_model", fromlist=["ElasticNet"]
        ).ElasticNet(alpha=1.0),
        "random_forest": lambda: __import__(
            "sklearn.ensemble", fromlist=["RandomForestRegressor"]
        ).RandomForestRegressor(n_estimators=100, max_depth=6, n_jobs=-1),
    }

    def load_data(self, data_path: Path) -> pl.DataFrame:
        return self._read_data(data_path)

    def clean_data(self, df: pl.DataFrame) -> pl.DataFrame:
        df = df.unique()
        target_col = self._get_target_col(df)
        if target_col:
            df = df.drop_nulls(subset=[target_col])
        return df

    def engineer_features(self, df: pl.DataFrame) -> pl.DataFrame:
        # Пропускаем feature engineering, используем только raw фичи
        return df

    def preprocess(self, df: pl.DataFrame) -> tuple[Any, Any]:
        from sklearn.preprocessing import StandardScaler
        import numpy as np

        feature_cols = [c for c in df.columns if c.startswith("numeric_")]

        target_col = self._get_target_col(df)
        if not target_col:
            raise ValueError("No target column found in data")

        # Удаляем строки с NaN
        df = df.drop_nulls(subset=feature_cols + [target_col])

        features = df.select(feature_cols).to_numpy()
        target = df.select(target_col).to_numpy().ravel()

        # Заполняем оставшиеся NaN (на всякий случай)
        features = np.nan_to_num(features, nan=0.0)

        scaler = StandardScaler()
        features = scaler.fit_transform(features)

        return features, target

    def split_data(
        self, features: Any, target: Any
    ) -> tuple[Any, Any, Any, Any]:
        from sklearn.model_selection import train_test_split

        return train_test_split(features, target, test_size=0.2, random_state=42)

    def train(self, features_train: Any, target_train: Any, model_name: str) -> Any:
        model = self.MODELS[model_name]()
        model.fit(features_train, target_train)
        return model

    def validate(
        self, model: Any, features_test: Any, target_test: Any
    ) -> dict[str, float]:
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        import numpy as np

        predictions = model.predict(features_test)

        rmse = np.sqrt(mean_squared_error(target_test, predictions))
        mae = mean_absolute_error(target_test, predictions)
        r2 = r2_score(target_test, predictions)

        # MAPE (с защитой от деления на ноль)
        mask = target_test != 0
        mape = np.mean(np.abs((target_test[mask] - predictions[mask]) / target_test[mask])) * 100

        return {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "mape": mape,
        }


# =============================================================================
# Multiclass Classification Pipeline
# =============================================================================


class MulticlassClassificationPipeline(MLPipelineBase):
    """Пайплайн многоклассовой классификации."""

    task_type = "multiclass_classification"

    MODELS = {
        "xgboost": lambda: __import__("xgboost").XGBClassifier(
            n_estimators=100, max_depth=6, n_jobs=-1, verbosity=0
        ),
        "lightgbm": lambda: __import__("lightgbm").LGBMClassifier(
            n_estimators=100, max_depth=6, n_jobs=-1, verbose=-1
        ),
        "catboost": lambda: __import__("catboost").CatBoostClassifier(
            n_estimators=100, max_depth=6, thread_count=-1, verbose=0
        ),
        "logistic_regression": lambda: __import__(
            "sklearn.linear_model", fromlist=["LogisticRegression"]
        ).LogisticRegression(max_iter=1000, n_jobs=-1, multi_class="multinomial"),
        "random_forest": lambda: __import__(
            "sklearn.ensemble", fromlist=["RandomForestClassifier"]
        ).RandomForestClassifier(n_estimators=100, max_depth=6, n_jobs=-1),
    }

    def load_data(self, data_path: Path) -> pl.DataFrame:
        return self._read_data(data_path)

    def clean_data(self, df: pl.DataFrame) -> pl.DataFrame:
        df = df.unique()
        target_col = self._get_target_col(df)
        if target_col:
            df = df.drop_nulls(subset=[target_col])
        return df

    def engineer_features(self, df: pl.DataFrame) -> pl.DataFrame:
        numeric_cols = [c for c in df.columns if c.startswith("numeric_")]
        if numeric_cols and "cat_category_0" in df.columns:
            for col in numeric_cols[:3]:
                new_col = f"{col}_cat0_mean"
                if new_col not in df.columns:
                    df = df.with_columns(
                        pl.col(col).mean().over("cat_category_0").alias(new_col)
                    )
        return df

    def preprocess(self, df: pl.DataFrame) -> tuple[Any, Any]:
        from sklearn.preprocessing import StandardScaler

        feature_cols = [c for c in df.columns if c.startswith("numeric_")]
        feature_cols += [c for c in df.columns if c.endswith("_cat0_mean")]

        features = df.select(feature_cols).to_numpy()

        target_col = self._get_target_col(df)
        if not target_col:
            raise ValueError("No target column found in data")
        target = df.select(target_col).to_numpy().ravel()

        scaler = StandardScaler()
        features = scaler.fit_transform(features)

        return features, target

    def split_data(
        self, features: Any, target: Any
    ) -> tuple[Any, Any, Any, Any]:
        from sklearn.model_selection import train_test_split

        return train_test_split(
            features, target, test_size=0.2, random_state=42, stratify=target
        )

    def train(self, features_train: Any, target_train: Any, model_name: str) -> Any:
        model = self.MODELS[model_name]()
        model.fit(features_train, target_train)
        return model

    def validate(
        self, model: Any, features_test: Any, target_test: Any
    ) -> dict[str, float]:
        from sklearn.metrics import accuracy_score, f1_score

        predictions = model.predict(features_test)

        return {
            "accuracy": accuracy_score(target_test, predictions),
            "f1_macro": f1_score(target_test, predictions, average="macro"),
            "f1_weighted": f1_score(target_test, predictions, average="weighted"),
        }


# =============================================================================
# Clustering Pipeline
# =============================================================================


class ClusteringPipeline(MLPipelineBase):
    """Пайплайн кластеризации."""

    task_type = "clustering"

    MODELS = {
        "kmeans": lambda n_clusters=5: __import__(
            "sklearn.cluster", fromlist=["KMeans"]
        ).KMeans(n_clusters=n_clusters, random_state=42, n_init=10),
        "dbscan": lambda: __import__(
            "sklearn.cluster", fromlist=["DBSCAN"]
        ).DBSCAN(eps=0.5, min_samples=5),
        "agglomerative": lambda n_clusters=5: __import__(
            "sklearn.cluster", fromlist=["AgglomerativeClustering"]
        ).AgglomerativeClustering(n_clusters=n_clusters),
    }

    def load_data(self, data_path: Path) -> pl.DataFrame:
        return self._read_data(data_path)

    def clean_data(self, df: pl.DataFrame) -> pl.DataFrame:
        df = df.unique()
        df = df.drop_nulls()
        return df

    def engineer_features(self, df: pl.DataFrame) -> pl.DataFrame:
        return df  # Для кластеризации используем сырые фичи

    def preprocess(self, df: pl.DataFrame) -> tuple[Any, Any]:
        from sklearn.preprocessing import StandardScaler

        feature_cols = [c for c in df.columns if c.startswith("numeric_")]
        features = df.select(feature_cols).to_numpy()

        scaler = StandardScaler()
        features = scaler.fit_transform(features)

        # Для кластеризации target не нужен, но вернём реальный если есть
        target_col = self._get_target_col(df)
        target = df.select(target_col).to_numpy().ravel() if target_col else None

        return features, target

    def split_data(
        self, features: Any, target: Any
    ) -> tuple[Any, Any, Any, Any]:
        # Для кластеризации не делим данные
        return features, features, target, target

    def train(self, features_train: Any, target_train: Any, model_name: str) -> Any:
        model = self.MODELS[model_name]()
        model.fit(features_train)
        return model

    def validate(
        self, model: Any, features_test: Any, target_test: Any
    ) -> dict[str, float]:
        from sklearn.metrics import (
            calinski_harabasz_score,
            davies_bouldin_score,
            silhouette_score,
        )

        labels = model.labels_ if hasattr(model, "labels_") else model.predict(features_test)

        # Проверяем, что есть хотя бы 2 кластера
        n_labels = len(set(labels)) - (1 if -1 in labels else 0)
        if n_labels < 2:
            return {"silhouette": 0.0, "calinski_harabasz": 0.0, "davies_bouldin": 0.0}

        return {
            "silhouette": silhouette_score(features_test, labels),
            "calinski_harabasz": calinski_harabasz_score(features_test, labels),
            "davies_bouldin": davies_bouldin_score(features_test, labels),
        }


# =============================================================================
# Dimensionality Reduction Pipeline
# =============================================================================


class DimensionalityReductionPipeline(MLPipelineBase):
    """Пайплайн снижения размерности."""

    task_type = "dimensionality_reduction"

    MODELS = {
        "pca": lambda n_components=2: __import__(
            "sklearn.decomposition", fromlist=["PCA"]
        ).PCA(n_components=n_components),
        "tsne": lambda n_components=2: __import__(
            "sklearn.manifold", fromlist=["TSNE"]
        ).TSNE(n_components=n_components, random_state=42),
        "umap": lambda n_components=2: __import__(
            "umap", fromlist=["UMAP"]
        ).UMAP(n_components=n_components, random_state=42),
    }

    def load_data(self, data_path: Path) -> pl.DataFrame:
        return self._read_data(data_path)

    def clean_data(self, df: pl.DataFrame) -> pl.DataFrame:
        df = df.unique()
        df = df.drop_nulls()
        # Для t-SNE/UMAP ограничим размер выборки
        if len(df) > 10000:
            df = df.sample(n=10000, seed=42)
        return df

    def engineer_features(self, df: pl.DataFrame) -> pl.DataFrame:
        return df

    def preprocess(self, df: pl.DataFrame) -> tuple[Any, Any]:
        from sklearn.preprocessing import StandardScaler

        feature_cols = [c for c in df.columns if c.startswith("numeric_")]
        features = df.select(feature_cols).to_numpy()

        scaler = StandardScaler()
        features = scaler.fit_transform(features)

        target_col = self._get_target_col(df)
        target = df.select(target_col).to_numpy().ravel() if target_col else None

        return features, target

    def split_data(
        self, features: Any, target: Any
    ) -> tuple[Any, Any, Any, Any]:
        return features, features, target, target

    def train(self, features_train: Any, target_train: Any, model_name: str) -> Any:
        model = self.MODELS[model_name]()
        model.fit(features_train)
        return model

    def validate(
        self, model: Any, features_test: Any, target_test: Any
    ) -> dict[str, float]:
        import numpy as np

        if hasattr(model, "transform"):
            transformed = model.transform(features_test)
        else:
            # t-SNE не имеет transform, используем embedding_
            transformed = model.embedding_

        metrics = {"n_components": transformed.shape[1]}

        # Для PCA — explained variance
        if hasattr(model, "explained_variance_ratio_"):
            metrics["explained_variance"] = sum(model.explained_variance_ratio_)

        # Reconstruction error для PCA
        if hasattr(model, "inverse_transform"):
            reconstructed = model.inverse_transform(transformed)
            mse = np.mean((features_test - reconstructed) ** 2)
            metrics["reconstruction_mse"] = mse

        return metrics


# =============================================================================
# Реестр ML пайплайнов
# =============================================================================

ALL_ML_PIPELINES: dict[str, type[MLPipelineBase]] = {
    "binary_classification": BinaryClassificationPipeline,
    "regression": RegressionPipeline,
    "multiclass_classification": MulticlassClassificationPipeline,
    "clustering": ClusteringPipeline,
    "dimensionality_reduction": DimensionalityReductionPipeline,
}


# =============================================================================
# Унифицированный API (аналог synthdator)
# =============================================================================


class BenchmarkSuite:
    """Набор бенчмарков для выполнения (аналог Pipeline в synthdator).

    Пример использования:
        config = BenchmarkConfig(data_path=Path("data.parquet"), data_size="1GB")
        suite = BenchmarkFactory().create(config)
        results = suite.run()
        print(results.summary())
    """

    def __init__(
        self,
        benchmarks: list[BaseBenchmark],
        ml_pipelines: list["MLPipelineBase"],
        config: BenchmarkConfig,
    ):
        """Инициализирует BenchmarkSuite.

        Args:
            benchmarks: Список бенчмарков для выполнения.
            ml_pipelines: Список ML пайплайнов для выполнения.
            config: Конфигурация бенчмарка.
        """
        self.benchmarks = benchmarks
        self.ml_pipelines = ml_pipelines
        self.config = config

    def _get_sizes_to_run(self) -> list[tuple[str, int | None]]:
        """Возвращает список (data_size, n_rows) для запуска.

        Returns:
            Список кортежей (размер, количество_строк).
            Если meta доступна — вычисляет n_rows из meta.rows_for_size().
            Если meta нет — n_rows=None (читать весь файл).
        """
        if not self.config.data_sizes:
            # Нет списка размеров — используем весь файл
            if self.config.meta:
                size_label = f"{self.config.meta.file_size_bytes / (1024**2):.0f}MB"
                return [(size_label, None)]
            return [("full", None)]

        # Есть список размеров — вычисляем n_rows для каждого
        sizes = []
        for size in self.config.data_sizes:
            if self.config.meta:
                n_rows = self.config.meta.rows_for_size(size)
                sizes.append((size, n_rows))
            else:
                # Нет meta — не можем вычислить n_rows, используем весь файл
                sizes.append((size, None))
        return sizes

    def run(self) -> BenchmarkResults:
        """Запускает все бенчмарки и ML пайплайны для каждого размера данных.

        Returns:
            BenchmarkResults с результатами всех запусков.
        """
        if self.config.data_path is None:
            raise ValueError("data_path must be set in BenchmarkConfig")

        results = BenchmarkResults()
        sizes_to_run = self._get_sizes_to_run()

        for data_size, n_rows in sizes_to_run:
            print(f"\n{'='*60}")
            print(f"Data size: {data_size}" + (f" ({n_rows:,} rows)" if n_rows else " (full)"))
            print(f"{'='*60}")

            # Запускаем бенчмарки
            for bench in self.benchmarks:
                print(f"Running benchmark: {bench.operation_id}")
                try:
                    bench_results = bench.run_all(
                        self.config.data_path,
                        data_size,
                        n_rows=n_rows,
                    )
                    results.benchmarks.extend(bench_results)
                    for r in bench_results:
                        status = "OK" if r.success else f"FAIL: {r.error_message}"
                        print(
                            f"  {r.framework}/{r.mode} run#{r.run_number}: "
                            f"{r.wall_time_seconds:.3f}s, mem={r.peak_memory_mb:.1f}MB [{status}]"
                        )
                except Exception as e:
                    print(f"  ERROR: {e}")
                    results.benchmarks.append(
                        BenchmarkResult(
                            operation=bench.operation_id,
                            framework="unknown",
                            mode="unknown",
                            data_size=data_size,
                            run_number=0,
                            is_cold=True,
                            wall_time_seconds=-1,
                            peak_memory_mb=-1,
                            cpu_percent=-1,
                            success=False,
                            error_message=str(e),
                        )
                    )

            # Запускаем ML пайплайны
            for pipeline in self.ml_pipelines:
                task_name = pipeline.__class__.__name__
                print(f"Running ML pipeline: {task_name}")
                for model_name in pipeline.MODELS:
                    try:
                        print(f"  Model: {model_name}")
                        result = pipeline.run(
                            self.config.data_path,
                            data_size,
                            model_name,
                            n_rows=n_rows,
                        )
                        results.ml_pipelines.append(result)
                        status = "OK" if result.success else f"FAIL: {result.error_message}"
                        print(
                            f"    {result.framework}: {result.time_total:.3f}s, "
                            f"mem={result.peak_memory_mb:.1f}MB [{status}]"
                        )
                    except Exception as e:
                        print(f"    ERROR: {e}")
                        results.ml_pipelines.append(
                            MLPipelineResult(
                                task_type=task_name,
                                model_name=model_name,
                                framework="unknown",
                                data_size=data_size,
                                feature_set="unknown",
                                preprocessing="unknown",
                                validation="unknown",
                                time_loading=0,
                                time_cleaning=0,
                                time_feature_engineering=0,
                                time_preprocessing=0,
                                time_splitting=0,
                                time_training=0,
                                time_validation=0,
                                time_total=-1,
                                peak_memory_mb=-1,
                                success=False,
                                error_message=str(e),
                            )
                        )

        return results


class BenchmarkFactory:
    """Фабрика для создания BenchmarkSuite из конфигурации (аналог PipelineFactory).

    Пример использования:
        # С существующими данными:
        config = BenchmarkConfig(
            data_path=Path("data.parquet"),
            data_sizes=["100MB", "1GB"],
        )

        # С автоматической генерацией:
        config = BenchmarkConfig(
            data_sizes=["100MB", "1GB", "10GB"],
            data_generation=DataGenerationConfig(
                tasks=["regression", "binary"],
                n_numeric=20,
            ),
        )

        suite = BenchmarkFactory().create(config)
        results = suite.run()
    """

    @staticmethod
    def _parse_size(size_str: str) -> int:
        """Парсит строку размера в байты."""
        size_str = size_str.strip().upper()
        multipliers = {
            "B": 1,
            "KB": 1024,
            "MB": 1024 ** 2,
            "GB": 1024 ** 3,
            "TB": 1024 ** 4,
        }
        for suffix, mult in sorted(multipliers.items(), key=lambda x: -len(x[0])):
            if size_str.endswith(suffix):
                num = float(size_str[: -len(suffix)])
                return int(num * mult)
        return int(size_str)

    def _get_max_size(self, config: BenchmarkConfig) -> str:
        """Определяет максимальный размер для генерации.

        Args:
            config: Конфигурация бенчмарка.

        Returns:
            Строка размера (например, "10GB").
        """
        if config.data_generation and config.data_generation.max_size:
            return config.data_generation.max_size

        # Берём максимальный из data_sizes
        if config.data_sizes:
            max_bytes = max(self._parse_size(s) for s in config.data_sizes)
            # Конвертируем обратно в строку
            if max_bytes >= 1024 ** 3:
                return f"{max_bytes // (1024 ** 3)}GB"
            elif max_bytes >= 1024 ** 2:
                return f"{max_bytes // (1024 ** 2)}MB"
            else:
                return f"{max_bytes}B"

        return "100MB"

    def _generate_data(self, config: BenchmarkConfig) -> tuple[Path, Any]:
        """Генерирует данные через Synthdator.

        Args:
            config: Конфигурация бенчмарка.

        Returns:
            Кортеж (путь к parquet файлу, meta объект).
        """
        # Импортируем synthdator
        import sys
        synthdator_path = Path(__file__).parent.parent / "synthdator"
        if str(synthdator_path) not in sys.path:
            sys.path.insert(0, str(synthdator_path))

        from generator import GeneratorConfig, PipelineFactory

        gen_config = config.data_generation
        max_size = self._get_max_size(config)
        data_dir = config.output_path / "generated_data"
        data_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Generating data: {max_size}")
        print(f"Output: {data_dir}")
        print(f"{'='*60}\n")

        # Создаём GeneratorConfig из DataGenerationConfig
        generator_config = GeneratorConfig(
            target_size=max_size,
            output_path=str(data_dir),
            seed=config.seed,
            n_numeric=gen_config.n_numeric,
            informative_ratio=gen_config.informative_ratio,
            n_categories=gen_config.n_categories,
            category_cardinality=gen_config.category_cardinality,
            tasks=gen_config.tasks,
            target_method=gen_config.target_method,
            target_noise=gen_config.target_noise,
            n_classes=gen_config.n_classes,
            datetime_range=gen_config.datetime_range,
            with_date=gen_config.with_date,
            n_booleans=gen_config.n_booleans,
            nullable_ratio=gen_config.nullable_ratio,
        )

        # Генерируем данные
        factory = PipelineFactory()
        pipeline = factory.create(generator_config)
        meta = pipeline.run()

        data_path = Path(meta.file_path)
        print(f"\nData generated: {data_path}")
        print(f"Rows: {meta.row_count:,}")
        print(f"Size: {meta.file_size_bytes / (1024**2):.1f} MB")
        print(f"Columns: {list(meta.columns.keys())}\n")

        return data_path, meta

    def _try_load_meta(self, data_path: Path) -> Any:
        """Пытается загрузить Meta из pickle рядом с parquet.

        Args:
            data_path: Путь к parquet файлу.

        Returns:
            Объект Meta или None если не найден.
        """
        import pickle

        # Ищем meta.pkl рядом с parquet
        meta_path = data_path.parent / "meta.pkl"
        if not meta_path.exists():
            # Попробуем заменить имя файла
            meta_path = Path(str(data_path).replace("main.parquet", "meta.pkl"))

        if meta_path.exists():
            try:
                with open(meta_path, "rb") as f:
                    meta = pickle.load(f)
                print(f"Loaded meta from: {meta_path}")
                return meta
            except Exception as e:
                print(f"Warning: Failed to load meta: {e}")

        return None

    def create(self, config: BenchmarkConfig) -> BenchmarkSuite:
        """Создаёт BenchmarkSuite на основе конфигурации.

        Если data_path не указан и есть data_generation — генерирует данные
        через Synthdator.

        Args:
            config: Конфигурация бенчмарка.

        Returns:
            BenchmarkSuite готовый к запуску.
        """
        # Генерируем данные если нужно
        if config.data_path is None and config.data_generation is not None:
            config.data_path, config.meta = self._generate_data(config)
        # Пытаемся загрузить meta если не передана
        elif config.meta is None and config.data_path is not None:
            config.meta = self._try_load_meta(config.data_path)

        # Выбираем бенчмарки
        if config.benchmark_filter:
            benchmarks = [
                ALL_BENCHMARKS[name](config)
                for name in config.benchmark_filter
                if name in ALL_BENCHMARKS
            ]
        else:
            benchmarks = [cls(config) for cls in ALL_BENCHMARKS.values()]

        # Выбираем ML пайплайны
        if config.ml_pipelines:
            ml_pipelines = [
                ALL_ML_PIPELINES[name](config)
                for name in config.ml_pipelines
                if name in ALL_ML_PIPELINES
            ]
        else:
            ml_pipelines = [cls(config) for cls in ALL_ML_PIPELINES.values()]

        return BenchmarkSuite(benchmarks, ml_pipelines, config)


# =============================================================================
# Runner и CLI (legacy API)
# =============================================================================


@dataclass
class BenchmarkRunner:
    """Класс для запуска бенчмарков и ML пайплайнов."""

    config: BenchmarkConfig
    output_dir: Path = field(default_factory=lambda: Path("results"))

    def __post_init__(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_benchmark(
        self,
        benchmark_name: str,
        data_path: Path,
        data_size: str = "unknown",
    ) -> list[BenchmarkResult]:
        """Запуск одного бенчмарка."""
        if benchmark_name not in ALL_BENCHMARKS:
            raise ValueError(f"Unknown benchmark: {benchmark_name}")

        benchmark_cls = ALL_BENCHMARKS[benchmark_name]
        benchmark = benchmark_cls(self.config)

        try:
            results = benchmark.run_all(data_path, data_size)
            for r in results:
                status = "OK" if r.success else f"FAIL: {r.error_message}"
                print(f"  {r.framework}/{r.mode} run#{r.run_number}: "
                      f"{r.wall_time_seconds:.3f}s, mem={r.peak_memory_mb:.1f}MB [{status}]")
            return results
        except Exception as e:
            print(f"  ERROR: {e}")
            return [BenchmarkResult(
                operation=benchmark_name,
                framework="unknown",
                mode="unknown",
                data_size=data_size,
                run_number=0,
                is_cold=True,
                wall_time_seconds=-1,
                peak_memory_mb=-1,
                cpu_percent=-1,
                success=False,
                error_message=str(e),
            )]

    def run_all_benchmarks(
        self,
        data_path: Path,
        data_size: str = "unknown",
        benchmark_filter: list[str] | None = None,
    ) -> list[BenchmarkResult]:
        """Запуск всех бенчмарков."""
        all_results = []
        benchmarks = benchmark_filter or list(ALL_BENCHMARKS.keys())

        for name in benchmarks:
            print(f"Running benchmark: {name}")
            results = self.run_benchmark(name, data_path, data_size)
            all_results.extend(results)

        return all_results

    def run_ml_pipeline(
        self,
        pipeline_name: str,
        data_path: Path,
        models: list[str] | None = None,
        data_size: str = "unknown",
    ) -> list[MLPipelineResult]:
        """Запуск одного ML пайплайна."""
        if pipeline_name not in ALL_ML_PIPELINES:
            raise ValueError(f"Unknown pipeline: {pipeline_name}")

        pipeline_cls = ALL_ML_PIPELINES[pipeline_name]
        pipeline = pipeline_cls(self.config)

        if models is None:
            models = list(pipeline.MODELS.keys())

        results = []
        for model_name in models:
            try:
                print(f"  Model: {model_name}")
                result = pipeline.run(data_path, data_size, model_name)
                results.append(result)
                status = "OK" if result.success else f"FAIL: {result.error_message}"
                print(f"    {result.framework}: {result.time_total:.3f}s, "
                      f"mem={result.peak_memory_mb:.1f}MB [{status}]")
            except Exception as e:
                print(f"    ERROR: {e}")
                results.append(MLPipelineResult(
                    task_type=pipeline_name,
                    model_name=model_name,
                    framework="unknown",
                    data_size=data_size,
                    feature_set="unknown",
                    preprocessing="unknown",
                    validation="unknown",
                    time_loading=0,
                    time_cleaning=0,
                    time_feature_engineering=0,
                    time_preprocessing=0,
                    time_splitting=0,
                    time_training=0,
                    time_validation=0,
                    time_total=-1,
                    peak_memory_mb=-1,
                    success=False,
                    error_message=str(e),
                ))

        return results

    def run_all_ml_pipelines(
        self,
        data_path: Path,
        data_size: str = "unknown",
        pipeline_filter: list[str] | None = None,
    ) -> list[MLPipelineResult]:
        """Запуск всех ML пайплайнов."""
        all_results = []
        pipelines = pipeline_filter or list(ALL_ML_PIPELINES.keys())

        for name in pipelines:
            print(f"Running ML pipeline: {name}")
            results = self.run_ml_pipeline(name, data_path, data_size=data_size)
            all_results.extend(results)

        return all_results

    def save_results(
        self,
        benchmark_results: list[BenchmarkResult],
        ml_results: list[MLPipelineResult],
        filename: str | None = None,
    ) -> Path:
        """Сохранение результатов в JSON."""
        import json
        from datetime import datetime

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}.json"

        output_path = self.output_dir / filename

        data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "config": {
                    "n_runs": self.config.n_runs,
                    "cold_runs": self.config.cold_runs,
                    "timeout_seconds": self.config.timeout_seconds,
                    "metrics_interval_ms": self.config.metrics_interval_ms,
                },
                "system": self._get_system_info(),
            },
            "benchmark_results": [self._result_to_dict(r) for r in benchmark_results],
            "ml_pipeline_results": [self._ml_result_to_dict(r) for r in ml_results],
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"Results saved to: {output_path}")
        return output_path

    def _get_system_info(self) -> dict[str, Any]:
        """Сбор информации о системе."""
        import platform

        info = {
            "platform": platform.system(),
            "platform_release": platform.release(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
        }

        try:
            info["polars_version"] = pl.__version__
        except Exception:
            pass

        try:
            info["duckdb_version"] = duckdb.__version__
        except Exception:
            pass

        try:
            info["cpu_count"] = psutil.cpu_count()
            info["memory_total_gb"] = round(psutil.virtual_memory().total / (1024**3), 2)
        except Exception:
            pass

        return info

    def _result_to_dict(self, result: BenchmarkResult) -> dict[str, Any]:
        """Конвертация BenchmarkResult в dict."""
        return {
            "operation": result.operation,
            "framework": result.framework,
            "mode": result.mode,
            "data_size": result.data_size,
            "run_number": result.run_number,
            "is_cold": result.is_cold,
            "wall_time_seconds": result.wall_time_seconds,
            "peak_memory_mb": result.peak_memory_mb,
            "cpu_percent": result.cpu_percent,
            "success": result.success,
            "error_message": result.error_message,
        }

    def _ml_result_to_dict(self, result: MLPipelineResult) -> dict[str, Any]:
        """Конвертация MLPipelineResult в dict."""
        return {
            "task_type": result.task_type,
            "model_name": result.model_name,
            "framework": result.framework,
            "data_size": result.data_size,
            "feature_set": result.feature_set,
            "preprocessing": result.preprocessing,
            "validation": result.validation,
            "time_loading": result.time_loading,
            "time_cleaning": result.time_cleaning,
            "time_feature_engineering": result.time_feature_engineering,
            "time_preprocessing": result.time_preprocessing,
            "time_splitting": result.time_splitting,
            "time_training": result.time_training,
            "time_validation": result.time_validation,
            "time_total": result.time_total,
            "peak_memory_mb": result.peak_memory_mb,
            "metrics": result.metrics,
            "success": result.success,
            "error_message": result.error_message,
        }


def parse_args() -> argparse.Namespace:
    """Парсинг аргументов командной строки."""
    parser = argparse.ArgumentParser(
        description="Polars vs DuckDB Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all benchmarks on data file
  python polars_duckdb_bench.py --data data.parquet

  # Run specific benchmarks
  python polars_duckdb_bench.py --data data.parquet --benchmarks read_parquet,filter_simple

  # Run only ML pipelines
  python polars_duckdb_bench.py --data data.parquet --ml-only

  # Run with specific modes
  python polars_duckdb_bench.py --data data.parquet --modes polars_eager,duckdb

  # List available benchmarks
  python polars_duckdb_bench.py --list
        """,
    )

    parser.add_argument(
        "--data", "-d",
        type=Path,
        help="Path to input data file (parquet or csv)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("results"),
        help="Output directory for results (default: results)",
    )
    parser.add_argument(
        "--benchmarks", "-b",
        type=str,
        help="Comma-separated list of benchmarks to run",
    )
    parser.add_argument(
        "--modes", "-m",
        type=str,
        default="polars_eager,polars_lazy,polars_streaming,duckdb",
        help="Comma-separated list of modes (default: all)",
    )
    parser.add_argument(
        "--ml-pipelines", "-p",
        type=str,
        help="Comma-separated list of ML pipelines to run",
    )
    parser.add_argument(
        "--ml-loader",
        type=str,
        default="polars",
        choices=["polars", "duckdb"],
        help="Data loader for ML pipelines (default: polars)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Number of warmup runs (default: 1)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of measurement runs (default: 3)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout per operation in seconds (default: 300)",
    )
    parser.add_argument(
        "--benchmarks-only",
        action="store_true",
        help="Run only operation benchmarks, skip ML pipelines",
    )
    parser.add_argument(
        "--ml-only",
        action="store_true",
        help="Run only ML pipelines, skip operation benchmarks",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available benchmarks and ML pipelines",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save results to file",
    )

    return parser.parse_args()


def main() -> None:
    """Главная функция запуска."""
    args = parse_args()

    # Список доступных бенчмарков
    if args.list:
        print("Available operation benchmarks:")
        for name in sorted(ALL_BENCHMARKS.keys()):
            print(f"  - {name}")
        print("\nAvailable ML pipelines:")
        for name in sorted(ALL_ML_PIPELINES.keys()):
            pipeline_cls = ALL_ML_PIPELINES[name]
            models = list(pipeline_cls.MODELS.keys())
            print(f"  - {name}: {', '.join(models)}")
        return

    # Проверка наличия данных
    if args.data is None:
        print("Error: --data is required. Use --list to see available benchmarks.")
        return

    if not args.data.exists():
        print(f"Error: Data file not found: {args.data}")
        return

    # Конфигурация
    config = BenchmarkConfig(
        n_runs=args.runs,
        cold_runs=args.warmup,
        timeout_seconds=args.timeout,
    )

    runner = BenchmarkRunner(config=config, output_dir=args.output)

    # Запуск бенчмарков
    benchmark_results: list[BenchmarkResult] = []
    ml_results: list[MLPipelineResult] = []

    if not args.ml_only:
        benchmark_filter = None
        if args.benchmarks:
            benchmark_filter = [b.strip() for b in args.benchmarks.split(",")]

        print("=" * 60)
        print("Running operation benchmarks")
        print("=" * 60)
        benchmark_results = runner.run_all_benchmarks(
            args.data, benchmark_filter=benchmark_filter
        )

    if not args.benchmarks_only:
        pipeline_filter = None
        if args.ml_pipelines:
            pipeline_filter = [p.strip() for p in args.ml_pipelines.split(",")]

        print("\n" + "=" * 60)
        print("Running ML pipelines")
        print("=" * 60)
        ml_results = runner.run_all_ml_pipelines(
            args.data, pipeline_filter=pipeline_filter
        )

    # Сохранение результатов
    if not args.no_save and (benchmark_results or ml_results):
        runner.save_results(benchmark_results, ml_results)

    # Итоговая статистика
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    if benchmark_results:
        successful = [r for r in benchmark_results if r.success]
        failed = [r for r in benchmark_results if not r.success]
        print(f"Operation benchmarks: {len(successful)} successful, {len(failed)} failed")

    if ml_results:
        successful = [r for r in ml_results if r.success]
        failed = [r for r in ml_results if not r.success]
        print(f"ML pipelines: {len(successful)} successful, {len(failed)} failed")


if __name__ == "__main__":
    main()
