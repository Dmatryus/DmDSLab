"""Генератор синтетических данных для ML."""

from __future__ import annotations

import logging
import os
import pickle
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime as dt
from enum import Enum
from typing import ClassVar, Literal

import duckdb
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.datasets import make_classification

# Логгер модуля
logger = logging.getLogger(__name__)


class DType(Enum):
    """Типы данных DuckDB для колонок."""

    INT8 = "INT8"
    INT64 = "INT64"
    DOUBLE = "DOUBLE"
    VARCHAR = "VARCHAR"
    BOOLEAN = "BOOLEAN"
    TIMESTAMP = "TIMESTAMP"
    DATE = "DATE"

    def __str__(self) -> str:
        """Возвращает строковое представление для использования в SQL."""
        return self.value


# Type alias для функций генерации таргета
TargetGeneratorFunc = Callable[[np.ndarray, np.random.Generator], np.ndarray]

# Type alias для progress callback
# Сигнатура: (step_name: str, current_step: int, total_steps: int, chunk_info: str | None)
# chunk_info содержит информацию о прогрессе внутри шага (например, "chunk 3/10")
ProgressCallback = Callable[[str, int, int, str | None], None]

# Константы генерации
DEFAULT_CHUNK_SIZE = 100_000
DEFAULT_KMEANS_SAMPLES = 100_000
DEFAULT_CALIBRATION_ROWS = 1_000
DEFAULT_STEP_BINS = 5
MIN_ROW_COUNT = 100

# Математические константы
EPSILON = 1e-10  # Защита от деления на ноль


def _normalize_to_unit(arr: np.ndarray, axis: int | None = 0) -> np.ndarray:
    """Нормализует массив в диапазон [0, 1].

    Args:
        arr: Входной массив.
        axis: Ось для вычисления min/max. None для всего массива.

    Returns:
        Нормализованный массив в диапазоне [0, 1].
    """
    arr_min = arr.min(axis=axis, keepdims=True) if axis is not None else arr.min()
    arr_max = arr.max(axis=axis, keepdims=True) if axis is not None else arr.max()
    return (arr - arr_min) / (arr_max - arr_min + EPSILON)

# Коэффициенты для генерации таргетов
COEF_SQUARED_SCALE = 0.5  # Масштаб квадратичных членов в polynomial
COEF_INTERACTION_SCALE = 0.3  # Масштаб взаимодействий в polynomial
COEF_EXP_SCALE = 0.1  # Масштаб коэффициентов в exponential
COEF_NONLINEAR_EXP_SCALE = 0.5  # Масштаб exp в nonlinear
COEF_OTHER_FEATURES_SCALE = 0.5  # Масштаб для дополнительных фич в step
STEP_VALUES_SCALE = 10  # Масштаб значений ступеней

# Ограничения для численной стабильности
EXP_CLIP_RANGE = (-3, 3)  # Клиппинг для exp в nonlinear
EXP_LINEAR_CLIP_RANGE = (-5, 5)  # Клиппинг для exp в exponential

# Параметры Friedman функций (оригинальные диапазоны)
FRIEDMAN_X0_MAX = 100
FRIEDMAN_X1_START = 40 * np.pi
FRIEDMAN_X1_SCALE = 520 * np.pi
FRIEDMAN_X3_MIN = 1
FRIEDMAN_X3_SCALE = 10

# Параметры RBF
RBF_MAX_CENTERS = 5
RBF_VARIANCE_DIVISOR = 2

# Коэффициенты функции Friedman #1.
# Источник: Friedman, J.H. (1991). "Multivariate Adaptive Regression Splines"
# The Annals of Statistics, 19(1), 1-67.
# Формула: 10*sin(π*x0*x1) + 20*(x2-0.5)² + 10*x3 + 5*x4
FRIEDMAN1_SIN_COEF = 10
FRIEDMAN1_SQUARED_COEF = 20
FRIEDMAN1_SQUARED_OFFSET = 0.5
FRIEDMAN1_X3_COEF = 10
FRIEDMAN1_X4_COEF = 5

# Параметры KMeans
DEFAULT_KMEANS_N_INIT = 10

# Защита от edge case в ранжировании: PERCENT_RANK()=1.0 даст n_levels,
# что выходит за диапазон [0, n_levels-1]
PERCENT_RANK_EPSILON = 0.001

# Приближение медианы распределения chi(2) для метода circles.
# Медиана chi(2) ≈ sqrt(2 * (1 - 2/9)^3) ≈ 1.386, делённая на sqrt(2) даёт ~0.98.
# Эмпирически 0.67 даёт лучшее разделение классов.
CHI2_MEDIAN_FACTOR = 0.67

# Параметры генерации таргетов
MAX_INTERACTION_PAIRS = 3  # Макс. пар взаимодействий в polynomial
# Параметры nonlinear: sin[0:3], cos[3:6], exp[6] — без пересечений
MAX_SIN_FEATURES = 3  # Макс. индекс фич для sin (exclusive): 0, 1, 2
MIN_COS_FEATURES = 3  # Мин. индекс фич для cos: 3
MAX_COS_FEATURES = 6  # Макс. индекс фич для cos (exclusive): 3, 4, 5


def _atomic_write(db: duckdb.DuckDBPyConnection, query: str, file_path: str) -> None:
    """Атомарная запись в parquet через временный файл.

    Args:
        db: Соединение с DuckDB.
        query: SQL запрос (SELECT).
        file_path: Путь к файлу.
    """
    tmp_path = f"{file_path}.tmp.parquet"
    try:
        db.execute(f"COPY ({query}) TO '{tmp_path}' (FORMAT PARQUET)")
        os.replace(tmp_path, file_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _iter_chunks(
    db: duckdb.DuckDBPyConnection,
    meta: "Meta",
    columns: list[str],
    chunk_size: int,
) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
    """Итератор по чанкам данных.

    Args:
        db: Соединение с DuckDB.
        meta: Текущая метаинформация.
        columns: Список колонок для чтения.
        chunk_size: Размер чанка.

    Yields:
        Кортеж (ids, features) где features — матрица (n_rows, n_cols).
    """
    cols_sql = ", ".join(columns)
    n_chunks = (meta.row_count + chunk_size - 1) // chunk_size

    for chunk_idx in range(n_chunks):
        offset = chunk_idx * chunk_size
        limit = min(chunk_size, meta.row_count - offset)

        chunk_data = db.execute(
            f"SELECT id, {cols_sql} FROM '{meta.file_path}' "
            f"ORDER BY id LIMIT {limit} OFFSET {offset}"
        ).fetchnumpy()

        ids = chunk_data["id"]
        features = np.column_stack([chunk_data[col] for col in columns])

        yield ids, features


def _iter_chunks_dict(
    db: duckdb.DuckDBPyConnection,
    file_path: str,
    row_count: int,
    columns: list[str],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> Generator[dict[str, np.ndarray], None, None]:
    """Итератор по чанкам данных, возвращающий словарь.

    Упрощённая версия для случаев, когда не нужна матрица фич.

    Args:
        db: Соединение с DuckDB.
        file_path: Путь к parquet файлу.
        row_count: Общее количество строк.
        columns: Список колонок для чтения (включая id если нужен).
        chunk_size: Размер чанка.

    Yields:
        Словарь {column_name: values} для каждого чанка.
    """
    cols_sql = ", ".join(columns)
    n_chunks = (row_count + chunk_size - 1) // chunk_size

    for chunk_idx in range(n_chunks):
        offset = chunk_idx * chunk_size
        limit = min(chunk_size, row_count - offset)

        chunk_data = db.execute(
            f"SELECT {cols_sql} FROM '{file_path}' "
            f"ORDER BY id LIMIT {limit} OFFSET {offset}"
        ).fetchnumpy()

        yield chunk_data


@contextmanager
def _chunked_target_writer(
    db: duckdb.DuckDBPyConnection,
    meta: "Meta",
    col_name: str,
    col_type: str,
) -> Generator[Callable[[int, np.ndarray, np.ndarray], None], None, None]:
    """Контекстный менеджер для чанкованной записи таргета.

    Управляет временными файлами и финальным объединением.

    Args:
        db: Соединение с DuckDB.
        meta: Текущая метаинформация.
        col_name: Имя создаваемой колонки.
        col_type: Тип колонки (DOUBLE, INT8, etc).

    Yields:
        Функция write_chunk(chunk_idx, ids, values) для записи чанков.
    """
    temp_files: list[str] = []

    def write_chunk(chunk_idx: int, ids: np.ndarray, values: np.ndarray) -> None:
        chunk_file = f"{meta.file_path}.chunk_{col_name}_{chunk_idx}.parquet"
        temp_files.append(chunk_file)

        db.register("chunk_data", {"id": ids, "value": values})
        try:
            db.execute(
                f"COPY (SELECT id, value FROM chunk_data) "
                f"TO '{chunk_file}' (FORMAT PARQUET)"
            )
        finally:
            db.unregister("chunk_data")

    try:
        yield write_chunk

        # Объединяем чанки и джойним с main
        if temp_files:
            chunks_union = " UNION ALL ".join(
                [f"SELECT * FROM '{f}'" for f in temp_files]
            )
            _atomic_write(
                db,
                f"""
                    SELECT m.*, CAST(t.value AS {col_type}) AS {col_name}
                    FROM '{meta.file_path}' AS m
                    JOIN ({chunks_union}) AS t ON m.id = t.id
                """,
                meta.file_path,
            )
    finally:
        for f in temp_files:
            if os.path.exists(f):
                os.remove(f)


@dataclass
class GeneratorConfig:
    """Конфигурация генератора синтетических данных.

    Attributes:
        target_size: Целевой размер файла (например, "100MB", "1GB").
        output_path: Директория для сохранения результатов.
        seed: Seed для воспроизводимости. None для случайной генерации.
        n_numeric: Количество числовых фич.
        informative_ratio: Доля информативных фич (0-1).
        n_categories: Количество категориальных колонок.
        category_cardinality: Количество уникальных значений в категории.
        category_method: Метод генерации категорий ("kmeans" или "quantile").
        task: Тип ML-задачи ("regression", "binary", "multiclass", "ranking").
        target_method: Метод генерации целевой переменной.
        target_noise: Уровень шума в таргете (0-1).
        n_classes: Количество классов для multiclass.
        with_datetime: Генерировать ли колонку timestamp.
        datetime_start: Начальная дата для timestamp.
        datetime_end: Конечная дата для timestamp.
        with_date: Генерировать ли колонку date из timestamp.
        n_booleans: Количество boolean колонок.
        nullable_ratio: Доля NULL значений (0-1).
    """

    target_size: str
    output_path: str
    seed: int | None = None

    # Числовые фичи
    n_numeric: int = 20
    informative_ratio: float = 0.5

    # Категориальные фичи
    n_categories: int = 0
    category_cardinality: int = 10
    category_method: Literal["kmeans", "quantile"] = "kmeans"

    # Таргет
    task: Literal["regression", "binary", "multiclass", "ranking"] | None = None
    target_method: Literal[
        "linear", "polynomial", "nonlinear",
        "friedman1", "friedman2", "friedman3",
        "exponential", "logarithmic", "step", "radial",
        "xor", "circles", "moons", "clusters",
    ] = "linear"
    target_noise: float = 0.1
    n_classes: int = 5

    # Временные данные
    with_datetime: bool = False
    datetime_start: str = "2020-01-01"
    datetime_end: str = "2024-01-01"
    with_date: bool = False

    # Boolean
    n_booleans: int = 0

    # NULL
    nullable_ratio: float = 0.0

    def __post_init__(self) -> None:
        """Валидация параметров конфига."""
        if self.n_numeric < 1:
            raise ValueError("n_numeric должен быть >= 1")

        if not 0.0 <= self.informative_ratio <= 1.0:
            raise ValueError("informative_ratio должен быть в диапазоне [0, 1]")

        if self.n_categories < 0:
            raise ValueError("n_categories должен быть >= 0")

        if self.category_cardinality < 2:
            raise ValueError("category_cardinality должен быть >= 2")

        if self.n_classes < 2:
            raise ValueError("n_classes должен быть >= 2")

        if not 0.0 <= self.target_noise <= 1.0:
            raise ValueError("target_noise должен быть в диапазоне [0, 1]")

        if self.n_booleans < 0:
            raise ValueError("n_booleans должен быть >= 0")

        if not 0.0 <= self.nullable_ratio <= 1.0:
            raise ValueError("nullable_ratio должен быть в диапазоне [0, 1]")

        # Валидация дат
        try:
            start = dt.strptime(self.datetime_start, "%Y-%m-%d")
            end = dt.strptime(self.datetime_end, "%Y-%m-%d")
            if start >= end:
                raise ValueError("datetime_start должен быть раньше datetime_end")
        except ValueError as e:
            if "должен быть" in str(e):
                raise
            raise ValueError(f"Некорректный формат даты (ожидается YYYY-MM-DD): {e}")

        # Валидация совместимости task и target_method
        regression_methods = {
            "linear", "polynomial", "nonlinear",
            "friedman1", "friedman2", "friedman3",
            "exponential", "logarithmic", "step", "radial",
        }
        classification_methods = {"xor", "circles", "moons", "clusters"}

        if self.task in ("binary", "multiclass") and self.target_method in regression_methods:
            logger.warning(
                "target_method '%s' является регрессионным и будет бинаризован "
                "для задачи '%s'",
                self.target_method,
                self.task,
            )

        if self.task == "regression" and self.target_method in classification_methods:
            raise ValueError(
                f"target_method '{self.target_method}' не поддерживается для задачи "
                f"'regression'. Используйте один из: {sorted(regression_methods)}"
            )

        if self.task == "multiclass" and self.target_method in classification_methods:
            raise ValueError(
                f"target_method '{self.target_method}' не поддерживается для задачи "
                f"'multiclass'. Используйте один из: {sorted(regression_methods)}"
            )


@dataclass
class Meta:
    """Метаинформация о состоянии данных.

    Хранит информацию о текущем состоянии сгенерированного датасета,
    включая схему колонок, теги и прогресс выполнения pipeline.

    Attributes:
        file_path: Путь к файлу main.parquet.
        row_count: Количество строк в датасете.
        columns: Словарь {имя_колонки: тип_данных}.
        column_tags: Словарь {имя_колонки: список_тегов}.
        completed_steps: Список имён выполненных шагов pipeline.
    """

    file_path: str
    row_count: int = 0
    columns: dict[str, DType] = field(default_factory=dict)
    column_tags: dict[str, list[str]] = field(default_factory=dict)
    completed_steps: list[str] = field(default_factory=list)

    def get_columns_by_tag(
        self, prefer_tag: str, fallback_tag: str | None = None
    ) -> list[str]:
        """Возвращает колонки по тегу с опциональным fallback.

        Args:
            prefer_tag: Предпочитаемый тег.
            fallback_tag: Fallback тег, если prefer_tag не найден.

        Returns:
            Список имён колонок.
        """
        cols = [col for col, tags in self.column_tags.items() if prefer_tag in tags]
        if not cols and fallback_tag:
            cols = [
                col for col, tags in self.column_tags.items() if fallback_tag in tags
            ]
        return cols


class Transformer(ABC):
    """Базовый класс трансформера pipeline."""

    default_name: ClassVar[str] = "transformer"
    requires: ClassVar[list[type["Transformer"]]] = []

    def __init__(self, name: str | None = None):
        """Инициализирует трансформер.

        Args:
            name: Кастомное имя трансформера. Если None, используется default_name.
        """
        self.name = name or self.default_name

    @abstractmethod
    def transform(self, db: duckdb.DuckDBPyConnection, meta: Meta) -> Meta:
        """Выполняет трансформацию.

        Args:
            db: Соединение с DuckDB.
            meta: Текущая метаинформация о данных.

        Returns:
            Обновлённая Meta.
        """
        raise NotImplementedError


class Init(Transformer):
    """Создаёт таблицу с колонкой id."""

    default_name: ClassVar[str] = "init"
    requires: ClassVar[list[type[Transformer]]] = []

    def __init__(self, row_count: int, name: str | None = None):
        """Инициализирует Init.

        Args:
            row_count: Количество строк для генерации.
            name: Кастомное имя трансформера.
        """
        super().__init__(name)
        self.row_count = row_count

    def transform(self, db: duckdb.DuckDBPyConnection, meta: Meta) -> Meta:
        """Создаёт таблицу main с колонкой id и сохраняет в parquet.

        Args:
            db: Соединение с DuckDB.
            meta: Текущая метаинформация.

        Returns:
            Обновлённая Meta с информацией о колонке id.
        """
        db.execute(
            f"""
            CREATE TABLE main AS
            SELECT CAST(range AS INT64) AS id
            FROM range({self.row_count})
        """
        )
        db.execute(f"COPY main TO '{meta.file_path}' (FORMAT PARQUET)")

        meta.row_count = self.row_count
        meta.columns["id"] = DType.INT64
        meta.completed_steps.append(self.name)
        return meta


class Numeric(Transformer):
    """Генерирует числовые фичи через sklearn с чанкованием."""

    default_name: ClassVar[str] = "numeric"
    requires: ClassVar[list[type[Transformer]]] = [Init]

    def __init__(
        self,
        n_features: int = 20,
        informative_ratio: float = 0.5,
        seed: int | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        name: str | None = None,
    ):
        """Инициализирует генератор числовых фич.

        Args:
            n_features: Общее количество фич.
            informative_ratio: Доля информативных фич (имеющих корреляцию с target).
            seed: Seed для воспроизводимости.
            chunk_size: Размер чанка для генерации.
            name: Кастомное имя трансформера.
        """
        super().__init__(name)
        self.n_features = n_features
        self.informative_ratio = informative_ratio
        self.seed = seed
        self.chunk_size = chunk_size

    def transform(self, db: duckdb.DuckDBPyConnection, meta: Meta) -> Meta:
        """Генерирует числовые фичи чанками и добавляет к main.

        Args:
            db: Соединение с DuckDB.
            meta: Текущая метаинформация.

        Returns:
            Обновлённая Meta с информацией о сгенерированных фичах.
        """
        feature_names = [f"numeric_{i}" for i in range(self.n_features)]
        n_informative = int(self.n_features * self.informative_ratio)

        n_chunks = (meta.row_count + self.chunk_size - 1) // self.chunk_size
        logger.debug(
            "Numeric: генерация %d фич (%d информативных) в %d чанках",
            self.n_features,
            n_informative,
            n_chunks,
        )
        temp_files: list[str] = []

        try:
            for chunk_idx in range(n_chunks):
                start_idx = chunk_idx * self.chunk_size
                end_idx = min(start_idx + self.chunk_size, meta.row_count)
                chunk_rows = end_idx - start_idx

                features, _ = make_classification(
                    n_samples=chunk_rows,
                    n_features=self.n_features,
                    n_informative=n_informative,
                    n_redundant=0,
                    n_clusters_per_class=1,
                    random_state=None if self.seed is None else self.seed + chunk_idx,
                )

                ids = np.arange(start_idx, end_idx, dtype=np.int64)
                chunk_data = {"id": ids}
                for i, name in enumerate(feature_names):
                    chunk_data[name] = features[:, i]
                db.register("chunk_np", chunk_data)

                chunk_file = f"{meta.file_path}.features_chunk_{chunk_idx}.parquet"
                temp_files.append(chunk_file)

                feature_cols = ", ".join(feature_names)
                db.execute(
                    f"""
                    COPY (
                        SELECT id, {feature_cols}
                        FROM chunk_np
                    ) TO '{chunk_file}' (FORMAT PARQUET)
                """
                )
                db.unregister("chunk_np")

            # Объединяем чанки и джойним с main
            chunks_union = " UNION ALL ".join([f"SELECT * FROM '{f}'" for f in temp_files])
            _atomic_write(
                db,
                f"""
                    SELECT m.*, f.* EXCLUDE (id)
                    FROM '{meta.file_path}' AS m
                    JOIN ({chunks_union}) AS f USING (id)
                """,
                meta.file_path,
            )
        finally:
            # Удаляем временные файлы в любом случае
            for f in temp_files:
                if os.path.exists(f):
                    os.remove(f)

        # Обновляем meta
        for i, col_name in enumerate(feature_names):
            meta.columns[col_name] = DType.DOUBLE
            tags = ["numeric"]
            if i < n_informative:
                tags.append("informative")
            meta.column_tags[col_name] = tags

        meta.completed_steps.append(self.name)
        return meta


class Category(Transformer):
    """Генерирует категориальную колонку на основе числовых фич."""

    default_name: ClassVar[str] = "category"
    requires: ClassVar[list[type[Transformer]]] = [Numeric]

    def __init__(
        self,
        cardinality: int = 10,
        method: Literal["kmeans", "quantile"] = "kmeans",
        noise_ratio: float = 0.0,
        seed: int | None = None,
        name: str | None = None,
        max_kmeans_samples: int = DEFAULT_KMEANS_SAMPLES,
    ):
        """Инициализирует генератор категорий.

        Args:
            cardinality: Количество уникальных категорий.
            method: Метод генерации — "kmeans" (кластеризация) или "quantile" (бакетизация).
            noise_ratio: Доля значений для случайной подмены (только для quantile).
            seed: Seed для воспроизводимости.
            name: Кастомное имя трансформера.
            max_kmeans_samples: Максимум строк для обучения KMeans (сэмплирование).
        """
        super().__init__(name)
        self.cardinality = cardinality
        self.method = method
        self.noise_ratio = noise_ratio
        self.seed = seed
        self.max_kmeans_samples = max_kmeans_samples

    def _generate_kmeans(
        self,
        db: duckdb.DuckDBPyConnection,
        meta: Meta,
        numeric_cols: list[str],
    ) -> np.ndarray:
        """Генерирует категории через MiniBatchKMeans.

        Для больших данных использует сэмплирование: обучает KMeans на выборке,
        затем применяет predict ко всем данным чанками.

        Args:
            db: Соединение с DuckDB.
            meta: Текущая метаинформация.
            numeric_cols: Список числовых колонок.

        Returns:
            Массив меток категорий.
        """
        cols_sql = ", ".join(numeric_cols)

        # Обучаем KMeans на сэмпле
        if meta.row_count > self.max_kmeans_samples:
            sample_query = f"""
                SELECT {cols_sql} FROM '{meta.file_path}'
                USING SAMPLE {self.max_kmeans_samples}
            """
            sample_data = db.execute(sample_query).fetchnumpy()
            sample_array = np.column_stack([sample_data[col] for col in numeric_cols])
        else:
            full_data = db.execute(f"SELECT {cols_sql} FROM '{meta.file_path}'").fetchnumpy()
            sample_array = np.column_stack([full_data[col] for col in numeric_cols])

        kmeans = MiniBatchKMeans(
            n_clusters=self.cardinality,
            random_state=self.seed,
            n_init=DEFAULT_KMEANS_N_INIT,
        )
        kmeans.fit(sample_array)

        # Predict чанками для экономии памяти
        chunk_size = DEFAULT_CHUNK_SIZE
        labels = np.empty(meta.row_count, dtype=np.int32)

        for offset in range(0, meta.row_count, chunk_size):
            limit = min(chunk_size, meta.row_count - offset)
            chunk_query = f"""
                SELECT {cols_sql} FROM '{meta.file_path}'
                LIMIT {limit} OFFSET {offset}
            """
            chunk_data = db.execute(chunk_query).fetchnumpy()
            chunk_array = np.column_stack([chunk_data[col] for col in numeric_cols])
            labels[offset:offset + limit] = kmeans.predict(chunk_array)

        return labels

    def _generate_quantile(
        self,
        db: duckdb.DuckDBPyConnection,
        meta: Meta,
        numeric_cols: list[str],
    ) -> np.ndarray:
        """Генерирует категории через бакетизацию квантилями + шум.

        Args:
            db: Соединение с DuckDB.
            meta: Текущая метаинформация.
            numeric_cols: Список числовых колонок.

        Returns:
            Массив меток категорий.
        """
        # Берём первую числовую колонку
        source_col = numeric_cols[0]

        data = db.execute(f"SELECT {source_col} FROM '{meta.file_path}'").fetchnumpy()
        values = data[source_col]

        # Бакетизация через квантили
        percentiles = np.linspace(0, 100, self.cardinality + 1)
        bins = np.percentile(values, percentiles)
        labels = np.digitize(values, bins[1:-1])

        # Добавляем шум — случайная подмена категории
        if self.noise_ratio > 0:
            rng = np.random.default_rng(self.seed)
            noise_mask = rng.random(len(values)) < self.noise_ratio
            random_labels = rng.integers(0, self.cardinality, size=len(values))
            labels = np.where(noise_mask, random_labels, labels)

        return labels

    def transform(self, db: duckdb.DuckDBPyConnection, meta: Meta) -> Meta:
        """Генерирует категориальную колонку и добавляет к main.

        Args:
            db: Соединение с DuckDB.
            meta: Текущая метаинформация.

        Returns:
            Обновлённая Meta с информацией о категориальной колонке.
        """
        numeric_cols = [
            col for col, tags in meta.column_tags.items() if "numeric" in tags
        ]

        if self.method == "kmeans":
            labels = self._generate_kmeans(db, meta, numeric_cols)
        else:
            labels = self._generate_quantile(db, meta, numeric_cols)

        # Имя колонки на основе имени трансформера
        col_name = f"cat_{self.name}"

        # Читаем id и добавляем labels
        ids = db.execute(f"SELECT id FROM '{meta.file_path}'").fetchnumpy()["id"]
        db.register("labels_np", {"id": ids, "label": labels})

        # Добавляем колонку к main
        _atomic_write(
            db,
            f"""
                SELECT m.*, CAST(l.label AS VARCHAR) AS {col_name}
                FROM '{meta.file_path}' AS m
                JOIN labels_np AS l ON m.id = l.id
            """,
            meta.file_path,
        )
        db.unregister("labels_np")

        # Обновляем meta
        meta.columns[col_name] = DType.VARCHAR
        meta.column_tags[col_name] = ["category"]
        meta.completed_steps.append(self.name)
        return meta


class TargetGeneratorMixin:
    """Миксин с методами генерации таргетов для регрессии и классификации."""

    def _generate_linear(
        self, features: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Линейная зависимость: y = X @ coef.

        Args:
            features: Матрица фич (n_samples, n_features).
            rng: Генератор случайных чисел.

        Returns:
            Вектор таргета.
        """
        n_features = features.shape[1]
        coef = rng.standard_normal(n_features)
        return features @ coef

    def _generate_polynomial(
        self, features: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Полиномиальная зависимость: квадраты + взаимодействия.

        Args:
            features: Матрица фич (n_samples, n_features).
            rng: Генератор случайных чисел.

        Returns:
            Вектор таргета.
        """
        n_features = features.shape[1]

        # Линейная часть
        coef_linear = rng.standard_normal(n_features)
        y = features @ coef_linear

        # Квадратичная часть
        coef_squared = rng.standard_normal(n_features) * COEF_SQUARED_SCALE
        y += (features**2) @ coef_squared

        # Взаимодействия (первые пары фич)
        for i in range(min(n_features - 1, MAX_INTERACTION_PAIRS)):
            coef_inter = rng.standard_normal() * COEF_INTERACTION_SCALE
            y += coef_inter * features[:, i] * features[:, i + 1]

        return y

    def _generate_nonlinear(
        self, features: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Нелинейная зависимость: sin, cos, exp.

        Args:
            features: Матрица фич (n_samples, n_features).
            rng: Генератор случайных чисел.

        Returns:
            Вектор таргета.
        """
        n_features = features.shape[1]
        y = np.zeros(features.shape[0])

        # sin от первых фич
        for i in range(min(n_features, MAX_SIN_FEATURES)):
            coef = rng.standard_normal()
            y += coef * np.sin(features[:, i])

        # cos от следующих
        for i in range(MIN_COS_FEATURES, min(n_features, MAX_COS_FEATURES)):
            coef = rng.standard_normal()
            y += coef * np.cos(features[:, i])

        # exp (с ограничением чтобы не взорвалось)
        if n_features > MAX_COS_FEATURES:
            coef = rng.standard_normal() * COEF_NONLINEAR_EXP_SCALE
            clipped = np.clip(features[:, MAX_COS_FEATURES], *EXP_CLIP_RANGE)
            y += coef * np.exp(clipped)

        # Взаимодействие
        if n_features >= 2:
            coef = rng.standard_normal()
            y += coef * features[:, 0] * np.sin(features[:, 1])

        return y

    def _generate_friedman1(
        self, features: np.ndarray, _rng: np.random.Generator
    ) -> np.ndarray:
        """Friedman #1: классический бенчмарк для нелинейной регрессии.

        Формула: 10*sin(π*x0*x1) + 20*(x2-0.5)² + 10*x3 + 5*x4
        Использует только первые 5 фич, остальные — шум.

        Args:
            features: Матрица фич (n_samples, n_features).
            _rng: Не используется (сохранён для единой сигнатуры TargetGeneratorFunc).

        Returns:
            Вектор таргета.
        """
        n_features = features.shape[1]

        # Нормализуем фичи в [0, 1] для соответствия оригинальной формуле
        f_norm = _normalize_to_unit(features, axis=0)

        y = np.zeros(features.shape[0])

        if n_features >= 2:
            y += FRIEDMAN1_SIN_COEF * np.sin(np.pi * f_norm[:, 0] * f_norm[:, 1])
        if n_features >= 3:
            y += FRIEDMAN1_SQUARED_COEF * (f_norm[:, 2] - FRIEDMAN1_SQUARED_OFFSET) ** 2
        if n_features >= 4:
            y += FRIEDMAN1_X3_COEF * f_norm[:, 3]
        if n_features >= 5:
            y += FRIEDMAN1_X4_COEF * f_norm[:, 4]

        return y

    def _generate_friedman2(
        self, features: np.ndarray, _rng: np.random.Generator
    ) -> np.ndarray:
        """Friedman #2: sqrt(x0² + (x1*x2 - 1/(x1*x3))²).

        Args:
            features: Матрица фич (n_samples, n_features).
            _rng: Не используется (сохранён для единой сигнатуры TargetGeneratorFunc).

        Returns:
            Вектор таргета.
        """
        n_features = features.shape[1]
        if n_features < 4:
            return self._generate_linear(features, _rng)

        # Масштабируем как в оригинале: x0 in [0,100], x1 in [40π, 560π], x2 in [0,1], x3 in [1,11]
        f_norm = _normalize_to_unit(features, axis=0)

        x0 = f_norm[:, 0] * FRIEDMAN_X0_MAX
        x1 = FRIEDMAN_X1_START + f_norm[:, 1] * FRIEDMAN_X1_SCALE
        x2 = f_norm[:, 2]
        x3 = FRIEDMAN_X3_MIN + f_norm[:, 3] * FRIEDMAN_X3_SCALE

        # Защита от деления на ноль
        denom = x1 * x3 + EPSILON
        inner = x1 * x2 - 1 / denom

        return np.sqrt(x0**2 + inner**2)

    def _generate_friedman3(
        self, features: np.ndarray, _rng: np.random.Generator
    ) -> np.ndarray:
        """Friedman #3: atan((x1*x2 - 1/(x1*x3)) / x0).

        Args:
            features: Матрица фич (n_samples, n_features).
            _rng: Не используется (сохранён для единой сигнатуры TargetGeneratorFunc).

        Returns:
            Вектор таргета.
        """
        n_features = features.shape[1]
        if n_features < 4:
            return self._generate_linear(features, _rng)

        f_norm = _normalize_to_unit(features, axis=0)

        x0 = f_norm[:, 0] * FRIEDMAN_X0_MAX + EPSILON  # Защита от деления на ноль
        x1 = FRIEDMAN_X1_START + f_norm[:, 1] * FRIEDMAN_X1_SCALE
        x2 = f_norm[:, 2]
        x3 = FRIEDMAN_X3_MIN + f_norm[:, 3] * FRIEDMAN_X3_SCALE

        denom = x1 * x3 + EPSILON
        inner = x1 * x2 - 1 / denom

        return np.arctan(inner / x0)

    def _generate_exponential(
        self, features: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Экспоненциальная зависимость: exp(X @ coef) с масштабированием.

        Args:
            features: Матрица фич (n_samples, n_features).
            rng: Генератор случайных чисел.

        Returns:
            Вектор таргета.
        """
        n_features = features.shape[1]

        # Малые коэффициенты чтобы exp не взорвался
        coef = rng.standard_normal(n_features) * COEF_EXP_SCALE
        linear = features @ coef

        # Клиппинг для стабильности
        linear = np.clip(linear, *EXP_LINEAR_CLIP_RANGE)

        return np.exp(linear)

    def _generate_logarithmic(
        self, features: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Логарифмическая зависимость: log(|X| + 1) @ coef.

        Args:
            features: Матрица фич (n_samples, n_features).
            rng: Генератор случайных чисел.

        Returns:
            Вектор таргета.
        """
        n_features = features.shape[1]
        coef = rng.standard_normal(n_features)

        # log(|x| + 1) для стабильности
        log_features = np.log(np.abs(features) + 1)

        return log_features @ coef

    def _generate_step(
        self, features: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Кусочно-постоянная (ступенчатая) функция.

        Args:
            features: Матрица фич (n_samples, n_features).
            rng: Генератор случайных чисел.

        Returns:
            Вектор таргета.
        """
        n_features = features.shape[1]
        n_steps = DEFAULT_STEP_BINS

        # Берём первую фичу и делим на ступени
        x = features[:, 0]
        percentiles = np.linspace(0, 100, n_steps + 1)
        bins = np.percentile(x, percentiles)

        # Случайные значения для каждой ступени
        step_values = rng.standard_normal(n_steps) * STEP_VALUES_SCALE
        step_idx = np.digitize(x, bins[1:-1])

        y = step_values[step_idx]

        # Добавляем влияние других фич
        if n_features > 1:
            coef = rng.standard_normal(n_features - 1) * COEF_OTHER_FEATURES_SCALE
            y += features[:, 1:] @ coef

        return y

    def _generate_radial(
        self, features: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Радиальная зависимость от расстояния до случайных центров (RBF-подобная).

        Args:
            features: Матрица фич (n_samples, n_features).
            rng: Генератор случайных чисел.

        Returns:
            Вектор таргета.
        """
        n_samples, n_features = features.shape
        n_centers = min(RBF_MAX_CENTERS, n_features)

        # Случайные центры в пространстве фич
        centers = rng.standard_normal((n_centers, n_features))
        weights = rng.standard_normal(n_centers)

        y = np.zeros(n_samples)

        for i in range(n_centers):
            # Евклидово расстояние до центра
            dist = np.sqrt(np.sum((features - centers[i]) ** 2, axis=1))
            # RBF-ядро
            y += weights[i] * np.exp(-(dist**2) / (RBF_VARIANCE_DIVISOR * n_features))

        return y

    def _get_regression_generators(self) -> dict:
        """Возвращает словарь методов генерации для регрессии.

        Returns:
            Словарь {имя_метода: функция_генерации}.
        """
        return {
            "linear": self._generate_linear,
            "polynomial": self._generate_polynomial,
            "nonlinear": self._generate_nonlinear,
            "friedman1": self._generate_friedman1,
            "friedman2": self._generate_friedman2,
            "friedman3": self._generate_friedman3,
            "exponential": self._generate_exponential,
            "logarithmic": self._generate_logarithmic,
            "step": self._generate_step,
            "radial": self._generate_radial,
        }


class RegressionTarget(TargetGeneratorMixin, Transformer):
    """Генерирует целевую переменную для регрессии на основе существующих фич."""

    default_name: ClassVar[str] = "regression_target"
    requires: ClassVar[list[type[Transformer]]] = [Numeric]

    def __init__(
        self,
        method: Literal[
            "linear",
            "polynomial",
            "nonlinear",
            "friedman1",
            "friedman2",
            "friedman3",
            "exponential",
            "logarithmic",
            "step",
            "radial",
        ] = "linear",
        noise: float = 0.1,
        seed: int | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        name: str | None = None,
    ):
        """Инициализирует генератор таргета регрессии.

        Args:
            method: Метод генерации зависимости:
                - "linear": y = X @ coef
                - "polynomial": квадраты + взаимодействия пар фич
                - "nonlinear": sin/cos/exp комбинации
                - "friedman1": 10*sin(π*x0*x1) + 20*(x2-0.5)² + 10*x3 + 5*x4
                - "friedman2": sqrt(x0² + (x1*x2 - 1/(x1*x3))²)
                - "friedman3": atan((x1*x2 - 1/(x1*x3)) / x0)
                - "exponential": exp(X @ coef) с масштабированием
                - "logarithmic": log(|X| + 1) @ coef
                - "step": кусочно-постоянная функция
                - "radial": зависимость от расстояния до случайных центров
            noise: Стандартное отклонение гауссова шума (относительно std(y)).
            seed: Seed для воспроизводимости.
            chunk_size: Размер чанка для обработки данных.
            name: Кастомное имя трансформера.
        """
        super().__init__(name)
        self.method = method
        self.noise = noise
        self.seed = seed
        self.chunk_size = chunk_size

    def _estimate_noise_scale(
        self,
        db: duckdb.DuckDBPyConnection,
        meta: Meta,
        informative_cols: list[str],
        generator: TargetGeneratorFunc,
        rng: np.random.Generator,
    ) -> float:
        """Оценивает масштаб шума на sample данных.

        Args:
            db: Соединение с DuckDB.
            meta: Текущая метаинформация.
            informative_cols: Список информативных колонок.
            generator: Функция генерации таргета.
            rng: Генератор случайных чисел.

        Returns:
            Масштаб шума (noise * std(target)).
        """
        if self.noise <= 0:
            return 0.0

        cols_sql = ", ".join(informative_cols)
        sample_size = min(self.chunk_size, meta.row_count)

        sample_data = db.execute(
            f"SELECT {cols_sql} FROM '{meta.file_path}' LIMIT {sample_size}"
        ).fetchnumpy()
        sample_features = np.column_stack(
            [sample_data[col] for col in informative_cols]
        )

        # Генерируем таргет на sample для оценки std
        sample_target = generator(sample_features, rng)
        return self.noise * np.std(sample_target)

    def transform(self, db: duckdb.DuckDBPyConnection, meta: Meta) -> Meta:
        """Генерирует таргет регрессии на основе информативных фич.

        Использует чанкованную обработку для экономии памяти.

        Args:
            db: Соединение с DuckDB.
            meta: Текущая метаинформация.

        Returns:
            Обновлённая Meta с информацией о таргете.
        """
        informative_cols = meta.get_columns_by_tag("informative", "numeric")
        col_name = "target_reg"

        generators = self._get_regression_generators()
        if self.method not in generators:
            raise ValueError(
                f"Неизвестный метод: {self.method}. Доступны: {list(generators.keys())}"
            )

        generator = generators[self.method]

        # Оцениваем масштаб шума на sample (отдельный rng для изоляции)
        estimation_rng = np.random.default_rng(self.seed)
        noise_scale = self._estimate_noise_scale(
            db, meta, informative_cols, generator, estimation_rng
        )

        # Основной rng для генерации (независим от estimation)
        rng = np.random.default_rng(self.seed)

        logger.debug(
            "RegressionTarget: метод=%s, noise_scale=%.4f, колонок=%d",
            self.method,
            noise_scale,
            len(informative_cols),
        )

        with _chunked_target_writer(db, meta, col_name, "DOUBLE") as write_chunk:
            for chunk_idx, (ids, features) in enumerate(
                _iter_chunks(db, meta, informative_cols, self.chunk_size)
            ):
                # Генерируем таргет для чанка
                target = generator(features, rng)

                # Добавляем шум
                if noise_scale > 0:
                    target = target + rng.normal(0, noise_scale, size=len(target))

                write_chunk(chunk_idx, ids, target)

        meta.columns[col_name] = DType.DOUBLE
        meta.column_tags[col_name] = ["target", "regression"]
        meta.completed_steps.append(self.name)
        return meta


class BinaryTarget(TargetGeneratorMixin, Transformer):
    """Генерирует целевую переменную для бинарной классификации."""

    default_name: ClassVar[str] = "binary_target"
    requires: ClassVar[list[type[Transformer]]] = [Numeric]

    def __init__(
        self,
        method: Literal[
            "linear",
            "polynomial",
            "nonlinear",
            "friedman1",
            "friedman2",
            "friedman3",
            "exponential",
            "logarithmic",
            "step",
            "radial",
            "xor",
            "circles",
            "moons",
            "clusters",
        ] = "linear",
        threshold: float = 0.5,
        flip_ratio: float = 0.0,
        seed: int | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        name: str | None = None,
    ):
        """Инициализирует генератор бинарного таргета.

        Args:
            method: Метод генерации. Регрессионные методы бинаризуются по квантилю:
                - "linear", "polynomial", "nonlinear", "friedman1", etc.
                Специфичные для классификации (возвращают метки напрямую):
                - "xor": XOR от первых двух фич (линейно неразделимо)
                - "circles": концентрические окружности
                - "moons": два полумесяца
                - "clusters": KMeans на 2 кластера
            threshold: Квантиль для порога бинаризации (для регрессионных методов).
            flip_ratio: Доля меток для случайного переключения (шум).
            seed: Seed для воспроизводимости.
            chunk_size: Размер чанка для обработки данных.
            name: Кастомное имя трансформера.
        """
        super().__init__(name)
        self.method = method
        self.threshold = threshold
        self.flip_ratio = flip_ratio
        self.seed = seed
        self.chunk_size = chunk_size

    def _compute_statistics(
        self,
        db: duckdb.DuckDBPyConnection,
        meta: Meta,
        informative_cols: list[str],
    ) -> dict:
        """Вычисляет статистики для классификационных методов на sample.

        Args:
            db: Соединение с DuckDB.
            meta: Текущая метаинформация.
            informative_cols: Список информативных колонок.

        Returns:
            Словарь со статистиками (медианы, mean, std).
        """
        cols_sql = ", ".join(informative_cols[:2])  # Нужны только первые 2 фичи
        sample_size = min(self.chunk_size, meta.row_count)

        sample_data = db.execute(
            f"SELECT {cols_sql} FROM '{meta.file_path}' USING SAMPLE {sample_size}"
        ).fetchnumpy()

        col0 = informative_cols[0]
        col1 = informative_cols[1] if len(informative_cols) > 1 else col0

        return {
            "median_0": np.median(sample_data[col0]),
            "median_1": np.median(sample_data[col1]),
            "mean_0": np.mean(sample_data[col0]),
            "mean_1": np.mean(sample_data[col1]),
            "std_0": np.std(sample_data[col0]),
            "std_1": np.std(sample_data[col1]),
        }

    def _generate_xor_chunked(
        self, features: np.ndarray, stats: dict
    ) -> np.ndarray:
        """XOR с предвычисленными медианами."""
        x0 = features[:, 0] > stats["median_0"]
        x1 = features[:, 1] > stats["median_1"] if features.shape[1] > 1 else x0
        return (x0 ^ x1).astype(np.int8)

    def _generate_circles_chunked(
        self, features: np.ndarray, stats: dict
    ) -> np.ndarray:
        """Circles с предвычисленными статистиками."""
        f = (
            features[:, :2]
            if features.shape[1] >= 2
            else np.column_stack([features[:, 0], features[:, 0]])
        )
        # Нормализуем используя предвычисленные статистики
        f_norm_0 = (f[:, 0] - stats["mean_0"]) / (stats["std_0"] + EPSILON)
        f_norm_1 = (f[:, 1] - stats["mean_1"]) / (stats["std_1"] + EPSILON)

        dist = np.sqrt(f_norm_0 ** 2 + f_norm_1 ** 2)
        # Используем приближённую медиану (sqrt(2) для стандартного нормального)
        median_dist = np.sqrt(2) * CHI2_MEDIAN_FACTOR
        return (dist > median_dist).astype(np.int8)

    def _generate_moons_chunked(
        self, features: np.ndarray, stats: dict
    ) -> np.ndarray:
        """Moons с предвычисленными статистиками."""
        f = (
            features[:, :2]
            if features.shape[1] >= 2
            else np.column_stack([features[:, 0], features[:, 0]])
        )
        f_norm_0 = (f[:, 0] - stats["mean_0"]) / (stats["std_0"] + EPSILON)
        f_norm_1 = (f[:, 1] - stats["mean_1"]) / (stats["std_1"] + EPSILON)

        boundary = np.sin(f_norm_0 * np.pi)
        return (f_norm_1 > boundary).astype(np.int8)

    def _estimate_threshold_value(
        self,
        db: duckdb.DuckDBPyConnection,
        meta: Meta,
        informative_cols: list[str],
        generator: TargetGeneratorFunc,
        rng: np.random.Generator,
    ) -> float:
        """Оценивает пороговое значение для бинаризации на sample.

        Args:
            db: Соединение с DuckDB.
            meta: Текущая метаинформация.
            informative_cols: Список информативных колонок.
            generator: Функция генерации latent.
            rng: Генератор случайных чисел.

        Returns:
            Пороговое значение для бинаризации.
        """
        cols_sql = ", ".join(informative_cols)
        sample_size = min(self.chunk_size, meta.row_count)

        sample_data = db.execute(
            f"SELECT {cols_sql} FROM '{meta.file_path}' LIMIT {sample_size}"
        ).fetchnumpy()
        sample_features = np.column_stack(
            [sample_data[col] for col in informative_cols]
        )

        sample_latent = generator(sample_features, rng)
        return np.percentile(sample_latent, self.threshold * 100)

    def transform(self, db: duckdb.DuckDBPyConnection, meta: Meta) -> Meta:
        """Генерирует бинарный таргет.

        Использует чанкованную обработку для экономии памяти.

        Args:
            db: Соединение с DuckDB.
            meta: Текущая метаинформация.

        Returns:
            Обновлённая Meta с информацией о таргете.
        """
        informative_cols = meta.get_columns_by_tag("informative", "numeric")
        col_name = "target_bin"
        cols_sql = ", ".join(informative_cols)

        classification_methods = {"xor", "circles", "moons", "clusters"}
        regression_generators = self._get_regression_generators()

        # Валидация метода
        all_methods = list(classification_methods) + list(regression_generators.keys())
        if self.method not in all_methods:
            raise ValueError(
                f"Неизвестный метод: {self.method}. Доступны: {all_methods}"
            )

        # Предвычисляем статистики/пороги (используем отдельный rng для изоляции)
        stats = None
        threshold_value = None
        kmeans = None

        if self.method in {"xor", "circles", "moons"}:
            stats = self._compute_statistics(db, meta, informative_cols)
        elif self.method == "clusters":
            # Для clusters обучаем KMeans на sample
            estimation_rng = np.random.default_rng(self.seed)
            sample_size = min(self.chunk_size, meta.row_count)
            sample_data = db.execute(
                f"SELECT {cols_sql} FROM '{meta.file_path}' USING SAMPLE {sample_size}"
            ).fetchnumpy()
            sample_features = np.column_stack(
                [sample_data[col] for col in informative_cols]
            )
            random_state = int(estimation_rng.integers(0, 2**31))
            kmeans = MiniBatchKMeans(
                n_clusters=2, random_state=random_state, n_init=DEFAULT_KMEANS_N_INIT
            )
            kmeans.fit(sample_features)
        else:
            # Регрессионные методы — оцениваем порог
            estimation_rng = np.random.default_rng(self.seed)
            threshold_value = self._estimate_threshold_value(
                db, meta, informative_cols, regression_generators[self.method], estimation_rng
            )

        # Основной rng для генерации (независим от estimation)
        rng = np.random.default_rng(self.seed)

        with _chunked_target_writer(db, meta, col_name, "INT8") as write_chunk:
            for chunk_idx, (ids, features) in enumerate(
                _iter_chunks(db, meta, informative_cols, self.chunk_size)
            ):
                # Генерируем таргет в зависимости от метода
                if self.method == "xor":
                    target = self._generate_xor_chunked(features, stats)
                elif self.method == "circles":
                    target = self._generate_circles_chunked(features, stats)
                elif self.method == "moons":
                    target = self._generate_moons_chunked(features, stats)
                elif self.method == "clusters":
                    target = kmeans.predict(features).astype(np.int8)
                else:
                    # Регрессионные методы
                    latent = regression_generators[self.method](features, rng)
                    target = (latent > threshold_value).astype(np.int8)

                # Добавляем шум через flip
                if self.flip_ratio > 0:
                    flip_mask = rng.random(len(target)) < self.flip_ratio
                    target = np.where(flip_mask, 1 - target, target)

                write_chunk(chunk_idx, ids, target)

        meta.columns[col_name] = DType.INT8
        meta.column_tags[col_name] = ["target", "binary"]
        meta.completed_steps.append(self.name)
        return meta


class MulticlassTarget(TargetGeneratorMixin, Transformer):
    """Генерирует целевую переменную для многоклассовой классификации."""

    default_name: ClassVar[str] = "multiclass_target"
    requires: ClassVar[list[type[Transformer]]] = [Numeric]

    def __init__(
        self,
        n_classes: int = 5,
        method: Literal[
            "linear",
            "polynomial",
            "nonlinear",
            "friedman1",
            "friedman2",
            "friedman3",
            "exponential",
            "logarithmic",
            "step",
            "radial",
        ] = "linear",
        seed: int | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        name: str | None = None,
    ):
        """Инициализирует генератор многоклассового таргета.

        Args:
            n_classes: Количество классов.
            method: Метод генерации скрытой переменной:
                - "linear", "polynomial", "nonlinear"
                - "friedman1", "friedman2", "friedman3"
                - "exponential", "logarithmic", "step", "radial"
            seed: Seed для воспроизводимости.
            chunk_size: Размер чанка для обработки данных.
            name: Кастомное имя трансформера.
        """
        super().__init__(name)
        self.n_classes = n_classes
        self.method = method
        self.seed = seed
        self.chunk_size = chunk_size

    def _estimate_bins(
        self,
        db: duckdb.DuckDBPyConnection,
        meta: Meta,
        informative_cols: list[str],
        generator: TargetGeneratorFunc,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Оценивает границы бинов для квантильного биннинга на sample.

        Args:
            db: Соединение с DuckDB.
            meta: Текущая метаинформация.
            informative_cols: Список информативных колонок.
            generator: Функция генерации latent.
            rng: Генератор случайных чисел.

        Returns:
            Массив границ бинов.
        """
        cols_sql = ", ".join(informative_cols)
        sample_size = min(self.chunk_size, meta.row_count)

        sample_data = db.execute(
            f"SELECT {cols_sql} FROM '{meta.file_path}' LIMIT {sample_size}"
        ).fetchnumpy()
        sample_features = np.column_stack(
            [sample_data[col] for col in informative_cols]
        )

        sample_latent = generator(sample_features, rng)
        percentiles = np.linspace(0, 100, self.n_classes + 1)
        return np.percentile(sample_latent, percentiles)

    def transform(self, db: duckdb.DuckDBPyConnection, meta: Meta) -> Meta:
        """Генерирует многоклассовый таргет через квантильный биннинг.

        Использует чанкованную обработку для экономии памяти.

        Args:
            db: Соединение с DuckDB.
            meta: Текущая метаинформация.

        Returns:
            Обновлённая Meta с информацией о таргете.
        """
        informative_cols = meta.get_columns_by_tag("informative", "numeric")
        col_name = "target_multi"

        generators = self._get_regression_generators()
        if self.method not in generators:
            raise ValueError(
                f"Неизвестный метод: {self.method}. Доступны: {list(generators.keys())}"
            )

        generator = generators[self.method]

        # Оцениваем границы бинов на sample (отдельный rng для изоляции)
        estimation_rng = np.random.default_rng(self.seed)
        bins = self._estimate_bins(db, meta, informative_cols, generator, estimation_rng)

        # Основной rng для генерации (независим от estimation)
        rng = np.random.default_rng(self.seed)

        with _chunked_target_writer(db, meta, col_name, "INT8") as write_chunk:
            for chunk_idx, (ids, features) in enumerate(
                _iter_chunks(db, meta, informative_cols, self.chunk_size)
            ):
                # Генерируем latent и биннинг
                latent = generator(features, rng)
                target = np.digitize(latent, bins[1:-1]).astype(np.int8)

                write_chunk(chunk_idx, ids, target)

        meta.columns[col_name] = DType.INT8
        meta.column_tags[col_name] = ["target", "multiclass"]
        meta.completed_steps.append(self.name)
        return meta


class RankingTarget(Transformer):
    """Генерирует целевую переменную для ранжирования."""

    default_name: ClassVar[str] = "ranking_target"
    requires: ClassVar[list[type[Transformer]]] = [RegressionTarget, Category]

    def __init__(
        self,
        n_levels: int = 5,
        category_source: str | None = None,
        name: str | None = None,
    ):
        """Инициализирует генератор таргета ранжирования.

        Args:
            n_levels: Количество уровней релевантности (0 до n_levels-1).
            category_source: Имя категориальной колонки для query_id. Если None,
                            берётся первая категориальная колонка.
            name: Кастомное имя трансформера.
        """
        super().__init__(name)
        self.n_levels = n_levels
        self.category_source = category_source

    def transform(self, db: duckdb.DuckDBPyConnection, meta: Meta) -> Meta:
        """Генерирует таргет ранжирования и query_id.

        query_id берётся из категориальной колонки.
        target_rank формируется через ранжирование target_reg внутри группы
        и последующий биннинг на n_levels уровней.

        Args:
            db: Соединение с DuckDB.
            meta: Текущая метаинформация.

        Returns:
            Обновлённая Meta с информацией о таргете и query_id.
        """
        # Находим категориальную колонку для query_id
        if self.category_source:
            if self.category_source not in meta.columns:
                raise ValueError(
                    f"Указанная категориальная колонка '{self.category_source}' "
                    f"не найдена. Доступные колонки: {list(meta.columns.keys())}"
                )
            query_col = self.category_source
        else:
            # Category гарантирован через requires, берём первую категориальную колонку
            category_cols = [
                col for col, tags in meta.column_tags.items() if "category" in tags
            ]
            query_col = category_cols[0]

        # Генерируем target_rank через SQL:
        # 1. Ранжируем target_reg внутри каждой группы (query_id)
        # 2. Нормализуем ранг в [0, 1]
        # 3. Бинним на n_levels уровней
        _atomic_write(
            db,
            f"""
            SELECT
                m.*,
                CAST({query_col} AS INT64) AS query_id,
                CAST(
                    FLOOR(
                        PERCENT_RANK() OVER (
                            PARTITION BY {query_col}
                            ORDER BY target_reg
                        ) * {self.n_levels - PERCENT_RANK_EPSILON}
                    ) AS INT8
                ) AS target_rank
            FROM '{meta.file_path}' AS m
            """,
            meta.file_path,
        )

        # Обновляем meta
        meta.columns["query_id"] = DType.INT64
        meta.column_tags["query_id"] = ["ranking"]

        meta.columns["target_rank"] = DType.INT8
        meta.column_tags["target_rank"] = ["target", "ranking"]

        meta.completed_steps.append(self.name)
        return meta


class Datetime(Transformer):
    """Генерирует колонку timestamp на основе таргета и фичи."""

    default_name: ClassVar[str] = "datetime"
    requires: ClassVar[list[type[Transformer]]] = [RegressionTarget]

    def __init__(
        self,
        start_date: str = "2020-01-01",
        end_date: str = "2024-01-01",
        target_weight: float = 0.7,
        feature_weight: float = 0.3,
        noise_seconds: int = 0,
        seed: int | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        name: str | None = None,
    ):
        """Инициализирует генератор временных меток.

        Args:
            start_date: Начальная дата диапазона (YYYY-MM-DD).
            end_date: Конечная дата диапазона (YYYY-MM-DD).
            target_weight: Вес таргета в линейной комбинации.
            feature_weight: Вес фичи в линейной комбинации.
            noise_seconds: Максимальное отклонение в секундах (±noise_seconds).
            seed: Seed для воспроизводимости.
            chunk_size: Размер чанка для обработки данных.
            name: Кастомное имя трансформера.
        """
        super().__init__(name)
        self.start_date = start_date
        self.end_date = end_date
        self.target_weight = target_weight
        self.feature_weight = feature_weight
        self.noise_seconds = noise_seconds
        self.seed = seed
        self.chunk_size = chunk_size

    def _compute_global_stats(
        self,
        db: duckdb.DuckDBPyConnection,
        meta: Meta,
        feature_col: str,
    ) -> dict[str, float]:
        """Вычисляет глобальные min/max для нормализации.

        Args:
            db: Соединение с DuckDB.
            meta: Текущая метаинформация.
            feature_col: Имя колонки фичи.

        Returns:
            Словарь с min/max значениями для target и feature.
        """
        stats = db.execute(
            f"""
            SELECT
                MIN(target_reg) as target_min,
                MAX(target_reg) as target_max,
                MIN({feature_col}) as feature_min,
                MAX({feature_col}) as feature_max
            FROM '{meta.file_path}'
            """
        ).fetchone()

        return {
            "target_min": stats[0],
            "target_max": stats[1],
            "feature_min": stats[2],
            "feature_max": stats[3],
        }

    def transform(self, db: duckdb.DuckDBPyConnection, meta: Meta) -> Meta:
        """Генерирует timestamp через линейную комбинацию таргета и фичи.

        Формула: normalized = w1 * norm(target_reg) + w2 * norm(feature)
        Затем масштабирование в диапазон дат + опциональный шум.

        Использует чанкованную обработку для экономии памяти.

        Args:
            db: Соединение с DuckDB.
            meta: Текущая метаинформация.

        Returns:
            Обновлённая Meta с колонкой timestamp.
        """
        # Берём первую информативную фичу
        informative_cols = meta.get_columns_by_tag("informative", "numeric")
        feature_col = informative_cols[0]

        # Вычисляем глобальные min/max для корректной нормализации
        stats = self._compute_global_stats(db, meta, feature_col)
        target_range = stats["target_max"] - stats["target_min"] + EPSILON
        feature_range = stats["feature_max"] - stats["feature_min"] + EPSILON

        # Временные границы
        start_ts = dt.strptime(self.start_date, "%Y-%m-%d").timestamp()
        end_ts = dt.strptime(self.end_date, "%Y-%m-%d").timestamp()
        ts_range = end_ts - start_ts

        # RNG для шума
        rng = np.random.default_rng(self.seed) if self.noise_seconds > 0 else None

        with _chunked_target_writer(db, meta, "timestamp", "DOUBLE") as write_chunk:
            for chunk_idx, (ids, features) in enumerate(
                _iter_chunks(db, meta, ["target_reg", feature_col], self.chunk_size)
            ):
                target = features[:, 0]
                feature = features[:, 1]

                # Нормализуем в [0, 1] используя глобальные min/max
                target_norm = (target - stats["target_min"]) / target_range
                feature_norm = (feature - stats["feature_min"]) / feature_range

                # Линейная комбинация
                combined = (
                    self.target_weight * target_norm
                    + self.feature_weight * feature_norm
                )
                # Нормализуем комбинацию (может выходить за [0,1] из-за весов)
                combined_min, combined_max = combined.min(), combined.max()
                combined = (combined - combined_min) / (combined_max - combined_min + EPSILON)

                # Преобразуем в timestamps
                timestamps = start_ts + combined * ts_range

                # Добавляем шум
                if rng is not None:
                    noise = rng.integers(
                        -self.noise_seconds, self.noise_seconds + 1, size=len(timestamps)
                    )
                    timestamps = timestamps + noise

                write_chunk(chunk_idx, ids, timestamps)

        # Конвертируем DOUBLE в TIMESTAMP через SQL
        _atomic_write(
            db,
            f"""
            SELECT
                * EXCLUDE (timestamp),
                TO_TIMESTAMP(timestamp) AS timestamp
            FROM '{meta.file_path}'
            """,
            meta.file_path,
        )

        # Обновляем meta
        meta.columns["timestamp"] = DType.TIMESTAMP
        meta.column_tags["timestamp"] = ["datetime"]
        meta.completed_steps.append(self.name)
        return meta


class Date(Transformer):
    """Извлекает date из timestamp."""

    default_name: ClassVar[str] = "date"
    requires: ClassVar[list[type[Transformer]]] = [Datetime]

    def transform(self, db: duckdb.DuckDBPyConnection, meta: Meta) -> Meta:
        """Создаёт колонку date из timestamp.

        Args:
            db: Соединение с DuckDB.
            meta: Текущая метаинформация.

        Returns:
            Обновлённая Meta с колонкой date.
        """
        _atomic_write(
            db,
            f"""
            SELECT m.*, CAST(timestamp AS DATE) AS date
            FROM '{meta.file_path}' AS m
            """,
            meta.file_path,
        )

        meta.columns["date"] = DType.DATE
        meta.column_tags["date"] = ["date"]
        meta.completed_steps.append(self.name)
        return meta


class Boolean(Transformer):
    """Генерирует boolean колонку через бинаризацию фичи."""

    default_name: ClassVar[str] = "boolean"
    requires: ClassVar[list[type[Transformer]]] = [Numeric]

    def __init__(
        self,
        threshold: float = 0.5,
        flip_ratio: float = 0.0,
        source_column: str | int | None = None,
        seed: int | None = None,
        name: str | None = None,
    ):
        """Инициализирует генератор boolean.

        Args:
            threshold: Порог для бинаризации (квантиль от 0 до 1).
            flip_ratio: Доля значений для случайного переключения.
            source_column: Колонка-источник для бинаризации:
                - str: имя конкретной колонки
                - int: индекс числовой колонки (0-based, с циклическим переходом)
                - None: использует первую числовую колонку
            seed: Seed для воспроизводимости.
            name: Кастомное имя трансформера.
        """
        super().__init__(name)
        self.threshold = threshold
        self.flip_ratio = flip_ratio
        self.source_column = source_column
        self.seed = seed

    def _resolve_source_column(self, meta: Meta) -> str:
        """Определяет колонку-источник для бинаризации.

        Args:
            meta: Текущая метаинформация.

        Returns:
            Имя колонки-источника.

        Raises:
            ValueError: Если указанная колонка не найдена.
        """
        numeric_cols = [
            col for col, tags in meta.column_tags.items() if "numeric" in tags
        ]

        if isinstance(self.source_column, str):
            if self.source_column not in numeric_cols:
                raise ValueError(
                    f"Колонка '{self.source_column}' не найдена среди числовых. "
                    f"Доступны: {numeric_cols}"
                )
            return self.source_column

        if isinstance(self.source_column, int):
            # Циклический выбор по индексу
            idx = self.source_column % len(numeric_cols)
            return numeric_cols[idx]

        # None — первая колонка
        return numeric_cols[0]

    def transform(self, db: duckdb.DuckDBPyConnection, meta: Meta) -> Meta:
        """Генерирует boolean колонку.

        Бинаризует числовую фичу по квантильному порогу,
        затем случайно переключает часть значений.

        Args:
            db: Соединение с DuckDB.
            meta: Текущая метаинформация.

        Returns:
            Обновлённая Meta с boolean колонкой.
        """
        source_col = self._resolve_source_column(meta)
        col_name = f"flag_{self.name}"

        # Читаем данные
        data = db.execute(
            f"SELECT id, {source_col} FROM '{meta.file_path}'"
        ).fetchnumpy()

        values = data[source_col]
        ids = data["id"]

        # Бинаризация по квантильному порогу
        threshold_value = np.percentile(values, self.threshold * 100)
        flags = (values > threshold_value).astype(np.int8)

        # Добавляем шум через flip
        if self.flip_ratio > 0:
            rng = np.random.default_rng(self.seed)
            flip_mask = rng.random(len(flags)) < self.flip_ratio
            flags = np.where(flip_mask, 1 - flags, flags)

        # Сохраняем
        db.register("flags_np", {"id": ids, "flag": flags})

        _atomic_write(
            db,
            f"""
            SELECT m.*, CAST(f.flag AS BOOLEAN) AS {col_name}
            FROM '{meta.file_path}' AS m
            JOIN flags_np AS f ON m.id = f.id
            """,
            meta.file_path,
        )
        db.unregister("flags_np")

        meta.columns[col_name] = DType.BOOLEAN
        meta.column_tags[col_name] = ["boolean"]
        meta.completed_steps.append(self.name)
        return meta


class Nullable(Transformer):
    """Добавляет NULL в существующие колонки."""

    default_name: ClassVar[str] = "nullable"
    requires: ClassVar[list[type[Transformer]]] = [Numeric]

    def __init__(
        self,
        tags: list[str] | None = None,
        columns: list[str] | None = None,
        ratio: float = 0.1,
        seed: int | None = None,
        name: str | None = None,
    ):
        """Инициализирует трансформер для добавления NULL.

        Args:
            tags: Теги колонок для обработки (например, ["numeric"]).
            columns: Конкретные колонки для обработки. Если указано, tags игнорируется.
            ratio: Доля NULL значений (от 0 до 1).
            seed: Seed для воспроизводимости.
            name: Кастомное имя трансформера.
        """
        super().__init__(name)
        self.tags = tags or ["numeric"]
        self.columns = columns
        self.ratio = ratio
        self.seed = seed

    def transform(self, db: duckdb.DuckDBPyConnection, meta: Meta) -> Meta:
        """Добавляет NULL в выбранные колонки.

        Args:
            db: Соединение с DuckDB.
            meta: Текущая метаинформация.

        Returns:
            Обновлённая Meta (колонки помечаются тегом nullable).
        """
        # Определяем колонки для обработки
        if self.columns:
            target_cols = self.columns
        else:
            target_cols = [
                col
                for col, col_tags in meta.column_tags.items()
                if any(tag in col_tags for tag in self.tags)
            ]

        if not target_cols:
            meta.completed_steps.append(self.name)
            return meta

        # Генерируем маску NULL для каждой колонки
        rng = np.random.default_rng(self.seed)

        # Формируем CASE выражения для каждой колонки
        case_expressions = []
        null_arrays = {}

        for i, col in enumerate(target_cols):
            null_mask = rng.random(meta.row_count) < self.ratio
            null_arrays[f"null_{i}"] = null_mask.astype(np.int8)

        # Создаём словарь с id и масками
        ids = np.arange(meta.row_count, dtype=np.int64)
        mask_data = {"id": ids}
        for i in range(len(target_cols)):
            mask_data[f"mask_{i}"] = null_arrays[f"null_{i}"]
        db.register("null_masks", mask_data)

        # Формируем SQL с CASE для каждой колонки
        select_parts = []
        for col in meta.columns:
            if col in target_cols:
                idx = target_cols.index(col)
                select_parts.append(
                    f"CASE WHEN n.mask_{idx} = 1 THEN NULL ELSE m.{col} END AS {col}"
                )
            else:
                select_parts.append(f"m.{col}")

        select_sql = ", ".join(select_parts)

        _atomic_write(
            db,
            f"""
            SELECT {select_sql}
            FROM '{meta.file_path}' AS m
            JOIN null_masks AS n ON m.id = n.id
            """,
            meta.file_path,
        )
        db.unregister("null_masks")

        # Обновляем meta — добавляем тег nullable
        for col in target_cols:
            if col in meta.column_tags:
                meta.column_tags[col].append("nullable")

        meta.completed_steps.append(self.name)
        return meta


class Pipeline:
    """Выполняет последовательность трансформеров для генерации данных.

    Pipeline управляет выполнением шагов генерации, поддерживает
    сохранение прогресса и возобновление с точки остановки.

    Attributes:
        steps: Список трансформеров для выполнения.
        config: Конфигурация генератора.
        progress_callback: Опциональный callback для отслеживания прогресса.
    """

    def __init__(
        self,
        steps: list[Transformer],
        config: GeneratorConfig,
        progress_callback: ProgressCallback | None = None,
    ):
        """Инициализирует pipeline.

        Args:
            steps: Список трансформеров для выполнения.
            config: Конфигурация генератора.
            progress_callback: Опциональный callback для отслеживания прогресса.
                Сигнатура: (step_name, current_step, total_steps, chunk_info)
                Пример использования с tqdm:
                    pbar = tqdm(total=len(steps))
                    def callback(name, curr, total, chunk):
                        pbar.set_description(f"{name} {chunk or ''}")
                        if chunk is None:  # Конец шага
                            pbar.update(1)
                    pipeline = Pipeline(steps, config, progress_callback=callback)
        """
        self.steps = steps
        self.config = config
        self.progress_callback = progress_callback

    def _meta_pickle_path(self) -> str:
        """Возвращает путь к файлу meta.pkl.

        Returns:
            Путь к файлу meta.pkl.
        """
        return f"{self.config.output_path}/meta.pkl"

    def _save_meta(self, meta: Meta) -> None:
        """Сохраняет meta в pickle.

        Args:
            meta: Объект Meta для сохранения.
        """
        with open(self._meta_pickle_path(), "wb") as f:
            pickle.dump(meta, f)

    def _load_meta(self) -> Meta | None:
        """Загружает meta из pickle если существует.

        Returns:
            Объект Meta если файл существует, иначе None.
        """
        path = self._meta_pickle_path()
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
        return None

    def run(self) -> Meta:
        """Запускает все шаги последовательно с поддержкой возобновления.

        Если файл meta.pkl существует, пропускает уже выполненные шаги.
        После каждого шага сохраняет прогресс в meta.pkl.

        Returns:
            Итоговая Meta после выполнения всех шагов.
        """
        logger.info("Запуск pipeline с %d шагами", len(self.steps))

        # Пытаемся загрузить существующий прогресс
        meta = self._load_meta()
        if meta is not None:
            logger.info(
                "Загружен прогресс: выполнено %d шагов", len(meta.completed_steps)
            )
        else:
            meta = Meta(file_path=f"{self.config.output_path}/main.parquet")

        db = duckdb.connect()
        try:
            for i, step in enumerate(self.steps, 1):
                # Пропускаем уже выполненные шаги
                if step.name in meta.completed_steps:
                    logger.info(
                        "[%d/%d] Пропуск (уже выполнен): %s",
                        i,
                        len(self.steps),
                        step.name,
                    )
                    continue

                logger.info("[%d/%d] Выполняется: %s", i, len(self.steps), step.name)

                # Уведомляем о начале шага
                if self.progress_callback:
                    self.progress_callback(step.name, i, len(self.steps), "starting")

                meta = step.transform(db, meta)
                logger.debug("Завершён шаг %s, строк: %d", step.name, meta.row_count)

                # Уведомляем о завершении шага
                if self.progress_callback:
                    self.progress_callback(step.name, i, len(self.steps), None)

                # Сохраняем прогресс после каждого шага
                self._save_meta(meta)

            logger.info("Pipeline завершён успешно, строк: %d", meta.row_count)
            return meta
        finally:
            db.close()


class PipelineFactory:
    """Фабрика для создания Pipeline из конфигурации.

    Автоматически определяет необходимые шаги на основе GeneratorConfig,
    выполняет калибровку размера и топологическую сортировку зависимостей.
    """

    def _validate_config(self, config: GeneratorConfig) -> None:
        """Ранняя валидация конфигурации на совместимость.

        Проверяет, что все требуемые зависимости могут быть удовлетворены
        на основе заданной конфигурации.

        Args:
            config: Конфигурация генератора.

        Raises:
            ValueError: Если конфигурация несовместима.
        """
        # with_date требует with_datetime
        if config.with_date and not config.with_datetime:
            raise ValueError(
                "with_date=True требует with_datetime=True. "
                "Колонка date извлекается из timestamp."
            )

        # Datetime требует таргет — проверяем что task задана или будет создан
        # implicit RegressionTarget
        if config.with_datetime and config.task is None:
            logger.warning(
                "with_datetime=True при task=None: будет создан вспомогательный "
                "RegressionTarget для генерации timestamp."
            )

        # Ranking требует category (явно или создастся неявно)
        if config.task == "ranking" and config.n_categories == 0:
            logger.info(
                "task='ranking' без категорий: будет создана вспомогательная "
                "категория для query_id."
            )

    def _parse_size(self, size_str: str) -> int:
        """Парсит строку размера в байты.

        Args:
            size_str: Строка вида "100MB", "1GB", "500KB".

        Returns:
            Размер в байтах.
        """
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

    def _estimate_row_count(self, config: GeneratorConfig) -> int:
        """Оценивает количество строк для достижения target_size.

        Использует калибровку: генерирует небольшую выборку,
        измеряет размер parquet-файла, экстраполирует.

        Args:
            config: Конфигурация генератора.

        Returns:
            Оценочное количество строк.
        """
        target_bytes = self._parse_size(config.target_size)
        logger.info("Калибровка: целевой размер %s (%d байт)", config.target_size, target_bytes)

        # Калибровочная выборка
        calibration_rows = DEFAULT_CALIBRATION_ROWS

        with tempfile.TemporaryDirectory() as tmpdir:
            calibration_path = f"{tmpdir}/calibration.parquet"

            # Создаём калибровочный pipeline
            calibration_config = GeneratorConfig(
                target_size="1MB",  # не используется
                output_path=tmpdir,
                seed=config.seed,
                n_numeric=config.n_numeric,
                informative_ratio=config.informative_ratio,
                n_categories=config.n_categories,
                category_cardinality=config.category_cardinality,
                category_method=config.category_method,
                task=config.task,
                target_method=config.target_method,
                target_noise=config.target_noise,
                n_classes=config.n_classes,
                with_datetime=config.with_datetime,
                datetime_start=config.datetime_start,
                datetime_end=config.datetime_end,
                with_date=config.with_date,
                n_booleans=config.n_booleans,
                nullable_ratio=config.nullable_ratio,
            )

            # Собираем шаги с фиксированным row_count
            steps = self._build_steps(calibration_config)

            # Заменяем Init на версию с calibration_rows
            for i, step in enumerate(steps):
                if isinstance(step, Init):
                    steps[i] = Init(row_count=calibration_rows)
                    break

            sorted_steps = self._topological_sort(steps)

            # Запускаем калибровку
            with duckdb.connect() as db:
                meta = Meta(file_path=calibration_path)

                for step in sorted_steps:
                    meta = step.transform(db, meta)

            # Измеряем размер
            calibration_size = os.path.getsize(calibration_path)
            bytes_per_row = calibration_size / calibration_rows

            # Экстраполируем
            estimated_rows = int(target_bytes / bytes_per_row)
            result_rows = max(MIN_ROW_COUNT, estimated_rows)

            logger.info(
                "Калибровка завершена: %.2f байт/строка, оценка %d строк",
                bytes_per_row,
                result_rows,
            )

            return result_rows

    def _has_step_of_type(
        self, steps: list[Transformer], step_type: type[Transformer]
    ) -> bool:
        """Проверяет наличие шага заданного типа.

        Args:
            steps: Список трансформеров.
            step_type: Тип трансформера для поиска.

        Returns:
            True если шаг такого типа уже есть в списке.
        """
        return any(isinstance(s, step_type) for s in steps)

    def _add_base_steps(
        self, config: GeneratorConfig, steps: list[Transformer]
    ) -> None:
        """Добавляет базовые шаги: Init и Numeric.

        Args:
            config: Конфигурация генератора.
            steps: Список шагов для модификации.
        """
        steps.append(Init(row_count=0))
        steps.append(
            Numeric(
                n_features=config.n_numeric,
                informative_ratio=config.informative_ratio,
                seed=config.seed,
            )
        )

    def _add_category_steps(
        self, config: GeneratorConfig, steps: list[Transformer]
    ) -> None:
        """Добавляет шаги категорий.

        Args:
            config: Конфигурация генератора.
            steps: Список шагов для модификации.
        """
        for i in range(config.n_categories):
            steps.append(
                Category(
                    cardinality=config.category_cardinality,
                    method=config.category_method,
                    seed=config.seed,
                    name=f"category_{i}",
                )
            )

    def _add_target_steps(
        self, config: GeneratorConfig, steps: list[Transformer]
    ) -> None:
        """Добавляет шаги таргетов в зависимости от задачи.

        Args:
            config: Конфигурация генератора.
            steps: Список шагов для модификации.
        """
        if config.task == "regression":
            steps.append(
                RegressionTarget(
                    method=config.target_method,
                    noise=config.target_noise,
                    seed=config.seed,
                )
            )
        elif config.task == "binary":
            steps.append(
                BinaryTarget(
                    method=config.target_method,
                    seed=config.seed,
                )
            )
        elif config.task == "multiclass":
            steps.append(
                MulticlassTarget(
                    n_classes=config.n_classes,
                    method=config.target_method,
                    seed=config.seed,
                )
            )
        elif config.task == "ranking":
            # Ranking требует regression target + category
            steps.append(
                RegressionTarget(
                    method=config.target_method,
                    noise=config.target_noise,
                    seed=config.seed,
                )
            )
            if not config.n_categories:
                steps.append(Category(seed=config.seed, name="category_for_ranking"))
            steps.append(RankingTarget())

    def _add_datetime_steps(
        self, config: GeneratorConfig, steps: list[Transformer]
    ) -> None:
        """Добавляет шаги datetime.

        Args:
            config: Конфигурация генератора.
            steps: Список шагов для модификации.
        """
        if not config.with_datetime:
            return

        # Datetime требует RegressionTarget — добавляем если ещё нет
        if not self._has_step_of_type(steps, RegressionTarget):
            steps.append(
                RegressionTarget(
                    method=config.target_method,
                    noise=config.target_noise,
                    seed=config.seed,
                )
            )

        steps.append(
            Datetime(
                start_date=config.datetime_start,
                end_date=config.datetime_end,
                seed=config.seed,
            )
        )

        if config.with_date:
            steps.append(Date())

    def _add_boolean_steps(
        self, config: GeneratorConfig, steps: list[Transformer]
    ) -> None:
        """Добавляет boolean шаги.

        Args:
            config: Конфигурация генератора.
            steps: Список шагов для модификации.
        """
        for i in range(config.n_booleans):
            steps.append(
                Boolean(seed=config.seed, name=f"boolean_{i}", source_column=i)
            )

    def _add_nullable_step(
        self, config: GeneratorConfig, steps: list[Transformer]
    ) -> None:
        """Добавляет Nullable шаг.

        Args:
            config: Конфигурация генератора.
            steps: Список шагов для модификации.
        """
        if config.nullable_ratio > 0:
            steps.append(Nullable(ratio=config.nullable_ratio, seed=config.seed))

    def _build_steps(self, config: GeneratorConfig) -> list[Transformer]:
        """Определяет какие шаги нужны на основе конфига.

        Args:
            config: Конфигурация генератора.

        Returns:
            Список трансформеров (ещё не отсортированный).
        """
        steps: list[Transformer] = []

        self._add_base_steps(config, steps)
        self._add_category_steps(config, steps)
        self._add_target_steps(config, steps)
        self._add_datetime_steps(config, steps)
        self._add_boolean_steps(config, steps)
        self._add_nullable_step(config, steps)

        return steps

    def _topological_sort(self, steps: list[Transformer]) -> list[Transformer]:
        """Топологическая сортировка шагов по зависимостям.

        Поддерживает множественные экземпляры одного класса (например, несколько Category).
        Зависимость считается выполненной, когда выполнен хотя бы один экземпляр
        требуемого класса.

        Args:
            steps: Список трансформеров.

        Returns:
            Отсортированный список трансформеров.
        """
        # Строим множество доступных классов
        available_classes: set[type[Transformer]] = {type(step) for step in steps}

        # in_degree[step] = количество невыполненных зависимостей
        in_degree: dict[Transformer, int] = {}

        for step in steps:
            count = 0
            for req_class in type(step).requires:
                if req_class in available_classes:
                    count += 1
            in_degree[step] = count

        # dependents[class] = список шагов, зависящих от этого класса
        dependents: dict[type[Transformer], list[Transformer]] = {}
        for step in steps:
            for req_class in type(step).requires:
                if req_class not in dependents:
                    dependents[req_class] = []
                dependents[req_class].append(step)

        # Отслеживаем какие классы уже выполнены
        executed_classes: set[type[Transformer]] = set()

        # Алгоритм Кана
        result: list[Transformer] = []
        queue: list[Transformer] = [s for s in steps if in_degree[s] == 0]

        while queue:
            current = queue.pop(0)
            result.append(current)

            current_class = type(current)

            # Если это первый экземпляр данного класса — уменьшаем in_degree зависимых
            if current_class not in executed_classes:
                executed_classes.add(current_class)

                if current_class in dependents:
                    for dependent in dependents[current_class]:
                        in_degree[dependent] -= 1
                        if in_degree[dependent] == 0:
                            queue.append(dependent)

        if len(result) != len(steps):
            raise ValueError("Обнаружен цикл в зависимостях трансформеров")

        return result

    def create(
        self,
        config: GeneratorConfig,
        progress_callback: ProgressCallback | None = None,
    ) -> Pipeline:
        """Собирает pipeline на основе конфига.

        Args:
            config: Конфигурация генератора.
            progress_callback: Опциональный callback для отслеживания прогресса.

        Returns:
            Готовый Pipeline с упорядоченными шагами.
        """
        # Ранняя валидация конфигурации
        self._validate_config(config)

        # Калибруем row_count
        row_count = self._estimate_row_count(config)

        # Собираем шаги
        steps = self._build_steps(config)

        # Устанавливаем row_count в Init
        for step in steps:
            if isinstance(step, Init):
                step.row_count = row_count
                break

        sorted_steps = self._topological_sort(steps)
        return Pipeline(sorted_steps, config, progress_callback)
