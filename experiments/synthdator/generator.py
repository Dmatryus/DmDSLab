"""Генератор синтетических данных для ML."""

from __future__ import annotations

import logging
import os
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime as dt
from typing import ClassVar, Literal

import duckdb
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.datasets import make_classification

# Логгер модуля
logger = logging.getLogger(__name__)

# Константы генерации
DEFAULT_CHUNK_SIZE = 100_000
DEFAULT_KMEANS_SAMPLES = 100_000
DEFAULT_CALIBRATION_ROWS = 1_000
DEFAULT_STEP_BINS = 5
MIN_ROW_COUNT = 100

# Математические константы
EPSILON = 1e-10  # Защита от деления на ноль

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

# Параметры генерации таргетов
MAX_INTERACTION_PAIRS = 3  # Макс. пар взаимодействий в polynomial
MAX_SIN_FEATURES = 3  # Макс. фич для sin в nonlinear
MIN_COS_FEATURES = 3  # Мин. индекс фич для cos в nonlinear
MAX_COS_FEATURES = 6  # Макс. фич для cos в nonlinear


def _atomic_write(db: duckdb.DuckDBPyConnection, query: str, file_path: str) -> None:
    """Атомарная запись в parquet через временный файл.

    Args:
        db: Соединение с DuckDB.
        query: SQL запрос (SELECT).
        file_path: Путь к файлу.
    """
    tmp_path = f"{file_path}.tmp.parquet"
    db.execute(f"COPY ({query}) TO '{tmp_path}' (FORMAT PARQUET)")
    os.replace(tmp_path, file_path)


@dataclass
class GeneratorConfig:
    """Конфигурация генератора."""

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


@dataclass
class Meta:
    """Метаинформация о состоянии данных."""

    file_path: str
    row_count: int = 0
    columns: dict[str, str] = field(default_factory=dict)
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
        pass


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
        meta.columns["id"] = "INT64"
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
            meta.columns[col_name] = "DOUBLE"
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
            n_init=10,
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
        meta.columns[col_name] = "VARCHAR"
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
        self, features: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Friedman #1: классический бенчмарк для нелинейной регрессии.

        Формула: 10*sin(π*x0*x1) + 20*(x2-0.5)² + 10*x3 + 5*x4
        Использует только первые 5 фич, остальные — шум.

        Args:
            features: Матрица фич (n_samples, n_features).
            rng: Генератор случайных чисел.

        Returns:
            Вектор таргета.
        """
        n_features = features.shape[1]

        # Нормализуем фичи в [0, 1] для соответствия оригинальной формуле
        f_norm = (features - features.min(axis=0)) / (
            features.max(axis=0) - features.min(axis=0) + EPSILON
        )

        y = np.zeros(features.shape[0])

        if n_features >= 2:
            y += 10 * np.sin(np.pi * f_norm[:, 0] * f_norm[:, 1])
        if n_features >= 3:
            y += 20 * (f_norm[:, 2] - 0.5) ** 2
        if n_features >= 4:
            y += 10 * f_norm[:, 3]
        if n_features >= 5:
            y += 5 * f_norm[:, 4]

        return y

    def _generate_friedman2(
        self, features: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Friedman #2: sqrt(x0² + (x1*x2 - 1/(x1*x3))²).

        Args:
            features: Матрица фич (n_samples, n_features).
            rng: Генератор случайных чисел.

        Returns:
            Вектор таргета.
        """
        n_features = features.shape[1]
        if n_features < 4:
            return self._generate_linear(features, rng)

        # Масштабируем как в оригинале: x0 in [0,100], x1 in [40π, 560π], x2 in [0,1], x3 in [1,11]
        f = features.copy()
        f_min, f_max = f.min(axis=0), f.max(axis=0)
        f_norm = (f - f_min) / (f_max - f_min + EPSILON)

        x0 = f_norm[:, 0] * FRIEDMAN_X0_MAX
        x1 = FRIEDMAN_X1_START + f_norm[:, 1] * FRIEDMAN_X1_SCALE
        x2 = f_norm[:, 2]
        x3 = FRIEDMAN_X3_MIN + f_norm[:, 3] * FRIEDMAN_X3_SCALE

        # Защита от деления на ноль
        denom = x1 * x3 + EPSILON
        inner = x1 * x2 - 1 / denom

        return np.sqrt(x0**2 + inner**2)

    def _generate_friedman3(
        self, features: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Friedman #3: atan((x1*x2 - 1/(x1*x3)) / x0).

        Args:
            features: Матрица фич (n_samples, n_features).
            rng: Генератор случайных чисел.

        Returns:
            Вектор таргета.
        """
        n_features = features.shape[1]
        if n_features < 4:
            return self._generate_linear(features, rng)

        f = features.copy()
        f_min, f_max = f.min(axis=0), f.max(axis=0)
        f_norm = (f - f_min) / (f_max - f_min + EPSILON)

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
            name: Кастомное имя трансформера.
        """
        super().__init__(name)
        self.method = method
        self.noise = noise
        self.seed = seed

    def transform(self, db: duckdb.DuckDBPyConnection, meta: Meta) -> Meta:
        """Генерирует таргет регрессии на основе информативных фич.

        Args:
            db: Соединение с DuckDB.
            meta: Текущая метаинформация.

        Returns:
            Обновлённая Meta с информацией о таргете.
        """
        informative_cols = meta.get_columns_by_tag("informative", "numeric")

        col_name = "target_reg"

        # Читаем информативные фичи
        cols_sql = ", ".join(informative_cols)
        data = db.execute(f"SELECT id, {cols_sql} FROM '{meta.file_path}'").fetchnumpy()

        ids = data["id"]
        features = np.column_stack([data[col] for col in informative_cols])

        # Генерируем таргет в зависимости от метода
        rng = np.random.default_rng(self.seed)
        generators = self._get_regression_generators()

        if self.method not in generators:
            raise ValueError(
                f"Неизвестный метод: {self.method}. Доступны: {list(generators.keys())}"
            )

        target = generators[self.method](features, rng)

        # Добавляем шум пропорционально std таргета
        if self.noise > 0:
            noise_scale = self.noise * np.std(target)
            target = target + rng.normal(0, noise_scale, size=len(target))

        # Сохраняем
        db.register("target_np", {"id": ids, "target": target})
        _atomic_write(
            db,
            f"""
                SELECT m.*, CAST(t.target AS DOUBLE) AS {col_name}
                FROM '{meta.file_path}' AS m
                JOIN target_np AS t ON m.id = t.id
            """,
            meta.file_path,
        )
        db.unregister("target_np")

        meta.columns[col_name] = "DOUBLE"
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
            name: Кастомное имя трансформера.
        """
        super().__init__(name)
        self.method = method
        self.threshold = threshold
        self.flip_ratio = flip_ratio
        self.seed = seed

    def _generate_xor(
        self, features: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """XOR от первых двух фич — линейно неразделимая задача.

        Args:
            features: Матрица фич.
            rng: Генератор случайных чисел.

        Returns:
            Бинарные метки.
        """
        # Бинаризуем первые 2 фичи по медиане
        x0 = features[:, 0] > np.median(features[:, 0])
        x1 = features[:, 1] > np.median(features[:, 1]) if features.shape[1] > 1 else x0

        return (x0 ^ x1).astype(np.int8)

    def _generate_circles(
        self, features: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Концентрические окружности — внутренний и внешний круг.

        Args:
            features: Матрица фич.
            rng: Генератор случайных чисел.

        Returns:
            Бинарные метки.
        """
        # Нормализуем первые 2 фичи
        f = (
            features[:, :2]
            if features.shape[1] >= 2
            else np.column_stack([features[:, 0], features[:, 0]])
        )
        f_norm = (f - f.mean(axis=0)) / (f.std(axis=0) + EPSILON)

        # Расстояние от центра
        dist = np.sqrt(f_norm[:, 0] ** 2 + f_norm[:, 1] ** 2)

        # Медианное расстояние как граница
        return (dist > np.median(dist)).astype(np.int8)

    def _generate_moons(
        self, features: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Два полумесяца — классическая нелинейная задача.

        Args:
            features: Матрица фич.
            rng: Генератор случайных чисел.

        Returns:
            Бинарные метки.
        """
        f = (
            features[:, :2]
            if features.shape[1] >= 2
            else np.column_stack([features[:, 0], features[:, 0]])
        )
        f_norm = (f - f.mean(axis=0)) / (f.std(axis=0) + EPSILON)

        # Полумесяцы: y > sin(x) для одного класса
        boundary = np.sin(f_norm[:, 0] * np.pi)
        return (f_norm[:, 1] > boundary).astype(np.int8)

    def _generate_clusters(
        self, features: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """KMeans кластеризация на 2 кластера.

        Args:
            features: Матрица фич.
            rng: Генератор случайных чисел.

        Returns:
            Бинарные метки.
        """
        kmeans = MiniBatchKMeans(n_clusters=2, random_state=self.seed, n_init="auto")
        return kmeans.fit_predict(features).astype(np.int8)

    def transform(self, db: duckdb.DuckDBPyConnection, meta: Meta) -> Meta:
        """Генерирует бинарный таргет.

        Args:
            db: Соединение с DuckDB.
            meta: Текущая метаинформация.

        Returns:
            Обновлённая Meta с информацией о таргете.
        """
        informative_cols = meta.get_columns_by_tag("informative", "numeric")

        col_name = "target_bin"

        # Читаем фичи
        cols_sql = ", ".join(informative_cols)
        data = db.execute(f"SELECT id, {cols_sql} FROM '{meta.file_path}'").fetchnumpy()

        ids = data["id"]
        features = np.column_stack([data[col] for col in informative_cols])

        rng = np.random.default_rng(self.seed)

        # Специфичные методы для классификации — возвращают метки напрямую
        classification_methods = {
            "xor": self._generate_xor,
            "circles": self._generate_circles,
            "moons": self._generate_moons,
            "clusters": self._generate_clusters,
        }

        if self.method in classification_methods:
            target = classification_methods[self.method](features, rng)
        else:
            # Регрессионные методы — бинаризуем по квантилю
            regression_generators = self._get_regression_generators()

            if self.method not in regression_generators:
                all_methods = list(classification_methods.keys()) + list(
                    regression_generators.keys()
                )
                raise ValueError(
                    f"Неизвестный метод: {self.method}. Доступны: {all_methods}"
                )

            latent = regression_generators[self.method](features, rng)
            threshold_value = np.percentile(latent, self.threshold * 100)
            target = (latent > threshold_value).astype(np.int8)

        # Добавляем шум через flip
        if self.flip_ratio > 0:
            flip_mask = rng.random(len(target)) < self.flip_ratio
            target = np.where(flip_mask, 1 - target, target)

        # Сохраняем
        db.register("target_np", {"id": ids, "target": target})
        _atomic_write(
            db,
            f"""
                SELECT m.*, CAST(t.target AS INT8) AS {col_name}
                FROM '{meta.file_path}' AS m
                JOIN target_np AS t ON m.id = t.id
            """,
            meta.file_path,
        )
        db.unregister("target_np")

        meta.columns[col_name] = "INT8"
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
            name: Кастомное имя трансформера.
        """
        super().__init__(name)
        self.n_classes = n_classes
        self.method = method
        self.seed = seed

    def transform(self, db: duckdb.DuckDBPyConnection, meta: Meta) -> Meta:
        """Генерирует многоклассовый таргет через квантильный биннинг.

        Args:
            db: Соединение с DuckDB.
            meta: Текущая метаинформация.

        Returns:
            Обновлённая Meta с информацией о таргете.
        """
        informative_cols = meta.get_columns_by_tag("informative", "numeric")

        col_name = "target_multi"

        # Читаем фичи
        cols_sql = ", ".join(informative_cols)
        data = db.execute(f"SELECT id, {cols_sql} FROM '{meta.file_path}'").fetchnumpy()

        ids = data["id"]
        features = np.column_stack([data[col] for col in informative_cols])

        rng = np.random.default_rng(self.seed)

        # Генерируем скрытую переменную
        generators = self._get_regression_generators()

        if self.method not in generators:
            raise ValueError(
                f"Неизвестный метод: {self.method}. Доступны: {list(generators.keys())}"
            )

        latent = generators[self.method](features, rng)

        # Биннинг по квантилям на n_classes классов
        percentiles = np.linspace(0, 100, self.n_classes + 1)
        bins = np.percentile(latent, percentiles)
        target = np.digitize(latent, bins[1:-1]).astype(np.int8)

        # Сохраняем
        db.register("target_np", {"id": ids, "target": target})
        _atomic_write(
            db,
            f"""
            SELECT m.*, CAST(t.target AS INT8) AS {col_name}
            FROM '{meta.file_path}' AS m
            JOIN target_np AS t ON m.id = t.id
            """,
            meta.file_path,
        )
        db.unregister("target_np")

        meta.columns[col_name] = "INT8"
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
            query_col = self.category_source
        else:
            category_cols = [
                col for col, tags in meta.column_tags.items() if "category" in tags
            ]
            if not category_cols:
                raise ValueError("Нет категориальных колонок для query_id")
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
                        ) * {self.n_levels - 0.001}
                    ) AS INT8
                ) AS target_rank
            FROM '{meta.file_path}' AS m
            """,
            meta.file_path,
        )

        # Обновляем meta
        meta.columns["query_id"] = "INT64"
        meta.column_tags["query_id"] = ["ranking"]

        meta.columns["target_rank"] = "INT8"
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
            name: Кастомное имя трансформера.
        """
        super().__init__(name)
        self.start_date = start_date
        self.end_date = end_date
        self.target_weight = target_weight
        self.feature_weight = feature_weight
        self.noise_seconds = noise_seconds
        self.seed = seed

    def transform(self, db: duckdb.DuckDBPyConnection, meta: Meta) -> Meta:
        """Генерирует timestamp через линейную комбинацию таргета и фичи.

        Формула: normalized = w1 * norm(target_reg) + w2 * norm(feature)
        Затем масштабирование в диапазон дат + опциональный шум.

        Args:
            db: Соединение с DuckDB.
            meta: Текущая метаинформация.

        Returns:
            Обновлённая Meta с колонкой timestamp.
        """
        # Берём первую информативную фичу
        informative_cols = meta.get_columns_by_tag("informative", "numeric")
        feature_col = informative_cols[0]

        # Читаем данные
        data = db.execute(
            f"SELECT id, target_reg, {feature_col} FROM '{meta.file_path}'"
        ).fetchnumpy()

        target = data["target_reg"]
        feature = data[feature_col]
        ids = data["id"]

        # Нормализуем в [0, 1]
        target_norm = (target - target.min()) / (target.max() - target.min() + EPSILON)
        feature_norm = (feature - feature.min()) / (
            feature.max() - feature.min() + EPSILON
        )

        # Линейная комбинация
        combined = self.target_weight * target_norm + self.feature_weight * feature_norm
        combined = (combined - combined.min()) / (
            combined.max() - combined.min() + EPSILON
        )

        # Преобразуем в timestamps
        start_ts = dt.strptime(self.start_date, "%Y-%m-%d").timestamp()
        end_ts = dt.strptime(self.end_date, "%Y-%m-%d").timestamp()

        timestamps = start_ts + combined * (end_ts - start_ts)

        # Добавляем шум
        if self.noise_seconds > 0:
            rng = np.random.default_rng(self.seed)
            noise = rng.integers(
                -self.noise_seconds, self.noise_seconds + 1, size=len(timestamps)
            )
            timestamps = timestamps + noise

        # Сохраняем через DuckDB
        db.register("ts_np", {"id": ids, "ts": timestamps})

        _atomic_write(
            db,
            f"""
            SELECT
                m.*,
                TO_TIMESTAMP(t.ts) AS timestamp
            FROM '{meta.file_path}' AS m
            JOIN ts_np AS t ON m.id = t.id
            """,
            meta.file_path,
        )
        db.unregister("ts_np")

        # Обновляем meta
        meta.columns["timestamp"] = "TIMESTAMP"
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

        meta.columns["date"] = "DATE"
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
        seed: int | None = None,
        name: str | None = None,
    ):
        """Инициализирует генератор boolean.

        Args:
            threshold: Порог для бинаризации (квантиль от 0 до 1).
            flip_ratio: Доля значений для случайного переключения.
            seed: Seed для воспроизводимости.
            name: Кастомное имя трансформера.
        """
        super().__init__(name)
        self.threshold = threshold
        self.flip_ratio = flip_ratio
        self.seed = seed

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
        # Берём первую числовую колонку
        numeric_cols = [
            col for col, tags in meta.column_tags.items() if "numeric" in tags
        ]
        source_col = numeric_cols[0]
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

        meta.columns[col_name] = "BOOLEAN"
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
    """Выполняет последовательность шагов."""

    def __init__(self, steps: list[Transformer], config: GeneratorConfig):
        """Инициализирует pipeline.

        Args:
            steps: Список трансформеров для выполнения.
            config: Конфигурация генератора.
        """
        self.steps = steps
        self.config = config

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
                meta = step.transform(db, meta)
                logger.debug("Завершён шаг %s, строк: %d", step.name, meta.row_count)

                # Сохраняем прогресс после каждого шага
                self._save_meta(meta)

            logger.info("Pipeline завершён успешно, строк: %d", meta.row_count)
            return meta
        finally:
            db.close()


class PipelineFactory:
    """Создаёт Pipeline из конфига."""

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
        import os
        import tempfile

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
            db = duckdb.connect()
            try:
                meta = Meta(file_path=calibration_path)

                for step in sorted_steps:
                    meta = step.transform(db, meta)
            finally:
                db.close()

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
            if config.n_categories == 0:
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
            steps.append(Boolean(seed=config.seed, name=f"boolean_{i}"))

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

    def create(self, config: GeneratorConfig) -> Pipeline:
        """Собирает pipeline на основе конфига.

        Args:
            config: Конфигурация генератора.

        Returns:
            Готовый Pipeline с упорядоченными шагами.
        """
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
        return Pipeline(sorted_steps, config)
