# Техническое задание: Бенчмарк Polars vs DuckDB

## Цель проекта

Комплексное сравнение производительности Polars и DuckDB:

**Часть 1: Операции**
- Типичные DS-операции
- Scaling на разных размерах данных (100MB → 20GB)
- In-memory и streaming режимы

**Часть 2: End-to-end ML пайплайны**
- Полный цикл: загрузка → cleaning → feature engineering → preprocessing → split → training → validation
- Разные типы задач: классификация, регрессия, ранжирование, кластеризация, снижение размерности
- Разные модели и вариации экспериментов

---

## Структура проекта

```
polars_duckdb_benchmark/
├── README.md
├── pyproject.toml
├── config.py
├── data/
│   ├── generator.py
│   └── schemas.py
├── benchmarks/
│   ├── __init__.py
│   ├── base.py
│   ├── io_bench.py
│   ├── filter_bench.py
│   ├── aggregation_bench.py
│   ├── join_bench.py
│   ├── sort_bench.py
│   ├── window_bench.py
│   ├── vector_bench.py
│   ├── transform_bench.py
│   ├── string_bench.py
│   ├── datetime_bench.py
│   ├── missing_bench.py
│   ├── stats_bench.py
│   ├── unique_bench.py
│   └── cast_bench.py
├── ml_pipelines/
│   ├── __init__.py
│   ├── base_pipeline.py
│   ├── data_loading.py
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   ├── preprocessing.py
│   ├── splitting.py
│   ├── training.py
│   ├── validation.py
│   └── tasks/
│       ├── __init__.py
│       ├── binary_classification.py
│       ├── multiclass_classification.py
│       ├── regression.py
│       ├── ranking.py
│       ├── clustering.py
│       └── dimensionality_reduction.py
├── runners/
│   ├── __init__.py
│   ├── polars_runner.py
│   └── duckdb_runner.py
├── utils/
│   ├── __init__.py
│   ├── metrics.py
│   ├── decorators.py
│   └── reporting.py
├── results/
│   ├── raw/
│   └── reports/
├── run_benchmark.py
├── run_ml_pipelines.py
└── analyze_results.py
```

---

## Зависимости (pyproject.toml)

```toml
[project]
name = "polars-duckdb-benchmark"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    # DataFrame frameworks
    "polars>=1.0.0",
    "duckdb>=1.0.0",
    "pyarrow>=15.0.0",
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    
    # Metrics collection
    "psutil>=5.9.0",
    
    # Visualization
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
    
    # ML: Gradient Boosting
    "xgboost>=2.0.0",
    "lightgbm>=4.0.0",
    "catboost>=1.2.0",
    
    # ML: sklearn
    "scikit-learn>=1.3.0",
    
    # ML: Dimensionality reduction
    "umap-learn>=0.5.0",
    
    # ML: Metrics
    "scipy>=1.11.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "ruff>=0.1.0",
]
```

---

## Конфигурация (config.py)

```python
from dataclasses import dataclass, field

@dataclass
class BenchmarkConfig:
    # Размеры данных
    data_sizes: list[str] = field(default_factory=lambda: [
        "100MB", "500MB", "1GB", "3GB", "10GB", "20GB"
    ])
    
    # Прогоны
    n_runs: int = 5
    cold_runs: int = 1
    warm_runs: int = 4
    
    # Метрики
    measure_time: bool = True
    measure_memory: bool = True
    measure_cpu: bool = True
    
    # Фреймворки и режимы
    frameworks: list[str] = field(default_factory=lambda: ["polars", "duckdb"])
    polars_modes: list[str] = field(default_factory=lambda: ["eager", "lazy", "streaming"])
    
    # Таймаут (секунды)
    timeout: int = 600
    
    # Пути
    data_dir: str = "data/generated"
    results_dir: str = "results"
```

---

## Операции для тестирования

### I/O
| ID | Операция | Polars | DuckDB |
|----|----------|--------|--------|
| io_read_parquet | Чтение Parquet | `pl.read_parquet()` | `SELECT * FROM 'file.parquet'` |
| io_read_csv | Чтение CSV | `pl.read_csv()` | `SELECT * FROM 'file.csv'` |
| io_write_parquet | Запись Parquet | `df.write_parquet()` | `COPY ... TO 'file.parquet'` |
| io_write_csv | Запись CSV | `df.write_csv()` | `COPY ... TO 'file.csv'` |

### Filter
| ID | Операция | Polars | DuckDB |
|----|----------|--------|--------|
| filter_simple | `col > X` | `df.filter(pl.col("a") > 0.5)` | `WHERE a > 0.5` |
| filter_complex | AND/OR | `(A > X) & (B < Y) \| (C == Z)` | `WHERE (a>X AND b<Y) OR c=Z` |
| filter_string_contains | contains | `col.str.contains("sub")` | `WHERE col LIKE '%sub%'` |
| filter_string_regex | regex | `col.str.contains(r"\d+")` | `WHERE regexp_matches(col, '\d+')` |

### Aggregation
| ID | Операция | Polars | DuckDB |
|----|----------|--------|--------|
| agg_single | groupby + sum | `group_by().agg(pl.sum())` | `GROUP BY ... SUM()` |
| agg_multi | несколько агрегаций | `agg([sum, mean, count, std, min, max])` | `SUM(), AVG(), COUNT(), STDDEV(), MIN(), MAX()` |
| agg_nunique | nunique | `n_unique()` | `COUNT(DISTINCT ...)` |
| agg_multiple_keys | 2+ ключа группировки | `group_by(["a", "b"])` | `GROUP BY a, b` |

### Join
| ID | Операция | Polars | DuckDB |
|----|----------|--------|--------|
| join_inner | inner | `df.join(other, on="key")` | `INNER JOIN` |
| join_left | left | `df.join(other, how="left")` | `LEFT JOIN` |
| join_outer | outer | `df.join(other, how="outer")` | `FULL OUTER JOIN` |
| join_cross | cross | `df.join(other, how="cross")` | `CROSS JOIN` |
| join_anti | anti | `df.join(other, how="anti")` | `WHERE NOT EXISTS` |
| join_semi | semi | `df.join(other, how="semi")` | `WHERE EXISTS` |

### Sort
| ID | Операция | Polars | DuckDB |
|----|----------|--------|--------|
| sort_single | 1 колонка | `df.sort("col")` | `ORDER BY col` |
| sort_multi | 3 колонки | `df.sort(["a", "b", "c"])` | `ORDER BY a, b, c` |
| sort_topk | top-k | `df.top_k(1000, by="col")` | `ORDER BY col LIMIT 1000` |

### Window
| ID | Операция | Polars | DuckDB |
|----|----------|--------|--------|
| window_rank | rank | `col.rank().over("partition")` | `RANK() OVER (PARTITION BY ...)` |
| window_dense_rank | dense_rank | `col.rank("dense")` | `DENSE_RANK() OVER ...` |
| window_row_number | row_number | `pl.int_range().over()` | `ROW_NUMBER() OVER ...` |
| window_rolling_mean | rolling mean | `col.rolling_mean(7)` | `AVG() OVER (ROWS 6 PRECEDING)` |
| window_rolling_sum | rolling sum | `col.rolling_sum(7)` | `SUM() OVER (ROWS 6 PRECEDING)` |
| window_lag | lag | `col.shift(1)` | `LAG(col, 1) OVER ...` |
| window_lead | lead | `col.shift(-1)` | `LEAD(col, 1) OVER ...` |
| window_cumsum | cumsum | `col.cum_sum()` | `SUM() OVER (ROWS UNBOUNDED PRECEDING)` |

### Vector Operations
| ID | Операция | Polars | DuckDB |
|----|----------|--------|--------|
| vec_add | сложение | `pl.col("a") + pl.col("b")` | `a + b` |
| vec_mul | умножение | `pl.col("a") * pl.col("b")` | `a * b` |
| vec_div | деление | `pl.col("a") / pl.col("b")` | `a / b` |
| vec_dot | dot product | `(a * b).sum()` | `SUM(a * b)` |
| vec_normalize | normalize | `col / col.sum()` | `col / SUM(col) OVER ()` |
| vec_compound | комплексное | `(a + b) * c - d / e` | `(a + b) * c - d / e` |

### Transform
| ID | Операция | Polars | DuckDB |
|----|----------|--------|--------|
| transform_concat_v | vertical concat | `pl.concat([df1, df2])` | `UNION ALL` |
| transform_concat_h | horizontal concat | `pl.concat([df1, df2], how="horizontal")` | несколько способов |
| transform_pivot | pivot | `df.pivot()` | `PIVOT` |
| transform_unpivot | unpivot/melt | `df.unpivot()` | `UNPIVOT` |
| transform_transpose | transpose | `df.transpose()` | сложный SQL |

### String Operations
| ID | Операция | Polars | DuckDB |
|----|----------|--------|--------|
| str_split | split | `col.str.split("_")` | `string_split(col, '_')` |
| str_replace | replace | `col.str.replace("a", "b")` | `replace(col, 'a', 'b')` |
| str_lower | lowercase | `col.str.to_lowercase()` | `lower(col)` |
| str_upper | uppercase | `col.str.to_uppercase()` | `upper(col)` |
| str_strip | strip | `col.str.strip_chars()` | `trim(col)` |
| str_length | length | `col.str.len_chars()` | `length(col)` |
| str_slice | slice | `col.str.slice(0, 5)` | `substr(col, 1, 5)` |

### DateTime Operations
| ID | Операция | Polars | DuckDB |
|----|----------|--------|--------|
| dt_extract_year | year | `col.dt.year()` | `year(col)` |
| dt_extract_month | month | `col.dt.month()` | `month(col)` |
| dt_extract_day | day | `col.dt.day()` | `day(col)` |
| dt_extract_hour | hour | `col.dt.hour()` | `hour(col)` |
| dt_diff | diff | `col - col_other` | `col - col_other` |
| dt_truncate | truncate | `col.dt.truncate("1d")` | `date_trunc('day', col)` |
| dt_floor | floor | `col.dt.truncate("1h")` | `date_trunc('hour', col)` |

### Missing Values
| ID | Операция | Polars | DuckDB |
|----|----------|--------|--------|
| missing_fillna | fill_null | `col.fill_null(0)` | `COALESCE(col, 0)` |
| missing_dropna | drop_nulls | `df.drop_nulls()` | `WHERE col IS NOT NULL` |
| missing_count | null_count | `col.null_count()` | `COUNT(*) - COUNT(col)` |
| missing_interpolate | interpolate | `col.interpolate()` | оконная функция |

### Statistics
| ID | Операция | Polars | DuckDB |
|----|----------|--------|--------|
| stats_describe | describe | `df.describe()` | несколько агрегаций |
| stats_quantile | quantile | `col.quantile(0.95)` | `quantile_cont(col, 0.95)` |
| stats_corr | correlation | `df.select(pl.corr("a", "b"))` | `corr(a, b)` |
| stats_value_counts | value_counts | `col.value_counts()` | `GROUP BY col, COUNT(*)` |

### Unique Operations
| ID | Операция | Polars | DuckDB |
|----|----------|--------|--------|
| unique_distinct | distinct rows | `df.unique()` | `SELECT DISTINCT *` |
| unique_col | unique values | `col.unique()` | `SELECT DISTINCT col` |
| unique_duplicated | is_duplicated | `col.is_duplicated()` | оконная функция |
| unique_is_in | is_in | `col.is_in([...])` | `col IN (...)` |

### Cast Operations
| ID | Операция | Polars | DuckDB |
|----|----------|--------|--------|
| cast_int_to_float | int → float | `col.cast(pl.Float64)` | `CAST(col AS DOUBLE)` |
| cast_float_to_int | float → int | `col.cast(pl.Int64)` | `CAST(col AS BIGINT)` |
| cast_to_string | → string | `col.cast(pl.String)` | `CAST(col AS VARCHAR)` |
| cast_to_category | → categorical | `col.cast(pl.Categorical)` | нет прямого аналога |
| cast_string_to_date | string → datetime | `col.str.to_datetime()` | `CAST(col AS TIMESTAMP)` |

---

## Часть 2: End-to-end ML пайплайны

### Этапы пайплайна

| Этап | Описание | Что измеряем |
|------|----------|--------------|
| **1. Data Loading** | Загрузка из parquet/csv | Время, память |
| **2. Data Cleaning** | Дубликаты, пропуски, выбросы | Время, память |
| **3. Feature Engineering** | Агрегации, lag, rolling, interactions | Время, память |
| **4. Preprocessing** | Encoding, нормализация | Время, память |
| **5. Train/Test Split** | Разбиение данных | Время |
| **6. Model Training** | Обучение модели | Время, память |
| **7. Validation** | CV, метрики качества | Время, качество модели |

### Типы задач и модели

| Тип задачи | Модели |
|------------|--------|
| **Бинарная классификация** | XGBoost, LightGBM, CatBoost, LogisticRegression, RandomForest |
| **Многоклассовая классификация** | XGBoost, LightGBM, CatBoost, LogisticRegression, RandomForest |
| **Регрессия** | XGBoost, LightGBM, CatBoost, Ridge, Lasso, ElasticNet, RandomForest |
| **Ранжирование** | XGBoost, LightGBM, CatBoost |
| **Кластеризация** | KMeans, DBSCAN, Agglomerative |
| **Снижение размерности** | PCA, t-SNE, UMAP |

### Вариации экспериментов

| Аспект | Варианты |
|--------|----------|
| **Feature sets** | Baseline (сырые фичи), Engineered (агрегации, lag), Full (все) |
| **Preprocessing** | С нормализацией / без, разные encoding стратегии |
| **Валидация** | KFold, StratifiedKFold, TimeSeriesSplit |
| **Размер данных** | 100MB, 500MB, 1GB, 3GB, 10GB, 20GB |

### ML метрики качества

| Тип задачи | Метрики |
|------------|---------|
| **Бинарная классификация** | ROC-AUC, PR-AUC, F1, Accuracy, Precision, Recall |
| **Многоклассовая классификация** | Macro F1, Weighted F1, Accuracy |
| **Регрессия** | RMSE, MAE, MAPE, R² |
| **Ранжирование** | NDCG, MAP, MRR |
| **Кластеризация** | Silhouette, Calinski-Harabasz, Davies-Bouldin |
| **Снижение размерности** | Explained variance, reconstruction error |

### Структура результата ML пайплайна

```python
@dataclass
class MLPipelineResult:
    task_type: str           # "binary_classification" | "regression" | ...
    model_name: str          # "xgboost" | "catboost" | ...
    framework: str           # "polars" | "duckdb"
    data_size: str           # "100MB" | "1GB" | ...
    feature_set: str         # "baseline" | "engineered" | "full"
    preprocessing: str       # "normalized" | "raw"
    validation: str          # "kfold" | "stratified" | "timeseries"
    
    # Время по этапам (секунды)
    time_loading: float
    time_cleaning: float
    time_feature_engineering: float
    time_preprocessing: float
    time_splitting: float
    time_training: float
    time_validation: float
    time_total: float
    
    # Память
    peak_memory_mb: float
    
    # Качество модели (зависит от задачи)
    metrics: dict[str, float]
    
    success: bool
    error_message: str | None = None
```

---

## Метрики

### Структура результата

```python
@dataclass
class BenchmarkResult:
    operation: str           # ID операции
    framework: str           # "polars" | "duckdb"
    mode: str                # "eager" | "lazy" | "streaming" | "sql"
    data_size: str           # "100MB" | "1GB" | ...
    run_number: int          # Номер прогона
    is_cold: bool            # Cold или warm run
    wall_time_seconds: float # Время выполнения
    peak_memory_mb: float    # Пиковая память
    cpu_percent: float       # Загрузка CPU
    success: bool            # Успех/ошибка
    error_message: str | None
```

### Сбор метрик

- **Время:** `time.perf_counter()`
- **Память:** `psutil.Process().memory_info().rss` в фоновом потоке
- **CPU:** `psutil.Process().cpu_percent()` в фоновом потоке
- **Интервал замеров:** 100ms

---

## Размеры данных

| Размер | Режим | Назначение |
|--------|-------|------------|
| 100MB | In-memory | Baseline |
| 500MB | In-memory | Типичный датасет |
| 1GB | In-memory | Комфортный предел |
| 3GB | На пределе | Memory pressure |
| 10GB | Streaming | Тест streaming |
| 20GB | Streaming | Устойчивость |

---

## Окружение

- **ОС:** Windows
- **RAM:** 16GB (~5GB свободно)
- **CPU:** 12 ядер
- **Диск:** SSD

---

## Порядок реализации

### Фаза 1: Инфраструктура
1. Создать структуру проекта
2. `pyproject.toml`
3. `config.py`
4. `utils/metrics.py` — сбор метрик
5. `utils/decorators.py` — декоратор `@benchmark`
6. `benchmarks/base.py` — базовый класс

### Фаза 2: Генерация данных
1. `data/schemas.py` — схема данных (отложено, согласовать отдельно)
2. `data/generator.py` — генератор синтетики

### Фаза 3: Бенчмарки операций
Реализовать по категориям:
1. `io_bench.py`
2. `filter_bench.py`
3. `aggregation_bench.py`
4. `join_bench.py`
5. `sort_bench.py`
6. `window_bench.py`
7. `vector_bench.py`
8. `transform_bench.py`
9. `string_bench.py`
10. `datetime_bench.py`
11. `missing_bench.py`
12. `stats_bench.py`
13. `unique_bench.py`
14. `cast_bench.py`

### Фаза 4: ML пайплайны — инфраструктура
1. `ml_pipelines/base_pipeline.py` — базовый класс
2. `ml_pipelines/data_loading.py`
3. `ml_pipelines/data_cleaning.py`
4. `ml_pipelines/feature_engineering.py`
5. `ml_pipelines/preprocessing.py`
6. `ml_pipelines/splitting.py`
7. `ml_pipelines/training.py`
8. `ml_pipelines/validation.py`

### Фаза 5: ML пайплайны — задачи
1. `ml_pipelines/tasks/binary_classification.py`
2. `ml_pipelines/tasks/multiclass_classification.py`
3. `ml_pipelines/tasks/regression.py`
4. `ml_pipelines/tasks/ranking.py`
5. `ml_pipelines/tasks/clustering.py`
6. `ml_pipelines/tasks/dimensionality_reduction.py`

### Фаза 6: Runner и отчётность
1. `run_benchmark.py` — точка входа для операций
2. `run_ml_pipelines.py` — точка входа для ML
3. `utils/reporting.py` — сохранение результатов
4. `analyze_results.py` — анализ и визуализация

### Фаза 7: Тестирование
1. Прогон операций на 100MB — проверка работоспособности
2. Прогон ML пайплайнов на 100MB
3. Полный прогон
4. Анализ результатов

---

## Важные требования

1. **Согласовывать план перед написанием кода**
2. **Не писать лишней документации**
3. **Имена переменных:** `features`, `target` (не `X`, `y`)
4. **Задавать уточняющие вопросы при необходимости**
5. **GC перед каждым замером:** `gc.collect()`
6. **Streaming Polars:** `collect(engine="streaming")` для Polars ≥1.31
7. **DuckDB streaming:** автоматический при `duckdb.sql()` на файлах
