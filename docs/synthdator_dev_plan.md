# План разработки: Генератор синтетических данных для ML

## Цель

Независимый модуль для генерации синтетических табличных данных с контролируемыми характеристиками. Данные подходят для:
- Тестирования DataFrame-фреймворков (Polars, DuckDB, Pandas, etc.)
- ML задач: регрессия, бинарная/многоклассовая классификация, ранжирование, кластеризация
- Тестирования операций: groupby, join, string ops, datetime ops, missing values

---

## Архитектура

### Основные принципы

- **Pipeline-подход:** генерация — цепочка шагов-трансформаций
- **File-based:** данные хранятся в parquet, шаги модифицируют один файл `main.parquet`
- **DuckDB:** основа для трансформаций (out-of-core, SQL, работа с parquet)
- **Модульность:** шаги можно комбинировать, переиспользовать, заменять
- **Фабрика:** `PipelineFactory` строит Pipeline по декларативному конфигу
- **Планировщик:** фабрика содержит планировщик, который строит порядок шагов на основе их зависимостей

### Структура модуля

```
data/
└── generator.py    # Config, Meta, Step, конкретные шаги, Pipeline, PipelineFactory
```

---

## Конфигурация

### Два уровня

1. **GeneratorConfig** — декларативное описание "что хочется сгенерировать"
2. **Параметры шагов** — детали реализации, определяются фабрикой или переопределяются

### GeneratorConfig

Описывает типы данных на высоком уровне, не конкретные колонки:

```python
@dataclass
class GeneratorConfig:
    # Обязательные
    target_size: str      # "100MB", "1GB", etc.
    output_path: str      # Директория для сохранения

    # Воспроизводимость
    seed: int | None = None

    # Что генерировать
    numeric_features: dict | None = None    # {"count": 20, "informative": 10, ...}
    categories: dict | None = None          # {"low": 5, "mid": 50, "high": 500} — кол-во кластеров
    targets: list | dict | None = None      # ["regression", "binary", "multiclass", "ranking"]
    strings: bool | dict = False            # {"noise_ratio": 0.1}
    dates: bool | dict = False              # {"start": "2020-01-01", "end": "2024-01-01"}
    booleans: bool | dict = False           # {"threshold": 0.5, "flip_ratio": 0.1}
    nullable: bool | dict = False           # {"tags": ["numeric"], "ratio": 0.1} или детальнее

    # Параметры ранжирования
    ranking: dict | None = None             # {"levels": 5, "category_source": "cat_low"}

    # Переопределения для отдельных шагов (по имени шага)
    step_overrides: dict | None = None
```

---

## PipelineFactory

```python
class PipelineFactory:
    def create(self, config: GeneratorConfig) -> Pipeline:
        """
        Строит Pipeline на основе конфига.
        Определяет какие шаги нужны и с какими параметрами.
        Планировщик упорядочивает шаги по зависимостям.
        """
        ...
```

### Логика фабрики

1. Читает конфиг
2. Определяет нужные шаги на основе заданных типов данных
3. Создаёт шаги с параметрами по умолчанию
4. Применяет `step_overrides` если заданы
5. **Планировщик** топологически сортирует шаги по зависимостям
6. Возвращает готовый Pipeline

---

## Pipeline

```python
class Pipeline:
    def __init__(self, steps: list[Step], config: GeneratorConfig):
        ...

    def run(self) -> None:
        """
        Выполняет все шаги последовательно.
        1. Калибровочный прогон на sample для определения row_count
        2. Полная генерация с чанкованием
        3. Сохранение прогресса в Meta для возможности продолжения
        """
        ...
```

### Калибровка размера

1. Pipeline собирается со всеми шагами
2. Выполняется калибровочный прогон на малом sample (например, 10000 строк)
3. Измеряется размер полученного parquet файла
4. Экстраполируется количество строк для достижения `target_size`
5. Pipeline выполняется на полном объёме

### Обработка ошибок

- Если шаг упал, прогресс сохраняется в Meta
- При повторном запуске pipeline продолжает с упавшего шага
- Meta хранит информацию о выполненных шагах

---

## Интерфейс шага

```python
class Step:
    name: str                    # Уникальный идентификатор
    requires: list[str] = []     # Зависимости от других шагов

    def transform(self, db: duckdb.DuckDBPyConnection, meta: Meta) -> Meta:
        """
        Модифицирует таблицу в db, возвращает обновлённую мета-информацию.
        """
        ...
```

### Зависимости шагов

Каждый шаг декларирует свои зависимости в атрибуте `requires`. Планировщик фабрики использует эту информацию для топологической сортировки.

---

## Meta

Информация о текущем состоянии данных, передаётся между шагами.

```python
@dataclass
class Meta:
    # Основное
    file_path: str              # Путь к main.parquet
    row_count: int              # Количество строк

    # Колонки и их теги
    columns: dict[str, str]     # {name: dtype}
    column_tags: dict[str, list[str]]  # {name: ["numeric", "informative", ...]}

    # Прогресс pipeline
    completed_steps: list[str]  # Список выполненных шагов

    # Расширяется по мере необходимости
```

### Теги колонок

Шаги добавляют теги к создаваемым колонкам. Другие шаги (например, `NullableStep`) используют теги для выбора колонок.

Примеры тегов:
- `numeric` — числовая колонка
- `informative` — информативная фича
- `category` — категориальная колонка
- `target` — целевая переменная

### Сериализация

Meta сериализуется в pickle для сохранения прогресса и возможности продолжения pipeline.

---

## Доступные шаги

### Инициализация

| Шаг | Описание | Зависимости |
|-----|----------|-------------|
| `InitStep` | Создаёт таблицу с id | — |
| `CalibrationStep` | Калибровочный прогон для расчёта row_count | Все шаги генерации |

### Числовые фичи

| Шаг | Описание | Зависимости |
|-----|----------|-------------|
| `NumericFeaturesStep` | Генерирует числовые фичи через sklearn | `InitStep` |

**Детали реализации:**
- Использует `make_classification` / `make_regression` из sklearn
- Генерация чанками для экономии памяти
- Seed для чанка: `seed + chunk_index`
- Добавляет теги: `numeric`, `informative` (для информативных фич)

### Targets

| Шаг | Описание | Зависимости |
|-----|----------|-------------|
| `RegressionTargetStep` | target_reg через make_regression | `NumericFeaturesStep` |
| `BinaryTargetStep` | target_bin через make_classification | `NumericFeaturesStep` |
| `MulticlassTargetStep` | target_multi через make_classification | `NumericFeaturesStep` |
| `RankingTargetStep` | target_rank + query_id | `RegressionTargetStep`, `CategoryStep` |

**RankingTargetStep детали:**
- `query_id` берётся из существующей категориальной колонки
- `target_rank` формируется через ранжирование по `target_reg` внутри группы + биннинг на N уровней
- Количество уровней — параметр конфига (по умолчанию 5)

### Категории

| Шаг | Описание | Зависимости |
|-----|----------|-------------|
| `CategoryStep` | Категория через MiniBatchKMeans на числовых фичах | `NumericFeaturesStep` |

**Детали реализации:**
- Использует `MiniBatchKMeans` для масштабируемости
- Разная кардинальность через разное количество кластеров:
  - `cat_low`: ~5-10 кластеров
  - `cat_mid`: ~50 кластеров
  - `cat_high`: ~500 кластеров
- Добавляет тег: `category`

### Строки

| Шаг | Описание | Зависимости |
|-----|----------|-------------|
| `StringStep` | Строка как бакетизация числовой фичи + шум | `NumericFeaturesStep` |

**Детали реализации:**
- Бакетизация числовой фичи в текстовые значения (`"low"`, `"medium"`, `"high"`)
- Шум: случайная подмена бакета с заданной вероятностью (`noise_ratio`)
- Добавляет тег: `string`

### Даты

| Шаг | Описание | Зависимости |
|-----|----------|-------------|
| `DatetimeStep` | timestamp из линейной комбинации target и фичи | `NumericFeaturesStep`, `RegressionTargetStep` |
| `DateStep` | date из timestamp | `DatetimeStep` |

**DatetimeStep детали:**
- Линейная комбинация: `w1 * target_reg + w2 * feature` (веса настраиваемые)
- Нормализация в [0, 1]
- Масштабирование в диапазон дат (`start_date`, `end_date`)
- Опциональный шум (±N секунд)
- Добавляет тег: `datetime`

### Прочее

| Шаг | Описание | Зависимости |
|-----|----------|-------------|
| `BooleanStep` | Boolean через бинаризацию фичи + flip | `NumericFeaturesStep` |
| `NullableStep` | Добавление NULL в существующие колонки | Зависит от целевых колонок |

**BooleanStep детали:**
- Бинаризация числовой фичи по порогу (`threshold`)
- Случайный flip с вероятностью (`flip_ratio`)
- Добавляет тег: `boolean`

**NullableStep детали:**
- Модифицирует существующие колонки, добавляя NULL
- Выбор колонок по тегам (например, `{"tags": ["numeric"]}`)
- Поддержка единого `ratio` и индивидуального для разных тегов/колонок
- Не создаёт новые колонки

---

## Чанковая генерация

Для экономии памяти sklearn-шаги генерируют данные чанками:

1. Определяется размер чанка (например, 100000 строк)
2. Для каждого чанка:
   - Seed = `base_seed + chunk_index`
   - Генерация данных в память
   - Append в parquet через DuckDB
3. Результат идентичен при повторном запуске (воспроизводимость)

---

## Пример использования

```python
# Через фабрику (рекомендуемый способ)
config = GeneratorConfig(
    target_size="1GB",
    output_path="./data",
    seed=42,
    numeric_features={"count": 20, "informative": 10},
    categories={"low": 5, "mid": 50, "high": 500},
    targets=["regression", "binary", "multiclass", "ranking"],
    strings={"noise_ratio": 0.1},
    dates={"start": "2020-01-01", "end": "2024-01-01"},
    booleans={"threshold": 0.5, "flip_ratio": 0.05},
    nullable={"tags": ["numeric"], "ratio": 0.1},
    ranking={"levels": 5, "category_source": "cat_low"},
)

factory = PipelineFactory()
pipeline = factory.create(config)
pipeline.run()
```

---

## Выходные данные

### main.parquet

Один файл со всеми колонками. Состав зависит от конфига:

| Колонка | Тип | Описание | Теги |
|---------|-----|----------|------|
| `id` | Int64 | PK | — |
| `feature_*` | Float64 | Числовые фичи | `numeric`, `informative` |
| `cat_low`, `cat_mid`, `cat_high` | String | Категории | `category` |
| `text` | String | Строковая колонка | `string` |
| `timestamp` | Datetime | Дата-время | `datetime` |
| `date` | Date | Дата | `date` |
| `flag` | Boolean | Бинарный признак | `boolean` |
| `target_reg` | Float64 | Регрессия | `target` |
| `target_bin` | Int8 | Бинарная классификация | `target` |
| `target_multi` | Int8 | Многоклассовая | `target` |
| `target_rank` | Int8 | Ранжирование (0 до N-1) | `target` |
| `query_id` | Int64 | Группа для ранжирования | — |

### meta.pkl

Pickle-файл с объектом Meta для сохранения прогресса и возможности продолжения.

---

## Зависимости

- `duckdb`
- `numpy`
- `scikit-learn` (make_regression, make_classification, MiniBatchKMeans)

---

## Граф зависимостей шагов

```
InitStep
    │
    ▼
NumericFeaturesStep
    │
    ├──────────────┬──────────────┬──────────────┐
    ▼              ▼              ▼              ▼
CategoryStep   StringStep   BooleanStep   DatetimeStep ◄── RegressionTargetStep
    │              │              │              │                │
    │              │              │              ▼                │
    │              │              │          DateStep             │
    │              │              │                               │
    ▼              ▼              ▼                               ▼
    └──────────────┴──────────────┴───────────────────────────────┘
                                  │
                                  ▼
                            NullableStep
                                  │
                                  ▼
                         RankingTargetStep (требует CategoryStep + RegressionTargetStep)
```

---

## Демо скрипт

Вместо полноценных тестов — демо скрипт `demo_generator.py`, проверяющий end-to-end работу pipeline:

```python
# demo_generator.py
from data.generator import GeneratorConfig, PipelineFactory

config = GeneratorConfig(
    target_size="100MB",
    output_path="./demo_data",
    seed=42,
    numeric_features={"count": 10, "informative": 5},
    categories={"low": 5},
    targets=["regression", "binary"],
    strings=True,
    dates=True,
    nullable={"tags": ["numeric"], "ratio": 0.05},
)

factory = PipelineFactory()
pipeline = factory.create(config)
pipeline.run()

# Проверка результата
import duckdb
db = duckdb.connect()
result = db.execute("SELECT COUNT(*) FROM './demo_data/main.parquet'").fetchone()
print(f"Generated {result[0]} rows")
```
