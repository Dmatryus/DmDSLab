# План разработки UCI Dataset Loader для DmDSLab

## Обновления на основе требований

План обновлен с учетом следующих ключевых требований:
- **Кеширование**: Использование pickle формата с возможностью перезаписи
- **Логирование**: Все уровни (DEBUG, INFO, WARNING, ERROR) с настройкой
- **Progress Bar**: Интеграция с tqdm для отображения прогресса загрузки
- **Множественная загрузка**: Последовательная загрузка нескольких датасетов
- **Категориальные признаки**: Автоматическое определение при необходимости
- **Упрощение**: Убраны валидация, сетевые компоненты, история, CLI и другие избыточные функции

## 1. Схема классов

### 1.1. Основные классы

```mermaid
classDiagram
    class UCIDatasetManager {
        -str cache_dir
        -bool raise_on_missing
        -Dict cache_index
        -Logger logger
        -str log_level
        -bool show_progress
        +__init__(cache_dir, raise_on_missing, log_level, show_progress)
        +load_dataset(dataset_id, force_reload) ModelData
        +load_datasets(dataset_ids) List[ModelData]
        -_setup_cache() void
        -_setup_logging(log_level) void
        -_check_cache(dataset_id) Tuple[bool, dict]
        -_load_from_cache(dataset_id) ModelData
        -_save_to_cache(dataset_id, data, metadata) bool
        -_fetch_from_uci(dataset_id, show_progress) object
        -_analyze_structure(raw_data) Tuple
        -_detect_task_type(features, target) TaskType
        -_detect_categorical_features(features) List[int]
        -_create_model_data(features, target, feature_names, metadata) ModelData
        +clear_cache(dataset_id) bool
        +get_cache_info() Dict
    }

    class DatasetInfo {
        +int dataset_id
        +str name
        +str description
        +TaskType task_type
        +Domain domain
        +int n_instances
        +int n_features
        +List[str] feature_types
        +bool has_missing_values
        +Dict additional_info
        +datetime cached_at
        +str cache_version
    }

    class UCIDatasetError {
        <<exception>>
        +str message
        +int dataset_id
        +str error_type
    }

    class TaskType {
        <<enumeration>>
        CLASSIFICATION
        REGRESSION
        CLUSTERING
        UNKNOWN
    }

    class Domain {
        <<enumeration>>
        BUSINESS
        COMPUTER
        ENGINEERING
        GAMES
        LIFE
        PHYSICAL
        SOCIAL
        OTHER
    }

    class CacheManager {
        -str cache_dir
        -Dict index
        -int protocol_version
        +load_index() Dict
        +save_index(index) void
        +get_cache_path(dataset_id) Path
        +exists(dataset_id) bool
        +save_dataset(dataset_id, data, metadata) void
        +load_dataset(dataset_id) Tuple[object, Dict]
        +invalidate(dataset_id) void
        +calculate_cache_size() float
        +clear_all() void
        +get_cached_datasets() List[int]
    }

    UCIDatasetManager --> DatasetInfo : creates
    UCIDatasetManager --> CacheManager : uses
    UCIDatasetManager --> UCIDatasetError : throws
    DatasetInfo --> TaskType : contains
    DatasetInfo --> Domain : contains
    UCIDatasetManager --> ModelData : returns
```

### 1.2. Вспомогательные компоненты

```mermaid
classDiagram
    class DatasetRegistry {
        <<singleton>>
        -Dict[int, DatasetInfo] registry
        +get_instance() DatasetRegistry
        +register(dataset_id, info) void
        +get_info(dataset_id) DatasetInfo
        +search_by_name(name) List[DatasetInfo]
        +filter_by_domain(domain) List[DatasetInfo]
        +filter_by_task_type(task_type) List[DatasetInfo]
    }

    class NetworkManager {
        -int timeout
        -int max_retries
        -Dict proxy_settings
        +check_connectivity() bool
        +download_with_retry(url, max_retries) Response
        +estimate_download_time(size) float
    }

    class DataValidator {
        +validate_features(features) List[str]
        +validate_target(target, task_type) List[str]
        +check_data_quality(features, target) Dict
        +detect_categorical_columns(features) List[int]
        +infer_feature_types(features) Dict[str, str]
        +suggest_preprocessing(validation_report) List[str]
    }

    class MetadataExtractor {
        +extract_from_uci(raw_data) Dict
        +infer_feature_types(features) List[str]
        +detect_categorical(features) List[int]
        +calculate_statistics(features, target) Dict
    }
```

## 2. Workflow процесса загрузки

### 2.1. Основной процесс

```mermaid
sequenceDiagram
    participant User
    participant UCIDatasetManager as Manager
    participant Logger
    participant CacheManager as Cache
    participant ProgressBar
    participant ucimlrepo
    participant ModelData

    User->>Manager: load_dataset(dataset_id, force_reload=False)
    Manager->>Logger: log(INFO, "Starting dataset load")
    Manager->>Manager: _setup_cache()
    
    alt not force_reload
        Manager->>Cache: check_cache(dataset_id)
        Cache-->>Manager: (exists, cache_metadata)
        alt Dataset in cache
            Manager->>Logger: log(DEBUG, "Loading from cache")
            Manager->>Cache: load_from_cache(dataset_id)
            Cache-->>Manager: cached_data
        else Dataset not in cache
            Manager->>Logger: log(INFO, "Cache miss, downloading")
        end
    else force_reload
        Manager->>Logger: log(INFO, "Force reload requested")
    end
    
    alt Need to download
        Manager->>ProgressBar: create("Downloading dataset")
        Manager->>ucimlrepo: fetch(dataset_id)
        ucimlrepo->>ProgressBar: update(progress)
        ucimlrepo-->>Manager: raw_data
        ProgressBar->>User: [Download complete]
        
        Manager->>Manager: _analyze_structure(raw_data)
        Manager->>Manager: _detect_categorical_features()
        Manager->>Cache: save_to_cache(dataset_id, data)
        Manager->>Logger: log(DEBUG, "Saved to cache")
    end
    
    Manager->>ModelData: create(features, target, metadata)
    ModelData-->>Manager: model_data_instance
    
    Manager->>Logger: log(INFO, "Dataset loaded successfully")
    Manager->>User: ModelData
```

## 3. Структура модулей

### 3.1. Файловая структура

```
dmdslab/
├── datasets/
│   ├── __init__.py
│   ├── ml_data_container.py (существующий)
│   └── uci/
│       ├── __init__.py
│       ├── manager.py          # Основной класс UCIDatasetManager
│       ├── cache.py            # CacheManager и работа с кешем
│       ├── metadata.py         # DatasetInfo и MetadataExtractor
│       ├── exceptions.py       # UCIDatasetError и другие исключения
│       ├── types.py           # TaskType, Domain и другие типы
│       └── utils.py           # Вспомогательные функции и progress bar
```

### 3.2. Разбивка на логические модули

```mermaid
graph TD
    subgraph "Модуль управления (manager.py)"
        A[UCIDatasetManager]
        A1[Основная логика загрузки]
        A2[Координация компонентов]
        A3[API для пользователя]
        A4[Логирование процессов]
    end
    
    subgraph "Модуль кеширования (cache.py)"
        B[CacheManager]
        B1[Управление индексом]
        B2[Pickle сериализация]
        B3[Перезапись кеша]
    end
    
    subgraph "Модуль метаданных (metadata.py)"
        D[DatasetInfo]
        D1[MetadataExtractor]
        D2[Определение типов задач]
        D3[Определение категориальных]
    end
    
    subgraph "Модуль утилит (utils.py)"
        E[Progress Bar]
        E1[Форматирование вывода]
        E2[Вспомогательные функции]
        E3[Обертки для tqdm]
    end
    
    subgraph "Модуль типов (types.py)"
        F[TaskType enum]
        F1[Domain enum]
        F2[LogLevel enum]
    end
    
    A --> B
    A --> D
    A --> E
```

## 4. Детальный план разработки

### 4.1. Этап 1: Базовая инфраструктура

#### Задачи:
1. **Создание структуры модуля**
   - Создать директорию `dmdslab/datasets/uci/`
   - Создать все необходимые файлы-заготовки
   - Настроить импорты в `__init__.py` файлах

2. **Реализация базовых типов (types.py)**
   ```python
   # Элементарные задачи:
   - [ ] Создать enum TaskType
   - [ ] Создать enum Domain  
   - [ ] Создать enum CacheStatus
   - [ ] Создать enum LogLevel (DEBUG, INFO, WARNING, ERROR)
   - [ ] Добавить type hints для всех типов
   ```

3. **Реализация исключений (exceptions.py)**
   ```python
   # Элементарные задачи:
   - [ ] Создать базовый класс UCIDatasetError
   - [ ] Создать CacheError
   - [ ] Создать ValidationError
   - [ ] Создать DatasetNotFoundError
   ```

4. **Настройка логирования**
   ```python
   # Элементарные задачи:
   - [ ] Создать функцию setup_logger() с настройкой уровня
   - [ ] Добавить форматирование логов
   - [ ] Создать декоратор для логирования методов
   ```

### 4.2. Этап 2: Модуль кеширования

#### Задачи:
1. **Класс CacheManager (cache.py)**
   ```python
   # Элементарные задачи:
   - [ ] Метод __init__ с настройкой директории
   - [ ] Метод load_index() для загрузки индекса
   - [ ] Метод save_index() для сохранения индекса
   - [ ] Метод get_cache_path() для путей к файлам
   - [ ] Метод save_dataset() с pickle
   - [ ] Метод load_dataset() с pickle
   - [ ] Метод exists() для проверки наличия
   - [ ] Метод invalidate() для перезаписи
   - [ ] Метод calculate_cache_size()
   - [ ] Метод clear_all() для полной очистки
   ```

2. **Утилиты для работы с pickle**
   ```python
   # Элементарные задачи:
   - [ ] Функция safe_pickle_dump() с обработкой ошибок
   - [ ] Функция safe_pickle_load() с обработкой ошибок
   - [ ] Обработка версий pickle протокола
   ```

3. **Тесты для CacheManager**
   ```python
   # Элементарные задачи:
   - [ ] Тест создания директории кеша
   - [ ] Тест сохранения/загрузки данных
   - [ ] Тест работы с индексом
   - [ ] Тест перезаписи кеша
   - [ ] Тест очистки кеша
   ```

### 4.3. Этап 3: Модуль метаданных

#### Задачи:
1. **Класс DatasetInfo (metadata.py)**
   ```python
   # Элементарные задачи:
   - [ ] Определить dataclass с полями
   - [ ] Метод from_uci_data() для создания из UCI
   - [ ] Метод to_dict() для сериализации
   - [ ] Метод from_dict() для десериализации
   ```

2. **Класс MetadataExtractor (metadata.py)**
   ```python
   # Элементарные задачи:
   - [ ] Метод extract_from_uci()
   - [ ] Метод infer_feature_types()
   - [ ] Метод detect_categorical()
   - [ ] Метод calculate_statistics()
   - [ ] Метод determine_task_type()
   ```

3. **Определение категориальных признаков**
   ```python
   # Элементарные задачи:
   - [ ] Анализ уникальных значений
   - [ ] Проверка соотношения уникальных/общих
   - [ ] Анализ типов данных (object, int с малым кардиналом)
   - [ ] Эвристики для определения категорий
   ```

### 4.4. Этап 4: Утилиты и Progress Bar

#### Задачи:
1. **Progress Bar утилиты (utils.py)**
   ```python
   # Элементарные задачи:
   - [ ] Функция create_progress_bar() с tqdm
   - [ ] Контекстный менеджер для progress bar
   - [ ] Обертки для загрузки с прогрессом
   - [ ] Настройка отображения (console/notebook)
   ```

2. **Вспомогательные функции**
   ```python
   # Элементарные задачи:
   - [ ] Функция format_dataset_info()
   - [ ] Функция estimate_download_size()
   - [ ] Функция validate_dataset_id()
   - [ ] Функция print_dataset_summary()
   ```

### 4.5. Этап 5: Основной менеджер

#### Задачи:
1. **Класс UCIDatasetManager (manager.py)**
   ```python
   # Элементарные задачи:
   - [ ] Метод __init__ с параметрами (cache_dir, raise_on_missing, log_level, show_progress)
   - [ ] Метод _setup_cache()
   - [ ] Метод _setup_logging()
   - [ ] Метод _check_cache()
   - [ ] Метод _load_from_cache()
   - [ ] Метод _save_to_cache()
   - [ ] Метод _fetch_from_uci() с progress bar
   - [ ] Метод _analyze_structure()
   - [ ] Метод _detect_task_type()
   - [ ] Метод _detect_categorical_features()
   - [ ] Метод _create_model_data()
   - [ ] Метод load_dataset() с force_reload параметром
   - [ ] Метод load_datasets() для загрузки нескольких
   - [ ] Метод clear_cache()
   - [ ] Метод get_cache_info()
   ```

2. **Логирование**
   ```python
   # Элементарные задачи:
   - [ ] Настройка уровней логирования
   - [ ] Логирование всех этапов загрузки
   - [ ] Форматирование сообщений
   - [ ] Интеграция с progress bar
   ```

3. **Загрузка нескольких датасетов**
   ```python
   # Элементарные задачи:
   - [ ] Последовательная загрузка
   - [ ] Общий progress bar для всех
   - [ ] Обработка ошибок для каждого
   - [ ] Возврат списка ModelData или dict с ошибками
   ```

### 4.6. Этап 6: Интеграция и тестирование

#### Задачи:
1. **Интеграционные тесты**
   ```python
   # Элементарные задачи:
   - [ ] Тест полного цикла загрузки
   - [ ] Тест работы без интернета
   - [ ] Тест повреждённого кеша
   - [ ] Тест различных типов датасетов
   ```

2. **Обновление документации**
   ```markdown
   # Элементарные задачи:
   - [ ] Обновить __init__.py файлы
   - [ ] Создать README для модуля
   - [ ] Добавить примеры использования
   - [ ] Создать Jupyter notebook с демо
   ```

### 4.7. Этап 7: Дополнительные возможности

#### Задачи:
1. **Обновление requirements**
   ```python
   # Элементарные задачи:
   - [ ] Добавить ucimlrepo в requirements
   - [ ] Добавить tqdm для progress bar
   - [ ] Создать extras_require для опциональных зависимостей
   - [ ] Обновить setup.py
   ```

2. **Утилиты (utils.py)**
   ```python
   # Элементарные задачи:
   - [ ] Функция print_dataset_summary()
   - [ ] Функция get_popular_datasets()
   - [ ] Функция format_cache_size()
   - [ ] Функция create_download_report()
   ```

## 5. Критерии готовности

### 5.1. Функциональные требования
- [ ] Загрузка датасета по ID работает корректно
- [ ] Кеширование с pickle функционирует без ошибок
- [ ] Перезапись кеша по force_reload работает
- [ ] Загрузка нескольких датасетов последовательно
- [ ] Progress bar отображается корректно
- [ ] Логирование с настраиваемым уровнем
- [ ] Определение категориальных признаков
- [ ] Интеграция с ModelData работает
- [ ] Все тесты проходят успешно

### 5.2. Нефункциональные требования
- [ ] Код соответствует PEP 8
- [ ] Покрытие тестами > 80%
- [ ] Документация complete
- [ ] Примеры работают корректно
- [ ] Производительность приемлема
- [ ] Логи информативны на всех уровнях

## 6. Риски и митигация

| Риск | Вероятность | Влияние | Митигация |
|------|-------------|---------|-----------|
| Изменение API ucimlrepo | Средняя | Высокое | Абстрактный слой для API |
| Большой размер кеша | Высокая | Среднее | Метод clear_all() и информация о размере |
| Недоступность UCI сервера | Низкая | Высокое | Использование кеша, информативные ошибки |
| Несовместимые форматы данных | Средняя | Среднее | Гибкая система обработки |
| Проблемы с pickle версиями | Низкая | Среднее | Фиксация protocol версии |