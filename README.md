# DmDSLab: Data Science Toolkit

[![PyPI version](https://badge.fury.io/py/DmDSLab.svg)](https://badge.fury.io/py/DmDSLab)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Downloads](https://pepy.tech/badge/dmdslab)](https://pepy.tech/project/dmdslab)

## 🎉 Новое в версии 2.0

- 🌐 **UCI Dataset Loader**: Доступ к датасетам из UCI ML Repository
- 💾 **Интеллектуальное кеширование**: Загрузил один раз - используй всегда
- 🎯 **Автоматическое определение**: Типы задач и категориальные признаки из коробки
- 📊 **Progress Bars**: Визуальный контроль процесса загрузки

## 🚀 Основные возможности

### Работа с UCI Repository
- **Простая загрузка**: Один вызов функции для получения датасета
- **Кеширование**: Автоматическое сохранение загруженных данных
- **Метаданные**: Полная информация о каждом датасете
- **Фильтрация**: Поиск датасетов по различным критериям

### Контейнеры данных
- **ModelData**: Универсальный контейнер для ML данных с автоматической валидацией
- **DataSplit**: Гибкие разбиения train/validation/test
- **DataInfo**: Структурированные метаданные о датасетах

### Препроцессинг
- **Автоматическое преобразование**: Pandas ↔ NumPy
- **Валидация данных**: Проверка размерностей и типов
- **Выборки**: Случайные и стратифицированные
- **K-fold**: Встроенная поддержка кросс-валидации

## 📦 Установка

```bash
# Базовая установка
pip install DmDSLab

# С поддержкой UCI (рекомендуется)
pip install DmDSLab[uci]

# Полная установка со всеми зависимостями
pip install DmDSLab[all]
```

## 🔥 Быстрый старт

### Загрузка из UCI Repository (НОВОЕ!)

```python
from dmdslab.datasets import load_uci_by_name, load_uci_dataset

# Самый простой способ - по имени
iris = load_uci_by_name('iris')
print(f"Загружен {iris.metadata['name']}: {iris.shape}")

# Загрузка по ID
wine = load_uci_dataset(109)  # Wine dataset

# Использование менеджера для больших возможностей
from dmdslab.datasets.uci import UCIDatasetManager

manager = UCIDatasetManager()
data = manager.load_dataset(53, show_progress=True)  # Iris с progress bar
```

### Работа с контейнерами данных

```python
from dmdslab.datasets import ModelData, create_data_split
import numpy as np

# Создание данных
X = np.random.randn(1000, 10)
y = np.random.randint(0, 2, 1000)

data = ModelData(features=X, target=y)
print(f"Dataset shape: {data.shape}")

# Автоматическое разбиение
split = create_data_split(X, y, test_size=0.2, validation_size=0.1)
print(f"Train: {split.train.n_samples}")
print(f"Val: {split.validation.n_samples}")
print(f"Test: {split.test.n_samples}")
```

## 📚 Примеры использования

### 🌐 UCI Dataset Loader

#### Пакетная загрузка датасетов
```python
from dmdslab.datasets.uci import UCIDatasetManager

manager = UCIDatasetManager()

# Загрузка нескольких датасетов
dataset_ids = [53, 17, 19, 45]  # Iris, Breast Cancer, Wine, Heart Disease
datasets = manager.load_datasets(dataset_ids, show_progress=True)

for data in datasets:
    if isinstance(data, ModelData):
        print(f"{data.metadata['name']}: {data.shape}")
```

#### Управление кешем
```python
# Проверка информации о кеше
info = manager.get_cache_info()
print(f"Датасетов в кеше: {info['total_datasets']}")
print(f"Размер кеша: {info['total_size']}")

# Очистка конкретного датасета
manager.clear_cache(dataset_id=53)

# Полная очистка кеша
manager.clear_cache()
```

#### Поиск датасетов (если установлена база метаданных)
```python
# Инициализация базы данных UCI
python scripts/initialize_uci_database.py

# Использование фильтров
from dmdslab.datasets.uci_dataset_manager import UCIDatasetManager, TaskType

manager = UCIDatasetManager()
binary_datasets = manager.filter_datasets(
    task_type=TaskType.BINARY_CLASSIFICATION,
    min_instances=1000,
    max_instances=10000
)
```

### 📊 Работа с разбиениями

```python
from dmdslab.datasets import create_data_split, create_kfold_data

# Разбиение со стратификацией
split = create_data_split(
    X, y, 
    test_size=0.2,
    validation_size=0.2,
    stratify=True,
    random_state=42
)

# K-fold кросс-валидация
kfold_splits = create_kfold_data(X, y, n_splits=5, random_state=42)

for i, fold in enumerate(kfold_splits):
    print(f"Fold {i}: train={fold.train.n_samples}, val={fold.validation.n_samples}")
```

### 🔬 Интеграция с scikit-learn

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Загрузка данных из UCI
data = load_uci_by_name('wine')
split = create_data_split(data.features, data.target, test_size=0.2)

# Обучение модели
model = RandomForestClassifier(random_state=42)
model.fit(split.train.features, split.train.target)

# Оценка
predictions = model.predict(split.test.features)
accuracy = accuracy_score(split.test.target, predictions)
print(f"Accuracy: {accuracy:.3f}")
```

## 🗄️ Доступные UCI датасеты

DmDSLab включает быстрый доступ к популярным датасетам:

| Имя | ID | Описание | Размер |
|-----|-----|----------|---------|
| `iris` | 53 | Классический датасет ирисов Фишера | 150×4 |
| `wine` | 109 | Классификация итальянских вин | 178×13 |
| `breast_cancer` | 17 | Диагностика рака груди Wisconsin | 569×30 |
| `adult` | 2 | Предсказание дохода >$50K | 48842×14 |
| `mushroom` | 73 | Классификация съедобных грибов | 8124×22 |
| `heart_disease` | 45 | Диагностика сердечных заболеваний | 303×13 |

И многие другие! Полный список доступен через `POPULAR_DATASETS`.

Вы можете взять любой датасет с 
[сайта](https://archive.ics.uci.edu/datasets?skip=0&take=10&sort=desc&orderBy=NumHits&search=&Python=true)
и выгрузить его по id (предпочтительно) или name.

## 🛠️ Архитектура

```
dmdslab/
├── datasets/              # Работа с датасетами
│   ├── ml_data_container.py    # Контейнеры ModelData, DataSplit
│   ├── uci/                    # UCI ML Repository интеграция
│   │   ├── uci_manager.py      # Основной менеджер
│   │   ├── uci_cache.py        # Система кеширования
│   │   ├── uci_metadata.py     # Работа с метаданными
│   │   └── uci_utils.py        # Вспомогательные функции
│   └── uci_dataset_manager.py  # База данных датасетов
└── scripts/              # Утилиты
    └── initialize_uci_database.py  # Инициализация БД
```

## 🔧 Настройка и конфигурация

### Настройка кеширования
```python
# Кастомная директория кеша
manager = UCIDatasetManager(cache_dir="./my_cache")

# Отключение кеша
manager = UCIDatasetManager(use_cache=False)

# Настройка логирования
manager = UCIDatasetManager(log_level="DEBUG")
```

### Работа без интернета
```python
# Загрузка только из кеша
manager = UCIDatasetManager(raise_on_missing=False)
data = manager.load_dataset(53)  # Вернет None если нет в кеше
```

## 🔬 Для исследователей

DmDSLab создан специально для исследователей в области машинного обучения:

- **Воспроизводимость**: Фиксированные random seeds и версионирование
- **Стандартизация**: Единый интерфейс для всех датасетов
- **Метаданные**: Полная информация о каждом датасете
- **Эффективность**: Кеширование экономит время и трафик

## 🚀 Разработка

```bash
# Клонирование репозитория
git clone https://github.com/Dmatryus/DmDSLab.git
cd DmDSLab

# Установка в режиме разработки
pip install -e .[dev,uci]

# Запуск тестов
pytest tests/

# Проверка стиля кода
ruff check dmdslab/
```

### Добавление новых UCI датасетов

1. Отредактируйте `scripts/initialize_uci_database.py`
2. Добавьте информацию о датасете
3. Запустите скрипт для обновления базы
4. Создайте PR с изменениями

## 📈 Производительность

- **Кеширование**: Повторная загрузка в 100+ раз быстрее
- **Оптимизация памяти**: Эффективное хранение больших датасетов
- **Параллельная загрузка**: В планах для версии 2.1

## 🤝 Вклад в проект

Мы приветствуем вклад от сообщества! См. [CONTRIBUTING.md](CONTRIBUTING.md) для деталей.

## 📄 Лицензия

Apache License 2.0. См. [LICENSE](LICENSE) для деталей.

## 👨‍💻 Автор

**Dmatryus Detry** - [GitHub](https://github.com/Dmatryus)
