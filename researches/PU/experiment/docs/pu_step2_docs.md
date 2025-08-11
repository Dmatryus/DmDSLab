# Эксперимент PU Learning - Шаг 2: Предобработка данных

## 📋 Обзор

Второй шаг в конвейере экспериментов PU Learning - **Предобработка данных**. Этот модуль преобразует валидированные данные в формат, готовый для машинного обучения, с особым вниманием к специфике PU Learning задач.

### Ключевые особенности:
- 🎯 **Стратифицированное разделение** - сохранение редких положительных примеров
- 🔧 **Интеллектуальная обработка пропусков** - разные стратегии для разных типов
- 🏷️ **Автоматическое определение типов признаков** - численные vs категориальные
- 📦 **Создание переиспользуемого пайплайна** - для применения к новым данным

## 🏗️ Архитектура модуля

### Общая схема работы

```mermaid
graph TB
    subgraph Input["📥 Входные данные"]
        DF[/"Валидированный DataFrame"/]
        Config[/"Конфигурация"/]
        Target[/"Имя целевой переменной"/]
    end
    
    subgraph Step2["🔧 Шаг 2: Предобработка"]
        S1["1️⃣ Разделение X и y"]
        S2["2️⃣ Удаление константных признаков"]
        S3["3️⃣ Определение типов признаков"]
        S4["4️⃣ Train/Test split со стратификацией"]
        S5["5️⃣ Обработка пропущенных значений"]
        S6["6️⃣ Кодирование категориальных"]
        S7["7️⃣ Масштабирование числовых"]
        S8["8️⃣ Сбор статистики"]
        
        S1 --> S2 --> S3 --> S4
        S4 --> S5 --> S6 --> S7 --> S8
    end
    
    subgraph Output["📤 Выходные данные"]
        XTrain["X_train"]
        XTest["X_test"]
        YTrain["y_train"]
        YTest["y_test"]
        Stats["📊 Статистика"]
        Pipeline["🔄 Pipeline"]
    end
    
    Input --> Step2
    Step2 --> Output
    
    style Step2 fill:#2C5282,stroke:#5BA0E5,stroke-width:3px,color:#fff
    style S4 fill:#FF9800,stroke:#F57C00,color:#fff
    style Pipeline fill:#4CAF50,stroke:#388E3C,color:#fff
```

### Интеграция с конвейером экспериментов

```mermaid
graph LR
    subgraph Previous["✅ Завершённые шаги"]
        style Previous fill:#E8F5E9,stroke:#4CAF50,stroke-width:2px
        S1["Шаг 1: Валидация"]
        style S1 fill:#4CAF50,color:#fff
    end
    
    subgraph Current["🎯 Текущий шаг"]
        style Current fill:#FFF3E0,stroke:#FF9800,stroke-width:3px
        S2["Шаг 2: Предобработка"]
        style S2 fill:#FF9800,color:#fff
    end
    
    subgraph Future["⏳ Будущие шаги"]
        style Future fill:#f9f9f9,stroke:#ddd,stroke-width:1px
        S3["Шаг 3: Симуляция PU"]
        S4["Шаг 4: Методы"]
        S5["Шаг 5: Обучение"]
        S6["Шаг 6: Метрики"]
        S7["Шаг 7: Визуализация"]
        S8["Шаг 8: Отчёт"]
    end
    
    Data[/"Данные"/] --> S1
    S1 -.->|"Валидированные<br/>данные"| S2
    S2 -.->|"X_train, X_test<br/>y_train, y_test"| S3
    S3 --> S4 --> S5 --> S6 --> S7 --> S8
    S8 --> Report[/"HTML отчёт"/]
```

## 🔧 Детали реализации

### 1. Определение типов признаков

```mermaid
graph TD
    Feature[/"Признак"/] --> CheckDtype{Проверка dtype}
    
    CheckDtype -->|Object| Cat1["🏷️ Категориальный"]
    
    CheckDtype -->|Numerical| CheckCardinality{Уникальных < 10<br/>И < 5% от данных?}
    CheckCardinality -->|Да| Cat2["🏷️ Категориальный<br/>(закодированный числами)"]
    CheckCardinality -->|Нет| Num["📊 Числовой"]
    
    Cat1 --> CheckHighCard{Уникальных > 50?}
    Cat2 --> CheckHighCard
    CheckHighCard -->|Да| Warning["⚠️ Высокая кардинальность"]
    CheckHighCard -->|Нет| OK["✅ Нормальная кардинальность"]
    
    style Cat1,Cat2 fill:#9C27B0,color:#fff
    style Num fill:#2196F3,color:#fff
    style Warning fill:#FF9800,color:#fff
    style OK fill:#4CAF50,color:#fff
```

### 2. Стратегия обработки пропущенных значений

```mermaid
graph LR
    subgraph Missing["🔍 Обработка пропусков"]
        subgraph Numerical["Числовые признаки"]
            N1["Медиана (default)"]
            N2["Среднее"]
            N3["Константа"]
        end
        
        subgraph Categorical["Категориальные признаки"]
            C1["Константа 'missing' (default)"]
            C2["Наиболее частое"]
            C3["Специальная категория"]
        end
    end
    
    style Numerical fill:#2196F3,color:#fff
    style Categorical fill:#9C27B0,color:#fff
    style N1,C1 fill:#4CAF50,color:#fff
```

### 3. Методы кодирования категориальных признаков

```mermaid
graph TD
    Cat[/"Категориальный признак"/] --> Method{Метод кодирования}
    
    Method -->|Target Encoding| TE["🎯 Target Encoding<br/>Лучше для PU Learning<br/>Обрабатывает высокую кардинальность"]
    Method -->|Label Encoding| LE["🏷️ Label Encoding<br/>Простой порядковый<br/>Быстрый"]
    Method -->|One-Hot| OH["🔥 One-Hot Encoding<br/>Создаёт много признаков<br/>Осторожно с кардинальностью"]
    
    TE --> Smoothing["Сглаживание для<br/>предотвращения переобучения"]
    
    style TE fill:#4CAF50,color:#fff
    style LE fill:#2196F3,color:#fff
    style OH fill:#FF9800,color:#fff
```

### 4. Стратифицированное разделение для PU Learning

```mermaid
graph TB
    subgraph Problem["❗ Проблема экстремального дисбаланса"]
        Original["Исходные данные<br/>1000 примеров<br/>10 положительных (1%)"]
    end
    
    subgraph RandomSplit["❌ Случайное разделение"]
        RTrain["Train: 800<br/>Положительных: 8 😊"]
        RTest["Test: 200<br/>Положительных: 2 😰"]
        RTestBad["Или хуже:<br/>Test: 200<br/>Положительных: 0 ☠️"]
    end
    
    subgraph StratifiedSplit["✅ Стратифицированное разделение"]
        STrain["Train: 800<br/>Положительных: 8 ✅<br/>(1% сохранён)"]
        STest["Test: 200<br/>Положительных: 2 ✅<br/>(1% сохранён)"]
    end
    
    Original --> RandomSplit
    Original --> StratifiedSplit
    
    RTrain -.->|"Риск потери<br/>положительных"| RTestBad
    
    style Problem fill:#FFEBEE,stroke:#F44336
    style RandomSplit fill:#FFF3E0,stroke:#FF9800
    style StratifiedSplit fill:#E8F5E9,stroke:#4CAF50
    style RTestBad fill:#F44336,color:#fff
```

## 📊 Структуры данных

### PreprocessingStatistics
Содержит полную информацию о процессе предобработки:

```mermaid
classDiagram
    class PreprocessingStatistics {
        +int train_size
        +int test_size
        +float train_positive_ratio
        +float test_positive_ratio
        +bool missing_handled
        +Dict missing_strategies
        +int n_numerical_features
        +int n_categorical_features
        +List categorical_features
        +List numerical_features
        +str categorical_encoding_method
        +bool scaling_applied
        +Dict scaling_stats
        +List constant_features_removed
        +List high_cardinality_warnings
        +to_dict() Dict
    }
```

### PreprocessingPipeline
Переиспользуемый пайплайн для новых данных:

```mermaid
classDiagram
    class PreprocessingPipeline {
        +Dict imputers
        +StandardScaler scaler
        +Dict encoders
        +List feature_order
        +List categorical_features
        +List numerical_features
        +transform(X) DataFrame
    }
```

## 🔄 Workflow предобработки

```mermaid
flowchart TD
    Start([Начало]) --> LoadData[/"Загрузка валидированных данных"/]
    
    LoadData --> SeparateXY[Разделение X и y]
    
    SeparateXY --> RemoveConstant{Удалять<br/>константные?}
    RemoveConstant -->|Да| RemoveConst[Удаление константных признаков]
    RemoveConstant -->|Нет| IdentifyTypes
    RemoveConst --> IdentifyTypes[Определение типов признаков]
    
    IdentifyTypes --> DetectCat["🏷️ Категориальные"]
    IdentifyTypes --> DetectNum["📊 Числовые"]
    
    DetectCat --> SplitData
    DetectNum --> SplitData[Train/Test split<br/>со стратификацией]
    
    SplitData --> CheckBalance{Баланс<br/>сохранён?}
    CheckBalance -->|Нет| WarnBalance[⚠️ Предупреждение]
    CheckBalance -->|Да| HandleMissing
    WarnBalance --> HandleMissing
    
    HandleMissing[Обработка пропусков] --> ImputeNum[Импутация числовых<br/>медиана/среднее]
    HandleMissing --> ImputeCat[Импутация категориальных<br/>константа/мода]
    
    ImputeNum --> Encode
    ImputeCat --> Encode[Кодирование категориальных]
    
    Encode --> EncMethod{Метод?}
    EncMethod -->|Target| TargetEnc[Target Encoding<br/>с smoothing]
    EncMethod -->|Label| LabelEnc[Label Encoding]
    EncMethod -->|OneHot| OneHotEnc[One-Hot Encoding]
    
    TargetEnc --> Scale
    LabelEnc --> Scale
    OneHotEnc --> Scale[Масштабирование числовых]
    
    Scale --> ScaleMethod{Масштабировать?}
    ScaleMethod -->|Да| StandardScale[StandardScaler<br/>mean=0, std=1]
    ScaleMethod -->|Нет| CollectStats
    
    StandardScale --> CollectStats[Сбор статистики]
    CollectStats --> CreatePipeline[Создание Pipeline]
    
    CreatePipeline --> Output([Вывод:<br/>X_train, X_test,<br/>y_train, y_test,<br/>stats, pipeline])
    
    style Start fill:#4CAF50,color:#fff
    style Output fill:#2196F3,color:#fff
    style SplitData fill:#FF9800,color:#fff
    style TargetEnc fill:#9C27B0,color:#fff
    style CreatePipeline fill:#4CAF50,color:#fff
```

## 💻 Примеры использования

### Базовое использование
```python
from pu_data_preprocessor import preprocess_data

# После валидации данных
X_train, X_test, y_train, y_test, stats, pipeline = preprocess_data(
    validated_df, 
    target_column='target'
)

print(stats)  # Просмотр статистики предобработки
```

### Пользовательская конфигурация
```python
config = {
    'test_size': 0.3,  # 30% на тест
    'scale_features': True,  # Масштабирование
    'encoding_method': 'target',  # Target encoding для PU
    'numerical_impute_strategy': 'median',
    'high_cardinality_threshold': 100
}

X_train, X_test, y_train, y_test, stats, pipeline = preprocess_data(
    validated_df, 'target', config
)
```

### Применение пайплайна к новым данным
```python
# Сохранённый пайплайн можно применить к новым данным
new_data = pd.read_csv('new_batch.csv')
X_new = new_data.drop(columns=['target'])

# Применение того же преобразования
X_new_transformed = pipeline.transform(X_new)
```

## 🎯 Особенности для PU Learning

### 1. Сохранение редких положительных примеров

```mermaid
graph LR
    subgraph Challenge["🎯 Вызов PU Learning"]
        Rare["Очень мало положительных<br/>примеров (< 1%)"]
    end
    
    subgraph Solution["✅ Решение"]
        Strat["Стратифицированное<br/>разделение"]
        Target["Target Encoding<br/>для категориальных"]
        Balance["Сохранение баланса<br/>в train и test"]
    end
    
    Rare --> Strat
    Rare --> Target
    Strat --> Balance
    
    style Challenge fill:#FFEBEE,stroke:#F44336
    style Solution fill:#E8F5E9,stroke:#4CAF50
```

### 2. Обработка высокой кардинальности

PU Learning датасеты часто имеют категориальные признаки с высокой кардинальностью (ID пользователей, продуктов и т.д.):

- **Target Encoding** - рекомендуется для PU Learning
- **Сглаживание** - предотвращает переобучение на редких категориях
- **Предупреждения** - для признаков с > 50 уникальными значениями

## ✅ Контрольный список

### Основная функциональность
- [x] Разделение признаков и целевой переменной
- [x] Автоматическое определение типов признаков
- [x] Стратифицированное train/test разделение
- [x] Обработка пропущенных значений
- [x] Кодирование категориальных признаков
- [x] Масштабирование числовых признаков
- [x] Удаление константных признаков

### Специфика PU Learning
- [x] Сохранение редких положительных примеров
- [x] Target encoding для высокой кардинальности
- [x] Проверка сохранения баланса классов
- [x] Предупреждения о потенциальных проблемах

### Инфраструктура
- [x] Сохранение пайплайна для новых данных
- [x] Детальная статистика предобработки
- [x] Конфигурируемые параметры
- [x] Обработка ошибок и предупреждений

### Демонстрация
- [x] Базовая предобработка
- [x] Пользовательская конфигурация
- [x] Обработка высокой кардинальности
- [x] Применение пайплайна
- [x] Экстремальные PU сценарии
- [x] Определение типов признаков

## 📁 Созданные файлы

- `researches/PU/experiment/pu_data_preprocessor.py` - Основной модуль предобработки
- `researches/PU/experiment/demo/pu_demo_preprocessing.py` - Демонстрационный скрипт
- `researches/PU/experiment/docs/pu_step2_docs.md` - Этот файл документации

---

*Следующий шаг: [Шаг 3 - Симуляция PU сценария →](step3_pu_simulation.md)*