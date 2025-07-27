# Структура эксперимента по сравнению методов выбора порогов для псевдо-разметки

## Оглавление
1. [Обзор эксперимента](#обзор-эксперимента)
2. [Общая архитектура](#общая-архитектура)
3. [Методы выбора порогов](#методы-выбора-порогов)
4. [Pipeline эксперимента](#pipeline-эксперимента)
5. [Структура данных](#структура-данных)
6. [Детальное описание модулей](#детальное-описание-модулей)
7. [План реализации](#план-реализации)

## Обзор эксперимента

### Цель
Сравнить эффективность различных методов выбора порогов вероятности для псевдо-разметки в задачах полуконтролируемого обучения на разных типах данных и моделях.

### Ключевые задачи
- Реализовать 12+ методов выбора порогов из руководства
- Протестировать на 3+ датасетах для каждого типа задач (бинарная/мультиклассовая)
- Использовать минимум 5 различных моделей классификации
- Создать comprehensive HTML отчет с анализом результатов

## Общая архитектура

```mermaid
graph TB
    subgraph "1. Подготовка данных"
        A[UCIDatasetManager] --> B[Загрузка датасетов]
        B --> C[Предобработка<br/>CatBoostEncoder<br/>SimpleImputer]
        C --> D[Разделение данных<br/>train/val/test/unlabeled]
    end
    
    subgraph "2. Базовое обучение"
        D --> E[CatBoostClassifier]
        D --> F[RandomForestClassifier]
        D --> G[ExtraTreesClassifier]
        D --> H[Другие модели]
    end
    
    subgraph "3. Методы порогов"
        E --> I[Бинарные методы]
        F --> I
        G --> I
        H --> I
        
        E --> J[Мультиклассовые методы]
        F --> J
        G --> J
        H --> J
        
        E --> K[Универсальные методы]
        F --> K
        G --> K
        H --> K
    end
    
    subgraph "4. Псевдо-разметка"
        I --> L[Отбор уверенных примеров]
        J --> L
        K --> L
        L --> M[Создание псевдо-меток]
        M --> N[Дообучение модели]
    end
    
    subgraph "5. Оценка и отчет"
        N --> O[Метрики качества]
        O --> P[Визуализация]
        P --> Q[HTML отчет]
    end
    
    style A fill:#e1f5fe
    style Q fill:#c8e6c9
```

## Методы выбора порогов

```mermaid
graph LR
    subgraph "Методы для бинарной классификации"
        B1[F1-оптимизация<br/>Максимизация F1-score]
        B2[Статистика Юдена<br/>TPR + TNR - 1]
        B3[Учет стоимости<br/>Cost-sensitive]
        B4[Precision-Recall<br/>баланс]
    end
    
    subgraph "Методы для мультиклассовой классификации"
        M1[Энтропийный<br/>H = -Σ p·log p]
        M2[Метод отрыва<br/>margin = p1 - p2]
        M3[Top-K уверенность<br/>sum top-k probs]
        M4[Температурная<br/>калибровка]
    end
    
    subgraph "Универсальные методы"
        U1[Процентильный<br/>quantile-based]
        U2[Фиксированный<br/>порог 0.9/0.95]
        U3[Адаптивный<br/>EMA-based]
        U4[MC Dropout<br/>uncertainty]
    end
    
    B1 --> E[Эксперимент]
    B2 --> E
    B3 --> E
    B4 --> E
    M1 --> E
    M2 --> E
    M3 --> E
    M4 --> E
    U1 --> E
    U2 --> E
    U3 --> E
    U4 --> E
    
    style B1 fill:#ffcdd2
    style B2 fill:#ffcdd2
    style B3 fill:#ffcdd2
    style B4 fill:#ffcdd2
    style M1 fill:#c5cae9
    style M2 fill:#c5cae9
    style M3 fill:#c5cae9
    style M4 fill:#c5cae9
    style U1 fill:#dcedc8
    style U2 fill:#dcedc8
    style U3 fill:#dcedc8
    style U4 fill:#dcedc8
```

### Детализация методов

#### Бинарная классификация
1. **F1-оптимизация**: Находит порог, максимизирующий F1-score на валидационной выборке
2. **Статистика Юдена**: Максимизирует сумму чувствительности и специфичности
3. **Учет стоимости**: Минимизирует взвешенную стоимость ошибок (FP × Cost_FP + FN × Cost_FN)
4. **Precision-Recall баланс**: Оптимизирует баланс между точностью и полнотой

#### Мультиклассовая классификация
1. **Энтропийный**: Отбирает примеры с низкой энтропией распределения вероятностей
2. **Метод отрыва**: Использует разность между top-2 вероятностями
3. **Top-K уверенность**: Суммирует вероятности K наиболее вероятных классов
4. **Температурная калибровка**: Применяет temperature scaling для улучшения калибровки

#### Универсальные методы
1. **Процентильный**: Выбирает порог на заданном процентиле распределения
2. **Фиксированный**: Использует предопределенные пороги (0.9, 0.95)
3. **Адаптивный**: Динамически корректирует порог через EMA
4. **MC Dropout**: Оценивает неопределенность через множественные прогоны

## Pipeline эксперимента

```mermaid
sequenceDiagram
    participant D as Датасет
    participant P as Препроцессор
    participant M as Модель
    participant T as Threshold Method
    participant PL as Псевдо-разметка
    participant E as Оценка
    participant R as Отчет
    
    Note over D,R: Цикл по датасетам
    D->>P: Загрузка данных
    P->>P: CatBoostEncoder + SimpleImputer
    P->>P: train_test_split (60/20/20)
    P->>P: Скрытие 80% меток от train
    
    Note over M,E: Цикл по моделям
    P->>M: Обучение на labeled data
    M->>M: Валидация и калибровка
    
    Note over T,PL: Цикл по методам порогов
    M->>T: Предсказания на unlabeled
    T->>T: Расчет порога
    T->>PL: Отбор уверенных примеров
    PL->>PL: Создание псевдо-меток
    PL->>M: Дообучение модели
    
    M->>E: Оценка на test set
    E->>E: Расчет метрик
    E->>R: Сохранение результатов
    
    Note over R: Генерация HTML отчета
    R->>R: Анализ по методам
    R->>R: Сравнительный анализ
    R->>R: Визуализация
```

## Структура данных

```mermaid
classDiagram
    class ExperimentResults {
        +dict dataset_results
        +dict model_results
        +dict method_results
        +datetime timestamp
        +save_results()
        +load_results()
    }
    
    class DatasetResult {
        +str dataset_name
        +str task_type
        +int n_samples
        +int n_features
        +int n_classes
        +dict class_distribution
        +dict preprocessing_params
    }
    
    class ModelResult {
        +str model_name
        +dict model_params
        +float baseline_score
        +dict calibration_metrics
        +float training_time
    }
    
    class MethodResult {
        +str method_name
        +str method_type
        +float optimal_threshold
        +int n_selected
        +float selection_ratio
        +dict performance_metrics
        +list threshold_history
    }
    
    class PerformanceMetrics {
        +float accuracy
        +float precision
        +float recall
        +float f1_score
        +float auc_roc
        +dict confusion_matrix
        +float improvement_ratio
    }
    
    ExperimentResults --> DatasetResult : contains
    ExperimentResults --> ModelResult : contains
    ExperimentResults --> MethodResult : contains
    MethodResult --> PerformanceMetrics : uses
```

## Детальное описание модулей

### 1. Модуль подготовки данных (`data_preparation.py`)

#### Функциональность
- Загрузка датасетов через UCIDatasetManager
- Автоматическая предобработка данных
- Создание сценария полуконтролируемого обучения

#### Датасеты
**Для бинарной классификации:**
- Breast Cancer Wisconsin
- Heart Disease
- Bank Marketing
- Adult Income
- Credit Card Default

**Для мультиклассовой классификации:**
- Iris
- Wine Quality
- Cover Type
- Letter Recognition
- Satellite Image

#### Предобработка
```python
# Псевдокод pipeline предобработки
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('encoder', CatBoostEncoder()),
    ('scaler', StandardScaler())  # опционально
])
```

### 2. Модуль базовых моделей (`base_models.py`)

#### Модели
1. **CatBoostClassifier**
   - Автоматическая обработка категориальных признаков
   - Встроенная регуляризация
   - GPU поддержка

2. **RandomForestClassifier**
   - Ансамблевый метод
   - Устойчивость к переобучению
   - Feature importance

3. **ExtraTreesClassifier**
   - Более случайный, чем RF
   - Быстрее в обучении
   - Хорошо для больших датасетов

4. **LightGBM**
   - Быстрое обучение
   - Эффективная работа с памятью
   - Хорошо для больших датасетов

5. **LogisticRegression**
   - Baseline модель
   - Интерпретируемость
   - Быстрое обучение

#### Калибровка вероятностей
- Platt Scaling для малых выборок
- Isotonic Regression для больших выборок
- Оценка калибровки через ECE (Expected Calibration Error)

### 3. Модуль методов выбора порогов (`threshold_methods.py`)

#### Базовая архитектура
```python
class ThresholdSelector(ABC):
    @abstractmethod
    def find_threshold(self, y_true, y_proba):
        pass
    
    @abstractmethod
    def select_samples(self, y_proba, threshold):
        pass
```

#### Реализации методов
Каждый метод имеет:
- Параметры инициализации
- Метод поиска оптимального порога
- Метод отбора примеров
- Визуализацию процесса выбора

### 4. Модуль псевдо-разметки (`pseudo_labeling.py`)

#### Стратегии
1. **Hard Pseudo-Labeling**
   - Присваивание жестких меток
   - Простая реализация
   - Может вносить шум

2. **Soft Pseudo-Labeling**
   - Использование вероятностей как весов
   - Более устойчиво к ошибкам
   - Требует поддержки от модели

3. **Iterative Pseudo-Labeling**
   - Постепенное добавление примеров
   - Адаптация порогов
   - Контроль качества на каждой итерации

### 5. Модуль оценки (`evaluation.py`)

#### Метрики
**Базовые метрики классификации:**
- Accuracy, Precision, Recall, F1
- AUC-ROC, AUC-PR
- Confusion Matrix
- Classification Report

**Метрики псевдо-разметки:**
- Coverage (процент отобранных примеров)
- Pseudo-label accuracy
- Confidence distribution
- Threshold stability

**Метрики улучшения:**
- Absolute improvement
- Relative improvement
- Statistical significance (paired t-test)

### 6. Модуль генерации отчетов (`report_generator.py`)

#### Структура HTML отчета

1. **Executive Summary**
   - Ключевые находки
   - Лучшие методы для каждого типа задач
   - Рекомендации

2. **Детальный анализ по датасетам**
   - Характеристики датасета
   - Результаты всех методов
   - Визуализации

3. **Анализ по моделям**
   - Сравнение baseline производительности
   - Влияние калибровки
   - Вычислительная эффективность

4. **Анализ методов выбора порогов**
   - Детальный анализ каждого метода
   - Сильные и слабые стороны
   - Рекомендации по применению

5. **Сравнительный анализ**
   - Heatmap результатов
   - Ranking методов
   - Statistical tests

6. **Интерактивные элементы**
   - Plotly графики
   - Фильтруемые таблицы
   - Drill-down возможности

### 7. Главный скрипт (`run_experiment.py`)

#### Конфигурация
```yaml
experiment_config:
  datasets:
    binary: [breast_cancer, heart_disease, bank_marketing]
    multiclass: [iris, wine, covertype]
  
  models:
    - name: CatBoost
      params: {iterations: 1000, early_stopping_rounds: 50}
    - name: RandomForest
      params: {n_estimators: 100, max_depth: 10}
  
  threshold_methods:
    binary: [f1_optimization, youden, cost_sensitive]
    multiclass: [entropy, margin, top_k]
    universal: [percentile, fixed, adaptive]
  
  pseudo_labeling:
    strategy: iterative
    max_iterations: 5
    
  output:
    save_intermediate: true
    report_format: html
    visualization_backend: plotly
```

## План реализации

### Этап 1: Базовая инфраструктура (текущий)
- [x] Проектирование архитектуры
- [x] Создание схем и документации
- [ ] Настройка окружения и зависимостей

### Этап 2: Подготовка данных
- [ ] Реализация data_preparation.py
- [ ] Загрузка и проверка датасетов
- [ ] Создание pipeline предобработки

### Этап 3: Основной функционал
- [ ] Реализация базовых моделей
- [ ] Реализация методов выбора порогов
- [ ] Модуль псевдо-разметки

### Этап 4: Анализ и отчетность
- [ ] Модуль оценки результатов
- [ ] Генератор HTML отчетов
- [ ] Финальный запуск эксперимента

## Ожидаемые результаты

1. **Сравнительная таблица** эффективности методов
2. **Рекомендации** по выбору метода для конкретных задач
3. **Визуализации** процесса отбора примеров
4. **Статистический анализ** значимости различий
5. **Практическое руководство** по применению методов

## Технологический стек

- **Python 3.8+**
- **Библиотеки ML**: scikit-learn, CatBoost, LightGBM
- **Обработка данных**: pandas, numpy
- **Визуализация**: matplotlib, seaborn, plotly
- **Отчеты**: Jinja2, HTML/CSS
- **Утилиты**: tqdm, joblib, pickle
