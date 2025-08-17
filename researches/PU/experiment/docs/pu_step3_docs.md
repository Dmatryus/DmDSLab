# Эксперимент PU Learning - Шаг 3: Симуляция PU сценариев

## 📋 Обзор

Третий шаг в конвейере экспериментов PU Learning - **Симуляция PU сценариев**. Этот модуль преобразует полностью размеченные данные в реалистичные сценарии PU Learning, имитируя реальные условия, где доступны только положительные и неразмеченные примеры.

### Ключевые особенности:
- 🎲 **Множественные стратегии симуляции** - SCAR, SAR, Prior Shift
- 📊 **Контроль искажений** - настраиваемые параметры селекции
- 🔍 **Метрики качества** - оценка реалистичности симуляции
- 📈 **Сохранение ground truth** - для последующего сравнения

## 🏗️ Архитектура модуля

### Общая схема работы

```mermaid
graph TB
    subgraph Input["📥 Входные данные"]
        XTrain[/"X_train"/]
        XTest[/"X_test"/]
        YTrain[/"y_train (ground truth)"/]
        YTest[/"y_test (ground truth)"/]
        Config[/"Конфигурация симуляции"/]
    end
    
    subgraph Step3["🎲 Шаг 3: Симуляция PU"]
        style Step3 fill:#9C27B0,stroke:#7B1FA2,stroke-width:3px,color:#fff
        
        S1["1️⃣ Валидация входных данных"]
        S2["2️⃣ Выбор стратегии симуляции"]
        S3["3️⃣ Применение селекции к положительным"]
        S4["4️⃣ Создание PU меток"]
        S5["5️⃣ Вычисление метрик качества"]
        S6["6️⃣ Сохранение ground truth"]
        
        S1 --> S2 --> S3 --> S4 --> S5 --> S6
        
        subgraph Strategies["Стратегии симуляции"]
            SCAR["🎯 SCAR<br/>Случайная селекция"]
            SAR["🔧 SAR<br/>Селекция с bias"]
            Prior["📈 Prior Shift<br/>Временные изменения"]
        end
        
        S2 --> SCAR
        S2 --> SAR
        S2 --> Prior
    end
    
    subgraph Output["📤 Выходные данные"]
        YTrainPU["y_train_pu (P+U labels)"]
        YTestPU["y_test_pu (P+U labels)"]
        YTrainTrue["y_train_true (ground truth)"]
        YTestTrue["y_test_true (ground truth)"]
        Stats["📊 Статистика симуляции"]
    end
    
    Input --> Step3
    Step3 --> Output
    
    style Strategies fill:#E1BEE7,stroke:#9C27B0,stroke-width:2px
    style SCAR fill:#4CAF50,color:#fff
    style SAR fill:#FF9800,color:#fff  
    style Prior fill:#2196F3,color:#fff
```

### Интеграция с конвейером экспериментов

```mermaid
graph LR
    subgraph Previous["✅ Завершённые шаги"]
        style Previous fill:#E8F5E9,stroke:#4CAF50,stroke-width:2px
        S1["Шаг 1: Валидация"]
        S2["Шаг 2: Предобработка"]
        style S1 fill:#4CAF50,color:#fff
        style S2 fill:#4CAF50,color:#fff
    end
    
    subgraph Current["🎯 Текущий шаг"]
        style Current fill:#F3E5F5,stroke:#9C27B0,stroke-width:3px
        S3["Шаг 3: Симуляция PU"]
        style S3 fill:#9C27B0,color:#fff
    end
    
    subgraph Future["⏳ Будущие шаги"]
        style Future fill:#f9f9f9,stroke:#ddd,stroke-width:1px
        S4["Шаг 4: Методы"]
        S5["Шаг 5: Обучение"]
        S6["Шаг 6: Метрики"]
        S7["Шаг 7: Визуализация"]
        S8["Шаг 8: Отчёт"]
    end
    
    Data[/"Данные"/] --> S1
    S1 -.->|"Валидированные<br/>данные"| S2
    S2 -.->|"X_train, X_test<br/>y_train, y_test"| S3
    S3 -.->|"PU данные +<br/>ground truth"| S4
    S4 --> S5 --> S6 --> S7 --> S8
    S8 --> Report[/"HTML отчёт"/]
```

## 🎲 Стратегии симуляции

### 1. SCAR (Selected Completely At Random)

**Принцип**: Положительные примеры выбираются для разметки случайным образом с вероятностью α.

```mermaid
graph TB
    subgraph SCAR["🎯 SCAR Стратегия"]
        AllPos["Все положительные примеры<br/>P = {p₁, p₂, ..., pₙ}"]
        Random["Случайная селекция<br/>P(выбран | положительный) = α"]
        Labeled["Размеченные положительные<br/>P_labeled ⊆ P"]
        Unlabeled["Неразмеченные<br/>U = P_hidden ∪ N"]
        
        AllPos --> Random
        Random --> Labeled
        Random --> Unlabeled
    end
    
    subgraph Properties["Свойства SCAR"]
        Unbiased["✅ Несмещенная селекция"]
        Simple["✅ Простая реализация"]
        Baseline["✅ Хороший baseline"]
        Unrealistic["⚠️ Нереалистично для некоторых задач"]
    end
    
    style SCAR fill:#E8F5E9,stroke:#4CAF50,stroke-width:2px
    style Properties fill:#FFF3E0,stroke:#FF9800,stroke-width:2px
```

**Математическая формулировка**:
```
P(s = 1 | y = 1, x) = α
P(s = 1 | y = 0, x) = 0
```

### 2. SAR (Selected At Random with bias)

**Принцип**: Вероятность селекции зависит от признаков, моделируя реальные сценарии с систематическими смещениями.

```mermaid
graph TB
    subgraph SAR["🔧 SAR Стратегия"]
        AllPos["Все положительные примеры<br/>с признаками X"]
        FeatureBias["Вычисление bias на основе признаков<br/>score(x) = f(x₁, x₂, ..., xₖ)"]
        BiasedProb["Смещенная вероятность<br/>P(выбран | x) = α + β·(score(x) - 0.5)"]
        Selection["Селекция по смещенным вероятностям"]
        
        AllPos --> FeatureBias
        FeatureBias --> BiasedProb
        BiasedProb --> Selection
    end
    
    subgraph UseCases["Применения SAR"]
        Medical["🏥 Медицина<br/>Диагноз зависит от симптомов"]
        Finance["🏦 Финансы<br/>Обнаружение зависит от суммы"]
        Web["🌐 Веб<br/>Клики зависят от позиции"]
    end
    
    style SAR fill:#FFF3E0,stroke:#FF9800,stroke-width:2px
    style UseCases fill:#E3F2FD,stroke:#2196F3,stroke-width:2px
```

**Математическая формулировка**:
```
P(s = 1 | y = 1, x) = α + β · bias_function(x)
где bias_function(x) создаёт зависимость от признаков
```

### 3. Prior Shift

**Принцип**: Вероятность селекции меняется со временем или по другим факторам.

```mermaid
graph TB
    subgraph PriorShift["📈 Prior Shift Стратегия"]
        TimeOrder["Порядок примеров<br/>(временной или другой)"]
        ChangingProb["Изменяющаяся вероятность<br/>P(t) = α · shift_function(t)"]
        TemporalBias["Временное смещение<br/>Ранние vs поздние примеры"]
        
        TimeOrder --> ChangingProb
        ChangingProb --> TemporalBias
    end
    
    subgraph Examples["Примеры Prior Shift"]
        Dataset["📊 Изменение критериев<br/>сбора данных"]
        Technology["💻 Улучшение методов<br/>детекции"]
        Policy["📋 Изменение политик<br/>разметки"]
    end
    
    style PriorShift fill:#E3F2FD,stroke:#2196F3,stroke-width:2px
    style Examples fill:#F3E5F5,stroke:#9C27B0,stroke-width:2px
```

## 📊 Метрики качества симуляции

### Структура SimulationStatistics

```mermaid
classDiagram
    class SimulationStatistics {
        +int original_positive_count
        +int original_negative_count
        +float original_positive_ratio
        +str simulation_strategy
        +float alpha_value
        +int labeled_positive_count
        +int unlabeled_count
        +int hidden_positive_count
        +int hidden_negative_count
        +float simulated_positive_ratio
        +float hidden_positive_ratio
        +float label_completeness
        +float pu_bias_score
        +float kl_divergence
        +float wasserstein_distance
        +to_dict() Dict
    }
    
    note for SimulationStatistics "Содержит полную статистику\nо качестве симуляции"
```

### Ключевые метрики

```mermaid
graph LR
    subgraph QualityMetrics["🔍 Метрики качества"]
        Completeness["📊 Label Completeness<br/>labeled_pos / original_pos<br/>Показывает долю найденных положительных"]
        
        Bias["🎯 PU Bias Score<br/>0 = нет смещения<br/>1 = максимальное смещение"]
        
        KL["📏 KL Divergence<br/>Измеряет сдвиг распределения<br/>D_KL(P_true || P_observed)"]
        
        Wasserstein["📐 Wasserstein Distance<br/>|mean(y_true) - mean(y_pu)|<br/>Простая мера различия"]
    end
    
    subgraph Interpretation["💡 Интерпретация"]
        Good["✅ Хорошая симуляция<br/>• Completeness > 10%<br/>• KL divergence < 0.5<br/>• Bias score отражает задачу"]
        
        Poor["⚠️ Проблемная симуляция<br/>• Completeness < 5%<br/>• KL divergence > 1.0<br/>• Нереалистичное смещение"]
    end
    
    style QualityMetrics fill:#E8F5E9,stroke:#4CAF50,stroke-width:2px
    style Interpretation fill:#FFF3E0,stroke:#FF9800,stroke-width:2px
```

## 🔄 Workflow симуляции

```mermaid
flowchart TD
    Start([Начало]) --> LoadData[/"Загрузка предобработанных данных<br/>X_train, X_test, y_train, y_test"/]
    
    LoadData --> ValidateInputs{Валидация входных данных}
    ValidateInputs -->|Ошибка| Error1[❌ Некорректные данные<br/>Небинарные метки, отсутствие классов]
    ValidateInputs -->|Успех| CheckAlpha{Проверка α}
    
    CheckAlpha -->|α ∉ (0,1]| Error2[❌ Некорректный α]
    CheckAlpha -->|α ∈ (0,1]| WarnLowAlpha{α < 0.1?}
    
    WarnLowAlpha -->|Да| Warn1[⚠️ Предупреждение:<br/>Мало размеченных примеров]
    WarnLowAlpha -->|Нет| SelectStrategy
    Warn1 --> SelectStrategy[Выбор стратегии симуляции]
    
    SelectStrategy --> SCAR{SCAR?}
    SelectStrategy --> SAR{SAR?}
    SelectStrategy --> PriorShift{Prior Shift?}
    
    SCAR -->|Да| RandomSelection[🎯 Случайная селекция<br/>np.random.choice(pos_indices, α)]
    SAR -->|Да| BiasedSelection[🔧 Селекция с bias<br/>Вероятность зависит от признаков]
    PriorShift -->|Да| TemporalSelection[📈 Временная селекция<br/>Изменяющаяся вероятность]
    
    RandomSelection --> CreatePULabels
    BiasedSelection --> CreatePULabels
    TemporalSelection --> CreatePULabels[Создание PU меток]
    
    CreatePULabels --> ApplyTrain[Применение к train set]
    CreatePULabels --> ApplyTest[Применение к test set]
    
    ApplyTrain --> CalcStats
    ApplyTest --> CalcStats[Вычисление статистики качества]
    
    CalcStats --> SaveGroundTruth[Сохранение ground truth]
    SaveGroundTruth --> Output([Вывод:<br/>y_train_pu, y_test_pu,<br/>y_train_true, y_test_true,<br/>statistics])
    
    Error1 --> End([Конец])
    Error2 --> End
    
    style Start fill:#4CAF50,color:#fff
    style Output fill:#2196F3,color:#fff
    style SelectStrategy fill:#9C27B0,color:#fff
    style RandomSelection fill:#4CAF50,color:#fff
    style BiasedSelection fill:#FF9800,color:#fff
    style TemporalSelection fill:#2196F3,color:#fff
    style Error1,Error2 fill:#F44336,color:#fff
    style Warn1 fill:#FF9800,color:#fff
```

## 🎯 Реалистичные сценарии PU Learning

### Примеры из реальных доменов

```mermaid
graph TB
    subgraph Domains["🌍 Реальные домены PU Learning"]
        
        subgraph Fraud["🏦 Детекция мошенничества"]
            F1["Обнаруженное мошенничество: 0.1-1%"]
            F2["α = 0.15 (15% случаев выявляются)"]
            F3["SAR: зависит от суммы и паттернов"]
        end
        
        subgraph Medical["🏥 Медицинская диагностика"]
            M1["Редкие заболевания: 0.01-0.5%"]
            M2["α = 0.4 (40% диагностируются)"]
            M3["SAR: зависит от симптомов"]
        end
        
        subgraph Web["🌐 Веб-аналитика"]
            W1["Конверсии: 1-5%"]
            W2["α = 0.6 (60% отслеживается)"]
            W3["Prior Shift: изменения трекинга"]
        end
        
        subgraph Drug["💊 Поиск лекарств"]
            D1["Активные соединения: 0.001%"]
            D2["α = 0.8 (80% находится скринингом)"]
            D3["SCAR: случайный скрининг"]
        end
    end
    
    style Fraud fill:#FFEBEE,stroke:#F44336,stroke-width:2px
    style Medical fill:#E8F5E9,stroke:#4CAF50,stroke-width:2px
    style Web fill:#E3F2FD,stroke:#2196F3,stroke-width:2px
    style Drug fill:#FFF3E0,stroke:#FF9800,stroke-width:2px
```

### Рекомендации по выбору параметров

```mermaid
graph LR
    subgraph Guidelines["📋 Руководство по параметрам"]
        
        subgraph AlphaChoice["🎯 Выбор α"]
            A1["α = 0.1-0.3: Сложные задачи<br/>(редкие события, плохая детекция)"]
            A2["α = 0.3-0.6: Умеренные задачи<br/>(типичные бизнес-сценарии)"]
            A3["α = 0.6-0.9: Простые задачи<br/>(хорошие методы детекции)"]
        end
        
        subgraph StrategyChoice["🔧 Выбор стратегии"]
            S1["SCAR: Baseline, несмещенные тесты"]
            S2["SAR: Реалистичные сценарии<br/>с feature-dependent bias"]
            S3["Prior Shift: Временные изменения<br/>в процессах сбора данных"]
        end
    end
    
    style AlphaChoice fill:#E8F5E9,stroke:#4CAF50,stroke-width:2px
    style StrategyChoice fill:#FFF3E0,stroke:#FF9800,stroke-width:2px
```

## 💻 Примеры использования

### Базовое использование

```python
from pu_scenario_simulator import simulate_pu_scenario

# После предобработки данных
y_train_pu, y_test_pu, y_train_true, y_test_true, stats = simulate_pu_scenario(
    X_train, X_test, y_train, y_test,
    alpha=0.3,  # 30% положительных размечены
    strategy='scar'  # Случайная селекция
)

print(stats)  # Просмотр статистики симуляции
```

### Пользовательская конфигурация

```python
config = {
    'sar_bias_strength': 0.8,  # Сильное смещение для SAR
    'sar_feature_indices': [0, 1, 2],  # Признаки для bias
    'min_labeled_positive': 10,  # Минимум размеченных
    'random_state': 42
}

y_train_pu, y_test_pu, y_train_true, y_test_true, stats = simulate_pu_scenario(
    X_train, X_test, y_train, y_test,
    alpha=0.4,
    strategy='sar',  # Смещенная селекция
    config=config
)
```

### Сравнение стратегий

```python
strategies = ['scar', 'sar', 'prior_shift']
results = {}

for strategy in strategies:
    y_train_pu, y_test_pu, y_train_true, y_test_true, stats = simulate_pu_scenario(
        X_train, X_test, y_train, y_test,
        alpha=0.3,
        strategy=strategy
    )
    results[strategy] = stats

# Сравнение bias scores
for strategy, stats in results.items():
    print(f"{strategy}: bias = {stats.pu_bias_score:.3f}, KL = {stats.kl_divergence:.4f}")
```

## 🔍 Анализ качества симуляции

### Проверка реалистичности

```mermaid
graph TB
    subgraph Analysis["🔍 Анализ качества"]
        
        subgraph Checks["Проверки"]
            C1["✅ Label completeness > 5%<br/>Достаточно размеченных примеров"]
            C2["✅ KL divergence < 1.0<br/>Умеренный сдвиг распределения"]
            C3["✅ Hidden positive ratio > 0<br/>Есть скрытые положительные"]
            C4["✅ Bias score соответствует стратегии<br/>SCAR: low, SAR: medium-high"]
        end
        
        subgraph Warnings["Предупреждения"]
            W1["⚠️ Очень низкий α<br/>Может быть мало данных для обучения"]
            W2["⚠️ Высокая KL divergence<br/>Слишком большое искажение"]
            W3["⚠️ Экстремальные значения bias<br/>Нереалистичная селекция"]
        end
    end
    
    style Checks fill:#E8F5E9,stroke:#4CAF50,stroke-width:2px
    style Warnings fill:#FFF3E0,stroke:#FF9800,stroke-width:2px
```

## ✅ Контрольный список

### Основная функциональность
- [x] Реализация стратегии SCAR
- [x] Реализация стратегии SAR с feature bias
- [x] Реализация стратегии Prior Shift
- [x] Валидация входных параметров
- [x] Вычисление метрик качества симуляции
- [x] Сохранение ground truth для оценки

### Специфика PU Learning
- [x] Обработка экстремально низких α значений
- [x] Предупреждения о недостаточном количестве примеров
- [x] Метрики для оценки реалистичности симуляции
- [x] Поддержка различных уровней смещения

### Метрики и статистика
- [x] Label completeness (полнота разметки)
- [x] PU bias score (мера смещения селекции)
- [x] KL divergence (сдвиг распределения)
- [x] Wasserstein distance (альтернативная мера)
- [x] Подробная статистика по скрытым классам

### Демонстрация и примеры
- [x] Демо базовых стратегий симуляции
- [x] Сравнение SCAR vs SAR vs Prior Shift
- [x] Анализ влияния α параметра
- [x] Реалистичные сценарии из разных доменов
- [x] Объяснение метрик качества

## 📁 Созданные файлы

- `researches/PU/experiment/pu_scenario_simulator.py` - Основной модуль симуляции
- `researches/PU/experiment/demo/pu_demo_simulation.py` - Демонстрационный скрипт
- `researches/PU/experiment/docs/pu_step3_docs.md` - Этот файл документации

---

*Следующий шаг: [Шаг 4 - Инициализация методов обучения →](step4_methods_initialization.md)*