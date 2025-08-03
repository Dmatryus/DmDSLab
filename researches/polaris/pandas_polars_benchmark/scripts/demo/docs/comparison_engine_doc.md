# ComparisonEngine - Движок сравнения результатов

## Обзор

ComparisonEngine - это модуль для статистического сравнения результатов бенчмарков между двумя библиотеками (Pandas и Polars). Он предоставляет комплексный анализ с использованием различных статистических методов.

## Архитектура

```mermaid
graph TD
    subgraph "Comparison Engine"
        A[ComparisonEngine] --> B[Statistical Tests]
        A --> C[Performance Metrics]
        A --> D[Result Export]
        
        B --> B1[T-Test Welch]
        B --> B2[Mann-Whitney U]
        B --> B3[Cohen's d]
        
        C --> C1[Relative Improvement]
        C --> C2[Speedup Factor]
        C --> C3[Confidence Intervals]
        
        D --> D1[JSON Export]
        D --> D2[CSV Export]
        D --> D3[Summary Report]
    end
    
    subgraph "Input Data"
        E[Benchmark Results] --> A
        F[Outlier Cleaned Data] --> A
    end
    
    subgraph "Output"
        A --> G[ComparisonResult]
        A --> H[ComparisonMatrix]
        H --> I[Summary Statistics]
    end
    
    style A fill:#f9f,stroke:#333,stroke-width:4px
```

## Основные компоненты

### 1. ComparisonMetric
Перечисление метрик для сравнения:
- `EXECUTION_TIME` - время выполнения
- `MEMORY_PEAK` - пиковое использование памяти
- `MEMORY_MEAN` - среднее использование памяти
- `CPU_USAGE` - использование CPU

### 2. SignificanceLevel
Уровни статистической значимости:
- `NOT_SIGNIFICANT` - p > 0.1
- `WEAKLY_SIGNIFICANT` - 0.05 < p < 0.1
- `SIGNIFICANT` - 0.01 < p < 0.05
- `HIGHLY_SIGNIFICANT` - p < 0.01

### 3. ComparisonResult
Результат сравнения двух выборок, содержащий:
- Базовые статистики (среднее, медиана, стандартное отклонение)
- Метрики производительности (относительное улучшение, фактор ускорения)
- Результаты статистических тестов
- Автоматическое определение победителя

### 4. ComparisonMatrix
Матрица сравнений для всех операций с агрегированной статистикой:
- Количество значимых различий
- Распределение побед между библиотеками
- Среднее и медианное улучшение

## Процесс сравнения

```mermaid
sequenceDiagram
    participant User
    participant Engine as ComparisonEngine
    participant Stats as Statistical Tests
    participant Export as Exporter
    
    User->>Engine: compare_two_samples(baseline, comparison)
    Engine->>Engine: Вычисление базовых статистик
    Engine->>Stats: T-test (Welch)
    Stats-->>Engine: t-statistic, p-value
    Engine->>Stats: Mann-Whitney U test
    Stats-->>Engine: U-statistic, p-value
    Engine->>Engine: Вычисление Cohen's d
    Engine->>Engine: Расчет доверительных интервалов
    Engine->>Engine: Определение уровня значимости
    Engine-->>User: ComparisonResult
    
    User->>Engine: compare_all_operations(results)
    loop Для каждой операции
        Engine->>Engine: compare_two_samples()
    end
    Engine->>Engine: Создание ComparisonMatrix
    Engine-->>User: ComparisonMatrix
    
    User->>Engine: export_results(matrix, format)
    Engine->>Export: Подготовка данных
    Export->>Export: Сохранение в файл
    Export-->>User: Путь к файлу
```

## Статистические методы

### 1. T-тест Уэлча
- Используется для сравнения средних двух выборок
- Не требует равенства дисперсий
- Предполагает нормальность распределения

### 2. Тест Манна-Уитни
- Непараметрический тест
- Не требует нормальности распределения
- Более робастный к выбросам

### 3. Размер эффекта (Cohen's d)
- Измеряет практическую значимость различий
- Интерпретация:
  - |d| < 0.2: незначительный эффект
  - 0.2 ≤ |d| < 0.5: малый эффект
  - 0.5 ≤ |d| < 0.8: средний эффект
  - |d| ≥ 0.8: большой эффект

## Пример использования

```python
# Создание движка
engine = ComparisonEngine(confidence_level=0.95)

# Сравнение одной операции
result = engine.compare_two_samples(
    baseline=pandas_times,
    comparison=polars_times,
    name="read_csv",
    baseline_library="pandas",
    comparison_library="polars"
)

# Сравнение всех операций
matrix = engine.compare_all_operations(benchmark_results)

# Экспорт результатов
engine.export_results(matrix, Path("results.json"), format="json")
```

## Интерпретация результатов

### Относительное улучшение
```
Improvement = ((baseline_mean - comparison_mean) / baseline_mean) × 100%
```
- Положительное значение: comparison быстрее baseline
- Отрицательное значение: baseline быстрее comparison

### Фактор ускорения
```
Speedup = baseline_mean / comparison_mean
```
- > 1: comparison быстрее в X раз
- < 1: baseline быстрее

### Определение победителя
1. Если p-value > 0.05: "tie" (нет значимой разницы)
2. Если p-value ≤ 0.05:
   - comparison_mean < baseline_mean: comparison побеждает
   - baseline_mean < comparison_mean: baseline побеждает

## Интеграция с бенчмарком

```mermaid
graph LR
    subgraph "Процесс анализа"
        A[Profiling Results] --> B[Outlier Detection]
        B --> C[Statistics Calculation]
        C --> D[Comparison Engine]
        D --> E[Report Generation]
    end
    
    style D fill:#f9f,stroke:#333,stroke-width:4px
```

ComparisonEngine является ключевым компонентом между статистическим анализом и генерацией отчетов:

1. **Входные данные**: Очищенные от выбросов результаты профилирования
2. **Обработка**: Статистическое сравнение и определение значимости
3. **Выходные данные**: Структурированные результаты для генерации отчетов

## Форматы экспорта

### JSON формат
```json
{
  "metadata": {
    "baseline_library": "pandas",
    "comparison_library": "polars",
    "metric": "execution_time"
  },
  "summary": {
    "total_operations": 10,
    "significant_differences": 7,
    "mean_improvement": 35.2
  },
  "detailed_results": {
    "operation_name": {
      "baseline_mean": 1.2,
      "comparison_mean": 0.3,
      "relative_improvement": 75.0,
      "p_value": 0.0001,
      "winner": "polars"
    }
  }
}
```

### CSV формат
```csv
operation,baseline_mean,comparison_mean,relative_improvement_%,speedup_factor,p_value,significance,winner
read_csv,1.2,0.3,75.0,4.0,0.0001,highly_significant,polars
```

## Файлы и изменения

### Созданные файлы:
- `src/analysis/comparison_engine.py` - реализация движка сравнения
- `scripts/demo/demo_comparison_engine.py` - демонстрация работы
- `docs/comparison_engine_doc.md` - эта документация

### Интеграция:
- Модуль готов к интеграции с существующими компонентами анализа
- Следующий шаг: создание модуля генерации отчетов

## Рекомендации по использованию

1. **Минимальный размер выборки**: Рекомендуется минимум 10-15 измерений для надежных результатов
2. **Очистка от выбросов**: Всегда применяйте детекцию выбросов перед сравнением
3. **Множественные сравнения**: При сравнении многих операций учитывайте коррекцию Бонферрони
4. **Практическая значимость**: Обращайте внимание не только на p-value, но и на размер эффекта