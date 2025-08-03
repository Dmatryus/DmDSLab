# 🚀 Pandas vs Polars Benchmark System

Комплексная система для сравнения производительности библиотек Pandas и Polars с автоматической генерацией интерактивных отчетов.

## ✨ Основные возможности

- 📊 **26 различных операций** для тестирования
- 🔄 **Автоматическая генерация** синтетических данных
- 📈 **Статистический анализ** с проверкой значимости
- 🎨 **Интерактивные визуализации** с Plotly
- 💾 **Система чекпоинтов** для восстановления после сбоев
- 📱 **Адаптивные HTML отчеты** с детальной аналитикой

## 🎯 Быстрый старт

### Вариант 1: Запуск в один клик

```bash
# Минимальный тест (2 минуты)
python one_click_benchmark.py --tiny

# Быстрый тест (5 минут) - по умолчанию
python one_click_benchmark.py

# Средний тест (30 минут)
python one_click_benchmark.py --medium

# Полный тест (2+ часа)
python one_click_benchmark.py --full
```

### Вариант 2: Стандартный запуск

```bash
# 1. Установка зависимостей
pip install -r requirements.txt

# 2. Запуск с конфигурацией по умолчанию
python scripts/run_benchmark.py --config configs/default_config.yaml

# 3. Просмотр отчета
open results/reports/benchmark_report_*.html  # macOS
# или
start results/reports/benchmark_report_*.html  # Windows
```

## 📋 Требования

- Python 3.11+
- 8 GB RAM (рекомендуется 16 GB)
- 10 GB свободного места на диске

## 🧪 Тестируемые операции

### IO операции
- Чтение/запись CSV
- Чтение/запись Parquet

### Фильтрация данных
- Простая фильтрация
- Сложные условия
- Фильтрация по списку значений
- Поиск по паттерну

### Группировка и агрегация
- Группировка по одной колонке
- Группировка по нескольким колонкам
- Множественные агрегации
- Оконные функции

### Сортировка
- По одной колонке
- По нескольким колонкам
- Кастомная сортировка

### Соединения (Joins)
- Inner join
- Left join
- Соединение по нескольким ключам
- As-of join

### Строковые операции
- Поиск подстроки
- Замена
- Извлечение по регулярным выражениям
- Конкатенация

## 📊 Пример результатов

```
┌─────────────────┬──────────┬──────────┬──────────┬─────────────┐
│ Операция        │ Pandas   │ Polars   │ Speedup  │ Значимость  │
├─────────────────┼──────────┼──────────┼──────────┼─────────────┤
│ read_csv        │ 2.45s    │ 0.89s    │ 2.8x     │ ***         │
│ groupby_agg     │ 1.23s    │ 0.31s    │ 4.0x     │ ***         │
│ complex_filter  │ 0.67s    │ 0.15s    │ 4.5x     │ ***         │
│ string_ops      │ 3.12s    │ 0.94s    │ 3.3x     │ ***         │
└─────────────────┴──────────┴──────────┴──────────┴─────────────┘
```

## 🔧 Настройка

### Создание собственной конфигурации

```yaml
# configs/my_config.yaml
benchmark:
  name: "My Custom Benchmark"

data_generation:
  sizes: [10000, 100000, 1000000]
  datasets:
    numeric:
      enabled: true
      columns: 20
    string:
      enabled: true
      columns: 10

operations:
  # Выберите нужные операции
  io: ["read_csv", "read_parquet"]
  filter: ["simple_filter", "complex_filter"]
  groupby: ["single_column_groupby", "multi_aggregation"]

profiling:
  min_runs: 5
  max_runs: 20
  target_cv: 0.05  # Коэффициент вариации
```

### Запуск с кастомной конфигурацией

```bash
python scripts/run_benchmark.py --config configs/my_config.yaml
```

## 🔄 Возобновление после прерывания

```bash
# Автоматически найдет последний чекпоинт
python scripts/run_benchmark.py --resume
```

## 📁 Структура результатов

```
results/
├── reports/
│   └── benchmark_report_20240115_143022.html  # Интерактивный отчет
├── raw_results/
│   ├── benchmark_results.json                  # Сырые данные
│   └── benchmark_results.csv                   # Табличный формат
├── figures/                                    # Отдельные графики
└── checkpoints/                                # Для восстановления
```

## 🤝 Вклад в проект

1. Fork репозитория
2. Создайте feature branch (`git checkout -b feature/amazing-feature`)
3. Commit изменения (`git commit -m 'Add amazing feature'`)
4. Push в branch (`git push origin feature/amazing-feature`)
5. Откройте Pull Request

## 📝 Лицензия

MIT License - см. файл LICENSE для деталей.

## 🙏 Благодарности

- Команде Pandas за отличную библиотеку обработки данных
- Команде Polars за инновационный подход к производительности
- Plotly за потрясающие возможности визуализации

---

**Примечание**: Этот проект создан для объективного сравнения производительности. Обе библиотеки имеют свои сильные стороны и оптимальные сценарии использования.