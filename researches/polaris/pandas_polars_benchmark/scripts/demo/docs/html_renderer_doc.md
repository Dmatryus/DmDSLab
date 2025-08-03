# HTML Renderer - Модуль генерации финальных отчетов

## Обзор

HTML Renderer - завершающий компонент системы генерации отчетов, который объединяет все графики, таблицы и статистику в единый интерактивный HTML отчет. Модуль использует шаблонизатор Jinja2 и создает адаптивные, готовые к печати отчеты.

## Архитектура модуля

```mermaid
graph TD
    subgraph "HTML Renderer Module"
        A[HTMLRenderer] --> B[Report Sections]
        
        B --> B1[Summary Section]
        B --> B2[Methodology Section]
        B --> B3[Performance Section]
        B --> B4[Analysis Section]
        B --> B5[Distribution Section]
        B --> B6[Recommendations Section]
        
        A --> C[Template Engine]
        C --> C1[Default Template]
        C --> C2[Custom Templates]
        
        A --> D[Content Generation]
        D --> D1[Key Findings]
        D --> D2[Recommendations]
        D --> D3[Interpretations]
        
        A --> E[HTML Generation]
        E --> E1[Section Rendering]
        E --> E2[TOC Generation]
        E --> E3[Style Injection]
        
        F[Plotly Figures] --> A
        G[Summary Data] --> A
        H[Report Config] --> A
        
        A --> I[Final HTML Report]
    end
    
    style A fill:#f9f,stroke:#333,stroke-width:4px
    style I fill:#bbf,stroke:#333,stroke-width:2px
```

## Workflow генерации отчета

```mermaid
sequenceDiagram
    participant User
    participant HTMLRenderer
    participant VisualizationEngine
    participant DataProcessor
    participant Template
    
    User->>DataProcessor: Обработка результатов
    DataProcessor-->>User: ProcessedData
    
    User->>VisualizationEngine: Создание графиков
    VisualizationEngine-->>User: Plotly Figures
    
    User->>HTMLRenderer: render_report()
    activate HTMLRenderer
    
    HTMLRenderer->>HTMLRenderer: Создание секций
    loop Для каждой секции
        HTMLRenderer->>HTMLRenderer: Генерация контента
        HTMLRenderer->>HTMLRenderer: Форматирование
    end
    
    HTMLRenderer->>Template: Рендеринг HTML
    Template-->>HTMLRenderer: HTML строка
    
    HTMLRenderer->>HTMLRenderer: Сохранение файла
    HTMLRenderer-->>User: Путь к отчету
    deactivate HTMLRenderer
```

## Структура отчета

```mermaid
graph TD
    subgraph "HTML Report Structure"
        A[Header] --> A1[Title]
        A --> A2[Metadata]
        A --> A3[Generation Info]
        
        B[Table of Contents] --> B1[Section Links]
        
        C[Summary Section] --> C1[Key Metrics Cards]
        C --> C2[Key Findings List]
        
        D[Methodology Section] --> D1[System Info]
        D --> D2[Test Parameters]
        D --> D3[Methods Description]
        
        E[Performance Section] --> E1[Comparison Charts]
        E --> E2[Speedup Analysis]
        
        F[Analysis Section] --> F1[Heatmaps]
        F --> F2[Scaling Graphs]
        F --> F3[Observations]
        
        G[Distribution Section] --> G1[Box Plots]
        G --> G2[Statistical Tables]
        
        H[Recommendations] --> H1[When to use Polars]
        H --> H2[When to use Pandas]
        H --> H3[General Advice]
        
        I[Footer] --> I1[Generation Info]
    end
```

## Основные компоненты

### 1. ReportSection
Представляет отдельную секцию отчета:
- **title**: Заголовок секции
- **content**: HTML содержимое
- **section_id**: Уникальный идентификатор для навигации
- **order**: Порядок отображения

### 2. ReportConfig
Конфигурация для настройки отчета:
- **title**: Основной заголовок
- **subtitle**: Подзаголовок
- **author**: Автор отчета
- **description**: Описание
- **include_toc**: Включить оглавление
- **include_summary**: Включить сводку
- **include_methodology**: Включить методологию
- **include_recommendations**: Включить рекомендации
- **theme**: Тема оформления (light/dark)

### 3. Методы генерации контента

#### Key Findings Generator
```mermaid
graph LR
    A[Summary Data] --> B[Analyze Speedup]
    B --> C{Speedup Level}
    C -->|>2x| D[Significant Advantage]
    C -->|1.5-2x| E[Notable Advantage]
    C -->|1.1-1.5x| F[Moderate Advantage]
    C -->|<1.1x| G[Comparable Performance]
    
    H[Operation Analysis] --> I[Top Operations]
    I --> J[Generate Finding]
    
    D --> K[Key Findings List]
    E --> K
    F --> K
    G --> K
    J --> K
```

#### Recommendations Generator
Автоматически анализирует результаты и генерирует рекомендации по выбору библиотеки на основе:
- Среднего ускорения по операциям
- Специфичных сильных сторон каждой библиотеки
- Общих паттернов использования

## Стилизация и дизайн

### CSS структура
- **Адаптивный дизайн**: Mobile-first подход
- **Цветовая схема**: Настраиваемые CSS переменные
- **Печать**: Оптимизированные стили для печати
- **Доступность**: Семантическая разметка и контрастные цвета

### Интерактивные элементы
- Навигация по секциям через TOC
- Интерактивные Plotly графики
- Hover эффекты на карточках метрик
- Responsive таблицы

## Примеры использования

### Базовая генерация отчета
```python
renderer = HTMLRenderer()

html = renderer.render_report(
    figures={'chart1': fig1, 'chart2': fig2},
    summary_data=processed_data,
    output_path=Path("report.html")
)
```

### Кастомная конфигурация
```python
config = ReportConfig(
    title="Мой бенчмарк",
    subtitle="Детальный анализ",
    author="Data Team",
    include_toc=True,
    theme="dark"
)

renderer.render_report(
    figures=figures,
    summary_data=data,
    config=config,
    output_path=Path("custom_report.html")
)
```

### Использование custom шаблона
```python
renderer = HTMLRenderer(
    template_dir=Path("templates/")
)
# Поместите report.html в папку templates/
```

## Расширяемость

### Добавление новых секций
```python
renderer.sections.append(ReportSection(
    title="Дополнительный анализ",
    content="<p>Мой контент</p>",
    section_id="custom-analysis",
    order=7
))
```

### Кастомизация стилей
Можно переопределить CSS переменные в custom шаблоне:
```css
:root {
    --primary-color: #your-color;
    --secondary-color: #your-color;
}
```

## Выходной формат

### Структура HTML файла
- Standalone HTML с встроенными стилями
- CDN ссылки на Plotly.js
- Минимальные внешние зависимости
- Оптимизирован для быстрой загрузки

### Размер файла
- Базовый отчет: ~50-100 KB
- С графиками: 200-500 KB (зависит от количества)
- Сжатие: поддерживается gzip

## Интеграция с системой

```mermaid
graph LR
    subgraph "Complete Pipeline"
        A[Benchmark Results] --> B[Statistical Analysis]
        B --> C[Data Processor]
        C --> D[Visualization Engine]
        D --> E[HTML Renderer]
        E --> F[Final Report]
        
        G[User Config] --> E
        H[Custom Templates] --> E
    end
    
    style E fill:#f96,stroke:#333,stroke-width:4px
    style F fill:#9f9,stroke:#333,stroke-width:2px
```

## Файлы и изменения

### Созданные файлы:
- `src/reporting/html_renderer.py` - основной модуль рендерера
- `scripts/demo/demo_html_renderer.py` - демонстрация работы
- `docs/html_renderer_doc.md` - эта документация

### Обновленные файлы:
- `src/reporting/__init__.py` - добавлен экспорт HTMLRenderer

## Результат работы

HTML Renderer создает профессиональный, интерактивный отчет, который:
- **Информативен**: Содержит все ключевые метрики и выводы
- **Интерактивен**: Plotly графики с zoom, pan, hover
- **Адаптивен**: Хорошо выглядит на всех устройствах
- **Готов к печати**: Оптимизированные стили для печати
- **Самодостаточен**: Не требует внешних файлов

## Завершение фазы 5

С созданием HTML Renderer завершена фаза 5 "Анализ и отчетность". Теперь система может:
1. Анализировать результаты (Statistical Analysis)
2. Обрабатывать данные для визуализации (Data Processor)
3. Создавать интерактивные графики (Visualization Engine)
4. Генерировать профессиональные HTML отчеты (HTML Renderer)