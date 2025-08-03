#!/usr/bin/env python3
"""
Демонстрация работы модуля VisualizationEngine.
Создает различные типы интерактивных графиков с Plotly.
"""

import os
import sys
from pathlib import Path

from researches.polaris.pandas_polars_benchmark.src import get_logger

# Добавляем путь к src в PYTHONPATH
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
)

from reporting.data_processor import DataProcessor, MetricType, ProcessedData
from reporting.visualization_engine import VisualizationEngine, ChartConfig, ChartType

# Импортируем функцию создания примерных данных из demo_data_processor
from demo_data_processor import create_sample_results


def demonstrate_bar_charts():
    """Демонстрация создания столбчатых диаграмм."""
    logger = get_logger(__name__)
    logger.info("=" * 60)
    logger.info("1. СОЗДАНИЕ СТОЛБЧАТЫХ ДИАГРАММ")
    logger.info("=" * 60)

    # Подготовка данных
    df = create_sample_results()
    processor = DataProcessor()
    viz_engine = VisualizationEngine()

    # Подготовка данных для сравнения
    comparison_data = processor.prepare_comparison_data(
        df=df,
        metric=MetricType.EXECUTION_TIME,
        groupby=["operation"],
        pivot_column="library",
    )

    # Создание базового графика
    config = ChartConfig(
        title="Сравнение времени выполнения операций: Pandas vs Polars",
        x_label="Операция",
        y_label="Время выполнения (сек)",
        height=600,
    )

    fig = viz_engine.create_comparison_bar_chart(comparison_data, config)

    # Сохранение
    output_dir = Path("demo_outputs")
    output_dir.mkdir(exist_ok=True)

    viz_engine.export_figure(
        fig, output_dir / "comparison_bar_chart.html", format="html"
    )

    logger.info(f"График сохранен в {output_dir / 'comparison_bar_chart.html'}")

    # Создание графика с логарифмической шкалой
    config_log = ChartConfig(
        title="Сравнение производительности (логарифмическая шкала)",
        x_label="Операция",
        y_label="Время выполнения (сек, log scale)",
        height=600,
        log_scale=True,
    )

    fig_log = viz_engine.create_comparison_bar_chart(comparison_data, config_log)
    viz_engine.export_figure(
        fig_log, output_dir / "comparison_bar_chart_log.html", format="html"
    )


def demonstrate_line_charts():
    """Демонстрация создания линейных графиков."""
    logger = get_logger(__name__)
    logger.info("\n" + "=" * 60)
    logger.info("2. СОЗДАНИЕ ЛИНЕЙНЫХ ГРАФИКОВ")
    logger.info("=" * 60)

    # Подготовка данных
    df = create_sample_results()
    processor = DataProcessor()
    viz_engine = VisualizationEngine()

    # Подготовка данных для timeline
    timeline_data = processor.prepare_timeline_data(
        df=df, metric=MetricType.EXECUTION_TIME, dataset_size_column="dataset_size"
    )

    # Создание графика зависимости от размера
    config = ChartConfig(
        title="Зависимость времени выполнения от размера данных",
        x_label="Размер датасета",
        y_label="Время выполнения (сек)",
        height=700,
        log_scale=True,
    )

    fig = viz_engine.create_line_chart(timeline_data, config)

    # Сохранение
    output_dir = Path("demo_outputs")
    viz_engine.export_figure(
        fig, output_dir / "timeline_line_chart.html", format="html"
    )

    logger.info(f"График сохранен в {output_dir / 'timeline_line_chart.html'}")


def demonstrate_heatmaps():
    """Демонстрация создания тепловых карт."""
    logger = get_logger(__name__)
    logger.info("\n" + "=" * 60)
    logger.info("3. СОЗДАНИЕ ТЕПЛОВЫХ КАРТ")
    logger.info("=" * 60)

    # Подготовка данных
    df = create_sample_results()
    processor = DataProcessor()
    viz_engine = VisualizationEngine()

    # Подготовка данных для heatmap
    heatmap_data = processor.prepare_heatmap_data(
        df=df,
        metric=MetricType.EXECUTION_TIME,
        row_column="operation",
        col_column="dataset_size",
    )

    # Создание тепловой карты
    config = ChartConfig(
        title="Тепловая карта производительности по операциям и размерам данных",
        x_label="Размер датасета",
        y_label="Операция",
        height=600,
    )

    fig = viz_engine.create_heatmap(heatmap_data, config)

    # Сохранение
    output_dir = Path("demo_outputs")
    viz_engine.export_figure(
        fig, output_dir / "performance_heatmap.html", format="html"
    )

    logger.info(f"График сохранен в {output_dir / 'performance_heatmap.html'}")


def demonstrate_box_plots():
    """Демонстрация создания box plots."""
    logger = get_logger(__name__)
    logger.info("\n" + "=" * 60)
    logger.info("4. СОЗДАНИЕ BOX PLOTS")
    logger.info("=" * 60)

    # Подготовка данных
    df = create_sample_results()
    processor = DataProcessor()
    viz_engine = VisualizationEngine()

    # Для box plot нужны данные распределения
    dist_data = processor.prepare_distribution_data(
        df=df, metric=MetricType.EXECUTION_TIME, library="polars", operation="groupby"
    )

    # Создание box plot
    config = ChartConfig(
        title="Распределение времени выполнения операций",
        x_label="Библиотека",
        y_label="Время выполнения (сек)",
        height=600,
    )

    # Для демонстрации создадим данные для обеих библиотек
    combined_data = ProcessedData(
        data=df[df["operation"].isin(["groupby", "filter", "sort"])],
        metadata={"libraries": ["pandas", "polars"]},
        aggregation_level=dist_data.aggregation_level,
        metric_type=dist_data.metric_type,
    )

    fig = viz_engine.create_box_plot(combined_data, config)

    # Сохранение
    output_dir = Path("demo_outputs")
    viz_engine.export_figure(
        fig, output_dir / "distribution_box_plot.html", format="html"
    )

    logger.info(f"График сохранен в {output_dir / 'distribution_box_plot.html'}")


def demonstrate_performance_table():
    """Демонстрация создания интерактивных таблиц."""
    logger = get_logger(__name__)
    logger.info("\n" + "=" * 60)
    logger.info("5. СОЗДАНИЕ ИНТЕРАКТИВНЫХ ТАБЛИЦ")
    logger.info("=" * 60)

    # Подготовка данных
    df = create_sample_results()
    processor = DataProcessor()
    viz_engine = VisualizationEngine()

    # Создание сводной таблицы
    summary_df = processor.create_summary_table(df)

    summary_data = ProcessedData(
        data=summary_df,
        metadata={"total_records": len(df)},
        aggregation_level=processor.AggregationLevel.OPERATION,
        metric_type=MetricType.EXECUTION_TIME,
    )

    # Создание таблицы
    config = ChartConfig(
        title="Сводная таблица производительности", x_label="", y_label="", height=400
    )

    fig = viz_engine.create_performance_table(summary_data, config)

    # Сохранение
    output_dir = Path("demo_outputs")
    viz_engine.export_figure(fig, output_dir / "performance_table.html", format="html")

    logger.info(f"Таблица сохранена в {output_dir / 'performance_table.html'}")


def demonstrate_speedup_chart():
    """Демонстрация создания графика ускорения."""
    logger = get_logger(__name__)
    logger.info("\n" + "=" * 60)
    logger.info("6. СОЗДАНИЕ ГРАФИКА УСКОРЕНИЯ (SPEEDUP)")
    logger.info("=" * 60)

    # Подготовка данных
    df = create_sample_results()
    processor = DataProcessor()
    viz_engine = VisualizationEngine()

    # Подготовка данных с расчетом speedup
    speedup_data = processor.prepare_comparison_data(
        df=df, metric=MetricType.SPEEDUP, groupby=["operation"], pivot_column="library"
    )

    # Создание графика ускорения
    config = ChartConfig(
        title="Ускорение Polars относительно Pandas",
        x_label="Операция",
        y_label="Ускорение (раз)",
        height=600,
    )

    fig = viz_engine.create_speedup_chart(speedup_data, config)

    # Сохранение
    output_dir = Path("demo_outputs")
    viz_engine.export_figure(fig, output_dir / "speedup_chart.html", format="html")

    logger.info(f"График сохранен в {output_dir / 'speedup_chart.html'}")


def demonstrate_dashboard():
    """Демонстрация создания комплексного дашборда."""
    logger = get_logger(__name__)
    logger.info("\n" + "=" * 60)
    logger.info("7. СОЗДАНИЕ КОМПЛЕКСНОГО ДАШБОРДА")
    logger.info("=" * 60)

    # Подготовка данных
    df = create_sample_results()
    processor = DataProcessor()
    viz_engine = VisualizationEngine()

    # Подготовка различных типов данных
    processed_data = {
        "comparison": processor.prepare_comparison_data(
            df=df,
            metric=MetricType.EXECUTION_TIME,
            groupby=["operation"],
            pivot_column="library",
        ),
        "timeline": processor.prepare_timeline_data(
            df=df, metric=MetricType.EXECUTION_TIME, dataset_size_column="dataset_size"
        ),
        "distribution": ProcessedData(
            data=df,
            metadata={"libraries": ["pandas", "polars"]},
            aggregation_level=processor.AggregationLevel.OPERATION,
            metric_type=MetricType.EXECUTION_TIME,
        ),
        "speedup": processor.prepare_comparison_data(
            df=df,
            metric=MetricType.SPEEDUP,
            groupby=["operation"],
            pivot_column="library",
        ),
    }

    # Создание дашборда
    fig = viz_engine.create_dashboard(
        processed_data, title="Pandas vs Polars: Комплексный анализ производительности"
    )

    # Сохранение
    output_dir = Path("demo_outputs")
    viz_engine.export_figure(
        fig, output_dir / "performance_dashboard.html", format="html"
    )

    logger.info(f"Дашборд сохранен в {output_dir / 'performance_dashboard.html'}")


def main():
    """Основная функция для запуска всех демонстраций."""
    # Настройка логирования

    # Запуск всех демонстраций
    demonstrate_bar_charts()
    demonstrate_line_charts()
    demonstrate_heatmaps()
    demonstrate_box_plots()
    demonstrate_performance_table()
    demonstrate_speedup_chart()
    demonstrate_dashboard()


if __name__ == "__main__":
    main()
