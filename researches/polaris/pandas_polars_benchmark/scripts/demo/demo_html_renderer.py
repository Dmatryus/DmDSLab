#!/usr/bin/env python3
"""
Демонстрация работы HTML Renderer для генерации финальных отчетов.
"""

import sys
from pathlib import Path
import json
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# Добавляем путь к src
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from reporting.html_renderer import HTMLRenderer, ReportConfig
from reporting.visualization_engine import VisualizationEngine, ChartConfig
from reporting.data_processor import DataProcessor, MetricType
from utils import setup_logging


def create_sample_figures():
    """Создание примеров графиков для отчета."""
    figures = {}

    # 1. График сравнения производительности
    operations = ["read_csv", "filter", "groupby", "sort", "join"]
    pandas_times = [0.15, 0.08, 0.25, 0.18, 0.35]
    polars_times = [0.05, 0.02, 0.06, 0.045, 0.09]

    fig_comparison = go.Figure()
    fig_comparison.add_trace(
        go.Bar(name="Pandas", x=operations, y=pandas_times, marker_color="#FF6B6B")
    )
    fig_comparison.add_trace(
        go.Bar(name="Polars", x=operations, y=polars_times, marker_color="#4ECDC4")
    )
    fig_comparison.update_layout(
        title="Сравнение времени выполнения операций",
        xaxis_title="Операция",
        yaxis_title="Время (сек)",
        barmode="group",
        height=500,
    )
    figures["comparison_bar"] = fig_comparison

    # 2. График speedup
    speedup = [p / t for p, t in zip(pandas_times, polars_times)]

    fig_speedup = go.Figure()
    fig_speedup.add_trace(
        go.Bar(
            x=operations,
            y=speedup,
            text=[f"{s:.1f}x" for s in speedup],
            textposition="outside",
            marker_color=["#10B981" if s > 1 else "#EF4444" for s in speedup],
        )
    )
    fig_speedup.add_hline(y=1, line_dash="dash", line_color="gray")
    fig_speedup.update_layout(
        title="Ускорение Polars относительно Pandas",
        xaxis_title="Операция",
        yaxis_title="Коэффициент ускорения",
        height=500,
    )
    figures["speedup_chart"] = fig_speedup

    # 3. Тепловая карта
    datasets = ["small (1K)", "medium (100K)", "large (1M)", "xlarge (10M)"]
    heatmap_data = np.random.rand(5, 4) * 5 + 1  # Speedup от 1x до 6x

    fig_heatmap = go.Figure(
        data=go.Heatmap(
            z=heatmap_data,
            x=datasets,
            y=operations,
            colorscale="RdYlGn",
            text=[[f"{val:.1f}x" for val in row] for row in heatmap_data],
            texttemplate="%{text}",
            textfont={"size": 12},
        )
    )
    fig_heatmap.update_layout(
        title="Ускорение Polars по операциям и размерам данных",
        xaxis_title="Размер датасета",
        yaxis_title="Операция",
        height=500,
    )
    figures["heatmap"] = fig_heatmap

    # 4. Box plot распределения
    np.random.seed(42)
    distribution_data = []

    for op in operations[:3]:  # Первые 3 операции
        for lib in ["Pandas", "Polars"]:
            base = 0.2 if lib == "Pandas" else 0.05
            times = np.random.normal(base, base * 0.1, 50)
            for t in times:
                distribution_data.append({"Operation": op, "Library": lib, "Time": t})

    df_dist = pd.DataFrame(distribution_data)

    fig_box = px.box(
        df_dist,
        x="Operation",
        y="Time",
        color="Library",
        title="Распределение времени выполнения",
        labels={"Time": "Время (сек)"},
    )
    fig_box.update_layout(height=500)
    figures["distribution_box"] = fig_box

    # 5. Линейный график зависимости от размера
    sizes = [1000, 10000, 100000, 1000000]
    pandas_scaling = [0.01, 0.15, 2.5, 35]
    polars_scaling = [0.005, 0.04, 0.5, 6]

    fig_timeline = go.Figure()
    fig_timeline.add_trace(
        go.Scatter(
            x=sizes,
            y=pandas_scaling,
            mode="lines+markers",
            name="Pandas",
            line=dict(color="#FF6B6B", width=3),
        )
    )
    fig_timeline.add_trace(
        go.Scatter(
            x=sizes,
            y=polars_scaling,
            mode="lines+markers",
            name="Polars",
            line=dict(color="#4ECDC4", width=3),
        )
    )
    fig_timeline.update_xaxes(type="log", title="Размер датасета")
    fig_timeline.update_yaxes(type="log", title="Время выполнения (сек)")
    fig_timeline.update_layout(title="Масштабирование производительности", height=500)
    figures["timeline"] = fig_timeline

    # 6. Сводная таблица
    summary_data = {
        "Операция": operations,
        "Pandas (сек)": pandas_times,
        "Polars (сек)": polars_times,
        "Ускорение": [f"{s:.1f}x" for s in speedup],
        "Победитель": ["Polars" if s > 1 else "Pandas" for s in speedup],
    }

    fig_table = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=list(summary_data.keys()),
                    fill_color="paleturquoise",
                    align="left",
                ),
                cells=dict(
                    values=list(summary_data.values()),
                    fill_color="lavender",
                    align="left",
                ),
            )
        ]
    )
    fig_table.update_layout(title="Сводная таблица результатов", height=400)
    figures["summary_table"] = fig_table

    return figures


def create_sample_summary():
    """Создание примера сводных данных."""
    return {
        "overall_summary": {
            "total_comparisons": 45,
            "avg_speedup": 3.2,
            "median_speedup": 2.8,
            "min_speedup": 0.8,
            "max_speedup": 6.5,
            "polars_wins": 40,
            "pandas_wins": 5,
            "polars_win_rate": 88.9,
        },
        "summary_by_operation": {
            "read_csv": {"avg_speedup": 3.0, "polars_wins": 8, "pandas_wins": 0},
            "filter": {"avg_speedup": 4.0, "polars_wins": 8, "pandas_wins": 0},
            "groupby": {"avg_speedup": 4.2, "polars_wins": 8, "pandas_wins": 0},
            "sort": {"avg_speedup": 4.0, "polars_wins": 8, "pandas_wins": 0},
            "join": {"avg_speedup": 3.9, "polars_wins": 8, "pandas_wins": 0},
        },
        "metadata": {
            "run_date": "2024-01-15",
            "benchmark_version": "1.0.0",
            "system_info": {
                "os": "Ubuntu 22.04",
                "cpu": "Intel Core i9-9900K",
                "memory": "32GB DDR4",
            },
        },
    }


def demonstrate_html_renderer():
    """Основная демонстрация."""
    logger = setup_logging("demo_html_renderer", console_level="INFO")

    logger.info("=" * 80)
    logger.info("ДЕМОНСТРАЦИЯ HTML RENDERER")
    logger.info("=" * 80)

    # 1. Создание тестовых данных
    logger.info("\n1. ПОДГОТОВКА ДАННЫХ")
    figures = create_sample_figures()
    summary_data = create_sample_summary()
    logger.info(f"Создано {len(figures)} графиков")
    logger.info(f"Подготовлены сводные данные")

    # 2. Создание рендерера
    logger.info("\n2. СОЗДАНИЕ HTML RENDERER")
    renderer = HTMLRenderer(logger=logger)

    # 3. Конфигурация отчета
    config = ReportConfig(
        title="Результаты бенчмаркинга Pandas vs Polars",
        subtitle="Комплексный анализ производительности",
        author="Система автоматического бенчмаркинга",
        description="Детальное сравнение производительности библиотек обработки данных",
        include_toc=True,
        include_summary=True,
        include_methodology=True,
        include_recommendations=True,
    )

    # 4. Генерация отчета
    logger.info("\n3. ГЕНЕРАЦИЯ HTML ОТЧЕТА")
    output_path = Path("demo_outputs") / "benchmark_report.html"
    output_path.parent.mkdir(exist_ok=True)

    html_content = renderer.render_report(
        figures=figures,
        summary_data=summary_data,
        config=config,
        output_path=output_path,
    )

    logger.info(f"Отчет сохранен: {output_path}")
    logger.info(f"Размер файла: {len(html_content) / 1024:.1f} KB")

    # 5. Генерация минимального отчета
    logger.info("\n4. ГЕНЕРАЦИЯ МИНИМАЛЬНОГО ОТЧЕТА")
    minimal_config = ReportConfig(
        title="Краткий отчет Pandas vs Polars",
        include_toc=False,
        include_methodology=False,
        include_recommendations=False,
    )

    minimal_path = Path("demo_outputs") / "benchmark_report_minimal.html"
    renderer.render_report(
        figures={"comparison_bar": figures["comparison_bar"]},
        summary_data=summary_data,
        config=minimal_config,
        output_path=minimal_path,
    )

    logger.info(f"Минимальный отчет сохранен: {minimal_path}")

    # 6. Вывод структуры отчета
    logger.info("\n5. СТРУКТУРА ОТЧЕТА")
    logger.info("Секции отчета:")
    for section in renderer.sections:
        logger.info(f"  - {section.title} (id: {section.section_id})")

    logger.info("\n" + "=" * 80)
    logger.info("ДЕМОНСТРАЦИЯ ЗАВЕРШЕНА")
    logger.info("Откройте файлы в браузере для просмотра отчетов")
    logger.info("=" * 80)


if __name__ == "__main__":
    demonstrate_html_renderer()
