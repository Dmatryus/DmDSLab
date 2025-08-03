"""
Модуль для создания интерактивных визуализаций с использованием Plotly.
Генерирует различные типы графиков на основе обработанных данных.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum

from researches.polaris.pandas_polars_benchmark.src import get_logger
from researches.polaris.pandas_polars_benchmark.src.reporting import (
    ProcessedData,
    MetricType,
)


class ChartType(Enum):
    """Типы доступных графиков."""

    BAR = "bar"
    LINE = "line"
    HEATMAP = "heatmap"
    BOX = "box"
    SCATTER = "scatter"
    VIOLIN = "violin"
    TABLE = "table"


@dataclass
class ChartConfig:
    """Конфигурация для графика."""

    title: str
    x_label: str
    y_label: str
    height: int = 500
    width: Optional[int] = None
    show_legend: bool = True
    color_scheme: str = "plotly"
    log_scale: bool = False


class ColorPalette:
    """Цветовые схемы для визуализаций."""

    PANDAS = "#1f77b4"  # Синий
    POLARS = "#ff7f0e"  # Оранжевый

    LIBRARIES = {"pandas": PANDAS, "polars": POLARS}

    # Градиенты для тепловых карт
    HEATMAP_SCALE = "RdYlGn_r"  # От зеленого (хорошо) к красному (плохо)
    DIVERGING_SCALE = "RdBu"

    # Палитра для операций
    OPERATIONS = px.colors.qualitative.Set2


class VisualizationEngine:
    """Движок для создания визуализаций."""

    def __init__(self, logger=None):
        """
        Инициализация движка визуализации.

        Args:
            logger: Логгер для вывода информации
        """
        self.logger = logger or get_logger(__name__)
        self.default_layout = self._get_default_layout()

    def _get_default_layout(self) -> Dict[str, Any]:
        """Получение стандартных настроек layout для всех графиков."""
        return {
            "font": {"family": "Arial, sans-serif", "size": 12},
            "hovermode": "closest",
            "plot_bgcolor": "white",
            "paper_bgcolor": "white",
            "margin": {"t": 80, "b": 80, "l": 80, "r": 80},
            "xaxis": {
                "showgrid": True,
                "gridcolor": "rgba(128, 128, 128, 0.2)",
                "showline": True,
                "linecolor": "rgba(128, 128, 128, 0.4)",
            },
            "yaxis": {
                "showgrid": True,
                "gridcolor": "rgba(128, 128, 128, 0.2)",
                "showline": True,
                "linecolor": "rgba(128, 128, 128, 0.4)",
            },
        }

    def create_comparison_bar_chart(
        self, data: ProcessedData, config: Optional[ChartConfig] = None
    ) -> go.Figure:
        """
        Создание столбчатой диаграммы для сравнения библиотек.

        Args:
            data: Обработанные данные
            config: Конфигурация графика

        Returns:
            go.Figure: График Plotly
        """
        if config is None:
            config = ChartConfig(
                title="Сравнение производительности Pandas vs Polars",
                x_label="Операция",
                y_label=self._get_metric_label(data.metric_type),
            )

        fig = go.Figure()

        df = (
            data.data.reset_index()
            if isinstance(data.data.index, pd.MultiIndex)
            else data.data
        )

        # Определяем колонки с данными
        if "library" in df.columns:
            # Данные не в pivot формате
            for library in df["library"].unique():
                lib_data = df[df["library"] == library]

                fig.add_trace(
                    go.Bar(
                        name=library.capitalize(),
                        x=lib_data[df.columns[0]],  # Первая колонка - категория
                        y=(
                            lib_data["mean"]
                            if "mean" in lib_data.columns
                            else lib_data[self._get_metric_column(data.metric_type)]
                        ),
                        error_y=(
                            dict(
                                type="data",
                                array=(
                                    lib_data["std"]
                                    if "std" in lib_data.columns
                                    else None
                                ),
                                visible=True,
                            )
                            if "std" in lib_data.columns
                            else None
                        ),
                        marker_color=ColorPalette.LIBRARIES.get(library, None),
                        text=[
                            f"{v:.3f}"
                            for v in (
                                lib_data["mean"]
                                if "mean" in lib_data.columns
                                else lib_data[self._get_metric_column(data.metric_type)]
                            )
                        ],
                        textposition="outside",
                    )
                )
        else:
            # Данные в pivot формате
            x_values = df.index.tolist() if df.index.name else df.iloc[:, 0].tolist()

            # Находим колонки для каждой библиотеки
            for col in df.columns:
                if isinstance(col, tuple):
                    metric, library = col
                    if metric == "mean":
                        std_col = ("std", library)
                        fig.add_trace(
                            go.Bar(
                                name=library.capitalize(),
                                x=x_values,
                                y=df[col],
                                error_y=(
                                    dict(
                                        type="data",
                                        array=(
                                            df[std_col]
                                            if std_col in df.columns
                                            else None
                                        ),
                                        visible=True,
                                    )
                                    if std_col in df.columns
                                    else None
                                ),
                                marker_color=ColorPalette.LIBRARIES.get(library, None),
                                text=[f"{v:.3f}" for v in df[col]],
                                textposition="outside",
                            )
                        )

        # Настройка layout
        fig.update_layout(
            **self.default_layout,
            title=config.title,
            xaxis_title=config.x_label,
            yaxis_title=config.y_label,
            barmode="group",
            height=config.height,
            width=config.width,
            showlegend=config.show_legend,
        )

        if config.log_scale:
            fig.update_yaxes(type="log")

        return fig

    def create_line_chart(
        self, data: ProcessedData, config: Optional[ChartConfig] = None
    ) -> go.Figure:
        """
        Создание линейного графика (например, зависимость от размера данных).

        Args:
            data: Обработанные данные
            config: Конфигурация графика

        Returns:
            go.Figure: График Plotly
        """
        if config is None:
            config = ChartConfig(
                title="Зависимость производительности от размера данных",
                x_label="Размер датасета",
                y_label=self._get_metric_label(data.metric_type),
                log_scale=True,
            )

        fig = go.Figure()

        df = data.data

        # Группируем по операциям и библиотекам
        for operation in df["operation"].unique():
            for library in df["library"].unique():
                mask = (df["operation"] == operation) & (df["library"] == library)
                op_data = df[mask]

                if len(op_data) > 0:
                    fig.add_trace(
                        go.Scatter(
                            name=f"{library.capitalize()} - {operation}",
                            x=op_data["dataset_size"],
                            y=op_data["mean"],
                            error_y=(
                                dict(type="data", array=op_data["std"], visible=True)
                                if "std" in op_data.columns
                                else None
                            ),
                            mode="lines+markers",
                            line=dict(
                                color=ColorPalette.LIBRARIES.get(library),
                                dash=(
                                    "solid"
                                    if operation in ["read_csv", "write_csv"]
                                    else (
                                        "dash"
                                        if operation in ["filter", "sort"]
                                        else "dot"
                                    )
                                ),
                            ),
                            marker=dict(size=8),
                        )
                    )

        # Настройка layout
        fig.update_layout(
            **self.default_layout,
            title=config.title,
            xaxis_title=config.x_label,
            yaxis_title=config.y_label,
            height=config.height,
            width=config.width,
            showlegend=config.show_legend,
            xaxis_type="log" if config.log_scale else "linear",
            yaxis_type="log" if config.log_scale else "linear",
        )

        return fig

    def create_heatmap(
        self, data: ProcessedData, config: Optional[ChartConfig] = None
    ) -> go.Figure:
        """
        Создание тепловой карты.

        Args:
            data: Обработанные данные
            config: Конфигурация графика

        Returns:
            go.Figure: График Plotly
        """
        if config is None:
            config = ChartConfig(
                title="Тепловая карта производительности",
                x_label="Размер датасета",
                y_label="Операция",
            )

        # Создаем отдельные тепловые карты для каждой библиотеки
        libraries = data.metadata.get("libraries", ["pandas", "polars"])

        fig = make_subplots(
            rows=1,
            cols=len(libraries),
            subplot_titles=[f"{lib.capitalize()}" for lib in libraries],
            horizontal_spacing=0.1,
        )

        for i, library in enumerate(libraries):
            # Извлекаем данные для библиотеки
            if library in data.data:
                z_data = data.data[library]
            else:
                # Фильтруем данные если они не в pivot формате
                lib_data = (
                    data.data[data.data["library"] == library]
                    if "library" in data.data.columns
                    else data.data
                )
                continue

            # Определяем текст для hover
            hover_text = [
                [
                    f"{library.capitalize()}<br>Операция: {y}<br>Размер: {x}<br>Значение: {z:.3f}"
                    for x, z in zip(z_data.columns, row)
                ]
                for y, row in zip(z_data.index, z_data.values)
            ]

            fig.add_trace(
                go.Heatmap(
                    z=z_data.values,
                    x=z_data.columns.tolist(),
                    y=z_data.index.tolist(),
                    colorscale=ColorPalette.HEATMAP_SCALE,
                    text=z_data.values,
                    texttemplate="%{text:.2f}",
                    hovertext=hover_text,
                    hovertemplate="%{hovertext}<extra></extra>",
                    colorbar=(
                        dict(title=self._get_metric_label(data.metric_type))
                        if i == len(libraries) - 1
                        else None
                    ),
                    showscale=(i == len(libraries) - 1),
                ),
                row=1,
                col=i + 1,
            )

        # Настройка layout
        fig.update_layout(
            title=config.title,
            height=config.height,
            width=config.width or 600 * len(libraries),
            showlegend=False,
        )

        # Обновление осей
        for i in range(len(libraries)):
            fig.update_xaxes(
                title_text=config.x_label if i == 0 else "", row=1, col=i + 1
            )
            fig.update_yaxes(
                title_text=config.y_label if i == 0 else "", row=1, col=i + 1
            )

        return fig

    def create_box_plot(
        self, data: ProcessedData, config: Optional[ChartConfig] = None
    ) -> go.Figure:
        """
        Создание box plot для анализа распределения.

        Args:
            data: Обработанные данные
            config: Конфигурация графика

        Returns:
            go.Figure: График Plotly
        """
        if config is None:
            config = ChartConfig(
                title="Распределение времени выполнения",
                x_label="Операция",
                y_label=self._get_metric_label(data.metric_type),
            )

        fig = go.Figure()

        df = data.data

        # Если данные содержат raw значения
        if "values" in df.columns:
            for library in df["library"].unique():
                lib_data = df[df["library"] == library]

                fig.add_trace(
                    go.Box(
                        name=library.capitalize(),
                        y=lib_data["values"],
                        x=(
                            lib_data["operation"]
                            if "operation" in lib_data.columns
                            else None
                        ),
                        marker_color=ColorPalette.LIBRARIES.get(library),
                        boxpoints="outliers",
                        jitter=0.3,
                        pointpos=-1.8,
                    )
                )
        else:
            # Если только агрегированные данные, используем violin plot
            for library in data.metadata.get("libraries", ["pandas", "polars"]):
                lib_data = (
                    df[df["library"] == library] if "library" in df.columns else df
                )

                # Генерируем примерные данные на основе mean и std
                if "mean" in lib_data.columns and "std" in lib_data.columns:
                    for _, row in lib_data.iterrows():
                        # Генерируем нормальное распределение
                        values = np.random.normal(row["mean"], row["std"], 100)
                        operation = (
                            row[lib_data.columns[0]]
                            if len(lib_data.columns) > 0
                            else "Operation"
                        )

                        fig.add_trace(
                            go.Violin(
                                name=f"{library.capitalize()} - {operation}",
                                y=values,
                                box_visible=True,
                                meanline_visible=True,
                                fillcolor=ColorPalette.LIBRARIES.get(library),
                                opacity=0.6,
                                x0=operation,
                            )
                        )

        # Настройка layout
        fig.update_layout(
            **self.default_layout,
            title=config.title,
            xaxis_title=config.x_label,
            yaxis_title=config.y_label,
            height=config.height,
            width=config.width,
            showlegend=config.show_legend,
            violinmode="group" if "Violin" in str(fig.data) else None,
        )

        if config.log_scale:
            fig.update_yaxis(type="log")

        return fig

    def create_performance_table(
        self, data: ProcessedData, config: Optional[ChartConfig] = None
    ) -> go.Figure:
        """
        Создание интерактивной таблицы с результатами.

        Args:
            data: Обработанные данные
            config: Конфигурация

        Returns:
            go.Figure: Таблица Plotly
        """
        if config is None:
            config = ChartConfig(
                title="Сводная таблица производительности", x_label="", y_label=""
            )

        df = data.data

        # Подготовка данных для таблицы
        if isinstance(df.columns, pd.MultiIndex):
            # Flatten multi-index columns
            df.columns = ["_".join(map(str, col)).strip() for col in df.columns.values]

        # Форматирование числовых значений
        for col in df.columns:
            if df[col].dtype in ["float64", "float32"]:
                df[col] = df[col].apply(lambda x: f"{x:.3f}")

        # Создание таблицы
        fig = go.Figure(
            data=[
                go.Table(
                    header=dict(
                        values=list(df.columns),
                        fill_color="lightgray",
                        align="left",
                        font=dict(size=12, color="black"),
                    ),
                    cells=dict(
                        values=[df[col] for col in df.columns],
                        fill_color="white",
                        align="left",
                        font=dict(size=11),
                        height=25,
                    ),
                )
            ]
        )

        # Настройка layout
        fig.update_layout(
            title=config.title,
            height=config.height or min(600, 50 + len(df) * 25),
            width=config.width,
            margin=dict(t=50, b=10, l=10, r=10),
        )

        return fig

    def create_speedup_chart(
        self, data: ProcessedData, config: Optional[ChartConfig] = None
    ) -> go.Figure:
        """
        Создание графика ускорения (speedup) относительно baseline.

        Args:
            data: Обработанные данные с метрикой SPEEDUP
            config: Конфигурация графика

        Returns:
            go.Figure: График Plotly
        """
        if config is None:
            config = ChartConfig(
                title="Ускорение Polars относительно Pandas",
                x_label="Операция",
                y_label="Ускорение (раз)",
            )

        fig = go.Figure()

        df = data.data

        # Добавляем горизонтальную линию на уровне 1 (равная производительность)
        fig.add_hline(
            y=1,
            line_dash="dash",
            line_color="gray",
            annotation_text="Равная производительность",
            annotation_position="right",
        )

        # Определяем данные для графика
        if "speedup" in df.columns:
            x_values = df[df.columns[0]].tolist()
            y_values = df["speedup"].tolist()

            # Цвета в зависимости от того, какая библиотека быстрее
            colors = [
                ColorPalette.POLARS if y > 1 else ColorPalette.PANDAS for y in y_values
            ]

            fig.add_trace(
                go.Bar(
                    x=x_values,
                    y=y_values,
                    marker_color=colors,
                    text=[f"{y:.2f}x" for y in y_values],
                    textposition="outside",
                    hovertemplate="%{x}<br>Ускорение: %{y:.2f}x<extra></extra>",
                )
            )
        else:
            # Если данные в другом формате
            self.logger.warning("Данные не содержат колонку 'speedup'")

        # Настройка layout
        fig.update_layout(
            **self.default_layout,
            title=config.title,
            xaxis_title=config.x_label,
            yaxis_title=config.y_label,
            height=config.height,
            width=config.width,
            showlegend=False,
        )

        # Добавляем аннотации для пояснения
        fig.add_annotation(
            x=0.02,
            y=0.98,
            xref="paper",
            yref="paper",
            text="Выше 1 = Polars быстрее<br>Ниже 1 = Pandas быстрее",
            showarrow=False,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="gray",
            borderwidth=1,
        )

        return fig

    def create_dashboard(
        self,
        processed_data_dict: Dict[str, ProcessedData],
        title: str = "Pandas vs Polars Performance Dashboard",
    ) -> go.Figure:
        """
        Создание комплексного дашборда с несколькими графиками.

        Args:
            processed_data_dict: Словарь с различными обработанными данными
            title: Заголовок дашборда

        Returns:
            go.Figure: Комплексный дашборд
        """
        # Определяем layout сетки
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Сравнение по операциям",
                "Зависимость от размера",
                "Распределение времени",
                "Ускорение",
            ),
            specs=[
                [{"type": "bar"}, {"type": "scatter"}],
                [{"type": "box"}, {"type": "bar"}],
            ],
            vertical_spacing=0.15,
            horizontal_spacing=0.1,
        )

        # График 1: Сравнение по операциям
        if "comparison" in processed_data_dict:
            bar_fig = self.create_comparison_bar_chart(
                processed_data_dict["comparison"]
            )
            for trace in bar_fig.data:
                fig.add_trace(trace, row=1, col=1)

        # График 2: Зависимость от размера
        if "timeline" in processed_data_dict:
            line_fig = self.create_line_chart(processed_data_dict["timeline"])
            for trace in line_fig.data:
                fig.add_trace(trace, row=1, col=2)

        # График 3: Распределение
        if "distribution" in processed_data_dict:
            box_fig = self.create_box_plot(processed_data_dict["distribution"])
            for trace in box_fig.data:
                fig.add_trace(trace, row=2, col=1)

        # График 4: Ускорение
        if "speedup" in processed_data_dict:
            speedup_fig = self.create_speedup_chart(processed_data_dict["speedup"])
            for trace in speedup_fig.data:
                fig.add_trace(trace, row=2, col=2)

        # Обновление layout
        fig.update_layout(
            title=title,
            height=1000,
            showlegend=True,
            legend=dict(
                orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5
            ),
        )

        return fig

    def export_figure(
        self, fig: go.Figure, output_path: str, format: str = "html", **kwargs
    ) -> None:
        """
        Экспорт фигуры в файл.

        Args:
            fig: График Plotly
            output_path: Путь для сохранения
            format: Формат файла (html, png, svg, pdf)
            **kwargs: Дополнительные параметры для экспорта
        """
        if format == "html":
            fig.write_html(
                output_path,
                include_plotlyjs="cdn",
                config={"displayModeBar": True, "displaylogo": False},
                **kwargs,
            )
        elif format in ["png", "svg", "pdf"]:
            # Требует kaleido
            fig.write_image(output_path, format=format, **kwargs)
        else:
            raise ValueError(f"Неподдерживаемый формат: {format}")

        self.logger.info(f"График экспортирован в {output_path}")

    def _get_metric_label(self, metric_type: MetricType) -> str:
        """Получение подписи для метрики."""
        labels = {
            MetricType.EXECUTION_TIME: "Время выполнения (сек)",
            MetricType.MEMORY_USAGE: "Использование памяти (МБ)",
            MetricType.MEMORY_PEAK: "Пиковая память (МБ)",
            MetricType.RELATIVE_PERFORMANCE: "Относительная производительность",
            MetricType.SPEEDUP: "Ускорение (раз)",
        }
        return labels.get(metric_type, "Значение")

    def _get_metric_column(self, metric_type: MetricType) -> str:
        """Получение названия колонки для метрики."""
        mapping = {
            MetricType.EXECUTION_TIME: "execution_time",
            MetricType.MEMORY_USAGE: "memory_usage",
            MetricType.MEMORY_PEAK: "memory_peak",
            MetricType.RELATIVE_PERFORMANCE: "relative_performance",
            MetricType.SPEEDUP: "speedup",
        }
        return mapping.get(metric_type, "value")
