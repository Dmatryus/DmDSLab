"""
Модуль для обработки и подготовки данных к визуализации.
Преобразует сырые результаты бенчмаркинга в форматы, удобные для построения графиков.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path

from researches.polaris.pandas_polars_benchmark.src import get_logger


class AggregationLevel(Enum):
    """Уровни агрегации данных."""
    OPERATION = "operation"
    LIBRARY = "library"
    DATASET = "dataset"
    OPERATION_TYPE = "operation_type"


class MetricType(Enum):
    """Типы метрик для визуализации."""
    EXECUTION_TIME = "execution_time"
    MEMORY_USAGE = "memory_usage"
    MEMORY_PEAK = "memory_peak"
    RELATIVE_PERFORMANCE = "relative_performance"
    SPEEDUP = "speedup"


@dataclass
class ProcessedData:
    """Контейнер для обработанных данных."""
    data: pd.DataFrame
    metadata: Dict[str, Any]
    aggregation_level: AggregationLevel
    metric_type: MetricType
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь для сериализации."""
        # Создаем копию для безопасной модификации
        data_copy = self.data.copy()
        
        # Обработка MultiIndex колонок (после pivot операций)
        if isinstance(data_copy.columns, pd.MultiIndex):
            # Преобразуем MultiIndex колонки в строковые имена
            data_copy.columns = ['_'.join(map(str, col)).strip('_') for col in data_copy.columns.values]
        
        # Обработка MultiIndex индекса
        if isinstance(data_copy.index, pd.MultiIndex):
            data_copy = data_copy.reset_index()
        
        return {
            "data": data_copy.to_dict("records"),
            "metadata": self.metadata,
            "aggregation_level": self.aggregation_level.value,
            "metric_type": self.metric_type.value
        }


class DataProcessor:
    """Процессор для подготовки данных к визуализации."""
    
    def __init__(self, logger=None):
        """
        Инициализация процессора.
        
        Args:
            logger: Логгер для вывода информации
        """
        self.logger = logger or get_logger(__name__)
        
    def load_results(self, results_path: Union[str, Path]) -> pd.DataFrame:
        """
        Загрузка результатов бенчмаркинга.
        
        Args:
            results_path: Путь к файлу с результатами (JSON или CSV)
            
        Returns:
            pd.DataFrame: Загруженные результаты
        """
        results_path = Path(results_path)
        
        if not results_path.exists():
            raise FileNotFoundError(f"Файл результатов не найден: {results_path}")
            
        self.logger.info(f"Загрузка результатов из {results_path}")
        
        if results_path.suffix == ".json":
            with open(results_path, "r") as f:
                data = json.load(f)
                # Если это список записей
                if isinstance(data, list):
                    df = pd.DataFrame(data)
                # Если это словарь с результатами
                elif isinstance(data, dict) and "results" in data:
                    df = pd.DataFrame(data["results"])
                else:
                    df = pd.DataFrame([data])
        elif results_path.suffix == ".csv":
            df = pd.read_csv(results_path)
        else:
            raise ValueError(f"Неподдерживаемый формат файла: {results_path.suffix}")
            
        self.logger.info(f"Загружено {len(df)} записей")
        return df
        
    def prepare_comparison_data(
        self,
        df: pd.DataFrame,
        metric: MetricType,
        groupby: List[str],
        pivot_column: str = "library"
    ) -> ProcessedData:
        """
        Подготовка данных для сравнительных графиков.
        
        Args:
            df: DataFrame с результатами
            metric: Тип метрики для сравнения
            groupby: Список колонок для группировки
            pivot_column: Колонка для создания отдельных серий (обычно "library")
            
        Returns:
            ProcessedData: Обработанные данные
        """
        self.logger.debug(f"Подготовка данных для сравнения по {metric.value}")
        
        # Выбор нужной метрики
        metric_column = self._get_metric_column(metric)
        
        # Группировка и агрегация
        grouped = df.groupby(groupby + [pivot_column])[metric_column].agg(['mean', 'std']).reset_index()
        
        # Создание pivot таблицы для удобства визуализации
        pivot_data = grouped.pivot_table(
            index=groupby,
            columns=pivot_column,
            values=['mean', 'std']
        )
        
        # Расчет дополнительных метрик
        if metric == MetricType.RELATIVE_PERFORMANCE:
            pivot_data = self._calculate_relative_performance(pivot_data)
        elif metric == MetricType.SPEEDUP:
            pivot_data = self._calculate_speedup(pivot_data)
        
        # Преобразование MultiIndex колонок в плоские имена для лучшей совместимости
        pivot_data.columns = ['_'.join(col).strip() for col in pivot_data.columns.values]
            
        # Определение уровня агрегации
        if len(groupby) == 1:
            agg_level = AggregationLevel(groupby[0])
        else:
            agg_level = AggregationLevel.OPERATION
            
        metadata = {
            "groupby_columns": groupby,
            "pivot_column": pivot_column,
            "total_records": len(df),
            "unique_values": {col: df[col].unique().tolist() for col in groupby + [pivot_column]}
        }
        
        return ProcessedData(
            data=pivot_data.reset_index(),
            metadata=metadata,
            aggregation_level=agg_level,
            metric_type=metric
        )
        
    def prepare_distribution_data(
        self,
        df: pd.DataFrame,
        metric: MetricType,
        library: str,
        operation: Optional[str] = None
    ) -> ProcessedData:
        """
        Подготовка данных для графиков распределения.
        
        Args:
            df: DataFrame с результатами
            metric: Тип метрики
            library: Название библиотеки
            operation: Конкретная операция (опционально)
            
        Returns:
            ProcessedData: Данные для построения распределения
        """
        # Фильтрация данных
        mask = df['library'] == library
        if operation:
            mask &= df['operation'] == operation
            
        filtered_df = df[mask].copy()
        metric_column = self._get_metric_column(metric)
        
        # Подготовка данных для гистограммы/box plot
        distribution_data = pd.DataFrame({
            'value': filtered_df[metric_column],
            'library': library,
            'operation': operation or 'all'
        })
        
        metadata = {
            "library": library,
            "operation": operation,
            "sample_size": len(distribution_data),
            "statistics": {
                "mean": distribution_data['value'].mean(),
                "std": distribution_data['value'].std(),
                "min": distribution_data['value'].min(),
                "max": distribution_data['value'].max(),
                "median": distribution_data['value'].median()
            }
        }
        
        return ProcessedData(
            data=distribution_data,
            metadata=metadata,
            aggregation_level=AggregationLevel.OPERATION if operation else AggregationLevel.LIBRARY,
            metric_type=metric
        )
        
    def prepare_timeline_data(
        self,
        df: pd.DataFrame,
        metric: MetricType,
        dataset_size_column: str = "dataset_size"
    ) -> ProcessedData:
        """
        Подготовка данных для графиков зависимости от размера данных.
        
        Args:
            df: DataFrame с результатами
            metric: Тип метрики
            dataset_size_column: Название колонки с размером датасета
            
        Returns:
            ProcessedData: Данные для timeline графиков
        """
        metric_column = self._get_metric_column(metric)
        
        # Группировка по размеру и библиотеке
        timeline_data = df.groupby([dataset_size_column, 'library', 'operation'])[metric_column].agg([
            'mean', 'std', 'count'
        ]).reset_index()
        
        # Сортировка по размеру
        timeline_data = timeline_data.sort_values(dataset_size_column)
        
        metadata = {
            "dataset_sizes": sorted(df[dataset_size_column].unique().tolist()),
            "operations": df['operation'].unique().tolist(),
            "libraries": df['library'].unique().tolist()
        }
        
        return ProcessedData(
            data=timeline_data,
            metadata=metadata,
            aggregation_level=AggregationLevel.DATASET,
            metric_type=metric
        )
        
    def prepare_heatmap_data(
        self,
        df: pd.DataFrame,
        metric: MetricType,
        row_column: str = "operation",
        col_column: str = "dataset_size"
    ) -> ProcessedData:
        """
        Подготовка данных для тепловых карт.
        
        Args:
            df: DataFrame с результатами
            metric: Тип метрики
            row_column: Колонка для строк тепловой карты
            col_column: Колонка для столбцов тепловой карты
            
        Returns:
            ProcessedData: Данные для heatmap
        """
        metric_column = self._get_metric_column(metric)
        
        heatmap_data = {}
        
        for library in df['library'].unique():
            library_df = df[df['library'] == library]
            
            # Создание pivot таблицы
            pivot = library_df.pivot_table(
                index=row_column,
                columns=col_column,
                values=metric_column,
                aggfunc='mean'
            )
            
            heatmap_data[library] = pivot
            
        # Объединение данных для сравнения
        combined_data = pd.concat(heatmap_data, names=['library']).reset_index()
        
        # Получение диапазона значений только из числовых столбцов
        numeric_columns = combined_data.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            min_value = combined_data[numeric_columns].min().min()
            max_value = combined_data[numeric_columns].max().max()
        else:
            min_value = max_value = 0
        
        metadata = {
            "row_values": df[row_column].unique().tolist(),
            "col_values": sorted(df[col_column].unique().tolist()),
            "libraries": list(heatmap_data.keys()),
            "value_range": (min_value, max_value)
        }
        
        return ProcessedData(
            data=combined_data,
            metadata=metadata,
            aggregation_level=AggregationLevel.OPERATION,
            metric_type=metric
        )
        
    def create_summary_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Создание сводной таблицы с основными метриками.
        
        Args:
            df: DataFrame с результатами
            
        Returns:
            pd.DataFrame: Сводная таблица
        """
        summary_data = []
        
        for library in df['library'].unique():
            lib_df = df[df['library'] == library]
            
            for operation in df['operation'].unique():
                op_df = lib_df[lib_df['operation'] == operation]
                
                if len(op_df) == 0:
                    continue
                    
                summary_data.append({
                    'library': library,
                    'operation': operation,
                    'mean_time': op_df['execution_time'].mean(),
                    'std_time': op_df['execution_time'].std(),
                    'mean_memory': op_df['memory_usage'].mean(),
                    'peak_memory': op_df['memory_peak'].max(),
                    'samples': len(op_df)
                })
                
        summary_df = pd.DataFrame(summary_data)
        
        # Добавление относительных метрик (Polars как baseline)
        if 'polars' in summary_df['library'].values:
            for operation in summary_df['operation'].unique():
                polars_time = summary_df[
                    (summary_df['library'] == 'polars') & 
                    (summary_df['operation'] == operation)
                ]['mean_time'].values
                
                if len(polars_time) > 0:
                    mask = summary_df['operation'] == operation
                    summary_df.loc[mask, 'relative_time'] = (
                        summary_df.loc[mask, 'mean_time'] / polars_time[0]
                    )
                    
        return summary_df
        
    def _get_metric_column(self, metric: MetricType) -> str:
        """Получение названия колонки для метрики."""
        mapping = {
            MetricType.EXECUTION_TIME: "execution_time",
            MetricType.MEMORY_USAGE: "memory_usage",
            MetricType.MEMORY_PEAK: "memory_peak",
            MetricType.RELATIVE_PERFORMANCE: "execution_time",
            MetricType.SPEEDUP: "execution_time"
        }
        return mapping.get(metric, "execution_time")
        
    def _calculate_relative_performance(self, pivot_data: pd.DataFrame) -> pd.DataFrame:
        """Расчет относительной производительности."""
        # Проверяем наличие колонок для polars
        mean_cols = [col for col in pivot_data.columns if col[0] == 'mean']
        polars_col = next((col for col in mean_cols if col[1] == 'polars'), None)
        
        if polars_col:
            baseline = pivot_data[polars_col]
            
            for col in mean_cols:
                lib_name = col[1]
                pivot_data[('relative', lib_name)] = pivot_data[col] / baseline
                
        return pivot_data
        
    def _calculate_speedup(self, pivot_data: pd.DataFrame) -> pd.DataFrame:
        """Расчет ускорения относительно pandas."""
        # Проверяем наличие колонок для pandas
        mean_cols = [col for col in pivot_data.columns if col[0] == 'mean']
        pandas_col = next((col for col in mean_cols if col[1] == 'pandas'), None)
        
        if pandas_col:
            baseline = pivot_data[pandas_col]
            
            for col in mean_cols:
                lib_name = col[1]
                pivot_data[('speedup', lib_name)] = baseline / pivot_data[col]
                
        return pivot_data
        
    def export_for_visualization(
        self,
        processed_data: ProcessedData,
        output_path: Union[str, Path],
        format: str = "json"
    ) -> None:
        """
        Экспорт обработанных данных для визуализации.
        
        Args:
            processed_data: Обработанные данные
            output_path: Путь для сохранения
            format: Формат экспорта (json или csv)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "json":
            with open(output_path, "w") as f:
                json.dump(processed_data.to_dict(), f, indent=2, default=str)
        elif format == "csv":
            processed_data.data.to_csv(output_path, index=False)
        else:
            raise ValueError(f"Неподдерживаемый формат: {format}")
            
        self.logger.info(f"Данные экспортированы в {output_path}")
