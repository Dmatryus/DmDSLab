"""
Модуль для расчета описательных статистик результатов бенчмаркинга.
"""

from typing import Dict, List, Union, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy import stats
import warnings

from utils import get_logger


@dataclass
class DescriptiveStats:
    """Описательные статистики для набора данных."""
    count: int
    mean: float
    std: float
    min: float
    max: float
    median: float
    q1: float  # 25-й процентиль
    q3: float  # 75-й процентиль
    iqr: float  # Межквартильный размах
    cv: float  # Коэффициент вариации
    skew: float  # Асимметрия
    kurtosis: float  # Эксцесс
    confidence_interval: Tuple[float, float]  # 95% доверительный интервал
    
    def to_dict(self) -> Dict[str, float]:
        """Преобразование в словарь."""
        return {
            'count': self.count,
            'mean': self.mean,
            'std': self.std,
            'min': self.min,
            'max': self.max,
            'median': self.median,
            'q1': self.q1,
            'q3': self.q3,
            'iqr': self.iqr,
            'cv': self.cv,
            'skew': self.skew,
            'kurtosis': self.kurtosis,
            'ci_lower': self.confidence_interval[0],
            'ci_upper': self.confidence_interval[1]
        }


@dataclass
class NormalityTest:
    """Результаты тестов на нормальность."""
    shapiro_statistic: float
    shapiro_pvalue: float
    shapiro_is_normal: bool
    dagostino_statistic: float
    dagostino_pvalue: float
    dagostino_is_normal: bool
    anderson_statistic: float
    anderson_critical_values: Dict[str, float]
    anderson_is_normal: Dict[str, bool]
    
    @property
    def is_normal(self) -> bool:
        """Общее заключение о нормальности (консервативная оценка)."""
        return self.shapiro_is_normal and self.dagostino_is_normal


class StatisticsCalculator:
    """Класс для расчета статистик результатов бенчмаркинга."""
    
    def __init__(self, confidence_level: float = 0.95, logger=None):
        """
        Инициализация калькулятора статистик.
        
        Args:
            confidence_level: Уровень доверия для интервалов (по умолчанию 0.95)
            logger: Логгер для вывода информации
        """
        self.confidence_level = confidence_level
        self.logger = logger or get_logger(__name__)
    
    def calculate_descriptive_stats(self, 
                                  data: Union[List[float], np.ndarray, pd.Series]) -> DescriptiveStats:
        """
        Расчет описательных статистик.
        
        Args:
            data: Массив данных для анализа
            
        Returns:
            DescriptiveStats: Объект с рассчитанными статистиками
        """
        # Преобразование в numpy array
        data = np.asarray(data)
        
        # Проверка на пустые данные
        if len(data) == 0:
            raise ValueError("Данные не могут быть пустыми")
        
        # Основные статистики
        count = len(data)
        mean = np.mean(data)
        std = np.std(data, ddof=1)  # Используем несмещенную оценку
        min_val = np.min(data)
        max_val = np.max(data)
        median = np.median(data)
        
        # Квартили
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        
        # Коэффициент вариации
        cv = (std / mean * 100) if mean != 0 else float('inf')
        
        # Асимметрия и эксцесс
        skew = stats.skew(data)
        kurt = stats.kurtosis(data)
        
        # Доверительный интервал для среднего
        ci_lower, ci_upper = self._calculate_confidence_interval(data, mean, std)
        
        self.logger.debug(f"Рассчитаны статистики для {count} значений")
        
        return DescriptiveStats(
            count=count,
            mean=mean,
            std=std,
            min=min_val,
            max=max_val,
            median=median,
            q1=q1,
            q3=q3,
            iqr=iqr,
            cv=cv,
            skew=skew,
            kurtosis=kurt,
            confidence_interval=(ci_lower, ci_upper)
        )
    
    def _calculate_confidence_interval(self, 
                                     data: np.ndarray, 
                                     mean: float, 
                                     std: float) -> Tuple[float, float]:
        """
        Расчет доверительного интервала для среднего.
        
        Args:
            data: Массив данных
            mean: Среднее значение
            std: Стандартное отклонение
            
        Returns:
            Tuple[float, float]: Нижняя и верхняя границы интервала
        """
        n = len(data)
        
        if n < 2:
            return (mean, mean)
        
        # Используем t-распределение для малых выборок
        alpha = 1 - self.confidence_level
        t_critical = stats.t.ppf(1 - alpha/2, n - 1)
        margin_error = t_critical * (std / np.sqrt(n))
        
        return (mean - margin_error, mean + margin_error)
    
    def test_normality(self, 
                      data: Union[List[float], np.ndarray, pd.Series],
                      alpha: float = 0.05) -> NormalityTest:
        """
        Тесты на нормальность распределения.
        
        Args:
            data: Массив данных для тестирования
            alpha: Уровень значимости (по умолчанию 0.05)
            
        Returns:
            NormalityTest: Результаты тестов на нормальность
        """
        data = np.asarray(data)
        
        # Минимальный размер выборки для тестов
        if len(data) < 3:
            self.logger.warning("Недостаточно данных для тестов на нормальность")
            return NormalityTest(
                shapiro_statistic=np.nan,
                shapiro_pvalue=np.nan,
                shapiro_is_normal=False,
                dagostino_statistic=np.nan,
                dagostino_pvalue=np.nan,
                dagostino_is_normal=False,
                anderson_statistic=np.nan,
                anderson_critical_values={},
                anderson_is_normal={}
            )
        
        # Тест Шапиро-Уилка
        shapiro_stat, shapiro_p = stats.shapiro(data)
        shapiro_normal = shapiro_p > alpha
        
        # Тест Д'Агостино-Пирсона (K²)
        if len(data) >= 8:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                dagostino_stat, dagostino_p = stats.normaltest(data)
            dagostino_normal = dagostino_p > alpha
        else:
            dagostino_stat, dagostino_p = np.nan, np.nan
            dagostino_normal = False
        
        # Тест Андерсона-Дарлинга
        anderson_result = stats.anderson(data, dist='norm')
        anderson_critical = dict(zip(
            ['15%', '10%', '5%', '2.5%', '1%'],
            anderson_result.critical_values
        ))
        anderson_normal = {
            level: anderson_result.statistic < crit_val
            for level, crit_val in anderson_critical.items()
        }
        
        self.logger.debug(f"Тесты на нормальность: Shapiro p={shapiro_p:.4f}, "
                         f"D'Agostino p={dagostino_p:.4f}")
        
        return NormalityTest(
            shapiro_statistic=shapiro_stat,
            shapiro_pvalue=shapiro_p,
            shapiro_is_normal=shapiro_normal,
            dagostino_statistic=dagostino_stat,
            dagostino_pvalue=dagostino_p,
            dagostino_is_normal=dagostino_normal,
            anderson_statistic=anderson_result.statistic,
            anderson_critical_values=anderson_critical,
            anderson_is_normal=anderson_normal
        )
    
    def calculate_stats_for_groups(self, 
                                 groups: Dict[str, Union[List[float], np.ndarray]]) -> Dict[str, DescriptiveStats]:
        """
        Расчет статистик для нескольких групп данных.
        
        Args:
            groups: Словарь с группами данных (имя -> данные)
            
        Returns:
            Dict[str, DescriptiveStats]: Статистики для каждой группы
        """
        results = {}
        
        for name, data in groups.items():
            try:
                stats = self.calculate_descriptive_stats(data)
                results[name] = stats
                self.logger.info(f"Рассчитаны статистики для {name}: "
                               f"mean={stats.mean:.4f}, std={stats.std:.4f}")
            except Exception as e:
                self.logger.error(f"Ошибка при расчете статистик для {name}: {e}")
                
        return results
    
    def create_summary_table(self, 
                           stats_dict: Dict[str, DescriptiveStats]) -> pd.DataFrame:
        """
        Создание сводной таблицы статистик.
        
        Args:
            stats_dict: Словарь со статистиками для разных групп
            
        Returns:
            pd.DataFrame: Таблица со статистиками
        """
        # Создаем список словарей для DataFrame
        data = []
        for name, stats in stats_dict.items():
            row = stats.to_dict()
            row['name'] = name
            data.append(row)
        
        # Создаем DataFrame
        df = pd.DataFrame(data)
        
        # Переупорядочиваем колонки
        columns_order = ['name', 'count', 'mean', 'std', 'cv', 'min', 
                        'q1', 'median', 'q3', 'max', 'iqr', 
                        'skew', 'kurtosis', 'ci_lower', 'ci_upper']
        
        # Фильтруем только существующие колонки
        columns_order = [col for col in columns_order if col in df.columns]
        df = df[columns_order]
        
        # Форматирование чисел
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col != 'count':
                df[col] = df[col].round(4)
        
        return df
    
    def calculate_relative_metrics(self,
                                 base_stats: DescriptiveStats,
                                 compare_stats: DescriptiveStats) -> Dict[str, float]:
        """
        Расчет относительных метрик между двумя группами.
        
        Args:
            base_stats: Базовые статистики (например, pandas)
            compare_stats: Статистики для сравнения (например, polars)
            
        Returns:
            Dict[str, float]: Относительные метрики
        """
        # Относительная разница в среднем
        mean_diff_pct = ((base_stats.mean - compare_stats.mean) / base_stats.mean * 100 
                        if base_stats.mean != 0 else float('inf'))
        
        # Ускорение (speedup)
        speedup = base_stats.mean / compare_stats.mean if compare_stats.mean != 0 else float('inf')
        
        # Относительная разница в вариабельности
        cv_diff = base_stats.cv - compare_stats.cv
        
        # Перекрытие доверительных интервалов
        ci_overlap = not (base_stats.confidence_interval[1] < compare_stats.confidence_interval[0] or
                         compare_stats.confidence_interval[1] < base_stats.confidence_interval[0])
        
        return {
            'mean_difference_pct': mean_diff_pct,
            'speedup': speedup,
            'cv_difference': cv_diff,
            'ci_overlap': ci_overlap,
            'base_is_faster': base_stats.mean < compare_stats.mean
        }


def format_stats_report(stats: DescriptiveStats, name: str = "Dataset") -> str:
    """
    Форматирование статистик в виде текстового отчета.
    
    Args:
        stats: Объект со статистиками
        name: Название набора данных
        
    Returns:
        str: Форматированный отчет
    """
    report = f"""
=== Статистики для {name} ===
Количество измерений: {stats.count}

Центральные тенденции:
  Среднее: {stats.mean:.4f}
  Медиана: {stats.median:.4f}
  
Разброс:
  Стандартное отклонение: {stats.std:.4f}
  Коэффициент вариации: {stats.cv:.2f}%
  Межквартильный размах: {stats.iqr:.4f}
  
Диапазон:
  Минимум: {stats.min:.4f}
  Q1 (25%): {stats.q1:.4f}
  Q3 (75%): {stats.q3:.4f}
  Максимум: {stats.max:.4f}
  
Форма распределения:
  Асимметрия: {stats.skew:.4f}
  Эксцесс: {stats.kurtosis:.4f}
  
95% доверительный интервал:
  [{stats.confidence_interval[0]:.4f}, {stats.confidence_interval[1]:.4f}]
"""
    return report
