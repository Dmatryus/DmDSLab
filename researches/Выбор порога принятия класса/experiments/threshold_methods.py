"""
Модуль с реализацией различных методов выбора порогов для псевдо-разметки.
Включает методы для бинарной и мультиклассовой классификации.
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Optional, Union, List
from sklearn.metrics import precision_recall_curve, roc_curve, f1_score
from scipy.stats import entropy
import warnings
warnings.filterwarnings('ignore')


class ThresholdSelector(ABC):
    """Базовый класс для всех методов выбора порогов."""
    
    def __init__(self, method_name: str, method_type: str = 'universal'):
        """
        Args:
            method_name: Название метода
            method_type: Тип метода ('binary', 'multiclass', 'universal')
        """
        self.method_name = method_name
        self.method_type = method_type
        self.optimal_threshold = None
        self.threshold_history = []
        self.selection_stats = {}
        
    @abstractmethod
    def find_threshold(self, y_true: Optional[np.ndarray] = None, 
                      y_proba: np.ndarray = None, **kwargs) -> float:
        """Поиск оптимального порога."""
        pass
    
    def select_samples(self, y_proba: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """
        Отбор примеров на основе порога.
        
        Args:
            y_proba: Вероятности предсказаний
            threshold: Порог (если None, используется self.optimal_threshold)
            
        Returns:
            Маска отобранных примеров
        """
        if threshold is None:
            threshold = self.optimal_threshold
            
        if threshold is None:
            raise ValueError("Порог не установлен. Вызовите find_threshold() сначала.")
            
        # Для бинарной классификации
        if y_proba.ndim == 1 or (y_proba.ndim == 2 and y_proba.shape[1] == 2):
            if y_proba.ndim == 2:
                proba = y_proba[:, 1]
            else:
                proba = y_proba
            mask = proba >= threshold
            
        # Для мультиклассовой классификации (максимальная вероятность)
        else:
            max_proba = np.max(y_proba, axis=1)
            mask = max_proba >= threshold
            
        # Сохраняем статистику
        self.selection_stats = {
            'threshold': threshold,
            'n_selected': np.sum(mask),
            'selection_ratio': np.mean(mask),
            'total_samples': len(y_proba)
        }
        
        return mask
    
    def get_statistics(self) -> Dict:
        """Получение статистики выбора."""
        return {
            'method_name': self.method_name,
            'method_type': self.method_type,
            'optimal_threshold': self.optimal_threshold,
            'selection_stats': self.selection_stats,
            'threshold_history': self.threshold_history
        }


# ========== Методы для бинарной классификации ==========

class F1OptimizationThreshold(ThresholdSelector):
    """Оптимизация порога по F1-мере."""
    
    def __init__(self):
        super().__init__("F1 Optimization", "binary")
        
    def find_threshold(self, y_true: np.ndarray, y_proba: np.ndarray, **kwargs) -> float:
        """Находит порог, максимизирующий F1-score."""
        if y_proba.ndim == 2:
            y_proba = y_proba[:, 1]
            
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
        
        # F1-scores для каждого порога
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        
        # Находим оптимальный порог (исключаем последний элемент)
        optimal_idx = np.argmax(f1_scores[:-1])
        self.optimal_threshold = float(thresholds[optimal_idx])
        
        # Сохраняем историю
        self.threshold_history.append({
            'threshold': self.optimal_threshold,
            'f1_score': f1_scores[optimal_idx],
            'precision': precisions[optimal_idx],
            'recall': recalls[optimal_idx]
        })
        
        return self.optimal_threshold


class YoudensJThreshold(ThresholdSelector):
    """Статистика Юдена (Youden's J statistic)."""
    
    def __init__(self):
        super().__init__("Youden's J", "binary")
        
    def find_threshold(self, y_true: np.ndarray, y_proba: np.ndarray, **kwargs) -> float:
        """Максимизирует сумму чувствительности и специфичности."""
        if y_proba.ndim == 2:
            y_proba = y_proba[:, 1]
            
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        
        # Youden's J = Sensitivity + Specificity - 1 = TPR - FPR
        j_scores = tpr - fpr
        
        optimal_idx = np.argmax(j_scores)
        self.optimal_threshold = float(thresholds[optimal_idx])
        
        self.threshold_history.append({
            'threshold': self.optimal_threshold,
            'j_score': j_scores[optimal_idx],
            'sensitivity': tpr[optimal_idx],
            'specificity': 1 - fpr[optimal_idx]
        })
        
        return self.optimal_threshold


class CostSensitiveThreshold(ThresholdSelector):
    """Учет стоимости ошибок."""
    
    def __init__(self, cost_fp: float = 1.0, cost_fn: float = 10.0):
        super().__init__("Cost Sensitive", "binary")
        self.cost_fp = cost_fp
        self.cost_fn = cost_fn
        
    def find_threshold(self, y_true: np.ndarray, y_proba: np.ndarray, **kwargs) -> float:
        """Минимизирует взвешенную стоимость ошибок."""
        if y_proba.ndim == 2:
            y_proba = y_proba[:, 1]
            
        thresholds = np.linspace(0, 1, 1000)
        costs = []
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            
            # Подсчет ошибок
            fp = np.sum((y_pred == 1) & (y_true == 0))
            fn = np.sum((y_pred == 0) & (y_true == 1))
            
            # Общая стоимость
            total_cost = fp * self.cost_fp + fn * self.cost_fn
            costs.append(total_cost)
        
        optimal_idx = np.argmin(costs)
        self.optimal_threshold = float(thresholds[optimal_idx])
        
        # Расчет статистики для оптимального порога
        y_pred_optimal = (y_proba >= self.optimal_threshold).astype(int)
        fp_optimal = np.sum((y_pred_optimal == 1) & (y_true == 0))
        fn_optimal = np.sum((y_pred_optimal == 0) & (y_true == 1))
        
        self.threshold_history.append({
            'threshold': self.optimal_threshold,
            'total_cost': costs[optimal_idx],
            'false_positives': fp_optimal,
            'false_negatives': fn_optimal,
            'cost_per_sample': costs[optimal_idx] / len(y_true)
        })
        
        return self.optimal_threshold


class PrecisionRecallBalanceThreshold(ThresholdSelector):
    """Баланс между точностью и полнотой."""
    
    def __init__(self, beta: float = 1.0):
        """
        Args:
            beta: Вес полноты относительно точности (beta > 1 - больше вес полноты)
        """
        super().__init__("Precision-Recall Balance", "binary")
        self.beta = beta
        
    def find_threshold(self, y_true: np.ndarray, y_proba: np.ndarray, **kwargs) -> float:
        """Оптимизирует F-beta score."""
        if y_proba.ndim == 2:
            y_proba = y_proba[:, 1]
            
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
        
        # F-beta score
        beta_squared = self.beta ** 2
        f_beta_scores = ((1 + beta_squared) * precisions * recalls) / \
                       (beta_squared * precisions + recalls + 1e-10)
        
        optimal_idx = np.argmax(f_beta_scores[:-1])
        self.optimal_threshold = float(thresholds[optimal_idx])
        
        self.threshold_history.append({
            'threshold': self.optimal_threshold,
            'f_beta_score': f_beta_scores[optimal_idx],
            'precision': precisions[optimal_idx],
            'recall': recalls[optimal_idx],
            'beta': self.beta
        })
        
        return self.optimal_threshold


# ========== Методы для мультиклассовой классификации ==========

class EntropyThreshold(ThresholdSelector):
    """Энтропийный метод."""
    
    def __init__(self, max_entropy: float = 0.5):
        super().__init__("Entropy-based", "multiclass")
        self.max_entropy = max_entropy
        
    def find_threshold(self, y_true: Optional[np.ndarray] = None, 
                      y_proba: np.ndarray = None, **kwargs) -> float:
        """Порог устанавливается как max_entropy."""
        self.optimal_threshold = self.max_entropy
        return self.optimal_threshold
    
    def select_samples(self, y_proba: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """Отбирает примеры с энтропией ниже порога."""
        if threshold is None:
            threshold = self.optimal_threshold
            
        # Вычисляем энтропию для каждого примера
        # Избегаем log(0)
        proba_clipped = np.clip(y_proba, 1e-10, 1)
        sample_entropy = -np.sum(proba_clipped * np.log(proba_clipped), axis=1)
        
        # Отбираем примеры с низкой энтропией
        mask = sample_entropy < threshold
        
        self.selection_stats = {
            'threshold': threshold,
            'n_selected': np.sum(mask),
            'selection_ratio': np.mean(mask),
            'mean_entropy': np.mean(sample_entropy),
            'median_entropy': np.median(sample_entropy)
        }
        
        return mask


class MarginThreshold(ThresholdSelector):
    """Метод отрыва (margin)."""
    
    def __init__(self, min_margin: float = 0.2):
        super().__init__("Margin-based", "multiclass")
        self.min_margin = min_margin
        
    def find_threshold(self, y_true: Optional[np.ndarray] = None, 
                      y_proba: np.ndarray = None, **kwargs) -> float:
        """Порог устанавливается как min_margin."""
        self.optimal_threshold = self.min_margin
        return self.optimal_threshold
    
    def select_samples(self, y_proba: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """Отбирает примеры с margin выше порога."""
        if threshold is None:
            threshold = self.optimal_threshold
            
        # Сортируем вероятности в убывающем порядке
        sorted_proba = np.sort(y_proba, axis=1)[:, ::-1]
        
        # Margin = разность между top-2 вероятностями
        margins = sorted_proba[:, 0] - sorted_proba[:, 1]
        
        mask = margins >= threshold
        
        self.selection_stats = {
            'threshold': threshold,
            'n_selected': np.sum(mask),
            'selection_ratio': np.mean(mask),
            'mean_margin': np.mean(margins),
            'median_margin': np.median(margins)
        }
        
        return mask


class TopKConfidenceThreshold(ThresholdSelector):
    """Top-K уверенность."""
    
    def __init__(self, k: int = 3, min_confidence: float = 0.8):
        super().__init__(f"Top-{k} Confidence", "multiclass")
        self.k = k
        self.min_confidence = min_confidence
        
    def find_threshold(self, y_true: Optional[np.ndarray] = None, 
                      y_proba: np.ndarray = None, **kwargs) -> float:
        """Порог для суммы top-k вероятностей."""
        self.optimal_threshold = self.min_confidence
        return self.optimal_threshold
    
    def select_samples(self, y_proba: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """Отбирает примеры где сумма top-k вероятностей выше порога."""
        if threshold is None:
            threshold = self.optimal_threshold
            
        # Берем top-k вероятностей для каждого примера
        k = min(self.k, y_proba.shape[1])
        top_k_proba = np.sort(y_proba, axis=1)[:, -k:]
        top_k_sum = np.sum(top_k_proba, axis=1)
        
        mask = top_k_sum >= threshold
        
        self.selection_stats = {
            'threshold': threshold,
            'k': k,
            'n_selected': np.sum(mask),
            'selection_ratio': np.mean(mask),
            'mean_top_k_sum': np.mean(top_k_sum),
            'median_top_k_sum': np.median(top_k_sum)
        }
        
        return mask


class TemperatureCalibrationThreshold(ThresholdSelector):
    """Температурная калибровка с порогом."""
    
    def __init__(self, temperature: float = 1.5, confidence_threshold: float = 0.8):
        super().__init__("Temperature Calibration", "multiclass")
        self.temperature = temperature
        self.confidence_threshold = confidence_threshold
        
    def find_threshold(self, y_true: np.ndarray, y_proba: np.ndarray, **kwargs) -> float:
        """Оптимизирует температуру на валидационных данных."""
        # Здесь можно реализовать поиск оптимальной температуры
        # Для простоты используем заданную температуру
        self.optimal_threshold = self.confidence_threshold
        return self.optimal_threshold
    
    def select_samples(self, y_proba: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """Применяет температурное масштабирование и отбирает уверенные примеры."""
        if threshold is None:
            threshold = self.optimal_threshold
            
        # Применяем температурное масштабирование
        # Предполагаем, что y_proba уже softmax вероятности
        # Для корректного применения нужны логиты, но мы аппроксимируем
        log_proba = np.log(np.clip(y_proba, 1e-10, 1))
        scaled_log_proba = log_proba / self.temperature
        
        # Применяем softmax
        exp_scaled = np.exp(scaled_log_proba - np.max(scaled_log_proba, axis=1, keepdims=True))
        calibrated_proba = exp_scaled / np.sum(exp_scaled, axis=1, keepdims=True)
        
        # Отбираем по максимальной вероятности
        max_proba = np.max(calibrated_proba, axis=1)
        mask = max_proba >= threshold
        
        self.selection_stats = {
            'threshold': threshold,
            'temperature': self.temperature,
            'n_selected': np.sum(mask),
            'selection_ratio': np.mean(mask),
            'mean_max_proba': np.mean(max_proba)
        }
        
        return mask


# ========== Универсальные методы ==========

class PercentileThreshold(ThresholdSelector):
    """Процентильный метод."""
    
    def __init__(self, percentile: float = 80):
        super().__init__("Percentile-based", "universal")
        self.percentile = percentile
        
    def find_threshold(self, y_true: Optional[np.ndarray] = None, 
                      y_proba: np.ndarray = None, **kwargs) -> float:
        """Находит порог на заданном процентиле."""
        if y_proba is None:
            raise ValueError("y_proba обязателен для процентильного метода")
            
        # Для мультиклассовой берем максимальные вероятности
        if y_proba.ndim > 1 and y_proba.shape[1] > 2:
            proba_values = np.max(y_proba, axis=1)
        elif y_proba.ndim == 2:
            proba_values = y_proba[:, 1]
        else:
            proba_values = y_proba
            
        self.optimal_threshold = float(np.percentile(proba_values, self.percentile))
        
        self.threshold_history.append({
            'threshold': self.optimal_threshold,
            'percentile': self.percentile,
            'n_samples': len(proba_values)
        })
        
        return self.optimal_threshold


class FixedThreshold(ThresholdSelector):
    """Фиксированный порог."""
    
    def __init__(self, threshold: float = 0.9):
        super().__init__("Fixed", "universal")
        self.fixed_threshold = threshold
        
    def find_threshold(self, y_true: Optional[np.ndarray] = None, 
                      y_proba: np.ndarray = None, **kwargs) -> float:
        """Возвращает фиксированный порог."""
        self.optimal_threshold = self.fixed_threshold
        return self.optimal_threshold


class AdaptiveThreshold(ThresholdSelector):
    """Адаптивный порог с EMA."""
    
    def __init__(self, initial_threshold: float = 0.9, momentum: float = 0.999):
        super().__init__("Adaptive EMA", "universal")
        self.current_threshold = initial_threshold
        self.momentum = momentum
        self.update_count = 0
        
    def find_threshold(self, y_true: Optional[np.ndarray] = None, 
                      y_proba: np.ndarray = None, **kwargs) -> float:
        """Обновляет порог на основе текущих предсказаний."""
        if y_proba is None:
            self.optimal_threshold = self.current_threshold
            return self.optimal_threshold
            
        # Для мультиклассовой берем максимальные вероятности
        if y_proba.ndim > 1 and y_proba.shape[1] > 2:
            proba_values = np.max(y_proba, axis=1)
        elif y_proba.ndim == 2:
            proba_values = y_proba[:, 1]
        else:
            proba_values = y_proba
            
        # Обновляем порог через EMA
        mean_confidence = np.mean(proba_values)
        
        if self.update_count == 0:
            self.current_threshold = mean_confidence
        else:
            self.current_threshold = (self.momentum * self.current_threshold + 
                                    (1 - self.momentum) * mean_confidence)
        
        self.update_count += 1
        self.optimal_threshold = self.current_threshold
        
        self.threshold_history.append({
            'threshold': self.optimal_threshold,
            'update_count': self.update_count,
            'mean_confidence': mean_confidence
        })
        
        return self.optimal_threshold


class MonteCarloDropoutThreshold(ThresholdSelector):
    """Monte Carlo Dropout для оценки неопределенности."""
    
    def __init__(self, uncertainty_threshold: float = 0.1, n_iterations: int = 50):
        super().__init__("MC Dropout", "universal")
        self.uncertainty_threshold = uncertainty_threshold
        self.n_iterations = n_iterations
        
    def find_threshold(self, y_true: Optional[np.ndarray] = None, 
                      y_proba: np.ndarray = None, **kwargs) -> float:
        """Устанавливает порог неопределенности."""
        self.optimal_threshold = self.uncertainty_threshold
        return self.optimal_threshold
    
    def select_samples_with_uncertainty(self, predictions_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Отбирает примеры на основе неопределенности из множественных предсказаний.
        
        Args:
            predictions_list: Список предсказаний от n_iterations прогонов
            
        Returns:
            mask: Маска отобранных примеров
            uncertainty: Неопределенность для каждого примера
        """
        predictions = np.array(predictions_list)
        
        # Средние предсказания
        mean_proba = np.mean(predictions, axis=0)
        
        # Эпистемическая неопределенность (дисперсия предсказаний)
        if mean_proba.ndim > 1:
            # Мультиклассовая - берем максимальную дисперсию по классам
            epistemic_uncertainty = np.max(np.var(predictions, axis=0), axis=1)
        else:
            epistemic_uncertainty = np.var(predictions, axis=0)
        
        # Отбираем примеры с низкой неопределенностью
        mask = epistemic_uncertainty < self.optimal_threshold
        
        self.selection_stats = {
            'threshold': self.optimal_threshold,
            'n_selected': np.sum(mask),
            'selection_ratio': np.mean(mask),
            'mean_uncertainty': np.mean(epistemic_uncertainty),
            'median_uncertainty': np.median(epistemic_uncertainty)
        }
        
        return mask, epistemic_uncertainty


# ========== Фабрика методов ==========

class ThresholdMethodFactory:
    """Фабрика для создания методов выбора порогов."""
    
    # Регистрация всех методов
    METHODS = {
        # Бинарные методы
        'f1_optimization': F1OptimizationThreshold,
        'youden': YoudensJThreshold,
        'cost_sensitive': CostSensitiveThreshold,
        'precision_recall_balance': PrecisionRecallBalanceThreshold,
        
        # Мультиклассовые методы
        'entropy': EntropyThreshold,
        'margin': MarginThreshold,
        'top_k': TopKConfidenceThreshold,
        'temperature': TemperatureCalibrationThreshold,
        
        # Универсальные методы
        'percentile': PercentileThreshold,
        'fixed': FixedThreshold,
        'adaptive': AdaptiveThreshold,
        'mc_dropout': MonteCarloDropoutThreshold
    }
    
    @classmethod
    def create_method(cls, method_name: str, **kwargs) -> ThresholdSelector:
        """Создает экземпляр метода по имени."""
        if method_name not in cls.METHODS:
            raise ValueError(f"Неизвестный метод: {method_name}. "
                           f"Доступные: {list(cls.METHODS.keys())}")
        
        method_class = cls.METHODS[method_name]
        return method_class(**kwargs)
    
    @classmethod
    def get_methods_for_task(cls, task_type: str) -> List[str]:
        """Получает список методов для типа задачи."""
        methods = []
        for name, method_class in cls.METHODS.items():
            instance = method_class()
            if instance.method_type == task_type or instance.method_type == 'universal':
                methods.append(name)
        return methods
    
    @classmethod
    def create_all_methods(cls, task_type: str = None) -> Dict[str, ThresholdSelector]:
        """Создает все методы для заданного типа задачи."""
        methods = {}
        
        for name, method_class in cls.METHODS.items():
            instance = method_class()
            if task_type is None or instance.method_type == task_type or instance.method_type == 'universal':
                methods[name] = instance
                
        return methods


# ========== Утилиты для анализа ==========

def analyze_threshold_stability(selector: ThresholdSelector, 
                               X_val_list: List[np.ndarray], 
                               y_val_list: List[np.ndarray],
                               model) -> Dict:
    """
    Анализирует стабильность порога на разных валидационных фолдах.
    
    Returns:
        Статистика стабильности порога
    """
    thresholds = []
    
    for X_val, y_val in zip(X_val_list, y_val_list):
        y_proba = model.predict_proba(X_val)
        
        # Находим порог на каждом фолде
        if selector.method_type == 'binary' or 'f1' in selector.method_name.lower():
            threshold = selector.find_threshold(y_val, y_proba)
        else:
            threshold = selector.find_threshold(y_proba=y_proba)
            
        thresholds.append(threshold)
    
    return {
        'mean_threshold': np.mean(thresholds),
        'std_threshold': np.std(thresholds),
        'cv_threshold': np.std(thresholds) / (np.mean(thresholds) + 1e-10),
        'min_threshold': np.min(thresholds),
        'max_threshold': np.max(thresholds),
        'thresholds': thresholds
    }


def compare_threshold_methods(methods: Dict[str, ThresholdSelector],
                            X_test: np.ndarray,
                            y_test: np.ndarray,
                            y_proba: np.ndarray) -> pd.DataFrame:
    """
    Сравнивает различные методы выбора порогов.
    
    Returns:
        DataFrame с результатами сравнения
    """
    results = []
    
    for method_name, selector in methods.items():
        # Находим порог
        if selector.method_type == 'binary' or hasattr(selector, 'find_threshold'):
            try:
                if 'y_true' in selector.find_threshold.__code__.co_varnames:
                    threshold = selector.find_threshold(y_test, y_proba)
                else:
                    threshold = selector.find_threshold(y_proba=y_proba)
            except:
                threshold = selector.find_threshold(y_proba=y_proba)
        else:
            threshold = selector.optimal_threshold or 0.5
            
        # Отбираем примеры
        mask = selector.select_samples(y_proba)
        
        # Вычисляем метрики
        stats = selector.get_statistics()
        
        results.append({
            'method': method_name,
            'threshold': threshold,
            'n_selected': stats['selection_stats']['n_selected'],
            'selection_ratio': stats['selection_stats']['selection_ratio'],
            'method_type': selector.method_type
        })
    
    return pd.DataFrame(results).sort_values('selection_ratio', ascending=False)


if __name__ == "__main__":
    # Тестирование методов
    print("Тестирование методов выбора порогов...")
    
    # Генерируем тестовые данные
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    
    # Бинарная классификация
    X_bin, y_bin = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X_bin, y_bin, test_size=0.3, random_state=42)
    
    # Обучаем модель
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)
    
    # Тестируем бинарные методы
    print("\nБинарные методы:")
    binary_methods = ThresholdMethodFactory.get_methods_for_task('binary')
    
    for method_name in binary_methods:
        selector = ThresholdMethodFactory.create_method(method_name)
        threshold = selector.find_threshold(y_test, y_proba)
        mask = selector.select_samples(y_proba)
        
        print(f"\n{method_name}:")
        print(f"  Порог: {threshold:.3f}")
        print(f"  Отобрано: {np.sum(mask)} из {len(mask)} ({np.mean(mask)*100:.1f}%)")
