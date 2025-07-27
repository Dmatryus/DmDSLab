"""
Модуль для псевдо-разметки неразмеченных данных.
Включает различные стратегии: hard, soft, iterative pseudo-labeling.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')


@dataclass
class PseudoLabelingResult:
    """Результат псевдо-разметки."""
    X_pseudo: np.ndarray  # Отобранные признаки
    y_pseudo: np.ndarray  # Псевдо-метки
    confidence_scores: np.ndarray  # Уверенность в предсказаниях
    selection_mask: np.ndarray  # Маска отбора
    weights: Optional[np.ndarray] = None  # Веса для примеров
    iteration_stats: Optional[List[Dict]] = None  # Статистика по итерациям


class PseudoLabelingStrategy(ABC):
    """Базовый класс для стратегий псевдо-разметки."""
    
    def __init__(self, strategy_name: str):
        self.strategy_name = strategy_name
        self.stats = {}
        
    @abstractmethod
    def generate_pseudo_labels(self, 
                             X_unlabeled: np.ndarray,
                             y_proba: np.ndarray,
                             selection_mask: np.ndarray,
                             **kwargs) -> PseudoLabelingResult:
        """Генерация псевдо-меток."""
        pass
    
    def get_statistics(self) -> Dict:
        """Получение статистики."""
        return self.stats


class HardPseudoLabeling(PseudoLabelingStrategy):
    """Жесткая псевдо-разметка."""
    
    def __init__(self):
        super().__init__("Hard Pseudo-Labeling")
        
    def generate_pseudo_labels(self, 
                             X_unlabeled: np.ndarray,
                             y_proba: np.ndarray,
                             selection_mask: np.ndarray,
                             **kwargs) -> PseudoLabelingResult:
        """
        Присваивает жесткие метки на основе максимальной вероятности.
        
        Args:
            X_unlabeled: Неразмеченные признаки
            y_proba: Вероятности предсказаний
            selection_mask: Маска отобранных примеров
            
        Returns:
            PseudoLabelingResult
        """
        # Отбираем уверенные примеры
        X_pseudo = X_unlabeled[selection_mask]
        proba_selected = y_proba[selection_mask]
        
        # Генерируем жесткие метки
        if proba_selected.ndim == 1:
            # Бинарная классификация
            y_pseudo = (proba_selected >= 0.5).astype(int)
            confidence_scores = np.maximum(proba_selected, 1 - proba_selected)
        else:
            # Мультиклассовая
            y_pseudo = np.argmax(proba_selected, axis=1)
            confidence_scores = np.max(proba_selected, axis=1)
        
        # Статистика
        self.stats = {
            'n_selected': len(X_pseudo),
            'selection_ratio': np.mean(selection_mask),
            'mean_confidence': np.mean(confidence_scores),
            'std_confidence': np.std(confidence_scores),
            'class_distribution': dict(zip(*np.unique(y_pseudo, return_counts=True)))
        }
        
        return PseudoLabelingResult(
            X_pseudo=X_pseudo,
            y_pseudo=y_pseudo,
            confidence_scores=confidence_scores,
            selection_mask=selection_mask
        )


class SoftPseudoLabeling(PseudoLabelingStrategy):
    """Мягкая псевдо-разметка с использованием вероятностей как весов."""
    
    def __init__(self, weight_power: float = 1.0, min_weight: float = 0.1):
        """
        Args:
            weight_power: Степень для вероятностей (больше 1 - усиление уверенных)
            min_weight: Минимальный вес для примера
        """
        super().__init__("Soft Pseudo-Labeling")
        self.weight_power = weight_power
        self.min_weight = min_weight
        
    def generate_pseudo_labels(self, 
                             X_unlabeled: np.ndarray,
                             y_proba: np.ndarray,
                             selection_mask: np.ndarray,
                             **kwargs) -> PseudoLabelingResult:
        """Генерирует псевдо-метки с весами на основе уверенности."""
        # Отбираем уверенные примеры
        X_pseudo = X_unlabeled[selection_mask]
        proba_selected = y_proba[selection_mask]
        
        # Генерируем метки
        if proba_selected.ndim == 1:
            # Бинарная классификация
            y_pseudo = (proba_selected >= 0.5).astype(int)
            confidence_scores = np.maximum(proba_selected, 1 - proba_selected)
        else:
            # Мультиклассовая
            y_pseudo = np.argmax(proba_selected, axis=1)
            confidence_scores = np.max(proba_selected, axis=1)
        
        # Вычисляем веса на основе уверенности
        weights = np.power(confidence_scores, self.weight_power)
        weights = np.maximum(weights, self.min_weight)
        
        # Нормализация весов
        weights = weights / np.mean(weights)
        
        # Статистика
        self.stats = {
            'n_selected': len(X_pseudo),
            'selection_ratio': np.mean(selection_mask),
            'mean_confidence': np.mean(confidence_scores),
            'mean_weight': np.mean(weights),
            'std_weight': np.std(weights),
            'weight_range': (np.min(weights), np.max(weights))
        }
        
        return PseudoLabelingResult(
            X_pseudo=X_pseudo,
            y_pseudo=y_pseudo,
            confidence_scores=confidence_scores,
            selection_mask=selection_mask,
            weights=weights
        )


class IterativePseudoLabeling(PseudoLabelingStrategy):
    """Итеративная псевдо-разметка с постепенным добавлением примеров."""
    
    def __init__(self, 
                 max_iterations: int = 5,
                 min_improvement: float = 0.001,
                 batch_size_ratio: float = 0.1,
                 confidence_growth_rate: float = 0.95):
        """
        Args:
            max_iterations: Максимальное количество итераций
            min_improvement: Минимальное улучшение для продолжения
            batch_size_ratio: Доля примеров для добавления на каждой итерации
            confidence_growth_rate: Коэффициент снижения порога уверенности
        """
        super().__init__("Iterative Pseudo-Labeling")
        self.max_iterations = max_iterations
        self.min_improvement = min_improvement
        self.batch_size_ratio = batch_size_ratio
        self.confidence_growth_rate = confidence_growth_rate
        
    def generate_pseudo_labels(self, 
                             X_unlabeled: np.ndarray,
                             y_proba: np.ndarray,
                             selection_mask: np.ndarray,
                             model=None,
                             X_labeled=None,
                             y_labeled=None,
                             X_val=None,
                             y_val=None,
                             threshold_selector=None,
                             **kwargs) -> PseudoLabelingResult:
        """
        Итеративно добавляет псевдо-размеченные примеры.
        
        Требует дополнительные параметры:
            model: Модель для переобучения
            X_labeled, y_labeled: Исходные размеченные данные
            X_val, y_val: Валидационные данные
            threshold_selector: Метод выбора порога
        """
        if model is None or X_labeled is None:
            # Fallback к обычной hard pseudo-labeling
            return HardPseudoLabeling().generate_pseudo_labels(
                X_unlabeled, y_proba, selection_mask
            )
        
        # Инициализация
        all_X_pseudo = []
        all_y_pseudo = []
        all_confidence = []
        all_weights = []
        iteration_stats = []
        
        # Копируем данные для итеративного процесса
        remaining_X = X_unlabeled.copy()
        remaining_indices = np.arange(len(X_unlabeled))
        current_threshold = threshold_selector.optimal_threshold if threshold_selector else None
        
        # Начальная оценка на валидации
        val_score = model.evaluate(X_val, y_val)['accuracy'] if X_val is not None else 0
        
        for iteration in range(self.max_iterations):
            # Предсказания для оставшихся неразмеченных
            if len(remaining_X) == 0:
                break
                
            current_proba = model.predict_proba(remaining_X)
            
            # Обновляем порог
            if threshold_selector and current_threshold is not None:
                current_threshold *= self.confidence_growth_rate
                threshold_selector.optimal_threshold = current_threshold
            
            # Отбираем примеры
            if threshold_selector:
                current_mask = threshold_selector.select_samples(current_proba)
            else:
                # Простой отбор по уверенности
                if current_proba.ndim == 1:
                    confidence = np.maximum(current_proba, 1 - current_proba)
                else:
                    confidence = np.max(current_proba, axis=1)
                
                n_to_select = int(len(remaining_X) * self.batch_size_ratio)
                n_to_select = max(1, min(n_to_select, len(remaining_X)))
                
                top_indices = np.argsort(confidence)[-n_to_select:]
                current_mask = np.zeros(len(remaining_X), dtype=bool)
                current_mask[top_indices] = True
            
            if not np.any(current_mask):
                break
            
            # Генерируем псевдо-метки для отобранных
            selected_X = remaining_X[current_mask]
            selected_proba = current_proba[current_mask]
            
            if selected_proba.ndim == 1:
                selected_y = (selected_proba >= 0.5).astype(int)
                selected_confidence = np.maximum(selected_proba, 1 - selected_proba)
            else:
                selected_y = np.argmax(selected_proba, axis=1)
                selected_confidence = np.max(selected_proba, axis=1)
            
            # Сохраняем
            all_X_pseudo.append(selected_X)
            all_y_pseudo.append(selected_y)
            all_confidence.append(selected_confidence)
            all_weights.append(np.ones(len(selected_X)) * (0.9 ** iteration))  # Уменьшаем вес с итерациями
            
            # Переобучаем модель
            if len(all_X_pseudo) > 0:
                combined_X_pseudo = np.vstack(all_X_pseudo)
                combined_y_pseudo = np.hstack(all_y_pseudo)
                combined_weights = np.hstack(all_weights)
                
                # Объединяем с исходными данными
                X_train_combined = np.vstack([X_labeled, combined_X_pseudo])
                y_train_combined = np.hstack([y_labeled, combined_y_pseudo])
                weights_combined = np.hstack([
                    np.ones(len(X_labeled)),
                    combined_weights * 0.5  # Псевдо-метки с меньшим весом
                ])
                
                # Переобучаем
                model.fit(X_train_combined, y_train_combined, 
                         X_val=X_val, y_val=y_val,
                         sample_weight=weights_combined)
            
            # Оценка улучшения
            new_val_score = model.evaluate(X_val, y_val)['accuracy'] if X_val is not None else 0
            improvement = new_val_score - val_score
            
            # Статистика итерации
            iter_stats = {
                'iteration': iteration,
                'n_selected': len(selected_X),
                'total_pseudo': len(np.hstack(all_y_pseudo)),
                'mean_confidence': np.mean(selected_confidence),
                'val_score': new_val_score,
                'improvement': improvement,
                'current_threshold': current_threshold
            }
            iteration_stats.append(iter_stats)
            
            # Проверка критерия остановки
            if improvement < self.min_improvement and iteration > 0:
                break
                
            val_score = new_val_score
            
            # Убираем отобранные примеры
            remaining_mask = ~current_mask
            remaining_X = remaining_X[remaining_mask]
            remaining_indices = remaining_indices[remaining_mask]
        
        # Финальный результат
        if len(all_X_pseudo) > 0:
            final_X = np.vstack(all_X_pseudo)
            final_y = np.hstack(all_y_pseudo)
            final_confidence = np.hstack(all_confidence)
            final_weights = np.hstack(all_weights)
            
            # Создаем полную маску отбора
            final_mask = np.zeros(len(X_unlabeled), dtype=bool)
            selected_indices = set(range(len(X_unlabeled))) - set(remaining_indices)
            final_mask[list(selected_indices)] = True
        else:
            final_X = np.array([])
            final_y = np.array([])
            final_confidence = np.array([])
            final_weights = np.array([])
            final_mask = np.zeros(len(X_unlabeled), dtype=bool)
        
        # Общая статистика
        self.stats = {
            'n_iterations': len(iteration_stats),
            'total_selected': len(final_y),
            'selection_ratio': np.mean(final_mask),
            'final_val_score': val_score,
            'total_improvement': val_score - iteration_stats[0]['val_score'] if iteration_stats else 0
        }
        
        return PseudoLabelingResult(
            X_pseudo=final_X,
            y_pseudo=final_y,
            confidence_scores=final_confidence,
            selection_mask=final_mask,
            weights=final_weights,
            iteration_stats=iteration_stats
        )


class SelfTrainingPseudoLabeling(PseudoLabelingStrategy):
    """Self-training с контролем качества псевдо-меток."""
    
    def __init__(self, 
                 quality_threshold: float = 0.9,
                 max_error_rate: float = 0.1,
                 use_ensemble: bool = False):
        """
        Args:
            quality_threshold: Минимальная уверенность для псевдо-метки
            max_error_rate: Максимально допустимая ошибка на контрольной выборке
            use_ensemble: Использовать ансамбль моделей
        """
        super().__init__("Self-Training")
        self.quality_threshold = quality_threshold
        self.max_error_rate = max_error_rate
        self.use_ensemble = use_ensemble
        
    def generate_pseudo_labels(self, 
                             X_unlabeled: np.ndarray,
                             y_proba: np.ndarray,
                             selection_mask: np.ndarray,
                             X_control: Optional[np.ndarray] = None,
                             y_control: Optional[np.ndarray] = None,
                             **kwargs) -> PseudoLabelingResult:
        """
        Генерирует псевдо-метки с контролем качества.
        
        Args:
            X_control, y_control: Контрольная выборка для оценки качества
        """
        # Отбираем примеры с высокой уверенностью
        if y_proba.ndim == 1:
            confidence = np.maximum(y_proba, 1 - y_proba)
            y_pred = (y_proba >= 0.5).astype(int)
        else:
            confidence = np.max(y_proba, axis=1)
            y_pred = np.argmax(y_proba, axis=1)
        
        # Дополнительная фильтрация по качеству
        quality_mask = confidence >= self.quality_threshold
        combined_mask = selection_mask & quality_mask
        
        # Если есть контрольная выборка, проверяем качество
        if X_control is not None and y_control is not None and self.use_ensemble:
            # Здесь можно добавить логику проверки на контрольной выборке
            pass
        
        # Отбираем финальные примеры
        X_pseudo = X_unlabeled[combined_mask]
        y_pseudo = y_pred[combined_mask]
        confidence_scores = confidence[combined_mask]
        
        # Веса на основе уверенности
        weights = confidence_scores / self.quality_threshold
        
        # Статистика
        self.stats = {
            'n_selected': len(X_pseudo),
            'selection_ratio': np.mean(combined_mask),
            'quality_filtered': np.sum(quality_mask) - np.sum(combined_mask),
            'mean_confidence': np.mean(confidence_scores) if len(confidence_scores) > 0 else 0,
            'min_confidence': np.min(confidence_scores) if len(confidence_scores) > 0 else 0
        }
        
        return PseudoLabelingResult(
            X_pseudo=X_pseudo,
            y_pseudo=y_pseudo,
            confidence_scores=confidence_scores,
            selection_mask=combined_mask,
            weights=weights
        )


class PseudoLabelingPipeline:
    """Полный пайплайн псевдо-разметки."""
    
    def __init__(self, 
                 threshold_selector,
                 labeling_strategy: PseudoLabelingStrategy,
                 quality_check: bool = True):
        """
        Args:
            threshold_selector: Метод выбора порога
            labeling_strategy: Стратегия псевдо-разметки
            quality_check: Проверять качество псевдо-меток
        """
        self.threshold_selector = threshold_selector
        self.labeling_strategy = labeling_strategy
        self.quality_check = quality_check
        self.pipeline_stats = {}
        
    def run(self, 
            model,
            X_unlabeled: np.ndarray,
            X_labeled: Optional[np.ndarray] = None,
            y_labeled: Optional[np.ndarray] = None,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            true_labels: Optional[np.ndarray] = None) -> PseudoLabelingResult:
        """
        Запуск полного пайплайна псевдо-разметки.
        
        Args:
            model: Обученная модель
            X_unlabeled: Неразмеченные данные
            X_labeled, y_labeled: Размеченные данные (для итеративных методов)
            X_val, y_val: Валидационные данные
            true_labels: Истинные метки для неразмеченных (для оценки)
            
        Returns:
            PseudoLabelingResult
        """
        # 1. Получаем предсказания
        y_proba = model.predict_proba(X_unlabeled)
        
        # 2. Выбираем порог
        if hasattr(self.threshold_selector, 'find_threshold'):
            if X_val is not None and y_val is not None and self.threshold_selector.method_type == 'binary':
                # Для бинарных методов используем валидацию
                val_proba = model.predict_proba(X_val)
                self.threshold_selector.find_threshold(y_val, val_proba)
            else:
                # Для остальных методов
                self.threshold_selector.find_threshold(y_proba=y_proba)
        
        # 3. Отбираем примеры
        selection_mask = self.threshold_selector.select_samples(y_proba)
        
        # 4. Генерируем псевдо-метки
        if isinstance(self.labeling_strategy, IterativePseudoLabeling):
            # Итеративный метод требует дополнительные параметры
            result = self.labeling_strategy.generate_pseudo_labels(
                X_unlabeled, y_proba, selection_mask,
                model=model,
                X_labeled=X_labeled,
                y_labeled=y_labeled,
                X_val=X_val,
                y_val=y_val,
                threshold_selector=self.threshold_selector
            )
        else:
            result = self.labeling_strategy.generate_pseudo_labels(
                X_unlabeled, y_proba, selection_mask
            )
        
        # 5. Оценка качества псевдо-меток (если есть истинные метки)
        if self.quality_check and true_labels is not None and len(result.y_pseudo) > 0:
            # Находим индексы отобранных примеров
            selected_indices = np.where(result.selection_mask)[0]
            true_selected = true_labels[selected_indices]
            
            # Вычисляем точность псевдо-меток
            pseudo_accuracy = np.mean(result.y_pseudo == true_selected)
            
            # Анализ по классам
            if len(np.unique(result.y_pseudo)) > 2:
                class_accuracies = {}
                for cls in np.unique(result.y_pseudo):
                    cls_mask = result.y_pseudo == cls
                    if np.any(cls_mask):
                        class_accuracies[cls] = np.mean(
                            result.y_pseudo[cls_mask] == true_selected[cls_mask]
                        )
            else:
                class_accuracies = None
            
            quality_stats = {
                'pseudo_accuracy': pseudo_accuracy,
                'n_correct': np.sum(result.y_pseudo == true_selected),
                'n_incorrect': np.sum(result.y_pseudo != true_selected),
                'class_accuracies': class_accuracies
            }
        else:
            quality_stats = None
        
        # Сохраняем общую статистику
        self.pipeline_stats = {
            'threshold_stats': self.threshold_selector.get_statistics(),
            'labeling_stats': self.labeling_strategy.get_statistics(),
            'quality_stats': quality_stats,
            'n_unlabeled': len(X_unlabeled),
            'n_selected': len(result.y_pseudo),
            'selection_ratio': len(result.y_pseudo) / len(X_unlabeled) if len(X_unlabeled) > 0 else 0
        }
        
        return result
    
    def get_pipeline_statistics(self) -> Dict:
        """Получение полной статистики пайплайна."""
        return self.pipeline_stats


def create_pseudo_labeling_pipeline(threshold_method: str,
                                  labeling_strategy: str = 'hard',
                                  **kwargs) -> PseudoLabelingPipeline:
    """
    Создание пайплайна псевдо-разметки.
    
    Args:
        threshold_method: Название метода выбора порога
        labeling_strategy: Стратегия разметки ('hard', 'soft', 'iterative', 'self_training')
        **kwargs: Дополнительные параметры
        
    Returns:
        Настроенный пайплайн
    """
    from threshold_methods import ThresholdMethodFactory
    
    # Создаем селектор порога
    threshold_selector = ThresholdMethodFactory.create_method(threshold_method, **kwargs)
    
    # Создаем стратегию разметки
    strategy_map = {
        'hard': HardPseudoLabeling,
        'soft': SoftPseudoLabeling,
        'iterative': IterativePseudoLabeling,
        'self_training': SelfTrainingPseudoLabeling
    }
    
    if labeling_strategy not in strategy_map:
        raise ValueError(f"Неизвестная стратегия: {labeling_strategy}")
    
    strategy = strategy_map[labeling_strategy]()
    
    return PseudoLabelingPipeline(threshold_selector, strategy)


if __name__ == "__main__":
    # Тестирование модуля
    print("Тестирование модуля псевдо-разметки...")
    
    # Генерируем тестовые данные
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    
    X, y = make_classification(n_samples=2000, n_features=20, n_classes=3, random_state=42)
    
    # Разделяем данные
    X_labeled, X_unlabeled, y_labeled, y_unlabeled = train_test_split(
        X, y, test_size=0.8, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_labeled, y_labeled, test_size=0.2, random_state=42
    )
    
    # Обучаем модель
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Тестируем разные стратегии
    strategies = ['hard', 'soft', 'iterative']
    
    for strategy in strategies:
        print(f"\n{strategy.upper()} псевдо-разметка:")
        
        # Создаем пайплайн
        pipeline = create_pseudo_labeling_pipeline(
            threshold_method='percentile',
            labeling_strategy=strategy,
            percentile=80
        )
        
        # Запускаем
        result = pipeline.run(
            model=model,
            X_unlabeled=X_unlabeled,
            X_labeled=X_train,
            y_labeled=y_train,
            X_val=X_val,
            y_val=y_val,
            true_labels=y_unlabeled  # Для оценки качества
        )
        
        # Выводим статистику
        stats = pipeline.get_pipeline_statistics()
        print(f"  Отобрано: {stats['n_selected']} из {stats['n_unlabeled']} "
              f"({stats['selection_ratio']*100:.1f}%)")
        
        if stats['quality_stats']:
            print(f"  Точность псевдо-меток: {stats['quality_stats']['pseudo_accuracy']:.3f}")
