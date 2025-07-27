"""
Основной модуль для запуска экспериментов по сравнению методов выбора порогов.
Координирует работу всех компонентов и сохраняет результаты.
"""

import numpy as np
import pandas as pd
import pickle
import json
import time
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from tqdm import tqdm
import warnings
from pathlib import Path

# Импорт наших модулей
from data_preparation import DatasetPreparer
from base_models import ModelFactory, train_model_with_pseudo_labels
from threshold_methods import ThresholdMethodFactory
from pseudo_labeling import create_pseudo_labeling_pipeline, PseudoLabelingResult

warnings.filterwarnings('ignore')


@dataclass
class ExperimentConfig:
    """Конфигурация эксперимента."""
    # Датасеты
    datasets: Dict[str, List[str]]
    
    # Модели
    models: List[Dict[str, Any]]
    
    # Методы выбора порогов
    threshold_methods: Dict[str, List[str]]
    
    # Стратегия псевдо-разметки
    pseudo_labeling_strategy: str = 'hard'
    
    # Параметры эксперимента
    random_state: int = 42
    n_runs: int = 1  # Количество запусков для усреднения
    calibration_method: Optional[str] = 'isotonic'
    
    # Параметры сохранения
    save_intermediate: bool = True
    results_dir: str = 'experiment_results'
    experiment_name: str = 'threshold_comparison'


@dataclass 
class SingleRunResult:
    """Результат одного запуска эксперимента."""
    dataset_name: str
    model_name: str
    threshold_method: str
    run_id: int
    
    # Метрики до псевдо-разметки
    baseline_metrics: Dict[str, float]
    
    # Метрики после псевдо-разметки
    pseudo_metrics: Dict[str, float]
    
    # Статистика псевдо-разметки
    n_selected: int
    selection_ratio: float
    pseudo_accuracy: Optional[float]
    
    # Параметры
    optimal_threshold: float
    
    # Время выполнения
    training_time: float
    pseudo_labeling_time: float
    
    # Дополнительная информация
    additional_info: Dict[str, Any]


class ResultsStorage:
    """Класс для сохранения и загрузки результатов экспериментов."""
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Создаем подпапки
        self.raw_results_dir = self.results_dir / 'raw'
        self.processed_results_dir = self.results_dir / 'processed'
        self.models_dir = self.results_dir / 'models'
        
        for dir_path in [self.raw_results_dir, self.processed_results_dir, self.models_dir]:
            dir_path.mkdir(exist_ok=True)
    
    def save_single_result(self, result: SingleRunResult, timestamp: str):
        """Сохранение результата одного запуска."""
        filename = f"{result.dataset_name}_{result.model_name}_{result.threshold_method}_{result.run_id}_{timestamp}.pkl"
        filepath = self.raw_results_dir / filename
        
        with open(filepath, 'wb') as f:
            pickle.dump(result, f)
    
    def save_all_results(self, results: List[SingleRunResult], config: ExperimentConfig, timestamp: str):
        """Сохранение всех результатов эксперимента."""
        # Сохраняем сырые результаты
        all_results_file = self.processed_results_dir / f"all_results_{timestamp}.pkl"
        with open(all_results_file, 'wb') as f:
            pickle.dump(results, f)
        
        # Сохраняем конфигурацию
        config_file = self.processed_results_dir / f"config_{timestamp}.json"
        with open(config_file, 'w') as f:
            json.dump(asdict(config), f, indent=2)
        
        # Создаем DataFrame для удобного анализа
        results_df = self._results_to_dataframe(results)
        csv_file = self.processed_results_dir / f"results_summary_{timestamp}.csv"
        results_df.to_csv(csv_file, index=False)
        
        # Сохраняем агрегированные метрики
        aggregated = self._aggregate_results(results_df)
        agg_file = self.processed_results_dir / f"aggregated_results_{timestamp}.pkl"
        with open(agg_file, 'wb') as f:
            pickle.dump(aggregated, f)
        
        return results_df, aggregated
    
    def _results_to_dataframe(self, results: List[SingleRunResult]) -> pd.DataFrame:
        """Преобразование результатов в DataFrame."""
        rows = []
        for result in results:
            row = {
                'dataset': result.dataset_name,
                'model': result.model_name,
                'method': result.threshold_method,
                'run_id': result.run_id,
                'threshold': result.optimal_threshold,
                'n_selected': result.n_selected,
                'selection_ratio': result.selection_ratio,
                'pseudo_accuracy': result.pseudo_accuracy,
                'training_time': result.training_time,
                'pseudo_labeling_time': result.pseudo_labeling_time
            }
            
            # Добавляем baseline метрики
            for metric, value in result.baseline_metrics.items():
                row[f'baseline_{metric}'] = value
            
            # Добавляем метрики после псевдо-разметки
            for metric, value in result.pseudo_metrics.items():
                row[f'pseudo_{metric}'] = value
                # Вычисляем улучшение
                if metric in result.baseline_metrics:
                    row[f'improvement_{metric}'] = value - result.baseline_metrics[metric]
                    row[f'improvement_{metric}_pct'] = (
                        (value - result.baseline_metrics[metric]) / result.baseline_metrics[metric] * 100
                        if result.baseline_metrics[metric] > 0 else 0
                    )
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def _aggregate_results(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Агрегация результатов по различным измерениям."""
        aggregated = {}
        
        # Группировка по методам
        aggregated['by_method'] = df.groupby('method').agg({
            'selection_ratio': ['mean', 'std'],
            'pseudo_accuracy': ['mean', 'std'],
            'improvement_accuracy': ['mean', 'std'],
            'improvement_f1_score': ['mean', 'std']
        }).round(4)
        
        # Группировка по моделям
        aggregated['by_model'] = df.groupby('model').agg({
            'baseline_accuracy': 'mean',
            'pseudo_accuracy': ['mean', 'std'],
            'improvement_accuracy': ['mean', 'std'],
            'training_time': 'mean'
        }).round(4)
        
        # Группировка по датасетам
        aggregated['by_dataset'] = df.groupby('dataset').agg({
            'baseline_accuracy': 'mean',
            'improvement_accuracy': ['mean', 'std'],
            'selection_ratio': ['mean', 'std']
        }).round(4)
        
        # Лучшие методы для каждого датасета
        best_methods = df.groupby(['dataset', 'model']).apply(
            lambda x: x.loc[x['pseudo_accuracy'].idxmax()]['method']
        )
        aggregated['best_methods'] = best_methods
        
        return aggregated


class ExperimentRunner:
    """Основной класс для запуска экспериментов."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.storage = ResultsStorage(config.results_dir)
        self.data_preparer = DatasetPreparer(random_state=config.random_state)
        self.results = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def run(self) -> Tuple[pd.DataFrame, Dict]:
        """Запуск полного эксперимента."""
        print(f"{'='*60}")
        print(f"Запуск эксперимента: {self.config.experiment_name}")
        print(f"Время начала: {datetime.now()}")
        print(f"{'='*60}\n")
        
        # Подготовка датасетов
        print("Этап 1: Подготовка данных...")
        datasets = self._prepare_datasets()
        
        # Основной цикл экспериментов
        print("\nЭтап 2: Проведение экспериментов...")
        total_experiments = (
            sum(len(names) for names in self.config.datasets.values()) *
            len(self.config.models) *
            sum(len(methods) for methods in self.config.threshold_methods.values()) *
            self.config.n_runs
        )
        
        with tqdm(total=total_experiments, desc="Прогресс") as pbar:
            for dataset_key, dataset_info in datasets.items():
                self._run_dataset_experiments(dataset_key, dataset_info, pbar)
        
        # Сохранение результатов
        print("\nЭтап 3: Сохранение результатов...")
        df_results, aggregated = self.storage.save_all_results(
            self.results, self.config, self.timestamp
        )
        
        # Вывод сводки
        self._print_summary(aggregated)
        
        print(f"\n{'='*60}")
        print(f"Эксперимент завершен!")
        print(f"Результаты сохранены в: {self.storage.results_dir}")
        print(f"Время окончания: {datetime.now()}")
        print(f"{'='*60}")
        
        return df_results, aggregated
    
    def _prepare_datasets(self) -> Dict[str, Dict]:
        """Подготовка всех датасетов."""
        # Загружаем сохраненные датасеты если есть
        prepared_file = 'prepared_datasets.pkl'
        if os.path.exists(prepared_file):
            print("Загрузка подготовленных датасетов из файла...")
            with open(prepared_file, 'rb') as f:
                return pickle.load(f)
        
        # Иначе готовим заново
        return self.data_preparer.prepare_all_datasets(self.config.datasets)
    
    def _run_dataset_experiments(self, dataset_key: str, dataset_info: Dict, pbar):
        """Запуск экспериментов для одного датасета."""
        print(f"\n\nДатасет: {dataset_key}")
        print(f"Тип задачи: {dataset_info['task_type']}")
        print(f"Размер: {dataset_info['preprocessing_info']['original_shape']}")
        
        splits = dataset_info['splits']
        task_type = dataset_info['task_type']
        
        # Получаем данные
        X_train, y_train = splits['train_labeled']
        X_unlabeled, _ = splits['train_unlabeled']
        X_unlabeled_true, y_unlabeled_true = splits['train_unlabeled_true']
        X_val, y_val = splits['val']
        X_test, y_test = splits['test']
        
        # Определяем методы для типа задачи
        if task_type == 'binary':
            threshold_methods = self.config.threshold_methods.get('binary', [])
        else:
            threshold_methods = self.config.threshold_methods.get('multiclass', [])
        
        # Добавляем универсальные методы
        threshold_methods.extend(self.config.threshold_methods.get('universal', []))
        
        # Цикл по моделям
        for model_config in self.config.models:
            model_name = model_config['name']
            model_params = model_config.get('params', {})
            
            print(f"\n  Модель: {model_name}")
            
            # Цикл по запускам
            for run_id in range(self.config.n_runs):
                # Создаем и обучаем базовую модель
                model = ModelFactory.create_model(
                    model_name,
                    model_params=model_params,
                    calibration_method=self.config.calibration_method,
                    random_state=self.config.random_state + run_id
                )
                
                start_time = time.time()
                model.fit(X_train, y_train, X_val, y_val)
                training_time = time.time() - start_time
                
                # Базовые метрики
                baseline_metrics = model.evaluate(X_test, y_test)
                
                # Цикл по методам выбора порогов
                for method_name in threshold_methods:
                    # Создаем пайплайн псевдо-разметки
                    pipeline = create_pseudo_labeling_pipeline(
                        threshold_method=method_name,
                        labeling_strategy=self.config.pseudo_labeling_strategy
                    )
                    
                    # Запускаем псевдо-разметку
                    start_time = time.time()
                    pseudo_result = pipeline.run(
                        model=model,
                        X_unlabeled=X_unlabeled,
                        X_labeled=X_train,
                        y_labeled=y_train,
                        X_val=X_val,
                        y_val=y_val,
                        true_labels=y_unlabeled_true
                    )
                    pseudo_labeling_time = time.time() - start_time
                    
                    # Дообучаем модель с псевдо-метками
                    if len(pseudo_result.y_pseudo) > 0:
                        new_model = train_model_with_pseudo_labels(
                            model,
                            X_train, y_train,
                            pseudo_result.X_pseudo, pseudo_result.y_pseudo,
                            pseudo_weights=pseudo_result.weights,
                            validation_data=(X_val, y_val)
                        )
                        
                        # Оценка после псевдо-разметки
                        pseudo_metrics = new_model.evaluate(X_test, y_test)
                    else:
                        # Если ничего не отобрано
                        pseudo_metrics = baseline_metrics.copy()
                    
                    # Получаем статистику
                    pipeline_stats = pipeline.get_pipeline_statistics()
                    
                    # Создаем результат
                    result = SingleRunResult(
                        dataset_name=dataset_key,
                        model_name=model_name,
                        threshold_method=method_name,
                        run_id=run_id,
                        baseline_metrics=baseline_metrics,
                        pseudo_metrics=pseudo_metrics,
                        n_selected=pipeline_stats['n_selected'],
                        selection_ratio=pipeline_stats['selection_ratio'],
                        pseudo_accuracy=pipeline_stats['quality_stats']['pseudo_accuracy'] 
                                       if pipeline_stats['quality_stats'] else None,
                        optimal_threshold=pipeline_stats['threshold_stats']['optimal_threshold'],
                        training_time=training_time,
                        pseudo_labeling_time=pseudo_labeling_time,
                        additional_info={
                            'threshold_stats': pipeline_stats['threshold_stats'],
                            'labeling_stats': pipeline_stats['labeling_stats']
                        }
                    )
                    
                    # Сохраняем результат
                    self.results.append(result)
                    if self.config.save_intermediate:
                        self.storage.save_single_result(result, self.timestamp)
                    
                    pbar.update(1)
    
    def _print_summary(self, aggregated: Dict[str, pd.DataFrame]):
        """Вывод сводки результатов."""
        print("\n" + "="*60)
        print("СВОДКА РЕЗУЛЬТАТОВ")
        print("="*60)
        
        print("\n1. Средние показатели по методам:")
        print(aggregated['by_method'])
        
        print("\n2. Средние показатели по моделям:")
        print(aggregated['by_model'])
        
        print("\n3. Лучшие методы для каждой пары (датасет, модель):")
        print(aggregated['best_methods'])
        
        # Находим общий лучший метод
        method_scores = aggregated['by_method']['improvement_accuracy']['mean']
        best_method = method_scores.idxmax()
        best_score = method_scores.max()
        
        print(f"\n4. Лучший метод в среднем: {best_method}")
        print(f"   Среднее улучшение accuracy: {best_score:.3f}")


def create_default_config() -> ExperimentConfig:
    """Создание конфигурации по умолчанию."""
    return ExperimentConfig(
        datasets={
            'binary': ['breast_cancer', 'heart_disease'],
            'multiclass': ['iris', 'wine_quality']
        },
        models=[
            {'name': 'catboost', 'params': {'iterations': 500, 'verbose': False}},
            {'name': 'random_forest', 'params': {'n_estimators': 100}},
            {'name': 'logistic_regression', 'params': {}}
        ],
        threshold_methods={
            'binary': ['f1_optimization', 'youden', 'cost_sensitive'],
            'multiclass': ['entropy', 'margin'],
            'universal': ['percentile', 'fixed', 'adaptive']
        },
        pseudo_labeling_strategy='hard',
        random_state=42,
        n_runs=1,
        calibration_method='isotonic',
        save_intermediate=True,
        results_dir='experiment_results',
        experiment_name='threshold_comparison_v1'
    )


def run_small_experiment():
    """Запуск небольшого тестового эксперимента."""
    config = ExperimentConfig(
        datasets={
            'binary': ['breast_cancer'],
            'multiclass': ['iris']
        },
        models=[
            {'name': 'random_forest', 'params': {'n_estimators': 50}},
            {'name': 'logistic_regression', 'params': {}}
        ],
        threshold_methods={
            'binary': ['f1_optimization'],
            'multiclass': ['entropy'],
            'universal': ['percentile']
        },
        pseudo_labeling_strategy='hard',
        random_state=42,
        n_runs=1,
        calibration_method=None,
        save_intermediate=True,
        results_dir='test_results',
        experiment_name='test_experiment'
    )
    
    runner = ExperimentRunner(config)
    return runner.run()


if __name__ == "__main__":
    # Запуск эксперимента
    print("Выберите режим запуска:")
    print("1. Тестовый эксперимент (быстрый)")
    print("2. Полный эксперимент")
    print("3. Пользовательская конфигурация")
    
    choice = input("\nВаш выбор (1/2/3): ").strip()
    
    if choice == '1':
        print("\nЗапуск тестового эксперимента...")
        df_results, aggregated = run_small_experiment()
        
    elif choice == '2':
        print("\nЗапуск полного эксперимента...")
        config = create_default_config()
        runner = ExperimentRunner(config)
        df_results, aggregated = runner.run()
        
    elif choice == '3':
        print("\nСоздание пользовательской конфигурации...")
        # Здесь можно добавить интерактивное создание конфигурации
        config = create_default_config()
        
        # Изменяем параметры
        n_runs = int(input("Количество запусков (по умолчанию 1): ") or "1")
        config.n_runs = n_runs
        
        runner = ExperimentRunner(config)
        df_results, aggregated = runner.run()
    
    else:
        print("Неверный выбор!")
