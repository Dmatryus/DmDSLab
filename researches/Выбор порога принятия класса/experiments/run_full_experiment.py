"""
Главный скрипт для запуска полного цикла эксперимента:
1. Подготовка данных
2. Проведение экспериментов
3. Мониторинг прогресса
4. Генерация отчета
"""

import os
import sys
import time
import argparse
import yaml
import json
from pathlib import Path
from datetime import datetime
import multiprocessing as mp
from typing import Dict, Optional

# Импорт наших модулей
from data_preparation import DatasetPreparer
from experiment_runner import ExperimentRunner, ExperimentConfig, create_default_config
from experiment_monitor import ExperimentMonitor
from report_generator import generate_experiment_report


class ExperimentOrchestrator:
    """Оркестратор для управления полным циклом эксперимента."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Args:
            config_path: Путь к файлу конфигурации YAML
        """
        self.config_path = config_path
        self.config = None
        self.results_dir = None
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def load_config(self) -> ExperimentConfig:
        """Загрузка конфигурации из файла или создание дефолтной."""
        if self.config_path and os.path.exists(self.config_path):
            print(f"Загрузка конфигурации из {self.config_path}")
            with open(self.config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            
            # Преобразуем в ExperimentConfig
            self.config = ExperimentConfig(**config_dict)
        else:
            print("Использование конфигурации по умолчанию")
            self.config = create_default_config()
        
        # Обновляем директорию результатов с timestamp
        self.results_dir = Path(self.config.results_dir) / f"run_{self.timestamp}"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.config.results_dir = str(self.results_dir)
        
        return self.config
    
    def prepare_data(self):
        """Подготовка датасетов."""
        print("\n" + "="*60)
        print("ЭТАП 1: ПОДГОТОВКА ДАННЫХ")
        print("="*60)
        
        preparer = DatasetPreparer(random_state=self.config.random_state)
        
        # Проверяем, есть ли уже подготовленные данные
        prepared_file = 'prepared_datasets.pkl'
        if os.path.exists(prepared_file):
            print("✓ Обнаружены подготовленные датасеты")
            response = input("Использовать существующие данные? (y/n): ").strip().lower()
            if response == 'y':
                return
        
        # Подготавливаем данные
        print("\nПодготовка датасетов...")
        datasets = preparer.prepare_all_datasets(self.config.datasets)
        
        print(f"\n✓ Подготовлено датасетов: {len(datasets)}")
        for name, info in datasets.items():
            print(f"  - {name}: {info['preprocessing_info']['processed_shape']}")
    
    def run_experiments(self, monitor: bool = True):
        """Запуск экспериментов с опциональным мониторингом."""
        print("\n" + "="*60)
        print("ЭТАП 2: ПРОВЕДЕНИЕ ЭКСПЕРИМЕНТОВ")
        print("="*60)
        
        # Создаем runner
        runner = ExperimentRunner(self.config)
        
        # Подсчитываем общее количество экспериментов
        total_experiments = (
            sum(len(names) for names in self.config.datasets.values()) *
            len(self.config.models) *
            sum(len(methods) for methods in self.config.threshold_methods.values()) *
            self.config.n_runs
        )
        
        print(f"Запланировано экспериментов: {total_experiments}")
        print(f"Результаты будут сохранены в: {self.results_dir}")
        
        if monitor:
            # Запускаем мониторинг в отдельном процессе
            monitor_process = mp.Process(
                target=self._run_monitor,
                args=(str(self.results_dir), total_experiments)
            )
            monitor_process.start()
            time.sleep(2)  # Даем время на запуск монитора
        
        # Запускаем эксперименты
        start_time = time.time()
        try:
            df_results, aggregated = runner.run()
            
            # Сохраняем финальную статистику
            stats = {
                'total_time': time.time() - start_time,
                'total_experiments': total_experiments,
                'completed_experiments': len(runner.results),
                'success_rate': len(runner.results) / total_experiments * 100
            }
            
            stats_file = self.results_dir / 'experiment_stats.json'
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
                
        finally:
            if monitor and monitor_process.is_alive():
                monitor_process.terminate()
                monitor_process.join()
        
        return df_results, aggregated
    
    def _run_monitor(self, results_dir: str, total_experiments: int):
        """Запуск монитора в отдельном процессе."""
        monitor = ExperimentMonitor(results_dir, update_interval=5)
        monitor.start_monitoring(total_experiments)
        
        try:
            while monitor.completed_experiments < total_experiments:
                monitor.update()
                time.sleep(monitor.update_interval)
        except KeyboardInterrupt:
            pass
        finally:
            monitor.save_monitoring_results()
    
    def generate_report(self):
        """Генерация HTML отчета."""
        print("\n" + "="*60)
        print("ЭТАП 3: ГЕНЕРАЦИЯ ОТЧЕТА")
        print("="*60)
        
        report_path = generate_experiment_report(
            str(self.results_dir),
            output_file=self.results_dir / f"report_{self.timestamp}.html"
        )
        
        return report_path
    
    def run_full_pipeline(self, skip_data_prep: bool = False, 
                         skip_experiments: bool = False,
                         monitor: bool = True):
        """Запуск полного пайплайна эксперимента."""
        print(f"\n{'='*60}")
        print(f"ЗАПУСК ПОЛНОГО ЭКСПЕРИМЕНТА")
        print(f"Время начала: {datetime.now()}")
        print(f"{'='*60}")
        
        # Загружаем конфигурацию
        self.load_config()
        
        # Сохраняем конфигурацию
        config_file = self.results_dir / 'experiment_config.yaml'
        with open(config_file, 'w') as f:
            yaml.dump(self.config.__dict__, f, default_flow_style=False)
        
        # 1. Подготовка данных
        if not skip_data_prep:
            self.prepare_data()
        
        # 2. Проведение экспериментов
        if not skip_experiments:
            df_results, aggregated = self.run_experiments(monitor=monitor)
        
        # 3. Генерация отчета
        report_path = self.generate_report()
        
        # Финальная статистика
        print(f"\n{'='*60}")
        print(f"ЭКСПЕРИМЕНТ ЗАВЕРШЕН")
        print(f"Время окончания: {datetime.now()}")
        print(f"{'='*60}")
        print(f"\n📁 Результаты сохранены в: {self.results_dir}")
        print(f"📊 HTML отчет: {report_path}")
        print(f"\n✅ Эксперимент успешно завершен!")
        
        return report_path


def create_config_template(output_path: str = 'experiment_config_template.yaml'):
    """Создание шаблона конфигурационного файла."""
    template = {
        'experiment_name': 'threshold_comparison_experiment',
        'datasets': {
            'binary': ['breast_cancer', 'heart_disease', 'bank_marketing'],
            'multiclass': ['iris', 'wine_quality', 'satellite']
        },
        'models': [
            {
                'name': 'catboost',
                'params': {
                    'iterations': 500,
                    'learning_rate': 0.03,
                    'depth': 6,
                    'verbose': False
                }
            },
            {
                'name': 'random_forest',
                'params': {
                    'n_estimators': 100,
                    'max_depth': 10
                }
            },
            {
                'name': 'lightgbm',
                'params': {
                    'n_estimators': 500,
                    'learning_rate': 0.05,
                    'num_leaves': 31
                }
            },
            {
                'name': 'extra_trees',
                'params': {
                    'n_estimators': 100,
                    'max_depth': 10
                }
            },
            {
                'name': 'logistic_regression',
                'params': {
                    'max_iter': 1000
                }
            }
        ],
        'threshold_methods': {
            'binary': ['f1_optimization', 'youden', 'cost_sensitive', 'precision_recall_balance'],
            'multiclass': ['entropy', 'margin', 'top_k', 'temperature'],
            'universal': ['percentile', 'fixed', 'adaptive']
        },
        'pseudo_labeling_strategy': 'hard',  # 'hard', 'soft', 'iterative', 'self_training'
        'random_state': 42,
        'n_runs': 3,  # Количество запусков для усреднения
        'calibration_method': 'isotonic',  # 'sigmoid', 'isotonic', None
        'save_intermediate': True,
        'results_dir': 'experiment_results'
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(template, f, default_flow_style=False, sort_keys=False)
    
    print(f"Шаблон конфигурации создан: {output_path}")


def main():
    """Главная функция с CLI интерфейсом."""
    parser = argparse.ArgumentParser(
        description='Запуск экспериментов по сравнению методов выбора порогов для псевдо-разметки'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Путь к файлу конфигурации YAML'
    )
    
    parser.add_argument(
        '--mode', '-m',
        choices=['full', 'quick', 'test', 'config'],
        default='full',
        help='Режим запуска: full (полный), quick (быстрый), test (тестовый), config (создать шаблон)'
    )
    
    parser.add_argument(
        '--skip-data-prep',
        action='store_true',
        help='Пропустить этап подготовки данных'
    )
    
    parser.add_argument(
        '--skip-experiments',
        action='store_true',
        help='Пропустить этап экспериментов (только отчет)'
    )
    
    parser.add_argument(
        '--no-monitor',
        action='store_true',
        help='Отключить мониторинг прогресса'
    )
    
    args = parser.parse_args()
    
    # Режим создания конфигурации
    if args.mode == 'config':
        create_config_template()
        return
    
    # Создаем оркестратор
    orchestrator = ExperimentOrchestrator(args.config)
    
    # Настройка для разных режимов
    if args.mode == 'test':
        # Минимальная конфигурация для тестирования
        orchestrator.config = ExperimentConfig(
            datasets={'binary': ['breast_cancer'], 'multiclass': ['iris']},
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
    elif args.mode == 'quick':
        # Быстрый режим - меньше моделей и методов
        if not args.config:
            orchestrator.config = create_default_config()
            orchestrator.config.n_runs = 1
            orchestrator.config.models = orchestrator.config.models[:2]  # Только 2 модели
            orchestrator.config.threshold_methods = {
                'binary': ['f1_optimization', 'youden'],
                'multiclass': ['entropy'],
                'universal': ['percentile']
            }
    
    # Запуск эксперимента
    try:
        orchestrator.run_full_pipeline(
            skip_data_prep=args.skip_data_prep,
            skip_experiments=args.skip_experiments,
            monitor=not args.no_monitor
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Эксперимент прерван пользователем")
    except Exception as e:
        print(f"\n\n❌ Ошибка при выполнении эксперимента: {e}")
        raise


if __name__ == "__main__":
    # Интерактивный режим если запущен без аргументов
    if len(sys.argv) == 1:
        print("🔬 Система экспериментов по выбору порогов для псевдо-разметки")
        print("="*60)
        print("\nВыберите режим работы:")
        print("1. Полный эксперимент")
        print("2. Быстрый эксперимент (сокращенный набор)")
        print("3. Тестовый запуск (минимальный)")
        print("4. Создать шаблон конфигурации")
        print("5. Запустить с существующей конфигурацией")
        print("6. Выход")
        
        choice = input("\nВаш выбор (1-6): ").strip()
        
        if choice == '1':
            orchestrator = ExperimentOrchestrator()
            orchestrator.run_full_pipeline()
        
        elif choice == '2':
            orchestrator = ExperimentOrchestrator()
            orchestrator.config = create_default_config()
            orchestrator.config.n_runs = 1
            orchestrator.config.models = orchestrator.config.models[:2]
            orchestrator.config.threshold_methods = {
                'binary': ['f1_optimization', 'youden'],
                'multiclass': ['entropy'],
                'universal': ['percentile']
            }
            orchestrator.run_full_pipeline()
        
        elif choice == '3':
            orchestrator = ExperimentOrchestrator()
            orchestrator.config = ExperimentConfig(
                datasets={'binary': ['breast_cancer'], 'multiclass': ['iris']},
                models=[
                    {'name': 'random_forest', 'params': {'n_estimators': 50}}
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
            orchestrator.run_full_pipeline()
        
        elif choice == '4':
            create_config_template()
        
        elif choice == '5':
            config_path = input("Путь к файлу конфигурации: ").strip()
            if os.path.exists(config_path):
                orchestrator = ExperimentOrchestrator(config_path)
                orchestrator.run_full_pipeline()
            else:
                print(f"❌ Файл не найден: {config_path}")
        
        elif choice == '6':
            print("До свидания!")
            sys.exit(0)
        
        else:
            print("❌ Неверный выбор")
    else:
        # Запуск через аргументы командной строки
        main()
