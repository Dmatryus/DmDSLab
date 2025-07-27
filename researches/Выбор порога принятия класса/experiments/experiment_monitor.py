"""
Модуль для мониторинга и визуализации хода экспериментов в реальном времени.
Позволяет отслеживать прогресс, промежуточные результаты и выявлять проблемы.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import time
from datetime import datetime, timedelta
from pathlib import Path
import pickle
import json
from collections import defaultdict
from IPython.display import clear_output, display
import warnings

warnings.filterwarnings('ignore')


class ExperimentMonitor:
    """Класс для мониторинга экспериментов в реальном времени."""
    
    def __init__(self, results_dir: str = 'experiment_results', update_interval: int = 5):
        """
        Args:
            results_dir: Директория с результатами
            update_interval: Интервал обновления в секундах
        """
        self.results_dir = Path(results_dir)
        self.update_interval = update_interval
        self.start_time = None
        self.total_experiments = 0
        self.completed_experiments = 0
        
        # Для хранения промежуточных результатов
        self.results_cache = []
        self.metrics_history = defaultdict(list)
        
        # Настройка визуализации
        plt.style.use('seaborn-v0_8-darkgrid')
        self.fig = None
        
    def start_monitoring(self, total_experiments: int):
        """Начало мониторинга."""
        self.start_time = datetime.now()
        self.total_experiments = total_experiments
        self.completed_experiments = 0
        
        print(f"Начало мониторинга эксперимента")
        print(f"Всего экспериментов: {total_experiments}")
        print(f"Время начала: {self.start_time}")
        print("-" * 60)
        
    def update(self):
        """Обновление состояния мониторинга."""
        # Загружаем новые результаты
        new_results = self._load_new_results()
        
        if new_results:
            self.results_cache.extend(new_results)
            self.completed_experiments = len(self.results_cache)
            
            # Обновляем историю метрик
            self._update_metrics_history(new_results)
            
            # Отображаем прогресс
            self._display_progress()
            
            # Обновляем визуализации
            self._update_visualizations()
    
    def _load_new_results(self) -> List:
        """Загрузка новых результатов из директории."""
        new_results = []
        raw_dir = self.results_dir / 'raw'
        
        if not raw_dir.exists():
            return new_results
        
        # Получаем список уже загруженных файлов
        loaded_files = {r.additional_info.get('filename', '') for r in self.results_cache}
        
        # Загружаем новые файлы
        for file_path in raw_dir.glob('*.pkl'):
            if file_path.name not in loaded_files:
                try:
                    with open(file_path, 'rb') as f:
                        result = pickle.load(f)
                        result.additional_info['filename'] = file_path.name
                        new_results.append(result)
                except:
                    continue
        
        return new_results
    
    def _update_metrics_history(self, new_results: List):
        """Обновление истории метрик."""
        for result in new_results:
            # Ключ для группировки
            key = f"{result.model_name}_{result.threshold_method}"
            
            # Вычисляем улучшение
            improvement = (result.pseudo_metrics['accuracy'] - 
                         result.baseline_metrics['accuracy'])
            
            self.metrics_history[key].append({
                'dataset': result.dataset_name,
                'improvement': improvement,
                'selection_ratio': result.selection_ratio,
                'threshold': result.optimal_threshold,
                'pseudo_accuracy': result.pseudo_accuracy
            })
    
    def _display_progress(self):
        """Отображение прогресса выполнения."""
        clear_output(wait=True)
        
        # Прогресс
        progress = self.completed_experiments / self.total_experiments * 100
        elapsed_time = datetime.now() - self.start_time
        
        # Оценка оставшегося времени
        if self.completed_experiments > 0:
            time_per_exp = elapsed_time.total_seconds() / self.completed_experiments
            remaining_exps = self.total_experiments - self.completed_experiments
            eta = timedelta(seconds=time_per_exp * remaining_exps)
        else:
            eta = timedelta(seconds=0)
        
        print(f"{'='*60}")
        print(f"МОНИТОРИНГ ЭКСПЕРИМЕНТА")
        print(f"{'='*60}")
        print(f"Прогресс: {self.completed_experiments}/{self.total_experiments} ({progress:.1f}%)")
        print(f"Прошло времени: {elapsed_time}")
        print(f"Осталось времени: {eta}")
        print(f"{'='*60}\n")
        
        # Текущая статистика
        if self.results_cache:
            self._display_current_stats()
    
    def _display_current_stats(self):
        """Отображение текущей статистики."""
        df = pd.DataFrame([{
            'dataset': r.dataset_name,
            'model': r.model_name,
            'method': r.threshold_method,
            'baseline_acc': r.baseline_metrics['accuracy'],
            'pseudo_acc': r.pseudo_metrics['accuracy'],
            'improvement': r.pseudo_metrics['accuracy'] - r.baseline_metrics['accuracy'],
            'selection_ratio': r.selection_ratio,
            'threshold': r.optimal_threshold
        } for r in self.results_cache])
        
        print("ТЕКУЩИЕ РЕЗУЛЬТАТЫ:")
        print("-" * 60)
        
        # Лучшие методы по улучшению
        best_by_improvement = df.groupby('method')['improvement'].agg(['mean', 'std', 'count'])
        best_by_improvement = best_by_improvement.sort_values('mean', ascending=False)
        
        print("\nЛучшие методы по среднему улучшению accuracy:")
        print(best_by_improvement.round(4))
        
        # Статистика по моделям
        model_stats = df.groupby('model').agg({
            'baseline_acc': 'mean',
            'improvement': ['mean', 'std']
        }).round(4)
        
        print("\nСтатистика по моделям:")
        print(model_stats)
        
        # Проблемные случаи
        problems = df[df['improvement'] < -0.01]
        if len(problems) > 0:
            print(f"\n⚠️  ВНИМАНИЕ: Обнаружено {len(problems)} случаев ухудшения качества!")
            print(problems[['dataset', 'model', 'method', 'improvement']].head())
    
    def _update_visualizations(self):
        """Обновление визуализаций."""
        if len(self.results_cache) < 10:  # Ждем накопления данных
            return
        
        # Создаем фигуру если еще нет
        if self.fig is None:
            self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 10))
            self.fig.suptitle('Мониторинг эксперимента', fontsize=16)
        
        # Очищаем оси
        for ax in self.axes.flat:
            ax.clear()
        
        # Подготовка данных
        df = pd.DataFrame([{
            'dataset': r.dataset_name,
            'model': r.model_name,
            'method': r.threshold_method,
            'improvement': r.pseudo_metrics['accuracy'] - r.baseline_metrics['accuracy'],
            'selection_ratio': r.selection_ratio,
            'threshold': r.optimal_threshold,
            'pseudo_accuracy': r.pseudo_accuracy if r.pseudo_accuracy else 0
        } for r in self.results_cache])
        
        # График 1: Распределение улучшений по методам
        ax1 = self.axes[0, 0]
        method_improvements = df.groupby('method')['improvement'].apply(list).to_dict()
        ax1.boxplot(method_improvements.values(), labels=method_improvements.keys())
        ax1.set_xlabel('Метод')
        ax1.set_ylabel('Улучшение accuracy')
        ax1.set_title('Распределение улучшений по методам')
        ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax1.tick_params(axis='x', rotation=45)
        
        # График 2: Зависимость улучшения от доли отобранных примеров
        ax2 = self.axes[0, 1]
        scatter = ax2.scatter(df['selection_ratio'], df['improvement'], 
                            c=df['method'].astype('category').cat.codes, 
                            alpha=0.6, cmap='tab10')
        ax2.set_xlabel('Доля отобранных примеров')
        ax2.set_ylabel('Улучшение accuracy')
        ax2.set_title('Улучшение vs Доля отбора')
        ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        # График 3: Точность псевдо-меток
        ax3 = self.axes[1, 0]
        if df['pseudo_accuracy'].sum() > 0:  # Если есть данные о точности
            pseudo_acc_by_method = df[df['pseudo_accuracy'] > 0].groupby('method')['pseudo_accuracy']
            ax3.bar(pseudo_acc_by_method.mean().index, 
                   pseudo_acc_by_method.mean().values,
                   yerr=pseudo_acc_by_method.std().values)
            ax3.set_xlabel('Метод')
            ax3.set_ylabel('Точность псевдо-меток')
            ax3.set_title('Качество псевдо-разметки')
            ax3.tick_params(axis='x', rotation=45)
        
        # График 4: Временная динамика
        ax4 = self.axes[1, 1]
        progress_data = []
        for i in range(0, len(self.results_cache), max(1, len(self.results_cache)//20)):
            subset = self.results_cache[:i+1]
            avg_improvement = np.mean([r.pseudo_metrics['accuracy'] - r.baseline_metrics['accuracy'] 
                                     for r in subset])
            progress_data.append((i+1, avg_improvement))
        
        if progress_data:
            x, y = zip(*progress_data)
            ax4.plot(x, y, 'b-', linewidth=2)
            ax4.fill_between(x, y, alpha=0.3)
            ax4.set_xlabel('Количество экспериментов')
            ax4.set_ylabel('Среднее улучшение')
            ax4.set_title('Динамика среднего улучшения')
            ax4.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.show()
    
    def generate_report(self) -> Dict:
        """Генерация промежуточного отчета."""
        if not self.results_cache:
            return {}
        
        df = pd.DataFrame([{
            'dataset': r.dataset_name,
            'model': r.model_name,
            'method': r.threshold_method,
            'baseline_accuracy': r.baseline_metrics['accuracy'],
            'pseudo_accuracy': r.pseudo_metrics['accuracy'],
            'improvement': r.pseudo_metrics['accuracy'] - r.baseline_metrics['accuracy'],
            'selection_ratio': r.selection_ratio,
            'threshold': r.optimal_threshold,
            'training_time': r.training_time,
            'pseudo_time': r.pseudo_labeling_time
        } for r in self.results_cache])
        
        report = {
            'summary': {
                'total_experiments': len(self.results_cache),
                'avg_improvement': df['improvement'].mean(),
                'best_improvement': df['improvement'].max(),
                'worst_improvement': df['improvement'].min(),
                'avg_selection_ratio': df['selection_ratio'].mean(),
                'total_time': df['training_time'].sum() + df['pseudo_time'].sum()
            },
            'by_method': df.groupby('method').agg({
                'improvement': ['mean', 'std', 'min', 'max'],
                'selection_ratio': ['mean', 'std'],
                'threshold': ['mean', 'std']
            }).round(4).to_dict(),
            'by_model': df.groupby('model').agg({
                'baseline_accuracy': 'mean',
                'improvement': ['mean', 'std'],
                'training_time': 'mean'
            }).round(4).to_dict(),
            'by_dataset': df.groupby('dataset').agg({
                'baseline_accuracy': 'mean',
                'improvement': ['mean', 'std']
            }).round(4).to_dict(),
            'problematic_cases': df[df['improvement'] < -0.01].to_dict('records')
        }
        
        return report
    
    def save_monitoring_results(self):
        """Сохранение результатов мониторинга."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Сохраняем отчет
        report = self.generate_report()
        report_file = self.results_dir / f'monitoring_report_{timestamp}.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Сохраняем историю метрик
        history_file = self.results_dir / f'metrics_history_{timestamp}.pkl'
        with open(history_file, 'wb') as f:
            pickle.dump(dict(self.metrics_history), f)
        
        print(f"\nРезультаты мониторинга сохранены:")
        print(f"  - Отчет: {report_file}")
        print(f"  - История метрик: {history_file}")


def monitor_experiment(results_dir: str = 'experiment_results', 
                      update_interval: int = 5,
                      total_experiments: Optional[int] = None):
    """
    Запуск мониторинга эксперимента.
    
    Args:
        results_dir: Директория с результатами
        update_interval: Интервал обновления в секундах
        total_experiments: Общее количество экспериментов (если известно)
    """
    monitor = ExperimentMonitor(results_dir, update_interval)
    
    # Если не указано количество, пытаемся определить из конфигурации
    if total_experiments is None:
        config_files = list(Path(results_dir).glob('processed/config_*.json'))
        if config_files:
            with open(config_files[-1], 'r') as f:
                config = json.load(f)
                # Подсчитываем общее количество
                n_datasets = sum(len(v) for v in config['datasets'].values())
                n_models = len(config['models'])
                n_methods = sum(len(v) for v in config['threshold_methods'].values())
                n_runs = config.get('n_runs', 1)
                total_experiments = n_datasets * n_models * n_methods * n_runs
    
    if total_experiments:
        monitor.start_monitoring(total_experiments)
    else:
        print("Не удалось определить общее количество экспериментов")
        return
    
    try:
        while monitor.completed_experiments < monitor.total_experiments:
            monitor.update()
            time.sleep(update_interval)
            
    except KeyboardInterrupt:
        print("\nМониторинг прерван пользователем")
    
    # Сохраняем результаты
    monitor.save_monitoring_results()
    
    # Финальный отчет
    print("\n" + "="*60)
    print("ФИНАЛЬНЫЙ ОТЧЕТ")
    print("="*60)
    
    report = monitor.generate_report()
    print(f"\nВсего выполнено экспериментов: {report['summary']['total_experiments']}")
    print(f"Среднее улучшение: {report['summary']['avg_improvement']:.4f}")
    print(f"Лучшее улучшение: {report['summary']['best_improvement']:.4f}")
    print(f"Худшее улучшение: {report['summary']['worst_improvement']:.4f}")
    print(f"Общее время: {report['summary']['total_time']:.2f} сек")


class ExperimentAnalyzer:
    """Класс для post-hoc анализа завершенных экспериментов."""
    
    def __init__(self, results_dir: str = 'experiment_results'):
        self.results_dir = Path(results_dir)
        self.results_df = None
        self.aggregated = None
        
    def load_results(self, timestamp: Optional[str] = None):
        """Загрузка результатов эксперимента."""
        processed_dir = self.results_dir / 'processed'
        
        if timestamp:
            results_file = processed_dir / f'all_results_{timestamp}.pkl'
            summary_file = processed_dir / f'results_summary_{timestamp}.csv'
        else:
            # Берем последние результаты
            results_files = list(processed_dir.glob('all_results_*.pkl'))
            if not results_files:
                raise FileNotFoundError("Результаты не найдены")
            results_file = max(results_files, key=lambda x: x.stat().st_mtime)
            
            timestamp = results_file.stem.split('_')[-1]
            summary_file = processed_dir / f'results_summary_{timestamp}.csv'
        
        # Загружаем данные
        with open(results_file, 'rb') as f:
            self.raw_results = pickle.load(f)
        
        if summary_file.exists():
            self.results_df = pd.read_csv(summary_file)
        
        print(f"Загружено {len(self.raw_results)} результатов")
        
    def create_detailed_analysis(self):
        """Создание детального анализа результатов."""
        if self.results_df is None:
            print("Сначала загрузите результаты с помощью load_results()")
            return
        
        # Создаем многоуровневые визуализации
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Heatmap улучшений
        ax1 = plt.subplot(3, 3, 1)
        pivot = self.results_df.pivot_table(
            values='improvement_accuracy', 
            index='method', 
            columns='dataset', 
            aggfunc='mean'
        )
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', center=0, ax=ax1)
        ax1.set_title('Среднее улучшение accuracy по методам и датасетам')
        
        # 2. Scatter plot: threshold vs improvement
        ax2 = plt.subplot(3, 3, 2)
        for method in self.results_df['method'].unique():
            method_data = self.results_df[self.results_df['method'] == method]
            ax2.scatter(method_data['threshold'], method_data['improvement_accuracy'], 
                       label=method, alpha=0.6)
        ax2.set_xlabel('Порог')
        ax2.set_ylabel('Улучшение accuracy')
        ax2.set_title('Зависимость улучшения от порога')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 3. Распределение времени выполнения
        ax3 = plt.subplot(3, 3, 3)
        time_data = self.results_df.groupby('method')['pseudo_labeling_time'].mean().sort_values()
        ax3.barh(time_data.index, time_data.values)
        ax3.set_xlabel('Среднее время (сек)')
        ax3.set_title('Время выполнения по методам')
        
        # 4. Box plot улучшений по моделям
        ax4 = plt.subplot(3, 3, 4)
        self.results_df.boxplot(column='improvement_accuracy', by='model', ax=ax4)
        ax4.set_title('Распределение улучшений по моделям')
        ax4.set_xlabel('Модель')
        ax4.set_ylabel('Улучшение accuracy')
        
        # 5. Корреляция между метриками
        ax5 = plt.subplot(3, 3, 5)
        ax5.scatter(self.results_df['selection_ratio'], 
                   self.results_df['pseudo_accuracy'])
        ax5.set_xlabel('Доля отобранных примеров')
        ax5.set_ylabel('Точность псевдо-меток')
        ax5.set_title('Качество vs Количество')
        
        # Добавляем линию тренда
        z = np.polyfit(self.results_df['selection_ratio'].dropna(), 
                      self.results_df['pseudo_accuracy'].dropna(), 1)
        p = np.poly1d(z)
        ax5.plot(self.results_df['selection_ratio'].sort_values(), 
                p(self.results_df['selection_ratio'].sort_values()), 
                "r--", alpha=0.8)
        
        # 6. Статистическая значимость
        ax6 = plt.subplot(3, 3, 6)
        # Считаем количество случаев значимого улучшения (> 1%)
        significant_improvements = self.results_df.groupby('method').apply(
            lambda x: (x['improvement_accuracy'] > 0.01).sum() / len(x) * 100
        )
        ax6.bar(significant_improvements.index, significant_improvements.values)
        ax6.set_ylabel('% случаев улучшения > 1%')
        ax6.set_title('Надежность методов')
        ax6.tick_params(axis='x', rotation=45)
        
        # 7. Radar chart для комплексной оценки
        ax7 = plt.subplot(3, 3, 7, projection='polar')
        methods = self.results_df['method'].unique()[:5]  # Топ 5 методов
        
        metrics = ['Улучшение', 'Стабильность', 'Скорость', 'Отбор', 'Качество']
        
        for method in methods:
            method_data = self.results_df[self.results_df['method'] == method]
            
            values = [
                method_data['improvement_accuracy'].mean() * 10,  # Масштабируем
                1 / (method_data['improvement_accuracy'].std() + 0.001),  # Обратная величина std
                1 / (method_data['pseudo_labeling_time'].mean() + 0.001),  # Обратное время
                method_data['selection_ratio'].mean(),
                method_data['pseudo_accuracy'].mean() if method_data['pseudo_accuracy'].notna().any() else 0
            ]
            
            # Нормализация
            values = [v / max(0.001, max([self.results_df[self.results_df['method'] == m][
                'improvement_accuracy'].mean() * 10 for m in methods])) for v in values]
            
            angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
            values += values[:1]
            angles += angles[:1]
            
            ax7.plot(angles, values, 'o-', linewidth=2, label=method)
            ax7.fill(angles, values, alpha=0.25)
        
        ax7.set_xticks(angles[:-1])
        ax7.set_xticklabels(metrics)
        ax7.set_title('Комплексная оценка методов')
        ax7.legend(bbox_to_anchor=(1.1, 1.1))
        
        # 8. Кумулятивное улучшение
        ax8 = plt.subplot(3, 3, 8)
        for model in self.results_df['model'].unique():
            model_data = self.results_df[self.results_df['model'] == model].sort_values('improvement_accuracy')
            cumulative = np.cumsum(model_data['improvement_accuracy'])
            ax8.plot(range(len(cumulative)), cumulative, label=model)
        
        ax8.set_xlabel('Эксперименты (отсортированы)')
        ax8.set_ylabel('Кумулятивное улучшение')
        ax8.set_title('Накопленное улучшение по моделям')
        ax8.legend()
        
        # 9. Итоговые рекомендации
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')
        
        # Находим лучшие комбинации
        best_overall = self.results_df.loc[self.results_df['improvement_accuracy'].idxmax()]
        best_by_dataset = self.results_df.groupby('dataset').apply(
            lambda x: x.loc[x['improvement_accuracy'].idxmax()]['method']
        )
        
        recommendations = f"""
        РЕКОМЕНДАЦИИ:
        
        Лучший результат:
        • Метод: {best_overall['method']}
        • Модель: {best_overall['model']}
        • Датасет: {best_overall['dataset']}
        • Улучшение: {best_overall['improvement_accuracy']:.3f}
        
        Лучшие методы по датасетам:
        """
        
        for dataset, method in best_by_dataset.items():
            recommendations += f"\n• {dataset}: {method}"
        
        ax9.text(0.1, 0.9, recommendations, transform=ax9.transAxes, 
                fontsize=12, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.show()
        
        return fig


if __name__ == "__main__":
    # Пример использования
    print("Инструменты мониторинга экспериментов")
    print("1. Мониторинг в реальном времени")
    print("2. Анализ завершенного эксперимента")
    
    choice = input("\nВыберите действие (1/2): ").strip()
    
    if choice == '1':
        results_dir = input("Директория с результатами (по умолчанию 'experiment_results'): ").strip()
        results_dir = results_dir or 'experiment_results'
        
        monitor_experiment(results_dir)
        
    elif choice == '2':
        analyzer = ExperimentAnalyzer()
        analyzer.load_results()
        analyzer.create_detailed_analysis()
