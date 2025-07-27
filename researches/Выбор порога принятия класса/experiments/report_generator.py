"""
Модуль для генерации подробных HTML отчетов по результатам экспериментов.
Создает интерактивные отчеты с визуализациями и статистическим анализом.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats
from jinja2 import Template
import base64
from io import BytesIO
import warnings

warnings.filterwarnings('ignore')


class ReportGenerator:
    """Генератор HTML отчетов для экспериментов по выбору порогов."""
    
    def __init__(self, results_dir: str = 'experiment_results'):
        self.results_dir = Path(results_dir)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results = None
        self.config = None
        self.df_results = None
        self.aggregated = None
        
        # Настройка стилей
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
        
    def load_experiment_results(self, timestamp: Optional[str] = None):
        """Загрузка результатов эксперимента."""
        processed_dir = self.results_dir / 'processed'
        
        if timestamp:
            results_file = processed_dir / f'all_results_{timestamp}.pkl'
            config_file = processed_dir / f'config_{timestamp}.json'
            summary_file = processed_dir / f'results_summary_{timestamp}.csv'
            agg_file = processed_dir / f'aggregated_results_{timestamp}.pkl'
        else:
            # Берем последние результаты
            results_files = list(processed_dir.glob('all_results_*.pkl'))
            if not results_files:
                raise FileNotFoundError("Результаты экспериментов не найдены")
            
            results_file = max(results_files, key=lambda x: x.stat().st_mtime)
            timestamp = results_file.stem.split('_', 2)[-1]
            
            config_file = processed_dir / f'config_{timestamp}.json'
            summary_file = processed_dir / f'results_summary_{timestamp}.csv'
            agg_file = processed_dir / f'aggregated_results_{timestamp}.pkl'
        
        # Загружаем данные
        with open(results_file, 'rb') as f:
            self.results = pickle.load(f)
            
        with open(config_file, 'r') as f:
            self.config = json.load(f)
            
        self.df_results = pd.read_csv(summary_file)
        
        if agg_file.exists():
            with open(agg_file, 'rb') as f:
                self.aggregated = pickle.load(f)
        
        print(f"Загружено {len(self.results)} результатов эксперимента от {timestamp}")
        
    def generate_report(self, output_file: Optional[str] = None):
        """Генерация полного HTML отчета."""
        if self.results is None:
            raise ValueError("Сначала загрузите результаты с помощью load_experiment_results()")
        
        # Генерируем все части отчета
        report_parts = {
            'metadata': self._generate_metadata(),
            'executive_summary': self._generate_executive_summary(),
            'individual_analysis': self._generate_individual_analysis(),
            'comparative_analysis': self._generate_comparative_analysis(),
            'general_conclusions': self._generate_general_conclusions(),
            'visualizations': self._generate_all_visualizations(),
            'statistical_tests': self._perform_statistical_tests(),
            'recommendations': self._generate_recommendations()
        }
        
        # Генерируем HTML
        html_content = self._render_html(report_parts)
        
        # Сохраняем отчет
        if output_file is None:
            output_file = self.results_dir / f'experiment_report_{self.timestamp}.html'
        else:
            output_file = Path(output_file)
            
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        print(f"Отчет сохранен: {output_file}")
        return output_file
    
    def _generate_metadata(self) -> Dict:
        """Генерация метаданных эксперимента."""
        return {
            'experiment_name': self.config['experiment_name'],
            'timestamp': self.timestamp,
            'total_experiments': len(self.results),
            'n_datasets': len(set(r.dataset_name for r in self.results)),
            'n_models': len(set(r.model_name for r in self.results)),
            'n_methods': len(set(r.threshold_method for r in self.results)),
            'n_runs': self.config.get('n_runs', 1),
            'total_time': sum(r.training_time + r.pseudo_labeling_time for r in self.results) / 60  # в минутах
        }
    
    def _generate_executive_summary(self) -> Dict:
        """Генерация краткой сводки результатов."""
        # Основные метрики
        avg_improvement = self.df_results['improvement_accuracy'].mean()
        best_improvement = self.df_results['improvement_accuracy'].max()
        worst_improvement = self.df_results['improvement_accuracy'].min()
        
        # Лучший метод в среднем
        method_avg = self.df_results.groupby('method')['improvement_accuracy'].mean()
        best_method = method_avg.idxmax()
        best_method_score = method_avg.max()
        
        # Лучшая модель
        model_avg = self.df_results.groupby('model')['improvement_accuracy'].mean()
        best_model = model_avg.idxmax()
        
        # Статистика отбора
        avg_selection = self.df_results['selection_ratio'].mean()
        
        # Лучшая комбинация
        best_row = self.df_results.loc[self.df_results['improvement_accuracy'].idxmax()]
        
        return {
            'avg_improvement': avg_improvement,
            'best_improvement': best_improvement,
            'worst_improvement': worst_improvement,
            'best_method': best_method,
            'best_method_score': best_method_score,
            'best_model': best_model,
            'avg_selection_ratio': avg_selection,
            'best_combination': {
                'dataset': best_row['dataset'],
                'model': best_row['model'],
                'method': best_row['method'],
                'improvement': best_row['improvement_accuracy'],
                'threshold': best_row['threshold']
            }
        }
    
    def _generate_individual_analysis(self) -> List[Dict]:
        """Анализ каждого отдельного эксперимента."""
        analyses = []
        
        # Группируем по датасетам
        for dataset in self.df_results['dataset'].unique():
            dataset_data = self.df_results[self.df_results['dataset'] == dataset]
            
            # Анализ для каждой модели в датасете
            for model in dataset_data['model'].unique():
                model_data = dataset_data[dataset_data['model'] == model]
                
                analysis = {
                    'dataset': dataset,
                    'model': model,
                    'baseline_accuracy': model_data['baseline_accuracy'].mean(),
                    'methods_performance': []
                }
                
                # Анализ каждого метода
                for method in model_data['method'].unique():
                    method_data = model_data[model_data['method'] == method]
                    
                    method_analysis = {
                        'method': method,
                        'avg_improvement': method_data['improvement_accuracy'].mean(),
                        'std_improvement': method_data['improvement_accuracy'].std(),
                        'avg_threshold': method_data['threshold'].mean(),
                        'avg_selection_ratio': method_data['selection_ratio'].mean(),
                        'avg_pseudo_accuracy': method_data['pseudo_accuracy'].mean() if 'pseudo_accuracy' in method_data else None,
                        'n_experiments': len(method_data)
                    }
                    
                    # Интерпретация результатов
                    if method_analysis['avg_improvement'] > 0.01:
                        method_analysis['interpretation'] = "✅ Значительное улучшение"
                    elif method_analysis['avg_improvement'] > 0:
                        method_analysis['interpretation'] = "✓ Небольшое улучшение"
                    else:
                        method_analysis['interpretation'] = "❌ Ухудшение качества"
                    
                    analysis['methods_performance'].append(method_analysis)
                
                # Сортируем методы по эффективности
                analysis['methods_performance'].sort(key=lambda x: x['avg_improvement'], reverse=True)
                analyses.append(analysis)
        
        return analyses
    
    def _generate_comparative_analysis(self) -> Dict:
        """Сравнительный анализ результатов."""
        comparative = {}
        
        # 1. Сравнение методов
        method_comparison = self.df_results.groupby('method').agg({
            'improvement_accuracy': ['mean', 'std', 'min', 'max', 'count'],
            'selection_ratio': ['mean', 'std'],
            'threshold': ['mean', 'std'],
            'pseudo_labeling_time': 'mean'
        }).round(4)
        
        comparative['methods'] = method_comparison.to_dict()
        
        # 2. Сравнение моделей
        model_comparison = self.df_results.groupby('model').agg({
            'baseline_accuracy': 'mean',
            'improvement_accuracy': ['mean', 'std'],
            'training_time': 'mean',
            'pseudo_labeling_time': 'mean'
        }).round(4)
        
        comparative['models'] = model_comparison.to_dict()
        
        # 3. Сравнение по типам задач (бинарная vs мультиклассовая)
        binary_datasets = [d for d, info in self.config['datasets'].items() if 'binary' in d]
        multiclass_datasets = [d for d, info in self.config['datasets'].items() if 'multiclass' in d]
        
        task_comparison = {
            'binary': self._analyze_task_type(binary_datasets),
            'multiclass': self._analyze_task_type(multiclass_datasets)
        }
        comparative['task_types'] = task_comparison
        
        # 4. Корреляционный анализ
        correlation_cols = ['threshold', 'selection_ratio', 'pseudo_accuracy', 
                          'improvement_accuracy', 'improvement_f1_score']
        correlation_cols = [col for col in correlation_cols if col in self.df_results.columns]
        
        correlation_matrix = self.df_results[correlation_cols].corr()
        comparative['correlations'] = correlation_matrix.to_dict()
        
        # 5. Ранжирование методов
        ranking = self._rank_methods()
        comparative['ranking'] = ranking
        
        return comparative
    
    def _analyze_task_type(self, datasets: List[str]) -> Dict:
        """Анализ результатов для типа задачи."""
        task_data = self.df_results[self.df_results['dataset'].isin(datasets)]
        
        if len(task_data) == 0:
            return {}
        
        return {
            'n_experiments': len(task_data),
            'avg_baseline_accuracy': task_data['baseline_accuracy'].mean(),
            'avg_improvement': task_data['improvement_accuracy'].mean(),
            'best_method': task_data.groupby('method')['improvement_accuracy'].mean().idxmax(),
            'avg_selection_ratio': task_data['selection_ratio'].mean()
        }
    
    def _rank_methods(self) -> List[Dict]:
        """Ранжирование методов по различным критериям."""
        methods = self.df_results['method'].unique()
        
        rankings = []
        for method in methods:
            method_data = self.df_results[self.df_results['method'] == method]
            
            # Вычисляем различные метрики для ранжирования
            ranking = {
                'method': method,
                'avg_improvement': method_data['improvement_accuracy'].mean(),
                'consistency': 1 / (method_data['improvement_accuracy'].std() + 0.001),  # Обратная величина std
                'reliability': (method_data['improvement_accuracy'] > 0).mean(),  # Доля улучшений
                'efficiency': 1 / (method_data['pseudo_labeling_time'].mean() + 0.001),  # Скорость
                'selection_quality': method_data['pseudo_accuracy'].mean() if 'pseudo_accuracy' in method_data else 0,
                'coverage': method_data['selection_ratio'].mean()
            }
            
            # Общий score (взвешенная сумма)
            weights = {
                'avg_improvement': 0.3,
                'consistency': 0.2,
                'reliability': 0.2,
                'efficiency': 0.1,
                'selection_quality': 0.15,
                'coverage': 0.05
            }
            
            # Нормализуем метрики
            for metric in weights:
                if metric in ranking:
                    all_values = [r.get(metric, 0) for r in rankings] + [ranking[metric]]
                    if max(all_values) > 0:
                        ranking[f'{metric}_normalized'] = ranking[metric] / max(all_values)
                    else:
                        ranking[f'{metric}_normalized'] = 0
            
            ranking['overall_score'] = sum(
                ranking.get(f'{metric}_normalized', 0) * weight 
                for metric, weight in weights.items()
            )
            
            rankings.append(ranking)
        
        # Сортируем по overall_score
        rankings.sort(key=lambda x: x['overall_score'], reverse=True)
        
        # Добавляем ранг
        for i, ranking in enumerate(rankings):
            ranking['rank'] = i + 1
        
        return rankings
    
    def _generate_general_conclusions(self) -> Dict:
        """Генерация общих выводов."""
        conclusions = {
            'main_findings': [],
            'best_practices': [],
            'warnings': [],
            'future_work': []
        }
        
        # Основные находки
        avg_improvement = self.df_results['improvement_accuracy'].mean()
        if avg_improvement > 0.01:
            conclusions['main_findings'].append(
                f"Псевдо-разметка в среднем улучшает качество на {avg_improvement:.1%}"
            )
        
        # Анализ по типам методов
        method_types = {
            'binary': ['f1_optimization', 'youden', 'cost_sensitive'],
            'multiclass': ['entropy', 'margin', 'top_k'],
            'universal': ['percentile', 'fixed', 'adaptive']
        }
        
        for type_name, methods in method_types.items():
            type_data = self.df_results[self.df_results['method'].isin(methods)]
            if len(type_data) > 0:
                avg_imp = type_data['improvement_accuracy'].mean()
                conclusions['main_findings'].append(
                    f"{type_name.capitalize()} методы показывают среднее улучшение {avg_imp:.1%}"
                )
        
        # Лучшие практики
        best_method = self.df_results.groupby('method')['improvement_accuracy'].mean().idxmax()
        conclusions['best_practices'].append(
            f"Используйте метод '{best_method}' для максимального улучшения качества"
        )
        
        # Анализ стабильности
        stable_methods = []
        for method in self.df_results['method'].unique():
            method_std = self.df_results[self.df_results['method'] == method]['improvement_accuracy'].std()
            if method_std < 0.01:
                stable_methods.append(method)
        
        if stable_methods:
            conclusions['best_practices'].append(
                f"Для стабильных результатов используйте: {', '.join(stable_methods)}"
            )
        
        # Предупреждения
        problematic = self.df_results[self.df_results['improvement_accuracy'] < -0.01]
        if len(problematic) > 0:
            conclusions['warnings'].append(
                f"Обнаружено {len(problematic)} случаев значительного ухудшения качества"
            )
            
            problem_methods = problematic['method'].value_counts().head(3).index.tolist()
            conclusions['warnings'].append(
                f"Методы с наибольшим риском: {', '.join(problem_methods)}"
            )
        
        # Рекомендации для будущих исследований
        if self.df_results['selection_ratio'].mean() < 0.3:
            conclusions['future_work'].append(
                "Исследовать методы для увеличения доли отбираемых примеров"
            )
        
        if 'mc_dropout' not in self.df_results['method'].unique():
            conclusions['future_work'].append(
                "Добавить методы оценки неопределенности (MC Dropout, ансамбли)"
            )
        
        return conclusions
    
    def _generate_all_visualizations(self) -> Dict[str, str]:
        """Генерация всех визуализаций."""
        visualizations = {}
        
        # 1. Интерактивная heatmap улучшений
        viz1 = self._create_improvement_heatmap()
        visualizations['improvement_heatmap'] = viz1
        
        # 2. Box plot по методам
        viz2 = self._create_methods_boxplot()
        visualizations['methods_boxplot'] = viz2
        
        # 3. Scatter plot: threshold vs improvement
        viz3 = self._create_threshold_scatter()
        visualizations['threshold_scatter'] = viz3
        
        # 4. Radar chart для методов
        viz4 = self._create_methods_radar()
        visualizations['methods_radar'] = viz4
        
        # 5. Временной анализ
        viz5 = self._create_time_analysis()
        visualizations['time_analysis'] = viz5
        
        # 6. Распределение псевдо-меток
        viz6 = self._create_selection_distribution()
        visualizations['selection_distribution'] = viz6
        
        # 7. Корреляционная матрица
        viz7 = self._create_correlation_matrix()
        visualizations['correlation_matrix'] = viz7
        
        # 8. Парные сравнения
        viz8 = self._create_pairwise_comparison()
        visualizations['pairwise_comparison'] = viz8
        
        return visualizations
    
    def _create_improvement_heatmap(self) -> str:
        """Создание интерактивной heatmap улучшений."""
        pivot = self.df_results.pivot_table(
            values='improvement_accuracy',
            index='method',
            columns='dataset',
            aggfunc='mean'
        )
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale='RdYlGn',
            zmid=0,
            text=np.round(pivot.values, 3),
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False,
            hovertemplate='Метод: %{y}<br>Датасет: %{x}<br>Улучшение: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Среднее улучшение accuracy по методам и датасетам',
            xaxis_title='Датасет',
            yaxis_title='Метод',
            height=600,
            width=800
        )
        
        return fig.to_html(include_plotlyjs='cdn', div_id="improvement_heatmap")
    
    def _create_methods_boxplot(self) -> str:
        """Box plot распределения улучшений по методам."""
        fig = go.Figure()
        
        for method in self.df_results['method'].unique():
            method_data = self.df_results[self.df_results['method'] == method]
            
            fig.add_trace(go.Box(
                y=method_data['improvement_accuracy'],
                name=method,
                boxpoints='outliers',
                marker_color=px.colors.qualitative.Set3[
                    list(self.df_results['method'].unique()).index(method) % len(px.colors.qualitative.Set3)
                ],
                hovertemplate='%{y:.3f}<extra></extra>'
            ))
        
        fig.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.5)
        
        fig.update_layout(
            title='Распределение улучшений качества по методам',
            yaxis_title='Улучшение accuracy',
            xaxis_title='Метод',
            showlegend=False,
            height=500
        )
        
        return fig.to_html(include_plotlyjs='cdn', div_id="methods_boxplot")
    
    def _create_threshold_scatter(self) -> str:
        """Scatter plot зависимости улучшения от порога."""
        fig = go.Figure()
        
        # Цветовая палитра для методов
        colors = px.colors.qualitative.Set3
        methods = self.df_results['method'].unique()
        
        for i, method in enumerate(methods):
            method_data = self.df_results[self.df_results['method'] == method]
            
            fig.add_trace(go.Scatter(
                x=method_data['threshold'],
                y=method_data['improvement_accuracy'],
                mode='markers',
                name=method,
                marker=dict(
                    size=8,
                    color=colors[i % len(colors)],
                    opacity=0.7
                ),
                text=[f"Датасет: {d}<br>Модель: {m}" 
                      for d, m in zip(method_data['dataset'], method_data['model'])],
                hovertemplate='Порог: %{x:.3f}<br>Улучшение: %{y:.3f}<br>%{text}<extra></extra>'
            ))
        
        fig.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.5)
        
        fig.update_layout(
            title='Зависимость улучшения качества от порога отбора',
            xaxis_title='Порог',
            yaxis_title='Улучшение accuracy',
            height=600,
            width=900,
            hovermode='closest'
        )
        
        return fig.to_html(include_plotlyjs='cdn', div_id="threshold_scatter")
    
    def _create_methods_radar(self) -> str:
        """Radar chart для комплексной оценки методов."""
        # Подготовка данных
        rankings = self._rank_methods()[:6]  # Топ 6 методов
        
        categories = ['Улучшение', 'Стабильность', 'Надежность', 
                     'Скорость', 'Качество отбора', 'Покрытие']
        
        fig = go.Figure()
        
        for ranking in rankings:
            values = [
                ranking.get('avg_improvement_normalized', 0),
                ranking.get('consistency_normalized', 0),
                ranking.get('reliability_normalized', 0),
                ranking.get('efficiency_normalized', 0),
                ranking.get('selection_quality_normalized', 0),
                ranking.get('coverage_normalized', 0)
            ]
            
            fig.add_trace(go.Scatterpolar(
                r=values + [values[0]],  # Замыкаем фигуру
                theta=categories + [categories[0]],
                fill='toself',
                name=ranking['method'],
                hovertemplate='%{theta}: %{r:.2f}<extra></extra>'
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=True,
            title="Комплексная оценка методов (нормализованные значения)",
            height=600,
            width=700
        )
        
        return fig.to_html(include_plotlyjs='cdn', div_id="methods_radar")
    
    def _create_time_analysis(self) -> str:
        """Анализ временных затрат."""
        # Подготовка данных
        time_data = self.df_results.groupby('method').agg({
            'training_time': 'mean',
            'pseudo_labeling_time': 'mean'
        }).reset_index()
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Время обучения',
            x=time_data['method'],
            y=time_data['training_time'],
            marker_color='lightblue'
        ))
        
        fig.add_trace(go.Bar(
            name='Время псевдо-разметки',
            x=time_data['method'],
            y=time_data['pseudo_labeling_time'],
            marker_color='lightgreen'
        ))
        
        fig.update_layout(
            title='Среднее время выполнения по методам',
            xaxis_title='Метод',
            yaxis_title='Время (секунды)',
            barmode='stack',
            height=500
        )
        
        return fig.to_html(include_plotlyjs='cdn', div_id="time_analysis")
    
    def _create_selection_distribution(self) -> str:
        """Распределение доли отбираемых примеров."""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Распределение по методам', 'Зависимость от порога')
        )
        
        # Violin plot по методам
        for i, method in enumerate(self.df_results['method'].unique()):
            method_data = self.df_results[self.df_results['method'] == method]
            
            fig.add_trace(
                go.Violin(
                    y=method_data['selection_ratio'],
                    name=method,
                    box_visible=True,
                    meanline_visible=True,
                    showlegend=False
                ),
                row=1, col=1
            )
        
        # Scatter plot: порог vs доля отбора
        fig.add_trace(
            go.Scatter(
                x=self.df_results['threshold'],
                y=self.df_results['selection_ratio'],
                mode='markers',
                marker=dict(
                    size=6,
                    color=self.df_results['improvement_accuracy'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Улучшение")
                ),
                text=self.df_results['method'],
                hovertemplate='Порог: %{x:.3f}<br>Доля отбора: %{y:.3f}<br>Метод: %{text}<extra></extra>',
                showlegend=False
            ),
            row=1, col=2
        )
        
        fig.update_xaxes(title_text="Метод", row=1, col=1)
        fig.update_xaxes(title_text="Порог", row=1, col=2)
        fig.update_yaxes(title_text="Доля отбора", row=1, col=1)
        fig.update_yaxes(title_text="Доля отбора", row=1, col=2)
        
        fig.update_layout(
            title='Анализ отбора примеров',
            height=500,
            width=1000
        )
        
        return fig.to_html(include_plotlyjs='cdn', div_id="selection_distribution")
    
    def _create_correlation_matrix(self) -> str:
        """Корреляционная матрица метрик."""
        # Выбираем числовые колонки для корреляции
        corr_columns = ['threshold', 'selection_ratio', 'improvement_accuracy', 
                       'improvement_f1_score', 'pseudo_labeling_time']
        
        # Фильтруем существующие колонки
        corr_columns = [col for col in corr_columns if col in self.df_results.columns]
        
        # Вычисляем корреляцию
        corr_matrix = self.df_results[corr_columns].corr()
        
        # Создаем heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}',
            textfont={"size": 12},
            hovertemplate='%{y} vs %{x}: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Корреляция между метриками',
            height=600,
            width=700
        )
        
        return fig.to_html(include_plotlyjs='cdn', div_id="correlation_matrix")
    
    def _create_pairwise_comparison(self) -> str:
        """Парное сравнение методов."""
        # Создаем матрицу парных сравнений
        methods = self.df_results['method'].unique()
        n_methods = len(methods)
        comparison_matrix = np.zeros((n_methods, n_methods))
        
        for i, method1 in enumerate(methods):
            for j, method2 in enumerate(methods):
                if i != j:
                    # Сравниваем методы на одинаковых датасетах и моделях
                    data1 = self.df_results[self.df_results['method'] == method1]
                    data2 = self.df_results[self.df_results['method'] == method2]
                    
                    # Находим общие эксперименты
                    merge_cols = ['dataset', 'model']
                    merged = pd.merge(
                        data1[merge_cols + ['improvement_accuracy']], 
                        data2[merge_cols + ['improvement_accuracy']], 
                        on=merge_cols, 
                        suffixes=('_1', '_2')
                    )
                    
                    if len(merged) > 0:
                        # Считаем процент случаев, когда method1 лучше method2
                        wins = (merged['improvement_accuracy_1'] > merged['improvement_accuracy_2']).sum()
                        comparison_matrix[i, j] = wins / len(merged) * 100
        
        # Создаем heatmap
        fig = go.Figure(data=go.Heatmap(
            z=comparison_matrix,
            x=methods,
            y=methods,
            colorscale='RdYlGn',
            zmid=50,
            text=np.round(comparison_matrix, 1),
            texttemplate='%{text}%',
            textfont={"size": 10},
            hovertemplate='%{y} лучше %{x} в %{z:.1f}% случаев<extra></extra>'
        ))
        
        fig.update_layout(
            title='Парное сравнение методов (% побед)',
            xaxis_title='Метод',
            yaxis_title='Метод',
            height=600,
            width=700
        )
        
        return fig.to_html(include_plotlyjs='cdn', div_id="pairwise_comparison")
    
    def _perform_statistical_tests(self) -> Dict:
        """Выполнение статистических тестов."""
        tests_results = {}
        
        # 1. ANOVA для сравнения методов
        methods = self.df_results['method'].unique()
        method_groups = [self.df_results[self.df_results['method'] == m]['improvement_accuracy'].values 
                        for m in methods]
        
        f_stat, p_value = stats.f_oneway(*method_groups)
        tests_results['anova_methods'] = {
            'f_statistic': f_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'interpretation': 'Методы значимо различаются' if p_value < 0.05 else 'Нет значимых различий между методами'
        }
        
        # 2. Парные t-тесты для топ методов
        top_methods = self.df_results.groupby('method')['improvement_accuracy'].mean().nlargest(3).index
        pairwise_tests = []
        
        for i, method1 in enumerate(top_methods):
            for method2 in top_methods[i+1:]:
                data1 = self.df_results[self.df_results['method'] == method1]['improvement_accuracy']
                data2 = self.df_results[self.df_results['method'] == method2]['improvement_accuracy']
                
                t_stat, p_val = stats.ttest_ind(data1, data2)
                
                pairwise_tests.append({
                    'method1': method1,
                    'method2': method2,
                    't_statistic': t_stat,
                    'p_value': p_val,
                    'significant': p_val < 0.05
                })
        
        tests_results['pairwise_ttests'] = pairwise_tests
        
        # 3. Тест на нормальность распределения улучшений
        _, normality_p = stats.normaltest(self.df_results['improvement_accuracy'])
        tests_results['normality_test'] = {
            'p_value': normality_p,
            'is_normal': normality_p > 0.05,
            'interpretation': 'Распределение нормальное' if normality_p > 0.05 else 'Распределение не является нормальным'
        }
        
        # 4. Корреляционные тесты
        if 'selection_ratio' in self.df_results.columns:
            corr, p_val = stats.pearsonr(
                self.df_results['selection_ratio'], 
                self.df_results['improvement_accuracy']
            )
            tests_results['selection_correlation'] = {
                'correlation': corr,
                'p_value': p_val,
                'significant': p_val < 0.05,
                'interpretation': f"{'Значимая' if p_val < 0.05 else 'Незначимая'} корреляция между долей отбора и улучшением"
            }
        
        return tests_results
    
    def _generate_recommendations(self) -> List[Dict]:
        """Генерация практических рекомендаций."""
        recommendations = []
        
        # 1. Рекомендации по выбору метода
        rankings = self._rank_methods()
        
        # Для общего случая
        best_overall = rankings[0]
        recommendations.append({
            'category': 'Общие рекомендации',
            'title': 'Лучший метод в целом',
            'content': f"Используйте метод '{best_overall['method']}' для большинства задач. "
                      f"Он показывает среднее улучшение {best_overall['avg_improvement']:.1%} "
                      f"с высокой надежностью {best_overall['reliability']:.1%}.",
            'priority': 'high'
        })
        
        # Для скорости
        fastest = max(rankings, key=lambda x: x.get('efficiency', 0))
        if fastest['method'] != best_overall['method']:
            recommendations.append({
                'category': 'Производительность',
                'title': 'Самый быстрый метод',
                'content': f"Если важна скорость, используйте '{fastest['method']}'. "
                          f"Он работает быстрее других, хотя и дает меньшее улучшение.",
                'priority': 'medium'
            })
        
        # Для стабильности
        most_stable = max(rankings, key=lambda x: x.get('consistency', 0))
        if most_stable['method'] not in [best_overall['method'], fastest['method']]:
            recommendations.append({
                'category': 'Стабильность',
                'title': 'Самый стабильный метод',
                'content': f"Для предсказуемых результатов используйте '{most_stable['method']}'. "
                          f"Он показывает наименьший разброс результатов.",
                'priority': 'medium'
            })
        
        # 2. Рекомендации по типам задач
        binary_methods = ['f1_optimization', 'youden', 'cost_sensitive']
        multiclass_methods = ['entropy', 'margin', 'top_k']
        
        binary_data = self.df_results[self.df_results['method'].isin(binary_methods)]
        if len(binary_data) > 0:
            best_binary = binary_data.groupby('method')['improvement_accuracy'].mean().idxmax()
            recommendations.append({
                'category': 'Бинарная классификация',
                'title': 'Лучший метод для бинарных задач',
                'content': f"Для бинарной классификации рекомендуется '{best_binary}'.",
                'priority': 'high'
            })
        
        multiclass_data = self.df_results[self.df_results['method'].isin(multiclass_methods)]
        if len(multiclass_data) > 0:
            best_multiclass = multiclass_data.groupby('method')['improvement_accuracy'].mean().idxmax()
            recommendations.append({
                'category': 'Мультиклассовая классификация',
                'title': 'Лучший метод для многоклассовых задач',
                'content': f"Для задач с множеством классов используйте '{best_multiclass}'.",
                'priority': 'high'
            })
        
        # 3. Предостережения
        problematic = self.df_results[self.df_results['improvement_accuracy'] < -0.01]
        if len(problematic) > 0:
            risky_methods = problematic['method'].value_counts().head(3).index.tolist()
            recommendations.append({
                'category': 'Предостережения',
                'title': 'Методы с риском ухудшения',
                'content': f"Будьте осторожны с методами: {', '.join(risky_methods)}. "
                          f"Они могут ухудшить качество модели в некоторых случаях.",
                'priority': 'high'
            })
        
        # 4. Оптимальные параметры
        if 'percentile' in self.df_results['method'].unique():
            percentile_data = self.df_results[self.df_results['method'] == 'percentile']
            optimal_percentile = percentile_data.loc[
                percentile_data['improvement_accuracy'].idxmax(), 'threshold'
            ]
            recommendations.append({
                'category': 'Параметры методов',
                'title': 'Оптимальный процентиль',
                'content': f"Для процентильного метода используйте порог около {optimal_percentile:.0%}.",
                'priority': 'low'
            })
        
        return recommendations
    
    def _render_html(self, report_parts: Dict) -> str:
        """Рендеринг HTML отчета."""
        template_str = """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Отчет эксперимента: {{ metadata.experiment_name }}</title>
    
    <style>
        :root {
            --primary-color: #2563eb;
            --secondary-color: #7c3aed;
            --success-color: #10b981;
            --warning-color: #f59e0b;
            --danger-color: #ef4444;
            --dark-color: #1f2937;
            --light-bg: #f9fafb;
            --border-color: #e5e7eb;
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif;
            line-height: 1.6;
            color: var(--dark-color);
            background-color: var(--light-bg);
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        
        header {
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            color: white;
            padding: 60px 0;
            text-align: center;
            margin-bottom: 40px;
            border-radius: 12px;
        }
        
        h1 {
            font-size: 2.5em;
            margin-bottom: 20px;
        }
        
        .metadata {
            display: flex;
            justify-content: center;
            gap: 30px;
            flex-wrap: wrap;
            margin-top: 20px;
        }
        
        .metadata-item {
            background: rgba(255, 255, 255, 0.1);
            padding: 10px 20px;
            border-radius: 8px;
        }
        
        nav {
            background: white;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            margin-bottom: 40px;
            position: sticky;
            top: 20px;
            z-index: 100;
        }
        
        nav ul {
            list-style: none;
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 20px;
        }
        
        nav a {
            color: var(--dark-color);
            text-decoration: none;
            padding: 10px 20px;
            border-radius: 8px;
            transition: all 0.3s;
            font-weight: 500;
        }
        
        nav a:hover {
            background-color: var(--primary-color);
            color: white;
        }
        
        .section {
            background: white;
            border-radius: 12px;
            padding: 40px;
            margin-bottom: 40px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }
        
        h2 {
            color: var(--primary-color);
            margin-bottom: 30px;
            font-size: 2em;
        }
        
        h3 {
            color: var(--dark-color);
            margin: 30px 0 20px;
            font-size: 1.5em;
        }
        
        h4 {
            color: var(--dark-color);
            margin: 20px 0 15px;
            font-size: 1.2em;
        }
        
        .metric-card {
            background: var(--light-bg);
            padding: 20px;
            border-radius: 8px;
            margin: 10px 0;
            border-left: 4px solid var(--primary-color);
        }
        
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: var(--primary-color);
        }
        
        .metric-label {
            color: #6b7280;
            font-size: 0.9em;
        }
        
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        
        th {
            background: var(--primary-color);
            color: white;
            padding: 12px;
            text-align: left;
        }
        
        td {
            padding: 12px;
            border-bottom: 1px solid var(--border-color);
        }
        
        tr:hover {
            background-color: var(--light-bg);
        }
        
        .positive {
            color: var(--success-color);
            font-weight: bold;
        }
        
        .negative {
            color: var(--danger-color);
            font-weight: bold;
        }
        
        .neutral {
            color: #6b7280;
        }
        
        .recommendation {
            background: #eef2ff;
            border-left: 4px solid var(--primary-color);
            padding: 20px;
            margin: 20px 0;
            border-radius: 0 8px 8px 0;
        }
        
        .warning {
            background: #fef3c7;
            border-left: 4px solid var(--warning-color);
            padding: 20px;
            margin: 20px 0;
            border-radius: 0 8px 8px 0;
        }
        
        .visualization {
            margin: 30px 0;
            text-align: center;
        }
        
        .viz-description {
            background: var(--light-bg);
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
            font-style: italic;
            color: #6b7280;
        }
        
        .badge {
            display: inline-block;
            padding: 4px 12px;
            background: var(--primary-color);
            color: white;
            border-radius: 20px;
            font-size: 0.85em;
            margin: 2px;
        }
        
        .badge.success {
            background: var(--success-color);
        }
        
        .badge.warning {
            background: var(--warning-color);
        }
        
        .badge.danger {
            background: var(--danger-color);
        }
        
        .interpretation-box {
            background: #f0f9ff;
            border: 1px solid #bfdbfe;
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
        }
        
        .interpretation-box h5 {
            color: var(--primary-color);
            margin-bottom: 10px;
        }
        
        @media print {
            nav {
                display: none;
            }
            
            .section {
                page-break-inside: avoid;
            }
        }
        
        @media (max-width: 768px) {
            .container {
                padding: 10px;
            }
            
            .section {
                padding: 20px;
            }
            
            .grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>📊 {{ metadata.experiment_name }}</h1>
            <p class="subtitle">Отчет по экспериментам сравнения методов выбора порогов для псевдо-разметки</p>
            <div class="metadata">
                <div class="metadata-item">
                    <strong>Дата:</strong> {{ metadata.timestamp }}
                </div>
                <div class="metadata-item">
                    <strong>Экспериментов:</strong> {{ metadata.total_experiments }}
                </div>
                <div class="metadata-item">
                    <strong>Время выполнения:</strong> {{ "%.1f"|format(metadata.total_time) }} мин
                </div>
            </div>
        </header>
        
        <nav>
            <ul>
                <li><a href="#summary">Краткая сводка</a></li>
                <li><a href="#individual">Анализ экспериментов</a></li>
                <li><a href="#comparison">Сравнительный анализ</a></li>
                <li><a href="#visualizations">Визуализации</a></li>
                <li><a href="#statistics">Статистика</a></li>
                <li><a href="#conclusions">Выводы</a></li>
                <li><a href="#recommendations">Рекомендации</a></li>
            </ul>
        </nav>
        
        <!-- КРАТКАЯ СВОДКА -->
        <section id="summary" class="section">
            <h2>📋 Краткая сводка результатов</h2>
            
            <div class="grid">
                <div class="metric-card">
                    <div class="metric-value">{{ "%.1%%"|format(executive_summary.avg_improvement * 100) }}</div>
                    <div class="metric-label">Среднее улучшение accuracy</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-value">{{ executive_summary.best_method }}</div>
                    <div class="metric-label">Лучший метод</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-value">{{ "%.1%%"|format(executive_summary.avg_selection_ratio * 100) }}</div>
                    <div class="metric-label">Средняя доля отбора</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-value">{{ executive_summary.best_model }}</div>
                    <div class="metric-label">Лучшая модель</div>
                </div>
            </div>
            
            <h3>🏆 Лучший результат</h3>
            <div class="recommendation">
                <p><strong>Датасет:</strong> {{ executive_summary.best_combination.dataset }}</p>
                <p><strong>Модель:</strong> {{ executive_summary.best_combination.model }}</p>
                <p><strong>Метод:</strong> {{ executive_summary.best_combination.method }}</p>
                <p><strong>Улучшение:</strong> <span class="positive">+{{ "%.1%%"|format(executive_summary.best_combination.improvement * 100) }}</span></p>
                <p><strong>Порог:</strong> {{ "%.3f"|format(executive_summary.best_combination.threshold) }}</p>
            </div>
        </section>
        
        <!-- ИНДИВИДУАЛЬНЫЙ АНАЛИЗ -->
        <section id="individual" class="section">
            <h2>🔍 Анализ отдельных экспериментов</h2>
            
            <div class="interpretation-box">
                <h5>📖 Как читать этот раздел:</h5>
                <p>Для каждой комбинации датасет-модель показаны результаты всех методов выбора порогов.
                Зеленым выделены улучшения качества, красным - ухудшения. 
                Обратите внимание на стабильность результатов (std) и долю отобранных примеров.</p>
            </div>
            
            {% for analysis in individual_analysis %}
            <h3>{{ analysis.dataset }} - {{ analysis.model }}</h3>
            <p><strong>Baseline accuracy:</strong> {{ "%.3f"|format(analysis.baseline_accuracy) }}</p>
            
            <table>
                <thead>
                    <tr>
                        <th>Метод</th>
                        <th>Улучшение</th>
                        <th>Порог</th>
                        <th>Доля отбора</th>
                        <th>Точность псевдо-меток</th>
                        <th>Интерпретация</th>
                    </tr>
                </thead>
                <tbody>
                    {% for method in analysis.methods_performance %}
                    <tr>
                        <td><strong>{{ method.method }}</strong></td>
                        <td class="{% if method.avg_improvement > 0 %}positive{% else %}negative{% endif %}">
                            {{ "%.3f"|format(method.avg_improvement) }} ± {{ "%.3f"|format(method.std_improvement) }}
                        </td>
                        <td>{{ "%.3f"|format(method.avg_threshold) }}</td>
                        <td>{{ "%.1%%"|format(method.avg_selection_ratio * 100) }}</td>
                        <td>{% if method.avg_pseudo_accuracy %}{{ "%.3f"|format(method.avg_pseudo_accuracy) }}{% else %}-{% endif %}</td>
                        <td>{{ method.interpretation }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
            {% endfor %}
        </section>
        
        <!-- СРАВНИТЕЛЬНЫЙ АНАЛИЗ -->
        <section id="comparison" class="section">
            <h2>📊 Сравнительный анализ</h2>
            
            <h3>Ранжирование методов</h3>
            
            <div class="interpretation-box">
                <h5>📖 Методика ранжирования:</h5>
                <p>Методы оцениваются по 6 критериям: улучшение качества (30%), стабильность (20%), 
                надежность (20%), скорость (10%), качество отбора (15%) и покрытие (5%). 
                Все метрики нормализованы для справедливого сравнения.</p>
            </div>
            
            <table>
                <thead>
                    <tr>
                        <th>Ранг</th>
                        <th>Метод</th>
                        <th>Общий балл</th>
                        <th>Улучшение</th>
                        <th>Стабильность</th>
                        <th>Надежность</th>
                        <th>Скорость</th>
                    </tr>
                </thead>
                <tbody>
                    {% for ranking in comparative_analysis.ranking %}
                    <tr>
                        <td><strong>{{ ranking.rank }}</strong></td>
                        <td><strong>{{ ranking.method }}</strong></td>
                        <td>{{ "%.3f"|format(ranking.overall_score) }}</td>
                        <td>{{ "%.1%%"|format(ranking.avg_improvement * 100) }}</td>
                        <td>{{ "%.3f"|format(ranking.consistency) }}</td>
                        <td>{{ "%.1%%"|format(ranking.reliability * 100) }}</td>
                        <td>{{ "%.3f"|format(ranking.efficiency) }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
            
            <h3>Анализ по типам задач</h3>
            
            {% if comparative_analysis.task_types.binary.n_experiments > 0 %}
            <h4>Бинарная классификация</h4>
            <div class="metric-card">
                <p><strong>Лучший метод:</strong> {{ comparative_analysis.task_types.binary.best_method }}</p>
                <p><strong>Среднее улучшение:</strong> {{ "%.1%%"|format(comparative_analysis.task_types.binary.avg_improvement * 100) }}</p>
                <p><strong>Средняя доля отбора:</strong> {{ "%.1%%"|format(comparative_analysis.task_types.binary.avg_selection_ratio * 100) }}</p>
            </div>
            {% endif %}
            
            {% if comparative_analysis.task_types.multiclass.n_experiments > 0 %}
            <h4>Мультиклассовая классификация</h4>
            <div class="metric-card">
                <p><strong>Лучший метод:</strong> {{ comparative_analysis.task_types.multiclass.best_method }}</p>
                <p><strong>Среднее улучшение:</strong> {{ "%.1%%"|format(comparative_analysis.task_types.multiclass.avg_improvement * 100) }}</p>
                <p><strong>Средняя доля отбора:</strong> {{ "%.1%%"|format(comparative_analysis.task_types.multiclass.avg_selection_ratio * 100) }}</p>
            </div>
            {% endif %}
        </section>
        
        <!-- ВИЗУАЛИЗАЦИИ -->
        <section id="visualizations" class="section">
            <h2>📈 Визуализации</h2>
            
            <div class="visualization">
                <h3>Heatmap улучшений по методам и датасетам</h3>
                <div class="viz-description">
                    Показывает среднее улучшение accuracy для каждой комбинации метод-датасет.
                    Зеленый цвет означает улучшение, красный - ухудшение.
                </div>
                {{ visualizations.improvement_heatmap|safe }}
            </div>
            
            <div class="visualization">
                <h3>Распределение улучшений по методам</h3>
                <div class="viz-description">
                    Box plot показывает медиану, квартили и выбросы для каждого метода.
                    Красная линия - нулевое улучшение (baseline).
                </div>
                {{ visualizations.methods_boxplot|safe }}
            </div>
            
            <div class="visualization">
                <h3>Зависимость улучшения от порога</h3>
                <div class="viz-description">
                    Scatter plot показывает, как выбранный порог влияет на улучшение качества.
                    Каждая точка - отдельный эксперимент.
                </div>
                {{ visualizations.threshold_scatter|safe }}
            </div>
            
            <div class="visualization">
                <h3>Комплексная оценка методов</h3>
                <div class="viz-description">
                    Radar chart сравнивает методы по 6 критериям. Чем больше площадь фигуры, 
                    тем лучше метод по совокупности критериев.
                </div>
                {{ visualizations.methods_radar|safe }}
            </div>
            
            <div class="visualization">
                <h3>Анализ времени выполнения</h3>
                <div class="viz-description">
                    Сравнение времени обучения базовой модели и времени псевдо-разметки.
                </div>
                {{ visualizations.time_analysis|safe }}
            </div>
            
            <div class="visualization">
                <h3>Анализ отбора примеров</h3>
                <div class="viz-description">
                    Слева: распределение доли отбираемых примеров по методам.
                    Справа: зависимость доли отбора от порога (цвет показывает улучшение).
                </div>
                {{ visualizations.selection_distribution|safe }}
            </div>
            
            <div class="visualization">
                <h3>Корреляция между метриками</h3>
                <div class="viz-description">
                    Показывает взаимосвязи между различными метриками эксперимента.
                </div>
                {{ visualizations.correlation_matrix|safe }}
            </div>
            
            <div class="visualization">
                <h3>Парное сравнение методов</h3>
                <div class="viz-description">
                    Показывает процент случаев, когда метод в строке превосходит метод в столбце
                    на одинаковых данных.
                </div>
                {{ visualizations.pairwise_comparison|safe }}
            </div>
        </section>
        
        <!-- СТАТИСТИЧЕСКИЕ ТЕСТЫ -->
        <section id="statistics" class="section">
            <h2>📊 Статистический анализ</h2>
            
            <div class="interpretation-box">
                <h5>📖 Интерпретация статистических тестов:</h5>
                <p>p-value < 0.05 означает статистически значимый результат. 
                ANOVA проверяет, есть ли различия между методами в целом. 
                Парные t-тесты показывают, какие конкретно методы значимо различаются.</p>
            </div>
            
            <h3>ANOVA для сравнения методов</h3>
            <div class="metric-card">
                <p><strong>F-статистика:</strong> {{ "%.3f"|format(statistical_tests.anova_methods.f_statistic) }}</p>
                <p><strong>p-value:</strong> {{ "%.4f"|format(statistical_tests.anova_methods.p_value) }}</p>
                <p><strong>Результат:</strong> 
                    <span class="badge {% if statistical_tests.anova_methods.significant %}success{% else %}warning{% endif %}">
                        {{ statistical_tests.anova_methods.interpretation }}
                    </span>
                </p>
            </div>
            
            {% if statistical_tests.pairwise_ttests %}
            <h3>Парные сравнения топ методов</h3>
            <table>
                <thead>
                    <tr>
                        <th>Метод 1</th>
                        <th>Метод 2</th>
                        <th>t-статистика</th>
                        <th>p-value</th>
                        <th>Значимость</th>
                    </tr>
                </thead>
                <tbody>
                    {% for test in statistical_tests.pairwise_ttests %}
                    <tr>
                        <td>{{ test.method1 }}</td>
                        <td>{{ test.method2 }}</td>
                        <td>{{ "%.3f"|format(test.t_statistic) }}</td>
                        <td>{{ "%.4f"|format(test.p_value) }}</td>
                        <td>
                            <span class="badge {% if test.significant %}success{% else %}neutral{% endif %}">
                                {% if test.significant %}Значимо{% else %}Незначимо{% endif %}
                            </span>
                        </td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
            {% endif %}
            
            <h3>Проверка нормальности распределения</h3>
            <div class="metric-card">
                <p><strong>p-value:</strong> {{ "%.4f"|format(statistical_tests.normality_test.p_value) }}</p>
                <p><strong>Результат:</strong> {{ statistical_tests.normality_test.interpretation }}</p>
            </div>
            
            {% if statistical_tests.selection_correlation %}
            <h3>Корреляция между долей отбора и улучшением</h3>
            <div class="metric-card">
                <p><strong>Коэффициент корреляции:</strong> {{ "%.3f"|format(statistical_tests.selection_correlation.correlation) }}</p>
                <p><strong>p-value:</strong> {{ "%.4f"|format(statistical_tests.selection_correlation.p_value) }}</p>
                <p><strong>Результат:</strong> {{ statistical_tests.selection_correlation.interpretation }}</p>
            </div>
            {% endif %}
        </section>
        
        <!-- ВЫВОДЫ -->
        <section id="conclusions" class="section">
            <h2>💡 Общие выводы</h2>
            
            <h3>Основные находки</h3>
            <ul>
                {% for finding in general_conclusions.main_findings %}
                <li>{{ finding }}</li>
                {% endfor %}
            </ul>
            
            <h3>Лучшие практики</h3>
            <div class="recommendation">
                <ul>
                    {% for practice in general_conclusions.best_practices %}
                    <li>{{ practice }}</li>
                    {% endfor %}
                </ul>
            </div>
            
            {% if general_conclusions.warnings %}
            <h3>Предупреждения</h3>
            <div class="warning">
                <ul>
                    {% for warning in general_conclusions.warnings %}
                    <li>{{ warning }}</li>
                    {% endfor %}
                </ul>
            </div>
            {% endif %}
            
            {% if general_conclusions.future_work %}
            <h3>Рекомендации для будущих исследований</h3>
            <ul>
                {% for work in general_conclusions.future_work %}
                <li>{{ work }}</li>
                {% endfor %}
            </ul>
            {% endif %}
        </section>
        
        <!-- ПРАКТИЧЕСКИЕ РЕКОМЕНДАЦИИ -->
        <section id="recommendations" class="section">
            <h2>🎯 Практические рекомендации</h2>
            
            {% for rec in recommendations %}
            <div class="{% if rec.priority == 'high' %}recommendation{% else %}metric-card{% endif %}">
                <h4>{{ rec.title }}</h4>
                <p><strong>Категория:</strong> {{ rec.category }}</p>
                <p>{{ rec.content }}</p>
            </div>
            {% endfor %}
            
            <h3>Краткая памятка</h3>
            <div class="interpretation-box">
                <h5>🚀 Quick Start Guide:</h5>
                <ol>
                    <li>Для большинства задач начните с метода <strong>{{ executive_summary.best_method }}</strong></li>
                    <li>Используйте процентильный метод (80-85%) для быстрого baseline</li>
                    <li>Для критичных приложений добавьте оценку неопределенности</li>
                    <li>Мониторьте качество псевдо-меток на контрольной выборке</li>
                    <li>Итеративно корректируйте пороги в процессе обучения</li>
                </ol>
            </div>
        </section>
        
        <footer style="text-align: center; padding: 40px 0; color: #6b7280;">
            <p>Отчет сгенерирован автоматически | {{ metadata.timestamp }}</p>
            <p>Эксперимент: {{ metadata.experiment_name }} | Всего экспериментов: {{ metadata.total_experiments }}</p>
        </footer>
    </div>
</body>
</html>
        """
        
        template = Template(template_str)
        return template.render(**report_parts)


def generate_experiment_report(results_dir: str = 'experiment_results', 
                             timestamp: Optional[str] = None,
                             output_file: Optional[str] = None):
    """
    Главная функция для генерации отчета.
    
    Args:
        results_dir: Директория с результатами эксперимента
        timestamp: Временная метка эксперимента (если не указана, берется последний)
        output_file: Путь для сохранения отчета
        
    Returns:
        Путь к сгенерированному отчету
    """
    generator = ReportGenerator(results_dir)
    generator.load_experiment_results(timestamp)
    return generator.generate_report(output_file)


if __name__ == "__main__":
    # Генерация отчета
    print("Генератор отчетов по экспериментам")
    print("-" * 50)
    
    results_dir = input("Директория с результатами (по умолчанию 'experiment_results'): ").strip()
    results_dir = results_dir or 'experiment_results'
    
    try:
        report_path = generate_experiment_report(results_dir)
        print(f"\n✅ Отчет успешно сгенерирован: {report_path}")
        print("\nОткройте файл в браузере для просмотра.")
    except Exception as e:
        print(f"\n❌ Ошибка при генерации отчета: {e}")
