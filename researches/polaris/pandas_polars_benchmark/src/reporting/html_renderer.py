"""
HTML Renderer для генерации финальных отчетов бенчмаркинга.
Объединяет графики, таблицы и статистику в единый интерактивный HTML отчет.
"""

from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from datetime import datetime
import json
from dataclasses import dataclass
from jinja2 import Environment, FileSystemLoader, Template
import plotly.graph_objects as go
import logging


@dataclass
class ReportSection:
    """Секция отчета."""
    title: str
    content: str
    section_id: str
    order: int = 0
    
    
@dataclass
class ReportConfig:
    """Конфигурация для генерации отчета."""
    title: str = "Pandas vs Polars Benchmark Report"
    subtitle: Optional[str] = None
    author: Optional[str] = None
    description: Optional[str] = None
    include_toc: bool = True
    include_summary: bool = True
    include_methodology: bool = True
    include_recommendations: bool = True
    theme: str = "light"  # light или dark
    

class HTMLRenderer:
    """Генератор HTML отчетов для результатов бенчмаркинга."""
    
    def __init__(self, template_dir: Optional[Path] = None, logger=None):
        """
        Инициализация рендерера.
        
        Args:
            template_dir: Директория с шаблонами (если None, используется встроенный шаблон)
            logger: Логгер для вывода информации
        """
        self.logger = logger or logging.getLogger(__name__)
        self.template_dir = template_dir
        
        if template_dir and template_dir.exists():
            self.env = Environment(loader=FileSystemLoader(str(template_dir)))
        else:
            # Используем встроенный шаблон
            self.env = Environment()
            self.env.from_string(self._get_default_template())
            
        self.sections: List[ReportSection] = []
        
    def render_report(
        self,
        figures: Dict[str, go.Figure],
        summary_data: Dict[str, Any],
        config: Optional[ReportConfig] = None,
        output_path: Optional[Path] = None
    ) -> str:
        """
        Генерация полного HTML отчета.
        
        Args:
            figures: Словарь с Plotly фигурами
            summary_data: Сводные данные и статистика
            config: Конфигурация отчета
            output_path: Путь для сохранения (если None, возвращает HTML строку)
            
        Returns:
            str: HTML содержимое отчета
        """
        if not config:
            config = ReportConfig()
            
        # Очистка секций
        self.sections = []
        
        # Добавление секций
        if config.include_summary:
            self._add_summary_section(summary_data)
            
        if config.include_methodology:
            self._add_methodology_section(summary_data)
            
        # Добавление графиков
        self._add_performance_comparison_section(figures)
        self._add_detailed_analysis_section(figures)
        self._add_distribution_section(figures)
        
        if config.include_recommendations:
            self._add_recommendations_section(summary_data)
            
        # Генерация HTML
        html_content = self._render_html(config)
        
        # Сохранение если указан путь
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(html_content, encoding='utf-8')
            if self.logger:
                self.logger.info(f"Отчет сохранен: {output_path}")
                
        return html_content
        
    def _add_summary_section(self, summary_data: Dict[str, Any]):
        """Добавление секции со сводкой результатов."""
        overall = summary_data.get('overall_summary', {})
        
        content = f"""
        <div class="summary-cards">
            <div class="card">
                <h3>Общее ускорение</h3>
                <div class="metric">{overall.get('avg_speedup', 0):.2f}x</div>
                <p>Среднее ускорение Polars</p>
            </div>
            
            <div class="card">
                <h3>Победы Polars</h3>
                <div class="metric">{overall.get('polars_win_rate', 0):.1f}%</div>
                <p>Процент операций где Polars быстрее</p>
            </div>
            
            <div class="card">
                <h3>Максимальное ускорение</h3>
                <div class="metric">{overall.get('max_speedup', 0):.2f}x</div>
                <p>Лучший результат Polars</p>
            </div>
            
            <div class="card">
                <h3>Всего сравнений</h3>
                <div class="metric">{overall.get('total_comparisons', 0)}</div>
                <p>Количество тестов</p>
            </div>
        </div>
        
        <div class="key-findings">
            <h3>Ключевые выводы</h3>
            <ul>
                {self._generate_key_findings(summary_data)}
            </ul>
        </div>
        """
        
        self.sections.append(ReportSection(
            title="Сводка результатов",
            content=content,
            section_id="summary",
            order=1
        ))
        
    def _add_methodology_section(self, summary_data: Dict[str, Any]):
        """Добавление секции с методологией."""
        metadata = summary_data.get('metadata', {})
        
        content = f"""
        <div class="methodology">
            <h3>Параметры тестирования</h3>
            
            <table class="params-table">
                <tr>
                    <td>Дата проведения:</td>
                    <td>{metadata.get('run_date', 'Н/Д')}</td>
                </tr>
                <tr>
                    <td>Версия бенчмарка:</td>
                    <td>{metadata.get('benchmark_version', 'Н/Д')}</td>
                </tr>
                <tr>
                    <td>Система:</td>
                    <td>{metadata.get('system_info', {}).get('os', 'Н/Д')}</td>
                </tr>
                <tr>
                    <td>CPU:</td>
                    <td>{metadata.get('system_info', {}).get('cpu', 'Н/Д')}</td>
                </tr>
                <tr>
                    <td>Память:</td>
                    <td>{metadata.get('system_info', {}).get('memory', 'Н/Д')}</td>
                </tr>
            </table>
            
            <h3>Методология измерений</h3>
            <ul>
                <li>Каждая операция выполнялась многократно для статистической достоверности</li>
                <li>Выбросы удалялись методом IQR (межквартильный размах)</li>
                <li>Время измерялось с помощью высокоточных таймеров</li>
                <li>Память отслеживалась в отдельном процессе</li>
            </ul>
        </div>
        """
        
        self.sections.append(ReportSection(
            title="Методология",
            content=content,
            section_id="methodology",
            order=2
        ))
        
    def _add_performance_comparison_section(self, figures: Dict[str, go.Figure]):
        """Добавление секции сравнения производительности."""
        charts_html = []
        
        # Основной график сравнения
        if 'comparison_bar' in figures:
            charts_html.append(self._figure_to_html(figures['comparison_bar']))
            
        # График speedup
        if 'speedup_chart' in figures:
            charts_html.append(self._figure_to_html(figures['speedup_chart']))
            
        content = f"""
        <div class="charts-container">
            {''.join(charts_html)}
        </div>
        
        <div class="interpretation">
            <p><strong>Интерпретация:</strong> Графики показывают сравнение времени выполнения 
            различных операций. Более высокие значения speedup означают большее преимущество Polars.</p>
        </div>
        """
        
        self.sections.append(ReportSection(
            title="Сравнение производительности",
            content=content,
            section_id="performance",
            order=3
        ))
        
    def _add_detailed_analysis_section(self, figures: Dict[str, go.Figure]):
        """Добавление секции детального анализа."""
        charts_html = []
        
        # Тепловая карта
        if 'heatmap' in figures:
            charts_html.append(self._figure_to_html(figures['heatmap']))
            
        # График зависимости от размера
        if 'timeline' in figures:
            charts_html.append(self._figure_to_html(figures['timeline']))
            
        content = f"""
        <div class="charts-container">
            {''.join(charts_html)}
        </div>
        
        <div class="analysis-notes">
            <h4>Наблюдения:</h4>
            <ul>
                <li>Производительность зависит от размера данных и типа операции</li>
                <li>Некоторые операции показывают нелинейное масштабирование</li>
                <li>Polars особенно эффективен на больших датасетах</li>
            </ul>
        </div>
        """
        
        self.sections.append(ReportSection(
            title="Детальный анализ",
            content=content,
            section_id="detailed-analysis",
            order=4
        ))
        
    def _add_distribution_section(self, figures: Dict[str, go.Figure]):
        """Добавление секции с распределениями."""
        charts_html = []
        
        # Box plots
        if 'distribution_box' in figures:
            charts_html.append(self._figure_to_html(figures['distribution_box']))
            
        # Таблица с детальными результатами
        if 'summary_table' in figures:
            charts_html.append(self._figure_to_html(figures['summary_table']))
            
        content = f"""
        <div class="charts-container">
            {''.join(charts_html)}
        </div>
        
        <div class="distribution-notes">
            <p>Box plots показывают распределение времени выполнения для каждой операции,
            включая медиану, квартили и выбросы.</p>
        </div>
        """
        
        self.sections.append(ReportSection(
            title="Распределение результатов",
            content=content,
            section_id="distributions",
            order=5
        ))
        
    def _add_recommendations_section(self, summary_data: Dict[str, Any]):
        """Добавление секции с рекомендациями."""
        recommendations = self._generate_recommendations(summary_data)
        
        content = f"""
        <div class="recommendations">
            <h3>Рекомендации по выбору библиотеки</h3>
            
            <div class="recommendation-box polars">
                <h4>Используйте Polars когда:</h4>
                <ul>
                    {recommendations['polars']}
                </ul>
            </div>
            
            <div class="recommendation-box pandas">
                <h4>Используйте Pandas когда:</h4>
                <ul>
                    {recommendations['pandas']}
                </ul>
            </div>
            
            <div class="general-advice">
                <h4>Общие рекомендации:</h4>
                <ul>
                    <li>Проведите тестирование на ваших реальных данных</li>
                    <li>Учитывайте не только скорость, но и экосистему библиотек</li>
                    <li>Рассмотрите возможность использования обеих библиотек в разных частях pipeline</li>
                </ul>
            </div>
        </div>
        """
        
        self.sections.append(ReportSection(
            title="Рекомендации",
            content=content,
            section_id="recommendations",
            order=6
        ))
        
    def _figure_to_html(self, fig: go.Figure, include_plotlyjs: str = 'cdn') -> str:
        """Преобразование Plotly фигуры в HTML."""
        return fig.to_html(
            include_plotlyjs=include_plotlyjs,
            div_id=None,
            config={'displayModeBar': True, 'displaylogo': False}
        )
        
    def _generate_key_findings(self, summary_data: Dict[str, Any]) -> str:
        """Генерация ключевых выводов."""
        findings = []
        overall = summary_data.get('overall_summary', {})
        
        avg_speedup = overall.get('avg_speedup', 1)
        if avg_speedup > 2:
            findings.append(f"<li>Polars показывает <strong>значительное преимущество</strong> со средним ускорением {avg_speedup:.1f}x</li>")
        elif avg_speedup > 1.5:
            findings.append(f"<li>Polars <strong>заметно быстрее</strong> со средним ускорением {avg_speedup:.1f}x</li>")
        elif avg_speedup > 1.1:
            findings.append(f"<li>Polars показывает <strong>умеренное преимущество</strong> со средним ускорением {avg_speedup:.1f}x</li>")
        else:
            findings.append("<li>Библиотеки показывают <strong>сопоставимую производительность</strong></li>")
            
        # Анализ по операциям
        by_operation = summary_data.get('summary_by_operation', {})
        best_ops = sorted(by_operation.items(), key=lambda x: x[1].get('avg_speedup', 0), reverse=True)[:3]
        
        if best_ops:
            ops_list = ', '.join([op[0] for op in best_ops])
            findings.append(f"<li>Наибольшее ускорение достигается в операциях: <strong>{ops_list}</strong></li>")
            
        return '\n'.join(findings)
        
    def _generate_recommendations(self, summary_data: Dict[str, Any]) -> Dict[str, str]:
        """Генерация рекомендаций на основе результатов."""
        by_operation = summary_data.get('summary_by_operation', {})
        
        polars_recs = []
        pandas_recs = []
        
        # Анализируем результаты по операциям
        for op_name, op_data in by_operation.items():
            if op_data.get('avg_speedup', 1) > 2:
                polars_recs.append(f"<li>Операции <strong>{op_name}</strong> (ускорение {op_data['avg_speedup']:.1f}x)</li>")
            elif op_data.get('avg_speedup', 1) < 0.8:
                pandas_recs.append(f"<li>Операции <strong>{op_name}</strong></li>")
                
        # Общие рекомендации
        if not polars_recs:
            polars_recs.append("<li>Работа с большими датасетами</li>")
            polars_recs.append("<li>Операции требующие параллелизации</li>")
            
        if not pandas_recs:
            pandas_recs.append("<li>Работа с небольшими датасетами</li>")
            pandas_recs.append("<li>Необходима совместимость с существующим кодом</li>")
            pandas_recs.append("<li>Требуется богатая экосистема библиотек</li>")
            
        return {
            'polars': '\n'.join(polars_recs),
            'pandas': '\n'.join(pandas_recs)
        }
        
    def _render_html(self, config: ReportConfig) -> str:
        """Рендеринг финального HTML."""
        # Сортировка секций
        sorted_sections = sorted(self.sections, key=lambda x: x.order)
        
        # Подготовка данных для шаблона
        template_data = {
            'title': config.title,
            'subtitle': config.subtitle,
            'author': config.author,
            'description': config.description,
            'generation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'sections': sorted_sections,
            'include_toc': config.include_toc,
            'theme': config.theme
        }
        
        # Рендеринг
        if self.template_dir:
            template = self.env.get_template('report.html')
        else:
            template = self.env.from_string(self._get_default_template())
            
        return template.render(**template_data)
        
    def _get_default_template(self) -> str:
        """Встроенный шаблон HTML отчета."""
        return '''
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        :root {
            --primary-color: #2563eb;
            --secondary-color: #10b981;
            --background: #ffffff;
            --text-color: #1f2937;
            --border-color: #e5e7eb;
            --card-background: #f9fafb;
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: var(--text-color);
            background: var(--background);
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
        }
        
        header {
            text-align: center;
            padding: 3rem 0;
            border-bottom: 2px solid var(--border-color);
            margin-bottom: 3rem;
        }
        
        h1 {
            font-size: 2.5rem;
            color: var(--primary-color);
            margin-bottom: 1rem;
        }
        
        h2 {
            font-size: 2rem;
            color: var(--text-color);
            margin: 2rem 0 1rem;
        }
        
        h3 {
            font-size: 1.5rem;
            color: var(--text-color);
            margin: 1.5rem 0 1rem;
        }
        
        .subtitle {
            font-size: 1.25rem;
            color: #6b7280;
        }
        
        .metadata {
            margin-top: 1rem;
            font-size: 0.875rem;
            color: #9ca3af;
        }
        
        .section {
            margin-bottom: 3rem;
            padding: 2rem;
            background: var(--card-background);
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }
        
        .summary-cards {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1.5rem;
            margin: 2rem 0;
        }
        
        .card {
            background: white;
            padding: 1.5rem;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            text-align: center;
        }
        
        .card h3 {
            font-size: 1rem;
            color: #6b7280;
            margin-bottom: 0.5rem;
        }
        
        .metric {
            font-size: 2.5rem;
            font-weight: bold;
            color: var(--primary-color);
            margin: 0.5rem 0;
        }
        
        .card p {
            font-size: 0.875rem;
            color: #9ca3af;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
        }
        
        th, td {
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }
        
        th {
            background: var(--card-background);
            font-weight: 600;
        }
        
        .charts-container {
            margin: 2rem 0;
        }
        
        .recommendation-box {
            margin: 1.5rem 0;
            padding: 1.5rem;
            border-radius: 8px;
            border-left: 4px solid;
        }
        
        .recommendation-box.polars {
            background: #f0fdf4;
            border-color: var(--secondary-color);
        }
        
        .recommendation-box.pandas {
            background: #eff6ff;
            border-color: var(--primary-color);
        }
        
        .toc {
            background: var(--card-background);
            padding: 1.5rem;
            border-radius: 8px;
            margin-bottom: 2rem;
        }
        
        .toc h3 {
            margin-bottom: 1rem;
        }
        
        .toc ul {
            list-style: none;
        }
        
        .toc a {
            color: var(--primary-color);
            text-decoration: none;
            display: block;
            padding: 0.5rem 0;
        }
        
        .toc a:hover {
            text-decoration: underline;
        }
        
        @media (max-width: 768px) {
            .container {
                padding: 1rem;
            }
            
            h1 {
                font-size: 2rem;
            }
            
            .summary-cards {
                grid-template-columns: 1fr;
            }
        }
        
        @media print {
            .section {
                page-break-inside: avoid;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>{{ title }}</h1>
            {% if subtitle %}
            <p class="subtitle">{{ subtitle }}</p>
            {% endif %}
            <div class="metadata">
                {% if author %}
                <span>Автор: {{ author }}</span> | 
                {% endif %}
                <span>Дата генерации: {{ generation_time }}</span>
            </div>
        </header>
        
        {% if include_toc %}
        <nav class="toc">
            <h3>Содержание</h3>
            <ul>
                {% for section in sections %}
                <li><a href="#{{ section.section_id }}">{{ section.title }}</a></li>
                {% endfor %}
            </ul>
        </nav>
        {% endif %}
        
        <main>
            {% for section in sections %}
            <section id="{{ section.section_id }}" class="section">
                <h2>{{ section.title }}</h2>
                {{ section.content|safe }}
            </section>
            {% endfor %}
        </main>
        
        <footer style="text-align: center; padding: 2rem; color: #9ca3af;">
            <p>Сгенерировано системой бенчмаркинга Pandas vs Polars</p>
        </footer>
    </div>
</body>
</html>
        '''