#!/usr/bin/env python3
"""
One-click benchmark script - Запуск бенчмарка Pandas vs Polars одной командой.

Использование:
    python one_click_benchmark.py              # Быстрый тест (5 минут)
    python one_click_benchmark.py --medium     # Средний тест (30 минут)
    python one_click_benchmark.py --full       # Полный тест (2+ часа)
    python one_click_benchmark.py --tiny       # Минимальный тест (2 минуты)
"""

import argparse
import subprocess
import sys
import yaml
from pathlib import Path
from datetime import datetime


def create_config(test_size='quick'):
    """Создает конфигурацию для выбранного размера теста."""
    
    configs = {
        'tiny': {
            'name': 'Tiny Test (2 minutes)',
            'sizes': [500, 1000],
            'operations': {
                'io': ['read_csv'],
                'filter': ['simple_filter'],
                'groupby': ['single_column_groupby']
            },
            'min_runs': 3,
            'max_runs': 5,
            'columns': 3
        },
        'quick': {
            'name': 'Quick Test (5 minutes)',
            'sizes': [1000, 5000, 10000],
            'operations': {
                'io': ['read_csv', 'write_csv'],
                'filter': ['simple_filter', 'complex_filter'],
                'groupby': ['single_column_groupby', 'multi_aggregation'],
                'sort': ['single_column_sort']
            },
            'min_runs': 3,
            'max_runs': 10,
            'columns': 5
        },
        'medium': {
            'name': 'Medium Test (30 minutes)',
            'sizes': [10000, 50000, 100000],
            'operations': {
                'io': ['read_csv', 'write_csv', 'read_parquet', 'write_parquet'],
                'filter': ['simple_filter', 'complex_filter', 'isin_filter'],
                'groupby': ['single_column_groupby', 'multi_column_groupby', 'multi_aggregation'],
                'sort': ['single_column_sort', 'multi_column_sort'],
                'join': ['inner_join', 'left_join']
            },
            'min_runs': 5,
            'max_runs': 20,
            'columns': 10
        },
        'full': {
            'name': 'Full Test (2+ hours)',
            'sizes': [10000, 100000, 1000000],
            'operations': {
                'io': ['read_csv', 'write_csv', 'read_parquet', 'write_parquet'],
                'filter': ['simple_filter', 'complex_filter', 'isin_filter', 'pattern_filter'],
                'groupby': ['single_column_groupby', 'multi_column_groupby', 'multi_aggregation', 'window_functions'],
                'sort': ['single_column_sort', 'multi_column_sort', 'custom_sort'],
                'join': ['inner_join', 'left_join', 'multi_key_join', 'asof_join'],
                'string': ['contains_operation', 'replace_operation', 'extract_operation']
            },
            'min_runs': 5,
            'max_runs': 50,
            'columns': 20
        }
    }
    
    config_data = configs[test_size]
    
    return {
        'benchmark': {
            'name': config_data['name'],
            'version': '1.0.0',
            'description': f'Automated {test_size} benchmark run'
        },
        'environment': {
            'seed': 42
        },
        'data_generation': {
            'sizes': config_data['sizes'],
            'datasets': {
                'numeric': {
                    'enabled': True,
                    'columns': config_data['columns']
                },
                'string': {
                    'enabled': test_size in ['medium', 'full'],
                    'columns': max(3, config_data['columns'] // 2)
                },
                'mixed': {
                    'enabled': test_size == 'full',
                    'numeric_columns': config_data['columns'] // 2,
                    'string_columns': config_data['columns'] // 4
                }
            }
        },
        'operations': config_data['operations'],
        'profiling': {
            'min_runs': config_data['min_runs'],
            'max_runs': config_data['max_runs'],
            'target_cv': 0.05 if test_size == 'full' else 0.1,
            'warmup_runs': 1 if test_size in ['tiny', 'quick'] else 2
        },
        'output': {
            'generate_plots': True,
            'create_report': True,
            'save_raw_results': True
        }
    }


def check_environment():
    """Проверяет, что окружение готово к запуску."""
    print("🔍 Проверка окружения...")
    
    # Проверка Python версии
    if sys.version_info < (3, 11):
        print("❌ Требуется Python 3.11 или выше")
        print(f"   Текущая версия: {sys.version}")
        return False
    
    # Проверка основных модулей
    required_modules = ['pandas', 'polars', 'plotly', 'numpy', 'yaml']
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        print("❌ Отсутствуют необходимые модули:")
        for module in missing_modules:
            print(f"   - {module}")
        print("\n💡 Установите зависимости командой:")
        print("   pip install -r requirements.txt")
        return False
    
    print("✅ Окружение готово к работе")
    return True


def main():
    """Основная функция."""
    parser = argparse.ArgumentParser(
        description='Запуск бенчмарка Pandas vs Polars одной командой'
    )
    parser.add_argument(
        '--tiny', action='store_true',
        help='Минимальный тест (2 минуты)'
    )
    parser.add_argument(
        '--medium', action='store_true',
        help='Средний тест (30 минут)'
    )
    parser.add_argument(
        '--full', action='store_true',
        help='Полный тест (2+ часа)'
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='Директория для сохранения результатов'
    )
    
    args = parser.parse_args()
    
    # Определение размера теста
    if args.tiny:
        test_size = 'tiny'
    elif args.medium:
        test_size = 'medium'
    elif args.full:
        test_size = 'full'
    else:
        test_size = 'quick'  # По умолчанию
    
    # Проверка окружения
    if not check_environment():
        sys.exit(1)
    
    # Создание временной конфигурации
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    config_path = Path(f'configs/auto_config_{test_size}_{timestamp}.yaml')
    
    print(f"\n🚀 Запуск {test_size.upper()} бенчмарка")
    print(f"📝 Создание конфигурации: {config_path}")
    
    # Сохранение конфигурации
    config_path.parent.mkdir(exist_ok=True)
    config = create_config(test_size)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    # Подготовка команды запуска
    cmd = [
        sys.executable,
        'scripts/run_benchmark.py',
        '--config', str(config_path)
    ]
    
    if args.output_dir:
        cmd.extend(['--output-dir', args.output_dir])
    
    # Запуск бенчмарка
    print(f"🏃 Запуск команды: {' '.join(cmd)}")
    print("-" * 80)
    
    try:
        result = subprocess.run(cmd, check=True)
        
        print("\n" + "=" * 80)
        print("✅ БЕНЧМАРК ЗАВЕРШЕН УСПЕШНО!")
        print("=" * 80)
        
        # Поиск созданного отчета
        report_pattern = "results/reports/benchmark_report_*.html"
        if args.output_dir:
            report_pattern = f"{args.output_dir}/reports/benchmark_report_*.html"
        
        report_files = list(Path(".").glob(report_pattern))
        if report_files:
            latest_report = max(report_files, key=lambda p: p.stat().st_mtime)
            print(f"\n📊 Отчет создан: {latest_report}")
            print(f"💡 Откройте в браузере для просмотра результатов")
            
            # Попытка автоматически открыть отчет
            if sys.platform == 'darwin':  # macOS
                subprocess.run(['open', str(latest_report)])
            elif sys.platform == 'win32':  # Windows
                subprocess.run(['start', '', str(latest_report)], shell=True)
            elif sys.platform.startswith('linux'):  # Linux
                subprocess.run(['xdg-open', str(latest_report)])
        
        # Удаление временной конфигурации
        config_path.unlink()
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Ошибка при выполнении бенчмарка: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Бенчмарк прерван пользователем")
        print("💡 Используйте 'python scripts/run_benchmark.py --resume' для продолжения")
        sys.exit(1)


if __name__ == '__main__':
    main()
