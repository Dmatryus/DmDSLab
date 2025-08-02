#!/usr/bin/env python3
"""
Скрипт быстрой настройки окружения для запуска бенчмарка.
"""

import os
import sys
import subprocess
from pathlib import Path


def run_command(cmd, description):
    """Выполняет команду и показывает результат."""
    print(f"\n{description}...")
    print(f"Команда: {cmd}")
    print("-" * 50)
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Успешно!")
        if result.stdout:
            print(result.stdout)
    else:
        print("❌ Ошибка!")
        if result.stderr:
            print(result.stderr)
        if result.stdout:
            print(result.stdout)
    
    return result.returncode == 0


def main():
    """Основная функция настройки."""
    print("=" * 70)
    print("БЫСТРАЯ НАСТРОЙКА ОКРУЖЕНИЯ PANDAS VS POLARS BENCHMARK")
    print("=" * 70)
    
    # Определяем пути
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # Переходим в корень проекта
    os.chdir(project_root)
    print(f"\nРабочая директория: {project_root}")
    
    # 1. Создаем __init__.py файлы
    print("\n1. Создание __init__.py файлов...")
    if (script_dir / 'init_modules.py').exists():
        run_command(f"{sys.executable} scripts/init_modules.py", "Инициализация модулей")
    else:
        print("⚠️  Скрипт init_modules.py не найден, создаем файлы вручную...")
        
        # Создаем __init__.py в основных директориях
        for dir_name in ['src', 'src/core', 'src/data', 'src/profiling', 
                        'src/operations', 'src/analysis', 'src/reporting', 
                        'src/utils', 'tests', 'tests/unit', 'tests/integration']:
            dir_path = project_root / dir_name
            if dir_path.exists():
                init_file = dir_path / '__init__.py'
                if not init_file.exists():
                    init_file.write_text('"""Package initialization."""\n')
                    print(f"✅ Создан: {init_file.relative_to(project_root)}")
    
    # 2. Устанавливаем минимальные зависимости для запуска валидации
    print("\n2. Установка минимальных зависимостей...")
    min_deps = "PyYAML jsonschema colorlog click"
    if not run_command(f"{sys.executable} -m pip install {min_deps}", 
                      "Установка базовых зависимостей"):
        print("❌ Не удалось установить зависимости!")
        print("Попробуйте выполнить вручную:")
        print(f"  pip install {min_deps}")
        return
    
    # 3. Проверяем валидацию конфигурации
    print("\n3. Проверка валидации конфигурации...")
    config_file = project_root / 'configs' / 'default_config.yaml'
    
    if config_file.exists():
        if run_command(f"{sys.executable} scripts/validate_config.py --config {config_file}", 
                      "Валидация конфигурации"):
            print("\n✅ Конфигурация валидна!")
        else:
            print("\n⚠️  Проблемы с конфигурацией, но это не критично для настройки.")
    else:
        print("⚠️  Файл конфигурации не найден!")
    
    # 4. Установка всех зависимостей
    print("\n4. Установка всех зависимостей из requirements.txt...")
    requirements_file = project_root / 'requirements.txt'
    
    if requirements_file.exists():
        if run_command(f"{sys.executable} -m pip install -r requirements.txt", 
                      "Установка всех зависимостей"):
            print("\n✅ Все зависимости установлены!")
        else:
            print("\n⚠️  Некоторые зависимости не удалось установить.")
            print("Это может быть нормально для опциональных пакетов.")
    
    # 5. Финальная проверка
    print("\n5. Финальная проверка окружения...")
    if (script_dir / 'check_dependencies.py').exists():
        run_command(f"{sys.executable} scripts/check_dependencies.py", 
                   "Проверка зависимостей")
    
    print("\n" + "=" * 70)
    print("НАСТРОЙКА ЗАВЕРШЕНА!")
    print("=" * 70)
    print("\nТеперь вы можете:")
    print("1. Проверить конфигурацию:")
    print(f"   python scripts/validate_config.py --config configs/default_config.yaml")
    print("\n2. Создать пример конфигурации:")
    print(f"   python scripts/validate_config.py --create-example")
    print("\n3. Запустить тесты:")
    print(f"   python -m pytest tests/unit/test_config.py -v")


if __name__ == '__main__':
    main()
