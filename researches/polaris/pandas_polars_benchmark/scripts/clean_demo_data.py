#!/usr/bin/env python3
"""
Скрипт для очистки временных данных после демонстрации.
"""

import shutil
from pathlib import Path
import sys


def clean_demo_data():
    """Очищает временные файлы и директории."""
    print("🧹 Очистка демонстрационных данных...")
    print("-" * 40)
    
    # Директории для очистки
    dirs_to_clean = [
        'data/demo',
        'data/test',
        'results/checkpoints',
        'logs'
    ]
    
    # Файлы для удаления
    patterns_to_clean = [
        'checkpoint_*.json',
        '*.log',
        'test_*.pkl',
        'result_*.pkl'
    ]
    
    total_removed = 0
    
    # Очистка директорий
    for dir_path in dirs_to_clean:
        path = Path(dir_path)
        if path.exists():
            try:
                if path.name == 'logs':
                    # Для логов удаляем только файлы, оставляем директорию
                    log_files = list(path.glob('*.log'))
                    for log_file in log_files:
                        log_file.unlink()
                        total_removed += 1
                    print(f"✓ Очищено {len(log_files)} лог-файлов в {dir_path}")
                else:
                    # Для остальных удаляем всю директорию
                    file_count = sum(1 for _ in path.rglob('*') if _.is_file())
                    shutil.rmtree(path)
                    total_removed += file_count
                    print(f"✓ Удалена директория {dir_path} ({file_count} файлов)")
            except Exception as e:
                print(f"⚠️  Ошибка при очистке {dir_path}: {e}")
    
    # Очистка временных файлов по паттернам
    for pattern in patterns_to_clean:
        files = list(Path('.').rglob(pattern))
        for file in files:
            try:
                file.unlink()
                total_removed += 1
            except Exception as e:
                print(f"⚠️  Ошибка при удалении {file}: {e}")
        
        if files:
            print(f"✓ Удалено {len(files)} файлов по паттерну '{pattern}'")
    
    print(f"\n✅ Очистка завершена. Удалено файлов: {total_removed}")
    
    # Воссоздаем пустые директории
    for dir_path in ['logs', 'data', 'results']:
        Path(dir_path).mkdir(exist_ok=True)
    
    print("\n📁 Структура директорий восстановлена")


def main():
    """Основная функция."""
    print("="*60)
    print("ОЧИСТКА ДЕМОНСТРАЦИОННЫХ ДАННЫХ")
    print("="*60)
    print()
    
    # Подтверждение
    response = input("Вы уверены, что хотите удалить все демонстрационные данные? (y/n): ")
    if response.lower() != 'y':
        print("Очистка отменена")
        return
    
    print()
    clean_demo_data()
    
    print("\n✨ Проект готов к новому запуску демонстрации!")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nОчистка прервана пользователем")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        sys.exit(1)
