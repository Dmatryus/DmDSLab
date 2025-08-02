#!/usr/bin/env python3
"""
Скрипт для проверки корректности импортов.
"""

import sys
import os
from pathlib import Path

print("=== ПРОВЕРКА ИМПОРТОВ ===")
print(f"Python версия: {sys.version}")
print(f"Текущая директория: {os.getcwd()}")
print(f"Путь к скрипту: {__file__}")

# Настройка путей
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
src_dir = project_root / 'src'

print(f"\nНастройка путей:")
print(f"  script_dir: {script_dir}")
print(f"  project_root: {project_root}")
print(f"  src_dir: {src_dir}")
print(f"  src_dir существует: {src_dir.exists()}")

# Добавляем src в sys.path
sys.path.insert(0, str(src_dir))
print(f"\nsys.path после добавления src:")
for i, p in enumerate(sys.path[:3]):  # Показываем первые 3 пути
    print(f"  [{i}] {p}")

# Пробуем импортировать модули
print("\nПроверка импортов:")

try:
    from core.config import Config
    print("✓ core.config импортирован успешно")
except ImportError as e:
    print(f"✗ Ошибка импорта core.config: {e}")

try:
    from data import DataGenerator, DataLoader, DataSaver
    print("✓ data модули импортированы успешно")
except ImportError as e:
    print(f"✗ Ошибка импорта data: {e}")

try:
    from utils.logging import setup_logging
    print("✓ utils.logging импортирован успешно")
except ImportError as e:
    print(f"✗ Ошибка импорта utils.logging: {e}")

try:
    import pandas as pd
    print(f"✓ pandas {pd.__version__} импортирован успешно")
except ImportError as e:
    print(f"✗ pandas не установлен: {e}")

try:
    import polars as pl
    print(f"✓ polars {pl.__version__} импортирован успешно")
except ImportError as e:
    print(f"✗ polars не установлен: {e}")

print("\n=== ПРОВЕРКА ЗАВЕРШЕНА ===")
