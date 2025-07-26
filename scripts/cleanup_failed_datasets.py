#!/usr/bin/env python3
"""
Автоматическая очистка недоступных датасетов из базы
"""

from dmdslab.datasets.uci_dataset_manager import UCIDatasetManager
import time

def cleanup_failed_datasets(auto_confirm=False):
    """Находит и удаляет все недоступные датасеты"""
    manager = UCIDatasetManager()
    all_datasets = manager.filter_datasets()
    
    print("🔍 Поиск недоступных датасетов...\n")
    
    failed_datasets = []
    
    # Проверяем каждый датасет
    for i, dataset in enumerate(all_datasets, 1):
        print(f"[{i}/{len(all_datasets)}] Проверка {dataset.name} (ID: {dataset.id})...", end=" ", flush=True)
        try:
            # Пытаемся загрузить только метаданные
            _ = manager.load_dataset(dataset.id)
            print("✅")
        except Exception as e:
            print("❌")
            error_msg = str(e)
            if "not available for import" in error_msg or "Failed to load" in error_msg:
                failed_datasets.append({
                    'id': dataset.id,
                    'name': dataset.name,
                    'error': error_msg
                })
    
    if not failed_datasets:
        print("\n✅ Все датасеты доступны!")
        return
    
    # Показываем найденные проблемные датасеты
    print(f"\n⚠️  Найдено {len(failed_datasets)} недоступных датасетов:")
    for ds in failed_datasets:
        print(f"   - {ds['name']} (ID: {ds['id']})")
    
    # Запрашиваем подтверждение
    if not auto_confirm:
        response = input(f"\n❓ Удалить эти {len(failed_datasets)} датасетов? (y/N): ")
        if response.lower() != 'y':
            print("❌ Отменено")
            return
    
    # Удаляем
    print("\n🗑️  Удаление недоступных датасетов...")
    deleted_count = 0
    
    for ds in failed_datasets:
        print(f"   Удаление {ds['name']}...", end=" ", flush=True)
        if manager.delete_dataset(ds['id']):
            print("✅")
            deleted_count += 1
        else:
            print("❌")
        time.sleep(0.1)  # Небольшая задержка для визуализации
    
    # Итоговая статистика
    print(f"\n📊 Результаты:")
    print(f"   - Проверено датасетов: {len(all_datasets)}")
    print(f"   - Найдено недоступных: {len(failed_datasets)}")
    print(f"   - Успешно удалено: {deleted_count}")
    
    stats = manager.get_statistics()
    print(f"   - Осталось в базе: {stats['total_datasets']}")

if __name__ == "__main__":
    import sys
    
    # Проверяем аргументы командной строки
    auto_confirm = "--auto" in sys.argv or "-y" in sys.argv
    
    if auto_confirm:
        print("🚀 Автоматический режим (без подтверждений)")
    
    cleanup_failed_datasets(auto_confirm)
