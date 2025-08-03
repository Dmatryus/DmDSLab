#!/usr/bin/env python3
"""
Финальная проверка готовности Фазы 3 после исправления проблем совместимости.
"""

import sys
import platform
from pathlib import Path

# Добавляем путь к src
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def print_header():
    """Выводит заголовок."""
    print("=" * 80)
    print("ФИНАЛЬНАЯ ПРОВЕРКА ФАЗЫ 3: СИСТЕМА ПРОФИЛИРОВАНИЯ")
    print("=" * 80)
    print(f"\nПлатформа: {platform.system()} {platform.version()}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Система: {sys.platform}")
    print()


def check_imports():
    """Проверяет основные импорты."""
    print("1. Проверка импортов...")
    print("-" * 40)

    try:
        # Core
        from core import (
            Config,
            CheckpointManager,
            BenchmarkState,
            TaskIdentifier,
            ProgressTracker,
        )

        print("✅ Core модули")

        # Profiling
        from profiling import (
            MemoryTracker,
            Timer,
            Profiler,
            ProfileResult,
            ProfilingConfig,
            get_profiler,
            measure_memory,
            measure_time,
        )

        print("✅ Profiling модули")

        # Utils
        from utils import setup_logging, get_logger

        print("✅ Utils модули")

        # Data
        from data import DataGenerator, DataLoader, DataSaver

        print("✅ Data модули")

        return True

    except ImportError as e:
        print(f"❌ Ошибка импорта: {e}")
        return False


def check_checkpoint_functionality():
    """Проверяет функциональность чекпоинтов."""
    print("\n2. Проверка системы чекпоинтов...")
    print("-" * 40)

    from core import CheckpointManager, TaskIdentifier
    from profiling import ProfileResult

    test_dir = Path("data/final_test")
    test_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Создание и сохранение
        manager = CheckpointManager(checkpoint_dir=test_dir)
        state = manager.initialize_state("final_test", {}, 3)

        # Добавляем результат
        result = ProfileResult(
            operation_name="test",
            library="test_lib",
            success=True,
            mean_time=0.1,
            peak_memory_mb=50.0,
        )
        manager.update_progress("task_1", result=result)

        # Сохраняем
        saved = manager.save_checkpoint(force=True)

        if saved:
            print("✅ Сохранение чекпоинтов работает")
        else:
            print("❌ Ошибка сохранения чекпоинтов")
            return False

        # Загружаем
        new_manager = CheckpointManager(checkpoint_dir=test_dir)
        loaded = new_manager.load_checkpoint()

        if loaded and loaded.completed_operations == 1:
            print("✅ Загрузка чекпоинтов работает")
        else:
            print("❌ Ошибка загрузки чекпоинтов")
            return False

        # Очистка
        manager.clear_checkpoint()
        import shutil

        shutil.rmtree(test_dir)

        return True

    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return False


def check_profiling():
    """Проверяет профилирование."""
    print("\n3. Проверка профилирования...")
    print("-" * 40)

    from profiling import get_profiler, ProfilingConfig
    import time

    try:
        config = ProfilingConfig(min_runs=2, max_runs=3, isolate_process=False)

        with get_profiler(config) as profiler:

            def simple_op():
                time.sleep(0.01)
                return [1, 2, 3]

            result = profiler.profile_operation(
                simple_op, operation_name="test", library="test"
            )

            if result.success and result.mean_time > 0:
                print("✅ Профилирование работает")
                print(f"   Режим: {'isolated' if config.isolate_process else 'inline'}")
                print(f"   Время: {result.mean_time:.3f}с")
                print(f"   Память: {result.peak_memory_mb:.1f}MB")
                return True
            else:
                print("❌ Ошибка профилирования")
                return False

    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return False


def check_progress_tracking():
    """Проверяет отслеживание прогресса."""
    print("\n4. Проверка отслеживания прогресса...")
    print("-" * 40)

    from core import ProgressTracker
    import time

    try:
        # Без progress bar для теста
        tracker = ProgressTracker(5, show_progress_bar=False)

        for i in range(3):
            tracker.start_operation(f"op_{i}", "test", "data")
            time.sleep(0.01)
            tracker.end_operation(success=True)

        info = tracker.get_progress_info()

        if info["completed"] == 3:
            print("✅ Отслеживание прогресса работает")
            print(f"   Выполнено: {info['completed']}/{info['total']}")
            print(f"   Прогресс: {info['progress_percentage']:.0f}%")
            return True
        else:
            print("❌ Ошибка отслеживания прогресса")
            return False

    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return False


def check_platform_specific():
    """Проверяет платформо-специфичные особенности."""
    print("\n5. Проверка платформо-специфичных функций...")
    print("-" * 40)

    # Проверка fcntl
    if sys.platform == "win32":
        try:
            import fcntl

            print("⚠️  fcntl импортируется на Windows (неожиданно)")
        except ImportError:
            print("✅ fcntl не импортируется на Windows (ожидаемо)")
    else:
        try:
            import fcntl

            print("✅ fcntl доступен на Unix-системе")
        except ImportError:
            print("❌ fcntl недоступен на Unix-системе")

    # Проверка изоляции процессов
    from profiling import ProfilingConfig

    config = ProfilingConfig(isolate_process=True)

    if sys.platform == "win32":
        if not config.isolate_process:
            print("✅ Изоляция автоматически отключена на Windows")
        else:
            print("⚠️  Изоляция включена на Windows (может не работать)")
    else:
        if config.isolate_process:
            print("✅ Изоляция процессов доступна")
        else:
            print("⚠️  Изоляция отключена на Unix-системе")

    return True


def main():
    """Основная функция проверки."""
    print_header()

    # Выполняем все проверки
    checks = [
        ("Импорты", check_imports),
        ("Чекпоинты", check_checkpoint_functionality),
        ("Профилирование", check_profiling),
        ("Прогресс", check_progress_tracking),
        ("Платформа", check_platform_specific),
    ]

    results = []
    for name, check_func in checks:
        try:
            success = check_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n❌ Критическая ошибка в {name}: {e}")
            import traceback

            traceback.print_exc()
            results.append((name, False))

    # Итоги
    print("\n" + "=" * 80)
    print("РЕЗУЛЬТАТЫ ФИНАЛЬНОЙ ПРОВЕРКИ")
    print("=" * 80)

    all_passed = True
    for name, success in results:
        status = "✅" if success else "❌"
        print(f"{status} {name}")
        if not success:
            all_passed = False

    print("\n" + "=" * 80)

    if all_passed:
        print("\n🎉 ФАЗА 3 ПОЛНОСТЬЮ ГОТОВА К ИСПОЛЬЗОВАНИЮ!")
        print("\nВсе компоненты работают корректно на вашей платформе:")
        print(f"  ✅ {platform.system()} {platform.release()}")
        print(f"  ✅ Python {sys.version.split()[0]}")
        print(f"  ✅ Все модули импортируются")
        print(f"  ✅ Чекпоинты сохраняются и загружаются")
        print(f"  ✅ Профилирование работает")
        print(f"  ✅ Прогресс отслеживается")

        print("\n📋 Рекомендуемые следующие шаги:")
        print("1. Запустите полную демонстрацию:")
        print(f"   python scripts{Path('/')}demo_phase3_full.py")
        print("\n2. Попробуйте интерактивную работу с чекпоинтами:")
        print(f"   python scripts{Path('/')}demo_checkpoint_progress.py")
        print("\n3. Переходите к Фазе 4 - реализации операций!")
    else:
        print("\n❌ Обнаружены проблемы. Рекомендации:")
        print("1. Проверьте установку зависимостей:")
        print("   pip install -r requirements.txt")
        print("\n2. Убедитесь, что вы в корневой директории проекта")
        print("\n3. Проверьте логи для деталей ошибок")

    return all_passed


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nПроверка прервана пользователем")
        sys.exit(1)
