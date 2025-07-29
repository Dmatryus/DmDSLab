"""
Тестируют функции быстрой загрузки и API верхнего уровня с реальными данными.
"""

import json
import logging
import os
import pickle
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

from dmdslab.datasets import ModelData
from dmdslab.datasets.uci import (
    POPULAR_DATASETS,
    CacheManager,
    NetworkError,
    UCIDatasetManager,
    UCILoader,  # Алиас
    clear_cache,
    get_cache_info,
    load_by_name,
    load_dataset,
    load_datasets,
)

# Вспомогательные функции для создания тестовых данных


def create_test_dataset(dataset_id, n_samples=100, n_features=4, seed=42):
    """Создает тестовый датасет с заданными параметрами."""
    np.random.seed(seed)

    features = np.random.randn(n_samples, n_features)
    target = np.random.randint(0, 3, n_samples)
    feature_names = [f"feature_{i}" for i in range(n_features)]

    metadata = {
        "name": f"Test Dataset {dataset_id}",
        "task": "classification",
        "area": "test",
        "num_instances": n_samples,
        "num_features": n_features,
        "has_missing_values": False,
        "dataset_id": dataset_id,
    }

    return {
        "features": features,
        "target": target,
        "feature_names": feature_names,
        "metadata": metadata,
    }


def create_cached_dataset(
    cache_dir: Path, dataset_id: int, test_data: Dict[str, Any]
) -> None:
    """Создание кешированного датасета для тестов."""
    import json

    from dmdslab.datasets.uci.uci_utils import format_cache_size, get_timestamp

    # Создаем имя файла
    timestamp = get_timestamp()
    filename = (
        f"dataset_{dataset_id}_{timestamp.replace(':', '-').replace(' ', '_')}.pkl"
    )
    cache_file = cache_dir / filename

    # Сохраняем данные
    with open(cache_file, "wb") as f:
        pickle.dump(test_data, f, protocol=4)

    # Загружаем существующий индекс или создаем новый
    index_path = cache_dir / "cache_index.json"
    if index_path.exists():
        with open(index_path, encoding="utf-8") as f:
            index = json.load(f)
    else:
        index = {}

    # Обновляем индекс
    file_size = cache_file.stat().st_size
    index[str(dataset_id)] = {
        "filename": filename,
        "dataset_id": dataset_id,
        "cached_at": timestamp,
        "size_bytes": file_size,
        "size_human": format_cache_size(file_size),
        "metadata": test_data.get("metadata", {}),
    }

    # Добавляем метаданные индекса
    index["_version"] = "1.0"
    index["_updated_at"] = timestamp

    # Сохраняем индекс
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)


class TestQuickLoadFunctions:
    """Тесты для функций быстрой загрузки."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Создает временную директорию для кеша."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def populated_cache(self, temp_cache_dir):
        """Создает кеш с тестовыми датасетами."""
        # Создаем тестовые датасеты в кеше
        datasets = {
            53: create_test_dataset(53, n_samples=150, n_features=4),  # Iris
            17: create_test_dataset(17, n_samples=569, n_features=30),  # Breast Cancer
            19: create_test_dataset(19, n_samples=178, n_features=13),  # Wine
        }

        for dataset_id, data in datasets.items():
            create_cached_dataset(temp_cache_dir, dataset_id, data)

        return temp_cache_dir, datasets

    def test_load_dataset_from_cache(self, populated_cache):
        """Тест загрузки датасета из кеша."""
        cache_dir, datasets = populated_cache

        # Загружаем датасет
        result = load_dataset(53, cache_dir=cache_dir)

        # Проверяем результат
        assert isinstance(result, ModelData)
        assert result.n_samples == 150
        assert result.n_features == 4
        assert result.feature_names == [f"feature_{i}" for i in range(4)]

    def test_load_dataset_cache_miss(self, temp_cache_dir):
        """Тест поведения при отсутствии датасета в кеше."""
        manager = UCIDatasetManager(
            cache_dir=temp_cache_dir,
            raise_on_missing=True,  # Выбрасывать исключения
            show_progress=False,
        )

        # НЕ создаем датасет в кеше!
        # Пытаемся загрузить несуществующий датасет
        with pytest.raises(NetworkError):
            manager.load_dataset(99999)  # Должна быть ошибка

    def test_load_datasets_multiple(self, populated_cache):
        """Тест загрузки нескольких датасетов."""
        cache_dir, datasets = populated_cache

        # Загружаем несколько
        results = load_datasets([53, 17, 19], cache_dir=cache_dir, show_progress=False)

        # Проверяем результаты
        assert len(results) == 3
        assert all(isinstance(r, ModelData) for r in results)
        assert results[0].n_samples == 150  # Iris
        assert results[1].n_samples == 569  # Breast Cancer
        assert results[2].n_samples == 178  # Wine

    def test_load_datasets_with_missing(self, populated_cache):
        """Тест загрузки с отсутствующими датасетами."""
        cache_dir, datasets = populated_cache

        # Загружаем с несуществующим ID
        results = load_datasets(
            [53, 999, 17], cache_dir=cache_dir, stop_on_error=False, show_progress=False
        )

        assert len(results) == 3
        assert isinstance(results[0], ModelData)
        assert isinstance(results[1], Exception)  # Ошибка для 999
        assert isinstance(results[2], ModelData)

    def test_load_by_name_with_cache(self, populated_cache):
        """Тест загрузки по имени из кеша."""
        cache_dir, datasets = populated_cache

        # Проверяем, что популярные датасеты определены
        assert "iris" in POPULAR_DATASETS
        assert POPULAR_DATASETS["iris"] == 53

        # Загружаем по имени
        result = load_by_name("iris", cache_dir=cache_dir)

        assert isinstance(result, ModelData)
        assert result.n_samples == 150

    def test_load_by_name_invalid(self):
        """Тест загрузки по неверному имени."""
        with pytest.raises(ValueError, match="не найден в списке популярных"):
            load_by_name("unknown_dataset")

    def test_load_by_name_variations(self, populated_cache):
        """Тест различных вариаций имен."""
        cache_dir, datasets = populated_cache

        # Создаем датасеты для разных вариаций
        breast_cancer_id = POPULAR_DATASETS.get("breast_cancer", 17)
        create_cached_dataset(cache_dir, breast_cancer_id, datasets[17])

        # Различные форматы имен
        result1 = load_by_name("IRIS", cache_dir=cache_dir)
        assert result1.n_samples == 150

        result2 = load_by_name("breast cancer", cache_dir=cache_dir)
        assert isinstance(result2, ModelData)

        result3 = load_by_name("breast-cancer", cache_dir=cache_dir)
        assert isinstance(result3, ModelData)

    def test_clear_cache_function(self, populated_cache):
        """Тест функции очистки кеша."""
        cache_dir, datasets = populated_cache

        # Проверяем, что датасеты есть
        cache_manager = CacheManager(cache_dir)
        assert cache_manager.exists(53)

        # Очищаем конкретный датасет
        success = clear_cache(53, cache_dir=cache_dir)
        assert success

        # Проверяем, что датасет удален
        assert not cache_manager.exists(53)
        # Но другие остались
        assert cache_manager.exists(17)

    def test_clear_cache_all(self, populated_cache):
        """Тест полной очистки кеша."""
        cache_dir, datasets = populated_cache

        # Очищаем весь кеш
        success = clear_cache(cache_dir=cache_dir)
        assert success

        # Проверяем, что кеш пуст
        cache_manager = CacheManager(cache_dir)
        assert len(cache_manager.get_cached_datasets()) == 0

    def test_get_cache_info_function(self, populated_cache):
        """Тест функции получения информации о кеше."""
        cache_dir, datasets = populated_cache

        # Получаем информацию
        info = get_cache_info(cache_dir=cache_dir)

        assert info["cache_enabled"]
        assert info["total_datasets"] == 3
        assert "total_size" in info
        assert len(info["datasets"]) == 3

    def test_no_cache_mode(self):
        """Тест работы без кеширования."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Создаем тестовый датасет
            test_data = create_test_dataset(1)
            create_cached_dataset(Path(temp_dir), 1, test_data)

            # Загружаем БЕЗ кеширования
            manager = UCIDatasetManager(
                cache_dir=temp_dir, use_cache=False, show_progress=False
            )

            # Кеш должен быть отключен
            info = manager.get_cache_info()
            assert not info["cache_enabled"]


class TestPopularDatasets:
    """Тесты для списка популярных датасетов."""

    def test_popular_datasets_structure(self):
        """Тест структуры словаря популярных датасетов."""
        assert isinstance(POPULAR_DATASETS, dict)
        assert len(POPULAR_DATASETS) > 0

        # Проверяем известные датасеты
        assert "iris" in POPULAR_DATASETS
        assert "wine" in POPULAR_DATASETS
        assert "breast_cancer" in POPULAR_DATASETS

        # Проверяем значения
        assert isinstance(POPULAR_DATASETS["iris"], int)
        assert POPULAR_DATASETS["iris"] > 0

    def test_all_popular_names_valid(self):
        """Тест валидности всех имен в списке."""
        for name, dataset_id in POPULAR_DATASETS.items():
            # Имя должно быть в нижнем регистре с подчеркиваниями
            assert name.islower()
            assert " " not in name
            assert "-" not in name

            # ID должен быть положительным числом
            assert isinstance(dataset_id, int)
            assert dataset_id > 0


class TestModuleImports:
    """Тесты импортов из модуля."""

    def test_main_imports(self):
        """Тест основных импортов из dmdslab.datasets.uci."""
        from dmdslab.datasets.uci import (
            CacheManager,
            DatasetInfo,
            Domain,
            TaskType,
            UCIDatasetError,
            UCIDatasetManager,
        )

        # Проверяем, что все импортировалось
        assert UCIDatasetManager is not None
        assert CacheManager is not None
        assert DatasetInfo is not None
        assert TaskType is not None
        assert Domain is not None
        assert UCIDatasetError is not None

    def test_uci_loader_alias(self):
        """Тест алиаса UCILoader."""
        assert UCILoader is UCIDatasetManager

    def test_dataset_imports(self):
        """Тест импортов из основного модуля datasets."""
        # Эти функции должны быть доступны через основной __init__
        from dmdslab.datasets import (
            UCI_POPULAR_DATASETS,
            clear_uci_cache,
            get_uci_cache_info,
            load_uci_by_name,
            load_uci_dataset,
            load_uci_datasets,
        )

        # Проверяем, что импортировалось
        assert load_uci_dataset is not None
        assert load_uci_datasets is not None
        assert load_uci_by_name is not None
        assert clear_uci_cache is not None
        assert get_uci_cache_info is not None
        assert UCI_POPULAR_DATASETS is not None


class TestLoggingIntegration:
    """Тесты интеграции с логированием."""

    def test_log_levels(self, caplog):
        """Тест различных уровней логирования."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # DEBUG уровень
            with caplog.at_level(logging.DEBUG):
                UCIDatasetManager(
                    cache_dir=temp_dir, log_level="DEBUG", show_progress=False
                )
                assert "инициализирован" in caplog.text

            caplog.clear()

            # ERROR уровень - не должно быть DEBUG сообщений
            with caplog.at_level(logging.ERROR):
                UCIDatasetManager(
                    cache_dir=temp_dir, log_level="ERROR", show_progress=False
                )
                assert "инициализирован" not in caplog.text


class TestErrorScenarios:
    """Тесты различных сценариев ошибок."""

    def test_cache_corruption_handling(self):
        """Тест обработки поврежденного кеша."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)

            # Создаем поврежденный файл кеша
            cache_file = cache_dir / "dataset_1_corrupted.pkl"
            cache_file.write_text("this is not pickle data")

            # Создаем индекс, указывающий на поврежденный файл
            index = {"1": {"filename": "dataset_1_corrupted.pkl"}}
            index_file = cache_dir / "cache_index.json"
            index_file.write_text(json.dumps(index))

            # Создаем менеджер
            manager = UCIDatasetManager(cache_dir=cache_dir, show_progress=False)

            # Создаем валидный датасет
            test_data = create_test_dataset(1)
            create_cached_dataset(cache_dir, 2, test_data)

            # Загрузка валидного датасета должна работать
            result = manager.load_dataset(2)
            assert isinstance(result, ModelData)

    def test_empty_cache_directory(self):
        """Тест работы с пустой директорией кеша."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = UCIDatasetManager(cache_dir=temp_dir, show_progress=False)

            # Информация о пустом кеше
            info = manager.get_cache_info()
            assert info["total_datasets"] == 0
            assert info["total_size_bytes"] == 0


class TestIntegrationScenarios:
    """Комплексные интеграционные сценарии."""

    def test_full_workflow(self):
        """Тест полного workflow работы с датасетами."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)

            # 1. Создаем менеджер
            manager = UCIDatasetManager(cache_dir=cache_dir, show_progress=False)

            # 2. Создаем несколько тестовых датасетов
            datasets_info = {
                1: {"n_samples": 100, "n_features": 5},
                2: {"n_samples": 200, "n_features": 10},
                3: {"n_samples": 150, "n_features": 7},
            }

            for dataset_id, params in datasets_info.items():
                test_data = create_test_dataset(dataset_id, **params)
                create_cached_dataset(cache_dir, dataset_id, test_data)

            # 3. Загружаем датасеты
            manager._load_cache_index()
            datasets = manager.load_datasets([1, 2, 3])

            # 4. Проверяем
            assert len(datasets) == 3
            assert all(isinstance(d, ModelData) for d in datasets)
            assert datasets[0].n_samples == 100
            assert datasets[1].n_samples == 200
            assert datasets[2].n_samples == 150

            # 5. Проверяем кеш
            info = manager.get_cache_info()
            assert info["total_datasets"] == 3

            # 6. Очищаем один датасет
            manager.clear_cache(1)
            info = manager.get_cache_info()
            assert info["total_datasets"] == 2

            # 7. Перезагружаем с force_reload
            test_data_new = create_test_dataset(2, n_samples=250, n_features=10)
            create_cached_dataset(cache_dir, 2, test_data_new)

            reloaded = manager.load_dataset(2, force_reload=True)
            assert reloaded.n_samples == 48842  # Новые данные

    def test_concurrent_cache_access(self):
        """Тест одновременного доступа к кешу."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)

            # Создаем тестовые данные
            for i in range(1, 5):
                test_data = create_test_dataset(i)
                create_cached_dataset(cache_dir, i, test_data)

            # Создаем несколько менеджеров
            managers = [
                UCIDatasetManager(cache_dir=cache_dir, show_progress=False)
                for _ in range(3)
            ]

            # Все должны видеть одинаковые данные
            for manager in managers:
                info = manager.get_cache_info()
                assert info["total_datasets"] == 4

                # Загружаем датасет
                data = manager.load_dataset(1)
                assert data.n_samples == 100


class TestRealDatasetLoading:
    """Тесты с реальной загрузкой (требуют интернет)."""

    def test_real_iris_loading(self):
        """Тест загрузки реального датасета Iris."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = UCIDatasetManager(cache_dir=temp_dir, show_progress=True)

            # Загружаем Iris
            iris = manager.load_dataset(53)

            assert iris.n_samples == 150
            assert iris.n_features == 4
            assert iris.feature_names == [
                "sepal length",
                "sepal width",
                "petal length",
                "petal width",
            ]

            # Проверяем кеширование
            info = manager.get_cache_info()
            assert info["total_datasets"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
