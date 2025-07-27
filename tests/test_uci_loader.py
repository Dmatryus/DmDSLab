"""
Тесты для UCI Dataset Loader.

Покрывают основную функциональность загрузки датасетов из UCI ML Repository.
"""

import json
import pickle
import shutil
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import logging

import numpy as np
import pandas as pd
import pytest

from dmdslab.datasets.uci import (
    UCIDatasetManager,
    CacheManager,
    DatasetInfo,
    MetadataExtractor,
    TaskType,
    Domain,
    CacheStatus,
    UCIDatasetError,
    CacheError,
    DatasetNotFoundError,
    ValidationError,
)
from dmdslab.datasets import ModelData


class TestCacheManager:
    """Тесты для класса CacheManager."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Создает временную директорию для кеша."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        # Очистка после теста
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def cache_manager(self, temp_cache_dir):
        """Создает экземпляр CacheManager."""
        return CacheManager(cache_dir=temp_cache_dir)

    def test_init_creates_directory(self, temp_cache_dir):
        """Тест создания директории кеша при инициализации."""
        # Удаляем директорию
        shutil.rmtree(temp_cache_dir)
        assert not temp_cache_dir.exists()

        # Создаем менеджер
        cache_manager = CacheManager(cache_dir=temp_cache_dir)

        # Директория должна быть создана
        assert temp_cache_dir.exists()
        assert temp_cache_dir.is_dir()

    def test_save_and_load_dataset(self, cache_manager):
        """Тест сохранения и загрузки датасета."""
        # Подготовка данных
        dataset_id = 53
        test_data = {
            "features": np.random.randn(100, 4),
            "target": np.random.randint(0, 3, 100),
            "feature_names": ["f1", "f2", "f3", "f4"],
            "metadata": {"name": "Test Dataset"},
        }

        # Сохранение
        cache_manager.save_dataset(dataset_id, test_data)

        # Проверка существования
        assert cache_manager.exists(dataset_id)

        # Загрузка
        loaded_data, metadata = cache_manager.load_dataset(dataset_id)

        # Проверка данных
        np.testing.assert_array_equal(loaded_data["features"], test_data["features"])
        np.testing.assert_array_equal(loaded_data["target"], test_data["target"])
        assert loaded_data["feature_names"] == test_data["feature_names"]
        assert loaded_data["metadata"] == test_data["metadata"]

    def test_invalidate_dataset(self, cache_manager):
        """Тест удаления датасета из кеша."""
        dataset_id = 17
        test_data = {"test": "data"}

        # Сохраняем
        cache_manager.save_dataset(dataset_id, test_data)
        assert cache_manager.exists(dataset_id)

        # Удаляем
        cache_manager.invalidate(dataset_id)

        # Проверяем
        assert not cache_manager.exists(dataset_id)

    def test_clear_all(self, cache_manager):
        """Тест полной очистки кеша."""
        # Сохраняем несколько датасетов
        for i in [1, 2, 3]:
            cache_manager.save_dataset(i, {"data": f"dataset_{i}"})

        # Проверяем, что все есть
        assert len(cache_manager.get_cached_datasets()) == 3

        # Очищаем
        cache_manager.clear_all()

        # Проверяем, что пусто
        assert len(cache_manager.get_cached_datasets()) == 0

    def test_cache_size_calculation(self, cache_manager):
        """Тест расчета размера кеша."""
        # Создаем данные известного размера
        large_data = {
            "array": np.zeros((1000, 100)),  # ~800KB
            "metadata": {"test": "data"},
        }

        cache_manager.save_dataset(1, large_data)

        # Проверяем размер
        size = cache_manager.calculate_cache_size()
        assert size > 0
        assert size < 10 * 1024 * 1024  # Меньше 10MB

    def test_cache_statistics(self, cache_manager):
        """Тест получения статистики кеша."""
        # Добавляем датасеты
        cache_manager.save_dataset(1, {"size": "small"})
        cache_manager.save_dataset(2, {"size": "medium", "data": np.zeros(1000)})
        cache_manager.save_dataset(3, {"size": "large", "data": np.zeros(10000)})

        # Получаем статистику
        stats = cache_manager.get_statistics()

        assert stats["total_datasets"] == 3
        assert stats["total_size_bytes"] > 0
        assert "largest_dataset" in stats
        assert "smallest_dataset" in stats
        assert "oldest_dataset" in stats
        assert "newest_dataset" in stats

    def test_cache_validation(self, cache_manager):
        """Тест валидации кеша."""
        # Сохраняем датасет
        cache_manager.save_dataset(1, {"test": "data"})

        # Валидация должна пройти
        results = cache_manager.validate_cache()
        assert len(results["valid"]) == 1
        assert len(results["corrupted"]) == 0
        assert len(results["missing"]) == 0

        # Портим файл
        cache_info = cache_manager.index["1"]
        cache_file = cache_manager.cache_dir / cache_info["filename"]
        cache_file.write_text("corrupted data")

        # Валидация должна обнаружить проблему
        results = cache_manager.validate_cache()
        assert len(results["corrupted"]) > 0


class TestMetadataExtractor:
    """Тесты для класса MetadataExtractor."""

    @pytest.fixture
    def extractor(self):
        """Создает экземпляр MetadataExtractor."""
        return MetadataExtractor()

    def test_detect_categorical_pandas(self, extractor):
        """Тест определения категориальных признаков в pandas DataFrame."""
        # Создаем тестовые данные
        df = pd.DataFrame(
            {
                "numeric": np.random.randn(100),
                "categorical_int": np.random.randint(0, 5, 100),
                "categorical_str": np.random.choice(["A", "B", "C"], 100),
                "binary": np.random.randint(0, 2, 100),
                "continuous": np.linspace(0, 100, 100),
            }
        )

        # Определяем категориальные
        cat_indices = extractor.detect_categorical(df)

        # Проверяем результаты
        assert 1 in cat_indices  # categorical_int
        assert 2 in cat_indices  # categorical_str
        assert 3 in cat_indices  # binary
        assert 0 not in cat_indices  # numeric
        assert 4 not in cat_indices  # continuous

    def test_detect_categorical_numpy(self, extractor):
        """Тест определения категориальных признаков в numpy array."""
        # Создаем тестовые данные
        features = np.column_stack(
            [
                np.random.randn(100),  # continuous
                np.random.randint(0, 3, 100),  # categorical
                np.random.randint(0, 2, 100),  # binary
                np.arange(100),  # continuous
            ]
        )

        # Определяем категориальные
        cat_indices = extractor.detect_categorical(features)

        assert 1 in cat_indices  # categorical
        assert 2 in cat_indices  # binary
        assert 0 not in cat_indices  # continuous
        assert 3 not in cat_indices  # continuous

    def test_determine_task_type(self, extractor):
        """Тест определения типа задачи."""
        features = np.random.randn(100, 5)

        # Классификация - мало уникальных значений
        target_class = np.random.randint(0, 3, 100)
        task_type = extractor.determine_task_type(features, target_class)
        assert task_type == TaskType.CLASSIFICATION

        # Регрессия - непрерывные значения
        target_reg = np.random.randn(100)
        task_type = extractor.determine_task_type(features, target_reg)
        assert task_type == TaskType.REGRESSION

        # Кластеризация - нет целевой переменной
        task_type = extractor.determine_task_type(features, None)
        assert task_type == TaskType.CLUSTERING

    def test_calculate_statistics(self, extractor):
        """Тест расчета статистики."""
        # Pandas данные
        df = pd.DataFrame({"A": [1, 2, 3, np.nan, 5], "B": ["a", "b", "a", "b", "c"]})
        target = pd.Series([0, 1, 0, 1, 0])

        stats = extractor.calculate_statistics(df, target)

        assert "features" in stats
        assert stats["features"]["shape"] == (5, 2)
        assert "missing_values" in stats["features"]
        assert stats["features"]["missing_values"]["A"] == 1

        assert "target" in stats
        assert stats["target"]["unique_values"] == 2


class TestDatasetInfo:
    """Тесты для класса DatasetInfo."""

    def test_from_uci_data(self):
        """Тест создания DatasetInfo из данных UCI."""
        # Мокаем данные UCI
        mock_uci_data = Mock()
        mock_uci_data.metadata = {
            "name": "Iris",
            "abstract": "Famous iris dataset",
            "task": "Classification",
            "area": "Life",
            "has_missing_values": False,
            "num_instances": 150,
            "num_features": 4,
        }

        features = np.random.randn(150, 4)
        target = np.random.randint(0, 3, 150)

        # Создаем DatasetInfo
        info = DatasetInfo.from_uci_data(
            dataset_id=53, uci_data=mock_uci_data, features=features, target=target
        )

        assert info.dataset_id == 53
        assert info.name == "Iris"
        assert info.description == "Famous iris dataset"
        assert info.task_type == TaskType.CLASSIFICATION
        assert info.domain == Domain.LIFE
        assert info.n_instances == 150
        assert info.n_features == 4
        assert not info.has_missing_values

    def test_serialization(self):
        """Тест сериализации/десериализации DatasetInfo."""
        # Создаем объект
        info = DatasetInfo(
            dataset_id=1,
            name="Test",
            task_type=TaskType.REGRESSION,
            domain=Domain.COMPUTER,
            n_instances=100,
            n_features=10,
        )

        # Сериализуем
        data_dict = info.to_dict()
        assert data_dict["task_type"] == "regression"
        assert data_dict["domain"] == "computer"

        # Десериализуем
        info_restored = DatasetInfo.from_dict(data_dict)
        assert info_restored.dataset_id == info.dataset_id
        assert info_restored.task_type == info.task_type
        assert info_restored.domain == info.domain


class TestUCIDatasetManager:
    """Тесты для основного класса UCIDatasetManager."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Создает временную директорию для кеша."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        # Очистка после теста
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def manager(self, temp_cache_dir):
        """Создает экземпляр UCIDatasetManager."""
        return UCIDatasetManager(
            cache_dir=temp_cache_dir,
            log_level="WARNING",  # Меньше логов в тестах
            show_progress=False,  # Отключаем progress bar
        )

    @pytest.fixture
    def mock_uci_data(self):
        """Создает мок данных из ucimlrepo."""
        mock_data = Mock()
        mock_data.data = {
            "features": pd.DataFrame(
                np.random.randn(150, 4),
                columns=["sepal_length", "sepal_width", "petal_length", "petal_width"],
            ),
            "targets": pd.DataFrame(np.random.randint(0, 3, 150), columns=["species"]),
        }
        mock_data.metadata = {
            "name": "Iris",
            "abstract": "Iris dataset for testing",
            "task": "Classification",
            "area": "Life",
            "has_missing_values": False,
            "uci_id": 53,
            "num_instances": 150,
            "num_features": 4,
        }
        return mock_data

    @patch("dmdslab.datasets.uci.uci_manager.ucimlrepo.fetch.fetch_ucirepo")
    def test_load_dataset_basic(self, mock_fetch, manager, mock_uci_data):
        """Тест базовой загрузки датасета."""
        # Настраиваем мок
        mock_fetch.return_value = mock_uci_data

        # Загружаем датасет
        result = manager.load_dataset(53)

        # Проверяем результат
        assert isinstance(result, ModelData)
        assert result.n_samples == 150
        assert result.n_features == 4
        assert result.feature_names == [
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width",
        ]
        assert result.has_target

        # Проверяем вызов
        mock_fetch.assert_called_once_with(id=53)

    @patch("dmdslab.datasets.uci.uci_manager.ucimlrepo.fetch.fetch_ucirepo")
    def test_load_dataset_with_cache(self, mock_fetch, manager, mock_uci_data):
        """Тест загрузки с кешированием."""
        mock_fetch.return_value = mock_uci_data

        # Первая загрузка - из сети
        result1 = manager.load_dataset(53)
        assert mock_fetch.call_count == 1

        # Вторая загрузка - из кеша
        result2 = manager.load_dataset(53)
        assert mock_fetch.call_count == 1  # Не должен вызываться снова

        # Данные должны быть идентичны
        assert result1.n_samples == result2.n_samples
        assert result1.n_features == result2.n_features
        np.testing.assert_array_equal(result1.features, result2.features)

    @patch("dmdslab.datasets.uci.uci_manager.ucimlrepo.fetch.fetch_ucirepo")
    def test_force_reload(self, mock_fetch, manager, mock_uci_data):
        """Тест принудительной перезагрузки."""
        mock_fetch.return_value = mock_uci_data

        # Первая загрузка
        manager.load_dataset(53)
        assert mock_fetch.call_count == 1

        # Принудительная перезагрузка
        manager.load_dataset(53, force_reload=True)
        assert mock_fetch.call_count == 2

    @patch("dmdslab.datasets.uci.uci_manager.ucimlrepo.fetch.fetch_ucirepo")
    def test_load_multiple_datasets(self, mock_fetch, manager, mock_uci_data):
        """Тест загрузки нескольких датасетов."""
        mock_fetch.return_value = mock_uci_data

        # Загружаем несколько
        results = manager.load_datasets([53, 17, 19])

        # Проверяем результаты
        assert len(results) == 3
        assert all(isinstance(r, ModelData) for r in results)
        assert mock_fetch.call_count == 3

    @patch("dmdslab.datasets.uci.uci_manager.ucimlrepo.fetch.fetch_ucirepo")
    def test_error_handling(self, mock_fetch, manager):
        """Тест обработки ошибок."""
        # Настраиваем ошибку
        mock_fetch.side_effect = ValueError("Dataset not found")

        # С raise_on_missing=True
        with pytest.raises(DatasetNotFoundError):
            manager.load_dataset(999)

        # С raise_on_missing=False
        manager_no_raise = UCIDatasetManager(
            cache_dir=manager.cache_dir, raise_on_missing=False, show_progress=False
        )

        # Должен вернуть None или обработать иначе
        # (зависит от реализации)

    def test_clear_cache(self, manager):
        """Тест очистки кеша."""
        # Создаем фейковые данные в кеше
        cache_manager = CacheManager(manager.cache_dir)
        cache_manager.save_dataset(1, {"test": "data1"})
        cache_manager.save_dataset(2, {"test": "data2"})

        manager._load_cache_index()

        # Очищаем конкретный датасет
        success = manager.clear_cache(1)
        assert success
        assert not cache_manager.exists(1)
        assert cache_manager.exists(2)

        # Очищаем весь кеш
        success = manager.clear_cache()
        assert success
        assert not cache_manager.exists(2)

    def test_get_cache_info(self, manager):
        """Тест получения информации о кеше."""
        # Добавляем данные в кеш
        cache_manager = CacheManager(manager.cache_dir)
        cache_manager.save_dataset(1, {"test": "small"})
        cache_manager.save_dataset(2, {"test": "large", "data": np.zeros(10000)})

        manager._load_cache_index()

        # Получаем информацию
        info = manager.get_cache_info()

        assert info["cache_enabled"] == True
        assert info["total_datasets"] == 2
        assert "total_size" in info
        assert "datasets" in info
        assert len(info["datasets"]) == 2

    def test_no_cache_mode(self, temp_cache_dir):
        """Тест работы без кеширования."""
        manager = UCIDatasetManager(use_cache=False, show_progress=False)

        # Проверяем, что кеш отключен
        info = manager.get_cache_info()
        assert info["cache_enabled"] == False
        assert "message" in info

    @patch("dmdslab.datasets.uci.uci_manager.ucimlrepo.fetch.fetch_ucirepo")
    def test_logging(self, mock_fetch, mock_uci_data, temp_cache_dir, caplog):
        """Тест логирования."""
        mock_fetch.return_value = mock_uci_data

        # Создаем менеджер с DEBUG уровнем
        manager = UCIDatasetManager(
            cache_dir=temp_cache_dir, log_level="DEBUG", show_progress=False
        )

        with caplog.at_level(logging.DEBUG):
            manager.load_dataset(53)

        # Проверяем, что есть логи
        assert len(caplog.records) > 0
        assert any(
            "Запрос на загрузку датасета 53" in r.message for r in caplog.records
        )

    @patch("dmdslab.datasets.uci.uci_manager.ucimlrepo.fetch.fetch_ucirepo")
    def test_categorical_detection(self, mock_fetch, manager):
        """Тест автоматического определения категориальных признаков."""
        # Создаем данные с явно категориальными признаками
        mock_data = Mock()
        mock_data.data = {
            "features": pd.DataFrame(
                {
                    "continuous": np.random.randn(100),
                    "categorical": np.random.randint(0, 5, 100),
                    "binary": np.random.randint(0, 2, 100),
                    "text": np.random.choice(["A", "B", "C"], 100),
                }
            ),
            "targets": pd.Series(np.random.randint(0, 2, 100)),
        }
        mock_data.metadata = {"name": "Test", "task": "Classification"}

        mock_fetch.return_value = mock_data

        # Загружаем
        result = manager.load_dataset(1)

        # Проверяем метаданные о категориальных признаках
        # (зависит от реализации, где хранится эта информация)


class TestUtilityFunctions:
    """Тесты для вспомогательных функций."""

    def test_format_cache_size(self):
        """Тест форматирования размера кеша."""
        from dmdslab.datasets.uci.uci_utils import format_cache_size

        assert format_cache_size(100) == "100.00 Б"
        assert format_cache_size(1024) == "1.00 КБ"
        assert format_cache_size(1024 * 1024) == "1.00 МБ"
        assert format_cache_size(1024 * 1024 * 1024) == "1.00 ГБ"

    def test_validate_dataset_id(self):
        """Тест валидации ID датасета."""
        from dmdslab.datasets.uci.uci_utils import validate_dataset_id

        # Валидные ID
        assert validate_dataset_id(53) == 53
        assert validate_dataset_id("53") == 53
        assert validate_dataset_id("iris") == "iris"

        # Невалидные ID
        with pytest.raises(ValueError):
            validate_dataset_id(0)

        with pytest.raises(ValueError):
            validate_dataset_id(-1)

        with pytest.raises(ValueError):
            validate_dataset_id("")

        with pytest.raises(ValueError):
            validate_dataset_id(None)

    @patch("dmdslab.datasets.uci.uci_utils.tqdm")
    def test_progress_bar_creation(self, mock_tqdm):
        """Тест создания progress bar."""
        from dmdslab.datasets.uci.uci_utils import create_progress_bar

        pbar = create_progress_bar(total=100, desc="Test", unit="items")

        mock_tqdm.assert_called_once()
        call_kwargs = mock_tqdm.call_args[1]
        assert call_kwargs["total"] == 100
        assert call_kwargs["desc"] == "Test"
        assert call_kwargs["unit"] == "items"


class TestIntegration:
    """Интеграционные тесты."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Создает временную директорию для кеша."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        # Очистка после теста
        shutil.rmtree(temp_dir)

    def test_real_dataset_loading(self):
        """Тест загрузки реального датасета (требует интернет)."""
        manager = UCIDatasetManager(show_progress=True)

        # Загружаем Iris (маленький датасет)
        iris = manager.load_dataset(53, force_reload=True)

        assert iris.n_samples == 150
        assert iris.n_features == 4
        assert iris.feature_names == [
            "sepal length",
            "sepal width",
            "petal length",
            "petal width",
        ]

    def test_full_workflow(self, temp_cache_dir):
        """Тест полного workflow работы с датасетами."""
        with patch(
            "dmdslab.datasets.uci.uci_manager.ucimlrepo.fetch.fetch_ucirepo"
        ) as mock_fetch:
            # Настраиваем моки для разных датасетов
            def side_effect(id):
                mock = Mock()
                if id == 1:
                    features = np.random.randn(100, 5)
                    name = "Dataset1"
                elif id == 2:
                    features = np.random.randn(200, 10)
                    name = "Dataset2"
                else:
                    raise ValueError("Dataset not found")

                mock.data = {
                    "features": pd.DataFrame(features),
                    "targets": pd.Series(np.random.randint(0, 2, len(features))),
                }
                mock.metadata = {"name": name, "task": "Classification"}
                return mock

            mock_fetch.side_effect = side_effect

            # Создаем менеджер
            manager = UCIDatasetManager(cache_dir=temp_cache_dir, show_progress=False)

            # Загружаем несколько датасетов
            datasets = manager.load_datasets([1, 2])

            # Проверяем
            assert len(datasets) == 2
            assert datasets[0].n_samples == 100
            assert datasets[1].n_samples == 200

            # Проверяем кеш
            info = manager.get_cache_info()
            assert info["total_datasets"] == 2

            # Очищаем один
            manager.clear_cache(1)
            info = manager.get_cache_info()
            assert info["total_datasets"] == 1

            # Перезагружаем с force_reload
            reloaded = manager.load_dataset(2, force_reload=True)
            assert reloaded.n_samples == 200


if __name__ == "__main__":
    pytest.main([__file__])
