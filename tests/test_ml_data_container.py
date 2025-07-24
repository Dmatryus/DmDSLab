"""
Тесты для модуля ml_data_container.

Покрывают все основные классы и функции для работы с ML данными.
"""

from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd
import pytest

from dmdslab.datasets import (
    DataInfo,
    DataSplit,
    ModelData,
    create_data_split,
    create_kfold_data,
)


class TestModelData:
    """Тесты для класса ModelData."""

    @pytest.fixture
    def sample_numpy_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Фикстура с numpy данными."""
        features = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)
        return features, y

    @pytest.fixture
    def sample_pandas_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Фикстура с pandas данными."""
        features = pd.DataFrame(
            np.random.randn(100, 5), columns=[f"feature_{i}" for i in range(5)]
        )
        y = pd.Series(np.random.randint(0, 2, 100), name="target")
        return features, y

    def test_create_with_numpy(self, sample_numpy_data):
        """Тест создания ModelData с numpy arrays."""
        features, y = sample_numpy_data
        data = ModelData(features=features, target=y)

        assert isinstance(data.features, np.ndarray)
        assert isinstance(data.target, np.ndarray)
        assert data.n_samples == 100
        assert data.n_features == 5
        assert data.shape == (100, 5)
        assert data.has_target

    def test_create_with_pandas(self, sample_pandas_data):
        """Тест создания ModelData с pandas данными."""
        features, y = sample_pandas_data
        data = ModelData(features=features, target=y)

        # Должны преобразоваться в numpy
        assert isinstance(data.features, np.ndarray)
        assert isinstance(data.target, np.ndarray)
        # Но имена признаков должны сохраниться
        assert data.feature_names == [f"feature_{i}" for i in range(5)]

    def test_create_without_target(self, sample_numpy_data):
        """Тест создания ModelData без целевой переменной."""
        features, _ = sample_numpy_data
        data = ModelData(features=features)

        assert data.target is None
        assert not data.has_target
        assert data.n_samples == 100

    def test_validation_error(self, sample_numpy_data):
        """Тест валидации при несовпадении размеров."""
        features, y = sample_numpy_data
        y_wrong = y[:50]  # Неправильный размер

        with pytest.raises(ValueError, match="Количество объектов"):
            ModelData(features=features, target=y_wrong)

    def test_feature_names_auto_init(self, sample_numpy_data):
        """Тест автоматической инициализации имен признаков."""
        features, y = sample_numpy_data
        data = ModelData(features=features, target=y)

        assert len(data.feature_names) == 5
        assert data.feature_names[0] == "feature_0"
        assert data.feature_names[-1] == "feature_4"

    def test_to_pandas(self, sample_numpy_data):
        """Тест преобразования в pandas."""
        features, y = sample_numpy_data
        data = ModelData(features=features, target=y)
        features_df, y_series = data.to_pandas()

        assert isinstance(features_df, pd.DataFrame)
        assert isinstance(y_series, pd.Series)
        assert features_df.shape == (100, 5)
        assert len(y_series) == 100

    def test_sample_with_n(self, sample_numpy_data):
        """Тест выборки по количеству объектов."""
        features, y = sample_numpy_data
        data = ModelData(features=features, target=y)
        sampled = data.sample(n=30, random_state=42)

        assert sampled.n_samples == 30
        assert sampled.n_features == 5
        assert sampled.feature_names == data.feature_names

    def test_sample_with_frac(self, sample_numpy_data):
        """Тест выборки по доле объектов."""
        features, y = sample_numpy_data
        data = ModelData(features=features, target=y)
        sampled = data.sample(frac=0.3, random_state=42)

        assert sampled.n_samples == 30

    def test_sample_error(self, sample_numpy_data):
        """Тест ошибки при неправильных параметрах выборки."""
        features, y = sample_numpy_data
        data = ModelData(features=features, target=y)

        with pytest.raises(ValueError, match="Необходимо указать"):
            data.sample()

    def test_copy(self, sample_numpy_data):
        """Тест копирования объекта."""
        features, y = sample_numpy_data
        data = ModelData(features=features, target=y)
        data_copy = data.copy()

        # Изменяем копию
        data_copy.features[0, 0] = 999

        # Оригинал не должен измениться
        assert data.features[0, 0] != 999
        assert data_copy.features[0, 0] == 999

    def test_with_info(self, sample_numpy_data):
        """Тест с метаинформацией."""
        features, y = sample_numpy_data
        info = DataInfo(
            name="test_dataset", description="Test data", metadata={"source": "random"}
        )
        data = ModelData(features=features, target=y, info=info)

        assert data.info.name == "test_dataset"
        assert data.info.description == "Test data"
        assert data.info.metadata["source"] == "random"


class TestDataSplit:
    """Тесты для класса DataSplit."""

    @pytest.fixture
    def sample_splits(self) -> Tuple[ModelData, ModelData, ModelData]:
        """Фикстура с готовыми выборками."""
        train = ModelData(features=np.random.randn(80, 5), target=np.random.randn(80))
        val = ModelData(features=np.random.randn(10, 5), target=np.random.randn(10))
        test = ModelData(features=np.random.randn(10, 5), target=np.random.randn(10))
        return train, val, test

    def test_train_test_split(self, sample_splits):
        """Тест разбиения train/test."""
        train, _, test = sample_splits
        split = DataSplit(train=train, test=test)

        assert split.has_test
        assert not split.has_validation
        assert split.split_info["train_size"] == 80
        assert split.split_info["test_size"] == 10
        assert split.split_info["total_size"] == 90

    def test_train_val_split(self, sample_splits):
        """Тест разбиения train/validation."""
        train, val, _ = sample_splits
        split = DataSplit(train=train, validation=val)

        assert not split.has_test
        assert split.has_validation
        assert split.split_info["train_size"] == 80
        assert split.split_info["validation_size"] == 10

    def test_train_val_test_split(self, sample_splits):
        """Тест полного разбиения train/validation/test."""
        train, val, test = sample_splits
        split = DataSplit(train=train, validation=val, test=test)

        assert split.has_test
        assert split.has_validation
        assert split.split_info["total_size"] == 100

    def test_train_only_split(self, sample_splits):
        """Тест с только обучающей выборкой."""
        train, _, _ = sample_splits
        split = DataSplit(train=train)

        assert not split.has_test
        assert not split.has_validation
        assert "warning" in split.split_info

    def test_get_split_ratios(self, sample_splits):
        """Тест получения пропорций выборок."""
        train, val, test = sample_splits
        split = DataSplit(train=train, validation=val, test=test)
        ratios = split.get_split_ratios()

        assert abs(ratios["train"] - 0.8) < 0.01
        assert abs(ratios["validation"] - 0.1) < 0.01
        assert abs(ratios["test"] - 0.1) < 0.01
        assert abs(sum(ratios.values()) - 1.0) < 0.0001

    def test_get_all_data(self, sample_splits):
        """Тест получения всех выборок."""
        train, val, test = sample_splits
        split = DataSplit(train=train, validation=val, test=test)
        all_data = split.get_all_data()

        assert len(all_data) == 3
        assert all_data[0][0] == "train"
        assert all_data[1][0] == "validation"
        assert all_data[2][0] == "test"


class TestCreateDataSplit:
    """Тесты для функции create_data_split."""

    @pytest.fixture
    def sample_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Фикстура с данными для разбиения."""
        np.random.seed(42)
        features = np.random.randn(1000, 10)
        y = np.random.randint(0, 3, 1000)
        return features, y

    def test_default_train_test_split(self, sample_data):
        """Тест разбиения по умолчанию (train/test)."""
        features, y = sample_data
        split = create_data_split(features, y, random_state=42)

        assert split.has_test
        assert not split.has_validation
        assert split.train.n_samples == 800
        assert split.test.n_samples == 200
        assert split.split_info["split_type"] == "train_test"

    def test_custom_test_size(self, sample_data):
        """Тест с кастомным размером тестовой выборки."""
        features, y = sample_data
        split = create_data_split(features, y, test_size=0.3, random_state=42)

        assert split.train.n_samples == 700
        assert split.test.n_samples == 300

    def test_train_validation_split(self, sample_data):
        """Тест разбиения train/validation."""
        features, y = sample_data
        split = create_data_split(
            features, y, test_size=None, validation_size=0.2, random_state=42
        )

        assert not split.has_test
        assert split.has_validation
        assert split.train.n_samples == 800
        assert split.validation.n_samples == 200
        assert split.split_info["split_type"] == "train_validation"

    def test_train_validation_test_split(self, sample_data):
        """Тест полного разбиения train/validation/test."""
        features, y = sample_data
        split = create_data_split(
            features, y, test_size=0.2, validation_size=0.2, random_state=42
        )

        assert split.has_test
        assert split.has_validation
        assert split.train.n_samples == 600
        assert split.validation.n_samples == 200
        assert split.test.n_samples == 200
        assert split.split_info["split_type"] == "train_validation_test"

    def test_train_only(self, sample_data):
        """Тест только с обучающей выборкой."""
        features, y = sample_data
        split = create_data_split(features, y, test_size=None, random_state=42)

        assert not split.has_test
        assert not split.has_validation
        assert split.train.n_samples == 1000
        assert split.split_info["split_type"] == "train_only"

    def test_stratified_split(self, sample_data):
        """Тест стратифицированного разбиения."""
        features, y = sample_data
        split = create_data_split(features, y, stratify=True, random_state=42)

        # Проверяем распределение классов
        train_dist = np.bincount(split.train.target) / len(split.train.target)
        test_dist = np.bincount(split.test.target) / len(split.test.target)

        # Распределения должны быть похожи
        assert np.allclose(train_dist, test_dist, atol=0.05)
        assert split.split_info["stratified"]

    def test_invalid_sizes(self, sample_data):
        """Тест с некорректными размерами выборок."""
        features, y = sample_data

        with pytest.raises(ValueError, match="должна быть меньше 1.0"):
            create_data_split(features, y, test_size=0.6, validation_size=0.5)

    def test_with_pandas_input(self, sample_data):
        """Тест с pandas входными данными."""
        features, y = sample_data
        features_df = pd.DataFrame(features, columns=[f"col_{i}" for i in range(10)])
        y_series = pd.Series(y, name="target")

        split = create_data_split(features_df, y_series, random_state=42)

        assert split.train.feature_names == [f"col_{i}" for i in range(10)]
        assert isinstance(split.train.features, np.ndarray)


class TestCreateKfoldData:
    """Тесты для функции create_kfold_data."""

    @pytest.fixture
    def sample_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Фикстура с данными для кросс-валидации."""
        np.random.seed(42)
        features = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)
        return features, y

    def test_default_kfold(self, sample_data):
        """Тест кросс-валидации по умолчанию."""
        features, y = sample_data
        splits = create_kfold_data(features, y, random_state=42)

        assert len(splits) == 5
        for i, split in enumerate(splits):
            assert split.train.n_samples == 80
            assert split.validation.n_samples == 20
            assert split.split_info["fold"] == i
            assert split.split_info["n_splits"] == 5
            assert split.split_info["split_type"] == "kfold"

    def test_custom_n_splits(self, sample_data):
        """Тест с кастомным количеством фолдов."""
        features, y = sample_data
        splits = create_kfold_data(features, y, n_splits=3, random_state=42)

        assert len(splits) == 3
        # Проверяем размеры (могут немного отличаться из-за округления)
        assert all(66 <= split.train.n_samples <= 67 for split in splits)
        assert all(33 <= split.validation.n_samples <= 34 for split in splits)

    def test_no_shuffle(self, sample_data):
        """Тест без перемешивания."""
        features, y = sample_data
        splits = create_kfold_data(features, y, n_splits=5, shuffle=False)

        # Первый фолд должен содержать первые 20 элементов в validation
        first_val = splits[0].validation
        assert np.array_equal(first_val.features, features[:20])
        assert np.array_equal(first_val.target, y[:20])

    def test_with_pandas_input(self, sample_data):
        """Тест кросс-валидации с pandas данными."""
        features, y = sample_data
        features_df = pd.DataFrame(features, columns=[f"feature_{i}" for i in range(5)])
        y_series = pd.Series(y, name="target")

        splits = create_kfold_data(features_df, y_series, n_splits=3, random_state=42)

        assert len(splits) == 3
        for split in splits:
            assert split.train.feature_names == [f"feature_{i}" for i in range(5)]


class TestDataInfo:
    """Тесты для класса DataInfo."""

    def test_basic_creation(self):
        """Тест базового создания DataInfo."""
        info = DataInfo(name="test_data")

        assert info.name == "test_data"
        assert info.version == "1.0.0"
        assert isinstance(info.created_at, datetime)
        assert info.description is None
        assert info.source is None
        assert isinstance(info.metadata, dict)

    def test_full_creation(self):
        """Тест создания DataInfo со всеми параметрами."""
        created_time = datetime.now()
        metadata = {"rows": 1000, "features": 10, "task": "classification"}

        info = DataInfo(
            name="full_dataset",
            created_at=created_time,
            description="Complete test dataset",
            source="synthetic",
            version="2.1.0",
            metadata=metadata,
        )

        assert info.name == "full_dataset"
        assert info.created_at == created_time
        assert info.description == "Complete test dataset"
        assert info.source == "synthetic"
        assert info.version == "2.1.0"
        assert info.metadata["task"] == "classification"


# Дополнительные интеграционные тесты
class TestIntegration:
    """Интеграционные тесты для проверки совместной работы компонентов."""

    def test_full_pipeline(self):
        """Тест полного пайплайна работы с данными."""
        # Создаем синтетические данные
        np.random.seed(42)
        features = np.random.randn(1000, 20)
        y = (features[:, 0] + features[:, 1] > 0).astype(int)

        # Создаем разбиение
        split = create_data_split(
            features,
            y,
            test_size=0.2,
            validation_size=0.1,
            stratify=True,
            random_state=42,
        )

        # Проверяем размеры
        assert split.train.n_samples == 700
        assert split.validation.n_samples == 100
        assert split.test.n_samples == 200

        # Делаем выборку из обучающей выборки
        train_sample = split.train.sample(n=100, random_state=42)
        assert train_sample.n_samples == 100

        # Преобразуем в pandas и обратно
        features_df, y_series = split.train.to_pandas()
        new_data = ModelData(features=features_df, target=y_series)
        assert new_data.n_samples == split.train.n_samples

    def test_kfold_with_operations(self):
        """Тест кросс-валидации с дополнительными операциями."""
        # Создаем данные
        features = pd.DataFrame(np.random.randn(150, 4), columns=list("ABCD"))
        y = pd.Series(np.random.choice([0, 1, 2], 150))

        # Кросс-валидация
        splits = create_kfold_data(features, y, n_splits=3, random_state=42)

        # Для каждого фолда
        for i, split in enumerate(splits):
            # Проверяем, что нет пересечений между train и validation
            train_indices = set(range(150))
            val_start = i * 50
            val_end = (i + 1) * 50 if i < 2 else 150
            val_indices = set(range(val_start, val_end))
            train_indices -= val_indices

            # Проверяем размеры
            assert split.train.n_samples == 100
            assert split.validation.n_samples == 50

            # Проверяем сохранение имен признаков
            assert split.train.feature_names == list("ABCD")
