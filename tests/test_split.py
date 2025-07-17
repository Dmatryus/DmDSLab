import pytest
import numpy as np
import pandas as pd
from dmdslab.data import ModelData, DataSplit
from dmdslab.splitting import split_dataset


@pytest.fixture
def sample_data():
    """Фикстура с тестовыми данными"""
    features = pd.DataFrame(
        {"feature1": np.random.rand(1000), "feature2": np.random.randint(0, 5, 1000)}
    )
    target = pd.Series(np.random.choice(["A", "B"], size=1000, p=[0.7, 0.3]))
    return ModelData(features=features, target=target)


def test_basic_split(sample_data):
    """Тест базового разделения без стратификации"""
    result = split_dataset(sample_data, test_size=0.2, tuning_size=0.1, random_state=42)

    # Проверка типов возвращаемых объектов
    assert isinstance(result, DataSplit)
    assert isinstance(result.train, ModelData)
    assert isinstance(result.tuning, ModelData)
    assert isinstance(result.test, ModelData)

    # Проверка размеров выборок
    total_samples = len(sample_data.features)
    assert len(result.tuning.features) == int(total_samples * 0.1)
    assert len(result.test.features) == int(total_samples * 0.2)


def test_stratified_split(sample_data):
    """Тест стратифицированного разделения"""
    result = split_dataset(
        sample_data, test_size=0.3, tuning_size=0.2, stratify=True, random_state=42
    )

    # Проверка сохранения распределения классов
    for part in [result.train, result.tuning, result.test]:
        target_dist = part.target.value_counts(normalize=True)
        assert np.isclose(target_dist["A"], 0.7, atol=0.05)
        assert np.isclose(target_dist["B"], 0.3, atol=0.05)


def test_no_validation_split(sample_data):
    """Тест без валидационной выборки"""
    result = split_dataset(sample_data, test_size=0.3, tuning_size=None)

    assert result.tuning is None
    assert len(result.train.features) + len(result.test.features) == 1000


def test_invalid_input():
    """Тест обработки неверного ввода"""
    with pytest.raises(ValueError):
        split_dataset("invalid_data", test_size=0.2)


def test_reproducibility(sample_data):
    """Тест воспроизводимости результатов"""
    result1 = split_dataset(sample_data, test_size=0.2, random_state=42)
    result2 = split_dataset(sample_data, test_size=0.2, random_state=42)

    pd.testing.assert_frame_equal(result1.train.features, result2.train.features)
    pd.testing.assert_series_equal(result1.train.target, result2.train.target)


def test_edge_cases(sample_data):
    """Тест пограничных случаев"""
    # Все данные в тестовой выборке
    with pytest.raises(ValueError):
        split_dataset(sample_data, test_size=1.0)

    # Нулевые размеры выборок
    result = split_dataset(sample_data, test_size=0.0, tuning_size=0.0)
    assert result.test is None
    assert result.tuning is None
    assert len(result.train.features) == 1000
