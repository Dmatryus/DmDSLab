import pytest
import pandas as pd
import numpy as np
from sklearn.exceptions import NotFittedError
from dmdslab.data import DataSplit, ModelData
from dmdslab.encoding import CategoricalEncoder


@pytest.fixture
def sample_data():
    """Фикстура с тестовыми данными"""
    train = ModelData(
        pd.DataFrame({
            'cat1': ['a', 'b', 'a', np.nan],
            'cat2': ['x', 'y', 'x', 'z'],
            'num': [1.0, 2.5, 3.0, 4.5]
        }),
        pd.Series([0, 1, 0, 1])
    )

    test = ModelData(
        pd.DataFrame({
            'cat1': ['c', np.nan],  # Новая категория 'c'
            'cat2': ['x', 'w'],  # Новая категория 'w'
            'num': [5.0, 6.0]
        }),
        pd.Series([1, 0])
    )

    return DataSplit(train=train, tuning=test, test=None)


def test_basic_fit_transform(sample_data):
    """Тест базового преобразования OneHot"""
    encoder = CategoricalEncoder(method='onehot')
    encoded = encoder.fit_transform(sample_data)

    # Проверка структуры данных
    assert encoded.train.features.shape[1] > 3  # Добавились новые колонки
    assert 'cat1_a' in encoded.train.features.columns
    assert 'num' in encoded.train.features.columns


def test_unknown_categories_handling(sample_data):
    """Тест обработки новых категорий"""
    encoder = CategoricalEncoder(method='ordinal')
    encoded = encoder.fit_transform(sample_data)

    # Проверка обработки 'c' в cat1
    assert encoded.tuning.features['cat1'].iloc[0] == -1  # Для handle_unknown='use_encoded_value'


def test_target_encoding(sample_data):
    """Тест Target Encoding"""
    encoder = CategoricalEncoder(method='target', cols=['cat1'])
    encoded = encoder.fit_transform(sample_data)

    # Проверка средних значений
    train_means = encoded.train.features['cat1']
    assert pytest.approx(train_means.mean(), 0.1) == 0.5

    # Проверка сглаживания
    assert encoded.tuning.features['cat1'].iloc[0] != 0.0


def test_catboost_encoding(sample_data):
    """Тест CatBoost Encoding"""
    encoder = CategoricalEncoder(method='catboost')
    encoded = encoder.fit_transform(sample_data)

    # Проверка что нет NaN
    assert not encoded.train.features.isna().any().any()
    assert encoded.tuning.features['cat1'].dtype == float


def test_binary_encoding(sample_data):
    """Тест Binary Encoding"""
    encoder = CategoricalEncoder(method='binary', cols=['cat2'])
    encoded = encoder.fit_transform(sample_data)

    # Проверка количества колонок
    original_cardinality = 4  # ['x', 'y', 'z', 'w']
    expected_columns = int(np.ceil(np.log2(original_cardinality)))
    assert sum('cat2_' in c for c in encoded.train.features.columns) == expected_columns


def test_error_handling(sample_data):
    """Тест обработки ошибок"""
    # Неправильный метод
    with pytest.raises(ValueError):
        CategoricalEncoder(method='invalid_method')

def test_numeric_columns_untouched(sample_data):
    """Тест неизменности числовых колонок"""
    encoder = CategoricalEncoder(method='ordinal')
    encoded = encoder.fit_transform(sample_data)

    # Проверка числовых данных
    pd.testing.assert_series_equal(
        encoded.train.features['num'],
        sample_data.train.features['num'],
        check_names=False
    )


def test_fit_transform_consistency(sample_data):
    """Тест согласованности fit+transform и fit_transform"""
    encoder1 = CategoricalEncoder(method='target').fit_transform(sample_data)
    encoder2 = CategoricalEncoder(method='target')
    encoder2.fit(sample_data)
    encoded2 = encoder2.transform(sample_data)

    pd.testing.assert_frame_equal(
        encoder1.train.features,
        encoded2.train.features
    )


def test_sparse_matrix_handling(sample_data):
    """Тест обработки разреженных матриц для OneHot"""
    encoder = CategoricalEncoder(method='onehot', handle_unknown='ignore')
    encoded = encoder.fit_transform(sample_data)

    # Проверка что все значения 0 или 1
    ohe_cols = [c for c in encoded.train.features.columns if 'cat' in c]
    assert encoded.train.features[ohe_cols].isin([0, 1]).all().all()
