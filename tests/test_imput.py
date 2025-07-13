import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from dmdslab.data import DataSplit, ModelData
from dmdslab.imput import DataImputer


@pytest.fixture
def sample_data_split():
    """Фикстура с тестовыми данными"""
    train_data = ModelData(
        features=pd.DataFrame({
            'num1': [1, 2, np.nan, 4],
            'num2': [np.nan, 20, 30, 40],
            'cat1': ['a', 'b', np.nan, 'a'],
            'cat2': ['x', np.nan, 'z', 'x']
        }),
        target=pd.Series([0, 1, 0, 1])
    )

    test_data = ModelData(
        features=pd.DataFrame({
            'num1': [5, np.nan],
            'num2': [np.nan, 60],
            'cat1': [np.nan, 'c'],
            'cat2': ['y', np.nan]
        }),
        target=pd.Series([1, 0])
    )

    return DataSplit(
        train=train_data,
        tuning=test_data,
        test=test_data.copy()
    )


def test_basic_imputation(sample_data_split):
    """Тест базового заполнения пропусков"""
    imputer = DataImputer(
        num_strategy='mean',
        cat_strategy='most_frequent'
    )

    imputed_data = imputer.fit_transform(sample_data_split)

    # Проверка числовых признаков
    assert not imputed_data.train.features['num1'].isna().any()
    assert imputed_data.train.features['num1'].mean() == pytest.approx(2.333, 0.1)

    # Проверка категориальных признаков
    assert imputed_data.train.features['cat1'].tolist() == ['a', 'b', 'a', 'a']
    assert imputed_data.tuning.features['cat2'].tolist() == ['y', 'x']


def test_iterative_imputation(sample_data_split):
    """Тест итеративного заполнения"""
    imputer = DataImputer(
        num_strategy='iterative',
        cat_strategy='constant',
        cat_constant='MISSING'
    )

    imputed_data = imputer.fit_transform(sample_data_split)

    # Проверка отсутствия пропусков
    assert not imputed_data.train.features.isna().any().any()
    assert not imputed_data.tuning.features.isna().any().any()

    # Проверка константного заполнения
    assert 'MISSING' in imputed_data.tuning.features['cat1'].values


def test_column_order_preservation(sample_data_split):
    """Тест сохранения порядка и имен колонок"""
    imputer = DataImputer()
    imputed_data = imputer.fit_transform(sample_data_split)

    original_columns = sample_data_split.train.features.columns.tolist()
    assert imputed_data.train.features.columns.tolist() == original_columns
    assert imputed_data.tuning.features.columns.tolist() == original_columns


def test_numeric_only_imputation():
    """Тест обработки только числовых данных"""
    data = DataSplit(
        train=ModelData(
            pd.DataFrame({'num': [1, np.nan, 3]}),
            pd.Series([0, 1, 0])
        ),
        tuning=ModelData(
            pd.DataFrame({'num': [np.nan, 5]}),
            pd.Series([1, 0])
        ),
        test=ModelData(
            pd.DataFrame({'num': [6, np.nan]}),
            pd.Series([0, 1])
        )
    )

    imputer = DataImputer(num_strategy='median')
    imputed_data = imputer.fit_transform(data)

    assert imputed_data.train.features['num'].tolist() == [1, 2, 3]
    assert imputed_data.tuning.features['num'].tolist() == [2, 5]


def test_categorical_only_imputation():
    """Тест обработки только категориальных данных"""
    data = DataSplit(
        train=ModelData(
            pd.DataFrame({'cat': ['a', np.nan, 'a', np.nan]}),
            pd.Series(range(4))
        ),
        tuning=ModelData(
            pd.DataFrame({'cat': [np.nan, 'b']}),
            pd.Series([1, 0])
        ),
        test=ModelData(
            pd.DataFrame({'cat': ['c', np.nan]}),
            pd.Series([0, 1])
        )
    )

    imputer = DataImputer(
        cat_strategy='constant',
        cat_constant='UNKNOWN'
    )
    imputed_data = imputer.fit_transform(data)

    assert imputed_data.train.features['cat'].tolist() == ['a', 'UNKNOWN', 'a', 'UNKNOWN']
    assert imputed_data.tuning.features['cat'].tolist() == ['UNKNOWN', 'b']


def test_custom_estimator(sample_data_split):
    """Тест использования кастомного estimator'а"""
    imputer = DataImputer(
        num_strategy='iterative',
        num_estimator=HistGradientBoostingRegressor(),
        cat_strategy='most_frequent'
    )

    imputed_data = imputer.fit_transform(sample_data_split)

    # Проверка что импутация выполнена
    assert not imputed_data.train.features.isna().any().any()
    assert isinstance(imputer.num_imputer_.estimator, HistGradientBoostingRegressor)


def test_missing_column_handling(sample_data_split):
    """Тест обработки данных с новыми колонками"""
    # Добавляем новую колонку в тестовые данные
    sample_data_split.tuning.features['new_num'] = [10, np.nan]
    sample_data_split.tuning.features['new_cat'] = [np.nan, 'new']

    imputer = DataImputer()
    imputed_data = imputer.fit_transform(sample_data_split)

    # Новые колонки не должны присутствовать в результате
    assert 'new_num' not in imputed_data.tuning.features.columns
    assert 'new_cat' not in imputed_data.tuning.features.columns


def test_error_handling():
    """Тест обработки некорректных входных данных"""
    # Неправильная стратегия
    with pytest.raises(ValueError):
        DataImputer(num_strategy='invalid_strategy')

    # Неправильный estimator
    with pytest.raises(TypeError):
        DataImputer(num_strategy='iterative', num_estimator="not_an_estimator")

    # Некорректный DataSplit
    with pytest.raises(AttributeError):
        imputer = DataImputer()
        imputer.transform("invalid_data")


def test_fit_transform_equivalence(sample_data_split):
    """Тест эквивалентности fit_transform и последовательных fit+transform"""
    imputer1 = DataImputer()
    result1 = imputer1.fit_transform(sample_data_split)

    imputer2 = DataImputer()
    imputer2.fit(sample_data_split)
    result2 = imputer2.transform(sample_data_split)

    pd.testing.assert_frame_equal(result1.train.features, result2.train.features)
    pd.testing.assert_frame_equal(result1.tuning.features, result2.tuning.features)


def test_missing_values_in_target(sample_data_split):
    """Тест обработки пропусков в target"""
    # Добавляем пропуски в target
    sample_data_split.train.target.iloc[1] = np.nan

    imputer = DataImputer()
    imputed_data = imputer.fit_transform(sample_data_split)

    # Target не должен быть изменен
    assert imputed_data.train.target.isna().sum() == 1
