import pytest
import pandas as pd
import numpy as np
from sklearn.exceptions import NotFittedError
from dmdslab.data import DataSplit, ModelData
from dmdslab.scaling import DataScaler


@pytest.fixture
def sample_data():
    train_features = pd.DataFrame(
        {
            "age": [25, 30, 35, 40],
            "income": [50_000, 60_000, 70_000, 80_000],
            "gender": ["M", "F", "M", "F"],
        }
    )
    test_features = pd.DataFrame(
        {"age": [45, 50], "income": [90_000, 100_000], "gender": ["M", "F"]}
    )
    return DataSplit(
        train=ModelData(features=train_features, target=pd.Series([0, 1, 0, 1])),
        test=ModelData(features=test_features, target=pd.Series([0, 1])),
        tuning=None,
    )


def test_initialization():
    scaler = DataScaler(scaler_type="standard", numeric_features=["age", "income"])
    assert scaler.scaler_type == "standard"
    assert scaler.numeric_features == ["age", "income"]
    assert not scaler._fitted


def test_auto_numeric_features_detection(sample_data):
    scaler = DataScaler(scaler_type="standard")
    scaler.fit(sample_data)
    assert set(scaler.numeric_features) == {"age", "income"}


def test_fit_transform_basic(sample_data):
    scaler = DataScaler(scaler_type="standard", numeric_features=["age", "income"])
    transformed = scaler.fit_transform(sample_data)

    # Проверка структуры данных
    assert isinstance(transformed.train.features, pd.DataFrame)
    assert transformed.train.features.shape == (4, 3)
    assert transformed.test.features.shape == (2, 3)

    # Проверка масштабирования
    assert np.allclose(transformed.train.features["scaler__age"].mean(), 0, atol=1e-8)
    assert np.allclose(transformed.train.features["scaler__age"].std(), 1, atol=0.2)

    # Категориальные признаки не изменены
    assert (
        transformed.train.features["remainder__gender"] == ["M", "F", "M", "F"]
    ).all()


def test_unfitted_transform_raises_error(sample_data):
    scaler = DataScaler(scaler_type="standard")
    with pytest.raises(NotFittedError):
        scaler.transform(sample_data)


def test_robust_scaler_with_outliers():
    data = DataSplit(
        train=ModelData(
            features=pd.DataFrame({"value": [1, 2, 3, 4, 100]}),
            target=pd.Series([0] * 5),
        ),
        tuning=None,
        test=None,
    )
    scaler = DataScaler(scaler_type="robust", numeric_features=["value"])
    transformed = scaler.fit_transform(data)

    # RobustScaler должен минимизировать влияние выброса
    assert abs(transformed.train.features["scaler__value"].iloc[-1]) < 50


def test_yeojohnson_transformation(sample_data):
    scaler = DataScaler(scaler_type="yeo-johnson", numeric_features=["income"])
    transformed = scaler.fit_transform(sample_data)

    # Проверка нормализации распределения
    from scipy.stats import skew

    assert np.allclose(
        transformed.train.features["scaler__income"].mean(), 0, atol=1e-8
    )
    assert np.allclose(transformed.train.features["scaler__income"].std(), 1, atol=0.2)


def test_missing_tuning_set():
    data = DataSplit(
        train=ModelData(
            features=pd.DataFrame({"age": [25, 30]}), target=pd.Series([0, 1])
        ),
        test=None,
        tuning=None,
    )
    scaler = DataScaler(scaler_type="standard")
    transformed = scaler.fit_transform(data)
    assert transformed.tuning is None


def test_preservation_of_indexes(sample_data):
    sample_data.train.features.index = ["a", "b", "c", "d"]
    scaler = DataScaler(scaler_type="standard")
    transformed = scaler.fit_transform(sample_data)
    assert transformed.train.features.index.tolist() == ["a", "b", "c", "d"]


def test_error_on_non_numeric_features():
    data = DataSplit(
        train=ModelData(
            features=pd.DataFrame({"text": ["foo", "bar"]}), target=pd.Series([0, 1])
        ),
        test=None,
        tuning=None,
    )
    scaler = DataScaler(scaler_type="standard", numeric_features=["text"])
    with pytest.raises(ValueError):
        scaler.fit(data)
