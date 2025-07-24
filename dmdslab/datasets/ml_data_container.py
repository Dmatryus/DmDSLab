"""
Модуль для работы с данными в машинном обучении.

Предоставляет удобные контейнеры для хранения и манипулирования
данными на различных этапах ML pipeline.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


@dataclass
class DataInfo:
    """Метаинформация о датасете."""

    name: str
    created_at: datetime = field(default_factory=datetime.now)
    description: Optional[str] = None
    source: Optional[str] = None
    version: str = "1.0.0"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelData:
    """
    Контейнер для хранения признаков и целевой переменной.

    Attributes:
        features: Матрица признаков (numpy array, pandas DataFrame)
        target: Целевая переменная (numpy array, pandas Series)
        feature_names: Имена признаков
        info: Метаинформация о данных
    """

    features: Union[np.ndarray, pd.DataFrame]
    target: Union[np.ndarray, pd.Series, None] = None
    feature_names: Optional[List[str]] = None
    info: Optional[DataInfo] = None

    def __post_init__(self):
        """Валидация и инициализация после создания объекта."""
        self._convert_to_numpy()
        self._validate()
        self._init_feature_names()

    def _convert_to_numpy(self):
        """Преобразование в numpy arrays для единообразия."""
        if isinstance(self.features, pd.DataFrame):
            if self.feature_names is None:
                self.feature_names = self.features.columns.tolist()
            self.features = self.features.values

        if isinstance(self.target, pd.Series):
            self.target = self.target.values

    def _validate(self):
        """Проверка корректности данных."""
        if self.target is not None and len(self.features) != len(self.target):
            raise ValueError(
                f"Количество объектов в features ({len(self.features)}) "
                f"не совпадает с target ({len(self.target)})"
            )

    def _init_feature_names(self):
        """Инициализация имен признаков если не заданы."""
        if self.feature_names is None:
            n_features = self.features.shape[1] if len(self.features.shape) > 1 else 1
            self.feature_names = [f"feature_{i}" for i in range(n_features)]

    @property
    def n_samples(self) -> int:
        """Количество объектов в датасете."""
        return len(self.features)

    @property
    def n_features(self) -> int:
        """Количество признаков."""
        return self.features.shape[1] if len(self.features.shape) > 1 else 1

    @property
    def shape(self) -> Tuple[int, int]:
        """Форма данных (n_samples, n_features)."""
        return self.n_samples, self.n_features

    @property
    def has_target(self) -> bool:
        """Проверка наличия целевой переменной."""
        return self.target is not None

    def to_pandas(self) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Преобразование в pandas DataFrame и Series.

        Returns:
            Tuple[pd.DataFrame, Optional[pd.Series]]: (features_df, target_series)
        """
        features_df = pd.DataFrame(self.features, columns=self.feature_names)
        target_series = (
            pd.Series(self.target, name="target") if self.has_target else None
        )
        return features_df, target_series

    def sample(
        self, n: int = None, frac: float = None, random_state: Optional[int] = None
    ) -> "ModelData":
        """
        Случайная выборка объектов.

        Args:
            n: Количество объектов для выборки
            frac: Доля объектов для выборки (от 0 до 1)
            random_state: Seed для воспроизводимости

        Returns:
            ModelData: Новый объект с выбранными данными
        """
        if n is None and frac is None:
            raise ValueError("Необходимо указать либо n, либо frac")

        if frac is not None:
            n = int(self.n_samples * frac)

        rng = np.random.RandomState(random_state)
        indices = rng.choice(self.n_samples, size=min(n, self.n_samples), replace=False)

        return ModelData(
            features=self.features[indices],
            target=self.target[indices] if self.has_target else None,
            feature_names=self.feature_names,
            info=self.info,
        )

    def copy(self) -> "ModelData":
        """Создание копии объекта."""
        return ModelData(
            features=self.features.copy(),
            target=self.target.copy() if self.has_target else None,
            feature_names=self.feature_names.copy() if self.feature_names else None,
            info=self.info,
        )


@dataclass
class DataSplit:
    """
    Контейнер для хранения разбиения данных.

    Поддерживает различные комбинации:
    - train + test
    - train + validation
    - train + validation + test
    - только train (для кросс-валидации)

    Attributes:
        train: Обучающая выборка
        test: Тестовая выборка (опционально)
        validation: Валидационная выборка (опционально)
        split_info: Информация о разбиении
    """

    train: ModelData
    test: Optional[ModelData] = None
    validation: Optional[ModelData] = None
    split_info: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Валидация и добавление информации о размерах выборок."""
        if self.test is None and self.validation is None:
            # Предупреждение о том, что есть только train
            self.split_info["warning"] = "Только обучающая выборка, нет test/validation"

        self._update_split_info()

    def _update_split_info(self):
        """Обновление информации о размерах выборок."""
        sizes = {"train_size": self.train.n_samples}

        if self.test is not None:
            sizes["test_size"] = self.test.n_samples

        if self.validation is not None:
            sizes["validation_size"] = self.validation.n_samples

        sizes["total_size"] = sum(sizes.values())
        self.split_info.update(sizes)

    @property
    def has_test(self) -> bool:
        """Проверка наличия тестовой выборки."""
        return self.test is not None

    @property
    def has_validation(self) -> bool:
        """Проверка наличия валидационной выборки."""
        return self.validation is not None

    def get_split_ratios(self) -> Dict[str, float]:
        """
        Получение относительных размеров выборок.

        Returns:
            Dict[str, float]: Словарь с долями каждой выборки
        """
        total = self.split_info["total_size"]
        ratios = {"train": self.train.n_samples / total}

        if self.has_test:
            ratios["test"] = self.test.n_samples / total

        if self.has_validation:
            ratios["validation"] = self.validation.n_samples / total

        return ratios

    def get_all_data(self) -> List[Tuple[str, ModelData]]:
        """
        Получение всех доступных выборок.

        Returns:
            List[Tuple[str, ModelData]]: Список пар (название, данные)
        """
        data_list = [("train", self.train)]

        if self.has_validation:
            data_list.append(("validation", self.validation))

        if self.has_test:
            data_list.append(("test", self.test))

        return data_list


def create_data_split(
    features: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    test_size: Optional[float] = 0.2,
    validation_size: Optional[float] = None,
    random_state: Optional[int] = None,
    stratify: bool = False,
) -> DataSplit:
    """
    Универсальная функция для создания разбиения данных.

    Создает разбиение в зависимости от переданных параметров:
    - Если указан только test_size: train/test разбиение
    - Если указаны test_size и validation_size: train/validation/test разбиение
    - Если указан только validation_size: train/validation разбиение
    - Если ничего не указано: возвращает только train (для кросс-валидации)

    Args:
        features: Матрица признаков
        y: Целевая переменная
        test_size: Доля тестовой выборки от всех данных
        validation_size: Доля валидационной выборки от всех данных
        random_state: Seed для воспроизводимости
        stratify: Использовать стратификацию по целевой переменной

    Returns:
        DataSplit: Объект с разбиением данных

    Examples:
        >>> # Train/test split
        >>> split = create_data_split(features, y, test_size=0.2)

        >>> # Train/validation/test split
        >>> split = create_data_split(features, y, test_size=0.2, validation_size=0.2)

        >>> # Train/validation split
        >>> split = create_data_split(features, y, validation_size=0.2)

        >>> # Only train (for cross-validation)
        >>> split = create_data_split(features, y, test_size=None)
    """
    from sklearn.model_selection import train_test_split

    # Определяем тип разбиения
    has_test = test_size is not None and test_size > 0
    has_validation = validation_size is not None and validation_size > 0

    if not has_test:
        if not has_validation:
            train = ModelData(features=features, target=y)
            return DataSplit(
                train=train,
                split_info={
                    "random_state": random_state,
                    "split_type": "train_only",
                },
            )

        features_train, features_val, y_train, y_val = train_test_split(
            features,
            y,
            test_size=validation_size,
            random_state=random_state,
            stratify=y if stratify else None,
        )

        train = ModelData(features=features_train, target=y_train)
        validation = ModelData(features=features_val, target=y_val)

        return DataSplit(
            train=train,
            validation=validation,
            split_info={
                "validation_size": validation_size,
                "random_state": random_state,
                "stratified": stratify,
                "split_type": "train_validation",
            },
        )

    # Train/test split
    if not has_validation:
        features_train, features_test, y_train, y_test = train_test_split(
            features,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y if stratify else None,
        )

        train = ModelData(features=features_train, target=y_train)
        test = ModelData(features=features_test, target=y_test)

        return DataSplit(
            train=train,
            test=test,
            split_info={
                "test_size": test_size,
                "random_state": random_state,
                "stratified": stratify,
                "split_type": "train_test",
            },
        )

    # Проверка корректности размеров
    if test_size + validation_size >= 1.0:
        raise ValueError(
            f"Сумма test_size ({test_size}) и validation_size ({validation_size}) "
            "должна быть меньше 1.0"
        )

    # Первое разбиение: train+val / test
    features_temp, features_test, y_temp, y_test = train_test_split(
        features,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if stratify else None,
    )

    # Второе разбиение: train / val
    val_ratio = validation_size / (1 - test_size)
    features_train, features_val, y_train, y_val = train_test_split(
        features_temp,
        y_temp,
        test_size=val_ratio,
        random_state=random_state,
        stratify=y_temp if stratify else None,
    )

    train = ModelData(features=features_train, target=y_train)
    validation = ModelData(features=features_val, target=y_val)
    test = ModelData(features=features_test, target=y_test)

    return DataSplit(
        train=train,
        validation=validation,
        test=test,
        split_info={
            "test_size": test_size,
            "validation_size": validation_size,
            "random_state": random_state,
            "stratified": stratify,
            "split_type": "train_validation_test",
        },
    )


def create_kfold_data(
    features: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: Optional[int] = None,
) -> List[DataSplit]:
    """
    Создание разбиений для кросс-валидации.

    Args:
        features: Матрица признаков
        y: Целевая переменная
        n_splits: Количество фолдов
        shuffle: Перемешивать данные
        random_state: Seed для воспроизводимости

    Returns:
        List[DataSplit]: Список разбиений для каждого фолда
    """
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    splits = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(features)):
        if isinstance(features, pd.DataFrame):
            features_train, features_val = (
                features.iloc[train_idx],
                features.iloc[val_idx],
            )
        else:
            features_train, features_val = features[train_idx], features[val_idx]

        if isinstance(y, pd.Series):
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        else:
            y_train, y_val = y[train_idx], y[val_idx]

        train = ModelData(features=features_train, target=y_train)
        validation = ModelData(features=features_val, target=y_val)

        split = DataSplit(
            train=train,
            validation=validation,
            split_info={
                "fold": fold_idx,
                "n_splits": n_splits,
                "split_type": "kfold",
            },
        )
        splits.append(split)

    return splits
