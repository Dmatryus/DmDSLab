from typing import Optional
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from dmdslab.data import ModelData, DataSplit


def split_dataset(
        data: ModelData,
        test_size: Optional[float] = None,
        tuning_size: Optional[float] = None,
        random_state: Optional[int] = None,
        stratify: bool = False
) -> DataSplit:
    """
    Разделяет данные на обучающую, валидационную и тестовую выборки.

    :param data: Объект ModelData с признаками и целевым значением
    :param test_size: Доля тестовой выборки
    :param tuning_size: Доля валидационной выборки
    :param random_state: Случайное начальное значение
    :param stratify: Использовать стратифицированное разделение
    :return: Объект DataSplit с разделёнными выборками
    """
    if not isinstance(data, ModelData):
        raise ValueError("Данные должны быть предоставлены в формате ModelData")

    def _split(features, target, split_size):
        """Вспомогательная функция для разделения данных"""
        if stratify:
            splitter = StratifiedShuffleSplit(
                n_splits=1,
                test_size=split_size,
                random_state=random_state
            )
            train_idx, test_idx = next(splitter.split(features, target))
            return (
                features.iloc[train_idx],
                features.iloc[test_idx],
                target.iloc[train_idx],
                target.iloc[test_idx]
            )
        return train_test_split(
            features, target,
            test_size=split_size,
            random_state=random_state
        )

    features, target = data.features, data.target
    test_data = tuning_data = None

    # Первое разделение: train + tuning | test
    if test_size:
        features, test_features, target, test_target = _split(features, target, test_size)
        test_data = ModelData(test_features, test_target)

    # Второе разделение: train | tuning
    if tuning_size:
        tuning_split_size = tuning_size / (1 - test_size) if test_size else tuning_size
        features, tuning_features, target, tuning_target = _split(features, target, tuning_split_size)
        tuning_data = ModelData(tuning_features, tuning_target)

    train_data = ModelData(features, target)

    return DataSplit(
        train=train_data,
        tuning=tuning_data,
        test=test_data
    )
