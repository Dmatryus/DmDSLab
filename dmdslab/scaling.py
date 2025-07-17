import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    PowerTransformer,
)
from sklearn.compose import ColumnTransformer
from typing import Literal, Optional

from dmdslab.data import DataSplit, ModelData


class DataScaler(BaseEstimator, TransformerMixin):
    """
    Масштабирование данных с сохранением структуры DataSplit

    Параметры:
    -----------
    scaler_type : {'standard', 'minmax', 'robust', 'yeo-johnson'}
        Тип масштабирования:
        - standard: StandardScaler (Z-нормализация)
        - minmax: MinMaxScaler [0, 1]
        - robust: RobustScaler (медиана и IQR)
        - yeo-johnson: PowerTransformer

    numeric_features : list
        Список числовых признаков для масштабирования

    copy : bool, default=True
        Создавать копию данных перед преобразованием
    """

    def __init__(
        self,
        scaler_type: Literal[
            "standard", "minmax", "robust", "yeo-johnson"
        ] = "standard",
        numeric_features: Optional[list] = None,
    ):
        self.scaler_type = scaler_type
        self.numeric_features = numeric_features
        self._fitted = False
        self.scaler_ = None

    def _init_scaler(self):
        scalers = {
            "standard": StandardScaler(),
            "minmax": MinMaxScaler(),
            "robust": RobustScaler(),
            "yeo-johnson": PowerTransformer(method="yeo-johnson"),
        }
        return scalers[self.scaler_type]

    def fit(self, data_split: DataSplit) -> "DataScaler":
        """Обучение скалера на тренировочных данных"""
        if self._fitted:
            raise RuntimeError("Scaler already fitted")

        # Определение числовых признаков
        if self.numeric_features is None:
            self.numeric_features = data_split.train.features.select_dtypes(
                include=["number"]
            ).columns.tolist()

        # Инициализация скалера
        self.scaler_ = ColumnTransformer(
            [("scaler", self._init_scaler(), self.numeric_features)],
            remainder="passthrough",
        )

        # Обучение только на train данных
        self.scaler_.fit(data_split.train.features)
        self._fitted = True
        return self

    def transform(self, data_split: DataSplit) -> DataSplit:
        """Применение масштабирования ко всем наборам данных"""
        if not self._fitted:
            raise NotFittedError("Scaler not fitted yet")

        def _transform_part(part: ModelData):
            if part is None:
                return None

            features = part.features
            transformed = self.scaler_.transform(features)

            # Сохранение имен признаков
            feature_names = self.scaler_.get_feature_names_out()
            return ModelData(
                features=pd.DataFrame(
                    transformed, columns=feature_names, index=features.index
                ),
                target=part.target,
            )

        return DataSplit(
            train=_transform_part(data_split.train),
            tuning=_transform_part(data_split.tuning),
            test=_transform_part(data_split.test),
        )

    def fit_transform(self, data_split: DataSplit) -> DataSplit:
        """Обучение и преобразование за один шаг"""
        return self.fit(data_split).transform(data_split)
