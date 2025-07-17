from typing import Optional

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils.validation import check_is_fitted
import pandas as pd
import numpy as np

from dmdslab.data import DataSplit, ModelData


class DataImputer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        num_strategy: str = "iterative",
        cat_strategy: str = "most_frequent",
        num_estimator: object = None,
        num_constant: float = 0,
        cat_constant: str = "MISSING",
        random_state: Optional[int] = None,
        max_iter: int = 10,
        verbose: bool = False,
    ):
        """
        Усовершенствованный импьютер с раздельной обработкой признаков

        :param num_strategy: Стратегия для числовых данных ('iterative', 'mean', 'median', 'constant')
        :param cat_strategy: Стратегия для категориальных данных ('most_frequent', 'constant')
        :param num_estimator: Кастомный estimator для числовых данных
        :param num_constant: Значение для числовых констант
        :param cat_constant: Значение для категориальных констант
        :param random_state: Сид для воспроизводимости
        :param max_iter: Максимальное число итераций
        :param verbose: Режим вывода информации
        """
        self.num_strategy = num_strategy
        self.cat_strategy = cat_strategy
        self.num_estimator = num_estimator
        self.num_constant = num_constant
        self.cat_constant = cat_constant
        self.random_state = random_state
        self.max_iter = max_iter
        self.verbose = verbose

    def fit(self, data_split: DataSplit) -> "DataImputer":
        X_train = data_split.train.features

        # Определяем типы признаков
        self.num_cols_ = X_train.select_dtypes(include=np.number).columns.tolist()
        self.cat_cols_ = X_train.select_dtypes(exclude=np.number).columns.tolist()

        # Инициализируем импьютеры
        self._init_numeric_imputer()
        self._init_categorical_imputer()

        # Обучение импьютеров
        if self.num_cols_:
            self.num_imputer_.fit(X_train[self.num_cols_])
        if self.cat_cols_:
            self.cat_imputer_.fit(X_train[self.cat_cols_])

        return self

    def transform(self, data_split: DataSplit) -> DataSplit:
        check_is_fitted(self, ["num_imputer_", "cat_imputer_"])

        return DataSplit(
            train=self._transform_part(data_split.train),
            tuning=self._transform_part(data_split.tuning),
            test=self._transform_part(data_split.test),
        )

    def _init_numeric_imputer(self):
        """Инициализация импьютера для числовых данных"""
        if self.num_strategy == "iterative":
            estimator = self.num_estimator or RandomForestRegressor(
                random_state=self.random_state
            )
            self.num_imputer_ = IterativeImputer(
                estimator=estimator,
                max_iter=self.max_iter,
                random_state=self.random_state,
            )
        else:
            self.num_imputer_ = SimpleImputer(
                strategy=self.num_strategy, fill_value=self.num_constant
            )

    def _init_categorical_imputer(self):
        """Инициализация импьютера для категориальных данных"""
        self.cat_imputer_ = SimpleImputer(
            strategy=self.cat_strategy, fill_value=self.cat_constant
        )

    def _transform_part(self, model_data: ModelData) -> ModelData:
        """Обработка отдельной части данных"""
        X = model_data.features.copy()
        y = model_data.target.copy()

        # Обработка числовых признаков
        if self.num_cols_:
            X_num = pd.DataFrame(
                self.num_imputer_.transform(X[self.num_cols_]),
                columns=self.num_cols_,
                index=X.index,
            )
        else:
            X_num = pd.DataFrame(index=X.index)

        # Обработка категориальных признаков
        if self.cat_cols_:
            X_cat = pd.DataFrame(
                self.cat_imputer_.transform(X[self.cat_cols_]),
                columns=self.cat_cols_,
                index=X.index,
            )
        else:
            X_cat = pd.DataFrame(index=X.index)

        # Объединение результатов
        X_imputed = pd.concat([X_num, X_cat], axis=1)[X.columns]

        return ModelData(features=X_imputed, target=y)

    def get_feature_imputers(self) -> dict:
        """Возвращает информацию о заполнении признаков"""
        return {"numeric": self.num_imputer_, "categorical": self.cat_imputer_}
