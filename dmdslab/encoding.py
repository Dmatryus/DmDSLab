from typing import Optional

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from category_encoders import CatBoostEncoder, BinaryEncoder, TargetEncoder
import pandas as pd
import numpy as np

from dmdslab.data import DataSplit, ModelData


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(
            self,
            method: str = 'onehot',
            cols: Optional[list] = None,
            handle_unknown: str = 'ignore',
            handle_missing: str = 'value',
            target_smoothing: float = 0.2,
            catboost_a: float = 10.0,
            random_state: Optional[int] = None
    ):
        """
        Расширенный кодировщик категориальных признаков

        :param method: Метод кодирования
            ('onehot', 'ordinal', 'target', 'frequency', 'catboost', 'binary')
        :param cols: Колонки для кодирования
        :param handle_unknown: Обработка неизвестных категорий
        :param handle_missing: Обработка пропусков
        :param target_smoothing: Сглаживание для target-based методов
        :param catboost_a: Параметр сглаживания для CatBoost
        :param binary_min_cardinality: Минимальная кардинальность для BinaryEncoder
        :param random_state: Сид для воспроизводимости
        """
        self.method = method
        self.cols = cols
        self.handle_unknown = handle_unknown
        self.handle_missing = handle_missing
        self.target_smoothing = target_smoothing
        self.catboost_a = catboost_a
        self.random_state = random_state
        self.encoder_ = None
        self.feature_names_ = None
        self._validate_parameters()

    def fit(self, data_split: DataSplit) -> 'CategoricalEncoder':
        X_train = data_split.train.features
        y_train = data_split.train.target

        self._detect_categorical_columns(X_train)
        self._init_encoder()
        self._fit_encoder(X_train, y_train)
        self._save_feature_names(X_train)

        return self

    def transform(self, data_split: DataSplit) -> DataSplit:
        return DataSplit(
            train=self._transform_part(data_split.train),
            tuning=self._transform_part(data_split.tuning),
            test=self._transform_part(data_split.test)
        )

    def _validate_parameters(self):
        """Проверка корректности параметров"""
        valid_methods = {'onehot', 'ordinal', 'target',
                         'frequency', 'catboost', 'binary'}
        if self.method not in valid_methods:
            raise ValueError(f"Invalid method: {self.method}. Valid options: {valid_methods}")

    def _detect_categorical_columns(self, X):
        """Определение категориальных колонок"""
        if self.cols is None:
            self.cols_ = X.select_dtypes(include=['object', 'category']).columns.tolist()
        else:
            self.cols_ = [c for c in self.cols if c in X.columns]

    def _init_encoder(self):
        """Инициализация выбранного энкодера"""
        encoder_map = {
            'onehot': OneHotEncoder(
                handle_unknown=self.handle_unknown,
                sparse_output=False
            ),
            'ordinal': OrdinalEncoder(
                handle_unknown='use_encoded_value',
                unknown_value=-1
            ),
            'target': TargetEncoder(
                smoothing=self.target_smoothing,
                handle_unknown='value'
            ),
            'frequency': self._frequency_encoder(),
            'catboost': CatBoostEncoder(
                a=self.catboost_a,
                handle_unknown='value',
                handle_missing='value'
            ),
            'binary': BinaryEncoder(
                cols=self.cols_,
                handle_unknown='value'
            )
        }
        self.encoder_ = encoder_map[self.method]

    def _fit_encoder(self, X, y):
        """Обучение энкодера"""
        if self.method in ['target', 'catboost']:
            self.encoder_.fit(X[self.cols_], y)
        else:
            self.encoder_.fit(X[self.cols_])

    def _transform_part(self, model_data: Optional[ModelData]) -> Optional[ModelData]:
        """Преобразование части данных"""
        if model_data is None:
            return None
        X = model_data.features.copy()
        y = model_data.target.copy()

        # Применяем кодирование
        encoded = self.encoder_.transform(X[self.cols_])

        # Создаем DataFrame для закодированных признаков
        if self.method in ['onehot', 'binary']:
            encoded_df = pd.DataFrame(
                encoded,
                columns=self.encoder_.get_feature_names_out(self.cols_),
                index=X.index
            )
        else:
            encoded_df = pd.DataFrame(
                encoded,
                columns=self.cols_,
                index=X.index
            )

        # Объединяем с числовыми признаками
        num_cols = [c for c in X.columns if c not in self.cols_]
        final_df = pd.concat([X[num_cols], encoded_df], axis=1)

        return ModelData(final_df, y)

    def _frequency_encoder(self):
        """Frequency Encoder с обработкой новых категорий"""

        class FrequencyEncoder:
            def __init__(self):
                self.mapping_ = {}
                self.default_ = 0.0

            def fit(self, X):
                for col in X.columns:
                    counts = X[col].value_counts(normalize=True)
                    self.mapping_[col] = counts.to_dict()
                    self.default_ = 1 / (len(counts) + 1)  # Сглаживание
                return self

            def transform(self, X):
                return X.apply(lambda col: col.map(
                    lambda x: self.mapping_[col.name].get(x, self.default_)
                ))

        return FrequencyEncoder()

    def _save_feature_names(self, X):
        """Сохранение имен фичей для совместимости с пайплайнами"""
        if self.method in ['onehot', 'binary']:
            self.feature_names_ = (
                    list(X.columns.difference(self.cols_)) +
                    self.encoder_.get_feature_names_out(self.cols_).tolist()
            )
        else:
            self.feature_names_ = X.columns.tolist()

    def get_feature_names_out(self):
        """Получение имен преобразованных признаков"""
        return self.feature_names_


# Пример использования новых методов
if __name__ == "__main__":
    # Создаем тестовые данные
    data = DataSplit(
        train=ModelData(
            pd.DataFrame({
                'product': ['A', 'B', 'A', 'C', 'B'],
                'region': ['East', 'West', 'East', 'South', 'West'],
                'sales': [100, 150, 200, 50, 300]
            }),
            pd.Series([1, 0, 1, 0, 1])
        ),
        tuning=ModelData(
            pd.DataFrame({
                'product': ['A', 'D'],
                'region': ['East', 'North'],
                'sales': [120, 80]
            }),
            pd.Series([1, 0])
        ),
        test=None
    )

    # Пример CatBoost Encoding
    cb_encoder = CategoricalEncoder(method='catboost')
    cb_encoded = cb_encoder.fit_transform(data)
    print("\nCatBoost Encoded:")
    print(cb_encoded.tuning.features)

    # Пример Binary Encoding
    bin_encoder = CategoricalEncoder(
        method='binary',
        cols=['product', 'region']
    )
    bin_encoded = bin_encoder.fit_transform(data)
    print("\nBinary Encoded:")
    print(bin_encoded.tuning.features)
