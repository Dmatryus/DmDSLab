"""
Модуль с базовыми моделями для экспериментов по выбору порогов.
Включает модели классификации и методы калибровки вероятностей.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List, Union
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from catboost import CatBoostClassifier
import lightgbm as lgb
import warnings
import time
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")


class BaseModelWrapper:
    """Базовый класс-обертка для всех моделей с единым интерфейсом."""

    def __init__(
        self,
        model_name: str,
        model_params: Dict = None,
        calibration_method: Optional[str] = None,
        random_state: int = 42,
    ):
        """
        Args:
            model_name: Название модели
            model_params: Параметры модели
            calibration_method: Метод калибровки ('sigmoid', 'isotonic', None)
            random_state: Seed для воспроизводимости
        """
        self.model_name = model_name
        self.model_params = model_params or {}
        self.calibration_method = calibration_method
        self.random_state = random_state
        self.model = None
        self.calibrated_model = None
        self.is_fitted = False
        self.training_time = 0
        self.calibration_time = 0
        self.model_info = {}

    def _create_model(self) -> BaseEstimator:
        """Создание экземпляра модели."""
        raise NotImplementedError("Должно быть реализовано в наследниках")

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "BaseModelWrapper":
        """
        Обучение модели с опциональной калибровкой.

        Args:
            X: Обучающие признаки
            y: Обучающие метки
            X_val: Валидационные признаки (для калибровки)
            y_val: Валидационные метки (для калибровки)
            sample_weight: Веса примеров
        """
        # Обучение основной модели
        start_time = time.time()
        self.model = self._create_model()

        if sample_weight is not None and hasattr(self.model, "fit"):
            # Проверяем поддержку весов
            try:
                self.model.fit(X, y, sample_weight=sample_weight)
            except:
                self.model.fit(X, y)
        else:
            self.model.fit(X, y)

        self.training_time = time.time() - start_time
        self.is_fitted = True

        # Калибровка вероятностей
        if self.calibration_method and X_val is not None and y_val is not None:
            start_time = time.time()
            self.calibrated_model = CalibratedClassifierCV(
                self.model, method=self.calibration_method, cv="prefit"
            )
            self.calibrated_model.fit(X_val, y_val)
            self.calibration_time = time.time() - start_time

        # Сохраняем информацию о модели
        self._update_model_info()

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Предсказание классов."""
        if not self.is_fitted:
            raise ValueError("Модель не обучена. Вызовите fit() сначала.")

        if self.calibrated_model is not None:
            return self.calibrated_model.predict(X)
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Предсказание вероятностей классов."""
        if not self.is_fitted:
            raise ValueError("Модель не обучена. Вызовите fit() сначала.")

        if self.calibrated_model is not None:
            return self.calibrated_model.predict_proba(X)
        return self.model.predict_proba(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Оценка модели на тестовых данных.

        Returns:
            Словарь с метриками качества
        """
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)

        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "f1_score": f1_score(y, y_pred, average="weighted"),
            "precision": precision_score(
                y, y_pred, average="weighted", zero_division=0
            ),
            "recall": recall_score(y, y_pred, average="weighted", zero_division=0),
        }

        # AUC для бинарной классификации
        if len(np.unique(y)) == 2:
            metrics["auc_roc"] = roc_auc_score(y, y_proba[:, 1])
        else:
            # Multiclass AUC
            try:
                metrics["auc_roc"] = roc_auc_score(y, y_proba, multi_class="ovr")
            except:
                metrics["auc_roc"] = None

        # Expected Calibration Error
        metrics["ece"] = self._calculate_ece(y, y_proba)

        return metrics

    def _calculate_ece(
        self, y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10
    ) -> float:
        """Расчет Expected Calibration Error."""
        # Для мультиклассовой классификации используем максимальную вероятность
        if y_proba.ndim > 1:
            confidences = np.max(y_proba, axis=1)
            predictions = np.argmax(y_proba, axis=1)
        else:
            confidences = y_proba
            predictions = (y_proba > 0.5).astype(int)

        accuracies = predictions == y_true

        ece = 0
        for bin_idx in range(n_bins):
            bin_lower = bin_idx / n_bins
            bin_upper = (bin_idx + 1) / n_bins

            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()

            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece

    def _update_model_info(self):
        """Обновление информации о модели."""
        self.model_info = {
            "model_name": self.model_name,
            "model_params": self.model_params,
            "calibration_method": self.calibration_method,
            "training_time": self.training_time,
            "calibration_time": self.calibration_time,
            "is_calibrated": self.calibrated_model is not None,
        }


class CatBoostWrapper(BaseModelWrapper):
    """Обертка для CatBoost."""

    def _create_model(self) -> CatBoostClassifier:
        default_params = {
            "iterations": 300,
            "learning_rate": 0.03,
            "depth": 6,
            "random_state": self.random_state,
            "verbose": False,
        }
        params = {**default_params, **self.model_params}
        return CatBoostClassifier(**params)


class RandomForestWrapper(BaseModelWrapper):
    """Обертка для Random Forest."""

    def _create_model(self) -> RandomForestClassifier:
        default_params = {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "random_state": self.random_state,
            "n_jobs": -1,
        }
        params = {**default_params, **self.model_params}
        return RandomForestClassifier(**params)


class ExtraTreesWrapper(BaseModelWrapper):
    """Обертка для Extra Trees."""

    def _create_model(self) -> ExtraTreesClassifier:
        default_params = {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "random_state": self.random_state,
            "n_jobs": -1,
        }
        params = {**default_params, **self.model_params}
        return ExtraTreesClassifier(**params)


class LightGBMWrapper(BaseModelWrapper):
    """Обертка для LightGBM."""

    def _create_model(self) -> lgb.LGBMClassifier:
        default_params = {
            "n_estimators": 1000,
            "learning_rate": 0.03,
            "num_leaves": 31,
            "max_depth": -1,
            "random_state": self.random_state,
            "n_jobs": -1,
            "verbose": -1,
            "objective": (
                "multiclass"
                if hasattr(self, "_n_classes") and self._n_classes > 2
                else "binary"
            ),
        }
        params = {**default_params, **self.model_params}
        return lgb.LGBMClassifier(**params)


class LogisticRegressionWrapper(BaseModelWrapper):
    """Обертка для Logistic Regression."""

    def _create_model(self) -> LogisticRegression:
        default_params = {
            "max_iter": 1000,
            "random_state": self.random_state,
            "n_jobs": -1,
        }
        params = {**default_params, **self.model_params}
        return LogisticRegression(**params)


class ModelFactory:
    """Фабрика для создания моделей."""

    MODEL_CLASSES = {
        "catboost": CatBoostWrapper,
        "random_forest": RandomForestWrapper,
        "extra_trees": ExtraTreesWrapper,
        "lightgbm": LightGBMWrapper,
        "logistic_regression": LogisticRegressionWrapper,
    }

    @classmethod
    def create_model(cls, model_type: str, **kwargs) -> BaseModelWrapper:
        """
        Создание модели по типу.

        Args:
            model_type: Тип модели ('catboost', 'random_forest', etc.)
            **kwargs: Параметры для инициализации модели

        Returns:
            Экземпляр модели
        """
        if model_type not in cls.MODEL_CLASSES:
            raise ValueError(
                f"Неизвестный тип модели: {model_type}. "
                f"Доступные: {list(cls.MODEL_CLASSES.keys())}"
            )

        model_class = cls.MODEL_CLASSES[model_type]
        return model_class(model_name=model_type, **kwargs)

    @classmethod
    def create_all_models(
        cls, calibration_method: Optional[str] = None, random_state: int = 42
    ) -> Dict[str, BaseModelWrapper]:
        """
        Создание всех доступных моделей.

        Returns:
            Словарь {название_модели: экземпляр_модели}
        """
        models = {}
        for model_type in cls.MODEL_CLASSES:
            models[model_type] = cls.create_model(
                model_type,
                calibration_method=calibration_method,
                random_state=random_state,
            )
        return models


class ModelEnsemble:
    """Ансамбль моделей для повышения качества псевдо-разметки."""

    def __init__(self, models: List[BaseModelWrapper], voting: str = "soft"):
        """
        Args:
            models: Список обученных моделей
            voting: Тип голосования ('soft' или 'hard')
        """
        self.models = models
        self.voting = voting

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Усредненные вероятности от всех моделей."""
        if self.voting != "soft":
            raise ValueError("predict_proba доступен только для soft voting")

        # Получаем предсказания от всех моделей
        predictions = []
        for model in self.models:
            pred = model.predict_proba(X)
            predictions.append(pred)

        # Усредняем
        return np.mean(predictions, axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Предсказание классов через голосование."""
        if self.voting == "soft":
            proba = self.predict_proba(X)
            return np.argmax(proba, axis=1)
        else:
            # Hard voting
            predictions = []
            for model in self.models:
                pred = model.predict(X)
                predictions.append(pred)

            predictions = np.array(predictions)
            # Мода по каждому примеру
            return np.apply_along_axis(
                lambda x: np.bincount(x).argmax(), axis=0, arr=predictions
            )


def train_model_with_pseudo_labels(
    model: BaseModelWrapper,
    X_labeled: np.ndarray,
    y_labeled: np.ndarray,
    X_pseudo: np.ndarray,
    y_pseudo: np.ndarray,
    pseudo_weights: Optional[np.ndarray] = None,
    validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> BaseModelWrapper:
    """
    Дообучение модели с использованием псевдо-меток.

    Args:
        model: Базовая модель
        X_labeled: Размеченные признаки
        y_labeled: Размеченные метки
        X_pseudo: Псевдо-размеченные признаки
        y_pseudo: Псевдо-метки
        pseudo_weights: Веса для псевдо-примеров
        validation_data: Кортеж (X_val, y_val) для калибровки

    Returns:
        Дообученная модель
    """
    # Объединяем данные
    X_combined = np.vstack([X_labeled, X_pseudo])
    y_combined = np.hstack([y_labeled, y_pseudo])

    # Создаем веса (1.0 для размеченных, заданные или 0.5 для псевдо)
    if pseudo_weights is None:
        pseudo_weights = np.full(len(X_pseudo), 0.5)

    weights = np.hstack([np.ones(len(X_labeled)), pseudo_weights])

    # Создаем новую модель с теми же параметрами
    new_model = ModelFactory.create_model(
        model.model_name,
        model_params=model.model_params,
        calibration_method=model.calibration_method,
        random_state=model.random_state,
    )

    # Обучаем
    if validation_data:
        new_model.fit(
            X_combined,
            y_combined,
            X_val=validation_data[0],
            y_val=validation_data[1],
            sample_weight=weights,
        )
    else:
        new_model.fit(X_combined, y_combined, sample_weight=weights)

    return new_model


def parallel_model_training(
    model_configs: List[Dict],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    n_jobs: int = -1,
) -> List[BaseModelWrapper]:
    """
    Параллельное обучение нескольких моделей.

    Args:
        model_configs: Список конфигураций моделей
        X_train, y_train: Обучающие данные
        X_val, y_val: Валидационные данные
        n_jobs: Количество параллельных процессов

    Returns:
        Список обученных моделей
    """

    def train_single_model(config):
        model = ModelFactory.create_model(**config)
        model.fit(X_train, y_train, X_val, y_val)
        return model

    trained_models = Parallel(n_jobs=n_jobs)(
        delayed(train_single_model)(config) for config in model_configs
    )

    return trained_models


if __name__ == "__main__":
    # Тестирование модулей
    print("Тестирование базовых моделей...")

    # Генерируем тестовые данные
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=1000, n_features=20, n_classes=3, n_informative=15, random_state=42
    )

    # Разбиваем на train/val
    from sklearn.model_selection import train_test_split

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Создаем и обучаем модели
    models = ModelFactory.create_all_models(calibration_method="isotonic")

    for model_name, model in models.items():
        print(f"\n{model_name}:")
        model.fit(X_train, y_train, X_val, y_val)
        metrics = model.evaluate(X_val, y_val)

        print(f"  Accuracy: {metrics['accuracy']:.3f}")
        print(f"  F1-score: {metrics['f1_score']:.3f}")
        print(f"  ECE: {metrics['ece']:.3f}")
        print(f"  Training time: {model.training_time:.2f}s")
