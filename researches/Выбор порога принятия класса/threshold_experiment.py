"""
Модуль для проведения экспериментов по сравнению методов выбора порогов.

Включает загрузку датасетов, обучение моделей и оценку методов.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
import warnings
from pathlib import Path
import json
import time
from datetime import datetime

# ML библиотеки
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    log_loss,
)
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB

# CatBoost
from catboost import CatBoostClassifier
from category_encoders import CatBoostEncoder

# Внутренние модули
from dmdslab.datasets.uci_dataset_manager import UCIDatasetManager, TaskType, ModelData
from threshold_methods import (
    BaseThresholdSelector,
    ThresholdMethodFactory,
    ThresholdResult,
)


@dataclass
class DatasetConfig:
    """Конфигурация датасета для эксперимента."""

    dataset_id: int
    name: str
    task_type: str
    n_classes: int = 2


@dataclass
class ModelConfig:
    """Конфигурация модели для эксперимента."""

    name: str
    model_class: Any
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentResult:
    """Результат эксперимента для одной комбинации датасет-модель-метод."""

    dataset_name: str
    model_name: str
    threshold_method: str
    threshold_value: float
    n_selected: int
    n_total: int
    selection_rate: float
    # Метрики на псевдо-размеченных данных
    pseudo_accuracy: float
    pseudo_f1: float
    pseudo_precision: float
    pseudo_recall: float
    # Метрики модели
    model_train_time: float
    model_inference_time: float
    # Дополнительная информация
    additional_info: Dict[str, Any] = field(default_factory=dict)


class ThresholdExperiment:
    """Класс для проведения экспериментов по сравнению методов выбора порогов."""

    def __init__(self, experiment_name: str = "threshold_comparison"):
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results: List[ExperimentResult] = []
        self.uci_manager = UCIDatasetManager()

    def get_datasets(self) -> Dict[str, List[DatasetConfig]]:
        """Получение датасетов для экспериментов."""
        datasets = {
            "binary": [
                DatasetConfig(222, "Bank Marketing", "binary", 2),
                DatasetConfig(350, "Credit Card Default", "binary", 2),
                DatasetConfig(159, "MAGIC Gamma Telescope", "binary", 2),
            ],
            "multiclass": [
                # Только действительно многоклассовые датасеты
                DatasetConfig(224, "Gas Sensor Array Drift", "multiclass", 6),
                DatasetConfig(459, "Avila", "multiclass", 12),
                DatasetConfig(102, "Thyroid Disease", "multiclass", 3),
            ],
        }
        return datasets

    def get_models(self) -> List[ModelConfig]:
        """Получение конфигураций моделей для экспериментов."""
        return [
            ModelConfig(
                "CatBoost",
                CatBoostClassifier,
                {
                    "iterations": 100,
                    "learning_rate": 0.1,
                    "depth": 6,
                    "verbose": False,
                    "random_state": 42,
                },
            ),
            ModelConfig(
                "RandomForest",
                RandomForestClassifier,
                {
                    "n_estimators": 100,
                    "max_depth": 10,
                    "random_state": 42,
                    "n_jobs": -1,
                },
            ),
            ModelConfig(
                "ExtraTrees",
                ExtraTreesClassifier,
                {
                    "n_estimators": 100,
                    "max_depth": 10,
                    "random_state": 42,
                    "n_jobs": -1,
                },
            ),
            ModelConfig(
                "LogisticRegression",
                LogisticRegression,
                {"max_iter": 1000, "random_state": 42},
            ),
            ModelConfig(
                "GaussianNB",
                GaussianNB,
                {},
            ),
            ModelConfig(
                "MLP",
                MLPClassifier,
                {
                    "hidden_layer_sizes": (100, 50),
                    "max_iter": 500,
                    "random_state": 42,
                    "early_stopping": True,
                },
            ),
        ]

    def prepare_data(
        self, model_data: ModelData, task_type: str
    ) -> Tuple[np.ndarray, np.ndarray, List[int], List[int]]:
        """
        Подготовка данных для эксперимента.

        Returns:
            X: Признаки
            y: Метки
            categorical_features: Индексы категориальных признаков
            numerical_features: Индексы численных признаков
        """
        X = model_data.features
        y = model_data.target

        # Преобразуем в numpy если нужно
        if hasattr(X, "values"):
            X = X.values
        if hasattr(y, "values"):
            y = y.values

        # Создаем копию для безопасной обработки
        X = X.copy()
        y = y.copy()

        # Определяем типы признаков более надежно
        categorical_features = []
        numerical_features = []

        for i in range(X.shape[1]):
            col = X[:, i]

            # Проверяем, является ли колонка числовой
            try:
                # Попытка преобразовать в float
                col_numeric = pd.to_numeric(col, errors="coerce")

                # Если больше 90% значений успешно преобразовались в числа
                if np.sum(~np.isnan(col_numeric)) / len(col) > 0.9:
                    # Это числовой признак
                    X[:, i] = col_numeric
                    numerical_features.append(i)
                else:
                    # Это категориальный признак
                    categorical_features.append(i)
            except:
                # Если не удалось преобразовать - категориальный
                categorical_features.append(i)

        # Label encoding для категориальных признаков
        if categorical_features:
            for idx in categorical_features:
                le = LabelEncoder()
                # Обрабатываем пропуски
                mask = pd.isna(X[:, idx]) | (X[:, idx] == "") | (X[:, idx] == "nan")
                X[~mask, idx] = le.fit_transform(X[~mask, idx])
                X[mask, idx] = -1  # Специальное значение для пропусков

        # Imputation для численных признаков
        if numerical_features:
            imputer = SimpleImputer(strategy="mean")
            X[:, numerical_features] = imputer.fit_transform(X[:, numerical_features])

        # Преобразуем все в float для единообразия
        X = X.astype(np.float32)

        # Label encoding для target если нужно
        if y.dtype == object or y.dtype.kind in ["U", "S"]:
            le = LabelEncoder()
            y = le.fit_transform(y)

        return X, y, categorical_features, numerical_features

    def run_single_experiment(
        self,
        dataset_config: DatasetConfig,
        model_config: ModelConfig,
        threshold_methods: List[BaseThresholdSelector],
    ) -> List[ExperimentResult]:
        """Запуск эксперимента для одной комбинации датасет-модель."""
        print(f"\n  Обработка: {dataset_config.name} + {model_config.name}")

        try:
            # Загружаем датасет
            model_data = self.uci_manager.load_dataset(dataset_config.dataset_id)
            X, y, cat_features, num_features = self.prepare_data(
                model_data, dataset_config.task_type
            )

            # Разбиваем данные: train/val/test/unlabeled
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            X_train, X_unlabeled, y_train, y_unlabeled = train_test_split(
                X_temp, y_temp, test_size=0.6, random_state=42, stratify=y_temp
            )

            X_train_full, X_val, y_train_full, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
            )

            print(
                f"    Размеры: train={len(X_train_full)}, val={len(X_val)}, "
                f"unlabeled={len(X_unlabeled)}, test={len(X_test)}"
            )

            # Кодирование категориальных признаков
            if cat_features and model_config.name != "CatBoost":
                encoder = CatBoostEncoder(cols=cat_features)
                X_train_full = encoder.fit_transform(X_train_full, y_train_full)
                X_val = encoder.transform(X_val)
                X_unlabeled = encoder.transform(X_unlabeled)
                X_test = encoder.transform(X_test)

            # Обучаем модель
            model = model_config.model_class(**model_config.params)

            # Специальная обработка для CatBoost
            if model_config.name == "CatBoost" and cat_features:
                start_time = time.time()
                model.fit(
                    X_train_full,
                    y_train_full,
                    cat_features=cat_features,
                    eval_set=(X_val, y_val),
                    verbose=False,
                )
                train_time = time.time() - start_time
            else:
                start_time = time.time()
                model.fit(X_train_full, y_train_full)
                train_time = time.time() - start_time

            # Получаем предсказания на неразмеченных данных
            start_time = time.time()
            if hasattr(model, "predict_proba"):
                y_proba_unlabeled = model.predict_proba(X_unlabeled)
            else:
                # Для моделей без predict_proba используем predict
                y_pred = model.predict(X_unlabeled)
                n_classes = len(np.unique(y_train_full))
                y_proba_unlabeled = np.zeros((len(y_pred), n_classes))
                y_proba_unlabeled[np.arange(len(y_pred)), y_pred] = 1.0

            inference_time = time.time() - start_time

            results = []

            # Тестируем каждый метод выбора порога
            for method in threshold_methods:
                try:
                    # Для методов, требующих истинные метки, используем валидацию
                    if method.name in [
                        "Optimal F1",
                        "Youden J Statistic",
                        "Cost Sensitive",
                    ]:
                        y_proba_val = (
                            model.predict_proba(X_val)
                            if hasattr(model, "predict_proba")
                            else None
                        )
                        if y_proba_val is None:
                            continue
                        # Получаем порог на валидации
                        threshold_result_val = method.select_threshold(
                            y_val, y_proba_val
                        )

                        # Применяем найденный порог к неразмеченным данным
                        if (
                            dataset_config.task_type == "binary"
                            and y_proba_unlabeled.shape[1] == 2
                        ):
                            proba_positive = y_proba_unlabeled[:, 1]
                        else:
                            proba_positive = np.max(y_proba_unlabeled, axis=1)

                        confident_mask = (
                            proba_positive >= threshold_result_val.threshold
                        )

                        # Создаем результат с найденным порогом
                        threshold_result = ThresholdResult(
                            threshold=threshold_result_val.threshold,
                            score=threshold_result_val.score,
                            method_name=method.name,
                            additional_info=threshold_result_val.additional_info,
                            confident_mask=confident_mask,
                        )
                    else:
                        threshold_result = method.select_threshold(
                            None, y_proba_unlabeled
                        )

                    # Оцениваем качество псевдо-разметки
                    confident_mask = threshold_result.confident_mask
                    n_selected = np.sum(confident_mask)

                    if n_selected > 0:
                        # Сравниваем псевдо-метки с истинными
                        if dataset_config.task_type == "binary":
                            if y_proba_unlabeled.shape[1] == 2:
                                pseudo_labels = (
                                    y_proba_unlabeled[confident_mask, 1]
                                    >= threshold_result.threshold
                                ).astype(int)
                            else:
                                pseudo_labels = (
                                    y_proba_unlabeled[confident_mask]
                                    >= threshold_result.threshold
                                ).astype(int)
                        else:
                            pseudo_labels = np.argmax(
                                y_proba_unlabeled[confident_mask], axis=1
                            )

                        true_labels = y_unlabeled[confident_mask]

                        # Вычисляем метрики
                        pseudo_accuracy = accuracy_score(true_labels, pseudo_labels)
                        pseudo_f1 = f1_score(
                            true_labels,
                            pseudo_labels,
                            average=(
                                "macro"
                                if dataset_config.task_type == "multiclass"
                                else "binary"
                            ),
                        )
                        pseudo_precision = precision_score(
                            true_labels,
                            pseudo_labels,
                            average=(
                                "macro"
                                if dataset_config.task_type == "multiclass"
                                else "binary"
                            ),
                            zero_division=0,
                        )
                        pseudo_recall = recall_score(
                            true_labels,
                            pseudo_labels,
                            average=(
                                "macro"
                                if dataset_config.task_type == "multiclass"
                                else "binary"
                            ),
                            zero_division=0,
                        )
                    else:
                        pseudo_accuracy = pseudo_f1 = pseudo_precision = (
                            pseudo_recall
                        ) = 0.0

                    result = ExperimentResult(
                        dataset_name=dataset_config.name,
                        model_name=model_config.name,
                        threshold_method=method.name,
                        threshold_value=threshold_result.threshold,
                        n_selected=n_selected,
                        n_total=len(y_unlabeled),
                        selection_rate=n_selected / len(y_unlabeled),
                        pseudo_accuracy=pseudo_accuracy,
                        pseudo_f1=pseudo_f1,
                        pseudo_precision=pseudo_precision,
                        pseudo_recall=pseudo_recall,
                        model_train_time=train_time,
                        model_inference_time=inference_time,
                        additional_info=threshold_result.additional_info,
                    )

                    results.append(result)

                except Exception as e:
                    print(f"      Ошибка в методе {method.name}: {str(e)}")
                    continue

            return results

        except Exception as e:
            print(f"    Ошибка в эксперименте: {str(e)}")
            return []

    def run_experiments(self):
        """Запуск всех экспериментов."""
        print(f"Начинаем эксперимент: {self.experiment_name}")
        print("=" * 80)

        datasets = self.get_datasets()
        models = self.get_models()
        methods = ThresholdMethodFactory.get_all_methods()

        # Эксперименты для бинарной классификации
        print("\n1. Эксперименты для бинарной классификации")
        print("-" * 40)
        for dataset_config in datasets["binary"]:
            for model_config in models:
                results = self.run_single_experiment(
                    dataset_config,
                    model_config,
                    methods["binary"] + methods["universal"],
                )
                self.results.extend(results)

        # Эксперименты для многоклассовой классификации
        print("\n2. Эксперименты для многоклассовой классификации")
        print("-" * 40)
        for dataset_config in datasets["multiclass"]:
            for model_config in models:
                results = self.run_single_experiment(
                    dataset_config,
                    model_config,
                    methods["multiclass"] + methods["universal"],
                )
                self.results.extend(results)

        print(f"\nВсего проведено экспериментов: {len(self.results)}")

    def save_results(self, output_dir: str = "results"):
        """Сохранение результатов эксперимента."""
        output_path = Path(output_dir) / self.experiment_name
        output_path.mkdir(parents=True, exist_ok=True)

        # Сохраняем как DataFrame
        results_df = pd.DataFrame([vars(r) for r in self.results])
        csv_path = output_path / f"results_{self.timestamp}.csv"
        results_df.to_csv(csv_path, index=False)
        print(f"\nРезультаты сохранены в: {csv_path}")

        # Сохраняем сводную статистику
        summary = self._generate_summary(results_df)
        summary_path = output_path / f"summary_{self.timestamp}.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        return results_df

    def _generate_summary(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """Генерация сводной статистики."""
        summary = {
            "experiment_name": self.experiment_name,
            "timestamp": self.timestamp,
            "total_experiments": len(results_df),
            "datasets": results_df["dataset_name"].unique().tolist(),
            "models": results_df["model_name"].unique().tolist(),
            "methods": results_df["threshold_method"].unique().tolist(),
        }

        # Лучшие методы по метрикам
        best_by_accuracy = (
            results_df.groupby("threshold_method")["pseudo_accuracy"]
            .mean()
            .sort_values(ascending=False)
            .head(5)
            .to_dict()
        )

        best_by_f1 = (
            results_df.groupby("threshold_method")["pseudo_f1"]
            .mean()
            .sort_values(ascending=False)
            .head(5)
            .to_dict()
        )

        best_by_selection = (
            results_df.groupby("threshold_method")["selection_rate"]
            .mean()
            .sort_values(ascending=False)
            .head(5)
            .to_dict()
        )

        summary["best_methods"] = {
            "by_accuracy": best_by_accuracy,
            "by_f1": best_by_f1,
            "by_selection_rate": best_by_selection,
        }

        # Статистика по моделям - исправляем формат для JSON
        model_stats_raw = results_df.groupby("model_name").agg(
            {
                "pseudo_accuracy": ["mean", "std"],
                "pseudo_f1": ["mean", "std"],
                "model_train_time": "mean",
            }
        )

        # Преобразуем MultiIndex в обычный словарь
        model_stats = {}
        for model in model_stats_raw.index:
            model_stats[model] = {
                "accuracy_mean": float(
                    model_stats_raw.loc[model, ("pseudo_accuracy", "mean")]
                ),
                "accuracy_std": float(
                    model_stats_raw.loc[model, ("pseudo_accuracy", "std")]
                ),
                "f1_mean": float(model_stats_raw.loc[model, ("pseudo_f1", "mean")]),
                "f1_std": float(model_stats_raw.loc[model, ("pseudo_f1", "std")]),
                "train_time_mean": float(
                    model_stats_raw.loc[model, ("model_train_time", "mean")]
                ),
            }

        summary["model_statistics"] = model_stats

        return summary
