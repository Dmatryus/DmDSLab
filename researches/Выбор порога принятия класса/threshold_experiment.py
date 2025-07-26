"""
Унифицированный эксперимент с автоматической обработкой категориальных признаков
===============================================================================

Добавлена поддержка категориальных данных через различные энкодеры.

Author: Dmatryus Detry
License: Apache 2.0
"""

import warnings
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import entropy
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from dmdslab.datasets import create_data_split
from dmdslab.datasets.uci_dataset_manager import TaskType, UCIDatasetManager

# Импортируем унифицированный модуль
from threshold_analysis import (
    create_selector,
    get_available_methods,
    select_confident_samples,
    TaskType as ThresholdTaskType,
)

warnings.filterwarnings("ignore")

# Попробуем импортировать CatBoostEncoder
try:
    from category_encoders import CatBoostEncoder

    HAS_CATBOOST_ENCODER = True
except ImportError:
    print(
        "Warning: CatBoostEncoder not available. Install with: pip install category_encoders"
    )
    HAS_CATBOOST_ENCODER = False


class DataPreprocessor:
    """Класс для автоматической предобработки данных"""

    def __init__(self, encoder_type="auto"):
        """
        Args:
            encoder_type: 'catboost', 'ordinal', 'auto'
        """
        self.encoder_type = encoder_type
        self.feature_encoder = None
        self.target_encoder = None
        self.numeric_features = []
        self.categorical_features = []
        self.fitted = False

    def _detect_feature_types(self, X):
        """Определение типов признаков"""
        if isinstance(X, pd.DataFrame):
            self.numeric_features = X.select_dtypes(
                include=[np.number]
            ).columns.tolist()
            self.categorical_features = X.select_dtypes(
                exclude=[np.number]
            ).columns.tolist()
        else:
            # Для numpy arrays пытаемся определить по содержимому
            self.numeric_features = []
            self.categorical_features = []

            for i in range(X.shape[1] if len(X.shape) > 1 else 1):
                col = X[:, i] if len(X.shape) > 1 else X
                try:
                    col.astype(float)
                    self.numeric_features.append(i)
                except (ValueError, TypeError):
                    self.categorical_features.append(i)

    def fit_transform(self, X, y):
        """Обучение и преобразование данных"""
        # Преобразуем в DataFrame для удобства
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Определяем типы признаков
        self._detect_feature_types(X)

        # Обрабатываем таргет
        self.target_encoder = LabelEncoder()
        y_encoded = self.target_encoder.fit_transform(y)

        # Если нет категориальных признаков, возвращаем как есть
        if not self.categorical_features:
            self.fitted = True
            return X.values, y_encoded

        # Выбираем энкодер
        if self.encoder_type == "auto":
            if HAS_CATBOOST_ENCODER and len(np.unique(y)) == 2:
                # CatBoostEncoder хорош для бинарной классификации
                encoder_type = "catboost"
            else:
                encoder_type = "ordinal"
        else:
            encoder_type = self.encoder_type

        # Создаем препроцессор
        transformers = []

        # Для числовых признаков - стандартизация
        if self.numeric_features:
            transformers.append(("num", StandardScaler(), self.numeric_features))

        # Для категориальных признаков - энкодинг
        if self.categorical_features:
            if encoder_type == "catboost" and HAS_CATBOOST_ENCODER:
                # CatBoostEncoder требует y для обучения
                encoder = CatBoostEncoder(
                    cols=self.categorical_features, return_df=False
                )
                # Обучаем отдельно
                X_cat = X[self.categorical_features]
                X_cat_encoded = encoder.fit_transform(X_cat, y_encoded)

                # Собираем обратно
                X_encoded = X.copy()
                X_encoded[self.categorical_features] = X_cat_encoded

                self.feature_encoder = encoder
                self.fitted = True

                # Стандартизируем все признаки
                scaler = StandardScaler()
                X_final = scaler.fit_transform(X_encoded)
                self.scaler = scaler

                return X_final, y_encoded
            else:
                # Используем OrdinalEncoder
                transformers.append(
                    (
                        "cat",
                        OrdinalEncoder(
                            handle_unknown="use_encoded_value", unknown_value=-1
                        ),
                        self.categorical_features,
                    )
                )

        # Создаем и обучаем ColumnTransformer
        self.feature_encoder = ColumnTransformer(
            transformers=transformers, remainder="passthrough"
        )

        X_encoded = self.feature_encoder.fit_transform(X)
        self.fitted = True

        return X_encoded, y_encoded

    def transform(self, X):
        """Преобразование новых данных"""
        if not self.fitted:
            raise ValueError("Preprocessor not fitted yet")

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        if not self.categorical_features:
            return X.values

        if isinstance(self.feature_encoder, CatBoostEncoder):
            X_encoded = X.copy()
            X_cat = X[self.categorical_features]
            X_cat_encoded = self.feature_encoder.transform(X_cat)
            X_encoded[self.categorical_features] = X_cat_encoded
            return self.scaler.transform(X_encoded)
        else:
            return self.feature_encoder.transform(X)

    def inverse_transform_target(self, y):
        """Обратное преобразование таргета"""
        if self.target_encoder is not None:
            return self.target_encoder.inverse_transform(y)
        return y


class UnifiedThresholdExperimentWithEncoding:
    """Эксперимент с автоматической обработкой категориальных данных"""

    def __init__(self, output_dir: str = "unified_experiments", encoder_type="auto"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.manager = UCIDatasetManager()
        self.results = []
        self.encoder_type = encoder_type

    def get_datasets(self) -> List[Dict]:
        """Получение разнообразных датасетов"""
        datasets_info = [
            # Бинарные датасеты
            {
                "id": 73,
                "name": "Mushroom",
                "expected_type": "binary",
            },  # Категориальные признаки!
            {"id": 2, "name": "Adult", "expected_type": "binary"},  # Смешанные типы
            {"id": 94, "name": "Spambase", "expected_type": "binary"},  # Числовые
            {"id": 222, "name": "Bank Marketing", "expected_type": "binary"},
            # Многоклассовые датасеты
            {"id": 53, "name": "Iris", "expected_type": "multiclass"},  # Числовые
            {"id": 80, "name": "Optical Recognition", "expected_type": "multiclass"},
            {"id": 459, "name": "Avila", "expected_type": "multiclass"},
        ]

        # Фильтруем доступные
        available = []
        for ds_info in datasets_info:
            try:
                if self.manager.get_dataset_info(ds_info["id"]):
                    available.append(ds_info)
            except:
                print(f"Dataset {ds_info['name']} (ID: {ds_info['id']}) not available")

        return available[:5]  # Ограничиваем для демонстрации

    def get_models(self) -> List[tuple]:
        """Модели для экспериментов"""
        return [
            ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=42)),
            (
                "Random Forest",
                RandomForestClassifier(n_estimators=100, random_state=42),
            ),
            ("Decision Tree", DecisionTreeClassifier(max_depth=10, random_state=42)),
        ]

    def get_threshold_methods(self) -> List[Dict[str, Any]]:
        """Методы выбора порогов с параметрами"""
        return [
            # Универсальные методы
            {"method": "max_prob", "params": {"threshold": 0.5}},
            {"method": "max_prob", "params": {"threshold": 0.7}},
            {"method": "percentile", "params": {"percentile": 80}},
            {"method": "percentile", "params": {"percentile": 90}},
            {"method": "entropy", "params": {}},
            {"method": "margin", "params": {"min_margin": 0.1}},
            {"method": "adaptive", "params": {"initial_threshold": 0.9}},
            # Специфичные для бинарной классификации
            {"method": "f1", "params": {}},
            {"method": "youden", "params": {}},
        ]

    def run_experiment(self):
        """Запуск эксперимента с предобработкой"""
        datasets = self.get_datasets()
        models = self.get_models()
        methods = self.get_threshold_methods()

        print(f"Starting experiment with {len(datasets)} datasets...")
        print(f"Encoder type: {self.encoder_type}")
        print(f"Methods to test: {len(methods)}")
        print(f"Models to evaluate: {len(models)}")

        for dataset_info in datasets:
            print(f"\n{'='*60}")
            print(
                f"Processing: {dataset_info['name']} (expected: {dataset_info['expected_type']})"
            )

            try:
                # Загружаем датасет
                split = self.manager.load_dataset_split(
                    dataset_info["id"],
                    test_size=0.2,
                    validation_size=0.2,
                    random_state=42,
                )

                # Создаем препроцессор
                preprocessor = DataPreprocessor(encoder_type=self.encoder_type)

                # Обучаем препроцессор и преобразуем данные
                print(f"  Preprocessing data...")
                X_train, y_train = preprocessor.fit_transform(
                    split.train.features, split.train.target
                )
                X_val = preprocessor.transform(split.validation.features)
                y_val = preprocessor.target_encoder.transform(split.validation.target)
                X_test = preprocessor.transform(split.test.features)
                y_test = preprocessor.target_encoder.transform(split.test.target)

                # Информация о предобработке
                print(
                    f"  Feature types - Numeric: {len(preprocessor.numeric_features)}, "
                    f"Categorical: {len(preprocessor.categorical_features)}"
                )

                # Определяем реальный тип задачи
                n_classes = len(np.unique(y_train))
                actual_type = "binary" if n_classes == 2 else "multiclass"

                print(f"  Actual type: {actual_type}, Classes: {n_classes}")
                print(
                    f"  Samples - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}"
                )
                print(f"  Features after encoding: {X_train.shape[1]}")

                dataset_info["n_classes"] = n_classes
                dataset_info["actual_type"] = actual_type
                dataset_info["n_categorical"] = len(preprocessor.categorical_features)
                dataset_info["n_numeric"] = len(preprocessor.numeric_features)

                # Для каждой модели
                for model_name, model in models:
                    print(f"\n  Training {model_name}...")

                    # Обучаем модель
                    model.fit(X_train, y_train)

                    # Получаем вероятности
                    val_proba = model.predict_proba(X_val)
                    test_proba = model.predict_proba(X_test)

                    # Для бинарной классификации извлекаем вероятность положительного класса
                    if actual_type == "binary":
                        val_proba_for_methods = val_proba[:, 1]
                        test_proba_for_methods = test_proba[:, 1]
                    else:
                        val_proba_for_methods = val_proba
                        test_proba_for_methods = test_proba

                    # Базовая точность модели
                    val_pred = model.predict(X_val)
                    val_accuracy = np.mean(val_pred == y_val)
                    print(f"    Validation accuracy: {val_accuracy:.3f}")

                    # Применяем все методы
                    method_results = []

                    for method_config in methods:
                        method_name = method_config["method"]
                        method_params = method_config["params"]

                        try:
                            # Применяем метод с автоматическим определением типа
                            result = select_confident_samples(
                                test_proba_for_methods,
                                method=method_name,
                                y_true=(
                                    y_test if method_name in ["f1", "youden"] else None
                                ),
                                return_result=True,
                                **method_params,
                            )

                            method_results.append(
                                {
                                    "method": result.method_name,
                                    "result": result,
                                    "execution_time": time.time(),
                                }
                            )

                            print(
                                f"    {result.method_name}: {result.selection_ratio:.1%} selected",
                                end="",
                            )
                            if result.metrics:
                                key_metric = (
                                    "f1" if actual_type == "binary" else "weighted_f1"
                                )
                                if key_metric in result.metrics:
                                    print(
                                        f" ({key_metric}: {result.metrics[key_metric]:.3f})"
                                    )
                                else:
                                    print()
                            else:
                                print()

                        except Exception as e:
                            print(f"    {method_name}: Error - {str(e)}")

                    # Визуализация результатов
                    self._create_visualizations(
                        val_proba_for_methods,
                        test_proba_for_methods,
                        y_val,
                        y_test,
                        method_results,
                        dataset_info,
                        model_name,
                    )

                    # Сохраняем результаты
                    self.results.append(
                        {
                            "dataset": dataset_info,
                            "model": model_name,
                            "val_accuracy": val_accuracy,
                            "method_results": method_results,
                            "preprocessing": {
                                "encoder_type": self.encoder_type,
                                "n_categorical": len(preprocessor.categorical_features),
                                "n_numeric": len(preprocessor.numeric_features),
                            },
                        }
                    )

            except Exception as e:
                print(f"  Error processing dataset: {str(e)}")
                import traceback

                traceback.print_exc()

        # Генерируем итоговый отчет
        self._generate_report()
        self._create_comparison_visualizations()

    def _create_visualizations(
        self,
        val_proba,
        test_proba,
        y_val,
        y_test,
        method_results,
        dataset_info,
        model_name,
    ):
        """Создание визуализаций"""
        task_type = dataset_info["actual_type"]

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Распределение вероятностей/уверенности
        ax = axes[0, 0]
        if task_type == "binary":
            ax.hist(val_proba, bins=50, alpha=0.7, density=True, edgecolor="black")
            ax.set_xlabel("P(y=1)")
            ax.set_title("Probability Distribution (Binary)")

            # Добавляем пороги
            for res_info in method_results[:3]:  # Показываем первые 3 для читаемости
                result = res_info["result"]
                if isinstance(result.threshold, float):
                    ax.axvline(
                        result.threshold,
                        linestyle="--",
                        alpha=0.7,
                        label=f"{result.method_name}: {result.threshold:.2f}",
                    )
        else:
            max_proba = np.max(val_proba, axis=1)
            ax.hist(max_proba, bins=50, alpha=0.7, density=True, edgecolor="black")
            ax.set_xlabel("Max Probability")
            ax.set_title("Max Probability Distribution (Multiclass)")

        ax.legend(fontsize=8)

        # 2. Метрики методов
        ax = axes[0, 1]
        method_names = []
        selection_ratios = []
        key_metrics = []

        for res_info in method_results:
            result = res_info["result"]
            method_names.append(result.method_name.split("(")[0])
            selection_ratios.append(result.selection_ratio * 100)

            if result.metrics:
                if task_type == "binary":
                    key_metrics.append(result.metrics.get("f1", 0) * 100)
                else:
                    key_metrics.append(result.metrics.get("weighted_f1", 0) * 100)
            else:
                key_metrics.append(0)

        # Показываем только топ методы для читаемости
        if len(method_names) > 7:
            # Сортируем по метрике и берем топ-7
            sorted_indices = np.argsort(key_metrics)[-7:]
            method_names = [method_names[i] for i in sorted_indices]
            selection_ratios = [selection_ratios[i] for i in sorted_indices]
            key_metrics = [key_metrics[i] for i in sorted_indices]

        x = np.arange(len(method_names))
        width = 0.35

        bars1 = ax.bar(
            x - width / 2, selection_ratios, width, label="Selection %", alpha=0.7
        )
        bars2 = ax.bar(x + width / 2, key_metrics, width, label="F1 Score %", alpha=0.7)

        ax.set_xlabel("Method")
        ax.set_ylabel("Percentage")
        ax.set_title("Method Performance Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(method_names, rotation=45, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Информация о данных
        ax = axes[1, 0]
        ax.axis("off")
        info_text = f"""Dataset Information:
        
Name: {dataset_info['name']}
Type: {dataset_info['actual_type']}
Classes: {dataset_info['n_classes']}
Model: {model_name}

Feature Types:
- Categorical: {dataset_info['n_categorical']}
- Numeric: {dataset_info['n_numeric']}

Validation Accuracy: {self.results[-1]['val_accuracy']:.3f}
        """
        ax.text(
            0.1,
            0.5,
            info_text,
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="center",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        # 4. Confusion matrix для лучшего метода
        ax = axes[1, 1]
        if method_results:
            best_result = max(
                (r for r in method_results if r["result"].metrics),
                key=lambda r: r["result"].metrics.get("accuracy", 0),
                default=None,
            )

            if best_result and len(best_result["result"].selected_indices) > 0:
                result = best_result["result"]
                cm = confusion_matrix(
                    y_test[result.selected_indices], result.predicted_labels
                )
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                ax.set_title(f"Confusion Matrix - {result.method_name}")
                ax.set_xlabel("Predicted")
                ax.set_ylabel("True")

        plt.suptitle(
            f"{dataset_info['name']} - {model_name} ({task_type})", fontsize=16
        )
        plt.tight_layout()

        save_path = (
            self.output_dir
            / f"{dataset_info['name']}_{model_name}_analysis.png".replace(" ", "_")
        )
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    def _create_comparison_visualizations(self):
        """Создание сравнительных визуализаций"""
        if not self.results:
            return

        # Анализ влияния категориальных признаков
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Собираем данные
        categorical_impact = []
        for result in self.results:
            n_cat = result["preprocessing"]["n_categorical"]
            best_score = max(
                (
                    r["result"].metrics.get(
                        (
                            "f1"
                            if result["dataset"]["actual_type"] == "binary"
                            else "weighted_f1"
                        ),
                        0,
                    )
                    for r in result["method_results"]
                    if r["result"].metrics
                ),
                default=0,
            )
            categorical_impact.append(
                {
                    "n_categorical": n_cat,
                    "best_score": best_score,
                    "dataset": result["dataset"]["name"],
                    "model": result["model"],
                }
            )

        df_impact = pd.DataFrame(categorical_impact)

        # График 1: Влияние количества категориальных признаков
        for model in df_impact["model"].unique():
            model_data = df_impact[df_impact["model"] == model]
            ax1.scatter(
                model_data["n_categorical"],
                model_data["best_score"],
                label=model,
                s=100,
                alpha=0.7,
            )

        ax1.set_xlabel("Number of Categorical Features")
        ax1.set_ylabel("Best F1 Score")
        ax1.set_title("Impact of Categorical Features on Performance")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # График 2: Производительность по датасетам
        datasets_perf = (
            df_impact.groupby("dataset")["best_score"]
            .mean()
            .sort_values(ascending=False)
        )
        bars = ax2.bar(range(len(datasets_perf)), datasets_perf.values)
        ax2.set_xticks(range(len(datasets_perf)))
        ax2.set_xticklabels(datasets_perf.index, rotation=45, ha="right")
        ax2.set_ylabel("Average Best F1 Score")
        ax2.set_title("Dataset Performance (with encoding)")

        # Цветовая кодировка по количеству категориальных признаков
        dataset_cat_features = df_impact.groupby("dataset")["n_categorical"].first()
        for i, (dataset, n_cat) in enumerate(dataset_cat_features.items()):
            if n_cat == 0:
                bars[i].set_color("green")
            elif n_cat < 10:
                bars[i].set_color("orange")
            else:
                bars[i].set_color("red")

        # Легенда
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor="green", label="No categorical"),
            Patch(facecolor="orange", label="1-9 categorical"),
            Patch(facecolor="red", label="10+ categorical"),
        ]
        ax2.legend(handles=legend_elements, loc="upper right")

        plt.tight_layout()
        plt.savefig(self.output_dir / "categorical_features_impact.png", dpi=300)
        plt.close()

    def _generate_report(self):
        """Генерация HTML отчета"""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Threshold Selection with Categorical Encoding</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1400px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; }}
        h1 {{ color: #2563eb; text-align: center; }}
        .encoding-info {{ background: #fef3c7; padding: 15px; border-radius: 8px; margin: 20px 0; }}
        .dataset-section {{ margin: 30px 0; padding: 20px; background: #f9fafb; border-radius: 8px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th {{ background: #3b82f6; color: white; padding: 12px; text-align: left; }}
        td {{ padding: 10px; border-bottom: 1px solid #e5e7eb; }}
        img {{ max-width: 100%; margin: 20px 0; border: 1px solid #e5e7eb; border-radius: 8px; }}
        .cat-features {{ background: #fee2e2; padding: 4px 8px; border-radius: 4px; }}
        .num-features {{ background: #dcfce7; padding: 4px 8px; border-radius: 4px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔬 Threshold Selection Experiment with Categorical Encoding</h1>
        
        <div class="encoding-info">
            <h2>⚙️ Encoding Configuration</h2>
            <p><strong>Encoder Type:</strong> {self.encoder_type}</p>
            <p><strong>Available Encoders:</strong> {'CatBoostEncoder, ' if HAS_CATBOOST_ENCODER else ''}OrdinalEncoder</p>
            <p>Categorical features are automatically detected and encoded before model training.</p>
        </div>
"""

        # Результаты по датасетам
        datasets_grouped = {}
        for result in self.results:
            dataset_name = result["dataset"]["name"]
            if dataset_name not in datasets_grouped:
                datasets_grouped[dataset_name] = {
                    "info": result["dataset"],
                    "results": [],
                }
            datasets_grouped[dataset_name]["results"].append(result)

        for dataset_name, data in datasets_grouped.items():
            dataset_info = data["info"]

            html_content += f"""
        <div class="dataset-section">
            <h2>{dataset_name}</h2>
            <p>
                <strong>Type:</strong> {dataset_info['actual_type']} ({dataset_info['n_classes']} classes) | 
                <strong>Features:</strong> 
                <span class="cat-features">Categorical: {dataset_info['n_categorical']}</span> 
                <span class="num-features">Numeric: {dataset_info['n_numeric']}</span>
            </p>
"""

            for result in data["results"]:
                model_name = result["model"]
                html_content += f"""
            <h3>Model: {model_name}</h3>
            <p><strong>Validation Accuracy:</strong> {result['val_accuracy']:.3f}</p>
            
            <img src="{dataset_name}_{model_name}_analysis.png" alt="Analysis">
            
            <table>
                <tr>
                    <th>Method</th>
                    <th>Selection %</th>
                    <th>Accuracy</th>
                    <th>F1 Score</th>
                </tr>
"""

                # Сортируем методы по F1 score
                sorted_methods = sorted(
                    result["method_results"],
                    key=lambda x: (
                        x["result"].metrics.get(
                            (
                                "f1"
                                if dataset_info["actual_type"] == "binary"
                                else "weighted_f1"
                            ),
                            0,
                        )
                        if x["result"].metrics
                        else 0
                    ),
                    reverse=True,
                )

                for method_result in sorted_methods[:10]:  # Топ-10
                    res = method_result["result"]

                    accuracy = res.metrics.get("accuracy", "-") if res.metrics else "-"
                    f1 = (
                        res.metrics.get(
                            (
                                "f1"
                                if dataset_info["actual_type"] == "binary"
                                else "weighted_f1"
                            ),
                            "-",
                        )
                        if res.metrics
                        else "-"
                    )

                    html_content += f"""
                <tr>
                    <td>{res.method_name}</td>
                    <td>{res.selection_ratio*100:.1f}%</td>
                    <td>{accuracy if isinstance(accuracy, str) else f'{accuracy:.3f}'}</td>
                    <td>{f1 if isinstance(f1, str) else f'{f1:.3f}'}</td>
                </tr>
"""

                html_content += """
            </table>
"""

            html_content += """
        </div>
"""

        # Добавляем анализ категориальных признаков
        html_content += """
        <div style="margin-top: 40px;">
            <h2>📊 Impact of Categorical Features</h2>
            <img src="categorical_features_impact.png" alt="Categorical Features Impact">
        </div>
        
        <div style="margin-top: 40px; padding: 20px; background: #f3f4f6; border-radius: 8px;">
            <h3>🔑 Key Insights with Encoding</h3>
            <ul>
                <li><strong>Categorical encoding is crucial</strong> - Datasets like Mushroom require proper encoding</li>
                <li><strong>CatBoostEncoder vs OrdinalEncoder</strong> - Target encoding can improve performance</li>
                <li><strong>Feature scaling matters</strong> - Standardization applied after encoding</li>
                <li><strong>Threshold methods still work</strong> - Unified API handles encoded features seamlessly</li>
            </ul>
        </div>
    </div>
</body>
</html>
"""

        # Сохраняем отчет
        report_path = self.output_dir / "experiment_report_with_encoding.html"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"\nReport saved to: {report_path}")


def main():
    """Запуск эксперимента с обработкой категориальных признаков"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run threshold experiment with categorical encoding"
    )
    parser.add_argument(
        "--encoder",
        choices=["auto", "catboost", "ordinal"],
        default="auto",
        help="Encoder type for categorical features",
    )
    parser.add_argument(
        "--output-dir",
        default="unified_experiments",
        help="Output directory for results",
    )

    args = parser.parse_args()

    # Проверяем доступность CatBoostEncoder
    if args.encoder == "catboost" and not HAS_CATBOOST_ENCODER:
        print("CatBoostEncoder requested but not available. Installing...")
        import subprocess

        subprocess.check_call(["pip", "install", "category_encoders"])
        print("Please restart the script.")
        return

    experiment = UnifiedThresholdExperimentWithEncoding(
        output_dir=args.output_dir, encoder_type=args.encoder
    )
    experiment.run_experiment()


if __name__ == "__main__":
    main()
