"""
Модуль визуализации результатов экспериментов по выбору порогов.

Создает информативные графики для анализа результатов.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# Настройка стиля графиков
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")


class ThresholdExperimentVisualizer:
    """Класс для визуализации результатов экспериментов с порогами."""

    def __init__(self, results_df: pd.DataFrame, output_dir: str = "results/plots"):
        self.results_df = results_df
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Цветовые палитры
        self.model_colors = sns.color_palette("Set2", n_colors=10)
        self.method_colors = sns.color_palette("Set3", n_colors=20)

    def create_all_visualizations(self):
        """Создание всех визуализаций."""
        print("Создание визуализаций...")

        # 1. Обзорная тепловая карта
        self.plot_performance_heatmap()

        # 2. Сравнение методов по метрикам
        self.plot_method_comparison()

        # 3. Trade-off между качеством и количеством
        self.plot_quality_vs_quantity_tradeoff()

        # 4. Анализ по моделям
        self.plot_model_performance()

        # 5. Анализ порогов
        self.plot_threshold_distribution()

        # 6. Временные характеристики
        self.plot_time_analysis()

        # 7. Детальный анализ для бинарной классификации
        self.plot_binary_classification_analysis()

        # 8. Детальный анализ для многоклассовой классификации
        self.plot_multiclass_analysis()

        # 9. Сводный дашборд
        self.create_summary_dashboard()

        print(f"Визуализации сохранены в: {self.output_dir}")

    def plot_performance_heatmap(self):
        """Тепловая карта производительности методов на разных датасетах."""
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))

        metrics = ["pseudo_accuracy", "pseudo_f1", "pseudo_precision", "pseudo_recall"]
        titles = ["Accuracy", "F1-Score", "Precision", "Recall"]

        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[idx // 2, idx % 2]

            # Создаем pivot table
            pivot_data = self.results_df.pivot_table(
                values=metric,
                index="threshold_method",
                columns="dataset_name",
                aggfunc="mean",
            )

            # Сортируем по среднему значению
            row_order = pivot_data.mean(axis=1).sort_values(ascending=False).index

            sns.heatmap(
                pivot_data.loc[row_order],
                annot=True,
                fmt=".3f",
                cmap="RdYlGn",
                vmin=0,
                vmax=1,
                cbar_kws={"label": title},
                ax=ax,
            )

            ax.set_title(f"Среднее {title} по методам и датасетам", fontsize=14, pad=15)
            ax.set_xlabel("Датасет", fontsize=12)
            ax.set_ylabel("Метод выбора порога", fontsize=12)

        plt.suptitle(
            "Производительность методов выбора порога на разных датасетах",
            fontsize=16,
            y=1.02,
        )
        plt.tight_layout()
        plt.savefig(self.output_dir / "performance_heatmap.png", dpi=300, bbox_inches="tight")
        plt.close()

    def plot_method_comparison(self):
        """Сравнение методов по основным метрикам."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Группируем по методам
        method_stats = self.results_df.groupby("threshold_method").agg(
            {
                "pseudo_accuracy": ["mean", "std"],
                "pseudo_f1": ["mean", "std"],
                "selection_rate": ["mean", "std"],
                "threshold_value": ["mean", "std"],
            }
        )

        # 1. Accuracy vs F1
        ax = axes[0, 0]
        ax.errorbar(
            method_stats[("pseudo_accuracy", "mean")],
            method_stats[("pseudo_f1", "mean")],
            xerr=method_stats[("pseudo_accuracy", "std")],
            yerr=method_stats[("pseudo_f1", "std")],
            fmt="o",
            capsize=5,
            markersize=8,
        )

        for idx, method in enumerate(method_stats.index):
            ax.annotate(
                method,
                (
                    method_stats.loc[method, ("pseudo_accuracy", "mean")],
                    method_stats.loc[method, ("pseudo_f1", "mean")],
                ),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                alpha=0.7,
            )

        ax.set_xlabel("Средняя Accuracy", fontsize=12)
        ax.set_ylabel("Средний F1-Score", fontsize=12)
        ax.set_title("Accuracy vs F1-Score по методам", fontsize=14)
        ax.grid(True, alpha=0.3)

        # 2. Box plot для accuracy
        ax = axes[0, 1]
        accuracy_data = []
        labels = []
        for method in method_stats.index[:10]:  # Top 10 методов
            data = self.results_df[self.results_df["threshold_method"] == method][
                "pseudo_accuracy"
            ]
            if len(data) > 0:
                accuracy_data.append(data)
                labels.append(method)

        bp = ax.boxplot(accuracy_data, labels=labels, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("lightblue")

        ax.set_xlabel("Метод", fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_title("Распределение Accuracy по методам (Top 10)", fontsize=14)
        ax.tick_params(axis="x", rotation=45)

        # 3. Selection rate vs Quality
        ax = axes[1, 0]
        scatter = ax.scatter(
            method_stats[("selection_rate", "mean")],
            method_stats[("pseudo_f1", "mean")],
            s=200,
            c=method_stats[("pseudo_accuracy", "mean")],
            cmap="viridis",
            alpha=0.7,
            edgecolors="black",
            linewidth=1,
        )

        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Accuracy", fontsize=10)

        ax.set_xlabel("Доля отобранных примеров", fontsize=12)
        ax.set_ylabel("F1-Score", fontsize=12)
        ax.set_title("Trade-off: Количество vs Качество", fontsize=14)
        ax.grid(True, alpha=0.3)

        # 4. Средние пороги по методам
        ax = axes[1, 1]
        threshold_means = method_stats[("threshold_value", "mean")].sort_values(ascending=False)
        threshold_stds = method_stats.loc[threshold_means.index, ("threshold_value", "std")]

        bars = ax.bar(
            range(len(threshold_means)),
            threshold_means.values,
            yerr=threshold_stds.values,
            capsize=5,
            color="coral",
            alpha=0.7,
            edgecolor="black",
        )

        ax.set_xticks(range(len(threshold_means)))
        ax.set_xticklabels(threshold_means.index, rotation=45, ha="right")
        ax.set_xlabel("Метод", fontsize=12)
        ax.set_ylabel("Средний порог", fontsize=12)
        ax.set_title("Средние значения порогов по методам", fontsize=14)
        ax.grid(True, axis="y", alpha=0.3)

        plt.suptitle("Сравнение методов выбора порога", fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig(self.output_dir / "method_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()

    def plot_quality_vs_quantity_tradeoff(self):
        """Визуализация trade-off между качеством и количеством отобранных примеров."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Группируем по типу задачи
        binary_data = self.results_df[
            self.results_df["dataset_name"].isin(["Mushroom", "Adult", "Spambase", "Bank Marketing", "Credit Card Default", "MAGIC Gamma Telescope"])
        ]
        multiclass_data = self.results_df[
            ~self.results_df["dataset_name"].isin(["Mushroom", "Adult", "Spambase", "Bank Marketing", "Credit Card Default", "MAGIC Gamma Telescope"])
        ]

        for idx, (data, title) in enumerate(
            [(binary_data, "Бинарная классификация"), (multiclass_data, "Многоклассовая классификация")]
        ):
            ax = axes[idx]

            # Для каждого метода строим кривую
            methods = data["threshold_method"].unique()

            for method in methods[:10]:  # Top 10 методов
                method_data = data[data["threshold_method"] == method]

                # Группируем по датасетам и усредняем
                grouped = method_data.groupby("dataset_name").agg(
                    {"selection_rate": "mean", "pseudo_f1": "mean"}
                )

                if len(grouped) > 1:
                    ax.plot(
                        grouped["selection_rate"],
                        grouped["pseudo_f1"],
                        marker="o",
                        markersize=8,
                        linewidth=2,
                        alpha=0.7,
                        label=method,
                    )

            ax.set_xlabel("Доля отобранных примеров", fontsize=12)
            ax.set_ylabel("F1-Score", fontsize=12)
            ax.set_title(f"Trade-off для {title}", fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)

        plt.suptitle("Анализ Trade-off: Качество vs Количество", fontsize=16)
        plt.tight_layout()
        plt.savefig(self.output_dir / "quality_quantity_tradeoff.png", dpi=300, bbox_inches="tight")
        plt.close()

    def plot_model_performance(self):
        """Анализ производительности по моделям."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Средние метрики по моделям
        ax = axes[0, 0]
        model_stats = self.results_df.groupby("model_name")[
            ["pseudo_accuracy", "pseudo_f1", "pseudo_precision", "pseudo_recall"]
        ].mean()

        model_stats.plot(kind="bar", ax=ax, width=0.8)
        ax.set_xlabel("Модель", fontsize=12)
        ax.set_ylabel("Значение метрики", fontsize=12)
        ax.set_title("Средние метрики по моделям", fontsize=14)
        ax.legend(["Accuracy", "F1", "Precision", "Recall"])
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, axis="y", alpha=0.3)

        # 2. Время обучения vs качество
        ax = axes[0, 1]
        model_time_quality = self.results_df.groupby("model_name").agg(
            {"model_train_time": "mean", "pseudo_f1": "mean", "pseudo_accuracy": "mean"}
        )

        scatter = ax.scatter(
            model_time_quality["model_train_time"],
            model_time_quality["pseudo_f1"],
            s=300,
            c=model_time_quality["pseudo_accuracy"],
            cmap="coolwarm",
            alpha=0.7,
            edgecolors="black",
            linewidth=2,
        )

        for idx, model in enumerate(model_time_quality.index):
            ax.annotate(
                model,
                (
                    model_time_quality.loc[model, "model_train_time"],
                    model_time_quality.loc[model, "pseudo_f1"],
                ),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=10,
            )

        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Accuracy", fontsize=10)

        ax.set_xlabel("Среднее время обучения (сек)", fontsize=12)
        ax.set_ylabel("Средний F1-Score", fontsize=12)
        ax.set_title("Время обучения vs Качество предсказаний", fontsize=14)
        ax.grid(True, alpha=0.3)

        # 3. Стабильность моделей
        ax = axes[1, 0]
        model_stability = self.results_df.groupby("model_name")[
            ["pseudo_accuracy", "pseudo_f1"]
        ].std()

        model_stability.plot(kind="bar", ax=ax, width=0.8)
        ax.set_xlabel("Модель", fontsize=12)
        ax.set_ylabel("Стандартное отклонение", fontsize=12)
        ax.set_title("Стабильность моделей (чем меньше, тем лучше)", fontsize=14)
        ax.legend(["Accuracy STD", "F1 STD"])
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, axis="y", alpha=0.3)

        # 4. Лучшие комбинации модель-метод
        ax = axes[1, 1]
        top_combinations = (
            self.results_df.groupby(["model_name", "threshold_method"])["pseudo_f1"]
            .mean()
            .sort_values(ascending=False)
            .head(15)
        )

        top_combinations.plot(kind="barh", ax=ax, color="skyblue", edgecolor="navy")
        ax.set_xlabel("F1-Score", fontsize=12)
        ax.set_ylabel("Модель + Метод", fontsize=12)
        ax.set_title("Top 15 комбинаций Модель-Метод по F1-Score", fontsize=14)
        ax.grid(True, axis="x", alpha=0.3)

        plt.suptitle("Анализ производительности моделей", fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig(self.output_dir / "model_performance.png", dpi=300, bbox_inches="tight")
        plt.close()

    def plot_threshold_distribution(self):
        """Распределение значений порогов."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

        # 1. Гистограмма всех порогов
        ax = axes[0, 0]
        ax.hist(self.results_df["threshold_value"], bins=50, alpha=0.7, color="purple", edgecolor="black")
        ax.axvline(
            self.results_df["threshold_value"].mean(),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f'Среднее: {self.results_df["threshold_value"].mean():.3f}',
        )
        ax.set_xlabel("Значение порога", fontsize=12)
        ax.set_ylabel("Частота", fontsize=12)
        ax.set_title("Распределение всех порогов", fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Пороги по типам методов
        ax = axes[0, 1]
        method_types = {
            "Оптимизационные": ["Optimal F1", "Youden J Statistic", "Cost Sensitive"],
            "Процентильные": [m for m in self.results_df["threshold_method"].unique() if "Percentile" in m],
            "Фиксированные": [m for m in self.results_df["threshold_method"].unique() if "Fixed" in m],
            "Энтропийные": [m for m in self.results_df["threshold_method"].unique() if "Entropy" in m],
            "Margin": [m for m in self.results_df["threshold_method"].unique() if "Margin" in m],
        }

        data_for_plot = []
        labels = []
        for method_type, methods in method_types.items():
            type_data = self.results_df[self.results_df["threshold_method"].isin(methods)][
                "threshold_value"
            ]
            if len(type_data) > 0:
                data_for_plot.append(type_data)
                labels.append(method_type)

        bp = ax.boxplot(data_for_plot, labels=labels, patch_artist=True)
        colors = sns.color_palette("Set2", len(labels))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)

        ax.set_xlabel("Тип метода", fontsize=12)
        ax.set_ylabel("Значение порога", fontsize=12)
        ax.set_title("Распределение порогов по типам методов", fontsize=14)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, axis="y", alpha=0.3)

        # 3. Зависимость порога от размера датасета
        ax = axes[1, 0]
        dataset_sizes = {
            "Mushroom": 8124,
            "Adult": 48842,
            "Spambase": 4601,
            "Bank Marketing": 45211,
            "Credit Card Default": 30000,
            "MAGIC Gamma Telescope": 19020,
            "Gas Sensor Array Drift": 13910,
            "Avila": 20867,
            "Thyroid Disease": 9172,
        }

        for dataset in dataset_sizes:
            dataset_data = self.results_df[self.results_df["dataset_name"] == dataset]
            if len(dataset_data) > 0:
                avg_threshold = dataset_data["threshold_value"].mean()
                ax.scatter(dataset_sizes[dataset], avg_threshold, s=100, alpha=0.7)
                ax.annotate(
                    dataset,
                    (dataset_sizes[dataset], avg_threshold),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                )

        ax.set_xscale("log")
        ax.set_xlabel("Размер датасета (log scale)", fontsize=12)
        ax.set_ylabel("Средний порог", fontsize=12)
        ax.set_title("Зависимость порога от размера датасета", fontsize=14)
        ax.grid(True, alpha=0.3)

        # 4. Корреляция порога с метриками
        ax = axes[1, 1]
        correlations = self.results_df[
            ["threshold_value", "pseudo_accuracy", "pseudo_f1", "selection_rate"]
        ].corr()

        sns.heatmap(
            correlations,
            annot=True,
            fmt=".3f",
            cmap="coolwarm",
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={"shrink": 0.8},
            ax=ax,
        )

        ax.set_title("Корреляция порога с метриками", fontsize=14)

        plt.suptitle("Анализ распределения порогов", fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig(self.output_dir / "threshold_distribution.png", dpi=300, bbox_inches="tight")
        plt.close()

    def plot_time_analysis(self):
        """Анализ временных характеристик."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # 1. Время обучения по моделям и датасетам
        ax = axes[0]
        pivot_time = self.results_df.pivot_table(
            values="model_train_time",
            index="model_name",
            columns="dataset_name",
            aggfunc="mean",
        )

        sns.heatmap(
            pivot_time,
            annot=True,
            fmt=".2f",
            cmap="YlOrRd",
            cbar_kws={"label": "Время (сек)"},
            ax=ax,
        )

        ax.set_title("Среднее время обучения (сек)", fontsize=14)
        ax.set_xlabel("Датасет", fontsize=12)
        ax.set_ylabel("Модель", fontsize=12)

        # 2. Эффективность: качество / время
        ax = axes[1]
        efficiency_data = self.results_df.groupby("model_name").agg(
            {"pseudo_f1": "mean", "model_train_time": "mean"}
        )
        efficiency_data["efficiency"] = efficiency_data["pseudo_f1"] / (
            efficiency_data["model_train_time"] + 0.01
        )

        efficiency_data["efficiency"].sort_values(ascending=True).plot(
            kind="barh", ax=ax, color="lightgreen", edgecolor="darkgreen"
        )

        ax.set_xlabel("Эффективность (F1 / время)", fontsize=12)
        ax.set_ylabel("Модель", fontsize=12)
        ax.set_title("Эффективность моделей", fontsize=14)
        ax.grid(True, axis="x", alpha=0.3)

        plt.suptitle("Анализ временных характеристик", fontsize=16)
        plt.tight_layout()
        plt.savefig(self.output_dir / "time_analysis.png", dpi=300, bbox_inches="tight")
        plt.close()

    def plot_binary_classification_analysis(self):
        """Детальный анализ для бинарной классификации."""
        binary_data = self.results_df[
            self.results_df["dataset_name"].isin(["Mushroom", "Adult", "Spambase", "Bank Marketing", "Credit Card Default", "MAGIC Gamma Telescope"])
        ]

        if len(binary_data) == 0:
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. ROC-подобная кривая: Recall vs Precision
        ax = axes[0, 0]
        for method in ["Optimal F1", "Youden J Statistic", "Cost Sensitive", "Percentile 80%", "Fixed 0.9"]:
            method_data = binary_data[binary_data["threshold_method"] == method]
            if len(method_data) > 0:
                grouped = method_data.groupby("threshold_value").agg(
                    {"pseudo_recall": "mean", "pseudo_precision": "mean"}
                )
                if len(grouped) > 1:
                    ax.plot(
                        grouped["pseudo_recall"],
                        grouped["pseudo_precision"],
                        marker="o",
                        label=method,
                        linewidth=2,
                        markersize=8,
                    )

        ax.set_xlabel("Recall", fontsize=12)
        ax.set_ylabel("Precision", fontsize=12)
        ax.set_title("Precision-Recall Trade-off", fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Влияние дисбаланса классов
        ax = axes[0, 1]
        # Примерные показатели дисбаланса
        imbalance_ratios = {
            "Mushroom": 1.0,  # Сбалансирован
            "Adult": 3.17,
            "Spambase": 1.5,
            "Bank Marketing": 7.85,
            "Credit Card Default": 3.52,
            "MAGIC Gamma Telescope": 1.86,
        }

        for dataset in imbalance_ratios:
            dataset_data = binary_data[binary_data["dataset_name"] == dataset]
            if len(dataset_data) > 0:
                avg_f1 = dataset_data["pseudo_f1"].mean()
                ax.scatter(imbalance_ratios[dataset], avg_f1, s=200, alpha=0.7)
                ax.annotate(
                    dataset,
                    (imbalance_ratios[dataset], avg_f1),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=9,
                )

        ax.set_xlabel("Степень дисбаланса классов", fontsize=12)
        ax.set_ylabel("Средний F1-Score", fontsize=12)
        ax.set_title("Влияние дисбаланса на качество псевдо-разметки", fontsize=14)
        ax.grid(True, alpha=0.3)

        # 3. Сравнение оптимизационных методов
        ax = axes[1, 0]
        opt_methods = ["Optimal F1", "Youden J Statistic", "Cost Sensitive"]
        opt_data = binary_data[binary_data["threshold_method"].isin(opt_methods)]

        metrics_comparison = opt_data.groupby("threshold_method")[
            ["pseudo_accuracy", "pseudo_f1", "pseudo_precision", "pseudo_recall"]
        ].mean()

        metrics_comparison.T.plot(kind="bar", ax=ax, width=0.8)
        ax.set_xlabel("Метрика", fontsize=12)
        ax.set_ylabel("Значение", fontsize=12)
        ax.set_title("Сравнение оптимизационных методов", fontsize=14)
        ax.legend(title="Метод")
        ax.tick_params(axis="x", rotation=0)
        ax.grid(True, axis="y", alpha=0.3)

        # 4. Анализ ошибок
        ax = axes[1, 1]
        # Вычисляем примерные FP и FN rates
        binary_data_copy = binary_data.copy()
        binary_data_copy["error_rate"] = 1 - binary_data_copy["pseudo_accuracy"]
        binary_data_copy["fp_proxy"] = (1 - binary_data_copy["pseudo_precision"]) * binary_data_copy[
            "selection_rate"
        ]
        binary_data_copy["fn_proxy"] = (1 - binary_data_copy["pseudo_recall"]) * (
            1 - binary_data_copy["selection_rate"]
        )

        error_analysis = binary_data_copy.groupby("threshold_method")[
            ["error_rate", "fp_proxy", "fn_proxy"]
        ].mean()
        error_analysis = error_analysis.sort_values("error_rate").head(10)

        error_analysis.plot(kind="bar", ax=ax, width=0.8)
        ax.set_xlabel("Метод", fontsize=12)
        ax.set_ylabel("Показатель ошибки", fontsize=12)
        ax.set_title("Анализ ошибок (Top 10 методов)", fontsize=14)
        ax.legend(["Общая ошибка", "Proxy FP rate", "Proxy FN rate"])
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, axis="y", alpha=0.3)

        plt.suptitle("Детальный анализ для бинарной классификации", fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig(self.output_dir / "binary_classification_analysis.png", dpi=300, bbox_inches="tight")
        plt.close()

    def plot_multiclass_analysis(self):
        """Детальный анализ для многоклассовой классификации."""
        multiclass_data = self.results_df[
            self.results_df["dataset_name"].isin(["Gas Sensor Array Drift", "Avila", "Thyroid Disease"])
        ]

        if len(multiclass_data) == 0:
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Эффективность методов по числу классов
        ax = axes[0, 0]
        n_classes_map = {
            "Gas Sensor Array Drift": 6,
            "Avila": 12,
            "Thyroid Disease": 3
        }

        for dataset, n_classes in n_classes_map.items():
            dataset_data = multiclass_data[multiclass_data["dataset_name"] == dataset]
            if len(dataset_data) > 0:
                # Топ-5 методов для датасета
                top_methods = dataset_data.groupby("threshold_method")["pseudo_f1"].mean().nlargest(5)

                x_offset = n_classes + np.linspace(-0.3, 0.3, len(top_methods))
                for idx, (method, f1) in enumerate(top_methods.items()):
                    ax.scatter(x_offset[idx], f1, s=100, alpha=0.7, label=method if n_classes == 3 else None)

        ax.set_xlabel("Количество классов", fontsize=12)
        ax.set_ylabel("F1-Score", fontsize=12)
        ax.set_title("Эффективность методов vs количество классов", fontsize=14)
        ax.set_xticks([3, 10])
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)

        # 2. Сравнение энтропийных и margin методов
        ax = axes[0, 1]
        entropy_margin_data = multiclass_data[
            multiclass_data["threshold_method"].str.contains("Entropy|Margin")
        ]

        if len(entropy_margin_data) > 0:
            comparison = entropy_margin_data.groupby(["dataset_name", "threshold_method"])[
                ["pseudo_f1", "selection_rate"]
            ].mean()

            # Разделяем на энтропийные и margin
            for dataset in n_classes_map:
                dataset_comp = comparison.loc[dataset] if dataset in comparison.index else None
                if dataset_comp is not None:
                    entropy_methods = dataset_comp[dataset_comp.index.str.contains("Entropy")]
                    margin_methods = dataset_comp[dataset_comp.index.str.contains("Margin")]

                    if len(entropy_methods) > 0:
                        ax.scatter(
                            entropy_methods["selection_rate"].mean(),
                            entropy_methods["pseudo_f1"].mean(),
                            s=200,
                            marker="o",
                            label=f"{dataset} (Entropy)",
                        )

                    if len(margin_methods) > 0:
                        ax.scatter(
                            margin_methods["selection_rate"].mean(),
                            margin_methods["pseudo_f1"].mean(),
                            s=200,
                            marker="s",
                            label=f"{dataset} (Margin)",
                        )

        ax.set_xlabel("Доля отобранных примеров", fontsize=12)
        ax.set_ylabel("F1-Score", fontsize=12)
        ax.set_title("Энтропийные vs Margin методы", fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Влияние числа классов на оптимальный порог
        ax = axes[1, 0]
        threshold_by_classes = []
        for dataset, n_classes in n_classes_map.items():
            dataset_data = multiclass_data[multiclass_data["dataset_name"] == dataset]
            if len(dataset_data) > 0:
                avg_threshold = dataset_data["threshold_value"].mean()
                threshold_by_classes.append((n_classes, avg_threshold, dataset))

        if threshold_by_classes:
            threshold_by_classes.sort(key=lambda x: x[0])
            x_vals = [item[0] for item in threshold_by_classes]
            y_vals = [item[1] for item in threshold_by_classes]
            labels = [item[2] for item in threshold_by_classes]

            ax.plot(x_vals, y_vals, marker="o", markersize=12, linewidth=2)
            for x, y, label in zip(x_vals, y_vals, labels):
                ax.annotate(label, (x, y), xytext=(5, 5), textcoords="offset points", fontsize=10)

        ax.set_xlabel("Количество классов", fontsize=12)
        ax.set_ylabel("Средний порог", fontsize=12)
        ax.set_title("Зависимость порога от числа классов", fontsize=14)
        ax.grid(True, alpha=0.3)

        # 4. Матрица методов для многоклассовой классификации
        ax = axes[1, 1]
        multiclass_methods = ["Entropy < 0.5", "Entropy < 1.0", "Margin > 0.1", "Margin > 0.2", "Margin > 0.3"]
        method_matrix = multiclass_data[multiclass_data["threshold_method"].isin(multiclass_methods)]

        if len(method_matrix) > 0:
            pivot = method_matrix.pivot_table(
                values="pseudo_f1",
                index="threshold_method",
                columns="dataset_name",
                aggfunc="mean",
            )

            sns.heatmap(
                pivot,
                annot=True,
                fmt=".3f",
                cmap="RdYlGn",
                cbar_kws={"label": "F1-Score"},
                ax=ax,
            )

        ax.set_title("Производительность специализированных методов", fontsize=14)
        ax.set_xlabel("Датасет", fontsize=12)
        ax.set_ylabel("Метод", fontsize=12)

        plt.suptitle("Детальный анализ для многоклассовой классификации", fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig(self.output_dir / "multiclass_analysis.png", dpi=300, bbox_inches="tight")
        plt.close()

    def create_summary_dashboard(self):
        """Создание сводного дашборда с ключевыми результатами."""
        fig = plt.figure(figsize=(20, 24))

        # Создаем сетку для размещения графиков
        gs = fig.add_gridspec(6, 3, hspace=0.3, wspace=0.3)

        # 1. Заголовок
        fig.suptitle("Сводный дашборд: Эксперимент по выбору порогов вероятности", fontsize=20, y=0.98)

        # 2. Общая статистика (текстовый блок)
        ax_stats = fig.add_subplot(gs[0, :])
        ax_stats.axis("off")

        total_experiments = len(self.results_df)
        n_datasets = self.results_df["dataset_name"].nunique()
        n_models = self.results_df["model_name"].nunique()
        n_methods = self.results_df["threshold_method"].nunique()

        stats_text = f"""
        Общая статистика эксперимента:
        • Всего экспериментов: {total_experiments:,}
        • Датасетов: {n_datasets}
        • Моделей: {n_models}
        • Методов выбора порога: {n_methods}
        
        Лучшие результаты:
        • Лучший F1-Score: {self.results_df['pseudo_f1'].max():.3f}
        • Лучшая Accuracy: {self.results_df['pseudo_accuracy'].max():.3f}
        • Средняя доля отобранных примеров: {self.results_df['selection_rate'].mean():.1%}
        """

        ax_stats.text(0.1, 0.5, stats_text, fontsize=14, verticalalignment="center", family="monospace")

        # 3. Top-5 методов по F1
        ax_top5 = fig.add_subplot(gs[1, 0])
        top5_f1 = self.results_df.groupby("threshold_method")["pseudo_f1"].mean().nlargest(5)
        top5_f1.plot(kind="barh", ax=ax_top5, color="lightcoral")
        ax_top5.set_title("Top-5 методов по F1-Score", fontsize=14)
        ax_top5.set_xlabel("F1-Score")

        # 4. Top-5 моделей по F1
        ax_models = fig.add_subplot(gs[1, 1])
        top5_models = self.results_df.groupby("model_name")["pseudo_f1"].mean().nlargest(5)
        top5_models.plot(kind="barh", ax=ax_models, color="lightblue")
        ax_models.set_title("Top-5 моделей по F1-Score", fontsize=14)
        ax_models.set_xlabel("F1-Score")

        # 5. Распределение порогов
        ax_dist = fig.add_subplot(gs[1, 2])
        ax_dist.hist(
            self.results_df["threshold_value"],
            bins=30,
            alpha=0.7,
            color="lightgreen",
            edgecolor="black",
        )
        ax_dist.set_title("Распределение порогов", fontsize=14)
        ax_dist.set_xlabel("Порог")
        ax_dist.set_ylabel("Частота")

        # 6. Scatter: Selection Rate vs F1
        ax_scatter = fig.add_subplot(gs[2, :2])
        scatter = ax_scatter.scatter(
            self.results_df["selection_rate"],
            self.results_df["pseudo_f1"],
            c=self.results_df["pseudo_accuracy"],
            s=30,
            alpha=0.5,
            cmap="viridis",
        )
        plt.colorbar(scatter, ax=ax_scatter, label="Accuracy")
        ax_scatter.set_xlabel("Доля отобранных примеров")
        ax_scatter.set_ylabel("F1-Score")
        ax_scatter.set_title("Trade-off: Количество vs Качество (все эксперименты)", fontsize=14)

        # 7. Лучшие комбинации
        ax_best = fig.add_subplot(gs[2, 2])
        best_combinations = (
            self.results_df.groupby(["model_name", "threshold_method"])["pseudo_f1"]
            .mean()
            .nlargest(10)
        )
        best_combinations.plot(kind="barh", ax=ax_best, color="gold")
        ax_best.set_title("Top-10 комбинаций", fontsize=14)
        ax_best.set_xlabel("F1-Score")

        # 8. Heatmap: методы vs датасеты
        ax_heatmap1 = fig.add_subplot(gs[3:5, :])
        pivot_f1 = self.results_df.pivot_table(
            values="pseudo_f1",
            index="threshold_method",
            columns="dataset_name",
            aggfunc="mean",
        )
        sns.heatmap(
            pivot_f1.loc[pivot_f1.mean(axis=1).nlargest(15).index],
            annot=True,
            fmt=".2f",
            cmap="RdYlGn",
            cbar_kws={"label": "F1-Score"},
            ax=ax_heatmap1,
        )
        ax_heatmap1.set_title("F1-Score: Top-15 методов × датасеты", fontsize=14)

        # 9. Выводы и рекомендации
        ax_conclusions = fig.add_subplot(gs[5, :])
        ax_conclusions.axis("off")

        # Находим лучшие методы для разных сценариев
        high_quality = (
            self.results_df[self.results_df["pseudo_f1"] > 0.9]
            .groupby("threshold_method")
            .size()
            .nlargest(3)
        )
        high_coverage = (
            self.results_df[self.results_df["selection_rate"] > 0.5]
            .groupby("threshold_method")["pseudo_f1"]
            .mean()
            .nlargest(3)
        )

        conclusions_text = f"""
        Ключевые выводы и рекомендации:
        
        1. Для максимального качества (F1 > 0.9):
           • {', '.join(high_quality.index[:3])}
        
        2. Для баланса качества и покрытия:
           • {', '.join(high_coverage.index[:3])}
        
        3. Универсальные методы:
           • Percentile 80-90% показывают стабильные результаты
           • Fixed thresholds хороши для быстрого baseline
        
        4. Рекомендации по выбору:
           • Критичные задачи → Fixed 0.95+ или MC Dropout
           • Сбалансированные данные → Optimal F1
           • Несбалансированные → Youden J или Cost Sensitive
           • Многоклассовая → Entropy или Margin методы
        """

        ax_conclusions.text(
            0.05,
            0.5,
            conclusions_text,
            fontsize=12,
            verticalalignment="center",
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.5),
        )

        plt.tight_layout()
        plt.savefig(self.output_dir / "summary_dashboard.png", dpi=300, bbox_inches="tight")
        plt.close()


def generate_experiment_report(results_df: pd.DataFrame, output_path: Path):
    """Генерация текстового отчета по эксперименту."""
    report = []
    report.append("=" * 80)
    report.append("ОТЧЕТ ПО ЭКСПЕРИМЕНТУ: СРАВНЕНИЕ МЕТОДОВ ВЫБОРА ПОРОГОВ")
    report.append("=" * 80)
    report.append("")

    # 1. Общая информация
    report.append("1. ОБЩАЯ ИНФОРМАЦИЯ")
    report.append("-" * 40)
    report.append(f"Всего экспериментов: {len(results_df):,}")
    report.append(f"Датасетов: {results_df['dataset_name'].nunique()}")
    report.append(f"Моделей: {results_df['model_name'].nunique()}")
    report.append(f"Методов: {results_df['threshold_method'].nunique()}")
    report.append("")

    # 2. Лучшие результаты
    report.append("2. ЛУЧШИЕ РЕЗУЛЬТАТЫ")
    report.append("-" * 40)

    best_f1 = results_df.loc[results_df["pseudo_f1"].idxmax()]
    report.append(f"Лучший F1-Score: {best_f1['pseudo_f1']:.3f}")
    report.append(
        f"  → Метод: {best_f1['threshold_method']}, "
        f"Модель: {best_f1['model_name']}, "
        f"Датасет: {best_f1['dataset_name']}"
    )

    best_acc = results_df.loc[results_df["pseudo_accuracy"].idxmax()]
    report.append(f"\nЛучшая Accuracy: {best_acc['pseudo_accuracy']:.3f}")
    report.append(
        f"  → Метод: {best_acc['threshold_method']}, "
        f"Модель: {best_acc['model_name']}, "
        f"Датасет: {best_acc['dataset_name']}"
    )
    report.append("")

    # 3. Рейтинг методов
    report.append("3. РЕЙТИНГ МЕТОДОВ (по среднему F1-Score)")
    report.append("-" * 40)

    method_ranking = results_df.groupby("threshold_method")["pseudo_f1"].agg(["mean", "std"]).sort_values(
        "mean", ascending=False
    )

    for i, (method, row) in enumerate(method_ranking.head(10).iterrows(), 1):
        report.append(f"{i:2d}. {method:<30} F1: {row['mean']:.3f} (±{row['std']:.3f})")
    report.append("")

    # 4. Рейтинг моделей
    report.append("4. РЕЙТИНГ МОДЕЛЕЙ (по среднему F1-Score)")
    report.append("-" * 40)

    model_ranking = results_df.groupby("model_name")["pseudo_f1"].agg(["mean", "std"]).sort_values(
        "mean", ascending=False
    )

    for i, (model, row) in enumerate(model_ranking.iterrows(), 1):
        report.append(f"{i}. {model:<20} F1: {row['mean']:.3f} (±{row['std']:.3f})")
    report.append("")

    # 5. Анализ по типам задач
    report.append("5. АНАЛИЗ ПО ТИПАМ ЗАДАЧ")
    report.append("-" * 40)

    binary_datasets = ["Mushroom", "Adult", "Spambase", "Bank Marketing", "Credit Card Default", "MAGIC Gamma Telescope"]
    binary_data = results_df[results_df["dataset_name"].isin(binary_datasets)]
    multiclass_data = results_df[~results_df["dataset_name"].isin(binary_datasets)]

    report.append("Бинарная классификация:")
    binary_best = binary_data.groupby("threshold_method")["pseudo_f1"].mean().nlargest(5)
    for method, f1 in binary_best.items():
        report.append(f"  • {method}: {f1:.3f}")

    report.append("\nМногоклассовая классификация:")
    multi_best = multiclass_data.groupby("threshold_method")["pseudo_f1"].mean().nlargest(5)
    for method, f1 in multi_best.items():
        report.append(f"  • {method}: {f1:.3f}")
    report.append("")

    # 6. Практические рекомендации
    report.append("6. ПРАКТИЧЕСКИЕ РЕКОМЕНДАЦИИ")
    report.append("-" * 40)
    report.append("• Для задач с высокими требованиями к качеству:")
    report.append("  → Используйте Fixed 0.95 или выше")
    report.append("  → Рассмотрите Optimal F1 с валидацией")
    report.append("")
    report.append("• Для сбалансированных датасетов:")
    report.append("  → Optimal F1 показывает лучшие результаты")
    report.append("  → Percentile 80-85% - хороший baseline")
    report.append("")
    report.append("• Для несбалансированных датасетов:")
    report.append("  → Youden J Statistic учитывает баланс классов")
    report.append("  → Cost Sensitive при известной стоимости ошибок")
    report.append("")
    report.append("• Для многоклассовой классификации:")
    report.append("  → Entropy методы эффективны при 10+ классах")
    report.append("  → Margin методы хороши для похожих классов")
    report.append("")

    # Сохранение отчета
    report_path = output_path / "experiment_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print(f"Текстовый отчет сохранен: {report_path}")

    return "\n".join(report)
