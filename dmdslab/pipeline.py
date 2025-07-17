from sklearn.base import BaseEstimator, TransformerMixin, clone
from joblib import dump, load
import pandas as pd
import graphviz
import html
from typing import Optional, Dict, Any
from sklearn.exceptions import NotFittedError
from dataclasses import dataclass, field

from dmdslab.data import ModelData, DataSplit


class MLPipeline(BaseEstimator):
    """Пайплайн с полной поддержкой ModelData и DataSplit"""

    def __init__(self, steps):
        """
        Инициализация пайплайна
        :param steps: список кортежей (name, transformer)
        """
        self.steps = steps
        self.named_steps = dict(steps)
        self.fitted_steps = []
        self.metadata = {}

    def _apply_transform(self, data: ModelData, mode: str = "transform") -> ModelData:
        """Внутренний метод для применения преобразований"""
        if data is None:
            return None

        transformed = ModelData(
            features=data.features.copy(),
            target=data.target.copy(),
        )

        for name, transformer in self.fitted_steps:
            if mode == "fit":
                transformer.fit(transformed)
            transformed = transformer.transform(transformed)
            transformed.metadata[f"pipeline_{name}"] = transformer.__dict__.copy()

        return transformed

    def fit(self, data: DataSplit) -> "MLPipeline":
        """Обучение пайплайна на тренировочных данных"""
        self.fitted_steps = []
        current_data = data.train

        for name, transformer in self.steps:
            cloned_transformer = clone(transformer)

            # Обучение только на тренировочных данных
            cloned_transformer.fit(current_data)

            # Применение преобразования ко всем данным
            transformed_train = cloned_transformer.transform(data.train)
            transformed_test = (
                cloned_transformer.transform(data.test) if data.test else None
            )
            transformed_tuning = (
                cloned_transformer.transform(data.tuning) if data.tuning else None
            )

            current_data = transformed_train
            self.fitted_steps.append((name, cloned_transformer))

            # Сохранение состояния данных
            data = DataSplit(
                train=transformed_train,
                test=transformed_test,
                tuning=transformed_tuning,
                split_params=data.split_params,
            )

        self.metadata = {
            "feature_names": data.train.features.columns.tolist(),
            "target_name": data.train.target.name,
            "steps_config": [t.get_params() for _, t in self.fitted_steps],
        }
        return self

    def transform(self, data: DataSplit) -> DataSplit:
        """Применение преобразований ко всем наборам"""
        if not self.fitted_steps:
            raise NotFittedError("Pipeline not fitted yet")

        return DataSplit(
            train=self._apply_transform(data.train),
            test=self._apply_transform(data.test),
            tuning=self._apply_transform(data.tuning),
            split_params=data.split_params,
        )

    def fit_transform(self, data: DataSplit) -> DataSplit:
        """Обучение и преобразование за один шаг"""
        return self.fit(data).transform(data)

    def __getitem__(self, key: str) -> TransformerMixin:
        """Доступ к шагам преобразований по имени"""
        return dict(self.fitted_steps)[key]

    def visualize(self) -> graphviz.Digraph:
        """Визуализация структуры пайплайна"""
        dot = graphviz.Digraph()
        for i, (name, est) in enumerate(self.fitted_steps):
            label = f"<B>{name}</B><BR/>{est.__class__.__name__}"
            dot.node(str(i), label=label)
            if i > 0:
                dot.edge(str(i - 1), str(i))
        return dot

    def save(self, path: str) -> None:
        """Сохранение пайплайна с метаданными"""
        dump({"steps": self.fitted_steps, "metadata": self.metadata}, path)

    @classmethod
    def load(cls, path: str) -> "MLPipeline":
        """Загрузка пайплайна с восстановлением состояния"""
        data = load(path)
        pipeline = cls(data["steps"])
        pipeline.fitted_steps = data["steps"]
        pipeline.metadata = data["metadata"]
        return pipeline

    def get_feature_names(self) -> list:
        """Получение имен финальных признаков"""
        return self.metadata.get("feature_names", [])

    def _repr_html_(self) -> str:
        """Визуализация для Jupyter Notebook"""
        html_str = (
            "<div style='border: 2px solid #eee; padding: 10px; border-radius: 5px'>"
        )
        for i, (name, est) in enumerate(self.fitted_steps):
            params = "<BR/>".join(f"{k}: {v}" for k, v in est.get_params().items())
            html_str += f"""
            <div style='margin: 5px; padding: 10px; border: 1px solid #ddd; border-radius: 3px'>
                <b>Step {i+1}: {name}</b><hr/>
                <b>Type:</b> {est.__class__.__name__}<br/>
                <b>Parameters:</b><br/>{html.escape(params)}
            </div>
            """
        html_str += "</div>"
        return html_str

    def __repr__(self) -> str:
        return f"MLPipeline(steps={[name for name, _ in self.steps]})"
