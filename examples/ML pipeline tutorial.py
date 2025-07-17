"""
Titanic Survival Prediction Tutorial
Полный цикл обработки данных с использованием кастомной ML-библиотеки
"""

# Импорт необходимых библиотек
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from dmdslab.data import ModelData, DataSplit
from dmdslab.encoding import CategoricalEncoder
from dmdslab.imputing import DataImputer
from dmdslab.pipeline import MLPipeline
from dmdslab.scaling import DataScaler
from dmdslab.splitting import split_dataset

# Константы
RANDOM_STATE = 42
DATA_URL = (
    "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
)
PIPELINE_PATH = "titanic_pipeline.joblib"


def load_and_preprocess_data() -> ModelData:
    """
    Загрузка и предварительная обработка данных
    Возвращает объект ModelData с признаками и целевой переменной
    """
    # Загрузка данных
    df = pd.read_csv(DATA_URL)

    # Выбор и переименование колонок
    df = df.rename(
        columns={
            "Survived": "target",
            "Pclass": "class",
            "Sex": "sex",
            "Age": "age",
            "Fare": "fare",
            "Embarked": "embarked",
        }
    )

    # Фильтрация и очистка данных
    df = df[["target", "class", "sex", "age", "fare", "embarked"]]
    df = df.dropna().reset_index(drop=True)

    # Создание ModelData
    return ModelData(features=df.drop("target", axis=1), target=df["target"])


def create_data_split(data: ModelData) -> DataSplit:
    """
    Разделение данных на тренировочные, валидационные и тестовые наборы
    """
    return split_dataset(
        data=data,
        test_size=0.2,
        tuning_size=0.1,
        random_state=RANDOM_STATE,
        stratify=True,
    )


def create_pipeline() -> MLPipeline:
    """
    Создание и конфигурация ML пайплайна
    """
    return MLPipeline(
        steps=[
            (
                "imputer",
                DataImputer(num_strategy="median", cat_strategy="most_frequent"),
            ),
            (
                "encoder",
                CategoricalEncoder(
                    method="catboost",
                    cols=["sex", "embarked"],
                    catboost_a=5.0,
                    random_state=RANDOM_STATE,
                ),
            ),
            (
                "scaler",
                DataScaler(scaler_type="standard", numeric_features=["age", "fare"]),
            ),
        ]
    )


def train_model(train_data: ModelData) -> RandomForestClassifier:
    """
    Обучение модели Random Forest
    """
    model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    model.fit(train_data.features, train_data.target)
    return model


def evaluate_model(model: RandomForestClassifier, data: ModelData) -> float:
    """
    Оценка точности модели на указанном наборе данных
    """
    pred = model.predict(data.features)
    return accuracy_score(data.target, pred)


def main():
    # Шаг 1: Загрузка данных
    raw_data = load_and_preprocess_data()
    print("Данные успешно загружены. Пример записей:")
    print(raw_data.features.head(3))

    # Шаг 2: Разделение данных
    data_split = create_data_split(raw_data)
    print(f"\nРазмеры наборов:")
    print(f"Train: {len(data_split.train.features)} записей")
    print(f"Validation: {len(data_split.tuning.features)} записей")
    print(f"Test: {len(data_split.test.features)} записей")

    # Шаг 3: Создание и применение пайплайна
    pipeline = create_pipeline()
    processed_data = pipeline.fit_transform(data_split)

    # Визуализация пайплайна
    pipeline.visualize().render("titanic_pipeline", format="png", view=True)
    print("\nПайплайн визуализирован в файле 'titanic_pipeline.png'")

    # Шаг 4: Обучение модели
    model = train_model(processed_data.train)

    # Шаг 5: Оценка модели
    print("\nОценка модели:")
    print(f"Train Accuracy: {evaluate_model(model, processed_data.train):.3f}")
    print(f"Validation Accuracy: {evaluate_model(model, processed_data.tuning):.3f}")
    print(f"Test Accuracy: {evaluate_model(model, processed_data.test):.3f}")

    # Шаг 6: Сохранение пайплайна
    pipeline.save(PIPELINE_PATH)
    print(f"\nПайплайн сохранен в файл: {PIPELINE_PATH}")

    # Пример использования пайплайна
    example_passenger = pd.DataFrame(
        [{"class": 2, "sex": "male", "age": 30, "fare": 25.0, "embarked": "S"}]
    )

    loaded_pipeline = MLPipeline.load(PIPELINE_PATH)
    processed_example = loaded_pipeline.transform(
        DataSplit(
            train=ModelData(example_passenger, pd.Series([1])), tuning=None, test=None
        )
    ).train.features

    print("\nПример преобразования новых данных:")
    print(processed_example)


if __name__ == "__main__":
    main()
