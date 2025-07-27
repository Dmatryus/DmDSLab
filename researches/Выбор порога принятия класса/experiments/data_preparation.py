"""
Модуль подготовки данных для эксперимента по сравнению методов выбора порогов
для псевдо-разметки в задачах полуконтролируемого обучения.

Использует UCIDatasetManager из библиотеки DmDSLab для загрузки датасетов.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from category_encoders import CatBoostEncoder
from typing import Dict, Tuple, List, Optional
import warnings

warnings.filterwarnings("ignore")

# Предполагаем, что DmDSLab установлен
try:
    from DmDSLab import UCIDatasetManager

    DMDSLAB_AVAILABLE = True
except ImportError:
    print("Библиотека DmDSLab не найдена. Будет использован прямой загрузчик UCI.")
    DMDSLAB_AVAILABLE = False

# Импортируем наш альтернативный загрузчик
from uci_direct_loader import UCIDirectLoader


class DatasetPreparer:
    """Класс для подготовки датасетов к экспериментам."""

    def __init__(self, random_state: int = 42):
        """
        Инициализация препарера данных.

        Args:
            random_state: Seed для воспроизводимости результатов
        """
        self.random_state = random_state

        # Инициализация загрузчиков
        if DMDSLAB_AVAILABLE:
            self.dataset_manager = UCIDatasetManager()
        else:
            self.dataset_manager = None

        self.direct_loader = UCIDirectLoader()

        # Маппинг датасетов и их идентификаторов в UCI репозитории
        self.binary_datasets = {
            "breast_cancer": "breast-cancer-wisconsin",
            "heart_disease": "heart-disease",
            "bank_marketing": "bank-marketing",
            "adult_income": "adult",
            "credit_default": "default-of-credit-card-clients",
        }

        self.multiclass_datasets = {
            "iris": "iris",
            "wine_quality": "wine-quality",
            "cover_type": "covertype",
            "letter_recognition": "letter-recognition",
            "satellite": "statlog-landsat-satellite",
        }

    def load_dataset(
        self, dataset_name: str, dataset_type: str = "binary"
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Загрузка датасета через UCIDatasetManager или альтернативный загрузчик.

        Args:
            dataset_name: Название датасета
            dataset_type: Тип задачи ('binary' или 'multiclass')

        Returns:
            X: Признаки
            y: Целевая переменная
        """
        # Определяем UCI идентификатор
        if dataset_type == "binary":
            uci_name = self.binary_datasets.get(dataset_name)
        else:
            uci_name = self.multiclass_datasets.get(dataset_name)

        if not uci_name:
            raise ValueError(f"Датасет {dataset_name} не найден в списке доступных")

        print(f"Загрузка датасета {dataset_name} (UCI: {uci_name})...")

        # Сначала пытаемся загрузить через прямой загрузчик
        try:
            X, y = self.direct_loader.load_dataset(dataset_name)
            print(f"Датасет успешно загружен через прямой загрузчик")
            return X, y
        except Exception as e:
            print(f"Ошибка при загрузке через прямой загрузчик: {e}")

        # Если есть DmDSLab, пытаемся через него
        if DMDSLAB_AVAILABLE and self.dataset_manager:
            try:
                dataset = self.dataset_manager.load_dataset(uci_name)
                X = dataset.drop("target", axis=1)
                y = dataset["target"]
                print(f"Датасет успешно загружен через DmDSLab")
                return X, y
            except Exception as e:
                print(f"Ошибка при загрузке через DmDSLab: {e}")

        # Если оба способа не сработали, генерируем синтетические данные
        print("Генерация синтетических данных...")
        return self._load_dataset_alternative(dataset_name, dataset_type)

    def _load_dataset_alternative(
        self, dataset_name: str, dataset_type: str
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Альтернативный способ загрузки датасетов (если UCIDatasetManager не сработал).
        """
        # Здесь можно реализовать прямую загрузку с UCI репозитория
        # Для примера возвращаем синтетические данные
        print(f"Генерация синтетических данных для {dataset_name}...")

        if dataset_type == "binary":
            from sklearn.datasets import make_classification

            X, y = make_classification(
                n_samples=1000,
                n_features=20,
                n_informative=15,
                n_redundant=5,
                n_classes=2,
                random_state=self.random_state,
            )
        else:
            from sklearn.datasets import make_classification

            n_classes = {
                "iris": 3,
                "wine_quality": 7,
                "cover_type": 7,
                "letter_recognition": 26,
                "satellite": 6,
            }.get(dataset_name, 4)
            X, y = make_classification(
                n_samples=1000,
                n_features=20,
                n_informative=15,
                n_redundant=5,
                n_classes=n_classes,
                random_state=self.random_state,
            )

        X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        y = pd.Series(y, name="target")

        return X, y

    def preprocess_data(
        self, X: pd.DataFrame, y: pd.Series, task_type: str = "binary"
    ) -> Tuple[pd.DataFrame, pd.Series, Dict]:
        """
        Предобработка данных: обработка пропусков, кодирование целевой переменной.

        Важно: Этот метод НЕ кодирует категориальные признаки в X!
        Для этого используется CatBoostEncoder отдельно после создания разбиений.

        Args:
            X: Признаки
            y: Целевая переменная
            task_type: Тип задачи

        Returns:
            X_processed: Обработанные признаки (с заполненными пропусками)
            y_processed: Обработанная целевая переменная (числовые метки)
            preprocessing_info: Информация о предобработке
        """
        preprocessing_info = {
            "original_shape": X.shape,
            "n_classes": len(np.unique(y)),
            "class_distribution": dict(y.value_counts()),
            "categorical_features": [],
            "numerical_features": [],
        }

        # Определяем типы признаков
        categorical_features = X.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
        numerical_features = X.select_dtypes(include=["number"]).columns.tolist()

        preprocessing_info["categorical_features"] = categorical_features
        preprocessing_info["numerical_features"] = numerical_features

        # Копируем данные
        X_processed = X.copy()
        y_processed = y.copy()

        # 1. ОБРАБОТКА ПРИЗНАКОВ (X)
        # Обработка пропусков в числовых признаках
        if numerical_features:
            num_imputer = SimpleImputer(strategy="mean")
            X_processed[numerical_features] = num_imputer.fit_transform(
                X_processed[numerical_features]
            )

        # Обработка пропусков в категориальных признаках
        if categorical_features:
            cat_imputer = SimpleImputer(strategy="most_frequent")
            X_processed[categorical_features] = cat_imputer.fit_transform(
                X_processed[categorical_features]
            )

        # 2. ОБРАБОТКА ЦЕЛЕВОЙ ПЕРЕМЕННОЙ (y) - используем LabelEncoder
        if y_processed.dtype == "object":
            label_encoder = LabelEncoder()
            y_processed = pd.Series(
                label_encoder.fit_transform(y_processed), index=y.index, name="target"
            )
            preprocessing_info["label_mapping"] = dict(
                zip(
                    label_encoder.classes_,
                    label_encoder.transform(label_encoder.classes_),
                )
            )
            preprocessing_info["label_encoder_used"] = True
        else:
            preprocessing_info["label_encoder_used"] = False

        # Для бинарной классификации убеждаемся, что классы 0 и 1
        if task_type == "binary":
            unique_classes = np.unique(y_processed)
            if len(unique_classes) > 2:
                # Преобразуем в бинарную задачу (первый класс vs остальные)
                y_processed = (y_processed == unique_classes[0]).astype(int)
                preprocessing_info["binary_conversion"] = (
                    f"Class {unique_classes[0]} vs rest"
                )

        preprocessing_info["processed_shape"] = X_processed.shape
        preprocessing_info["final_n_classes"] = len(np.unique(y_processed))

        # Замечание: категориальные признаки в X остаются как есть!
        # Они будут закодированы CatBoostEncoder'ом позже

        return X_processed, y_processed, preprocessing_info

    def create_semi_supervised_split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        val_size: float = 0.2,
        unlabeled_ratio: float = 0.8,
    ) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
        """
        Создание разбиения для полуконтролируемого обучения.

        Args:
            X: Признаки
            y: Целевая переменная
            test_size: Доля тестовой выборки
            val_size: Доля валидационной выборки от обучающей
            unlabeled_ratio: Доля неразмеченных данных от обучающей выборки

        Returns:
            Словарь с разбиениями: {
                'train_labeled': (X, y),
                'train_unlabeled': (X, None),
                'val': (X, y),
                'test': (X, y)
            }
        """
        # Сначала отделяем тестовую выборку
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )

        # Затем отделяем валидационную выборку
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=val_size,
            random_state=self.random_state,
            stratify=y_temp,
        )

        # Создаем полуконтролируемое разбиение обучающей выборки
        n_train = len(X_train)
        n_unlabeled = int(n_train * unlabeled_ratio)

        # Случайно выбираем индексы для неразмеченных данных
        unlabeled_indices = np.random.RandomState(self.random_state).choice(
            X_train.index, size=n_unlabeled, replace=False
        )
        labeled_indices = X_train.index.difference(unlabeled_indices)

        # Формируем финальные разбиения
        splits = {
            "train_labeled": (
                X_train.loc[labeled_indices],
                y_train.loc[labeled_indices],
            ),
            "train_unlabeled": (X_train.loc[unlabeled_indices], None),  # Метки скрыты
            "train_unlabeled_true": (
                X_train.loc[unlabeled_indices],
                y_train.loc[unlabeled_indices],
            ),  # Для оценки
            "val": (X_val, y_val),
            "test": (X_test, y_test),
        }

        # Добавляем статистику
        splits["statistics"] = {
            "total_samples": len(X),
            "train_labeled_samples": len(labeled_indices),
            "train_unlabeled_samples": len(unlabeled_indices),
            "val_samples": len(X_val),
            "test_samples": len(X_test),
            "unlabeled_ratio_actual": len(unlabeled_indices) / n_train,
        }

        return splits

    def prepare_all_datasets(
        self, dataset_names: Optional[Dict[str, List[str]]] = None
    ) -> Dict:
        """
        Подготовка всех датасетов для эксперимента.

        Args:
            dataset_names: Словарь с именами датасетов по типам задач

        Returns:
            Словарь со всеми подготовленными датасетами
        """
        if dataset_names is None:
            dataset_names = {
                "binary": list(self.binary_datasets.keys()),
                "multiclass": list(self.multiclass_datasets.keys()),
            }

        all_datasets = {}

        # Обработка бинарных датасетов
        for dataset_name in dataset_names.get("binary", []):
            print(f"\n{'='*50}")
            print(f"Обработка бинарного датасета: {dataset_name}")
            print(f"{'='*50}")

            try:
                # Загрузка
                X, y = self.load_dataset(dataset_name, "binary")

                # Предобработка
                X_processed, y_processed, prep_info = self.preprocess_data(
                    X, y, "binary"
                )

                # Создание полуконтролируемого разбиения
                splits = self.create_semi_supervised_split(X_processed, y_processed)

                # Кодирование категориальных признаков ПОСЛЕ разбиения
                encoder = None
                if prep_info["categorical_features"]:
                    encoder = CatBoostEncoder(cols=prep_info["categorical_features"])
                    # Обучаем энкодер только на размеченных обучающих данных
                    X_train_labeled, y_train_labeled = splits["train_labeled"]
                    encoder.fit(X_train_labeled, y_train_labeled)

                    # Применяем энкодер ко всем разбиениям
                    for split_name in [
                        "train_labeled",
                        "train_unlabeled",
                        "train_unlabeled_true",
                        "val",
                        "test",
                    ]:
                        if split_name in ["train_unlabeled", "train_unlabeled_true"]:
                            X, _ = splits[split_name]
                            X_encoded = encoder.transform(X)
                            if split_name == "train_unlabeled":
                                splits[split_name] = (X_encoded, None)
                            else:
                                _, y = splits[split_name]
                                splits[split_name] = (X_encoded, y)
                        else:
                            X, y = splits[split_name]
                            X_encoded = encoder.transform(X)
                            splits[split_name] = (X_encoded, y)

                # Сохранение результатов
                all_datasets[f"{dataset_name}_binary"] = {
                    "splits": splits,
                    "preprocessing_info": prep_info,
                    "task_type": "binary",
                    "encoder": encoder if prep_info["categorical_features"] else None,
                }

                # Вывод статистики
                self._print_dataset_statistics(dataset_name, splits, prep_info)

            except Exception as e:
                print(f"Ошибка при обработке {dataset_name}: {e}")
                continue

        # Обработка мультиклассовых датасетов
        for dataset_name in dataset_names.get("multiclass", []):
            print(f"\n{'='*50}")
            print(f"Обработка мультиклассового датасета: {dataset_name}")
            print(f"{'='*50}")

            try:
                # Загрузка
                X, y = self.load_dataset(dataset_name, "multiclass")

                # Предобработка
                X_processed, y_processed, prep_info = self.preprocess_data(
                    X, y, "multiclass"
                )

                # Кодирование категориальных признаков
                if prep_info["categorical_features"]:
                    encoder = CatBoostEncoder(cols=prep_info["categorical_features"])
                    X_temp, _, y_temp, _ = train_test_split(
                        X_processed,
                        y_processed,
                        test_size=0.8,
                        random_state=self.random_state,
                    )
                    encoder.fit(X_temp, y_temp)
                    X_processed = encoder.transform(X_processed)

                # Создание полуконтролируемого разбиения
                splits = self.create_semi_supervised_split(X_processed, y_processed)

                # Сохранение результатов
                all_datasets[f"{dataset_name}_multiclass"] = {
                    "splits": splits,
                    "preprocessing_info": prep_info,
                    "task_type": "multiclass",
                    "encoder": encoder if prep_info["categorical_features"] else None,
                }

                # Вывод статистики
                self._print_dataset_statistics(dataset_name, splits, prep_info)

            except Exception as e:
                print(f"Ошибка при обработке {dataset_name}: {e}")
                continue

        return all_datasets

    def _print_dataset_statistics(self, name: str, splits: Dict, prep_info: Dict):
        """Вывод статистики по датасету."""
        stats = splits["statistics"]
        print(f"\nСтатистика датасета {name}:")
        print(f"  - Всего примеров: {stats['total_samples']}")
        print(f"  - Размеченные данные для обучения: {stats['train_labeled_samples']}")
        print(f"  - Неразмеченные данные: {stats['train_unlabeled_samples']}")
        print(f"  - Валидационная выборка: {stats['val_samples']}")
        print(f"  - Тестовая выборка: {stats['test_samples']}")
        print(f"  - Количество признаков: {prep_info['processed_shape'][1]}")
        print(f"  - Количество классов: {prep_info['final_n_classes']}")
        print(f"  - Распределение классов: {prep_info['class_distribution']}")


def main():
    """Основная функция для тестирования модуля."""
    # Инициализация
    preparer = DatasetPreparer(random_state=42)

    # Подготовка подмножества датасетов для тестирования
    test_datasets = {
        "binary": ["breast_cancer", "heart_disease"],
        "multiclass": ["iris", "wine_quality"],
    }

    # Подготовка данных
    all_datasets = preparer.prepare_all_datasets(test_datasets)

    # Сохранение подготовленных данных
    import pickle

    with open("prepared_datasets.pkl", "wb") as f:
        pickle.dump(all_datasets, f)

    print(f"\n{'='*50}")
    print(f"Подготовка данных завершена!")
    print(f"Обработано датасетов: {len(all_datasets)}")
    print(f"Данные сохранены в файл: prepared_datasets.pkl")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
