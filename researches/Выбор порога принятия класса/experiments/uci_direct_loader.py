"""
Модуль для прямой загрузки датасетов с UCI Machine Learning Repository.
Используется как альтернатива, если датасет недоступен через DmDSLab.
"""

import pandas as pd
import numpy as np
import requests
import zipfile
import io
from typing import Tuple, Optional
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
import warnings
warnings.filterwarnings('ignore')


class UCIDirectLoader:
    """Класс для прямой загрузки датасетов с UCI репозитория."""
    
    def __init__(self):
        """Инициализация загрузчика с URL датасетов."""
        self.dataset_urls = {
            # Бинарные датасеты
            'breast_cancer': {
                'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data',
                'names': ['id', 'diagnosis'] + [f'feature_{i}' for i in range(30)],
                'target': 'diagnosis',
                'drop_cols': ['id']
            },
            'heart_disease': {
                'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data',
                'names': ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                         'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'],
                'target': 'target',
                'na_values': ['?']
            },
            'bank_marketing': {
                'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip',
                'file_in_zip': 'bank-additional/bank-additional-full.csv',
                'separator': ';',
                'target': 'y'
            },
            'adult_income': {
                'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',
                'names': ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 
                         'marital-status', 'occupation', 'relationship', 'race', 'sex',
                         'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'],
                'target': 'income',
                'na_values': [' ?']
            },
            'credit_default': {
                'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls',
                'target': 'default payment next month',
                'header': 1,
                'drop_cols': ['ID']
            },
            
            # Мультиклассовые датасеты
            'wine_quality': {
                'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv',
                'separator': ';',
                'target': 'quality'
            },
            'letter_recognition': {
                'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data',
                'names': ['letter'] + [f'feature_{i}' for i in range(16)],
                'target': 'letter'
            },
            'satellite': {
                'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/satimage/sat.trn',
                'names': [f'feature_{i}' for i in range(36)] + ['class'],
                'target': 'class',
                'separator': ' '
            }
        }
    
    def load_dataset(self, dataset_name: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Загрузка датасета по имени.
        
        Args:
            dataset_name: Название датасета
            
        Returns:
            X: Признаки
            y: Целевая переменная
        """
        # Проверяем встроенные датасеты sklearn
        if dataset_name == 'iris':
            return self._load_sklearn_dataset('iris')
        elif dataset_name == 'wine':
            return self._load_sklearn_dataset('wine')
        elif dataset_name == 'breast_cancer' and dataset_name not in self.dataset_urls:
            return self._load_sklearn_dataset('breast_cancer')
        
        # Загружаем с UCI
        if dataset_name not in self.dataset_urls:
            raise ValueError(f"Датасет {dataset_name} не найден")
        
        config = self.dataset_urls[dataset_name]
        print(f"Загрузка {dataset_name} с UCI Repository...")
        
        try:
            # Загрузка данных
            if dataset_name == 'credit_default':
                # Excel файл
                df = pd.read_excel(config['url'], header=config.get('header', 0))
            elif dataset_name == 'bank_marketing':
                # ZIP файл
                df = self._load_from_zip(config['url'], config['file_in_zip'], 
                                       separator=config.get('separator', ','))
            else:
                # CSV файлы
                df = pd.read_csv(
                    config['url'],
                    names=config.get('names'),
                    sep=config.get('separator', ','),
                    na_values=config.get('na_values', []),
                    header=0 if 'names' not in config else None
                )
            
            # Разделение на признаки и целевую переменную
            target_col = config['target']
            drop_cols = config.get('drop_cols', [])
            
            if drop_cols:
                df = df.drop(columns=drop_cols)
            
            y = df[target_col]
            X = df.drop(columns=[target_col])
            
            # Преобразование бинарной целевой переменной
            if dataset_name in ['breast_cancer', 'bank_marketing', 'adult_income']:
                if dataset_name == 'breast_cancer':
                    y = (y == 'M').astype(int)  # M=1 (malignant), B=0 (benign)
                elif dataset_name == 'bank_marketing':
                    y = (y == 'yes').astype(int)
                elif dataset_name == 'adult_income':
                    y = (y.str.strip() == '>50K').astype(int)
            
            # Для heart_disease: преобразуем в бинарную классификацию
            if dataset_name == 'heart_disease':
                y = (y > 0).astype(int)  # 0 = нет болезни, 1-4 = есть болезнь
            
            print(f"Датасет {dataset_name} загружен: {X.shape[0]} примеров, {X.shape[1]} признаков")
            
            return X, y
            
        except Exception as e:
            print(f"Ошибка при загрузке {dataset_name}: {e}")
            print("Генерация синтетических данных...")
            return self._generate_synthetic_data(dataset_name)
    
    def _load_sklearn_dataset(self, name: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Загрузка встроенных датасетов sklearn."""
        if name == 'iris':
            data = load_iris()
        elif name == 'wine':
            data = load_wine()
        elif name == 'breast_cancer':
            data = load_breast_cancer()
        else:
            raise ValueError(f"Неизвестный sklearn датасет: {name}")
        
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target, name='target')
        
        return X, y
    
    def _load_from_zip(self, url: str, file_in_zip: str, separator: str = ',') -> pd.DataFrame:
        """Загрузка CSV файла из ZIP архива."""
        response = requests.get(url)
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            with z.open(file_in_zip) as f:
                return pd.read_csv(f, sep=separator)
    
    def _generate_synthetic_data(self, dataset_name: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Генерация синтетических данных для демонстрации."""
        from sklearn.datasets import make_classification
        
        # Параметры для разных датасетов
        params = {
            'breast_cancer': {'n_samples': 569, 'n_features': 30, 'n_classes': 2},
            'heart_disease': {'n_samples': 303, 'n_features': 13, 'n_classes': 2},
            'bank_marketing': {'n_samples': 4119, 'n_features': 20, 'n_classes': 2},
            'adult_income': {'n_samples': 32561, 'n_features': 14, 'n_classes': 2},
            'credit_default': {'n_samples': 30000, 'n_features': 23, 'n_classes': 2},
            'wine_quality': {'n_samples': 4898, 'n_features': 11, 'n_classes': 7},
            'letter_recognition': {'n_samples': 20000, 'n_features': 16, 'n_classes': 26},
            'satellite': {'n_samples': 4435, 'n_features': 36, 'n_classes': 6},
        }
        
        p = params.get(dataset_name, {'n_samples': 1000, 'n_features': 20, 'n_classes': 2})
        
        X, y = make_classification(
            n_samples=p['n_samples'],
            n_features=p['n_features'],
            n_informative=int(p['n_features'] * 0.7),
            n_redundant=int(p['n_features'] * 0.2),
            n_classes=p['n_classes'],
            random_state=42
        )
        
        X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        y = pd.Series(y, name='target')
        
        return X, y
    
    def get_dataset_info(self, dataset_name: str) -> dict:
        """Получение информации о датасете."""
        info = {
            'breast_cancer': {
                'description': 'Wisconsin Breast Cancer Dataset',
                'task': 'binary',
                'n_features': 30,
                'n_classes': 2,
                'source': 'UCI ML Repository'
            },
            'heart_disease': {
                'description': 'Cleveland Heart Disease Dataset',
                'task': 'binary',
                'n_features': 13,
                'n_classes': 2,
                'source': 'UCI ML Repository'
            },
            'bank_marketing': {
                'description': 'Bank Marketing Dataset',
                'task': 'binary',
                'n_features': 20,
                'n_classes': 2,
                'source': 'UCI ML Repository'
            },
            'adult_income': {
                'description': 'Adult Income Dataset',
                'task': 'binary',
                'n_features': 14,
                'n_classes': 2,
                'source': 'UCI ML Repository'
            },
            'credit_default': {
                'description': 'Default of Credit Card Clients Dataset',
                'task': 'binary',
                'n_features': 23,
                'n_classes': 2,
                'source': 'UCI ML Repository'
            },
            'iris': {
                'description': 'Iris Flower Dataset',
                'task': 'multiclass',
                'n_features': 4,
                'n_classes': 3,
                'source': 'sklearn'
            },
            'wine': {
                'description': 'Wine Dataset',
                'task': 'multiclass',
                'n_features': 13,
                'n_classes': 3,
                'source': 'sklearn'
            },
            'wine_quality': {
                'description': 'Wine Quality Dataset',
                'task': 'multiclass',
                'n_features': 11,
                'n_classes': 7,
                'source': 'UCI ML Repository'
            },
            'letter_recognition': {
                'description': 'Letter Recognition Dataset',
                'task': 'multiclass',
                'n_features': 16,
                'n_classes': 26,
                'source': 'UCI ML Repository'
            },
            'satellite': {
                'description': 'Statlog Satellite Image Dataset',
                'task': 'multiclass',
                'n_features': 36,
                'n_classes': 6,
                'source': 'UCI ML Repository'
            }
        }
        
        return info.get(dataset_name, {})


def test_loader():
    """Тестирование загрузчика."""
    loader = UCIDirectLoader()
    
    # Тестируем загрузку нескольких датасетов
    test_datasets = ['iris', 'breast_cancer', 'wine_quality']
    
    for dataset_name in test_datasets:
        print(f"\n{'='*50}")
        print(f"Тестирование загрузки: {dataset_name}")
        print(f"{'='*50}")
        
        try:
            X, y = loader.load_dataset(dataset_name)
            info = loader.get_dataset_info(dataset_name)
            
            print(f"Описание: {info.get('description', 'N/A')}")
            print(f"Источник: {info.get('source', 'N/A')}")
            print(f"Размер: {X.shape}")
            print(f"Классы: {np.unique(y)}")
            print(f"Распределение классов: {dict(pd.Series(y).value_counts())}")
            
        except Exception as e:
            print(f"Ошибка: {e}")


if __name__ == "__main__":
    test_loader()
