"""FS-методы группы ``filter`` — отбор по статистикам признаков.

Реализованы четыре метода (план E-005 §2, E-005.2):

- `VarianceThresholdMethod` — отбор по порогу дисперсии признака;
- `CorrelationMethod` — отбор по корреляции с целевой переменной
  (Pearson / Spearman, `scipy.stats`);
- `MutualInformationMethod` — отбор по взаимной информации с целевой
  переменной (`sklearn.feature_selection.mutual_info_*`);
- `MRMRMethod` — minimum-Redundancy-Maximum-Relevance (опц. пакет
  `mrmr-selection`, ленивый импорт; при отсутствии метод регистрируется
  недоступным — план E-005, OQ-2 / R-1).

Каждый метод объявляет declarative-пространство поиска гиперпараметров
словарём ``имя → HyperParam`` (контракт ADR `0002`, см. `base.py`); слой
`methods/` не импортирует `optuna`. Built-in методы саморегистрируются —
строки `register(...)` в конце файла (план E-005, A-1).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_selection import (
    mutual_info_classif,
    mutual_info_regression,
)

from .base import FSMethod, HyperParam, MethodInfo
from .registry import register

__all__ = [
    "VarianceThresholdMethod",
    "CorrelationMethod",
    "MutualInformationMethod",
    "MRMRMethod",
]


def _is_classification(y: pd.Series) -> bool:
    """Эвристически определяет, является ли целевая переменная классовой.

    Args:
        y: Целевая переменная.

    Returns:
        ``True``, если `y` похожа на метки классов (нечисловой dtype или
        малое число уникальных целочисленных значений).
    """
    if not pd.api.types.is_numeric_dtype(y):
        return True
    nunique = y.nunique(dropna=True)
    if nunique <= 2:
        return True
    # Целочисленные значения с небольшим числом уникальных — классификация.
    values = y.dropna()
    is_integer = bool(np.all(np.equal(np.mod(values, 1), 0)))
    return is_integer and nunique <= max(20, int(0.05 * len(values)))


def _top_k(scores: dict[str, float], k: int) -> list[str]:
    """Возвращает имена `k` признаков с наибольшим скором.

    Сортировка детерминирована (критерий К2): по убыванию скора, при
    равенстве — по имени признака.

    Args:
        scores: Отображение ``имя признака → скор``.
        k: Число отбираемых признаков.

    Returns:
        Список имён отобранных признаков.
    """
    k = max(1, min(k, len(scores)))
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return [name for name, _ in ordered[:k]]


class VarianceThresholdMethod(FSMethod):
    """Отбор признаков по порогу дисперсии."""

    name = "variance_threshold"
    group = "filter"
    supported_tasks = ["classification", "regression"]

    @classmethod
    def method_info(cls) -> MethodInfo:
        """Возвращает метаданные метода и пространство поиска."""
        return MethodInfo(
            name=cls.name,
            description="Отбор признаков с дисперсией строго выше "
            "заданного порога; константные признаки отсеиваются.",
            requires_target=False,
            supports_multiclass=True,
            output_type="subset",
            hyperparameters={
                "threshold": HyperParam(
                    kind="float",
                    default=0.0,
                    low=0.0,
                    high=1.0,
                    log=False,
                ),
            },
        )

    def fit_select(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        **hyperparams: Any,
    ) -> list[str]:
        """Отбирает признаки с дисперсией выше порога `threshold`.

        Args:
            X_train: Обучающая выборка признаков.
            y_train: Целевая переменная (не используется — метод
                безнадзорный).
            **hyperparams: ``threshold`` — порог дисперсии (по умолчанию
                ``0.0`` — отбрасываются только константы).

        Returns:
            Имена признаков с дисперсией строго выше порога. Если ни один
            признак не прошёл порог — возвращается признак с максимальной
            дисперсией (метод обязан вернуть непустое подмножество).
        """
        threshold = float(hyperparams.get("threshold", 0.0))
        variances = dict(
            zip(
                X_train.columns,
                X_train.var(axis=0, ddof=0).to_numpy(),
                strict=True,
            )
        )
        selected = [
            col
            for col, var in variances.items()
            if var > threshold
        ]
        if selected:
            return selected
        # Все признаки ниже порога — вернуть признак с макс. дисперсией
        # (метод обязан вернуть непустое подмножество).
        return _top_k(variances, 1)


class CorrelationMethod(FSMethod):
    """Отбор признаков по корреляции с целевой переменной (Pearson/Spearman)."""

    name = "correlation"
    group = "filter"
    supported_tasks = ["classification", "regression"]

    @classmethod
    def method_info(cls) -> MethodInfo:
        """Возвращает метаданные метода и пространство поиска."""
        return MethodInfo(
            name=cls.name,
            description="Отбор top-k признаков по абсолютной корреляции "
            "с целевой переменной (Pearson или Spearman).",
            requires_target=True,
            supports_multiclass=False,
            output_type="ranking",
            hyperparameters={
                "method": HyperParam(
                    kind="categorical",
                    default="pearson",
                    choices=["pearson", "spearman"],
                ),
                "k": HyperParam(
                    kind="int",
                    default=10,
                    low=1,
                    high=50,
                    log=False,
                ),
            },
        )

    def fit_select(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        **hyperparams: Any,
    ) -> list[str]:
        """Отбирает `k` признаков с наибольшей |корреляцией| к цели.

        Args:
            X_train: Обучающая выборка признаков.
            y_train: Целевая переменная.
            **hyperparams: ``method`` — ``"pearson"`` | ``"spearman"``;
                ``k`` — число отбираемых признаков.

        Returns:
            Имена `k` признаков с наибольшим модулем корреляции.
        """
        method = str(hyperparams.get("method", "pearson"))
        k = int(hyperparams.get("k", 10))
        corr_fn = pearsonr if method == "pearson" else spearmanr
        y_values = pd.to_numeric(y_train, errors="coerce").to_numpy()

        scores: dict[str, float] = {}
        for col in X_train.columns:
            x_values = X_train[col].to_numpy()
            if np.std(x_values) == 0 or np.std(y_values) == 0:
                # Константа — корреляция не определена; нулевой скор.
                scores[col] = 0.0
                continue
            stat = corr_fn(x_values, y_values).statistic
            scores[col] = 0.0 if np.isnan(stat) else abs(float(stat))
        return _top_k(scores, k)


class MutualInformationMethod(FSMethod):
    """Отбор признаков по взаимной информации с целевой переменной."""

    name = "mutual_information"
    group = "filter"
    supported_tasks = ["classification", "regression"]

    @classmethod
    def method_info(cls) -> MethodInfo:
        """Возвращает метаданные метода и пространство поиска."""
        return MethodInfo(
            name=cls.name,
            description="Отбор top-k признаков по взаимной информации "
            "с целевой переменной (sklearn mutual_info_*).",
            requires_target=True,
            supports_multiclass=True,
            output_type="ranking",
            hyperparameters={
                "k": HyperParam(
                    kind="int",
                    default=10,
                    low=1,
                    high=50,
                    log=False,
                ),
            },
        )

    def fit_select(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        **hyperparams: Any,
    ) -> list[str]:
        """Отбирает `k` признаков с наибольшей взаимной информацией.

        Тип задачи (классификация / регрессия) определяется эвристически
        по целевой переменной — выбирается `mutual_info_classif` или
        `mutual_info_regression`.

        Args:
            X_train: Обучающая выборка признаков.
            y_train: Целевая переменная.
            **hyperparams: ``k`` — число отбираемых признаков;
                ``random_state`` — зерно (MI-оценка использует kNN со
                случайностью; для воспроизводимости — К2).

        Returns:
            Имена `k` признаков с наибольшей взаимной информацией.
        """
        k = int(hyperparams.get("k", 10))
        random_state = hyperparams.get("random_state", 0)
        mi_fn = (
            mutual_info_classif
            if _is_classification(y_train)
            else mutual_info_regression
        )
        mi_values = mi_fn(
            X_train.to_numpy(),
            y_train.to_numpy(),
            random_state=random_state,
        )
        scores = dict(
            zip(
                X_train.columns,
                (float(v) for v in mi_values),
                strict=True,
            )
        )
        return _top_k(scores, k)


class MRMRMethod(FSMethod):
    """Отбор признаков методом minimum-Redundancy-Maximum-Relevance."""

    name = "mrmr"
    group = "filter"
    supported_tasks = ["classification", "regression"]

    @classmethod
    def check_availability(cls) -> str | None:
        """Проверяет наличие опц. пакета `mrmr-selection`.

        Returns:
            ``None``, если пакет `mrmr` импортируется; иначе — строка с
            причиной недоступности (план E-005, OQ-2 / R-1).
        """
        try:
            import mrmr  # noqa: F401
        except ImportError:
            return (
                "требуется пакет 'mrmr-selection' "
                "(extra `[methods]`: pip install "
                "'feature_selection_benchmark[methods]')"
            )
        return None

    @classmethod
    def method_info(cls) -> MethodInfo:
        """Возвращает метаданные метода и пространство поиска."""
        return MethodInfo(
            name=cls.name,
            description="minimum-Redundancy-Maximum-Relevance: отбор k "
            "признаков с максимальной релевантностью цели и минимальной "
            "взаимной избыточностью (пакет mrmr-selection).",
            requires_target=True,
            supports_multiclass=True,
            output_type="subset",
            hyperparameters={
                "k": HyperParam(
                    kind="int",
                    default=10,
                    low=1,
                    high=50,
                    log=False,
                ),
            },
        )

    def fit_select(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        **hyperparams: Any,
    ) -> list[str]:
        """Отбирает `k` признаков методом mRMR.

        Опц. пакет `mrmr-selection` импортируется лениво — при его
        отсутствии метод помечается недоступным через `check_availability`
        и не попадает в дефолтный набор прогона; прямой вызов
        `fit_select` поднимает понятный `ImportError`.

        Args:
            X_train: Обучающая выборка признаков.
            y_train: Целевая переменная.
            **hyperparams: ``k`` — число отбираемых признаков.

        Returns:
            Имена `k` признаков, отобранных mRMR.

        Raises:
            ImportError: Если пакет `mrmr-selection` не установлен.
        """
        try:
            from mrmr import mrmr_classif, mrmr_regression
        except ImportError as exc:  # pragma: no cover - зависит от среды
            raise ImportError(
                "mrmr: " + (self.check_availability() or str(exc))
            ) from exc

        k = int(hyperparams.get("k", 10))
        k = max(1, min(k, X_train.shape[1]))
        mrmr_fn = (
            mrmr_classif
            if _is_classification(y_train)
            else mrmr_regression
        )
        selected = mrmr_fn(X=X_train, y=y_train, K=k, show_progress=False)
        return list(selected)


register("variance_threshold", VarianceThresholdMethod, "filter")
register("correlation", CorrelationMethod, "filter")
register("mutual_information", MutualInformationMethod, "filter")
register("mrmr", MRMRMethod, "filter")
