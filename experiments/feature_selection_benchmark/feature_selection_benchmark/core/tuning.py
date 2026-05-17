"""Подбор гиперпараметров FS-методов.

Заглушка каркаса (E-001.4): сигнатуры зафиксированы по ARCHITECTURE §2/§3,
тела поднимают `NotImplementedError`.
"""

from __future__ import annotations

from ..api import Dataset
from ..methods.base import FSMethod

__all__ = ["tune_hyperparams"]


def tune_hyperparams(
    method: FSMethod,
    dataset: Dataset,
    n_trials: int = 20,
    random_seed: int | None = None,
) -> dict[str, object]:
    """Подбирает гиперпараметры FS-метода на датасете.

    Args:
        method: Метод отбора признаков.
        dataset: Входной датасет.
        n_trials: Число итераций подбора.
        random_seed: Зерно случайности для воспроизводимости.

    Returns:
        Словарь лучших найденных гиперпараметров.
    """
    raise NotImplementedError
