"""Форматирование вывода: ранжирование, прогресс-бар, уведомление о
baseline.

Заглушка каркаса (E-001.4): сигнатуры зафиксированы по ARCHITECTURE §3,
тела поднимают `NotImplementedError`.
"""

from __future__ import annotations

import pandas as pd

__all__ = ["report_progress", "format_leaderboard", "check_baseline_notification"]


def report_progress(completed: int, total: int, current_method: str) -> None:
    """Отображает прогресс прогона через tqdm.

    Args:
        completed: Число завершённых методов.
        total: Общее число методов.
        current_method: Имя текущего метода.
    """
    raise NotImplementedError


def format_leaderboard(ranked_results: pd.DataFrame) -> str:
    """Форматирует ранжированный лидерборд для текстового вывода.

    Args:
        ranked_results: Ранжированная таблица результатов.

    Returns:
        Человекочитаемое строковое представление лидерборда.
    """
    raise NotImplementedError


def check_baseline_notification(
    ranked_results: pd.DataFrame,
    baseline_score: float | None,
) -> bool | None:
    """Определяет, побит ли baseline, и формирует уведомление.

    Args:
        ranked_results: Ранжированная таблица результатов.
        baseline_score: Оценка baseline. ``None`` — baseline не задан.

    Returns:
        ``True`` / ``False`` — побит ли baseline; ``None`` — baseline не
        задавался.
    """
    raise NotImplementedError
