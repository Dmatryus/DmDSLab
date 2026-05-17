"""Сохранение и загрузка состояния прогона (checkpointing).

Позволяет возобновить прерванный прогон, пропустив уже завершённые методы
(ARCHITECTURE §3).

Заглушка каркаса (E-001.4): сигнатуры зафиксированы, тела поднимают
`NotImplementedError`.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from ..core.orchestrator import CVResult

__all__ = ["CheckpointState", "resolve_run_id", "save_checkpoint", "load_checkpoint"]


@dataclass
class CheckpointState:
    """Сохранённое состояние прерванного прогона.

    Attributes:
        run_id: Идентификатор прогона.
        completed_methods: Имена уже завершённых методов.
        partial_results: Результаты завершённых методов по их именам.
        checkpoint_path: Путь к файлу checkpoint на диске.
    """

    run_id: str
    completed_methods: list[str]
    partial_results: dict[str, CVResult]
    checkpoint_path: Path


def resolve_run_id(conn: sqlite3.Connection, params_hash: str) -> str:
    """Возвращает идентификатор прогона по хэшу параметров.

    Args:
        conn: Соединение с базой данных.
        params_hash: Хэш параметров прогона.

    Returns:
        Идентификатор прогона (`run_id`).
    """
    raise NotImplementedError


def save_checkpoint(
    conn: sqlite3.Connection,
    run_id: str,
    method_name: str,
    result: CVResult,
    checkpoint_dir: str | Path | None = None,
) -> None:
    """Сохраняет промежуточный результат завершённого метода.

    Args:
        conn: Соединение с базой данных.
        run_id: Идентификатор прогона.
        method_name: Имя завершённого метода.
        result: Результат кросс-валидации метода.
        checkpoint_dir: Директория для файлов checkpoint. ``None`` —
            временная системная папка.
    """
    raise NotImplementedError


def load_checkpoint(
    conn: sqlite3.Connection,
    run_id: str,
) -> CheckpointState | None:
    """Загружает сохранённое состояние прогона.

    Args:
        conn: Соединение с базой данных.
        run_id: Идентификатор прогона.

    Returns:
        Состояние прогона или ``None``, если checkpoint отсутствует.
    """
    raise NotImplementedError
