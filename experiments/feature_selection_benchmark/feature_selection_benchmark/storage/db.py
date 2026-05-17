"""SQLite-соединение и миграции схемы хранилища.

Хранилище — единый SQLite-файл (ARCHITECTURE §5). `Dataset.data` на диск
никогда не пишется; storage-слой принимает только метрики и имена
признаков.

Заглушка каркаса (E-001.4): сигнатуры зафиксированы, тела поднимают
`NotImplementedError`.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

__all__ = ["connect", "init_schema", "get_default_db_path"]


def get_default_db_path() -> Path:
    """Возвращает путь к SQLite-файлу хранилища по умолчанию.

    Returns:
        Путь к файлу базы данных.
    """
    raise NotImplementedError


def connect(db_path: str | Path | None = None) -> sqlite3.Connection:
    """Открывает соединение с SQLite-хранилищем.

    Args:
        db_path: Путь к файлу базы данных. ``None`` — путь по умолчанию.

    Returns:
        Соединение с базой данных.
    """
    raise NotImplementedError


def init_schema(conn: sqlite3.Connection) -> None:
    """Создаёт таблицы схемы хранилища, если их ещё нет.

    Создаёт таблицы ``runs``, ``method_results``, ``checkpoints``,
    ``leaderboard`` (ARCHITECTURE §5).

    Args:
        conn: Соединение с базой данных.
    """
    raise NotImplementedError
