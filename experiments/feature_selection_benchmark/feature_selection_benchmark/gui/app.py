"""Десктопный GUI поверх storage и core (Should-уровень).

Таблица лидерборда, графики метрик, живой прогресс. Реализуется на Dear
PyGui в эпике E-009; ядро от GUI не зависит (ARCHITECTURE §2/§8).

Заглушка каркаса (E-001.4): сигнатура точки входа зафиксирована, тело
поднимает `NotImplementedError`.
"""

from __future__ import annotations

__all__ = ["launch_gui"]


def launch_gui(db_path: str | None = None) -> None:
    """Запускает десктопное GUI-приложение лидерборда.

    Args:
        db_path: Путь к SQLite-файлу хранилища. ``None`` — путь по
            умолчанию.
    """
    raise NotImplementedError
