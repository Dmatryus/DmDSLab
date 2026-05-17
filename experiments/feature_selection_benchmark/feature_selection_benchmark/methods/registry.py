"""Реестр FS-методов: ``name → (callable, group, supported_tasks)``.

Реестр — точка расширения для Персоны 2: добавить метод = один файл +
одна строка регистрации, ядро не трогается (ARCHITECTURE §2).

Заглушка каркаса (E-001.4): сигнатуры зафиксированы, тела поднимают
`NotImplementedError`.
"""

from __future__ import annotations

from .base import FSMethod, MethodInfo

__all__ = [
    "register",
    "get_method",
    "list_registered",
    "is_registered",
]


def register(name: str, method: FSMethod, group: str) -> None:
    """Регистрирует FS-метод в реестре.

    Args:
        name: Уникальное имя метода.
        method: Экземпляр класса-наследника `FSMethod`.
        group: Группа метода — ``"filter"`` | ``"wrapper"`` |
            ``"embedded"`` | ``"shap"``.
    """
    raise NotImplementedError


def get_method(name: str) -> FSMethod:
    """Возвращает зарегистрированный FS-метод по имени.

    Args:
        name: Имя метода.

    Returns:
        Экземпляр FS-метода.
    """
    raise NotImplementedError


def list_registered(task: str | None = None) -> list[MethodInfo]:
    """Возвращает метаданные зарегистрированных методов.

    Args:
        task: Тип задачи для фильтрации. ``None`` — все методы.

    Returns:
        Список метаданных методов.
    """
    raise NotImplementedError


def is_registered(name: str) -> bool:
    """Проверяет, зарегистрирован ли метод с данным именем.

    Args:
        name: Имя метода.

    Returns:
        ``True``, если метод зарегистрирован.
    """
    raise NotImplementedError
