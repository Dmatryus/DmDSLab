"""Реестр FS-методов: ``name → (FSMethod, group, supported_tasks)``.

Реестр — точка расширения для Персоны 2: добавить метод = один файл +
одна строка регистрации, ядро не трогается (ARCHITECTURE §2, CLAUDE
NEVER §7).

Built-in методы саморегистрируются: каждый группа-модуль
(`filter_methods` / `wrapper_methods` / `embedded_methods` /
`shap_methods`) вызывает `register(...)` для своих методов, а
`methods/__init__.py` импортирует все группа-модули — поэтому импорт
пакета наполняет реестр (план E-005, A-1).

Недоступные методы (OQ-2)
-------------------------

Опциональная зависимость метода (`shap`, `boruta`, `catboost` и др.)
может быть не установлена. При регистрации реестр вызывает
`FSMethod.check_availability`: если метод недоступен, он всё равно
**регистрируется** (виден в `list_registered` со статусом), но
**исключается** из дефолтного набора `default_method_names()` — того,
что соответствует ``methods=None`` в `run_benchmark`. Сама регистрация
никогда не падает из-за отсутствующей опц. зависимости.
"""

from __future__ import annotations

from dataclasses import dataclass

from .base import FSMethod, MethodInfo

__all__ = [
    "register",
    "get_method",
    "list_registered",
    "is_registered",
    "default_method_names",
]
# `clear_registry` намеренно НЕ в `__all__`: служебный хелпер для
# изоляции тестов, не часть публичного API. Тесты импортируют его явно
# (`registry.clear_registry`).


@dataclass
class _Entry:
    """Внутренняя запись реестра для одного метода."""

    method: FSMethod
    group: str
    supported_tasks: list[str]
    available: bool
    unavailable_reason: str | None


# Внутренний словарь реестра: имя метода → запись.
_REGISTRY: dict[str, _Entry] = {}


def register(
    name: str,
    method: FSMethod | type[FSMethod],
    group: str,
) -> None:
    """Регистрирует FS-метод в реестре.

    Если метод передан классом — он инстанцируется. Доступность метода
    (наличие опц. зависимостей) определяется через
    `FSMethod.check_availability`: недоступный метод регистрируется, но
    помечается недоступным и исключается из дефолтного набора прогона
    (OQ-2). Регистрация недоступного метода не является ошибкой.

    Args:
        name: Уникальное имя метода.
        method: Экземпляр или класс-наследник `FSMethod`.
        group: Группа метода — ``"filter"`` | ``"wrapper"`` |
            ``"embedded"`` | ``"shap"`` | ``"permutation"``.

    Raises:
        TypeError: Если `method` не является `FSMethod` (ни экземпляром,
            ни подклассом).
        ValueError: Если имя `name` уже зарегистрировано.
    """
    if isinstance(method, type):
        if not issubclass(method, FSMethod):
            raise TypeError(
                f"register: класс {method!r} не наследует FSMethod"
            )
        instance: FSMethod = method()
    elif isinstance(method, FSMethod):
        instance = method
    else:
        raise TypeError(
            f"register: method должен быть FSMethod или его классом, "
            f"получено {type(method)!r}"
        )

    if name in _REGISTRY:
        raise ValueError(f"register: метод {name!r} уже зарегистрирован")

    reason = instance.check_availability()
    _REGISTRY[name] = _Entry(
        method=instance,
        group=group,
        supported_tasks=list(getattr(instance, "supported_tasks", [])),
        available=reason is None,
        unavailable_reason=reason,
    )


def get_method(name: str) -> FSMethod:
    """Возвращает зарегистрированный FS-метод по имени.

    Args:
        name: Имя метода.

    Returns:
        Экземпляр FS-метода.

    Raises:
        KeyError: Если метод с данным именем не зарегистрирован.
    """
    entry = _REGISTRY.get(name)
    if entry is None:
        raise KeyError(f"get_method: метод {name!r} не зарегистрирован")
    return entry.method


def list_registered(task: str | None = None) -> list[MethodInfo]:
    """Возвращает метаданные зарегистрированных методов.

    Включает как доступные, так и недоступные методы — статус доступности
    отражён в полях `MethodInfo.available` / `unavailable_reason` (OQ-2).
    Для фильтрации только доступных методов используйте
    `default_method_names`.

    Args:
        task: Тип задачи (``"classification"`` | ``"regression"``) для
            фильтрации по `supported_tasks`. ``None`` — все методы.

    Returns:
        Список метаданных методов, отсортированный по имени метода
        (детерминированный порядок — критерий К2).
    """
    infos: list[MethodInfo] = []
    for name in sorted(_REGISTRY):
        entry = _REGISTRY[name]
        if task is not None and task not in entry.supported_tasks:
            continue
        infos.append(_build_info(name, entry))
    return infos


def is_registered(name: str) -> bool:
    """Проверяет, зарегистрирован ли метод с данным именем.

    Args:
        name: Имя метода.

    Returns:
        ``True``, если метод зарегистрирован (независимо от доступности).
    """
    return name in _REGISTRY


def default_method_names(task: str | None = None) -> list[str]:
    """Возвращает имена методов дефолтного набора прогона (``methods=None``).

    В набор входят только **доступные** методы (с установленными опц.
    зависимостями). Недоступные методы исключаются (OQ-2).

    Args:
        task: Тип задачи для фильтрации по `supported_tasks`. ``None`` —
            без фильтрации по задаче.

    Returns:
        Отсортированный список имён доступных методов.
    """
    return [
        name
        for name in sorted(_REGISTRY)
        if _REGISTRY[name].available
        and (task is None or task in _REGISTRY[name].supported_tasks)
    ]


def clear_registry() -> None:
    """Полностью очищает реестр.

    Служебная функция — предназначена для изоляции тестов. В прикладном
    коде не используется.
    """
    _REGISTRY.clear()


def _build_info(name: str, entry: _Entry) -> MethodInfo:
    """Собирает `MethodInfo` из записи реестра.

    Метаданные метода (`description`, `output_type`, declarative-
    пространство поиска и т.п.) метод объявляет сам через метод-фабрику
    `method_info`, если он есть; иначе используются нейтральные дефолты
    (актуально для застабленных в волне 1 методов — их метаданные
    наполняет волна 2).
    """
    method = entry.method
    factory = getattr(method, "method_info", None)
    if callable(factory):
        info = factory()
        # Поля группы / доступности проставляет реестр — он владеет
        # этим знанием, метод их не дублирует.
        info.group = entry.group
        info.supported_tasks = list(entry.supported_tasks)
        info.available = entry.available
        info.unavailable_reason = entry.unavailable_reason
        return info

    return MethodInfo(
        name=name,
        description=(method.__doc__ or "").strip().splitlines()[0]
        if method.__doc__
        else "",
        requires_target=True,
        supports_multiclass=True,
        output_type="subset",
        hyperparameters={},
        group=entry.group,
        supported_tasks=list(entry.supported_tasks),
        available=entry.available,
        unavailable_reason=entry.unavailable_reason,
    )
