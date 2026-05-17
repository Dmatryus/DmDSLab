"""Smoke-тест каркаса пакета `feature_selection_benchmark`.

Минимальная проверка работоспособности каркаса (E-001.7): пакет и все его
подмодули импортируются, публичный API реэкспортирован корневым `__init__.py`,
стабильные контракты доступны, а тела заглушек поднимают `NotImplementedError`.

Тест не проверяет бизнес-логику — она появится в функциональных эпиках
(E-004…E-009). Здесь только каркас.
"""

from __future__ import annotations

import importlib
from abc import ABC
from dataclasses import fields, is_dataclass

import pytest

# --- Список подмодулей пакета (ARCHITECTURE §2 Module Map) ------------------

PACKAGE = "feature_selection_benchmark"

SUBMODULES = [
    "feature_selection_benchmark.api",
    "feature_selection_benchmark.core.orchestrator",
    "feature_selection_benchmark.core.tuning",
    "feature_selection_benchmark.core.reproducibility",
    "feature_selection_benchmark.methods.base",
    "feature_selection_benchmark.methods.registry",
    "feature_selection_benchmark.methods.filter_methods",
    "feature_selection_benchmark.methods.wrapper_methods",
    "feature_selection_benchmark.methods.embedded_methods",
    "feature_selection_benchmark.methods.shap_methods",
    "feature_selection_benchmark.storage.db",
    "feature_selection_benchmark.storage.leaderboard",
    "feature_selection_benchmark.storage.checkpoint",
    "feature_selection_benchmark.reporting.formatter",
    "feature_selection_benchmark.gui.app",
]

# Публичные имена, реэкспортируемые корневым __init__.py.
PUBLIC_NAMES = [
    "run_benchmark",
    "get_leaderboard",
    "list_methods",
    "register_method",
    "Dataset",
    "BenchmarkResult",
    "FSMethod",
    "MethodInfo",
]

# Поля датакласса MethodInfo. Базовые 6 зафиксированы каркасом E-001.3
# (ARCHITECTURE §4); E-005.1 добавил group / supported_tasks / available /
# unavailable_reason — метаданные группы и статус доступности метода для
# `list_registered` (план E-005, OQ-2). Все добавленные поля имеют
# дефолты, поэтому контракт построения `MethodInfo` обратно совместим.
METHOD_INFO_CORE_FIELDS = {
    "name",
    "description",
    "requires_target",
    "supports_multiclass",
    "output_type",
    "hyperparameters",
}
METHOD_INFO_FIELDS = METHOD_INFO_CORE_FIELDS | {
    "group",
    "supported_tasks",
    "available",
    "unavailable_reason",
}


# --- Импорт пакета и подмодулей ---------------------------------------------


def test_package_imports() -> None:
    """Корневой пакет импортируется без ошибок."""
    module = importlib.import_module(PACKAGE)
    assert module is not None


@pytest.mark.parametrize("submodule", SUBMODULES)
def test_submodule_imports(submodule: str) -> None:
    """Каждый подмодуль каркаса импортируется без ошибок."""
    module = importlib.import_module(submodule)
    assert module is not None


# --- Реэкспорт публичного API -----------------------------------------------


def test_public_api_in_all() -> None:
    """`__all__` корневого пакета содержит ровно 8 публичных имён."""
    import feature_selection_benchmark as fsb

    assert set(fsb.__all__) == set(PUBLIC_NAMES)


@pytest.mark.parametrize("name", PUBLIC_NAMES)
def test_public_name_available(name: str) -> None:
    """Каждое публичное имя доступно как атрибут корневого пакета."""
    import feature_selection_benchmark as fsb

    assert hasattr(fsb, name), f"{name} не реэкспортирован из __init__.py"


# --- Контракт FSMethod ------------------------------------------------------


def test_fsmethod_is_abc() -> None:
    """`FSMethod` — корректный ABC: наследник `abc.ABC`."""
    from feature_selection_benchmark import FSMethod

    assert issubclass(FSMethod, ABC)


def test_fsmethod_cannot_be_instantiated() -> None:
    """`FSMethod` нельзя инстанцировать напрямую — есть абстрактный метод."""
    from feature_selection_benchmark import FSMethod

    with pytest.raises(TypeError):
        FSMethod()  # type: ignore[abstract]


def test_fsmethod_fit_select_is_abstract() -> None:
    """`fit_select` объявлен абстрактным методом интерфейса."""
    from feature_selection_benchmark import FSMethod

    assert "fit_select" in FSMethod.__abstractmethods__


# --- Контракт MethodInfo ----------------------------------------------------


def test_method_info_is_dataclass() -> None:
    """`MethodInfo` — датакласс."""
    from feature_selection_benchmark import MethodInfo

    assert is_dataclass(MethodInfo)


def test_method_info_has_expected_fields() -> None:
    """`MethodInfo` содержит ожидаемые поля.

    6 базовых полей каркаса E-001.3 + 4 поля E-005.1 (group /
    supported_tasks / available / unavailable_reason).
    """
    from feature_selection_benchmark import MethodInfo

    actual = {f.name for f in fields(MethodInfo)}
    assert METHOD_INFO_CORE_FIELDS <= actual
    assert actual == METHOD_INFO_FIELDS


# --- Контракты-датаклассы Dataset / BenchmarkResult -------------------------


def test_dataset_is_dataclass() -> None:
    """`Dataset` — датакласс."""
    from feature_selection_benchmark import Dataset

    assert is_dataclass(Dataset)


def test_benchmark_result_is_dataclass() -> None:
    """`BenchmarkResult` — датакласс."""
    from feature_selection_benchmark import BenchmarkResult

    assert is_dataclass(BenchmarkResult)


# --- Заглушки поднимают NotImplementedError ---------------------------------


def test_public_functions_raise_not_implemented() -> None:
    """Публичные функции-заглушки поднимают `NotImplementedError`."""
    from feature_selection_benchmark import (
        get_leaderboard,
        list_methods,
        register_method,
        run_benchmark,
    )

    with pytest.raises(NotImplementedError):
        run_benchmark(dataset=None)  # type: ignore[arg-type]
    with pytest.raises(NotImplementedError):
        get_leaderboard()
    with pytest.raises(NotImplementedError):
        list_methods()
    with pytest.raises(NotImplementedError):
        register_method(name="x", method=None, group="filter")


def test_fit_select_raises_not_implemented() -> None:
    """`fit_select` конкретного метода-заглушки поднимает `NotImplementedError`.

    Берём конкретный наследник `FSMethod` (заглушку filter-метода): сам ABC
    инстанцировать нельзя, поэтому проверяем тело `fit_select` на наследнике.
    """
    from feature_selection_benchmark.methods.filter_methods import (
        VarianceThresholdMethod,
    )

    method = VarianceThresholdMethod()
    with pytest.raises(NotImplementedError):
        method.fit_select(X_train=None, y_train=None)  # type: ignore[arg-type]
