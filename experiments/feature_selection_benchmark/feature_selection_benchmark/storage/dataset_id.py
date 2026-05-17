"""Генерация идентификатора датасета (`dataset_id`) для записей лидерборда.

Стратегия (ADR `docs/adr/0001-dataset-id-strategy.md`): `dataset_id` —
комбинация пользовательского имени и структурного fingerprint схемы
датасета. Fingerprint берётся по именам колонок и их dtype, а не по
значениям ячеек и не по числу строк, поэтому `dataset_id` стабилен при
добавлении/удалении строк и меняется при изменении схемы.

`Dataset.data` на диск не пишется: в хэш идут только метаданные схемы
(имена колонок и dtype) — это согласуется с моделью безопасности
(ARCHITECTURE §7).
"""

from __future__ import annotations

import hashlib

import pandas as pd

__all__ = ["compute_dataset_id"]

# Длина hex-fingerprint в составе dataset_id (48 бит — пренебрежимая
# вероятность случайной коллизии при человеко-масштабном числе датасетов).
_FINGERPRINT_LEN = 12


def compute_dataset_id(name: str, data: pd.DataFrame) -> str:
    """Возвращает идентификатор датасета вида ``"{name}::{fingerprint}"``.

    ``fingerprint`` — первые 12 hex-символов SHA-256 от канонической строки
    схемы: имена колонок в порядке ``data.columns``, каждая со своим
    каноническим dtype. Число строк и значения ячеек в хэш не входят, поэтому
    id не меняется при добавлении/удалении строк и различает датасеты с
    разной схемой.

    Используется SHA-256 (`hashlib`), а не встроенный `hash()`: последний
    солится по процессу и дал бы разный id между запусками.

    Args:
        name: Пользовательское имя датасета/задачи. Должно быть непустым.
        data: Таблица датасета (признаки + целевая переменная).

    Returns:
        Идентификатор датасета — стабильная строка для записей лидерборда.

    Raises:
        ValueError: Если ``name`` пустой или состоит только из пробелов.
    """
    if not name or not name.strip():
        raise ValueError("name датасета должно быть непустым")

    schema = "|".join(
        f"{column}:{dtype.name}"
        for column, dtype in zip(data.columns, data.dtypes, strict=True)
    )
    fingerprint = hashlib.sha256(schema.encode("utf-8")).hexdigest()[:_FINGERPRINT_LEN]
    return f"{name}::{fingerprint}"
