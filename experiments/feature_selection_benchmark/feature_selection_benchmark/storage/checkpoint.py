"""Сохранение и загрузка состояния прогона (checkpointing).

Edit log:
    - 2026-05-17 · v0.3.0 · execution-agent · реализация E-006.3
      (resolve_run_id / save_checkpoint / load_checkpoint).

Позволяет возобновить прерванный прогон, пропустив уже завершённые методы
(ARCHITECTURE §3).

Состояние прогона разделено на две части:

* строка в таблице ``checkpoints`` (ARCHITECTURE §5) — список завершённых
  методов, метка времени и путь к файлу состояния;
* файл состояния на диске — сериализованные результаты завершённых методов
  (`partial_results`: ``dict[str, CVResult]``).

Таблица ``checkpoints`` хранит только имена методов, поэтому сами результаты
кросс-валидации (`CVResult` — метрики и имена признаков) персистируются в
JSON-файле. Сырые данные ``Dataset.data`` в checkpoint никогда не попадают
(CLAUDE NEVER §3): сохраняются только агрегированные метрики и имена
отобранных признаков.

Соединение ``conn`` принимается извне; модуль ``storage/db.py`` (схема и
миграции, эпик E-007) не импортируется.
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from ..core.orchestrator import CVResult

__all__ = ["CheckpointState", "resolve_run_id", "save_checkpoint", "load_checkpoint"]

# Префикс имени файла состояния checkpoint в директории `checkpoint_dir`.
_CHECKPOINT_FILE_PREFIX = "checkpoint_"
_CHECKPOINT_FILE_SUFFIX = ".json"


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


def _utc_now() -> str:
    """Возвращает текущую метку времени в формате ISO 8601 (UTC)."""
    return datetime.now(timezone.utc).isoformat()


def _checkpoint_file(run_id: str, checkpoint_dir: str | Path | None) -> Path:
    """Возвращает путь к файлу состояния checkpoint для прогона.

    Args:
        run_id: Идентификатор прогона.
        checkpoint_dir: Директория для файлов checkpoint. ``None`` —
            временная системная папка.

    Returns:
        Путь к JSON-файлу состояния прогона.
    """
    if checkpoint_dir is not None:
        directory = Path(checkpoint_dir)
    else:
        directory = Path(tempfile.gettempdir())
    file_name = f"{_CHECKPOINT_FILE_PREFIX}{run_id}{_CHECKPOINT_FILE_SUFFIX}"
    return directory / file_name


def resolve_run_id(conn: sqlite3.Connection, params_hash: str) -> str:
    """Возвращает идентификатор прогона по хэшу параметров.

    Если для данного хэша уже существует строка checkpoint, возвращается её
    ``run_id`` — прогон будет возобновлён. Иначе создаётся новый
    идентификатор. Идентификатор прогона совпадает с хэшем параметров
    (ARCHITECTURE §5: ``runs.id`` — хэш параметров прогона), что делает
    функцию идемпотентной: повторный вызов с тем же хэшем даёт тот же id.

    Args:
        conn: Соединение с базой данных.
        params_hash: Хэш параметров прогона.

    Returns:
        Идентификатор прогона (`run_id`).
    """
    if not params_hash:
        raise ValueError("params_hash не может быть пустым")

    row = conn.execute(
        "SELECT run_id FROM checkpoints WHERE run_id = ?",
        (params_hash,),
    ).fetchone()
    if row is not None:
        return str(row[0])
    return params_hash


def save_checkpoint(
    conn: sqlite3.Connection,
    run_id: str,
    method_name: str,
    result: CVResult,
    checkpoint_dir: str | Path | None = None,
) -> None:
    """Сохраняет промежуточный результат завершённого метода.

    Обновляет (или создаёт) строку прогона в таблице ``checkpoints`` и файл
    состояния на диске. ``method_name`` добавляется к списку завершённых
    методов, его ``result`` — к результатам в файле состояния. Повторное
    сохранение того же метода перезаписывает его результат, не дублируя имя
    в списке завершённых.

    Args:
        conn: Соединение с базой данных.
        run_id: Идентификатор прогона.
        method_name: Имя завершённого метода.
        result: Результат кросс-валидации метода.
        checkpoint_dir: Директория для файлов checkpoint. ``None`` —
            временная системная папка.
    """
    checkpoint_path = _checkpoint_file(run_id, checkpoint_dir)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # Текущее состояние: читаем существующий checkpoint, если он есть.
    state = load_checkpoint(conn, run_id)
    if state is not None:
        completed_methods = list(state.completed_methods)
        partial_results = dict(state.partial_results)
    else:
        completed_methods = []
        partial_results = {}

    partial_results[method_name] = result
    if method_name not in completed_methods:
        completed_methods.append(method_name)

    # Файл состояния: сериализуем результаты завершённых методов.
    payload = {
        "run_id": run_id,
        "completed_methods": completed_methods,
        "partial_results": {name: asdict(res) for name, res in partial_results.items()},
    }
    checkpoint_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Строка в таблице `checkpoints`: upsert по первичному ключу `run_id`.
    conn.execute(
        """
        INSERT INTO checkpoints (run_id, completed_methods, saved_at, checkpoint_path)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(run_id) DO UPDATE SET
            completed_methods = excluded.completed_methods,
            saved_at = excluded.saved_at,
            checkpoint_path = excluded.checkpoint_path
        """,
        (
            run_id,
            json.dumps(completed_methods),
            _utc_now(),
            str(checkpoint_path),
        ),
    )
    conn.commit()


def load_checkpoint(
    conn: sqlite3.Connection,
    run_id: str,
) -> CheckpointState | None:
    """Загружает сохранённое состояние прогона.

    Args:
        conn: Соединение с базой данных.
        run_id: Идентификатор прогона.

    Returns:
        Состояние прогона или ``None``, если checkpoint отсутствует —
        прогон ещё не сохранялся либо файл состояния не найден на диске.
    """
    row = conn.execute(
        "SELECT completed_methods, checkpoint_path FROM checkpoints WHERE run_id = ?",
        (run_id,),
    ).fetchone()
    if row is None:
        return None

    completed_methods: list[str] = json.loads(row[0])
    checkpoint_path = Path(row[1])

    # Файл состояния — источник результатов методов. Без него восстановить
    # `partial_results` нельзя: считаем checkpoint недоступным.
    if not checkpoint_path.is_file():
        return None

    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    stored_results: dict = payload.get("partial_results", {})
    partial_results = {
        name: CVResult(**res) for name, res in stored_results.items()
    }

    return CheckpointState(
        run_id=run_id,
        completed_methods=completed_methods,
        partial_results=partial_results,
        checkpoint_path=checkpoint_path,
    )
