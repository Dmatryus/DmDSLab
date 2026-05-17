"""Тесты checkpointing (подзадача E-006.3).

Покрывают критерии DoD: работу ``resolve_run_id`` / ``save_checkpoint`` /
``load_checkpoint``, восстановление прерванного прогона с пропуском
завершённых методов, отсутствие checkpoint (``None``). Таблица
``checkpoints`` создаётся in-memory по схеме ARCHITECTURE §5 — ``storage/db.py``
(эпик E-007) не задействован.
"""

import sqlite3

import pytest

from feature_selection_benchmark.core.orchestrator import CVResult
from feature_selection_benchmark.storage.checkpoint import (
    CheckpointState,
    load_checkpoint,
    resolve_run_id,
    save_checkpoint,
)

# Схема таблицы `checkpoints` — дословно из ARCHITECTURE §5.
_CREATE_CHECKPOINTS = """
CREATE TABLE checkpoints (
    run_id            TEXT PRIMARY KEY REFERENCES runs(id),
    completed_methods TEXT NOT NULL,
    saved_at          TEXT NOT NULL,
    checkpoint_path   TEXT NOT NULL
);
"""


@pytest.fixture
def conn() -> sqlite3.Connection:
    """In-memory соединение с созданной таблицей ``checkpoints``."""
    connection = sqlite3.connect(":memory:")
    connection.execute(_CREATE_CHECKPOINTS)
    connection.commit()
    yield connection
    connection.close()


def _result(score: float = 0.9, features: list[str] | None = None) -> CVResult:
    """Эталонный результат кросс-валидации одного метода."""
    return CVResult(
        cv_score=score,
        score_std=0.01,
        selected_features=features if features is not None else ["f1", "f2"],
        duration_sec=1.5,
    )


# --- resolve_run_id ---------------------------------------------------------


def test_resolve_run_id_new_returns_hash(conn: sqlite3.Connection) -> None:
    """Для нового хэша ``run_id`` совпадает с хэшем параметров."""
    assert resolve_run_id(conn, "hash-abc") == "hash-abc"


def test_resolve_run_id_idempotent(conn: sqlite3.Connection) -> None:
    """Повторный вызов с тем же хэшем даёт тот же ``run_id``."""
    first = resolve_run_id(conn, "hash-xyz")
    second = resolve_run_id(conn, "hash-xyz")
    assert first == second == "hash-xyz"


def test_resolve_run_id_existing_checkpoint(conn: sqlite3.Connection, tmp_path) -> None:
    """Существующий checkpoint возвращает свой ``run_id`` для возобновления."""
    run_id = resolve_run_id(conn, "hash-resume")
    save_checkpoint(conn, run_id, "filter_corr", _result(), checkpoint_dir=tmp_path)
    assert resolve_run_id(conn, "hash-resume") == run_id


def test_resolve_run_id_rejects_empty(conn: sqlite3.Connection) -> None:
    """Пустой ``params_hash`` отвергается."""
    with pytest.raises(ValueError):
        resolve_run_id(conn, "")


# --- load_checkpoint без сохранения -----------------------------------------


def test_load_checkpoint_absent_returns_none(conn: sqlite3.Connection) -> None:
    """Для неизвестного прогона ``load_checkpoint`` возвращает ``None``."""
    assert load_checkpoint(conn, "unknown-run") is None


# --- save / load round-trip -------------------------------------------------


def test_save_then_load_single_method(conn: sqlite3.Connection, tmp_path) -> None:
    """Сохранённый результат метода восстанавливается без потерь."""
    run_id = resolve_run_id(conn, "hash-1")
    result = _result(score=0.87, features=["a", "b", "c"])
    save_checkpoint(conn, run_id, "filter_corr", result, checkpoint_dir=tmp_path)

    state = load_checkpoint(conn, run_id)
    assert isinstance(state, CheckpointState)
    assert state.run_id == run_id
    assert state.completed_methods == ["filter_corr"]
    assert state.partial_results["filter_corr"] == result
    assert state.checkpoint_path.is_file()


def test_save_checkpoint_writes_file_in_checkpoint_dir(
    conn: sqlite3.Connection, tmp_path
) -> None:
    """Файл состояния создаётся именно в указанной ``checkpoint_dir``."""
    run_id = resolve_run_id(conn, "hash-dir")
    save_checkpoint(conn, run_id, "embedded_lasso", _result(), checkpoint_dir=tmp_path)

    state = load_checkpoint(conn, run_id)
    assert state is not None
    assert state.checkpoint_path.parent == tmp_path


def _checkpoint_cleanup_path(run_id: str):
    """Путь к временному файлу checkpoint для очистки после теста."""
    import tempfile
    from pathlib import Path

    return Path(tempfile.gettempdir()) / f"checkpoint_{run_id}.json"


def test_save_checkpoint_default_dir_is_temp(conn: sqlite3.Connection) -> None:
    """При ``checkpoint_dir=None`` файл пишется во временную системную папку."""
    import tempfile
    from pathlib import Path

    run_id = resolve_run_id(conn, "hash-temp")
    try:
        save_checkpoint(conn, run_id, "shap_method", _result())
        state = load_checkpoint(conn, run_id)
        assert state is not None
        assert state.checkpoint_path.parent == Path(tempfile.gettempdir())
        assert state.checkpoint_path.is_file()
    finally:
        path = _checkpoint_cleanup_path("hash-temp")
        if path.exists():
            path.unlink()


# --- многометодный прогон / восстановление ----------------------------------


def test_save_accumulates_multiple_methods(conn: sqlite3.Connection, tmp_path) -> None:
    """Несколько сохранений накапливают завершённые методы и их результаты."""
    run_id = resolve_run_id(conn, "hash-multi")
    r1 = _result(score=0.80, features=["a"])
    r2 = _result(score=0.91, features=["a", "b"])

    save_checkpoint(conn, run_id, "filter_corr", r1, checkpoint_dir=tmp_path)
    save_checkpoint(conn, run_id, "wrapper_rfe", r2, checkpoint_dir=tmp_path)

    state = load_checkpoint(conn, run_id)
    assert state is not None
    assert state.completed_methods == ["filter_corr", "wrapper_rfe"]
    assert state.partial_results["filter_corr"] == r1
    assert state.partial_results["wrapper_rfe"] == r2


def test_resume_skips_completed_methods(conn: sqlite3.Connection, tmp_path) -> None:
    """Прерванный прогон восстанавливается с пропуском завершённых методов."""
    all_methods = ["filter_corr", "wrapper_rfe", "embedded_lasso", "shap_method"]
    run_id = resolve_run_id(conn, "hash-interrupted")

    # Первый запуск: успели завершить только два метода и «прервались».
    save_checkpoint(conn, run_id, "filter_corr", _result(), checkpoint_dir=tmp_path)
    save_checkpoint(conn, run_id, "wrapper_rfe", _result(), checkpoint_dir=tmp_path)

    # Возобновление: новое соединение, новый resolve по тому же хэшу.
    resumed_id = resolve_run_id(conn, "hash-interrupted")
    state = load_checkpoint(conn, resumed_id)
    assert state is not None

    remaining = [m for m in all_methods if m not in state.completed_methods]
    assert remaining == ["embedded_lasso", "shap_method"]


def test_resave_same_method_overwrites_without_duplicate(
    conn: sqlite3.Connection, tmp_path
) -> None:
    """Повторное сохранение метода обновляет результат, не дублируя имя."""
    run_id = resolve_run_id(conn, "hash-resave")
    save_checkpoint(
        conn, run_id, "filter_corr", _result(score=0.5), checkpoint_dir=tmp_path
    )
    save_checkpoint(
        conn, run_id, "filter_corr", _result(score=0.95), checkpoint_dir=tmp_path
    )

    state = load_checkpoint(conn, run_id)
    assert state is not None
    assert state.completed_methods == ["filter_corr"]
    assert state.partial_results["filter_corr"].cv_score == 0.95


def test_checkpoints_table_has_single_row_per_run(
    conn: sqlite3.Connection, tmp_path
) -> None:
    """В таблице ``checkpoints`` остаётся ровно одна строка на прогон (upsert)."""
    run_id = resolve_run_id(conn, "hash-onerow")
    save_checkpoint(conn, run_id, "filter_corr", _result(), checkpoint_dir=tmp_path)
    save_checkpoint(conn, run_id, "wrapper_rfe", _result(), checkpoint_dir=tmp_path)

    count = conn.execute(
        "SELECT COUNT(*) FROM checkpoints WHERE run_id = ?", (run_id,)
    ).fetchone()[0]
    assert count == 1


def test_load_returns_none_if_state_file_missing(
    conn: sqlite3.Connection, tmp_path
) -> None:
    """Строка в БД есть, но файл состояния удалён — checkpoint недоступен."""
    run_id = resolve_run_id(conn, "hash-nofile")
    save_checkpoint(conn, run_id, "filter_corr", _result(), checkpoint_dir=tmp_path)

    state = load_checkpoint(conn, run_id)
    assert state is not None
    state.checkpoint_path.unlink()

    assert load_checkpoint(conn, run_id) is None
