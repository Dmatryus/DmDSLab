# Changelog

All notable changes to `feature_selection_benchmark` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-05-17

Эпик E-002 (Research): зафиксирована стратегия идентификации записей
глобального лидерборда и поставлена функция генерации `dataset_id`.

### Added

- `storage/dataset_id.py` — `compute_dataset_id(name, data)`: генерация
  идентификатора датасета по стратегии «имя + структурный fingerprint схемы».
  `dataset_id = "{name}::{fingerprint}"`, где `fingerprint` — SHA-256 от имён
  колонок и dtype (не от значений ячеек и не от числа строк).
- `docs/adr/0001-dataset-id-strategy.md` — ADR с принятым решением,
  рассмотренными альтернативами и спецификацией контракта для эпиков
  E-004 (вход `dataset_name`) и E-007.
- `tests/test_dataset_id.py` — 9 тестов: детерминизм между процессами,
  различение схем и dtype, стабильность при изменении числа строк, валидация.

### Changed

- `OVERVIEW.md` §9 Q1 (идентификация записи лидерборда) помечен resolved
  со ссылкой на ADR 0001.
- `ARCHITECTURE.md` §2 — карта пакета дополнена модулем `storage/dataset_id.py`.

### Known Limitations

- `compute_dataset_id` не реэкспортируется в `storage/__init__.py` — следует
  конвенции пакета (`db` / `leaderboard` / `checkpoint` тоже не реэкспортируются);
  импорт — `from feature_selection_benchmark.storage.dataset_id import …`.
- Стратегия не различает датасеты с одинаковыми именем и схемой, но разными
  значениями — такие записи сливаются в лидерборде (см. ADR 0001).
- Вход `dataset_name` в публичный API (`run_benchmark`) ещё не добавлен —
  это правка контракта для эпика E-004, специфицирована в ADR 0001.

## [0.1.0] - 2026-05-17

Первый релиз — каркас пакета. Этот релиз поставляет структуру проекта,
стабильные публичные контракты и заглушки модулей; реализация FS-методов и
оркестрации появится в последующих эпиках (E-004…E-009).

### Added

- Дерево директорий и упаковка пакета `feature_selection_benchmark`
  (flat-layout, `__init__.py` для всех подпакетов: `methods`, `gui`, `tests`).
- `pyproject.toml`: декларативная конфигурация (`setuptools.build_meta`),
  зависимости рантайма, extras `[dev]` и `[methods]`, требование Python >= 3.10.
- Стабильные публичные контракты: `Dataset`, `BenchmarkResult`, `FSMethod`,
  `MethodInfo`, `CVResult`, `CheckpointState` (с `from __future__ import annotations`).
- Заглушки всех модулей с финальными сигнатурами: API-слой (`run_benchmark`,
  `get_leaderboard`, `list_methods`, `register_method`), реестр методов,
  оркестрация, хранилище (SQLite), лидерборды — каждая бросает
  `NotImplementedError`.
- `README.md`: обзор проекта, Quickstart, раздел Contributing.
- Конфигурация линтера: `ruff` (правила `E,W,F,I,UP,B`), `mypy` (пониженная
  строгость под каркас), `.pre-commit-config.yaml`.
- Smoke-тест `tests/test_scaffold.py` — 34 теста: проверка структуры пакета,
  импортируемости контрактов и `NotImplementedError` у заглушек.

### Known Limitations

- Все модули — заглушки: `run_benchmark` и FS-методы бросают
  `NotImplementedError`. Функциональная реализация распределена по эпикам
  E-004…E-009.
- Нижние границы версий зависимостей в `pyproject.toml` проставлены по
  актуальным релизам и будут выверены при интеграции ML-методов (OQ-1).
- Генерация `dataset_id` не определена — отложена в эпик E-002; каркас
  фиксирует только тип `dataset_id: str`.
- `pre-commit` требует флага `--config experiments/feature_selection_benchmark/.pre-commit-config.yaml`,
  поскольку конфиг лежит в подкаталоге пакета, а не в корне монорепо
  (отражено в README Contributing).

[0.2.0]: https://github.com/Dmatryus/DmDSLab/releases/tag/v0.2.0
[0.1.0]: https://github.com/Dmatryus/DmDSLab/releases/tag/v0.1.0
