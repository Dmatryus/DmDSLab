# Changelog

All notable changes to `feature_selection_benchmark` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.1.0]: https://github.com/Dmatryus/DmDSLab/releases/tag/v0.1.0
