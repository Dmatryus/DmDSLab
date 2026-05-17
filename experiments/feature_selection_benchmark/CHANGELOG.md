# Changelog

All notable changes to `feature_selection_benchmark` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-05-17

Эпик E-005 (Functional): поставлен реестр FS-методов и реализованы все
14 built-in методов четырёх групп (filter / wrapper / embedded / SHAP +
permutation). Импорт пакета автоматически регистрирует все методы.

### Added

- 14 реализованных FS-методов в `methods/{filter,wrapper,embedded,shap}_methods.py`,
  покрывающих группы filter, wrapper, embedded и SHAP/permutation. Каждый метод
  объявляет `MethodInfo` и декларативное пространство поиска гиперпараметров
  согласно ADR `docs/adr/0002-hyperparameter-tuning-strategy.md`.
- Реестр методов `methods/registry.py` — `register`, `get_method`,
  `list_registered`, `is_registered`, `default_method_names`; built-in методы
  саморегистрируются при импорте подпакета.
- Тип `HyperParam` — дескриптор пространства поиска одного гиперпараметра
  (`kind`, `default`, границы), используется методами при объявлении
  `MethodInfo.hyperparameters`.
- `FSMethod.check_availability()` — classmethod, проверяющий наличие опциональных
  зависимостей метода; даёт graceful degradation при отсутствии extra `[methods]`.
- Модуль `methods/_utils.py` — утилиты методов, в том числе каноническая
  функция `is_classification` (единое определение типа задачи для всех групп).
- Реэкспорт `HyperParam`, `FSMethod`, `MethodInfo` из корневого
  `feature_selection_benchmark/__init__.py` — публичный API для разработчиков
  FS-методов (Персона 2).
- Тесты `tests/test_methods/` — `test_filter.py`, `test_wrapper.py`,
  `test_embedded.py`, `test_shap.py`.

### Changed

- **Breaking.** `MethodInfo.hyperparameters` сменил тип с `dict[str, Any]` на
  `dict[str, HyperParam]`. Код, конструирующий `MethodInfo` напрямую с
  произвольными значениями в `hyperparameters`, несовместим с новым контрактом
  и функциями реестра — см. раздел Migration ниже.
- `MethodInfo` расширен 4 полями с дефолтами (`group`, `supported_tasks`,
  `available`, `unavailable_reason`) — обратно совместимо для конструктора.

### Migration

`MethodInfo.hyperparameters` теперь типизирован как `dict[str, HyperParam]`.

При создании `MethodInfo` вручную замените произвольные значения в
`hyperparameters` на экземпляры `HyperParam`:

```python
# Было (0.2.0 и ранее):
MethodInfo(name="my_method", hyperparameters={"k": 10})

# Стало (0.3.0):
from feature_selection_benchmark import HyperParam
MethodInfo(name="my_method", hyperparameters={"k": HyperParam(kind="int", default=10)})
```

Старый `dict[str, Any]` несовместим с новыми функциями реестра. Подробная
migration-note — `.task/migration-v0.3.0.md`.

### Known Limitations

- Методы `mrmr`, `boruta`, `shap_importance`, `null_importance` зависят от
  опциональных пакетов extra `[methods]` (`shap`, `BorutaShap`,
  `mrmr-selection`, `target-permutation-importances`). В тестовой среде эти
  пакеты не установлены — соответствующие пути `fit_select` не верифицированы
  исполнением (5 skipped-тестов, graceful degradation через
  `FSMethod.check_availability()`). Реальный прогон требует CI с extra
  `[methods]`.
- `run_benchmark` и API-слой (`api.py`) — ещё заглушки, бросают
  `NotImplementedError`; реализация — эпик E-004.
- Ядро оркестрации (CV, подбор гиперпараметров, checkpointing) — заглушки;
  реализация — эпик E-006.
- Хранилище и лидерборды (SQLite) — заглушки; реализация — эпик E-007.

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

[0.3.0]: https://github.com/Dmatryus/DmDSLab/releases/tag/v0.3.0
[0.2.0]: https://github.com/Dmatryus/DmDSLab/releases/tag/v0.2.0
[0.1.0]: https://github.com/Dmatryus/DmDSLab/releases/tag/v0.1.0
