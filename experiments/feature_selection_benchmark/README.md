# feature_selection_benchmark

> **Edit log:**
> - 2026-05-17 · v0.3.0 · execution-agent · создан

Приложение для автоматического сравнения методов отбора признаков
(filter / wrapper / embedded / гибридные) на табличных данных.

## Назначение

`feature_selection_benchmark` устраняет ручной несистематичный перебор
методов отбора признаков. Data scientist передаёт датасет и тип задачи
(классификация или регрессия), приложение прогоняет все зарегистрированные
FS-методы — filter, wrapper, embedded, SHAP-based, permutation importance,
null importance и другие — и возвращает ранжированный лидерборд лучших
наборов признаков:

- **локальный лидерборд** — результаты текущего запуска;
- **глобальный лидерборд** — лучшие результаты по всем запускам
  (персистентность в SQLite).

Одновременно инструмент даёт разработчикам AutoML единый интерфейс и
зафиксированные best practices для добавления новых FS-методов.

Подробнее: [`OVERVIEW.md`](OVERVIEW.md) (обзор и скоуп),
[`ARCHITECTURE.md`](ARCHITECTURE.md) (стек и API).

## Статус

> **В разработке.** Реализованы слой FS-методов и реестр; ядро
> оркестрации, хранилище и API-слой пока остаются заглушками.

Что уже реализовано:

- дерево пакета по `ARCHITECTURE.md` §2;
- упаковка через `pyproject.toml` (PEP 517/518);
- стабильные контракты данных (`Dataset`, `BenchmarkResult`, `FSMethod`,
  `MethodInfo`, `HyperParam`);
- **реестр FS-методов** (`methods/registry.py`) — саморегистрация
  built-in методов при импорте пакета, graceful degradation для методов
  с отсутствующими опц. зависимостями;
- **все 14 FS-методов** четырёх групп — filter, wrapper, embedded,
  shap (включая permutation importance).

Ещё является заглушкой (поднимает `NotImplementedError`):

- `run_benchmark` и API-слой валидации (эпик E-004);
- ядро оркестрации — CV, подбор гиперпараметров, checkpointing
  (эпик E-006);
- хранилище — SQLite, лидерборд (эпик E-007).

## Требования

- **Python ≥ 3.10** (используются `match`/`case` и синтаксис `X | Y`
  для типов; верхней границы версии нет).
- Только локальная Python-среда — облачный деплой и Docker не предусмотрены.

## Установка

Пакет расположен в монорепо DmDSLab по пути
`experiments/feature_selection_benchmark/`, имя пакета —
`feature_selection_benchmark`. Установка через `pip` в режиме editable.

**Core-каркас** (`numpy`, `pandas`, `scikit-learn`, `joblib`, `tqdm`,
`tqdm_joblib`):

```bash
pip install -e "experiments/feature_selection_benchmark"
```

**Core + тяжёлые ML-зависимости FS-методов** (extra `[methods]`: `shap`,
`catboost`, `BorutaShap`, `mrmr-selection`, `target-permutation-importances`):

```bash
pip install -e "experiments/feature_selection_benchmark[methods]"
```

**Core + dev-инструменты** (extra `[dev]`: `pytest`, `ruff`, `mypy`,
`pre-commit`):

```bash
pip install -e "experiments/feature_selection_benchmark[dev]"
```

Extra можно комбинировать, например `[methods,dev]`.

### Заметка для Windows

Extra `[methods]` тянет `catboost` и `shap`, для сборки которых на Windows
могут потребоваться **Microsoft C++ Build Tools** (если для вашей версии
Python/платформы нет готовых wheel-пакетов). Их можно установить из
[Build Tools for Visual Studio](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
(компонент «Desktop development with C++»).

Core-каркас (`pip install -e "experiments/feature_selection_benchmark"`)
ставится **без** `[methods]` и C++ build tools не требует — для работы с
текущим каркасом достаточно core-установки.

## Quickstart

> Приведённый пример отражает целевой API. Слой FS-методов и реестр уже
> реализованы, но `run_benchmark` пока остаётся заглушкой, поднимающей
> `NotImplementedError`. Пример показывает, как вызов будет выглядеть
> после реализации API-слоя и оркестрации (эпики E-004, E-006, E-007).

```python
from feature_selection_benchmark import run_benchmark, Dataset

result = run_benchmark(
    Dataset(
        data=df,                       # pandas.DataFrame: признаки + target
        features_list=[...],           # список имён колонок-признаков
        target="target",               # имя целевой колонки
        id="row_id",                   # колонка-идентификатор строки
    ),
    task="classification",             # или "regression"; None → автоопределение
    cv=5,                              # число разбиений кросс-валидации
    n_trials=20,                       # итераций подбора гиперпараметров
    random_seed=42,                    # фиксация для воспроизводимости
)
print(result.ranked_results)
```

`run_benchmark` возвращает `BenchmarkResult` с полем `ranked_results`
(`pandas.DataFrame` с колонками `method_name`, `group`, `selected_features`,
`cv_score`, `score_std`, `rank`, `duration_sec`). Полное описание сигнатур —
в [`ARCHITECTURE.md`](ARCHITECTURE.md) §4.

## Contributing

1. Клонировать репозиторий DmDSLab.
2. Установить пакет с dev-инструментами:

   ```bash
   pip install -e "experiments/feature_selection_benchmark[dev]"
   ```

3. Установить git-хуки pre-commit (ruff + mypy). Конфиг `.pre-commit-config.yaml`
   лежит в каталоге пакета (монорепо), поэтому команды запускаются из корня
   репозитория с явным `--config`:

   ```bash
   pre-commit install --config experiments/feature_selection_benchmark/.pre-commit-config.yaml
   ```

   После этого `ruff` и `mypy` запускаются автоматически на каждом коммите.

4. (Опционально) Прогнать все хуки по всей кодовой базе вручную — например,
   перед первым коммитом или после обновления конфига:

   ```bash
   pre-commit run --all-files --config experiments/feature_selection_benchmark/.pre-commit-config.yaml
   ```

Запуск тестов:

```bash
pytest experiments/feature_selection_benchmark/tests/
```

При добавлении нового FS-метода соблюдайте интерфейс `FSMethod` и регистрацию
через `register_method` — ядро (`core/orchestrator.py`, `methods/registry.py`)
при этом не меняется. Подробности — в `CLAUDE.md` (раздел HOW) и
`ARCHITECTURE.md` §4.

## Лицензия

MIT.
