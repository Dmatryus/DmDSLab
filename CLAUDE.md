# feature_selection_benchmark

> **Edit log:**
> - 2026-05-16 · v0.3.0 · claude-md-generator · создан

## WHY

`feature_selection_benchmark` существует, чтобы устранить ручной несистематичный перебор методов отбора признаков: data scientist передаёт датасет и тип задачи, приложение прогоняет все методы (filter, wrapper, embedded, SHAP-based и другие) и возвращает ранжированный лидерборд. Одновременно инструмент даёт разработчикам AutoML единый интерфейс и зафиксированные best practices для добавления новых FS-методов.

## WHAT

Единая точка входа — `run_benchmark(dataset, task, methods, cv, n_trials, checkpoint_dir, random_seed, baseline_model)`. Функция запускает зарегистрированные FS-методы через оркестратор с кросс-валидацией, подбором гиперпараметров и checkpointing; результаты сохраняются в локальный и глобальный лидерборд (SQLite). Пакет находится в монорепо DmDSLab по пути `experiments/feature_selection_benchmark/`.

```
experiments/feature_selection_benchmark/
├── pyproject.toml
├── feature_selection_benchmark/
│   ├── __init__.py                  # публичный API: run_benchmark, get_leaderboard, list_methods, register_method
│   ├── api.py                       # точка входа, валидация, Dataset dataclass
│   ├── core/
│   │   ├── orchestrator.py          # прогон методов, CV, checkpointing
│   │   ├── tuning.py                # подбор гиперпараметров
│   │   └── reproducibility.py      # random_seed, детерминированная сортировка
│   ├── methods/
│   │   ├── registry.py              # реестр: name → (callable, group, supported_tasks)
│   │   ├── base.py                  # базовый класс FSMethod
│   │   ├── filter_methods.py
│   │   ├── wrapper_methods.py
│   │   ├── embedded_methods.py
│   │   └── shap_methods.py
│   ├── storage/
│   │   ├── db.py                    # SQLite-соединение, миграции
│   │   ├── leaderboard.py           # чтение/запись/ранжирование
│   │   └── checkpoint.py            # сохранение/загрузка состояния прогона
│   └── reporting/
│       └── formatter.py             # вывод, прогресс tqdm, уведомление о baseline
└── tests/
    ├── test_api.py
    ├── test_orchestrator.py
    ├── test_tuning.py
    ├── test_reproducibility.py      # допуск 1e-6, критерий воспроизводимости
    ├── test_leaderboard.py
    └── test_methods/
        ├── test_filter.py
        ├── test_wrapper.py
        ├── test_embedded.py
        └── test_shap.py
```

## HOW

**Установка:**
```bash
pip install -e "experiments/feature_selection_benchmark[dev]"
```

**Запуск бенчмарка (Python API):**
```python
from feature_selection_benchmark import run_benchmark, Dataset

result = run_benchmark(
    dataset=Dataset(data=df, features_list=[...], target="target", id="row_id"),
    task="classification",       # или "regression"; None → автоопределение
    methods=None,                # None → все зарегистрированные методы
    cv=5,
    n_trials=20,
    random_seed=42,
)
print(result.ranked_results)
```

**Тесты:**
```bash
pytest experiments/feature_selection_benchmark/tests/
```

**Добавление нового FS-метода:**
```python
from feature_selection_benchmark import register_method
from feature_selection_benchmark.methods.base import FSMethod

class MyMethod(FSMethod):
    name = "my_method"
    group = "filter"
    supported_tasks = ["classification", "regression"]

    def fit_select(self, X_train, y_train, **hyperparams) -> list[str]:
        ...

register_method("my_method", MyMethod(), group="filter")
```

Конфиг-файлов нет — все параметры передаются аргументами `run_benchmark`. Дефолты: `cv=5`, `n_trials=20`, `checkpoint_dir=None` (временная папка), `random_seed=None`.

## CONSTRAINTS

1. **Только табличные данные.** Вход — `numpy`-массивы или `pandas.DataFrame`. Изображения, текст, временные ряды не поддерживаются.
2. **Задачи — только классификация и регрессия.** Кластеризация, ранжирование и другие типы задач вне скоупа.
3. **Производительность зависит от железа пользователя.** Жёсткого лимита на размер датасета нет; предел — доступная память и CPU.
4. **Предобработка — ответственность пользователя.** Пропущенные значения, кодирование категориальных переменных и нормализация должны быть выполнены до вызова `run_benchmark`.
5. **Только локальная Python-среда.** Облачный деплой и Docker не предусмотрены.
6. **Python ≥ 3.10** — используется `match`/`case` и синтаксис `X | Y` для типов.
7. **GUI — Should-уровень.** Ядро не зависит от `gui/`; Dear PyGui не является блокером для v1.

## NEVER

1. Не обучать финальную модель и не делать предсказания — инструмент только отбирает признаки.
2. Не подбирать алгоритм или гиперпараметры финальной модели — это не AutoML целевой модели.
3. Не хранить, не версионировать и не каталогизировать пользовательские датасеты — `Dataset.data` живёт только в RAM и никогда не пишется на диск.
4. Не использовать SHAP как инструмент объяснимости модели — только как метод отбора признаков.
5. Не добавлять поддержку нетабличных данных (изображения, текст, временные ряды) ни в какой версии.
6. Не реализовывать облачный деплой, веб-сервис с публичным лидербордом или интеграцию с MLflow / W&B.
7. Не вносить изменения в `core/orchestrator.py` или `methods/registry.py` при добавлении нового FS-метода — достаточно одного файла метода и одной строки регистрации.

## PRINCIPLES

1. **Think Before Coding** — сначала думать, потом писать код. Прочитай релевантные файлы, сформулируй план, и только потом меняй код.
2. **Simplicity First** — простота важнее «гибкости» и преждевременных абстракций. Три похожих строки лучше, чем абстракция ради воображаемого будущего.
3. **Surgical Changes** — менять только то, что просили. Никаких попутных улучшений, рефакторингов, переименований, которые пользователь не запрашивал.
4. **Goal-Driven Execution** — двигаться к заявленной цели. Не отвлекаться на побочные задачи, не уходить в ненужные исследования.
5. **One Question at a Time** — если нужно задать пользователю несколько вопросов, задавать их развёрнуто и строго по одному. Один вопрос → ответ → следующий вопрос. Не группировать, не нумеровать списком.
