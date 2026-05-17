"""Подбор гиперпараметров FS-методов.

`tune_hyperparams` подбирает гиперпараметры одного FS-метода через
Optuna со сэмплером TPE (Tree-structured Parzen Estimator) — стратегия
зафиксирована в ADR `0002` (вариант E).

Кратко (спецификация ADR 0002):

- метод декларирует пространство поиска словарём
  ``MethodInfo.hyperparameters: dict[str, HyperParam]`` — обычные данные,
  без зависимости от Optuna; трансляцию `HyperParam` → `trial.suggest_*`
  делает только этот модуль;
- objective сэмплирует гиперпараметры из этого пространства и возвращает
  CV-оценку набора признаков через `CVRunner.run_cv` (направление
  «больше = лучше» → study с ``direction="maximize"``);
- `TPESampler` сидируется `random_seed` — при фиксированном seed
  последовательность trials и `best_params` детерминированы (критерий К2);
- при ``n_trials == 0`` study не создаётся — возвращаются дефолтные
  гиперпараметры метода (поле `HyperParam.default`);
- метод без настраиваемых гиперпараметров (`hyperparameters == {}`) —
  подбор пропускается, возвращается пустой dict.
"""

from __future__ import annotations

import optuna

from ..api import Dataset
from ..methods.base import FSMethod, HyperParam
from .orchestrator import CVRunner

__all__ = ["tune_hyperparams"]


def _suggest(trial: optuna.Trial, name: str, param: HyperParam) -> object:
    """Транслирует declarative-дескриптор `HyperParam` в вызов `trial.suggest_*`.

    Args:
        trial: Текущий Optuna-trial.
        name: Имя гиперпараметра.
        param: Declarative-дескриптор пространства поиска параметра.

    Returns:
        Сэмплированное значение гиперпараметра.

    Raises:
        ValueError: Если `param.kind` не из ``{"int", "float",
            "categorical"}`` (защищено `Literal`-типом и валидацией
            `HyperParam.__post_init__`).
    """
    match param.kind:
        case "int":
            return trial.suggest_int(
                name, int(param.low), int(param.high), log=param.log
            )
        case "float":
            return trial.suggest_float(
                name, float(param.low), float(param.high), log=param.log
            )
        case "categorical":
            return trial.suggest_categorical(name, param.choices)
        case _:  # pragma: no cover - защищено Literal-типом HyperParam
            raise ValueError(f"HyperParam: неизвестный kind {param.kind!r}")


def tune_hyperparams(
    method: FSMethod,
    dataset: Dataset,
    n_trials: int = 20,
    random_seed: int | None = None,
) -> dict[str, object]:
    """Подбирает гиперпараметры FS-метода на датасете.

    Стратегия — Optuna TPE (ADR `0002`). Создаётся study с
    ``direction="maximize"`` и `TPESampler(seed=random_seed)`; objective
    сэмплирует гиперпараметры из declarative-пространства поиска метода и
    возвращает CV-оценку набора признаков (`CVRunner.run_cv`). После
    `n_trials` итераций возвращается `study.best_params`.

    Особые режимы:

    - ``n_trials == 0`` — study не создаётся; возвращаются дефолтные
      гиперпараметры метода (поле `HyperParam.default`);
    - метод без настраиваемых гиперпараметров — возвращается пустой dict.

    Воспроизводимость (К2): при фиксированном `random_seed` сэмплер
    детерминирован, поэтому `best_params` совпадают между запусками. При
    ``random_seed=None`` подбор невоспроизводим.

    Args:
        method: Метод отбора признаков.
        dataset: Входной датасет.
        n_trials: Число итераций подбора. ``0`` — режим дефолтов без study.
        random_seed: Зерно случайности для воспроизводимости — сидирует
            `TPESampler` и `CVRunner` (фолды, прокси-модель).

    Returns:
        Словарь лучших найденных гиперпараметров (``имя → значение``).
        Пустой dict — если у метода нет настраиваемых гиперпараметров.
    """
    search_space = method.method_info().hyperparameters

    # Метод без настраиваемых гиперпараметров — подбор пропускается.
    if not search_space:
        return {}

    # Режим дефолтов: study не создаётся (ADR 0002, n_trials=0).
    if n_trials == 0:
        return {
            name: param.default for name, param in search_space.items()
        }

    runner = CVRunner(random_seed=random_seed)

    def objective(trial: optuna.Trial) -> float:
        """Сэмплирует гиперпараметры и возвращает CV-оценку набора признаков.

        Args:
            trial: Текущий Optuna-trial.

        Returns:
            `cv_score` набора признаков, отобранного методом на
            сэмплированных гиперпараметрах (больше — лучше).
        """
        params = {
            name: _suggest(trial, name, param)
            for name, param in search_space.items()
        }
        return runner.run_cv(method, dataset, params).cv_score

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_seed),
    )
    study.optimize(objective, n_trials=n_trials)
    return study.best_params
