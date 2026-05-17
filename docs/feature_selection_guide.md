# Отбор признаков для табличных моделей: прикладной гид с теорией, CatBoost-фокусом и Kaggle-практикой

## 0. Зачем вообще отбирать признаки

Стандартный аргумент «больше данных — лучше модель» в реальной практике разбивается о несколько ограничений:

- **Bias–variance и проклятие размерности.** При фиксированном `n` рост `p` увеличивает дисперсию оценки и разрежает пространство признаков; для бустингов это менее остро, чем для kNN/SVM, но проявляется через переобучение split-структуры.
- **Регуляризация ≠ отбор.** L2 «давит» все коэффициенты, но не зануляет; деревья при `max_depth=6` всё равно «съедают» шумовые признаки в ранних сплитах, особенно высококардинальные (см. Strobl и др., ниже).
- **Время обучения и инференса.** Для CatBoost/LightGBM `O(p·n·iter)` — снижение `p` с 2000 до 200 ускоряет обучение в 5–10× и инференс линейно по числу used features.
- **Robustness к дрифту.** Меньшее число фичей — меньше каналов для распределённого сдвига; ниже стоимость мониторинга.
- **Интерпретируемость и compliance.** В скоринге/медицине регуляторы требуют объяснимости каждого признака; «1000 фичей с автоматическим энкодингом» не пройдёт валидацию.

Классическая таксономия ([Guyon & Elisseeff, JMLR 2003](https://www.jmlr.org/papers/v3/guyon03a.html)): **filter** (модель-агностики, статистики X↔y), **wrapper** (внешняя модель оценивает подмножества), **embedded** (отбор внутри обучения — L1, деревья), и **hybrid** (например, фильтр → wrapper).

---

## 1. Filter-методы

Считаются один раз, дёшевы (`O(n·p)` или `O(n·p²)` для попарных корреляций), не учитывают модель.

| Метод | Что измеряет | Когда работает | Когда ломается |
|---|---|---|---|
| Variance threshold | Дисперсия X_j | Удалить квази-константы | Не учитывает y |
| Pearson | Линейная связь X_j ↔ y | Числовые, линейные эффекты | Нелинейности, выбросы |
| Spearman / Kendall | Монотонная связь | Robust к выбросам | Немонотонные зависимости |
| ANOVA F-test | Различие средних X_j по классам | Числовой X, категориальный y | Гетероскедастичность |
| χ² | Независимость категорий | Категориальный X и y | Малые ожидаемые частоты |
| Mutual Information | Произвольная зависимость | Нелинейности | Шумно при малых n, нужна оценка плотности |
| **mRMR** ([Peng, Long, Ding 2005](https://ieeexplore.ieee.org/document/1453511)) | MI(X_j;y) − ср. MI(X_j;X_k) | Когда важна нередундантность | Жадный, не глобальный оптимум |
| **IV / WoE** | Σ(g/G − b/B)·ln((g/G)/(b/B)) | Кредитный скоринг, монотонные бины | Чувствителен к биннингу |

В кредитном скоринге исторически закрепилось правило большого пальца: IV<0.02 — бесполезный, 0.02–0.1 — слабый, 0.1–0.3 — средний, 0.3–0.5 — сильный, >0.5 — подозрительно (вероятна утечка). WoE-преобразование обеспечивает монотонность для логистической регрессии и облегчает интерпретацию.

Фильтры используются как **первичный фильтр** до тяжёлых методов: «срезать» 5000 → 500 признаков за секунды.

```python
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
# Быстрый фильтр
keep = VarianceThreshold(0.0).fit(X).get_support()
mi = mutual_info_classif(X.loc[:, keep], y, random_state=0)
top = X.columns[keep][mi.argsort()[-500:]]
```

Для mRMR в Python — пакет [`mrmr-selection`](https://github.com/smazzanti/mrmr); работает поверх `pandas` и быстр на сотнях тысяч строк.

---

## 2. Wrapper-методы

Многократно обучают модель на подмножествах — точнее, но дорого: `O(p·model_train)` для backward, `O(p²)` для исчерпывающего поиска.

- **Forward/Backward/Stepwise selection** — стартует с пустого/полного набора, шагает по выигрышу метрики на CV. В sklearn — [`SequentialFeatureSelector`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SequentialFeatureSelector.html), в [`mlxtend`](http://rasbt.github.io/mlxtend/user_guide/feature_selection/SequentialFeatureSelector/) — `SFS` с расширениями (SFFS/SBFS).
- **RFE / RFECV** — рекурсивно удаляет признак с минимальной важностью базовой модели. [RFECV](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html) дополнительно подбирает количество фичей по CV.
- **Boruta** ([Kursa & Rudnicki, *Journal of Statistical Software*, 2010](https://www.jstatsoft.org/article/view/v036i11)) — wrapper по принципу «всех релевантных» признаков (`all relevant`, а не `minimal optimal`). Алгоритм:
  1. Для каждого признака `X_j` создаётся «теневой» (`shadow`) — копия с перемешанными значениями.
  2. Обучается Random Forest на расширенной матрице `[X | X_shadow]`.
  3. Считаются Z-scores важности; находится `MZSA = max(Z(shadow))`.
  4. Признак получает «hit», если его Z > MZSA.
  5. После M повторов выполняется биномиальный тест: фичи, систематически бьющие MZSA — `Confirmed`; систематически проигрывающие — `Rejected`; остальные — `Tentative`.
  6. Shadow-копии пересоздаются каждую итерацию; алгоритм останавливается, когда всё классифицировано или достигнут `maxRuns`.

  Сложность пропорциональна `n_objects × n_features × maxRuns × cost(RF)` — для 100K×500 на CPU это часы.

- **BorutaShap** (Eoghan Keany, 2020, репозиторий [Ekeany/Boruta-Shap](https://github.com/Ekeany/Boruta-Shap)) заменяет gini-importance на mean(|SHAP|) от TreeExplainer (XGBoost/LightGBM/CatBoost/sklearn) — стабильнее, без bias к high-cardinality, и поддерживает выборку подмножеств строк через isolation-forest + KS-test для ускорения до 5×.
- **Stability selection** ([Meinshausen & Bühlmann, JRSS-B 2010](https://rss.onlinelibrary.wiley.com/doi/10.1111/j.1467-9868.2010.00740.x)) — bootstrap × отбор; признак отбирается, если попадает в селекцию в ≥π_thr долях бутстрэпов. Контролирует FDR.
- **Генетические алгоритмы / SA** — [`sklearn-genetic-opt`](https://github.com/rodrigo-arenas/Sklearn-genetic-opt), [`DEAP`](https://github.com/DEAP/deap); редко окупаются по сравнению с Boruta+SHAP, но полезны при сложных constraint-ах (стоимость фичи, латентность).

---

## 3. Embedded: главное блюдо для CatBoost/XGBoost/LightGBM

### 3.1 Линейные модели — Lasso и Elastic Net
L1-регуляризация даёт точечную селекцию ([Tibshirani, JRSS-B 1996](https://www.jstor.org/stable/2346178)). При мультиколлинеарности Lasso «случайно» оставляет один из коррелированных — Elastic Net ([Zou & Hastie, JRSS-B 2005](https://hastie.su.domains/Papers/B67.2%20(2005)%20301-320%20Zou%20&%20Hastie.pdf)) лечит это через дополнительный L2.

### 3.2 Tree-based importance: что именно считается

**Gain (Gini importance, MDI):** сумма уменьшений критерия по всем сплитам с участием признака. В [XGBoost](https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.Booster.get_score) — `importance_type='gain'`; в [LightGBM](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.Booster.html#lightgbm.Booster.feature_importance) — `importance_type='gain'`; в [CatBoost](https://catboost.ai/docs/concepts/fstr.html) — `PredictionValuesChange` (по умолчанию для несимметричных метрик) или `LossFunctionChange`.

**Split (cover, weight):** количество использований признака (или взвешенное по hessian).

**Известные искажения** ([Strobl et al., *BMC Bioinformatics* 8:25, 2007](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-8-25); [Strobl et al., *BMC Bioinformatics* 9:307, 2008](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-9-307)):
1. **Смещение к high-cardinality / числовым.** Числовой признак с 1000 уникальных значений имеет больше потенциальных сплитов, чем категориальный с 2 уровнями, поэтому импьюрити-импортанс ему завышен. CatBoost частично страхует это через **ordered TS-кодирование** категорий, но не полностью.
2. **Корреляция «размывает» важность** — два почти одинаковых признака делят сигнал; ни один не выглядит важным.
3. **Маскировочный эффект:** один признак-доминант «скрывает» остальные, даже если у них есть unique сигнал.

Strobl et al. предложили **conditional permutation importance** — пермутировать `X_j` в стратах, определённых корреляциями с другими признаками; реализовано в R-пакете [`party`](https://cran.r-project.org/package=party).

### 3.3 CatBoost specifics — `select_features()`

Документация: [catboost.ai/docs/concepts/python-reference_catboostclassifier_select_features](https://catboost.ai/docs/concepts/python-reference_catboostclassifier_select_features). Метод реализует **Recursive Feature Elimination** с тремя вариантами оценки важности на каждом шаге:

| `algorithm` | Что используется | Скорость / точность |
|---|---|---|
| `RecursiveByPredictionValuesChange` | Δ предсказания при дефолте признака | Самый быстрый, неточный, не для ranking-loss |
| `RecursiveByLossFunctionChange` | Δ функции потерь на eval-set | **Оптимальный баланс**, рекомендован документацией |
| `RecursiveByShapValues` | TreeSHAP-агрегат | Самый точный, самый дорогой |

Ключевые параметры: `num_features_to_select` (или диапазон с `features_for_select`), `steps` (число переобучений — больше → точнее ранжирование), `train_final_model=True`, `shap_calc_type ∈ {Regular, Approximate, Exact}`.

```python
from catboost import CatBoostClassifier, Pool, EShapCalcType, EFeaturesSelectionAlgorithm

pool = Pool(X_train, y_train, cat_features=cat_cols)
eval_pool = Pool(X_val, y_val, cat_features=cat_cols)

model = CatBoostClassifier(iterations=2000, learning_rate=0.05, random_seed=42)
summary = model.select_features(
    pool,
    eval_set=eval_pool,
    features_for_select=list(range(X_train.shape[1])),
    num_features_to_select=50,
    steps=5,
    algorithm=EFeaturesSelectionAlgorithm.RecursiveByLossFunctionChange,
    shap_calc_type=EShapCalcType.Regular,
    train_final_model=True,
    logging_level='Silent',
    plot=True,
)
selected = summary['selected_features_names']
```

Замечание: `LossFunctionChange` требует `eval_set` и работает с любой loss; `PredictionValuesChange` не рекомендован для `YetiRank`/`PairLogit` (документация явно это указывает).

CatBoost также экспонирует **внутренние feature interactions** (`get_feature_importance(type='Interaction')`) — попарные силы взаимодействия по дереву.

### 3.4 LightGBM и XGBoost importance

LightGBM: `booster.feature_importance(importance_type='gain'|'split')`. XGBoost: `'gain'`, `'weight'`, `'cover'`, `'total_gain'`, `'total_cover'`. Gain — наиболее семантически осмысленный; weight/split — самый шумный, но иногда полезен для диагностики «переиспользуемых» признаков в shallow-деревьях.

---

## 4. SHAP-based отбор

SHAP ([Lundberg & Lee, NeurIPS 2017, arXiv:1705.07874](https://arxiv.org/abs/1705.07874)) — переоткрытие Shapley values из кооперативной теории игр для атрибуции вклада признака в индивидуальное предсказание. Свойства: **local accuracy, missingness, consistency** — последнее принципиально, поскольку [Lundberg, Erion & Lee, arXiv:1802.03888](https://arxiv.org/abs/1802.03888) показали, что классические gain-importance **inconsistent**: можно изменить модель так, что истинный вклад признака вырос, а его gain-importance упал.

**TreeSHAP** ([Lundberg et al., *Nature Machine Intelligence*, 2020](https://www.nature.com/articles/s42256-019-0138-9)) — точный полиномиальный алгоритм за `O(T·L·D²)` вместо `O(T·L·2^M)` для KernelSHAP; интегрирован в XGBoost, LightGBM, CatBoost (`get_feature_importance(type='ShapValues')`).

**Практическое правило отбора:** ранжировать признаки по `mean(|SHAP_j|)` на out-of-fold предсказаниях, отсекать по «локтю» или сравнить с null-распределением (см. §6). Для тонкого анализа взаимодействий — **SHAP interaction values** (Лундберг 2018), которые декомпозируют попарные эффекты.

```python
import shap
explainer = shap.TreeExplainer(model)
sv = explainer.shap_values(X_val)               # (n, p)
imp = np.abs(sv).mean(axis=0)
```

**SHAP vs gain.** Эмпирически (Lundberg 2020 и многократно подтверждено на Kaggle) SHAP стабильнее при перекрёстных корреляциях и не страдает high-cardinality bias в той же мере, поскольку учитывает реальный output модели, а не структуру дерева. Но TreeSHAP в интервенциональном режиме делает допущения о фон-распределении — это релевантно при сильной зависимости признаков ([Aas, Jullum, Løland, *Artificial Intelligence* 2021](https://www.sciencedirect.com/science/article/pii/S0004370221000539)).

Глава по SHAP в Interpretable ML Book Christoph Molnar: [christophm.github.io/interpretable-ml-book/shap.html](https://christophm.github.io/interpretable-ml-book/shap.html).

---

## 5. Permutation importance

Идея Брейман (2001): обучить модель, перемешать `X_j` в hold-out, измерить падение метрики. Формализовано как **Model Reliance** ([Fisher, Rudin & Dominici, *JMLR* 2019](https://www.jmlr.org/papers/v20/18-760.html)) с доверительными интервалами.

**Преимущества над gain:** model-agnostic, считается на hold-out, отражает реальное влияние на качество.

**Главный pitfall** ([Hooker, Mentch & Zhou, *Statistics and Computing* 31:82, 2021, arXiv:1905.03151](https://arxiv.org/abs/1905.03151), «Unrestricted Permutation Forces Extrapolation»): при коррелированных признаках перемешивание создаёт точки вне совместного распределения данных, и модель экстраполирует, что приводит к завышению важности коррелированных признаков и абсурдным выводам. Авторы прямо рекомендуют **не использовать «бесплатный» permute-and-predict**, а вместо этого либо переобучать модель без признака (drop-column importance, дорого), либо использовать **conditional permutation** (Strobl 2008) с пересэмплированием в стратах.

```python
from sklearn.inspection import permutation_importance
r = permutation_importance(model, X_val, y_val,
                           n_repeats=30, random_state=0,
                           scoring='roc_auc')
imp = pd.Series(r.importances_mean, index=X_val.columns)
```

[`eli5.show_weights(PermutationImportance(model).fit(X_val, y_val))`](https://eli5.readthedocs.io/en/latest/blackbox/permutation_importance.html) — удобный wrapper.

---

## 6. Null importance / target permutation

Популяризован [Olivier Grellier на Home Credit Default Risk](https://www.kaggle.com/code/ogrellier/feature-selection-with-null-importances), ставший de-facto стандартом на Kaggle. Идея:
1. Обучить модель — получить «actual» важности.
2. Перемешать **y** (а не X), переобучить — повторить N раз → распределение null importances.
3. Признак значим, если actual важность > 75-го (или 95-го) перцентиля null-распределения; типичный score — `log((actual + 1) / (percentile_null + 1))`.

Это **target permutation**, не путать со `sklearn.inspection.permutation_importance`. Реализация: пакет [`target-permutation-importances`](https://github.com/kingychiu/target-permutation-importances).

Метод нативно элиминирует шум, high-cardinality bias и leakage-каналы — поэтому применялся в призовых решениях Santander Customer Transaction, IEEE-CIS Fraud Detection и др.

---

## 7. Прочие продвинутые методы

- **Adversarial validation для FS:** обучить классификатор «train vs test»; признаки с высокой важностью различают train/test и являются кандидатами на дрифт — их можно удалить или нормировать. Особенно полезно на Kaggle с большой задержкой между train/test (Santander, IEEE-CIS).
- **Stability selection с bootstrap** ([Meinshausen & Bühlmann, JRSS-B 2010](https://rss.onlinelibrary.wiley.com/doi/10.1111/j.1467-9868.2010.00740.x)) — повторить отбор на K бутстрэпах, оставить часто выбираемые.
- **Knockoffs** ([Barber & Candès, *Annals of Statistics* 2015](https://projecteuclid.org/journals/annals-of-statistics/volume-43/issue-5/Controlling-the-false-discovery-rate-via-knockoffs/10.1214/15-AOS1337.full); Model-X knockoffs, [Candès et al. 2018](https://arxiv.org/abs/1610.02351)) — строят «фальшивые» копии X с теми же ковариациями, но independence от y; гарантируют FDR-контроль. На практике используются мало из-за вычислительной сложности и зависимости от модели X-распределения; в табличных пайплайнах CatBoost/LightGBM проигрывают по простоте Boruta + null importance.
- **Probe / shadow features** — фундамент Boruta; самый простой вариант — добавить колонку гауссовского шума и удалить всё, что слабее.

---

## 8. Kaggle: lore и конкретика

**Home Credit Default Risk (2018).** Победители (top-1) явно подчёркивали: «feature engineering > model tuning > stacking». Из ~1500 сгенерированных агрегатов из 7 таблиц использовали LightGBM gain importance + null importance для отбора ~300 финальных. Стандарт того конкурса — LightGBM «rf» mode + null importance.

**IEEE-CIS Fraud Detection (2019).** Из-за временного сплита `train/test` участники массово применяли **adversarial validation** для отсева дрейфующих фичей; null importance тоже использовался (см. публичный kernel [Viraj Bagal](https://github.com/VirajBagal/IEEE-Fraud/blob/master/feature-selection-using-null-importance.ipynb) со ссылкой на Olivier).

**Santander Customer Transaction Prediction (2019).** Особый случай — 200 анонимизированных независимых признаков; FS почти не давал выигрыша, ключ был в обнаружении «magic feature» (synthetic vs real rows).

**Numerai.** Из-за крайне низкого signal-to-noise FS обычно ухудшает результат: ML-команды оставляют все ~1000 фичей и полагаются на ансамбли + регуляризацию.

**Two Sigma Financial Modeling.** Финансовые временные ряды; типично mRMR / корреляционный фильтр + stability selection.

**Когда Kaggler-ы НЕ делают FS.** При сильных GBDT и >100K строк бустинги робастны к шумовым признакам: gain-импортанс отсеивает их в split-ах естественно, а время CV-итерации важнее. На task-ах с n ≈ p (M5, многие табличные конкурсы с 50–200 фичами) явный FS почти не даёт выигрыша.

---

## 9. Какие методы FS используются внутри AutoML-фреймворков

Полезный практический вопрос — не «какой AutoML лучше», а **какие конкретно методы отбора фичей подсмотрены и интегрированы у мейнтейнеров**. По этим решениям видно, что считается «бесплатным» (всегда включено), что — опциональным, а что — слишком дорогим для дефолта.

### AutoGluon Tabular (Amazon)

Два разных механизма:

- **`feature_prune_kwargs`** в [`TabularPredictor.fit()`](https://auto.gluon.ai/stable/api/autogluon.tabular.TabularPredictor.fit.html) — **layer-wise RFE с permutation importance**. По документации: «fits all models in a stack layer once, discovers a pruned set of features, fits all models in the stack layer again with the pruned set of features, and updates input feature lists for models whose validation score improved». **По умолчанию выключено** (`None`); пользователь должен явно передать пустой dict для включения. То есть для типичного `fit(presets='best_quality')` отбора фичей **нет** — ансамбль из LightGBM/XGBoost/CatBoost/NN сам разбирается с шумом.
- **[`predictor.feature_importance(data, ...)`](https://auto.gluon.ai/stable/api/autogluon.tabular.TabularPredictor.feature_importance.html)** — post-hoc **permutation importance** с доверительными интервалами (`include_confidence_band`, `confidence_level=0.99`), `num_shuffle_sets` повторов перемешивания и `subsample_size=5000` для скорости. Метод чисто диагностический и сам ничего не удаляет.

### H2O AutoML

**Явного шага FS нет.** [Документация](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html) прямо описывает «light data preparation» — импьютация, стандартизация, one-hot и (с версии 3.32.0.1) опциональный target encoding для high-cardinality. Никаких filter/wrapper/RFE-шагов в пайплайне. Отбор делегирован разнообразию моделей (GLM с регуляризацией, GBM/XGBoost с их native importance, DRF, DeepLearning) и stacked ensemble. Получить `varimp()` по конкретной модели — да, но это уже post-hoc анализ.

### FLAML (Microsoft)

**Явного FS нет.** Фокус — экономный hyperparameter search через CFO/BlendSearch ([Wang et al., MLSys 2021, arXiv:1911.04706](https://arxiv.org/abs/1911.04706)). [Препроцессинг](https://microsoft.github.io/FLAML/docs/Use-Cases/Task-Oriented-AutoML/) состоит из data-validation и task/estimator-level трансформаций (sparse-conversion, label-encoding); отбор признаков не входит. Native importance модели (`automl.model.estimator.feature_importances_` для LightGBM) — для пользовательской диагностики, не для автоматического отбора.

### LightAutoML (Sber AI Lab)

Самый структурированный FS-модуль среди open-source AutoML. В [`tabular_config.yml`](https://github.com/sb-ai-lab/LightAutoML/blob/master/lightautoml/automl/presets/tabular_config.yml) — три режима через параметр `selection_params.mode`:
- **0** — без отбора;
- **1** — **`ImportanceCutoffSelector`** (по умолчанию): обучает LightGBM на холдауте, считает importance (`importance_type: 'gain'` по умолчанию, либо `'permutation'`), удаляет признаки ниже `cutoff` (по умолчанию 0 — то есть всё с нулевым gain);
- **2** — **iterative / forward selection**: фичи сортируются по убыванию важности, добавляются блоками (`feature_group_size`), модель переобучается, блок сохраняется только если качество улучшилось — это hill-climbing forward selection.

Селекторная модель — `gbm` (под капотом CatBoost + LightGBM) или `linear_l2`. Реализация в [`lightautoml.pipelines.selection.importance_based`](https://lightautoml.readthedocs.io/en/latest/_modules/lightautoml/pipelines/selection/importance_based.html). Статья: [Vakhrushev et al., CIKM 2022](https://arxiv.org/abs/2109.01528).

### MLJAR-supervised

Два связанных шага:
1. **[Golden Features](https://supervised.mljar.com/features/golden_features/)** — генерация фичей: перебирает все пары признаков (или 250 000 случайных), для каждой пары строит `X_i − X_j` и `X_i / X_j`, обучает дерево глубины 3 на 2 500 точках, оценивает logloss/MSE на 2 500 тестовых. Топовые новые фичи (5% от исходного числа, но 5–50) добавляются в датасет.
2. **[Features Selection](https://supervised.mljar.com/features/features_selection/)** — **probe-feature / random-feature метод**: вставляет колонку `random_feature` с равномерным распределением `U[0,1]`, переобучает лучшую модель, считает **permutation importance**, всё с важностью ниже `random_feature` уходит в `drop_features.json`. Это open-source аналог идеи Boruta без множественных reps и Z-тестов.

Дополнительно: для каждого алгоритма после обучения автоматически считаются SHAP-объяснения (importance, dependence, decision plots).

### auto-sklearn

FS как **один из preprocessor-узлов в configspace**, который BO выбирает наравне с моделями. Полный список из [`FeaturePreprocessorChoice.get_components()`](https://automl.github.io/auto-sklearn/master/examples/40_advanced/example_interpretable_models.html):

- **Filter:** `select_percentile_classification` / `select_percentile_regression` (с `score_func ∈ {chi2, f_classif, mutual_info}`), `select_rates_classification` / `select_rates_regression`.
- **Embedded:** `extra_trees_preproc_for_classification/regression` (SelectFromModel на ExtraTrees), `liblinear_svc_preprocessor` (L1-SVC).
- **Не-FS, но в той же группе** — снижение размерности: `pca`, `kernel_pca`, `fast_ica`, `truncatedSVD`, `feature_agglomeration`, `nystroem_sampler`, `kitchen_sinks`, `random_trees_embedding`, `polynomial`, `no_preprocessing`, `densifier`.

Какой конкретно preprocessor сработает — решает мета-обученный BO ([Feurer et al., JMLR 2022, «Auto-Sklearn 2.0»](https://www.jmlr.org/papers/v23/21-0992.html)).

### TPOT (EpistasisLab)

FS — как **операторы в дереве пайплайна для генетического программирования**. Из [главы 8 «AutoML Book» (Olson & Moore)](https://www.automl.org/wp-content/uploads/2019/05/AutoML_Book_Chapter8.pdf):

> Feature Selection Operators: VarianceThreshold, SelectKBest, SelectPercentile, SelectFwe, and Recursive Feature Elimination (RFE).

Плюс [`FeatureSetSelector`](http://epistasislab.github.io/tpot/latest/Tutorial/3_Feature_Set_Selector/) для группированных фичей (например, известных подмножеств генов) и MDR / MultiSURF-операторы (Relief-семейство) — TPOT исторически создавался под геномные задачи, отсюда подбор. Каждый оператор в популяции мутирует свои параметры (`k`, `percentile`, `alpha`, выбор `score_func`).

### PyCaret

Параметры [`setup()`](https://pycaret.gitbook.io/docs/get-started/preprocessing/feature-selection):
- **`feature_selection=True`** + **`feature_selection_method`**:
  - `'classic'` (по умолчанию) — `SelectFromModel` поверх `feature_selection_estimator` (LightGBM по умолчанию, можно подменить);
  - `'univariate'` — `SelectKBest`;
  - `'sequential'` — `SequentialFeatureSelector` (forward/backward).
- **`n_features_to_select`** — целое или доля.
- Отдельные параметры: **`remove_multicollinearity`** + **`multicollinearity_threshold`**, **`low_variance_threshold`**, **`pca`**.

Источник — [`pycaret/classification/oop.py`](https://github.com/pycaret/pycaret/blob/master/pycaret/classification/oop.py) и официальная документация.

### Коммерческие платформы (по публичной документации)

- **[DataRobot](https://docs.datarobot.com/en/docs/modeling/analyze-models/understand/feature-impact.html)** — «Feature Impact» (permutation importance на холдауте) + автоматические **Feature Lists**, отбираемые алгоритмом FIRE (Feature Importance Rank Ensembling); пользователь может выбрать готовый список или построить свой.
- **[H2O Driverless AI](https://docs.h2o.ai/driverless-ai/latest-stable/docs/userguide/feature-engineering.html)** — генетический поиск feature engineering + отбор по `permutation importance` модели-проксии (обычно LightGBM); поддерживает MLI-секцию с SHAP.
- **[Google Vertex AI Tabular](https://cloud.google.com/vertex-ai/docs/explainable-ai/overview)** — деталей внутреннего FS не публикует; экспонирует post-hoc feature attribution (Integrated Gradients / Sampled Shapley).

### Суммарная картина

| Фреймворк | Метод(ы) FS | Включён по умолчанию? |
|---|---|---|
| AutoGluon | RFE + permutation importance (layer-wise) | Нет, opt-in |
| H2O AutoML | — (доверяет ансамблю) | — |
| FLAML | — | — |
| LightAutoML | LGBM gain-cutoff / iterative forward по блокам | Да (cutoff=0) |
| MLJAR | Golden Features + probe-feature permutation | Да |
| auto-sklearn | SelectPercentile (chi2/F/MI), SelectRates, SelectFromModel(ExtraTrees), L1-SVC — выбирается BO | Условно (в configspace) |
| TPOT | VarianceThreshold, SelectKBest, SelectPercentile, SelectFwe, RFE — в генетическом дереве | Да (операторы в популяции) |
| PyCaret | SelectFromModel / SelectKBest / SequentialFS + remove_multicollinearity | Нет, opt-in |
| DataRobot | Permutation (Feature Impact) + FIRE Feature Lists | Да |
| H2O Driverless AI | Permutation importance проксии + GA feature engineering | Да |

Два устойчивых паттерна:

1. **«GBDT-first»-фреймворки** (AutoGluon, H2O AutoML, FLAML) не делают агрессивный FS по умолчанию — экономят time-budget и полагаются на устойчивость бустингов к шуму.
2. **«Pipeline-search»-фреймворки** (auto-sklearn, TPOT) включают FS как один из операторов в пространстве поиска, потому что их базовые модели (включая линейные и kNN) чувствительнее к мусорным признакам.

Между ними — LightAutoML / MLJAR / PyCaret с **явным выделенным FS-шагом**, обычно опирающимся на **gain или permutation importance LightGBM / CatBoost** с тем или иным правилом отсечения (cutoff 0, random-probe, forward по блокам). То есть в open-source AutoML мире как универсальный «дефолтный» инструмент де-факто закрепилась пара **GBDT importance + (опционально) probe/random feature** — те же кирпичики, что лежат в основе рецептов §12.

---

## 10. Анализ trade-off-ов

| Семейство | Сложность | Тип. время на 100K×500 | Hands-off? | Drift-robust | Интерпретация |
|---|---|---|---|---|---|
| Variance/corr filter | O(n·p) | секунды | Да | Высокая | Простая |
| MI / mRMR | O(n·p log n) | минуты | Полу-авто | Средняя | Простая |
| L1 / Elastic Net | O(n·p·iter) | минуты | Нужен tuning α | Средняя | Высокая (коэф-ы) |
| Tree gain (XGB/LGBM/CB) | бесплатно с обучением | — | Да | Средняя (есть bias) | Средняя |
| CatBoost `select_features` (Loss) | k·train ≈ steps·iter | 5–30 мин | Да (3 параметра) | Высокая | Высокая |
| Permutation importance | O(n_repeats·n·p·predict) | минуты–часы | Да | Низкая (extrapolation) | Высокая |
| Null importance | N·train | 30 мин – часы | Полу-авто | Высокая | Высокая |
| SHAP-importance | O(T·L·D²·n) | минуты | Да | Высокая | Очень высокая |
| Boruta(-Shap) | maxRuns·train | часы | Низкая, нужен maxRuns | Высокая | Очень высокая |
| Stability selection | K_bootstrap·train | часы | Низкая | Очень высокая | Высокая |
| Knockoffs | сложный фит X-модели | часы–дни | Низкая | Гарантия FDR | Средняя |

**Размер выборки:**
- **n < 1000:** жёсткий риск selection bias ([Ambroise & McLachlan, *PNAS* 2002](https://www.pnas.org/doi/10.1073/pnas.102102699)) — FS обязательно внутри nested-CV; предпочтительны устойчивые методы — stability selection, Boruta с большим pValue, кросс-валидированный Lasso.
- **1K ≤ n ≤ 100K:** CatBoost `select_features(LossFunctionChange)` + null importance — sweet spot.
- **n > 1M:** агрессивный pre-filter (variance, корреляция, IV) → GBDT-importance; Boruta-Shap на сэмпле; permutation на hold-out.

**Wide data (p ≫ n, геномика):** Knockoffs, mRMR, Lasso с stability selection. CatBoost здесь обычно не первое решение.

### Decision-flow для табличной задачи с CatBoost

```
                ┌─ p < 50, n > 10K ──────► НЕ делать явный FS, доверять CB
START ──► p ?  │
                ├─ 50 ≤ p ≤ 1000 ────────► CB.select_features(LossFunctionChange, steps=3-5)
                │                            + (опц.) null importance для верификации
                ├─ 1000 < p ≤ 10K ───────► Filter (variance + IV / MI) до 500–1000
                │                            → CB.select_features или Boruta-SHAP
                └─ p > 10K (wide) ───────► Lasso / Knockoffs / mRMR (CB не первый)
```

---

## 11. Best practices и pitfalls

1. **Selection bias.** [Ambroise & McLachlan (PNAS 2002)](https://www.pnas.org/doi/10.1073/pnas.102102699) на микрочиповых данных показали: если FS выполняется **до** CV-сплита, оценка ошибки занижена до нуля. **Правило:** FS — внутри каждого CV-fold, как часть пайплайна (`sklearn.pipeline`). [Cawley & Talbot (*JMLR* 2010)](https://www.jmlr.org/papers/v11/cawley10a.html) обобщают это на model selection.
2. **Data leakage.** Любая статистика, использующая `y`, должна считаться только на train-fold (касается target encoding, WoE, mean target, IV-биннинга).
3. **Мультиколлинеарность.** Для линейных моделей удаляйте по VIF > 5–10; для GBDT — обычно неважно для качества, но критично для интерпретации; используйте clustering по корреляциям (`scipy.cluster.hierarchy`) и выбирайте представителя кластера.
4. **High-cardinality.** CatBoost native handles категории через ordered TS — снижает (но не устраняет) bias, описанный Strobl 2007. Для XGBoost/LightGBM используйте target encoding с regularization внутри fold-ов.
5. **FS до или после тюнинга?** Эмпирический рецепт: сначала FS на дефолтных гиперпараметрах CatBoost (быстро), затем тюнинг на выбранных фичах. Совместная оптимизация (FLAML, AutoGluon) теоретически лучше, но дороже.
6. **Воспроизводимость.** Фиксируйте `random_seed` в CatBoost и в bootstrap-ах stability selection; для CatBoost — также `thread_count` (на GPU/multi-thread порядок плавающей-точки даёт детерминированные, но платформо-зависимые результаты).

---

## 12. Практические рецепты

### Recipe A — Quick & Dirty (5 минут)

```python
import pandas as pd, numpy as np
from sklearn.feature_selection import VarianceThreshold
from catboost import CatBoostClassifier

vt = VarianceThreshold(0.0).fit(X)
X1 = X.loc[:, vt.get_support()]
# отсечь сильно скоррелированные (Spearman > 0.95)
corr = X1.corr(method='spearman').abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
drop = [c for c in upper.columns if (upper[c] > 0.95).any()]
X2 = X1.drop(columns=drop)

m = CatBoostClassifier(iterations=500, verbose=0).fit(X2, y, cat_features=cats)
imp = pd.Series(m.get_feature_importance(), index=X2.columns).sort_values(ascending=False)
selected = imp.head(50).index.tolist()
```

### Recipe B — Solid baseline (CatBoost + null importance)

```python
from catboost import CatBoostClassifier
from sklearn.model_selection import KFold

def actual_imp(X, y):
    m = CatBoostClassifier(iterations=500, verbose=0, random_seed=0).fit(X, y, cat_features=cats)
    return pd.Series(m.get_feature_importance(), index=X.columns)

actual = actual_imp(X, y)
null_runs = []
rng = np.random.RandomState(0)
for i in range(50):
    y_perm = pd.Series(rng.permutation(y), index=y.index)
    null_runs.append(actual_imp(X, y_perm))
null_df = pd.concat(null_runs, axis=1)

score = np.log((actual + 1) / (null_df.quantile(0.75, axis=1) + 1))
selected = score[score > 0].sort_values(ascending=False).index.tolist()
```

### Recipe C — Maximum quality (Boruta-SHAP + stability + nested CV)

```python
from BorutaShap import BorutaShap
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
votes = pd.Series(0, index=X.columns)

for tr, va in skf.split(X, y):
    Xtr, ytr = X.iloc[tr], y.iloc[tr]
    base = CatBoostClassifier(iterations=500, verbose=0, random_seed=42)
    fs = BorutaShap(model=base, importance_measure='shap', classification=True)
    fs.fit(X=Xtr, y=ytr, n_trials=100, sample=True, verbose=False)
    accepted = set(fs.accepted)
    votes[list(accepted)] += 1

# stability threshold: признак отобран в ≥4 из 5 fold-ов
stable = votes[votes >= 4].index.tolist()
```

Для production-pipeline оборачивайте всю эту логику в `sklearn.Pipeline` и оценивайте качество **внешним** CV — иначе схватите selection bias по Ambroise & McLachlan.

---

## Резюме для прикладника на CatBoost

1. Базовая рекомендация — **`CatBoost.select_features(algorithm=RecursiveByLossFunctionChange, steps=3–5)`** + проверка через **null importance**. Покрывает 90% задач.
2. Если данных мало (`n < 5K`) или нужны статгарантии — **Boruta-SHAP внутри CV** + stability selection.
3. Не доверяйте **gain-importance** на коррелированных и high-cardinality признаках (Strobl et al. 2007, 2008); не доверяйте **сырому permutation importance** в тех же ситуациях (Hooker, Mentch, Zhou 2021).
4. Для интерпретации — **SHAP (TreeSHAP)**, для атрибуции взаимодействий — **SHAP interaction values**.
5. На больших Kaggle-данных с GBDT часто **выгоднее не делать FS** — это подтверждается решениями open-source AutoML, у которых FS либо выключен по умолчанию (AutoGluon), либо отсутствует (H2O AutoML, FLAML).
6. Делайте FS **внутри CV**. Всегда. Selection bias (Ambroise & McLachlan, PNAS 2002) — самая частая и тихая ошибка, искажающая offline-метрики до неузнаваемости.

Эти принципы переносятся и на другие табличные алгоритмы; в **NLP/CV** они работают на уровне «отбор фичей» только если фичи табличные (embedding-агрегаты, метаданные) — для сырого текста/изображений FS заменяется регуляризацией, dropout и архитектурными выборами.
