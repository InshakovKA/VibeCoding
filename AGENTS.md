# AGENTS.md

## Goal

Find the most accurate solution to the churn-prediction task defined by
`data/synthetic_customer_behavior_and_churn.csv`. Two modules, run in this order:

1. **`DataProcessing.py`** — missing-value imputation, feature normalization, feature
   selection. Produces two CSVs: training and testing data. Test = **20% of all rows (3000 of 15000)**.
2. **`ModelTuning.py`** — model selection + hyperparameter tuning via K-fold CV, then
   validation of the final tuned model on the test set. Produces the trained model
   saved with `pickle` **and** a markdown report describing the selected model and its
   test-set results.

## Environment

`python` on `PATH` is the Windows Store alias stub and **fails silently** — it prints
nothing and exits non-zero. Always go through the project venv (`.venv`, Python 3.11):

```powershell
& .venv\Scripts\python.exe DataProcessing.py
& .venv\Scripts\python.exe ModelTuning.py
```

Do not *run* the project with a `py -3.x` launcher or with an Anaconda interpreter. Those
are separate environments carrying whatever stack they happen to have (this machine's
Anaconda has pandas 1.5.3 / scikit-learn 1.3.0), so results will not reproduce. `.venv`
was deliberately created from a standalone, non-conda Python 3.11 so that it inherits
nothing from a base environment. To recreate it, use any standalone 3.11 to *create* the
venv, then install into it:

```powershell
py -3.11 -m venv .venv
& .venv\Scripts\python.exe -m pip install -r requirements.txt
```

Pinned and verified: `pandas 3.0.6`, `numpy 2.4.6`, `scikit-learn 1.9.1`, `scipy 1.17.1`,
`joblib 1.6.0`, `lightgbm 4.7.0`, `xgboost 3.2.0`, `imbalanced-learn 0.14.2`,
`matplotlib 3.11.2`, `seaborn 0.13.2`. Network access to PyPI works.

### pandas 3.x / sklearn 1.9 API traps (this is a major-version stack)

- **String columns are `str` dtype, not `object`.** `df.select_dtypes("object")` emits a
  `Pandas4Warning` and will break outright once the shim is removed. Use
  `select_dtypes(include=["str"])` or `select_dtypes(exclude="number")`.
- `OneHotEncoder` takes `sparse_output=`, not `sparse=`.
- `HistGradientBoostingClassifier` **does** now support native `categorical_features`
  and `monotonic_cst` (1.4+); `class_weight` is available. Older notes claiming it has
  no categorical support are obsolete.
- NumPy is 2.x: no `np.float_`/`np.int_`, and `np.NaN` is `np.nan`.
- `imbalanced-learn`, `lightgbm`, and `xgboost` are installed — `SMOTE`, `LightGBM`, and
  `XGBClassifier` are all fair game for the tuning pool.

### Boosters need numeric input

`lightgbm` / `xgboost` do not accept the one-hot-plus-imputer `ColumnTransformer` output
as-is with NaN present. Encode categoricals with `OrdinalEncoder` (unknown → `-1`),
`fillna(-1)`, and pass `categorical_feature` to LightGBM, or compare against
`HistGradientBoostingClassifier` which handles a `ColumnTransformer` directly.

## Dataset (verified)

15,000 rows x 21 columns; target is `churn` (`int64` 0/1, **68.11% / 31.89%** — moderately
imbalanced, no target missing values).

- **Missing values in every feature column** (~155–182 each, ≈1%); 2,707 rows have at
  least one missing value. Imputation is mandatory, not optional.
- `customer_id` — unique per row (15,000 distinct), string `CUST_000001`. **Drop it**;
  it is an identifier, not a feature.
- `signup_date` — string `YYYY-MM-DD`, spans 2022-01-01 to 2024-12-31 (1,096 distinct).
  Must be parsed and either dropped or decomposed into features.
- Numeric: `tenure_months` (1–35), `age` (18–80), `monthly_charges`, `total_charges`,
  `avg_session_duration_minutes`, `number_of_logins_per_month` (49 distinct),
  `number_of_support_tickets` (12 distinct), `last_login_days_ago` (90 distinct),
  `satisfaction_score` (ordinal, only values 1–5).
- Categorical: `gender` (3, incl. `Non-binary`), `region` (5), `income_level` (3),
  `subscription_type` (3), `usage_frequency` (3), `payment_method` (4),
  `contract_type` (2: monthly/yearly), `promotional_response` (2), `discount_used` (2).
- No duplicate rows apart from the ID.

## Pipeline contract

- **The train/test split must be made in `DataProcessing.py` and be identical for
  `ModelTuning.py`.** Do not re-split in `ModelTuning.py` — that silently invalidates the
  test score. Fix `random_state=42` and use `stratify=y` (the 68/32 split is stable, but
  stratification makes it exact).
- **Split before fitting any preprocessing.** Fitting imputers/scalers/selectors on the
  full dataset and then splitting leaks test statistics into training. Impute/scale
  inside a `sklearn.pipeline.Pipeline` (per CV fold) wherever the step is re-fit.
- `ModelTuning.py` reads the two CSVs, tunes on train only, and touches the test set
  exactly once for the final report. Nested CV or a train-internal validation split is
  needed to keep that number honest.
- Default filenames: `data/processed_train.csv`, `data/processed_test.csv`, the pickled
  model, and the markdown report. `.gitignore` already covers `.venv/`, those two CSVs,
  and `*.pkl` / `*.joblib` — add to it rather than committing artifacts.

## Stage 1 output schema (`DataProcessing.py`, already written)

Run it first; it is deterministic and its CSVs are the only sanctioned input to stage 2.
Do **not** re-read the raw CSV, re-impute, re-encode, or re-select features in
`ModelTuning.py` — that would re-introduce the leakage stage 1 exists to prevent.

- `data/processed_train.csv` — 12,000 rows; `data/processed_test.csv` — 3,000 rows.
  Both are **16 numeric features + `churn` as the last column**, fully dense (no NaN).
- Column names carry their origin block: `num__<feature>` for numeric features,
  `cat__<feature>_<level>` for one-hot levels (`drop="first"`, so one level per nominal
  feature is the reference). `customer_id` and `signup_date` are already gone.
- Scaling is `MinMaxScaler` into [0, 1] from the **train** min/max, applied by a local
  `MinMaxSubsetScaler` to **11 of the 12** numeric features. The exception is
  `satisfaction_score`: a 1-5 Likert scale that is imputed but deliberately left
  **unscaled** (listed in `UNSCALED_NUMERIC` at the top of the file), so the CSV reads
  1.0-5.0 rather than 0.00-1.00. `MaxAbsScaler` is an equally accurate drop-in —
  measured, see below.
- **`satisfaction_score` is not guaranteed integer.** `IterativeImputer` fills ~1% of its
  missing values fractionally (e.g. 2.34); all values stay within 1-5. Rounding them
  back measurably *costs* accuracy — test AUC 0.97573 -> 0.97512, CV 0.97407 -> 0.97367 —
  because it merges genuinely distinct imputations (4.4 and 4.6 both -> 5). Leave it
  continuous; do not add a rounding or clipping step.
- **Test rows can fall outside [0, 1].** `MinMaxScaler` does not clip; it maps new values
  against the *fitted* train min/max. One test row currently lands at
  `num__monthly_charges = -0.0128` (a real charge below the training minimum). Do not
  write stage 2 code that assumes a hard [0, 1] bound, and do not add clipping as a
  "fix" — it would make train/serve behaviour differ.
- Features kept: `tenure_months`, `monthly_charges`, `total_charges`,
  `avg_session_duration_minutes`, `number_of_logins_per_month`,
  `number_of_support_tickets`, `satisfaction_score`, `last_login_days_ago`,
  `signup_year`, `signup_month`, plus one-hot levels of `income_level`,
  `subscription_type`, `usage_frequency` (x2), `contract_type`, `promotional_response`.
  Dropped for |r| < 0.02: `gender`, `region`, `payment_method`, `discount_used`,
  `age`, `signup_dayofweek`, and 3 one-hot levels.
- Imputation is `IterativeImputer` for numerics and mode for categoricals; the choice is
  measured, not assumed (see below). `CORRELATION_THRESHOLD = 0.02` at the top of the
  file is the knob to turn if the feature set needs revisiting — the printed report
  lists every kept and dropped column with its correlation, so changing it is cheap.
- **Imputation stays joint across all 12 numerics, including `satisfaction_score`** — only
  scaling is split, via `MinMaxSubsetScaler`. Do not "tidy" `satisfaction_score` out into
  its own `ColumnTransformer` branch: that would isolate it from the other columns and
  throw away the multivariate reconstruction that makes iterative imputation worth using
  here (CV 0.97429 vs 0.97376 for median). Scale selectively; impute jointly.
- `ModelTuning.py` needs no scaler of its own — the CSVs are already on their final
  scales. Trees ignore scale; KNN/SVM/linear models depend on it being consistent.

## Modeling traps (verified in the data)

- **Temporal drift.** Churn rate is ~0.226 (2022 signups) and ~0.229 (2023) but **0.497
  for 2024**. A random split mixes three eras with very different base rates, so any
  date-derived feature silently encodes the era and inflates in-distribution scores.
  Decide the split strategy (random+stratified per the spec vs. temporal holdout) and
  state it in the report — the two answer different questions and give different
  numbers. Do not present an inflated random-split score as temporal generalization.
- **Redundant features.** `tenure_months` is ≈0.994 correlated with months elapsed since
  `signup_date`; `total_charges` is ≈0.954 correlated with
  `monthly_charges * tenure_months`. These are near-duplicates, not independent signal.
- Strongest single predictors: `satisfaction_score` (churn 0.796 at score 1 → 0.059 at
  score 5), `contract_type` (0.477 monthly vs 0.197 yearly), `usage_frequency` (0.490
  low vs 0.232 high).
- **Reference performance, measured in this venv.** Two points of comparison, both
  20% stratified holdout, `random_state=42`:
  - *Raw* features, median imputation, no selection, default
    `HistGradientBoostingClassifier` → AUC **0.9748** (a 300-tree `LGBMClassifier`
    reaches 0.9750).
  - *Stage-1 CSVs* (`processed_train`/`processed_test`), default
    `HistGradientBoostingClassifier` → AUC **0.97573**, AP 0.96228, accuracy 0.9350;
    `LogisticRegression(max_iter=2000)` → AUC **0.97241**; `SVC(rbf)` → 0.96938;
    `KNeighborsClassifier(5)` → 0.93005. 5-fold CV on the train CSV alone:
    HGB 0.97407, LogReg 0.96945.
  Treat ~0.975 as the floor; a stage 2 that cannot beat it is a regression. The figure
  moves by ~0.001 across scikit-learn versions, so compare like-for-like inside `.venv`.
- **Scaler choice is provably a no-op for this pipeline.** Measured 5-fold CV AUC under
  Standard / MinMax / MaxAbs: HGB 0.97496 in all three (trees are scale-invariant),
  LogReg 0.96923 / 0.96932 / 0.96928 (noise). Correlation-based selection is also
  untouched — max per-column |Δr| ~2e-15 and the identical 16-column keep set, because
  all three are monotone linear maps and Pearson r is invariant to those. So changing
  the scaler cannot change which features survive; do not spend stage-2 time on it.
- **Imputation evidence** (5-fold CV on train, same pipeline otherwise): iterative
  0.97429 > median 0.97376 > KNN-5 0.97335. The gap over median is ~1/3 of a standard
  deviation, so this is a defensible default rather than a decisive win.
- **Correlation-based selection is safe here, but is a linear filter.** Dropping
  `gender`, `region`, `payment_method`, `income_level`, or `age` moves test AUC by at
  most ±0.0003 — they carry no marginal signal. But Pearson r only catches *linear*
  dependence, so a feature with a purely non-monotone effect would be discarded
  spuriously. If stage 2 ever underperforms, re-check by fitting once on the full
  30-column set before blaming the model.

## Misc gotchas

- **PowerShell reports exit code 1 on success if the script writes anything to stderr**
  (e.g. a pandas warning). Check for a traceback before assuming a run failed.
- `SVC(probability=True)` is deprecated in scikit-learn 1.9 and will be removed in 1.11 —
  it emits a `FutureWarning` at fit time. If stage 2 needs SVC probabilities, wrap it in
  `CalibratedClassifierCV(SVC(), ...)` instead.
- `IterativeImputer` fills from model predictions and can emit physically impossible
  values here: it produces `total_charges` as low as **-551.75** and `last_login_days_ago`
  as low as **-6.84** (a "login" in the future), and `tenure_months` near 0. With
  min-max scaling these map cleanly into [0, 1] and look innocuous, but they are
  artifacts. They are rare enough not to move AUC measurably; if a domain-validity
  clamp is ever wanted, it belongs here and must be applied to train and test alike.
- The repo path and Windows console are not UTF-8: non-ASCII output and errors come out
  as mojibake. Keep script output and the markdown report ASCII-only.
- The CSV reads cleanly with a bare `pd.read_csv(path)` — no `encoding=`, no BOM, no
  ragged lines (15,001 lines = header + 15,000).
- `pickle` files are bound to the writing interpreter's class definitions
  (Python 3.11) — a model pickled under a different scikit-learn or Python
  version may fail to load. Record the versions in the report, and never unpickle an
  untrusted file.
