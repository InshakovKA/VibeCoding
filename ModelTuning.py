"""
ModelTuning.py -- stage 2 of the churn-prediction pipeline.

Reads the two CSVs produced by DataProcessing.py, runs K-fold cross-validated
model selection and hyperparameter tuning over every candidate declared in
`model_tuning.toml`, then validates the single winning model on the held-out test
set.

Outputs (paths configurable in the config's [report] table):
    churn_model.pkl   the fitted winning estimator, via pickle
    model_report.md   a short description of the model and its test-set results

Design rules this file must keep
--------------------------------
* **The candidate list comes from the config, not from code.** Each `[[models]]`
  entry names an importable `class_path` plus a parameter grid. Adding a model
  must not require editing this file.
* **The test set is touched exactly once**, after selection is complete, by the
  winner alone. No test row ever influences a hyperparameter choice, a model
  choice, or a decision threshold. That is what makes the reported test number
  meaningful rather than a second round of tuning.
* **Selection is by a threshold-free metric** (ROC AUC by default). The data is
  68/32 imbalanced, so accuracy would reward always predicting the majority
  class.
* **No re-splitting and no re-preprocessing.** Stage 1 already produced the final
  feature space, scaled and imputed, fitted on train only. Re-doing either here
  would reintroduce exactly the leakage stage 1 exists to prevent.

Run:
    .venv\\Scripts\\python.exe ModelTuning.py
"""

from __future__ import annotations

import importlib
import inspect
import pickle
import platform
import sys
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
CONFIG_PATH = Path("model_tuning.toml")
DATA_DIR = Path("data")
TRAIN_CSV = DATA_DIR / "processed_train.csv"
TEST_CSV = DATA_DIR / "processed_test.csv"
TARGET = "churn"

# TOML has no null literal, so configs spell "no value" as this sentinel. It is
# matched case-insensitively and exactly, which leaves strings like gamma="scale"
# untouched.
NONE_SENTINEL = "none"


class ConfigError(RuntimeError):
    """Raised when the config file is missing, malformed, or self-inconsistent."""


# --------------------------------------------------------------------------- #
# Config loading
# --------------------------------------------------------------------------- #
@dataclass
class Candidate:
    """One entry from the config's [[models]] array."""

    name: str
    class_path: str
    params: dict[str, list[Any]]
    search: str = "random"
    enabled: bool = True
    notes: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def grid_size(self) -> int:
        size = 1
        for values in self.params.values():
            size *= max(len(values), 1)
        return size


@dataclass
class Config:
    cv: dict[str, Any]
    search: dict[str, Any]
    report: dict[str, Any]
    candidates: list[Candidate]

    @property
    def n_splits(self) -> int:
        return int(self.cv.get("n_splits", 5))

    @property
    def scoring(self) -> str:
        return str(self.cv.get("scoring", "roc_auc"))

    @property
    def n_jobs(self) -> int:
        return int(self.cv.get("n_jobs", -1))

    @property
    def random_iterations(self) -> int:
        return int(self.search.get("random_iterations", 32))

    @property
    def random_state(self) -> int:
        return int(self.cv.get("random_state", 42))

    @property
    def decision_threshold(self) -> float:
        return float(self.report.get("decision_threshold", 0.5))

    @property
    def top_k(self) -> int:
        return int(self.report.get("top_k", 10))

    @property
    def importance_rows(self) -> int:
        return int(self.report.get("importance_rows", 10))

    @property
    def model_path(self) -> Path:
        return Path(self.report.get("output_dir", ".")) / self.report.get(
            "model_filename", "churn_model.pkl"
        )

    @property
    def report_path(self) -> Path:
        return Path(self.report.get("output_dir", ".")) / self.report.get(
            "report_filename", "model_report.md"
        )


def coerce_params(params: dict[str, Any]) -> dict[str, list[Any]]:
    """Turn TOML param values into something scikit-learn will accept.

    Resolves the "none" sentinel to Python None, and leaves every other string
    untouched. Raises on an empty grid, because an empty list means the search
    would silently do nothing.
    """
    if not params:
        # `params = {}` is present, so the required-key check does not catch it,
        # and it would otherwise become a legitimate-looking 1-point search that
        # competes in the leaderboard on plain defaults.
        raise ConfigError(
            "params is empty; name at least one hyperparameter to search. "
            "An empty table evaluates only the estimator's defaults, which is "
            "almost never what a tuning step is meant to do."
        )
    coerced: dict[str, list[Any]] = {}
    for name, values in params.items():
        if not isinstance(values, list):
            raise ConfigError(
                f"parameter {name!r} must be a list of candidate values, got {values!r}"
            )
        if not values:
            raise ConfigError(f"parameter {name!r} has an empty candidate list")
        out = []
        for value in values:
            if isinstance(value, str) and value.lower() == NONE_SENTINEL:
                out.append(None)
            else:
                out.append(value)
        coerced[name] = out
    return coerced


def load_config(path: Path = CONFIG_PATH) -> Config:
    """Read and validate the TOML config."""
    if not path.exists():
        raise ConfigError(
            f"{path} not found. It declares the candidate models; run from the "
            f"repository root or pass a different path."
        )
    try:
        with path.open("rb") as handle:
            raw = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path} is not valid TOML: {exc}") from exc

    entries = raw.get("models")
    if not entries:
        raise ConfigError(f"{path} declares no [[models]] entries; nothing to tune.")
    if not isinstance(entries, list):
        raise ConfigError(f"{path}: [[models]] must be an array of tables.")

    candidates: list[Candidate] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ConfigError(f"[[models]] entry #{index} is not a table.")
        for required in ("name", "class_path", "params"):
            if required not in entry:
                raise ConfigError(f"[[models]] entry #{index} is missing {required!r}.")
        params = coerce_params(entry["params"])
        candidates.append(
            Candidate(
                name=str(entry["name"]),
                class_path=str(entry["class_path"]),
                params=params,
                search=str(entry.get("search", "random")).lower(),
                enabled=bool(entry.get("enabled", True)),
                notes=str(entry.get("notes", "")).strip(),
                raw=entry,
            )
        )

    active = [c for c in candidates if c.enabled]
    if not active:
        raise ConfigError("every [[models]] entry is disabled; nothing to tune.")

    names = [c.name for c in active]
    if len(set(names)) != len(names):
        raise ConfigError(f"duplicate candidate names in the config: {names}")

    # Resolve every class_path now, while the fix is still a one-line config edit.
    # Doing it lazily at search time means a typo only surfaces after the previous
    # candidate has finished tuning -- i.e. minutes into the run.
    for candidate in active:
        estimator_class = import_estimator(candidate.class_path)
        accepted = inspect.signature(estimator_class.__init__).parameters
        unknown = sorted(set(candidate.params) - set(accepted))
        if unknown:
            raise ConfigError(
                f"candidate {candidate.name!r}: {estimator_class.__name__} does not "
                f"accept {unknown}. Check for a name from a different scikit-learn "
                f"version (e.g. HistGradientBoostingClassifier.max_features exists "
                f"in 1.5+ but not 1.3)."
            )

    return Config(
        cv=dict(raw.get("cv", {})),
        search=dict(raw.get("search", {})),
        report=dict(raw.get("report", {})),
        candidates=candidates,
    )


def import_estimator(class_path: str):
    """Import `module.ClassName` from a config string."""
    if "." not in class_path:
        raise ConfigError(
            f"class_path {class_path!r} must look like 'package.module.ClassName'."
        )
    module_name, _, attribute = class_path.rpartition(".")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ConfigError(f"cannot import module {module_name!r}: {exc}") from exc
    try:
        return getattr(module, attribute)
    except AttributeError as exc:
        raise ConfigError(f"{module_name!r} has no attribute {attribute!r}") from exc


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_split(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise ConfigError(
            f"{path} not found. Run DataProcessing.py first -- stage 2 consumes "
            f"its output and must not build the split itself."
        )
    return pd.read_csv(path)


def split_xy(frame: pd.DataFrame, name: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if TARGET not in frame.columns:
        raise ConfigError(f"{name} has no {TARGET!r} column.")
    features = [c for c in frame.columns if c != TARGET]
    x = frame[features].to_numpy(dtype=float)
    y = frame[TARGET].to_numpy()
    if not np.isfinite(x).all():
        raise ConfigError(f"{name} contains non-finite values; stage 1 output should be dense.")
    return x, y, features


# --------------------------------------------------------------------------- #
# Scoring helpers
# --------------------------------------------------------------------------- #
def positive_scores(estimator, x: np.ndarray) -> np.ndarray:
    """Score for the positive class, whichever interface the model offers.

    A model with predict_proba uses it; otherwise decision_function is used.
    Both are fine for ROC AUC, which only needs a ranking -- which is exactly
    why SVC runs without the deprecated probability=True.
    """
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(x)[:, 1]
    if hasattr(estimator, "decision_function"):
        decision = estimator.decision_function(x)
        return decision[:, 1] if decision.ndim == 2 else decision
    raise ConfigError(
        f"{type(estimator).__name__} exposes neither predict_proba nor decision_function."
    )


def best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Threshold maximising F1 on the given (TRAIN) data. Never called on test.

    Uses `precision_recall_curve`, whose contract is that `precision[i]` pairs with
    `thresholds[i]`. `roc_curve` is *not* usable here: it returns a
    precision/recall array whose last element reports precision 1.0 at the lowest
    threshold, instead of the base rate, so an argmax over it selects the
    degenerate "predict every row positive" solution.

    The trailing element of precision/recall has no matching threshold (it is the
    all-positive point) and is dropped for the same reason.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if thresholds.size == 0:
        return 0.5
    precision, recall = precision[:-1], recall[:-1]
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    return float(thresholds[int(np.argmax(f1))])


def threshold_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    predicted = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


# --------------------------------------------------------------------------- #
# Tuning
# --------------------------------------------------------------------------- #
@dataclass
class Trial:
    """Outcome of tuning one candidate."""

    candidate: Candidate
    best_score: float
    best_params: dict[str, Any]
    best_index: int
    n_candidates: int
    n_fits: int
    seconds: float
    results: list[dict[str, Any]]
    estimator: Any


def seed_estimator(estimator, candidate: Candidate, config: Config):
    """Force a deterministic seed unless the config asked for something specific.

    This is not cosmetic. `RandomizedSearchCV` is seeded, but the estimator it
    refits is not: `HistGradientBoostingClassifier` with early_stopping='auto'
    carves a validation set using the *global* RNG when random_state is None, and
    `RandomForestClassifier` samples bootstraps the same way. On this data that
    made the winning model's CV score drift between runs (0.97616 -> 0.97622) with
    n_iter_ varying 185/190/182. Seeding the estimator, not just the search, is
    what makes the reported numbers reproducible.
    """
    if "random_state" in candidate.params:
        return estimator  # config is explicit; respect it
    accepted = inspect.signature(type(estimator).__init__).parameters
    if "random_state" not in accepted:
        return estimator  # estimator is deterministic anyway
    estimator.set_params(
        random_state=int(config.search.get("random_state", config.random_state))
    )
    return estimator


def make_search(candidate: Candidate, config: Config, cv) -> GridSearchCV | RandomizedSearchCV:
    estimator = import_estimator(candidate.class_path)()
    estimator = seed_estimator(estimator, candidate, config)
    shared = dict(
        estimator=estimator,
        cv=cv,
        scoring=config.scoring,
        n_jobs=config.n_jobs,
        refit=True,
        return_train_score=False,
        error_score="raise",
    )
    if candidate.search == "grid":
        return GridSearchCV(param_grid=candidate.params, **shared)
    if candidate.search == "random":
        iterations = min(config.random_iterations, candidate.grid_size)
        if iterations < candidate.grid_size:
            print(
                f"    note: {candidate.name} grid holds {candidate.grid_size} points; "
                f"sampling {iterations}"
            )
        return RandomizedSearchCV(
            param_distributions=candidate.params,
            n_iter=iterations,
            random_state=int(config.search.get("random_state", config.random_state)),
            **shared,
        )
    raise ConfigError(
        f"candidate {candidate.name!r} has search={candidate.search!r}; "
        f"use 'grid' or 'random'."
    )


def tune(candidate: Candidate, x: np.ndarray, y: np.ndarray, config: Config, cv) -> Trial:
    search = make_search(candidate, config, cv)
    started = time.perf_counter()
    search.fit(x, y)
    seconds = time.perf_counter() - started

    ranked = sorted(
        search.cv_results_["mean_test_score"],
        key=lambda score: -score,
    )
    results = [
        {
            "params": search.cv_results_["params"][i],
            "mean_score": float(search.cv_results_["mean_test_score"][i]),
            "std_score": float(search.cv_results_["std_test_score"][i]),
        }
        for i in np.argsort(-search.cv_results_["mean_test_score"])
    ]
    return Trial(
        candidate=candidate,
        best_score=float(search.best_score_),
        best_params=dict(search.best_params_),
        best_index=int(search.best_index_),
        n_candidates=len(ranked),
        n_fits=int(search.n_splits_ * len(ranked)),
        seconds=seconds,
        results=results,
        estimator=search.best_estimator_,
    )


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def format_params(params: dict[str, Any]) -> str:
    if not params:
        return "(defaults)"
    return ", ".join(f"{k}={params[k]!r}" for k in sorted(params))


def build_report(
    config: Config,
    trials: list[Trial],
    winner: Trial,
    test_metrics: dict[str, Any],
    test_at_tuned_threshold: dict[str, Any],
    train_threshold: float,
    importances: list[tuple[str, float]],
    feature_names: list[str],
    n_train: int,
    n_test: int,
) -> str:
    lines: list[str] = []
    add = lines.append

    add("# Churn model -- selection and test results")
    add("")
    add(f"Generated by `ModelTuning.py` from `{CONFIG_PATH.name}`. "
        f"All selection used {config.n_splits}-fold cross-validation on the "
        f"{n_train:,}-row training split only; the {n_test:,}-row test split was "
        f"scored once, by the winner.")
    add("")

    add("## Selected model")
    add("")
    add(f"**{winner.candidate.name}** (`{winner.candidate.class_path}`)")
    add("")
    add(f"- Cross-validated {config.scoring}: **{winner.best_score:.5f}**")
    add(f"- Hyperparameters: `{format_params(winner.best_params)}`")
    add(f"- Search: {winner.candidate.search} over {winner.n_candidates} "
        f"configurations ({winner.n_fits} fits, {winner.seconds:.1f}s)")
    if winner.candidate.notes:
        add("")
        for chunk in winner.candidate.notes.strip().splitlines():
            if chunk.strip():
                add(f"> {chunk.strip()}")
    add("")
    add("The model is saved with `pickle` as "
        f"`{config.model_path.name}`. The pickle holds the fitted estimator "
        f"only -- no feature list, no config -- so the column order of "
        f"`{TRAIN_CSV.name}` is part of the model's contract. Passing columns in a "
        f"different order silently produces wrong scores, and passing a "
        f"DataFrame makes scikit-learn warn that the estimator was fitted without "
        f"feature names (it was fitted on a numpy array; pass a numpy array back).")
    add("")
    add("Feature order, positionally as the pickle expects it:")
    add("")
    add("```")
    for index, name in enumerate(feature_names):
        add(f"{index:2d}  {name}")
    add("```")
    add("")

    add("## Test-set results")
    add("")
    add("| metric | value |")
    add("| --- | --- |")
    add(f"| ROC AUC | **{test_metrics['roc_auc']:.5f}** |")
    add(f"| Average precision (PR-AUC) | {test_metrics['average_precision']:.5f} |")
    if test_metrics["brier"] is not None:
        add(f"| Brier score | {test_metrics['brier']:.5f} |")
    else:
        add("| Brier score | n/a -- the model has no `predict_proba` |")
    add("")
    add(f"At the default decision threshold {test_metrics['default']['threshold']:.2f}:")
    add("")
    add("| metric | value |")
    add("| --- | --- |")
    add(f"| accuracy | {test_metrics['default']['accuracy']:.4f} |")
    add(f"| balanced accuracy | {test_metrics['default']['balanced_accuracy']:.4f} |")
    add(f"| precision | {test_metrics['default']['precision']:.4f} |")
    add(f"| recall | {test_metrics['default']['recall']:.4f} |")
    add(f"| F1 | {test_metrics['default']['f1']:.4f} |")
    add(f"| confusion (tn/fp/fn/tp) | {test_metrics['default']['tn']} / {test_metrics['default']['fp']} / "
        f"{test_metrics['default']['fn']} / {test_metrics['default']['tp']} |")
    add("")
    add(f"Because the classes are 68/32 imbalanced, accuracy alone is misleading -- "
        f"always predicting the majority class would score 0.68. The threshold that "
        f"maximises F1 was chosen on **train** out-of-fold predictions "
        f"({train_threshold:.4f}) and applied to test unchanged:")
    add("")
    add("| metric | value |")
    add("| --- | --- |")
    add(f"| accuracy | {test_at_tuned_threshold['accuracy']:.4f} |")
    add(f"| balanced accuracy | {test_at_tuned_threshold['balanced_accuracy']:.4f} |")
    add(f"| precision | {test_at_tuned_threshold['precision']:.4f} |")
    add(f"| recall | {test_at_tuned_threshold['recall']:.4f} |")
    add(f"| F1 | {test_at_tuned_threshold['f1']:.4f} |")
    add(f"| confusion (tn/fp/fn/tp) | {test_at_tuned_threshold['tn']} / {test_at_tuned_threshold['fp']} / "
        f"{test_at_tuned_threshold['fn']} / {test_at_tuned_threshold['tp']} |")
    add("")

    add("## Model selection leaderboard")
    add("")
    add(f"Ranked by {config.n_splits}-fold CV {config.scoring} on the training split. "
        f"The winner is the top row; nothing below it influenced the choice.")
    add("")
    add("| rank | model | estimator | CV score | search | configs | fits | seconds |")
    add("| --- | --- | --- | --- | --- | --- | --- | --- |")
    ordered = sorted(trials, key=lambda t: -t.best_score)
    for rank, trial in enumerate(ordered, start=1):
        marker = " **<- selected**" if trial is winner else ""
        add(
            f"| {rank} | {trial.candidate.name}{marker} | `{trial.candidate.class_path}` | "
            f"{trial.best_score:.5f} | {trial.candidate.search} | {trial.n_candidates} | "
            f"{trial.n_fits} | {trial.seconds:.1f} |"
        )
    add("")

    add(f"### Best configurations tried for the selected model (top {min(config.top_k, len(winner.results))})")
    add("")
    add("| CV score | +/- std | params |")
    add("| --- | --- | --- |")
    for row in winner.results[: config.top_k]:
        add(f"| {row['mean_score']:.5f} | {row['std_score']:.5f} | `{format_params(row['params'])}` |")
    add("")

    if importances:
        add(f"### Permutation importance (top {len(importances)})")
        add("")
        add("| feature | importance |")
        add("| --- | --- |")
        for name, value in importances:
            add(f"| `{name}` | {value:.5f} |")
        add("")
        add("Measured on a stratified 25% holdout **carved out of the training set**, "
            "with a probe model refit on the remaining 75%. The winner itself is "
            "refit on all of train, so permuting train directly would be an "
            "in-sample measurement and would distort this ranking. The test split is "
            "not involved. Treat these as ranking, not as absolute contribution: "
            "correlated features share credit.")
        add("")

    add("## Caveats")
    add("")
    add("- **This is a random-split score, not a temporal one.** Churn is ~0.23 for "
        "2022/2023 signups but ~0.50 for 2024, and `signup_year` is among the "
        "features, so the model benefits from era information. Do not read this "
        "number as performance on future data; that needs a temporal holdout.")
    add("- **Selection used one CV pass, not nested CV.** The reported test score is "
        "unbiased because the test split was untouched, but the *CV* score is a "
        "mildly optimistic estimate of what the search would score on fresh data.")
    if not hasattr(winner.estimator, "predict_proba"):
        add("- **The selected model ships uncalibrated** (it was scored via "
            "`decision_function`). Ranking metrics such as AUC are unaffected, but "
            "wrap it in `CalibratedClassifierCV` and refit if you need to trust its "
            "probabilities or the Brier score.")
    add(f"- Pinned to python {platform.python_version()}, scikit-learn {sklearn.__version__}, "
        f"numpy {np.__version__}, pandas {pd.__version__}. A pickle will not load "
        f"across incompatible versions.")
    add("- Re-running the module reproduces every number here. The estimator is "
        "seeded as well as the search, because an unseeded "
        "`HistGradientBoostingClassifier` varies run to run "
        "(`n_iter_` 185/190/182) and an unseeded forest resamples its bootstrap.")
    add("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    print("=" * 70)
    print("ModelTuning.py")
    print("=" * 70)

    config = load_config()
    print(f"Config      : {CONFIG_PATH} "
          f"(cv={config.n_splits}-fold {config.scoring}, n_jobs={config.n_jobs})")
    active = [c for c in config.candidates if c.enabled]
    for candidate in config.candidates:
        state = "enabled" if candidate.enabled else "DISABLED"
        print(f"  - {candidate.name:32} {candidate.class_path:52} "
              f"[{candidate.search}, {state}]")
    if len(active) != len(config.candidates):
        print(f"  ({len(config.candidates) - len(active)} entr"
              f"{'y' if len(config.candidates) - len(active) == 1 else 'ies'} disabled in config)")

    train = load_split(TRAIN_CSV)
    test = load_split(TEST_CSV)
    x_train, y_train, features = split_xy(train, TRAIN_CSV.name)
    x_test, y_test, test_features = split_xy(test, TEST_CSV.name)
    if features != test_features:
        raise ConfigError(
            "train and test feature columns differ or are in a different order; "
            "rerun DataProcessing.py so both files come from the same run."
        )
    print(f"\nData        : {TRAIN_CSV} {x_train.shape}, "
          f"{TEST_CSV} {x_test.shape}, {len(features)} features")
    print(f"Class balance: train {np.mean(y_train == 1) * 100:.2f}% positive, "
          f"test {np.mean(y_test == 1) * 100:.2f}% positive")

    cv = StratifiedKFold(
        n_splits=config.n_splits,
        shuffle=bool(config.cv.get("shuffle", True)),
        random_state=config.random_state,
    )

    trials: list[Trial] = []
    for index, candidate in enumerate(active, start=1):
        print(f"\n[{index}/{len(active)}] {candidate.name} ({candidate.search} search, "
              f"grid size {candidate.grid_size})")
        trial = tune(candidate, x_train, y_train, config, cv)
        trials.append(trial)
        print(f"    best CV {config.scoring}: {trial.best_score:.5f} "
              f"({trial.n_candidates} configs, {trial.n_fits} fits, {trial.seconds:.1f}s)")
        print(f"    params: {format_params(trial.best_params)}")

    if not trials:
        raise ConfigError("no candidates were tuned.")

    winner = max(trials, key=lambda t: t.best_score)
    print(f"\n{'=' * 70}")
    print(f"Selected: {winner.candidate.name}  CV {config.scoring} {winner.best_score:.5f}")
    print("=" * 70)

    # The one and only pass over the test split.
    test_scores = positive_scores(winner.estimator, x_test)
    test_auc = float(roc_auc_score(y_test, test_scores))

    # Threshold chosen on TRAIN out-of-fold predictions, never on test.
    print("\nChoosing a decision threshold from train out-of-fold predictions...")
    oof_scores = cross_val_predict(
        clone_estimator(winner),
        x_train,
        y_train,
        cv=cv,
        method="predict_proba" if hasattr(winner.estimator, "predict_proba") else "decision_function",
        n_jobs=config.n_jobs,
    )
    if oof_scores.ndim == 2:
        oof_scores = oof_scores[:, 1]
    train_threshold = best_f1_threshold(y_train, oof_scores)
    print(f"    F1-optimal train threshold: {train_threshold:.4f}")

    has_proba = hasattr(winner.estimator, "predict_proba")
    test_metrics = {
        "roc_auc": test_auc,
        "average_precision": float(average_precision_score(y_test, test_scores)),
        "brier": float(brier_score_loss(y_test, test_scores)) if has_proba else None,
        "default": threshold_metrics(y_test, test_scores, config.decision_threshold),
    }
    test_at_tuned_threshold = threshold_metrics(y_test, test_scores, train_threshold)

    print(f"\nTest results (scored once, by the winner only):")
    print(f"    ROC AUC            {test_auc:.5f}")
    print(f"    Average precision  {test_metrics['average_precision']:.5f}")
    if has_proba:
        print(f"    Brier score        {test_metrics['brier']:.5f}")
    for label, block in (
        (f"threshold {config.decision_threshold:.2f}", test_metrics["default"]),
        (f"threshold {train_threshold:.4f} (train-tuned)", test_at_tuned_threshold),
    ):
        print(f"    {label:32} acc {block['accuracy']:.4f}  bal-acc "
              f"{block['balanced_accuracy']:.4f}  P {block['precision']:.4f}  "
              f"R {block['recall']:.4f}  F1 {block['f1']:.4f}")

    # Feature importance needs a fitted model and data it has NOT seen. The winner
    # was refit on every training row, so permuting x_train would be an in-sample
    # measurement and would distort the ranking. Carve a stratified holdout out of
    # train instead: honest, and the test split stays untouched.
    print("\nPermutation importance on a train-internal holdout (test not used)...")
    importances: list[tuple[str, float]] = []
    try:
        fit_x, hold_x, fit_y, hold_y = train_test_split(
            x_train,
            y_train,
            test_size=0.25,
            stratify=y_train,
            random_state=config.random_state,
        )
        probe = clone_estimator(winner).fit(fit_x, fit_y)
        result = permutation_importance(
            probe,
            hold_x,
            hold_y,
            n_repeats=10,
            random_state=config.random_state,
            scoring=config.scoring,
            n_jobs=config.n_jobs,
        )
        order = np.argsort(-result.importances_mean)[: config.importance_rows]
        importances = [
            (features[i], float(result.importances_mean[i])) for i in order
        ]
    except Exception as exc:  # pragma: no cover - diagnostics must not kill the run
        print(f"    skipped: {exc}")

    # Artifacts.
    config.model_path.parent.mkdir(parents=True, exist_ok=True)
    with config.model_path.open("wb") as handle:
        pickle.dump(winner.estimator, handle)
    print(f"\nSaved model : {config.model_path}")

    report = build_report(
        config=config,
        trials=trials,
        winner=winner,
        test_metrics=test_metrics,
        test_at_tuned_threshold=test_at_tuned_threshold,
        train_threshold=train_threshold,
        importances=importances,
        feature_names=features,
        n_train=len(y_train),
        n_test=len(y_test),
    )
    config.report_path.write_text(report, encoding="utf-8")
    print(f"Saved report: {config.report_path}")
    print("=" * 70)


def clone_estimator(trial: Trial):
    """An unfitted copy of the winner, for out-of-fold threshold selection.

    permutation/estimator instances are reused otherwise, and cross_val_predict
    needs a fresh estimator per fold.
    """
    from sklearn.base import clone

    return clone(trial.estimator).set_params(**trial.best_params)


if __name__ == "__main__":
    try:
        main()
    except ConfigError as error:
        sys.exit(f"ERROR: {error}")
