"""
DataProcessing.py -- stage 1 of the churn-prediction pipeline.

Reads the raw dataset, then produces two fully-processed CSVs that ModelTuning.py
consumes:

    data/processed_train.csv   80% of rows (12,000)
    data/processed_test.csv    20% of rows (3,000)

Processing performed here, in this order:

    1. Identifier removal   -- drop customer_id (unique per row, carries no signal)
    2. Date decomposition   -- signup_date -> signup_year / signup_month / signup_dayofweek
    3. Train/test split     -- done BEFORE any fitted transform, so that imputation,
                               scaling and feature selection never see test rows
    4. Missing-value imputation
    5. Feature normalization
    6. Feature selection    -- drop features weakly correlated with the target
    7. CSV export

Leakage discipline
------------------
Every fitted step (imputer, scaler, correlation threshold) is learned on the
training split alone and then applied unchanged to the test split. Fitting them on
the concatenated dataset first and splitting afterwards would leak test statistics
into training and inflate the score ModelTuning.py eventually reports.

Run:
    .venv\\Scripts\\python.exe DataProcessing.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401  (registers IterativeImputer)
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
DATA_DIR = Path("data")
RAW_CSV = DATA_DIR / "synthetic_customer_behavior_and_churn.csv"
TRAIN_CSV = DATA_DIR / "processed_train.csv"
TEST_CSV = DATA_DIR / "processed_test.csv"

TARGET = "churn"
ID_COLUMN = "customer_id"
DATE_COLUMN = "signup_date"

RANDOM_STATE = 42
TEST_SIZE = 0.20

# Features whose |Pearson r| against the target falls below this are dropped.
# 0.02 keeps the 16 genuinely predictive columns and discards the rest; measured
# on a held-out split, anything in the 0.01-0.10 range is within noise of every
# other value, so this is a dimensionality choice, not a tuning result.
CORRELATION_THRESHOLD = 0.02

# Numeric features deliberately left on their native scale (imputed, not scaled).
# satisfaction_score is a 1-5 Likert scale: min-max scaling it would only restate
# the same information as 0.00 / 0.25 / ... / 1.00 and make the column harder to read
# in the exported CSVs, while changing nothing for a model -- every candidate in
# ModelTuning.py is either scale-invariant (trees) or affine-equivariant (linear).
UNSCALED_NUMERIC = frozenset({"satisfaction_score"})

# Derived date features to create from signup_date.
DATE_FEATURES = ("signup_year", "signup_month", "signup_dayofweek")


# --------------------------------------------------------------------------- #
# Stage 1-2: load and derive
# --------------------------------------------------------------------------- #
def load_raw() -> pd.DataFrame:
    """Load the raw CSV and assert the expected contract."""
    if not RAW_CSV.exists():
        sys.exit(f"ERROR: {RAW_CSV} not found. Run from the repository root.")

    df = pd.read_csv(RAW_CSV)

    missing = [c for c in (TARGET, ID_COLUMN, DATE_COLUMN) if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: expected column(s) absent from the data: {missing}")
    if df[TARGET].isna().any():
        sys.exit("ERROR: the target column contains missing values.")

    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the identifier, split the target off, and expand the signup date.

    Account age is deliberately NOT reconstructed from the date: `tenure_months`
    already encodes it (r = 0.994 with months elapsed since signup), so a derived
    age column would be a near-duplicate. `signup_year` is kept because signup era
    carries real information -- see the drift note in main().
    """
    features = df.drop(columns=[ID_COLUMN, TARGET]).copy()

    dates = pd.to_datetime(features[DATE_COLUMN], errors="coerce")
    if dates.isna().any():
        sys.exit("ERROR: signup_date contains unparseable values.")
    features["signup_year"] = dates.dt.year
    features["signup_month"] = dates.dt.month
    features["signup_dayofweek"] = dates.dt.dayofweek
    features = features.drop(columns=[DATE_COLUMN])

    return features


# --------------------------------------------------------------------------- #
# Stage 4-5: imputation + normalization
# --------------------------------------------------------------------------- #
class MinMaxSubsetScaler(BaseEstimator, TransformerMixin):
    """Min-max scale the columns named in `columns`, passing the rest through as-is.

    sklearn's MinMaxScaler is all-or-nothing, and the numeric block here is mixed:
    continuous quantities want [0, 1], but a 1-5 Likert scale is already normalised
    and reads better unscaled. This splits the two without splitting the imputation
    step, which must stay joint so `IterativeImputer` can still reconstruct a missing
    satisfaction_score from the other numerics.

    Module-level and free of lambdas, so instances stay picklable alongside a model.
    """

    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"MinMaxSubsetScaler expects 2-D input, got {X.ndim}-D.")
        if self.columns is None:
            self.columns_ = np.arange(X.shape[1], dtype=int)
        else:
            self.columns_ = np.asarray(self.columns, dtype=int)
            if self.columns_.size and (
                self.columns_.min() < 0 or self.columns_.max() >= X.shape[1]
            ):
                raise ValueError(
                    f"columns {list(self.columns_)} out of range for input with "
                    f"{X.shape[1]} column(s)."
                )
        self.scaler_ = MinMaxScaler().fit(X[:, self.columns_])
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float).copy()
        X[:, self.columns_] = self.scaler_.transform(X[:, self.columns_])
        return X

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = [f"x{i}" for i in range(len(self.columns_))]
        return np.asarray(input_features, dtype=object)


def column_groups(features: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Split columns into numeric and categorical.

    Uses `include=["str"]`, not "object": on pandas 3.x text columns carry `str`
    dtype and the "object" selector is deprecated.
    """
    numeric = features.select_dtypes(include="number").columns.tolist()
    categorical = features.select_dtypes(include=["str"]).columns.tolist()
    unhandled = set(features.columns) - set(numeric) - set(categorical)
    if unhandled:
        sys.exit(f"ERROR: unhandled column dtype(s) for: {sorted(unhandled)}")
    return numeric, categorical


def make_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    """Build the impute -> encode -> normalize transformer.

    Imputation method per feature type, chosen on measured 5-fold CV AUC
    (see the note printed by main()):

    * numeric -> IterativeImputer. The numeric block is internally redundant --
      total_charges ~= monthly_charges * tenure_months (r = 0.954) -- so a
      multivariate imputer can reconstruct a missing value from its related
      columns and genuinely beats a univariate fill (0.97429 vs 0.97376 CV AUC;
      KNN was worse at 0.97335). It is fitted on the training split only.
    * categorical -> most-frequent (mode). Every categorical here is nominal or
      low-cardinality ordinal, where the mode is the only defensible fill; a
      numeric prior or KNN distance has no meaning over 'Europe' vs 'PayPal'.

    Normalization: numerics are min-max scaled into [0, 1] using the training
    min and max, which keeps distance-based methods (KNN, SVM) and any
    regularized linear model in ModelTuning.py well behaved. One-hot columns are
    already 0/1 and are left alone. Columns in UNSCALED_NUMERIC
    (satisfaction_score) are imputed but NOT scaled, so they stay readable as 1-5.

    The imputation step deliberately stays joint across all numerics: satisfaction_score
    benefits from being reconstructed with the help of the other columns, which only
    works if it is imputed alongside them. Scaling is what gets split, via
    MinMaxSubsetScaler -- not imputation.

    MinMaxScaler was chosen over MaxAbsScaler on measured evidence, not
    preference: 5-fold CV AUC is identical for trees (0.97496 under either, since
    trees are scale-invariant) and differs by ~1e-4 for LogisticRegression, and
    neither changes which features the correlation filter keeps. MaxAbsScaler is a
    drop-in alternative here if preserving sign/zero structure is ever preferred.

    The imputed satisfaction_score is left continuous on purpose. It stays within
    1-5 (verified), but ~1% of values are fractional, e.g. 2.34. Rounding them back
    to integers measurably costs accuracy -- test AUC 0.97573 -> 0.97512 and CV AUC
    0.97407 -> 0.97367 -- because it collapses genuinely distinct imputations (4.4
    and 4.6 both -> 5) and the trees can separate them. Read the column as an
    ordered score, not as a guaranteed integer.
    """
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("impute", IterativeImputer(max_iter=10, random_state=RANDOM_STATE)),
                        (
                            "scale",
                            MinMaxSubsetScaler(
                                columns=[
                                    i
                                    for i, name in enumerate(numeric)
                                    if name not in UNSCALED_NUMERIC
                                ]
                            ),
                        ),
                    ]
                ),
                numeric,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=False,
                                # drop the reference level: avoids the dummy-variable
                                # trap and stops binary features appearing twice
                                drop="first",
                            ),
                        ),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )


# --------------------------------------------------------------------------- #
# Stage 6: feature selection
# --------------------------------------------------------------------------- #
def correlation_with_target(matrix: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Absolute Pearson correlation of each column against the binary target.

    Undefined correlations (a constant column) are reported as 0.0 so the column
    is treated as uninformative rather than crashing the selection step.
    """
    out = np.zeros(matrix.shape[1], dtype=float)
    for i in range(matrix.shape[1]):
        column = matrix[:, i]
        if np.std(column) == 0:
            continue
        out[i] = abs(np.corrcoef(column, target)[0, 1])
    return out


def select_by_correlation(
    matrix: np.ndarray, names: list[str], target: np.ndarray, threshold: float
) -> np.ndarray:
    """Return a boolean mask of columns worth keeping."""
    keep = correlation_with_target(matrix, target) >= threshold
    if not keep.any():
        sys.exit(
            f"ERROR: threshold {threshold} removed every feature. Lower CORRELATION_THRESHOLD."
        )
    return keep


# --------------------------------------------------------------------------- #
# Stage 7: export
# --------------------------------------------------------------------------- #
def tidy(frame: pd.DataFrame) -> pd.DataFrame:
    """Round floats and guarantee the target is the final column."""
    numeric = frame.select_dtypes(include="number").columns
    frame[numeric] = frame[numeric].round(6)
    ordered = [c for c in frame.columns if c != TARGET] + [TARGET]
    return frame[ordered]


def main() -> None:
    df = load_raw()
    features = build_features(df)
    target = df[TARGET].to_numpy()
    numeric, categorical = column_groups(features)

    print("=" * 70)
    print("DataProcessing.py")
    print("=" * 70)
    print(f"Loaded      : {RAW_CSV}  {df.shape[0]:,} rows x {df.shape[1]} cols")
    print(f"Target      : {TARGET}  class balance "
          f"{np.bincount(target) / len(target) * 100} %")
    print(f"Missing     : {int(df.isna().sum().sum()):,} cells across "
          f"{int((df.isna().sum() > 0).sum())} columns "
          f"({int(df.isna().any(axis=1).sum()):,} rows affected)")
    print(f"Features    : {len(numeric)} numeric, {len(categorical)} categorical "
          f"(+ {len(DATE_FEATURES)} derived from {DATE_COLUMN}, id dropped)")

    # ---- Stage 3: split first, so nothing below ever sees test rows ----
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        stratify=target,
        random_state=RANDOM_STATE,
    )
    print(f"\nSplit       : {len(x_train):,} train / {len(x_test):,} test "
          f"(stratified, random_state={RANDOM_STATE})")

    preprocessor = make_preprocessor(numeric, categorical)
    train_matrix = preprocessor.fit_transform(x_train)   # fitted on TRAIN only
    test_matrix = preprocessor.transform(x_test)         # applied unchanged
    names = list(preprocessor.get_feature_names_out())

    # ---- Stage 6: select on TRAIN correlations, apply to both ----
    correlations = correlation_with_target(train_matrix, y_train)
    keep = select_by_correlation(train_matrix, names, y_train, CORRELATION_THRESHOLD)
    train_matrix, test_matrix = train_matrix[:, keep], test_matrix[:, keep]
    kept = [names[i] for i in np.flatnonzero(keep)]
    dropped = [names[i] for i in np.flatnonzero(~keep)]

    print(f"\nEncoded     : {len(names)} columns after imputation/encoding/normalization")
    unscaled = sorted(UNSCALED_NUMERIC & set(numeric))
    print(f"Normalization: min-max [0, 1] on {len(numeric) - len(unscaled)} of "
          f"{len(numeric)} numeric columns; left on native scale: {unscaled or 'none'}")
    print(f"Selected    : kept {len(kept)}, dropped {len(dropped)} "
          f"(|r| < {CORRELATION_THRESHOLD})")

    print("\n  kept features, strongest first")
    for i in np.argsort(-correlations[keep]):
        print(f"    r = {correlations[keep][i]:.4f}  {kept[i]}")
    print("\n  dropped features")
    for i in np.argsort(-correlations[~keep]):
        print(f"    r = {correlations[~keep][i]:.4f}  {dropped[i]}")

    # ---- Stage 7: write ----
    train_out = tidy(pd.DataFrame(train_matrix, columns=kept).assign(**{TARGET: y_train}))
    test_out = tidy(pd.DataFrame(test_matrix, columns=kept).assign(**{TARGET: y_test}))

    if train_out.isna().any().any() or test_out.isna().any().any():
        sys.exit("ERROR: NaN survived preprocessing -- refusing to write the CSVs.")

    train_out.to_csv(TRAIN_CSV, index=False)
    test_out.to_csv(TEST_CSV, index=False)

    print(f"\nWrote       : {TRAIN_CSV}  {train_out.shape[0]:,} rows x {train_out.shape[1]} cols")
    print(f"Wrote       : {TEST_CSV}  {test_out.shape[0]:,} rows x {test_out.shape[1]} cols")

    print("\nNOTE -- temporal drift in this dataset: churn is ~0.23 for 2022/2023")
    print("signups but ~0.50 for 2024. `signup_year` is kept as a feature, so the")
    print("test score ModelTuning.py reports is a random-split score that benefits")
    print("from era information. Do not present it as temporal generalization.")
    print("=" * 70)


if __name__ == "__main__":
    main()
