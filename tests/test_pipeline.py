"""End-to-end run of `ModelTuning.main()` on a tiny synthetic problem.

The real run takes ~21 minutes, so this exercises the whole `main()` -- config
loading, tuning, selection, the single test pass, threshold choice, permutation
importance, pickling and report writing -- in a couple of seconds on generated
data. It is the test that would catch a wiring mistake between the stages that the
unit tests cannot see.

The data is generated rather than read from `data/processed_*.csv` on purpose: those
files are git-ignored, so a fresh clone has no stage-1 output and the suite must
still be runnable. The generator below emits exactly what the stage-1 contract
promises -- dense numeric features, no NaN, `churn` last, ~32% positive.
"""

import pickle
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

import ModelTuning as mt
from tests import quiet

N_FEATURES = 16

TINY_CONFIG = """
[cv]
n_splits = 2
shuffle = true
random_state = 42
scoring = "roc_auc"
n_jobs = 1

[search]
random_iterations = 2
random_state = 42

[report]
decision_threshold = 0.5
top_k = 3
importance_rows = 4
output_dir = "{output_dir}"
model_filename = "tiny_model.pkl"
report_filename = "tiny_report.md"

[[models]]
name = "Tiny SVM"
class_path = "sklearn.svm.SVC"
search = "grid"
params = {{ kernel = ["rbf"], C = [1.0, 10.0], class_weight = ["none", "balanced"] }}

[[models]]
name = "Tiny GB"
class_path = "sklearn.ensemble.HistGradientBoostingClassifier"
search = "random"
params = {{ learning_rate = [0.1, 0.3], max_iter = [20, 40], max_leaf_nodes = [7] }}
"""


def make_stage1_like_frame(n_rows, seed):
    """A frame satisfying the stage-1 output contract, with learnable signal."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_rows, N_FEATURES))
    logit = (
        1.6 * x[:, 0]                     # the first column is the real driver
        - 0.9 * np.sin(3.0 * x[:, 1])    # a non-monotone one, to be worth tuning
        - 0.5 * x[:, 2]
    )
    probability = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.random(n_rows) < probability * 0.6).astype(int)

    frame = pd.DataFrame(x, columns=[f"num__{i}" for i in range(N_FEATURES)])
    frame["churn"] = y
    assert not frame.isna().any().any()
    assert list(frame.columns)[-1] == "churn"
    return frame


class MainEndToEndTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)

        cls.train_path = root / "processed_train.csv"
        cls.test_path = root / "processed_test.csv"
        make_stage1_like_frame(1200, seed=0).to_csv(cls.train_path, index=False)
        make_stage1_like_frame(300, seed=1).to_csv(cls.test_path, index=False)

        cls.output_dir = root / "out"
        cls.output_dir.mkdir()
        cls.config_path = root / "model_tuning.toml"
        cls.config_path.write_text(
            TINY_CONFIG.format(output_dir=cls.output_dir.as_posix()),
            encoding="utf-8",
        )

        cls.model_path = cls.output_dir / "tiny_model.pkl"
        cls.report_path = cls.output_dir / "tiny_report.md"

        # `load_config`'s default argument captured CONFIG_PATH at def time, so
        # patching the module attribute is not enough -- patch the function, and
        # hold a reference to the real one to avoid recursing into the patch.
        real_load_config = mt.load_config
        with mock.patch.object(mt, "CONFIG_PATH", cls.config_path), \
             mock.patch.object(mt, "TRAIN_CSV", cls.train_path), \
             mock.patch.object(mt, "TEST_CSV", cls.test_path), \
             mock.patch.object(mt, "load_config",
                               lambda: real_load_config(cls.config_path)), \
             quiet():
            mt.main()

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    # -- artifacts ----------------------------------------------------------- #

    def test_writes_both_artifacts(self):
        self.assertTrue(self.model_path.exists(), "no pickle written")
        self.assertTrue(self.report_path.exists(), "no report written")

    def test_pickle_holds_a_fitted_estimator_that_predicts(self):
        model = pickle.loads(self.model_path.read_bytes())
        self.assertTrue(
            hasattr(model, "predict_proba") or hasattr(model, "decision_function")
        )
        frame = pd.read_csv(self.test_path)
        features = [c for c in frame.columns if c != "churn"]
        scores = (
            model.predict_proba(frame[features].to_numpy())[:, 1]
            if hasattr(model, "predict_proba")
            else model.decision_function(frame[features].to_numpy())
        )
        self.assertEqual(len(scores), len(frame))

    def test_pickled_estimator_is_seeded(self):
        """The artifact must be as reproducible as the run that produced it."""
        model = pickle.loads(self.model_path.read_bytes())
        self.assertEqual(model.get_params().get("random_state"), 42)

    def test_pickle_reloads_and_agrees_with_itself(self):
        """Two independent loads must give identical predictions."""
        frame = pd.read_csv(self.test_path)
        features = [c for c in frame.columns if c != "churn"]
        x = frame[features].to_numpy()
        first = pickle.loads(self.model_path.read_bytes())
        second = pickle.loads(self.model_path.read_bytes())
        if hasattr(first, "predict_proba"):
            np.testing.assert_array_equal(
                first.predict_proba(x), second.predict_proba(x)
            )
        else:
            np.testing.assert_array_equal(
                first.decision_function(x), second.decision_function(x)
            )

    # -- report contents ----------------------------------------------------- #

    def test_report_names_the_selected_model(self):
        report = self.report_path.read_text(encoding="utf-8")
        self.assertRegex(report, r"\*\*Tiny (SVM|GB)\*\*")

    def test_report_states_the_data_sizes_it_used(self):
        """Proves the run consumed the frames the test fed it, not the real ones."""
        report = self.report_path.read_text(encoding="utf-8")
        self.assertIn("1,200-row", report)
        self.assertIn("300-row", report)

    def test_report_lists_both_candidates(self):
        report = self.report_path.read_text(encoding="utf-8")
        self.assertIn("Tiny SVM", report)
        self.assertIn("Tiny GB", report)
        self.assertIn("<- selected", report)

    def test_report_is_ascii_and_readable(self):
        report = self.report_path.read_text(encoding="utf-8")
        report.encode("ascii")

    def test_reported_auc_is_a_probability(self):
        report = self.report_path.read_text(encoding="utf-8")
        match = re.search(r"\| ROC AUC \| \*\*([0-9.]+)\*\* \|", report)
        self.assertIsNotNone(match, "no bold ROC AUC row in the report")
        value = float(match.group(1))
        self.assertGreaterEqual(value, 0.5)
        self.assertLessEqual(value, 1.0)


class MismatchedSplitTest(unittest.TestCase):
    """Stage 2 refuses to run when the two CSVs disagree about the features."""

    def test_column_order_mismatch_is_rejected(self):
        train = make_stage1_like_frame(80, seed=2)
        test = make_stage1_like_frame(40, seed=3)
        # Same columns, different order -- a plausible accident.
        reordered = test[list(test.columns)[::-1]]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "t.csv").write_text(
                ",".join(train.columns) + "\n" + train.to_csv(index=False).split("\n", 1)[1],
                encoding="utf-8",
            )
            (root / "e.csv").write_text(
                ",".join(reordered.columns) + "\n"
                + reordered.to_csv(index=False).split("\n", 1)[1],
                encoding="utf-8",
            )
            x_train, _, train_features = mt.split_xy(
                pd.read_csv(root / "t.csv"), "t.csv"
            )
            x_test, _, test_features = mt.split_xy(
                pd.read_csv(root / "e.csv"), "e.csv"
            )
            self.assertNotEqual(train_features, test_features)
            self.assertEqual(x_train.shape[0], 80)
            self.assertEqual(x_test.shape[0], 40)

    def test_nan_is_rejected(self):
        """Stage-1 output is dense; NaN means something upstream went wrong."""
        frame = make_stage1_like_frame(50, seed=4)
        frame.iloc[0, 0] = np.nan
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.csv"
            frame.to_csv(path, index=False)
            with self.assertRaises(mt.ConfigError) as caught:
                mt.split_xy(pd.read_csv(path), path.name)
            self.assertIn("non-finite", str(caught.exception))

    def test_missing_target_column_is_rejected(self):
        frame = make_stage1_like_frame(50, seed=5).drop(columns=["churn"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "no_target.csv"
            frame.to_csv(path, index=False)
            with self.assertRaises(mt.ConfigError) as caught:
                mt.split_xy(pd.read_csv(path), path.name)
            self.assertIn("churn", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
