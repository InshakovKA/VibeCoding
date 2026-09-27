"""Contract checks on the real stage-1 output, when it is present.

`data/processed_train.csv` and `data/processed_test.csv` are git-ignored, so a
fresh clone has neither file. Every test here skips in that case rather than
failing -- the suite must be runnable without first running the 21-minute stage-2
step, and without committing generated data.

When the files *are* present these assert the invariants stage 2 silently relies
on. They are the tests that would notice if `DataProcessing.py` started emitting a
different column order or left NaN behind, which is the kind of drift that shows
up much later as a mysteriously worse model rather than as an error.
"""

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import ModelTuning as mt

TRAIN = Path("data/processed_train.csv")
TEST = Path("data/processed_test.csv")

HAVE_STAGE1 = TRAIN.exists() and TEST.exists()
skip_without_stage1 = unittest.skipUnless(
    HAVE_STAGE1,
    "stage-1 output not present; run DataProcessing.py to exercise these",
)


@skip_without_stage1
class Stage1OutputContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = pd.read_csv(TRAIN)
        cls.test = pd.read_csv(TEST)

    def test_shapes_match_the_specified_split(self):
        self.assertEqual((len(self.train), len(self.test)), (12000, 3000))
        self.assertEqual(len(self.train) / (len(self.train) + len(self.test)), 0.8)

    def test_target_is_the_last_column(self):
        for name, frame in (("train", self.train), ("test", self.test)):
            with self.subTest(split=name):
                self.assertEqual(list(frame.columns)[-1], mt.TARGET)

    def test_no_missing_values(self):
        for name, frame in (("train", self.train), ("test", self.test)):
            with self.subTest(split=name):
                self.assertFalse(frame.isna().any().any())
                self.assertTrue(np.isfinite(frame.to_numpy(dtype=float)).all())

    def test_train_and_test_share_the_feature_columns_in_order(self):
        train_features = [c for c in self.train.columns if c != mt.TARGET]
        test_features = [c for c in self.test.columns if c != mt.TARGET]
        self.assertEqual(train_features, test_features)

    def test_identifier_and_raw_date_are_gone(self):
        columns = set(self.train.columns)
        self.assertNotIn("customer_id", columns)
        self.assertNotIn("signup_date", columns)

    def test_column_names_carry_their_origin_prefix(self):
        for column in self.train.columns:
            if column == mt.TARGET:
                continue
            with self.subTest(column=column):
                self.assertTrue(
                    column.startswith("num__") or column.startswith("cat__"),
                    f"{column} has neither a num__ nor a cat__ prefix",
                )

    def test_satisfaction_score_is_left_unscaled(self):
        """The one feature deliberately kept on its native 1-5 scale."""
        column = "num__satisfaction_score"
        self.assertIn(column, self.train.columns)
        values = self.train[column].to_numpy()
        self.assertGreater(values.min(), 0.5, "looks min-max scaled, should be 1-5")
        self.assertLessEqual(values.max(), 5.0 + 1e-9)

    def test_remaining_numeric_features_are_min_max_scaled(self):
        """Nine of the ten `num__` columns should sit in [0, 1] from the train range.

        Values slightly outside [0, 1] are expected and legitimate: `MinMaxScaler`
        does not clip, so a test row below the training minimum lands negative.
        The check is therefore on the *train* split only.
        """
        numeric = [c for c in self.train.columns
                   if c.startswith("num__") and c != "num__satisfaction_score"]
        self.assertTrue(numeric)
        for column in numeric:
            with self.subTest(column=column):
                values = self.train[column].to_numpy()
                self.assertGreaterEqual(values.min(), -1e-9)
                self.assertLessEqual(values.max(), 1.0 + 1e-9)

    def test_class_balance_matches_the_known_distribution(self):
        """~32% positive, the imbalance that makes accuracy a poor selection metric."""
        for name, frame in (("train", self.train), ("test", self.test)):
            with self.subTest(split=name):
                rate = frame[mt.TARGET].mean()
                self.assertGreater(rate, 0.25)
                self.assertLess(rate, 0.40)

    def test_split_xy_accepts_both_files(self):
        """The exact call `ModelTuning.main` makes must succeed."""
        x_train, y_train, features = mt.split_xy(self.train, TRAIN.name)
        x_test, y_test, test_features = mt.split_xy(self.test, TEST.name)
        self.assertEqual(features, test_features)
        self.assertEqual(x_train.shape[1], x_test.shape[1])
        self.assertEqual(x_train.shape[0], len(self.train))
        self.assertEqual(x_test.shape[0], len(self.test))
        self.assertIn(0, np.unique(y_train))
        self.assertIn(1, np.unique(y_train))


if __name__ == "__main__":
    unittest.main()
