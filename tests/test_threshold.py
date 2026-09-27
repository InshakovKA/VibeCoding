"""`best_f1_threshold` must return a threshold that actually achieves the optimum.

This is the regression test for a real bug. The first implementation computed F1
from `roc_curve`'s returned `precision`/`recall` arrays and took an argmax over
them. Those arrays are not index-aligned with `thresholds`: at the lowest
threshold `roc_curve` reports precision 1.0 where the base rate is ~0.55, so the
argmax selected the degenerate "predict every row positive" solution. On the real
data that produced a threshold of 0.0015 and a test accuracy of 0.319.

The fix uses `precision_recall_curve`, whose alignment contract is documented.

The invariant asserted here is behavioural rather than a threshold comparison:
applying the returned threshold must reproduce the brute-force optimum F1.
Comparing threshold *values* across implementations is fragile -- the optimum is
a range, and `precision_recall_curve` reports a score value where brute force
searches midpoints.
"""

import unittest
import warnings

import numpy as np
from sklearn.metrics import f1_score

import ModelTuning as mt


def brute_force_best_f1(y_true, scores):
    """Ground truth: F1 at every midpoint between distinct scores, plus the extremes."""
    candidates = np.unique(scores)
    if candidates.size > 1:
        grid = np.concatenate(
            [[0.0], (candidates[:-1] + candidates[1:]) / 2, [1.0]]
        )
    else:
        grid = candidates
    best_f1, best_threshold = -1.0, 0.5
    for threshold in grid:
        score = f1_score(y_true, (scores >= threshold).astype(int), zero_division=0)
        if score > best_f1:
            best_f1, best_threshold = score, float(threshold)
    return best_threshold, best_f1


def apply_threshold(y_true, scores, threshold):
    return float(f1_score(y_true, (scores >= threshold).astype(int), zero_division=0))


class BestF1ThresholdTest(unittest.TestCase):
    def assertOptimal(self, y_true, scores):
        """The returned threshold must achieve the brute-force optimum F1."""
        threshold = mt.best_f1_threshold(y_true, scores)

        self.assertIsInstance(threshold, float)
        achieved = apply_threshold(y_true, scores, threshold)
        _, optimum = brute_force_best_f1(y_true, scores)

        self.assertAlmostEqual(
            achieved, optimum, places=9,
            msg=f"threshold {threshold!r} yields F1 {achieved}, optimum is {optimum}",
        )
        return threshold, achieved

    # -- the cases that caught the original bug, plus the awkward ones -------- #

    def test_separable_scores_reach_perfect_f1(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 400)
        scores = y * 0.6 + 0.2 + rng.random(400) * 0.05
        _, achieved = self.assertOptimal(y, scores)
        self.assertAlmostEqual(achieved, 1.0, places=9)

    def test_tied_scores(self):
        """Heavy ties break midpoint-style reasoning; the sweep must still be exact."""
        rng = np.random.default_rng(1)
        y = rng.integers(0, 2, 300)
        scores = np.round(0.2 + 0.5 * y + rng.random(300), 1)
        unique = len(np.unique(scores))
        self.assertLess(unique, 25, "rounding was supposed to collapse many values")
        self.assertGreater(len(scores) / unique, 5, "expected heavy duplication")
        self.assertOptimal(y, scores)

    def test_scores_saturated_at_the_ends(self):
        """Clipping creates exact 0.0 and 1.0 masses, i.e. a big tie group at each end."""
        rng = np.random.default_rng(2)
        y = rng.integers(0, 2, 500)
        scores = np.clip(0.3 + 0.3 * y + rng.normal(0, 0.15, 500), 0, 1)
        self.assertGreater(np.sum(scores == 0.0) + np.sum(scores == 1.0), 5)
        self.assertOptimal(y, scores)

    def test_pure_noise_beats_the_degenerate_all_positive_answer(self):
        """The original bug's exact failure mode: no signal, so F1 is nearly flat.

        An argmax over misaligned arrays returns the lowest threshold, which
        predicts everything positive. That must never win *unless* it genuinely
        ties for best -- with a random draw it sometimes does, so the strong
        assertion is conditional on a strictly better cut existing.
        """
        rng = np.random.default_rng(3)
        y = rng.integers(0, 2, 300)
        scores = rng.random(300)  # no relationship to y at all

        threshold, achieved = self.assertOptimal(y, scores)
        degenerate = apply_threshold(y, scores, 0.0)
        best_non_degenerate = max(
            apply_threshold(y, scores, t) for t in np.unique(scores)
        )

        if best_non_degenerate > degenerate:
            self.assertGreater(
                achieved, degenerate,
                msg="chose the degenerate all-positive solution",
            )
            self.assertGreater(threshold, 0.0)

    def test_extreme_imbalance(self):
        """~1% positives, the regime where a degenerate threshold looks attractive."""
        rng = np.random.default_rng(4)
        y = np.zeros(1000, dtype=int)
        positives = rng.choice(1000, 10, replace=False)
        y[positives] = 1
        scores = rng.random(1000)
        scores[positives] += 0.3
        self.assertOptimal(y, scores)

    def test_single_distinct_score(self):
        """Degenerate input: the sweep must still return a usable float."""
        y = np.array([0, 1, 0, 1])
        scores = np.full(4, 0.5)
        threshold = mt.best_f1_threshold(y, scores)
        self.assertIsInstance(threshold, float)

    def test_constant_labels(self):
        """A fold with one class only: no meaningful optimum, must not raise."""
        y = np.zeros(20, dtype=int)
        scores = np.linspace(0, 1, 20)
        with warnings.catch_warnings():
            # sklearn warns that recall is forced to 1 with no positive class.
            warnings.simplefilter("ignore", UserWarning)
            threshold = mt.best_f1_threshold(y, scores)
        self.assertIsInstance(threshold, float)

    def test_returns_a_probability_range_threshold_when_given_probabilities(self):
        """The real use case: OOF probabilities in [0, 1] must yield a sane cut."""
        rng = np.random.default_rng(5)
        y = np.zeros(2000, dtype=int)
        y[640:] = 1  # ~32% positive, matching the real class balance
        scores = np.clip(0.3 + 0.35 * y + rng.normal(0, 0.1, 2000), 0, 1)
        threshold = mt.best_f1_threshold(y, scores)
        self.assertGreater(threshold, 0.0)
        self.assertLess(threshold, 1.0)


if __name__ == "__main__":
    unittest.main()
