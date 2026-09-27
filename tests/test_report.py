"""`build_report` output contract.

The report is a deliverable, so its shape is worth pinning down: it must stay
ASCII (the repo path and Windows console are not UTF-8), must never leak `nan`
into a markdown table, and must carry the information the pickle cannot -- namely
the positional feature order.

Two branch-gated behaviours are covered: the uncalibrated-model caveat only fires
when the winner has no `predict_proba`, and the Brier row degrades to `n/a`
instead of printing `nan` in that case.
"""

import unittest

import ModelTuning as mt

FEATURES = [f"num__f{i}" for i in range(6)] + [f"cat__c{i}" for i in range(10)]


class HasProba:
    def predict_proba(self, x):
        return x


class HasDecisionFunctionOnly:
    def decision_function(self, x):
        return x


def threshold_block():
    return {
        "threshold": 0.5, "accuracy": 0.9, "balanced_accuracy": 0.9,
        "precision": 0.9, "recall": 0.9, "f1": 0.9,
        "tn": 1, "fp": 2, "fn": 3, "tp": 4,
    }


class BuildReportTest(unittest.TestCase):
    def setUp(self):
        self.config = mt.load_config(mt.CONFIG_PATH)
        candidate = [c for c in self.config.candidates
                     if "Gradient" in c.name][0]
        self.trial = mt.Trial(
            candidate=candidate,
            best_score=0.9762,
            best_params={"max_iter": 200, "random_state": 42},
            best_index=0,
            n_candidates=48,
            n_fits=240,
            seconds=202.6,
            results=[{"params": {"a": 1}, "mean_score": 0.97, "std_score": 0.003}],
            estimator=HasProba(),
        )

    def render(self, estimator, brier=0.048, importances=(("num__f0", 0.15),)):
        self.trial.estimator = estimator
        return mt.build_report(
            config=self.config,
            trials=[self.trial],
            winner=self.trial,
            test_metrics={
                "roc_auc": 0.9769,
                "average_precision": 0.9646,
                "brier": brier,
                "default": threshold_block(),
            },
            test_at_tuned_threshold=threshold_block(),
            train_threshold=0.4193,
            importances=list(importances),
            feature_names=FEATURES,
            n_train=12000,
            n_test=3000,
        )

    # -- output hygiene ------------------------------------------------------ #

    def test_report_is_pure_ascii(self):
        report = self.render(HasProba())
        report.encode("ascii")  # raises if any non-ASCII character slipped in

    def test_report_never_contains_a_bare_nan(self):
        """A `nan` in a markdown table is worse than a blank cell."""
        self.assertNotIn("nan", self.render(HasProba(), brier=0.048))

    def test_brier_is_reported_when_the_model_has_probabilities(self):
        self.assertIn("| Brier score | 0.04800 |", self.render(HasProba()))

    def test_brier_degrades_to_na_when_there_are_no_probabilities(self):
        report = self.render(HasDecisionFunctionOnly(), brier=None)
        self.assertIn("no `predict_proba`", report)
        self.assertNotIn("nan", report)

    # -- the pickle's missing context ---------------------------------------- #

    def test_feature_order_is_printed_positionally(self):
        """The pickle holds only the estimator, so column order is documented here."""
        report = self.render(HasProba())
        self.assertIn("Feature order, positionally", report)
        for index, name in enumerate(FEATURES):
            self.assertIn(f"{index:2d}  {name}", report)

    def test_reproducibility_note_is_present(self):
        self.assertIn("n_iter_` 185/190/182", self.render(HasProba()))

    # -- branch-gated caveats ------------------------------------------------ #

    def test_calibration_caveat_fires_only_without_predict_proba(self):
        self.assertNotIn("ships uncalibrated", self.render(HasProba()))
        self.assertIn("ships uncalibrated",
                      self.render(HasDecisionFunctionOnly()))

    def test_stale_svc_specific_caveat_is_gone(self):
        """The caveat used to name SVC unconditionally, even when SVC did not win."""
        for estimator in (HasProba(), HasDecisionFunctionOnly()):
            report = self.render(estimator)
            self.assertNotIn("`SVC` ships uncalibrated", report)

    def test_split_caveat_says_the_score_is_not_temporal(self):
        report = self.render(HasProba())
        self.assertIn("not a temporal one", report)
        self.assertIn("temporal holdout", report)

    # -- structure ----------------------------------------------------------- #

    def test_sections_present(self):
        report = self.render(HasProba())
        for heading in ("## Selected model", "## Test-set results",
                        "## Model selection leaderboard", "## Caveats"):
            self.assertIn(heading, report)

    def test_importance_section_omitted_when_there_is_none(self):
        report = self.render(HasProba(), importances=())
        self.assertNotIn("Permutation importance", report)

    def test_importance_provenance_is_stated(self):
        """Must make clear the model was refit on all of train."""
        report = self.render(HasProba())
        self.assertIn("stratified 25% holdout", report)
        self.assertIn("in-sample", report)


if __name__ == "__main__":
    unittest.main()
