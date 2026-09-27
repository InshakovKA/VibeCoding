"""Config parsing, the "none" sentinel, and every rejection path.

Two real bugs are covered here:

* `params = {}` passed the "is `params` a required key" check, so it became a
  legitimate-looking 1-point search that competed in the leaderboard on plain
  defaults. It must be rejected.
* `class_path` was originally imported lazily, so a typo only surfaced *after* the
  previous candidate had finished tuning -- minutes into a 21-minute run. Loading
  must resolve every class path and reject parameter names the estimator does not
  accept, which is also the guard that catches a stale-version parameter name.
"""

import tempfile
import unittest
from pathlib import Path

import ModelTuning as mt

VALID = """
[cv]
n_splits = 5
shuffle = true
random_state = 42
scoring = "roc_auc"
n_jobs = -1

[search]
random_iterations = 4
random_state = 42

[report]
decision_threshold = 0.5
output_dir = "."
model_filename = "m.pkl"
report_filename = "r.md"

[[models]]
name = "SVM"
class_path = "sklearn.svm.SVC"
search = "grid"
params = { C = [1.0, 10.0], class_weight = ["none", "balanced"] }

[[models]]
name = "RF"
class_path = "sklearn.ensemble.RandomForestClassifier"
search = "random"
params = { n_estimators = [100], max_depth = ["none", 4] }
"""


def load_text(body):
    """Write `body` to a temp file and load it, returning the Config."""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "model_tuning.toml"
        path.write_text(body, encoding="utf-8")
        return mt.load_config(path)


class ConfigLoadingTest(unittest.TestCase):
    def test_loads_candidates_and_settings(self):
        config = load_text(VALID)
        self.assertEqual(config.n_splits, 5)
        self.assertEqual(config.scoring, "roc_auc")
        self.assertEqual(config.n_jobs, -1)
        self.assertEqual(config.random_iterations, 4)
        self.assertEqual(config.decision_threshold, 0.5)
        self.assertEqual([c.name for c in config.candidates], ["SVM", "RF"])
        self.assertTrue(all(c.enabled for c in config.candidates))

    def test_grid_size_is_the_product_of_parameter_lists(self):
        config = load_text(VALID)
        svm, rf = config.candidates
        self.assertEqual(svm.grid_size, 2 * 2)  # C x class_weight
        self.assertEqual(rf.grid_size, 1 * 2)  # n_estimators x max_depth

    def test_model_and_report_paths_come_from_the_report_table(self):
        config = load_text(VALID)
        self.assertEqual(config.model_path, Path("m.pkl"))
        self.assertEqual(config.report_path, Path("r.md"))


class NoneSentinelTest(unittest.TestCase):
    """TOML has no null literal, so "no value" is spelled "none"."""

    def test_none_becomes_python_none(self):
        config = load_text(VALID)
        svm = config.candidates[0]
        self.assertEqual(svm.params["class_weight"], [None, "balanced"])
        self.assertEqual(config.candidates[1].params["max_depth"], [None, 4])

    def test_sentinel_is_case_insensitive(self):
        body = VALID.replace(
            'class_weight = ["none", "balanced"]',
            'class_weight = ["none", "NONE", "None", "scale"]',
        )
        config = load_text(body)
        self.assertEqual(
            config.candidates[0].params["class_weight"],
            [None, None, None, "scale"],
        )

    def test_sentinel_does_not_clobber_other_strings(self):
        """`gamma = "scale"` must survive; the match is exact, not a substring test."""
        body = VALID.replace(
            'params = { C = [1.0, 10.0], class_weight = ["none", "balanced"] }',
            'params = { gamma = ["scale", "auto", "none"], C = [1.0] }',
        )
        config = load_text(body)
        self.assertEqual(
            config.candidates[0].params["gamma"], ["scale", "auto", None]
        )


class RejectionTest(unittest.TestCase):
    def assertRejected(self, body, needle):
        with self.assertRaises(mt.ConfigError) as caught:
            load_text(body)
        self.assertIn(needle.lower(), str(caught.exception).lower())

    def test_malformed_toml(self):
        self.assertRejected("[cv]\nbroken = ", "not valid toml")

    def test_no_models_declared(self):
        self.assertRejected(VALID.split("[[models]]")[0], "no [[models]]")

    def test_missing_params_key(self):
        self.assertRejected(
            VALID + '\n[[models]]\nname="x"\nclass_path="sklearn.svm.SVC"\n',
            "missing 'params'",
        )

    def test_empty_params_table(self):
        """Present but empty -- the bug a required-key check cannot catch."""
        self.assertRejected(
            VALID + '\n[[models]]\nname="x"\nclass_path="sklearn.svm.SVC"\nparams={}\n',
            "params is empty",
        )

    def test_empty_parameter_list(self):
        self.assertRejected(
            VALID.replace(
                'params = { C = [1.0, 10.0], class_weight = ["none", "balanced"] }',
                "params = { C = [] }",
            ),
            "empty candidate list",
        )

    def test_parameter_value_is_not_a_list(self):
        self.assertRejected(
            VALID.replace(
                'params = { C = [1.0, 10.0], class_weight = ["none", "balanced"] }',
                "params = { C = 1.0 }",
            ),
            "must be a list",
        )

    def test_duplicate_candidate_names(self):
        self.assertRejected(
            VALID + '\n[[models]]\nname="SVM"\nclass_path="sklearn.svm.SVC"\nparams={C=[1]}\n',
            "duplicate",
        )

    def test_all_candidates_disabled(self):
        body = (VALID.replace('name = "SVM"', 'name = "SVM"\nenabled = false')
                    .replace('name = "RF"', 'name = "RF"\nenabled = false'))
        self.assertRejected(body, "disabled")

    def test_missing_config_file(self):
        with self.assertRaises(mt.ConfigError) as caught:
            mt.load_config(Path("definitely-not-here.toml"))
        self.assertIn("not found", str(caught.exception))

    def test_class_path_without_a_module(self):
        self.assertRejected(VALID.replace("sklearn.svm.SVC", "SVC"), "must look like")

    def test_unknown_module(self):
        self.assertRejected(
            VALID.replace("sklearn.svm.SVC", "nosuchmodule.SVC"), "cannot import"
        )

    def test_unknown_class_attribute(self):
        self.assertRejected(
            VALID.replace("sklearn.svm.SVC", "sklearn.svm.NoSuchClassifier"),
            "no attribute",
        )

    def test_parameter_the_estimator_does_not_accept(self):
        self.assertRejected(
            VALID.replace(
                'params = { C = [1.0, 10.0], class_weight = ["none", "balanced"] }',
                "params = { n_estimators = [100] }",
            ),
            "does not accept",
        )


class DisablingCandidatesTest(unittest.TestCase):
    def test_disabling_one_entry_keeps_the_others_enabled(self):
        body = VALID.replace('name = "RF"', 'name = "RF"\nenabled = false')
        config = load_text(body)
        self.assertEqual([c.name for c in config.candidates if c.enabled], ["SVM"])
        self.assertEqual(len(config.candidates), 2)


class ClassPathResolutionTest(unittest.TestCase):
    def test_current_sklearn_version_accepts_the_parameters_we_configure(self):
        """Guards the version trap: `max_features` exists on HGB in 1.5+, not 1.3.

        An earlier note in AGENTS.md was written from a signature dump taken in the
        Anaconda environment (sklearn 1.3), which lacked this parameter and would
        have raised `ValueError` at fit time. Asserting against the venv in use is
        the cheap way to notice that class of drift.
        """
        body = (
            VALID.split("[[models]]")[0]
            + '\n[[models]]\nname="HGB"\n'
            'class_path="sklearn.ensemble.HistGradientBoostingClassifier"\n'
            'search="random"\n'
            'params={ max_features=[0.6,1.0], max_iter=[200] }\n'
        )
        config = load_text(body)
        self.assertEqual(
            config.candidates[0].params["max_features"], [0.6, 1.0]
        )


class SearchKindTest(unittest.TestCase):
    def test_unknown_search_kind_is_rejected_when_the_search_is_built(self):
        config = load_text(VALID.replace('search = "grid"', 'search = "bayesian"'))
        self.assertEqual(config.candidates[0].search, "bayesian")
        with self.assertRaises(mt.ConfigError) as caught:
            mt.make_search(config.candidates[0], config, cv=object())
        self.assertIn("use 'grid' or 'random'", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
