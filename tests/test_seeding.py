"""Determinism: the estimator must be seeded, not just the search.

`RandomizedSearchCV(random_state=...)` does *not* make the estimator it refits
deterministic. `HistGradientBoostingClassifier` with `early_stopping='auto'`
carves a validation set out of the **global** RNG when `random_state` is None, and
`RandomForestClassifier` samples bootstraps the same way. On this data an unseeded
HistGB gave `n_iter_` of 185 / 190 / 182 across three runs, which moved the winning
model's CV score between runs (0.97616 -> 0.97622) and would have made every number
in the report irreproducible.

These tests are deliberately cheap: they fit small models on small slices. The
claim "the full 21-minute run is bit-identical across runs" is checked manually,
not here.
"""

import unittest

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

import ModelTuning as mt
from tests import quiet

SEED = 42


class SeedEstimatorTest(unittest.TestCase):
    def setUp(self):
        self.config = mt.Config(
            cv={"random_state": SEED},
            search={"random_state": SEED},
            report={},
            candidates=[],
        )

    def candidate(self, name="m", **params):
        return mt.Candidate(name=name, class_path="sklearn.ensemble.HistGradientBoostingClassifier",
                            params=params if params else {"max_iter": [10]})

    def test_injects_a_seed_when_the_config_is_silent(self):
        candidate = self.candidate()
        estimator = mt.seed_estimator(
            HistGradientBoostingClassifier(), candidate, self.config
        )
        self.assertEqual(estimator.get_params()["random_state"], SEED)

    def test_respects_an_explicit_config_seed(self):
        """A `random_state` in the config grid means 'search over it'; do not pin it.

        `get_params()` always contains the key, so the check is that the value is
        still the estimator's own default rather than the injected seed.
        """
        candidate = self.candidate(random_state=[7])
        default = HistGradientBoostingClassifier().get_params()["random_state"]
        estimator = mt.seed_estimator(
            HistGradientBoostingClassifier(), candidate, self.config
        )
        self.assertEqual(estimator.get_params()["random_state"], default)

    def test_leaves_deterministic_estimators_alone(self):
        """An estimator with no random_state parameter must not be given one."""
        candidate = mt.Candidate(
            name="knn", class_path="sklearn.neighbors.KNeighborsClassifier",
            params={"n_neighbors": [5]},
        )
        estimator = mt.seed_estimator(
            KNeighborsClassifier(), candidate, self.config
        )
        self.assertNotIn("random_state", estimator.get_params())


class SearchSeedingTest(unittest.TestCase):
    """The estimator that the search will refit must already be seeded."""

    def setUp(self):
        self.config = mt.load_config(mt.CONFIG_PATH)
        self.cv = StratifiedKFold(2, shuffle=True, random_state=SEED)

    def test_every_configured_candidate_is_seeded(self):
        for candidate in self.config.candidates:
            if candidate.enabled:
                with self.subTest(candidate=candidate.name), quiet():
                    search = mt.make_search(candidate, self.config, self.cv)
                    seed = search.estimator.get_params().get("random_state")
                    self.assertIsNotNone(
                        seed,
                        f"{candidate.name} would be refit unseeded",
                    )
                    self.assertEqual(seed, 42)

    def test_real_config_candidates_all_accept_the_parameters_used(self):
        for candidate in self.config.candidates:
            if candidate.enabled:
                with self.subTest(candidate=candidate.name), quiet():
                    mt.make_search(candidate, self.config, self.cv)


class FittedModelDeterminismTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.x = rng.random((600, 8))
        self.y = (self.x[:, 0] + self.x[:, 1] + rng.normal(0, 0.3, 600) > 1.0).astype(int)

    def params(self):
        return dict(max_iter=40, max_leaf_nodes=8, random_state=SEED)

    def test_seeded_gradient_boosting_fits_identically_twice(self):
        first = HistGradientBoostingClassifier(**self.params()).fit(self.x, self.y)
        second = HistGradientBoostingClassifier(**self.params()).fit(self.x, self.y)
        np.testing.assert_array_equal(
            first.predict_proba(self.x), second.predict_proba(self.x)
        )
        self.assertEqual(first.n_iter_, second.n_iter_)

    def test_seeded_forest_fits_identically_twice(self):
        params = dict(n_estimators=25, random_state=SEED)
        first = RandomForestClassifier(**params).fit(self.x, self.y)
        second = RandomForestClassifier(**params).fit(self.x, self.y)
        np.testing.assert_array_equal(
            first.predict_proba(self.x), second.predict_proba(self.x)
        )

    def test_gradient_boosting_seed_comes_from_seed_estimator(self):
        """End to end: a bare estimator plus seed_estimator is reproducible."""
        config = mt.Config(cv={"random_state": SEED}, search={"random_state": SEED},
                           report={}, candidates=[])
        candidate = mt.Candidate(
            name="hgb",
            class_path="sklearn.ensemble.HistGradientBoostingClassifier",
            params={"max_iter": [40]},
        )
        params = {"max_iter": 40, "max_leaf_nodes": 8}
        first = mt.seed_estimator(HistGradientBoostingClassifier(**params), candidate, config)
        second = mt.seed_estimator(HistGradientBoostingClassifier(**params), candidate, config)
        np.testing.assert_array_equal(
            first.fit(self.x, self.y).predict_proba(self.x),
            second.fit(self.x, self.y).predict_proba(self.x),
        )

    def test_estimators_without_the_parameter_do_not_explode(self):
        """SVC has random_state but barely uses it; KNN has none. Neither may break."""
        svc = mt.seed_estimator(
            SVC(C=1.0), mt.Candidate("s", "sklearn.svm.SVC", params={"C": [1.0]}),
            mt.Config(cv={"random_state": SEED}, search={}, report={}, candidates=[]),
        )
        self.assertEqual(svc.get_params()["random_state"], SEED)
        knn = mt.seed_estimator(
            KNeighborsClassifier(n_neighbors=3),
            mt.Candidate("k", "sklearn.neighbors.KNeighborsClassifier", params={"n_neighbors": [3]}),
            mt.Config(cv={"random_state": SEED}, search={}, report={}, candidates=[]),
        )
        knn.fit(self.x, self.y)
        self.assertTrue(knn.predict(self.x).shape[0] == len(self.y))


if __name__ == "__main__":
    unittest.main()
