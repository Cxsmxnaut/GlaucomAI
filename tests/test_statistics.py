import math

import pytest

from glaucoma.statistics import cluster_bootstrap, roc_auc, score_statistics, wilson_interval


def test_wilson_intervals_include_uncertainty_even_for_perfect_small_samples():
    low, high = wilson_interval(14, 14)
    assert low == pytest.approx(.7846891973)
    assert high == pytest.approx(1.)
    assert wilson_interval(0, 0) is None
    with pytest.raises(ValueError):
        wilson_interval(4, 3)


def test_auc_handles_ties_and_degenerate_labels():
    assert roc_auc([0, 1], [.1, .9]) == 1.
    assert roc_auc([0, 1], [.9, .1]) == 0.
    assert roc_auc([0, 1], [.5, .5]) == .5
    assert roc_auc([1, 1], [.1, .9]) is None
    assert roc_auc([0, 1, 1, 0], [.2, .2, .8, .8]) == .5


def test_statistics_match_confusion_counts_and_observed_brier_score():
    result = score_statistics([0, 1, 0, 1], [.1, .7, .8, .4], .5)
    assert result["true_positive"] == 1 and result["false_negative"] == 1
    assert result["sample_ppv"] == .5
    assert result["brier_score"] == pytest.approx((.01 + .09 + .64 + .36) / 4)
    assert result["sensitivity_wilson_95"][0] < .5 < result["sensitivity_wilson_95"][1]


def test_bootstrap_resamples_patients_and_reports_missing_class_replicates():
    labels = [0, 0, 1, 1]
    scores = [.1, .1, .9, .1]
    groups = ["healthy", "healthy", "glaucoma", "glaucoma"]
    result = cluster_bootstrap(labels, scores, groups, .5, repetitions=200, seed=42)
    assert result["groups"] == 2
    assert result == cluster_bootstrap(labels, scores, groups, .5, repetitions=200, seed=42)
    sensitivity = result["intervals"]["sensitivity"]
    assert sensitivity["lower"] == sensitivity["upper"] == .5
    assert sensitivity["valid_replicates"] < 200
    assert math.isfinite(result["intervals"]["accuracy"]["lower"])
    with pytest.raises(ValueError):
        cluster_bootstrap(labels, scores, ["same"] * 4, .5)
