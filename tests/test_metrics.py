import numpy as np
import pytest
from scipy.stats import pearsonr
import torch

from tt_ecog.losses import make_loss, mse_cosine_loss, temporal_cosine
from tt_ecog.metrics import trajectory_metrics

pytestmark = pytest.mark.verification("V14")
REL = 1e-9


def ccc(t, p):
    return 2 * np.mean((t - t.mean()) * (p - p.mean())) / (t.var() + p.var() + (t.mean() - p.mean()) ** 2)


@pytest.fixture
def series():
    rng = np.random.default_rng(0)
    t = np.cumsum(rng.normal(size=(5, 800)), axis=1)
    p = np.vstack([t[0] + rng.normal(scale=3, size=800), -t[1] + 2, t[2], 0.5 * t[3] + 10, rng.normal(size=800)])
    return t, p


def test_pearson_ccc_and_errors_match_manual_references(series):
    t, p = series
    valid = np.ones(800, dtype=bool)
    valid[:37] = False
    result = trajectory_metrics(t, p, valid, degenerate_std_relative=REL)
    assert result["valid"] and result["n_samples"] == 763 and not result["diagnostics"]
    for f in range(5):
        tt, pp = t[f, valid], p[f, valid]
        assert result["pearson"][f] == pytest.approx(pearsonr(tt, pp)[0], abs=1e-12)
        assert result["ccc"][f] == pytest.approx(ccc(tt, pp), abs=1e-12)
        assert result["mse"][f] == pytest.approx(np.mean((pp - tt) ** 2), rel=1e-12)
        assert result["rmse"][f] == pytest.approx(np.sqrt(np.mean((pp - tt) ** 2)), rel=1e-12)
    assert result["pearson"][2] == pytest.approx(1.0) and result["pearson"][1] == pytest.approx(-1.0)
    assert result["pearson_macro"] == pytest.approx(np.mean(result["pearson"]), abs=1e-15)


def test_ccc_penalises_scale_and_offset_while_pearson_does_not(series):
    t, _ = series
    valid = np.ones(800, dtype=bool)
    shifted = trajectory_metrics(t, 2 * t + 5, valid, degenerate_std_relative=REL)
    assert np.allclose(shifted["pearson"], 1.0) and max(shifted["ccc"]) < 0.9


@pytest.mark.parametrize("what", ["constant_prediction", "constant_target", "nan_prediction", "near_constant"])
def test_undefined_pearson_invalidates_the_evaluation(series, what):
    t, p = (a.copy() for a in series)
    if what == "constant_prediction":
        p[3] = 4.2
    elif what == "constant_target":
        t[3] = -1.0
    elif what == "nan_prediction":
        p[3, 100] = np.nan
    else:
        p[3] = 1e6 + 1e-6 * np.sin(np.arange(800))
    result = trajectory_metrics(t, p, np.ones(800, dtype=bool), degenerate_std_relative=REL)
    assert not result["valid"] and result["pearson_macro"] is None and result["pearson"][3] is None
    assert all(value is not None for k, value in enumerate(result["pearson"]) if k != 3)
    assert any(d.startswith("ring") for d in result["diagnostics"])


def test_too_few_samples_and_bad_shapes(series):
    t, p = series
    assert not trajectory_metrics(t, p, np.zeros(800, dtype=bool), degenerate_std_relative=REL)["valid"]
    with pytest.raises(ValueError):
        trajectory_metrics(t, p[:, :10], np.ones(800, dtype=bool), degenerate_std_relative=REL)


def test_loss_matches_manual_formula(config):
    rng = np.random.default_rng(1)
    pred, target = rng.normal(size=(4, 5, 256)), rng.uniform(size=(4, 5, 256))
    cos = (pred * target).sum(-1) / np.sqrt((pred ** 2).sum(-1) * (target ** 2).sum(-1))
    expected = 0.5 * np.mean((pred - target) ** 2) + 0.5 * (1 - cos.mean())
    value = make_loss(config)(torch.tensor(pred), torch.tensor(target))
    assert float(value) == pytest.approx(expected, rel=1e-12)
    perfect = make_loss(config)(torch.tensor(target), torch.tensor(target))
    assert float(perfect) == pytest.approx(0.0, abs=1e-12)


def test_cosine_is_temporal_per_finger_and_finite_at_zero():
    pred = torch.zeros(2, 5, 64, requires_grad=True)
    target = torch.rand(2, 5, 64)
    target[0, 2] = 0
    cosine = temporal_cosine(pred, target, 1e-8)
    assert cosine.shape == (2, 5) and torch.equal(cosine, torch.zeros(2, 5))
    loss = mse_cosine_loss(pred, target, mse_weight=0.5, cosine_weight=0.5, eps=1e-8)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(pred.grad).all()
    with pytest.raises(ValueError):
        temporal_cosine(torch.zeros(2, 5, 64), torch.zeros(2, 5, 63), 1e-8)
