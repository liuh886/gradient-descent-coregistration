import numpy as np
import pytest

from gdc.core.stats import (
    compute_mix_loss,
    compute_nmad,
    compute_residuals_statistics,
    compute_rmse,
    filter_outliers,
)


def test_nmad_normal_distribution():
    rng = np.random.default_rng(123)
    sigma = 2.5
    data = rng.normal(loc=10.0, scale=sigma, size=50000)
    nmad_val = compute_nmad(data)
    # NMAD of normal distribution should match sigma within ~1%
    assert pytest.approx(sigma, rel=0.03) == nmad_val


def test_nmad_outlier_resistance():
    data = np.array([0.1, -0.2, 0.05, -0.05, 0.1, -0.1, 100.0, -200.0])
    nmad_val = compute_nmad(data)
    # The extreme outliers 100.0 and -200.0 should not blow up the NMAD
    assert nmad_val < 1.0


def test_nmad_nan_handling():
    data = [1.0, 2.0, np.nan, 3.0, 2.0]
    val = compute_nmad(data, nan_policy="omit")
    assert np.isfinite(val)
    assert val > 0.0

    empty_val = compute_nmad([])
    assert np.isnan(empty_val)


def test_rmse():
    data = np.array([3.0, -4.0])
    # sqrt((9 + 16)/2) = sqrt(12.5) ~ 3.5355
    val = compute_rmse(data)
    assert pytest.approx(np.sqrt(12.5)) == val


def test_mix_loss():
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    nmad_val = compute_nmad(data)
    rmse_val = compute_rmse(data)
    expected_mix = (2.0 * nmad_val + rmse_val) / 3.0
    assert pytest.approx(expected_mix) == compute_mix_loss(data)


def test_filter_outliers():
    data = np.concatenate([np.zeros(100), [999.0, -999.0]])
    filtered, mask = filter_outliers(data, perc_t=99.0)
    assert 999.0 not in filtered
    assert -999.0 not in filtered
    assert len(filtered) < len(data)


def test_compute_residuals_statistics():
    data = np.array([1.0, 2.0, 1.5, 1.8, 1.2, 500.0])
    stats = compute_residuals_statistics(data, d_lim=50.0)
    assert stats["outlier_count"] == 1
    assert stats["count"] == 5
    assert pytest.approx(stats["median"], abs=0.5) == 1.5
