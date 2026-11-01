import json
import numpy as np
import pytest

from gdc.core.sampling import AffineTransform, sample_grid_at_coordinates
from gdc.qc.checker import check_dem


@pytest.fixture
def dem_and_points():
    res = 10.0
    trans = AffineTransform(res, 0.0, 1000.0, 0.0, -res, 2000.0)
    cols, rows = np.meshgrid(np.arange(100), np.arange(100))
    grid = (
        0.04 * cols * res
        - 0.03 * rows * res
        + 8.0 * np.sin(cols / 8.0)
        + 12.0 * np.cos(rows / 9.0)
        + 500.0
    )

    rng = np.random.default_rng(101)
    n = 1200
    x_pts = rng.uniform(1100.0, 1800.0, n)
    y_pts = rng.uniform(1100.0, 1800.0, n)
    z_pts = sample_grid_at_coordinates(grid, trans, x_pts, y_pts, order=1)

    return (grid, trans), (x_pts, y_pts, z_pts)


def test_qc_report_reliable_tier(dem_and_points):
    dem, pts = dem_and_points
    report = check_dem(
        dem=dem,
        points=pts,
        bounds_px=3.0,
        multi_start=False,
    )

    assert report.tier == "reliable"
    assert "reliable" in report.tier
    assert report.coreg_result.nmad_after < 0.1
    assert "translation" in report.to_dict()
    assert "uncertainty" in report.to_dict()

    json_str = report.to_json()
    data = json.loads(json_str)
    assert data["tier"] == "reliable"


def test_qc_report_cautious_tier_due_to_sparse_sample(dem_and_points):
    dem, pts = dem_and_points
    x, y, z = pts
    # Cut down to 500 points (below 1000 threshold)
    sparse_pts = (x[:500], y[:500], z[:500])

    report = check_dem(
        dem=dem,
        points=sparse_pts,
        bounds_px=3.0,
    )

    assert report.tier == "cautious"
    assert any("Sparse point sample" in r for r in report.tier_reasons)


def test_qc_report_unreliable_tier_few_points(dem_and_points):
    dem, pts = dem_and_points
    x, y, z = pts
    # Cut down to 100 points (below 300 threshold)
    very_sparse_pts = (x[:100], y[:100], z[:100])

    report = check_dem(
        dem=dem,
        points=very_sparse_pts,
        bounds_px=3.0,
    )

    assert report.tier == "unreliable"
    assert any("Insufficient valid points" in r for r in report.tier_reasons)


def test_qc_multi_start_consistency(dem_and_points):
    dem, pts = dem_and_points
    report = check_dem(
        dem=dem,
        points=pts,
        multi_start=True,
    )

    assert report.multi_start_spread is not None
    assert report.multi_start_spread["max_spread_px"] < 0.2
