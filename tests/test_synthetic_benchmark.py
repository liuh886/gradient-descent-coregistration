"""Synthetic displacement benchmarks as outlined in GDC PRD (§6 and §7)."""

import numpy as np
import pytest

from gdc.core.coreg import coreg_points_to_dem
from gdc.core.sampling import AffineTransform, sample_grid_at_coordinates


@pytest.fixture
def synthetic_benchmark_dem():
    """Create a realistic synthetic topography DEM matching DTM10 characteristics."""
    # 200 x 200 cells at 10m resolution (2km x 2km)
    nx, ny = 200, 200
    res = 10.0
    x0, y0 = 500000.0, 6800000.0
    trans = AffineTransform(res, 0.0, x0, 0.0, -res, y0 + ny * res)

    cols, rows = np.meshgrid(np.arange(nx), np.arange(ny))
    xs = x0 + cols * res
    ys = (y0 + ny * res) - rows * res

    # Multi-frequency realistic terrain
    grid = (
        0.08 * (xs - x0)
        - 0.05 * (ys - y0)
        + 25.0 * np.sin((xs - x0) / 250.0)
        + 18.0 * np.cos((ys - y0) / 180.0)
        + 10.0 * np.sin((xs - x0 + ys - y0) / 120.0)
        + 850.0
    )
    return grid, trans


def test_synthetic_known_shift_recovery(synthetic_benchmark_dem):
    """PRD §7: Benchmark test recovering known shift (1.2 px, -0.7 px, 0.16 m)."""
    grid, trans = synthetic_benchmark_dem
    res = 10.0

    # Ground truth values from PRD §7 synthetic DTM10 benchmark
    true_dx_px = 1.2
    true_dy_px = -0.7
    true_dz = 0.16

    true_dx_m = true_dx_px * res
    true_dy_m = true_dy_px * res

    rng = np.random.default_rng(2026)
    n_points = 2000
    # Sample points in interior region
    x_pts = rng.uniform(500300.0, 501700.0, n_points)
    y_pts = rng.uniform(6800300.0, 6801700.0, n_points)

    # Displaced elevation (points sampled from shifted position)
    z_true = sample_grid_at_coordinates(
        grid, trans, x_pts + true_dx_m, y_pts + true_dy_m, order=3
    ) + true_dz

    # Add minor realistic Gaussian observation noise (sigma = 0.05 m)
    noise = rng.normal(0, 0.05, n_points)
    z_obs = z_true + noise

    # Coregistration
    result = coreg_points_to_dem(
        x=x_pts,
        y=y_pts,
        z=z_obs,
        dem_grid=grid,
        transform=trans,
        optimizer="lbfgsb",
        bounds_px=(-3.0, 3.0),
        interpolation_order=1,
    )

    # Truth recovery assertions
    assert pytest.approx(true_dx_px, abs=0.05) == result.dx_px
    assert pytest.approx(true_dy_px, abs=0.05) == result.dy_px
    assert pytest.approx(true_dz, abs=0.04) == result.dz_m
    assert result.nmad_after < 0.10


@pytest.mark.parametrize("n_sample", [200, 500, 1000, 2000])
def test_sparse_point_accuracy_curve(synthetic_benchmark_dem, n_sample):
    """PRD §7: Test convergence and accuracy across sample sizes (200, 500, 1000, 2000)."""
    grid, trans = synthetic_benchmark_dem
    res = 10.0

    true_dx_px = 0.8
    true_dy_px = -0.6
    true_dz = 0.20

    true_dx_m = true_dx_px * res
    true_dy_m = true_dy_px * res

    rng = np.random.default_rng(42)
    x_pts = rng.uniform(500300.0, 501700.0, n_sample)
    y_pts = rng.uniform(6800300.0, 6801700.0, n_sample)

    z_true = sample_grid_at_coordinates(
        grid, trans, x_pts + true_dx_m, y_pts + true_dy_m, order=3
    ) + true_dz

    result = coreg_points_to_dem(
        x=x_pts,
        y=y_pts,
        z=z_true,
        dem_grid=grid,
        transform=trans,
        optimizer="lbfgsb",
        bounds_px=(-3.0, 3.0),
        interpolation_order=1,
    )

    err_x = abs(result.dx_px - true_dx_px)
    err_y = abs(result.dy_px - true_dy_px)
    err_z = abs(result.dz_m - true_dz)

    # For >= 1000 points, error is strictly sub-pixel (< 0.05 px)
    if n_sample >= 1000:
        assert err_x < 0.05
        assert err_y < 0.05
        assert err_z < 0.03
    else:
        # For smaller sets, accuracy should still be well within 0.15 px
        assert err_x < 0.15
        assert err_y < 0.15
