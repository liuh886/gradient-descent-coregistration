import numpy as np
import pytest

from gdc.core.coreg import coreg_points_to_dem
from gdc.core.sampling import AffineTransform


@pytest.fixture
def synthetic_terrain():
    """Create a synthetic undulating 10m resolution DEM."""
    # 100 x 100 pixels, 10m resolution
    nx, ny = 120, 120
    res = 10.0
    x0, y0 = 10000.0, 50000.0
    trans = AffineTransform(res, 0.0, x0, 0.0, -res, y0 + ny * res)

    # Coordinates grid
    cols, rows = np.meshgrid(np.arange(nx), np.arange(ny))
    xs = x0 + cols * res
    ys = (y0 + ny * res) - rows * res

    # Smooth undulating topography
    grid = (
        0.04 * (xs - x0)
        - 0.03 * (ys - y0)
        + 8.0 * np.sin((xs - x0) / 80.0)
        + 12.0 * np.cos((ys - y0) / 90.0)
        + 500.0
    )

    return grid, trans


def test_coreg_recovers_known_shift_lbfgsb(synthetic_terrain):
    grid, trans = synthetic_terrain
    res = 10.0

    # True shift: dx = 1.2 px (12.0 m), dy = -0.7 px (-7.0 m), dz = 0.25 m
    true_dx_px = 1.2
    true_dy_px = -0.7
    true_dz = 0.25

    true_dx_m = true_dx_px * res
    true_dy_m = true_dy_px * res

    # Sample random points in the central region
    rng = np.random.default_rng(42)
    n_pts = 1200
    x_pts = rng.uniform(10200.0, 11000.0, n_pts)
    y_pts = rng.uniform(50200.0, 51000.0, n_pts)

    # Compute unshifted terrain elevation at shifted coordinates to simulate displaced points
    from gdc.core.sampling import sample_grid_at_coordinates

    z_dem_at_true_pos = sample_grid_at_coordinates(
        grid, trans, x_pts + true_dx_m, y_pts + true_dy_m, order=3
    )
    z_pts = z_dem_at_true_pos + true_dz

    # Run coregistration
    result = coreg_points_to_dem(
        x=x_pts,
        y=y_pts,
        z=z_pts,
        dem_grid=grid,
        transform=trans,
        optimizer="lbfgsb",
        bounds_px=(-3.0, 3.0),
        interpolation_order=1,
    )

    assert result.optimizer_result.success
    assert pytest.approx(true_dx_px, abs=0.08) == result.dx_px
    assert pytest.approx(true_dy_px, abs=0.08) == result.dy_px
    assert pytest.approx(true_dz, abs=0.05) == result.dz_m
    assert result.nmad_after < result.nmad_before
    assert result.nmad_after < 0.15


def test_coreg_recovers_known_shift_compass(synthetic_terrain):
    grid, trans = synthetic_terrain
    res = 10.0

    true_dx_px = 1.0
    true_dy_px = -0.5
    true_dz = 0.30

    true_dx_m = true_dx_px * res
    true_dy_m = true_dy_px * res

    rng = np.random.default_rng(42)
    n_pts = 800
    x_pts = rng.uniform(10200.0, 11000.0, n_pts)
    y_pts = rng.uniform(50200.0, 51000.0, n_pts)

    from gdc.core.sampling import sample_grid_at_coordinates

    z_dem_at_true_pos = sample_grid_at_coordinates(
        grid, trans, x_pts + true_dx_m, y_pts + true_dy_m, order=3
    )
    z_pts = z_dem_at_true_pos + true_dz

    result = coreg_points_to_dem(
        x=x_pts,
        y=y_pts,
        z=z_pts,
        dem_grid=grid,
        transform=trans,
        optimizer="compass",
        bounds_px=(-3.0, 3.0),
        deltainit=0.8,
        deltatol=0.01,
        interpolation_order=1,
    )

    assert pytest.approx(true_dx_px, abs=0.15) == result.dx_px
    assert pytest.approx(true_dy_px, abs=0.15) == result.dy_px
    assert pytest.approx(true_dz, abs=0.10) == result.dz_m
