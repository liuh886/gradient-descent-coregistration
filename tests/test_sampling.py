import numpy as np
import pytest

from gdc.core.sampling import AffineTransform, sample_grid_at_coordinates


def test_affine_transform_roundtrip():
    # 10m grid, upper-left corner at (500000, 6000000)
    trans = AffineTransform(10.0, 0.0, 500000.0, 0.0, -10.0, 6000000.0)

    x = np.array([500050.0, 500100.0])
    y = np.array([5999950.0, 5999900.0])

    row, col = trans.xy_to_rowcol(x, y)
    x_rec, y_rec = trans.rowcol_to_xy(row, col)

    assert np.allclose(x, x_rec)
    assert np.allclose(y, y_rec)
    assert trans.resolution == (10.0, 10.0)


def test_grid_sampling_interpolation():
    # 5x5 grid with ramp elevation z = 10 * row + 2 * col
    grid = np.zeros((5, 5), dtype=np.float64)
    for r in range(5):
        for c in range(5):
            grid[r, c] = 10.0 * r + 2.0 * c

    # 1m resolution, origin at (0, 0), pixel center coordinates
    trans = AffineTransform(1.0, 0.0, 0.0, 0.0, -1.0, 5.0)

    # Point at row=1.5, col=2.5 -> z = 10*1.5 + 2*2.5 = 15 + 5 = 20
    x, y = trans.rowcol_to_xy(1.5, 2.5)
    sampled = sample_grid_at_coordinates(grid, trans, [x], [y], order=1)
    assert pytest.approx(20.0, abs=1e-5) == sampled[0]


def test_grid_sampling_out_of_bounds():
    grid = np.ones((5, 5))
    trans = AffineTransform(1.0, 0.0, 0.0, 0.0, -1.0, 5.0)

    # Coordinates far outside grid
    sampled = sample_grid_at_coordinates(grid, trans, [100.0], [-100.0], order=1)
    assert np.isnan(sampled[0])
