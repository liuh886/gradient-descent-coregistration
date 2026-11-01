"""Point sampling and coordinate transformation on raster elevation grids."""

from typing import Any, Optional, Tuple, Union
import numpy as np
from scipy import ndimage


class AffineTransform:
    """Lightweight 2D Affine Transform (world coordinates to grid pixel indices).

    Represents:
        x = a * col + b * row + c
        y = d * col + e * row + f

    Matches rasterio / GDAL Affine(a, b, c, d, e, f).
    """

    def __init__(
        self,
        a: float,
        b: float,
        c: float,
        d: float,
        e: float,
        f: float,
    ):
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)
        self.d = float(d)
        self.e = float(e)
        self.f = float(f)

        det = self.a * self.e - self.b * self.d
        if abs(det) < 1e-15:
            raise ValueError("Affine transform determinant is zero; transform is non-invertible.")
        self.det = det

    @classmethod
    def from_tuple_or_affine(cls, transform: Any) -> "AffineTransform":
        """Construct from affine object, tuple (a,b,c,d,e,f), or list."""
        if hasattr(transform, "a") and hasattr(transform, "e"):
            return cls(
                transform.a,
                transform.b,
                transform.c,
                transform.d,
                transform.e,
                transform.f,
            )
        seq = list(transform)
        if len(seq) >= 6:
            return cls(seq[0], seq[1], seq[2], seq[3], seq[4], seq[5])
        raise ValueError(f"Cannot construct AffineTransform from {transform}")

    def xy_to_rowcol(
        self,
        x: np.ndarray,
        y: np.ndarray,
        area_or_point: str = "Area",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert world coordinates (x, y) to fractional (row, col) grid coordinates.

        Parameters
        ----------
        x, y : array-like
            World coordinates (e.g. Easting, Northing in projected CRS).
        area_or_point : str, default 'Area'
            GDAL pixel representation ('Area' or 'Point').
            'Point' centers the coordinate at integer (row, col) offsets.

        Returns
        -------
        row, col : np.ndarray
            Fractional row (i) and column (j) coordinates.
        """
        x_arr = np.asarray(x, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)

        dx = x_arr - self.c
        dy = y_arr - self.f

        col = (dx * self.e - dy * self.b) / self.det
        row = (dy * self.a - dx * self.d) / self.det

        if area_or_point.lower() == "point":
            col -= 0.5
            row -= 0.5

        return row, col

    def rowcol_to_xy(
        self,
        row: np.ndarray,
        col: np.ndarray,
        area_or_point: str = "Area",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert grid (row, col) indices back to world coordinates (x, y)."""
        r_arr = np.asarray(row, dtype=np.float64)
        c_arr = np.asarray(col, dtype=np.float64)

        if area_or_point.lower() == "point":
            c_arr = c_arr + 0.5
            r_arr = r_arr + 0.5

        x = self.a * c_arr + self.b * r_arr + self.c
        y = self.d * c_arr + self.e * r_arr + self.f
        return x, y

    @property
    def resolution(self) -> Tuple[float, float]:
        """Pixel width (abs(a)) and height (abs(e))."""
        res_x = float(np.hypot(self.a, self.b))
        res_y = float(np.hypot(self.d, self.e))
        return (res_x, res_y)


def sample_grid_at_coordinates(
    grid: np.ndarray,
    transform: Union[AffineTransform, Any],
    x: np.ndarray,
    y: np.ndarray,
    order: int = 1,
    nodata: Optional[float] = None,
    mask: Optional[np.ndarray] = None,
    area_or_point: str = "Area",
) -> np.ndarray:
    """Sample elevation values from a 2D raster grid at continuous coordinates (x, y).

    Parameters
    ----------
    grid : np.ndarray
        2D raster grid array of shape (height, width).
    transform : AffineTransform or rasterio.Affine
        Affine transformation for the grid.
    x, y : np.ndarray
        Coordinates of sample points.
    order : int, default 1
        Spline interpolation order (0=nearest, 1=bilinear, 3=cubic).
    nodata : float, optional
        Nodata value in grid.
    mask : np.ndarray of bool, optional
        Mask array where True indicates invalid cells.
    area_or_point : str, default 'Area'
        'Area' or 'Point' convention.

    Returns
    -------
    np.ndarray
        Interpolated elevation values at (x, y), with np.nan for invalid/out-of-bound points.
    """
    if not isinstance(transform, AffineTransform):
        transform = AffineTransform.from_tuple_or_affine(transform)

    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    out_shape = x_arr.shape

    row, col = transform.xy_to_rowcol(x_arr.ravel(), y_arr.ravel(), area_or_point=area_or_point)

    h, w = grid.shape[-2], grid.shape[-1]
    arr_2d = grid if grid.ndim == 2 else grid[0]

    # Check bounds
    in_bounds = (row >= 0.0) & (row <= (h - 1.0)) & (col >= 0.0) & (col <= (w - 1.0))

    data_work = arr_2d.astype(np.float64, copy=True)
    invalid_mask = np.zeros((h, w), dtype=bool)

    if nodata is not None and not np.isnan(nodata):
        invalid_mask |= np.isclose(data_work, nodata)
    if mask is not None:
        invalid_mask |= (mask if mask.ndim == 2 else mask[0])
    invalid_mask |= ~np.isfinite(data_work)

    # For points near invalid cells or out of bounds, handle cleanly
    if np.any(invalid_mask):
        # Fill invalid cells with local nearest for continuous interpolation where valid
        data_work[invalid_mask] = np.nan

    # Use map_coordinates with constant cval=nan
    sampled = ndimage.map_coordinates(
        data_work,
        [row, col],
        order=order,
        mode="constant",
        cval=np.nan,
    )

    sampled[~in_bounds] = np.nan
    return sampled.reshape(out_shape)
