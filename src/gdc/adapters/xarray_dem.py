"""Secondary adapter for xarray.DataArray elevation models."""

from dataclasses import dataclass
from typing import Any, Optional, Tuple
import numpy as np

from gdc.adapters.rasterio_dem import RasterDEMAdapter
from gdc.core.sampling import AffineTransform


def from_xarray(da: Any) -> RasterDEMAdapter:
    """Create a RasterDEMAdapter from an xarray.DataArray."""
    grid = da.values
    if grid.ndim == 3:
        grid = grid[0]

    # If rioxarray accessor is available
    if hasattr(da, "rio"):
        trans = AffineTransform.from_tuple_or_affine(da.rio.transform())
        crs = da.rio.crs
        nodata = da.rio.nodata
        bounds = tuple(da.rio.bounds())
        res = tuple(abs(r) for r in da.rio.resolution())
    else:
        # Fallback to coordinate dimension spacing
        x_coords = da.coords.get("x", da.coords.get("lon")).values
        y_coords = da.coords.get("y", da.coords.get("lat")).values
        dx = float(x_coords[1] - x_coords[0]) if len(x_coords) > 1 else 1.0
        dy = float(y_coords[1] - y_coords[0]) if len(y_coords) > 1 else -1.0
        x0 = float(x_coords[0]) - 0.5 * dx
        y0 = float(y_coords[0]) - 0.5 * dy
        trans = AffineTransform(dx, 0.0, x0, 0.0, dy, y0)
        crs = getattr(da, "attrs", {}).get("crs", None)
        nodata = getattr(da, "attrs", {}).get("nodata", None)
        bounds = (float(np.min(x_coords)), float(np.min(y_coords)), float(np.max(x_coords)), float(np.max(y_coords)))
        res = (abs(dx), abs(dy))

    return RasterDEMAdapter(
        grid=np.asarray(grid, dtype=np.float64),
        transform=trans,
        crs=crs,
        nodata=nodata,
        bounds=bounds,
        resolution=res,
        source_info="xarray.DataArray",
    )
