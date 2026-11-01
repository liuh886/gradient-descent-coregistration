"""Adapter for rasterio datasets and GeoTIFF raster files."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple, Union
import numpy as np

from gdc.core.sampling import AffineTransform


@dataclass
class RasterDEMAdapter:
    """Read-only container for DEM raster data."""

    grid: np.ndarray
    transform: AffineTransform
    crs: Any
    nodata: Optional[float]
    bounds: Tuple[float, float, float, float]
    resolution: Tuple[float, float]
    source_info: str

    @classmethod
    def from_source(cls, dem_source: Any) -> "RasterDEMAdapter":
        """Load DEM from filepath, rasterio dataset, or xarray object."""
        # Check if filepath or Path
        if isinstance(dem_source, (str, Path)):
            import rasterio

            with rasterio.open(dem_source) as src:
                arr = src.read(1)
                trans = AffineTransform.from_tuple_or_affine(src.transform)
                crs = src.crs
                nodata = src.nodata
                bounds = tuple(src.bounds)
                res = (abs(src.res[0]), abs(src.res[1]))
                return cls(
                    grid=arr,
                    transform=trans,
                    crs=crs,
                    nodata=nodata,
                    bounds=bounds,
                    resolution=res,
                    source_info=str(dem_source),
                )

        # Check if already open rasterio DatasetReader
        if hasattr(dem_source, "read") and hasattr(dem_source, "transform"):
            arr = dem_source.read(1)
            trans = AffineTransform.from_tuple_or_affine(dem_source.transform)
            crs = getattr(dem_source, "crs", None)
            nodata = getattr(dem_source, "nodata", None)
            bounds = tuple(dem_source.bounds)
            res = (abs(dem_source.res[0]), abs(dem_source.res[1]))
            return cls(
                grid=arr,
                transform=trans,
                crs=crs,
                nodata=nodata,
                bounds=bounds,
                resolution=res,
                source_info=getattr(dem_source, "name", "rasterio.DatasetReader"),
            )

        # Check if xdem.DEM or geoutils.Raster
        if hasattr(dem_source, "data") and hasattr(dem_source, "transform"):
            data_arr = dem_source.data
            grid_2d = data_arr[0] if data_arr.ndim == 3 else data_arr
            trans = AffineTransform.from_tuple_or_affine(dem_source.transform)
            crs = getattr(dem_source, "crs", None)
            nodata = getattr(dem_source, "nodata", None)
            bounds = tuple(getattr(dem_source, "bounds", (0, 0, 0, 0)))
            res = getattr(dem_source, "res", (abs(trans.a), abs(trans.e)))
            return cls(
                grid=np.asarray(grid_2d),
                transform=trans,
                crs=crs,
                nodata=nodata,
                bounds=bounds,
                resolution=(abs(res[0]), abs(res[1])),
                source_info="xdem.DEM/Raster",
            )

        raise TypeError(
            f"Unsupported DEM type: {type(dem_source)}. Expected file path, rasterio.DatasetReader, or DEM raster object."
        )
