"""Adapter for GeoPandas GeoDataFrames and tabular point datasets."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union
import numpy as np


@dataclass
class PointCloudAdapter:
    """Read-only container for 3D reference point data."""

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    crs: Any
    z_column_name: str
    point_count: int
    source_info: str

    @classmethod
    def from_source(
        cls,
        points_source: Any,
        z_col: Optional[str] = None,
        target_crs: Optional[Any] = None,
    ) -> "PointCloudAdapter":
        """Extract point coordinates and elevations from GeoDataFrame, DataFrame, file, or coordinate tuples."""
        # Check if tuple of (x, y, z)
        if isinstance(points_source, (tuple, list)) and len(points_source) == 3:
            x_arr = np.asarray(points_source[0], dtype=np.float64)
            y_arr = np.asarray(points_source[1], dtype=np.float64)
            z_arr = np.asarray(points_source[2], dtype=np.float64)
            return cls(
                x=x_arr,
                y=y_arr,
                z=z_arr,
                crs=target_crs,
                z_column_name=z_col or "z",
                point_count=len(x_arr),
                source_info="CoordinateArrays(x, y, z)",
            )

        # Handle GeoDataFrame or DataFrame
        import pandas as pd

        df = None
        source_label = "DataFrame"

        if isinstance(points_source, (str, Path)):
            p_str = str(points_source)
            if p_str.endswith(".csv"):
                df = pd.read_csv(p_str)
                source_label = p_str
            elif p_str.endswith((".parquet", ".geoparquet")):
                import geopandas as gpd
                try:
                    df = gpd.read_parquet(p_str)
                except Exception:
                    df = pd.read_parquet(p_str)
                source_label = p_str
            else:
                import geopandas as gpd
                df = gpd.read_file(p_str)
                source_label = p_str
        elif hasattr(points_source, "copy"):
            df = points_source.copy()
            source_label = type(points_source).__name__

        if df is None:
            raise TypeError(f"Cannot extract points from source of type: {type(points_source)}")

        crs = getattr(df, "crs", None)

        # Reproject if needed and both CRSs available
        if crs is not None and target_crs is not None:
            try:
                import geopandas as gpd
                if isinstance(df, gpd.GeoDataFrame) and df.crs != target_crs:
                    df = df.to_crs(target_crs)
                    crs = df.crs
            except Exception:
                pass

        # Extract X, Y
        if hasattr(df, "geometry") and df.geometry is not None:
            # GeoDataFrame with Point geometries
            valid_geom = df.geometry.notna()
            df = df[valid_geom]
            x_arr = np.asarray(df.geometry.x.values, dtype=np.float64)
            y_arr = np.asarray(df.geometry.y.values, dtype=np.float64)
        else:
            # Check for candidate X, Y columns
            col_lower = {str(c).lower(): c for c in df.columns}
            x_col = None
            y_col = None
            for cand_x in ["e", "easting", "x", "lon", "longitude"]:
                if cand_x in col_lower:
                    x_col = col_lower[cand_x]
                    break
            for cand_y in ["n", "northing", "y", "lat", "latitude"]:
                if cand_y in col_lower:
                    y_col = col_lower[cand_y]
                    break

            if x_col is None or y_col is None:
                raise ValueError(
                    f"Could not automatically detect X and Y coordinate columns in DataFrame. Columns: {list(df.columns)}"
                )
            x_arr = np.asarray(df[x_col].values, dtype=np.float64)
            y_arr = np.asarray(df[y_col].values, dtype=np.float64)

        # Detect Z column
        detected_z_col = z_col
        if detected_z_col is None:
            candidates = [
                "h_te_best_fit",
                "h_te_interp",
                "h_te_mean",
                "h_te_median",
                "z",
                "elevation",
                "height",
                "dem",
                "h",
            ]
            for cand in candidates:
                for col in df.columns:
                    if str(col).lower() == cand:
                        detected_z_col = col
                        break
                if detected_z_col is not None:
                    break

        if detected_z_col is None:
            raise ValueError(
                f"Could not identify elevation column. Please specify z_col explicitly. Columns: {list(df.columns)}"
            )

        z_arr = np.asarray(df[detected_z_col].values, dtype=np.float64)

        return cls(
            x=x_arr,
            y=y_arr,
            z=z_arr,
            crs=crs,
            z_column_name=detected_z_col,
            point_count=len(x_arr),
            source_info=source_label,
        )
