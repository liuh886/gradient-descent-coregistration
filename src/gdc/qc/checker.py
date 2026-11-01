"""High-level DEM quality control interface."""

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union
import numpy as np

from gdc.adapters.geopandas_pts import PointCloudAdapter
from gdc.adapters.rasterio_dem import RasterDEMAdapter
from gdc.core.coreg import coreg_points_to_dem
from gdc.core.sampling import AffineTransform
from gdc.qc.report import QCReport


def check_dem(
    dem: Any,
    points: Any,
    z_col: Optional[str] = None,
    bounds_px: Union[float, Tuple[float, float], Sequence[Tuple[float, float]]] = (-3.0, 3.0),
    loss: str = "nmad",
    optimizer: str = "lbfgsb",
    interpolation_order: int = 1,
    multi_start: bool = False,
    downsample: Optional[int] = 6000,
    random_state: int = 42,
    perc_t: Optional[float] = 99.75,
    d_lim: Optional[float] = None,
    std_t: Optional[float] = None,
    area_or_point: str = "Area",
    disp: bool = False,
    errorcontrol: bool = False,
    batch_size: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None,
    **optimizer_kwargs: Any,
) -> QCReport:
    """Run sparse-point DEM quality control and coregistration.

    Evaluates elevation difference residuals, finds optimal horizontal and vertical translations
    (dx, dy, dz), and outputs a structured Quality Control Report with tier grading
    ('reliable', 'cautious', 'unreliable').

    Does NOT modify the input DEM or point datasets in-place.

    Parameters
    ----------
    dem : str, Path, rasterio.DatasetReader, or RasterDEMAdapter
        Input DEM raster.
    points : str, Path, gpd.GeoDataFrame, pd.DataFrame, tuple of (x,y,z), or PointCloudAdapter
        Reference sparse point cloud.
    z_col : str, optional
        Elevation column name in points dataset (e.g. 'h_te_best_fit', 'z').
    bounds_px : float or tuple of tuples, default (-3.0, 3.0)
        Pixel search range for horizontal offset.
    loss : str, default 'nmad'
        Loss function: 'nmad', 'rmse', or 'mix'.
    optimizer : str, default 'lbfgsb'
        Optimizer algorithm: 'lbfgsb', 'compass', 'neldermead', 'powell'.
    interpolation_order : int, default 1
        Spline order for DEM sampling (1=bilinear, 3=cubic).
    multi_start : bool, default False
        Whether to perform multi-start consistency checks from perturbed offsets.
    downsample : int, optional, default 6000
        Number of points to sample during optimization loop (full set used for final evaluation).
    random_state : int, default 42
        Reproducibility seed.
    perc_t : float, default 99.75
        Percentile trimming threshold.
    d_lim : float, optional
        Elevation threshold in meters (|dh| < d_lim).
    std_t : float, optional
        Std threshold factor.
    area_or_point : str, default 'Area'
        GDAL pixel interpretation.
    disp : bool, default False
        Print iteration output.
    metadata : dict, optional
        User-defined metadata to include in the QC report.

    Returns
    -------
    QCReport
    """
    # 1. Adapt DEM
    if isinstance(dem, RasterDEMAdapter):
        dem_adapter = dem
    elif hasattr(dem, "grid") and hasattr(dem, "transform"):
        dem_adapter = dem
    elif isinstance(dem, (tuple, list)) and len(dem) == 2 and isinstance(dem[1], AffineTransform):
        dem_adapter = RasterDEMAdapter(
            grid=dem[0],
            transform=dem[1],
            crs=None,
            nodata=None,
            bounds=(0, 0, 0, 0),
            resolution=dem[1].resolution,
            source_info="CustomGrid",
        )
    else:
        dem_adapter = RasterDEMAdapter.from_source(dem)

    # 2. Adapt Points
    if isinstance(points, PointCloudAdapter):
        pt_adapter = points
    else:
        pt_adapter = PointCloudAdapter.from_source(
            points_source=points,
            z_col=z_col,
            target_crs=dem_adapter.crs,
        )

    # Normalize bounds format
    if isinstance(bounds_px, (int, float)):
        b_val = float(abs(bounds_px))
        b_tuples = ((-b_val, b_val), (-b_val, b_val))
    elif len(bounds_px) == 2 and isinstance(bounds_px[0], (int, float)):
        b_tuples = (bounds_px, bounds_px)
    else:
        b_tuples = tuple(bounds_px)

    # 3. Primary coregistration run
    coreg_res = coreg_points_to_dem(
        x=pt_adapter.x,
        y=pt_adapter.y,
        z=pt_adapter.z,
        dem_grid=dem_adapter.grid,
        transform=dem_adapter.transform,
        bounds_px=b_tuples,
        x0_px=(0.0, 0.0),
        loss=loss,
        optimizer=optimizer,
        interpolation_order=interpolation_order,
        downsample=downsample,
        random_state=random_state,
        perc_t=perc_t,
        d_lim=d_lim,
        std_t=std_t,
        area_or_point=area_or_point,
        nodata=dem_adapter.nodata,
        disp=disp,
        errorcontrol=errorcontrol,
        batch_size=batch_size,
        **optimizer_kwargs,
    )

    # 4. Optional Multi-start consistency check
    multi_start_spread = None
    if multi_start:
        offsets = [(0.0, 0.0), (+0.5, +0.5), (-0.5, +0.5), (+0.5, -0.5), (-0.5, -0.5)]
        xs = []
        ys = []
        for x0_test in offsets:
            r_test = coreg_points_to_dem(
                x=pt_adapter.x,
                y=pt_adapter.y,
                z=pt_adapter.z,
                dem_grid=dem_adapter.grid,
                transform=dem_adapter.transform,
                bounds_px=b_tuples,
                x0_px=x0_test,
                loss=loss,
                optimizer=optimizer,
                interpolation_order=interpolation_order,
                downsample=min(downsample or 2000, 2000),
                random_state=random_state,
                perc_t=perc_t,
                d_lim=d_lim,
                std_t=std_t,
                area_or_point=area_or_point,
                nodata=dem_adapter.nodata,
                disp=False,
            )
            xs.append(r_test.dx_px)
            ys.append(r_test.dy_px)

        spread_x = float(np.max(xs) - np.min(xs))
        spread_y = float(np.max(ys) - np.min(ys))
        multi_start_spread = {
            "spread_dx_px": round(spread_x, 4),
            "spread_dy_px": round(spread_y, 4),
            "max_spread_px": round(max(spread_x, spread_y), 4),
            "start_runs_count": len(offsets),
        }

    # 5. Build and return QC Report
    meta = metadata.copy() if metadata else {}
    meta["dem_source"] = dem_adapter.source_info
    meta["points_source"] = pt_adapter.source_info
    meta["z_column"] = pt_adapter.z_column_name

    return QCReport.evaluate(
        coreg_result=coreg_res,
        bounds_px=b_tuples,
        points_x=pt_adapter.x,
        points_y=pt_adapter.y,
        multi_start_spread=multi_start_spread,
        metadata=meta,
    )
