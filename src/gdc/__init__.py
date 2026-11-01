"""Gradient Descent Coregistration (GDC).

A point-to-surface coregistration and DEM quality control library.
"""

from gdc.adapters.geopandas_pts import PointCloudAdapter
from gdc.adapters.rasterio_dem import RasterDEMAdapter
from gdc.core.coreg import CoregResult, coreg_points_to_dem
from gdc.core.optimizers import OptimizerResult, minimize_compass, minimize_scipy
from gdc.core.sampling import AffineTransform, sample_grid_at_coordinates
from gdc.core.stats import (
    compute_mix_loss,
    compute_nmad,
    compute_residuals_statistics,
    compute_rmse,
    filter_outliers,
)
from gdc.qc.checker import check_dem
from gdc.qc.report import QCReport

__version__ = "0.1.0"

__all__ = [
    "AffineTransform",
    "CoregResult",
    "OptimizerResult",
    "PointCloudAdapter",
    "QCReport",
    "RasterDEMAdapter",
    "check_dem",
    "compute_mix_loss",
    "compute_nmad",
    "compute_residuals_statistics",
    "compute_rmse",
    "coreg_points_to_dem",
    "filter_outliers",
    "minimize_compass",
    "minimize_scipy",
    "sample_grid_at_coordinates",
]
