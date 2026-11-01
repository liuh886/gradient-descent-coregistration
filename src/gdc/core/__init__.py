"""Core algorithmic modules for GDC."""

from gdc.core.coreg import CoregResult, coreg_points_to_dem
from gdc.core.optimizers import OptimizerResult, minimize_compass, minimize_scipy, run_optimizer
from gdc.core.sampling import AffineTransform, sample_grid_at_coordinates
from gdc.core.stats import (
    compute_mix_loss,
    compute_nmad,
    compute_residuals_statistics,
    compute_rmse,
    filter_outliers,
)

__all__ = [
    "AffineTransform",
    "CoregResult",
    "OptimizerResult",
    "compute_mix_loss",
    "compute_nmad",
    "compute_residuals_statistics",
    "compute_rmse",
    "coreg_points_to_dem",
    "filter_outliers",
    "minimize_compass",
    "minimize_scipy",
    "run_optimizer",
    "sample_grid_at_coordinates",
]
