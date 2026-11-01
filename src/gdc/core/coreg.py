"""Core point-to-surface coregistration engine."""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union
import numpy as np

from gdc.core.optimizers import OptimizerResult, run_optimizer
from gdc.core.sampling import AffineTransform, sample_grid_at_coordinates
from gdc.core.stats import (
    compute_mix_loss,
    compute_nmad,
    compute_residuals_statistics,
    compute_rmse,
    filter_outliers,
)


@dataclass
class CoregResult:
    """Coregistration result structure containing translations, statistics, and metadata."""

    dx_px: float
    dy_px: float
    dx_m: float
    dy_m: float
    dz_m: float  # Systematic vertical bias (median of dh after horizontal shift)
    loss_name: str
    loss_final: float
    optimizer_name: str
    stats_before: Dict[str, float]
    stats_after: Dict[str, float]
    optimizer_result: OptimizerResult
    dh_initial: np.ndarray = field(repr=False)
    dh_final: np.ndarray = field(repr=False)
    inlier_mask_final: np.ndarray = field(repr=False)

    @property
    def nmad_before(self) -> float:
        return self.stats_before.get("nmad", float(np.nan))

    @property
    def nmad_after(self) -> float:
        return self.stats_after.get("nmad", float(np.nan))

    @property
    def nmad_improvement_pct(self) -> float:
        b = self.nmad_before
        a = self.nmad_after
        if np.isnan(b) or np.isnan(a) or b <= 1e-12:
            return 0.0
        return float((b - a) / b * 100.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "shift_px": {
                "dx_px": round(self.dx_px, 4),
                "dy_px": round(self.dy_px, 4),
            },
            "translation_m": {
                "dx_m": round(self.dx_m, 4),
                "dy_m": round(self.dy_m, 4),
                "dz_m": round(self.dz_m, 4),
            },
            "loss": {
                "metric": self.loss_name,
                "final_value": round(self.loss_final, 6),
            },
            "optimizer": self.optimizer_result.to_dict(),
            "stats_before": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.stats_before.items()},
            "stats_after": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.stats_after.items()},
            "nmad_improvement_pct": round(self.nmad_improvement_pct, 2),
        }


def coreg_points_to_dem(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    dem_grid: np.ndarray,
    transform: Union[AffineTransform, Any],
    bounds_px: Union[float, Tuple[float, float], Sequence[Tuple[float, float]]] = (-3.0, 3.0),
    x0_px: Tuple[float, float] = (0.0, 0.0),
    loss: str = "nmad",
    optimizer: str = "lbfgsb",
    interpolation_order: int = 1,
    downsample: Optional[int] = 6000,
    random_state: Optional[int] = 42,
    weights: Optional[np.ndarray] = None,
    perc_t: Optional[float] = 99.75,
    d_lim: Optional[float] = None,
    std_t: Optional[float] = None,
    area_or_point: str = "Area",
    nodata: Optional[float] = None,
    mask: Optional[np.ndarray] = None,
    disp: bool = False,
    errorcontrol: bool = False,
    batch_size: Optional[int] = None,
    **optimizer_kwargs: Any,
) -> CoregResult:
    """Core function to find optimal translation (dx, dy, dz) between points and a DEM.

    Shifts point coordinates (x, y) rather than resampling the DEM raster grid at each iteration.

    Parameters
    ----------
    x, y : np.ndarray
        Point world coordinates (projected CRS, e.g. UTM Easting / Northing in meters).
    z : np.ndarray
        Point elevations in meters.
    dem_grid : np.ndarray
        2D raster grid array representing DEM elevation values.
    transform : AffineTransform or rasterio.Affine
        Affine transform mapping pixel indices to world coordinates.
    bounds_px : float or tuple of tuples, default (-3.0, 3.0)
        Pixel search bounds for (dx, dy).
    x0_px : tuple of float, default (0.0, 0.0)
        Initial shift guess in pixel units.
    loss : str, default 'nmad'
        Loss function to minimize: 'nmad', 'rmse', or 'mix'.
    optimizer : str, default 'lbfgsb'
        Optimization method: 'lbfgsb', 'compass', 'neldermead', 'powell'.
    interpolation_order : int, default 1
        DEM interpolation order (1=bilinear, 3=cubic).
    downsample : int, optional, default 6000
        Downsample point cloud to this count during optimization for speed.
        Final statistics are always evaluated on the complete point dataset.
    random_state : int, optional, default 42
        Seed for reproducible downsampling.
    weights : np.ndarray, optional
        Weight array for points.
    perc_t : float, optional, default 99.75
        Percentile trimming threshold.
    d_lim : float, optional
        Absolute difference threshold in meters (|dh| < d_lim).
    std_t : float, optional
        Standard deviation threshold factor (|dh - mean| < std * std_t).
    area_or_point : str, default 'Area'
        'Area' or 'Point' pixel convention.
    nodata : float, optional
        Nodata value in DEM grid.
    mask : np.ndarray, optional
        Mask array for DEM grid.
    disp : bool, default False
        Display optimization iterations.

    Returns
    -------
    CoregResult
    """
    if not isinstance(transform, AffineTransform):
        transform = AffineTransform.from_tuple_or_affine(transform)

    x_full = np.asarray(x, dtype=np.float64).ravel()
    y_full = np.asarray(y, dtype=np.float64).ravel()
    z_full = np.asarray(z, dtype=np.float64).ravel()

    n_points = len(x_full)
    if n_points == 0:
        raise ValueError("Cannot perform coregistration on empty point dataset.")
    if len(y_full) != n_points or len(z_full) != n_points:
        raise ValueError("Dimensions of x, y, and z arrays must match.")

    res_x, res_y = transform.resolution

    # Standardize bounds
    if isinstance(bounds_px, (int, float)):
        b_val = float(abs(bounds_px))
        b_tuples = ((-b_val, b_val), (-b_val, b_val))
    elif len(bounds_px) == 2 and isinstance(bounds_px[0], (int, float)):
        b_tuples = (bounds_px, bounds_px)
    else:
        b_tuples = tuple(bounds_px)

    # Downsampling for optimization phase if requested
    if downsample and n_points > downsample:
        rng = np.random.default_rng(random_state)
        sample_indices = rng.choice(n_points, size=downsample, replace=False)
        x_opt = x_full[sample_indices]
        y_opt = y_full[sample_indices]
        z_opt = z_full[sample_indices]
        weights_opt = weights[sample_indices] if weights is not None else None
    else:
        x_opt = x_full
        y_opt = y_full
        z_opt = z_full
        weights_opt = weights

    # Select loss function
    loss_norm = loss.lower().strip()
    if loss_norm == "nmad":
        loss_fn = compute_nmad
    elif loss_norm == "rmse":
        loss_fn = compute_rmse
    elif loss_norm == "mix":
        loss_fn = compute_mix_loss
    else:
        raise ValueError(f"Unknown loss '{loss}'. Supported: 'nmad', 'rmse', 'mix'.")

    # Objective function
    def objective_func(p_px: np.ndarray, seed: Optional[int] = None) -> float:
        dx_px_curr, dy_px_curr = p_px[0], p_px[1]
        shift_x = dx_px_curr * res_x
        shift_y = dy_px_curr * res_y

        if batch_size is not None and batch_size < n_points:
            rng_b = np.random.default_rng(seed if seed is not None else random_state)
            sub_idx = rng_b.choice(n_points, size=batch_size, replace=False)
            x_curr = x_full[sub_idx]
            y_curr = y_full[sub_idx]
            z_curr = z_full[sub_idx]
            w_curr = weights[sub_idx] if weights is not None else None
        else:
            x_curr = x_opt
            y_curr = y_opt
            z_curr = z_opt
            w_curr = weights_opt

        # Shift points
        x_shifted = x_curr + shift_x
        y_shifted = y_curr + shift_y

        z_dem = sample_grid_at_coordinates(
            grid=dem_grid,
            transform=transform,
            x=x_shifted,
            y=y_shifted,
            order=interpolation_order,
            nodata=nodata,
            mask=mask,
            area_or_point=area_or_point,
        )

        dh = z_curr - z_dem
        if w_curr is not None:
            dh = dh * w_curr

        filtered_dh, _ = filter_outliers(dh, perc_t=perc_t, d_lim=d_lim, std_t=std_t)
        if filtered_dh.size < 5:
            return 1e9

        val = loss_fn(filtered_dh)
        return float(val) if np.isfinite(val) else 1e9

    # Run optimizer
    opt_result = run_optimizer(
        func=objective_func,
        x0=x0_px,
        optimizer=optimizer,
        bounds=b_tuples,
        disp=disp,
        errorcontrol=errorcontrol,
        **optimizer_kwargs,
    )

    best_dx_px = float(opt_result.x[0])
    best_dy_px = float(opt_result.x[1])
    best_dx_m = best_dx_px * res_x
    best_dy_m = best_dy_px * res_y

    # Evaluate on FULL dataset at (0, 0)
    z_dem_initial = sample_grid_at_coordinates(
        grid=dem_grid,
        transform=transform,
        x=x_full,
        y=y_full,
        order=interpolation_order,
        nodata=nodata,
        mask=mask,
        area_or_point=area_or_point,
    )
    dh_initial = z_full - z_dem_initial
    stats_before = compute_residuals_statistics(
        dh_initial, perc_t=perc_t, d_lim=d_lim, std_t=std_t
    )

    # Evaluate on FULL dataset at optimal (best_dx_px, best_dy_px)
    z_dem_final = sample_grid_at_coordinates(
        grid=dem_grid,
        transform=transform,
        x=x_full + best_dx_m,
        y=y_full + best_dy_m,
        order=interpolation_order,
        nodata=nodata,
        mask=mask,
        area_or_point=area_or_point,
    )
    dh_final_raw = z_full - z_dem_final
    stats_after_intermediate = compute_residuals_statistics(
        dh_final_raw, perc_t=perc_t, d_lim=d_lim, std_t=std_t
    )

    # Vertical bias dz is median dh after horizontal alignment
    best_dz_m = float(stats_after_intermediate["median"])
    dh_final = dh_final_raw - best_dz_m
    stats_after = compute_residuals_statistics(
        dh_final, perc_t=perc_t, d_lim=d_lim, std_t=std_t
    )
    # Restore the true vertical bias metric in stats_after
    stats_after["bias_dz"] = best_dz_m

    _, inlier_mask_final = filter_outliers(dh_final, perc_t=perc_t, d_lim=d_lim, std_t=std_t)

    return CoregResult(
        dx_px=best_dx_px,
        dy_px=best_dy_px,
        dx_m=best_dx_m,
        dy_m=best_dy_m,
        dz_m=best_dz_m,
        loss_name=loss_norm,
        loss_final=float(opt_result.fun),
        optimizer_name=opt_result.optimizer_name,
        stats_before=stats_before,
        stats_after=stats_after,
        optimizer_result=opt_result,
        dh_initial=dh_initial,
        dh_final=dh_final,
        inlier_mask_final=inlier_mask_final,
    )
