"""Robust statistics and loss functions for elevation difference (dh) evaluation."""

from typing import Any, Dict, Optional, Tuple, Union
import numpy as np


def compute_nmad(
    data: Union[np.ndarray, list],
    nan_policy: str = "omit",
) -> float:
    """Compute the Normalized Median Absolute Deviation (NMAD).

    NMAD = 1.4826 * median(|x - median(x)|)

    Parameters
    ----------
    data : array-like
        Input elevation differences.
    nan_policy : str, default "omit"
        How to handle NaNs: 'omit' removes NaNs before calculation.

    Returns
    -------
    float
        The NMAD value. Returns np.nan if data is empty or all NaNs.
    """
    arr = np.asarray(data)
    if isinstance(arr, np.ma.MaskedArray):
        arr = arr.compressed()
    else:
        arr = arr.ravel()

    if nan_policy == "omit":
        arr = arr[np.isfinite(arr)]

    if arr.size == 0:
        return float(np.nan)

    med = np.median(arr)
    mad = np.median(np.abs(arr - med))
    return float(1.482602218505602 * mad)


def compute_rmse(
    data: Union[np.ndarray, list],
    nan_policy: str = "omit",
) -> float:
    """Compute Root Mean Square Error (RMSE).

    Parameters
    ----------
    data : array-like
        Input differences.
    nan_policy : str, default "omit"
        How to handle NaNs: 'omit' removes NaNs before calculation.

    Returns
    -------
    float
        RMSE value.
    """
    arr = np.asarray(data)
    if isinstance(arr, np.ma.MaskedArray):
        arr = arr.compressed()
    else:
        arr = arr.ravel()

    if nan_policy == "omit":
        arr = arr[np.isfinite(arr)]

    if arr.size == 0:
        return float(np.nan)

    return float(np.sqrt(np.mean(arr**2)))


def compute_mix_loss(
    data: Union[np.ndarray, list],
    nan_policy: str = "omit",
) -> float:
    """Compute the mixed loss function: (2 * NMAD + RMSE) / 3.

    This loss balances robust outlier resistance (NMAD) with penalty
    for large deviations (RMSE).
    """
    nmad_val = compute_nmad(data, nan_policy=nan_policy)
    rmse_val = compute_rmse(data, nan_policy=nan_policy)
    if np.isnan(nmad_val) or np.isnan(rmse_val):
        return float(np.nan)
    return float((2.0 * nmad_val + rmse_val) / 3.0)


def filter_outliers(
    dh: np.ndarray,
    perc_t: Optional[float] = 99.75,
    d_lim: Optional[float] = None,
    std_t: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Filter gross outliers from elevation difference array.

    Parameters
    ----------
    dh : np.ndarray
        Array of elevation differences.
    perc_t : float, optional
        Percentile threshold (e.g. 99.75 for 0.25% - 99.75% range).
    d_lim : float, optional
        Absolute elevation threshold (|dh| < d_lim).
    std_t : float, optional
        Standard deviation threshold (|dh| < std * std_t).

    Returns
    -------
    filtered_dh : np.ndarray
        Array containing only valid inliers.
    inlier_mask : np.ndarray of bool
        Boolean mask of matching shape indicating valid inliers.
    """
    arr = np.asarray(dh)
    mask = np.isfinite(arr)

    if d_lim is not None:
        mask &= np.abs(arr) < d_lim

    if perc_t is not None and np.any(mask):
        valid_vals = arr[mask]
        # Apply percentile filtering when sample is reasonably large
        if valid_vals.size >= 20 and perc_t < 100.0:
            lower_p = max(0.0, 100.0 - perc_t)
            upper_p = min(100.0, perc_t)
            low_val = np.nanpercentile(valid_vals, lower_p)
            high_val = np.nanpercentile(valid_vals, upper_p)
            mask &= (arr >= low_val) & (arr <= high_val)

    if std_t is not None and np.any(mask):
        valid_vals = arr[mask]
        mean_val = np.nanmean(valid_vals)
        std_val = np.nanstd(valid_vals)
        mask &= np.abs(arr - mean_val) <= (std_val * std_t)

    return arr[mask], mask


def compute_residuals_statistics(
    dh: Union[np.ndarray, list],
    perc_t: Optional[float] = 99.75,
    d_lim: Optional[float] = None,
    std_t: Optional[float] = None,
) -> Dict[str, float]:
    """Compute comprehensive statistical metrics of elevation differences dh.

    Returns dict containing mean, median, std, rmse, nmad, mix, valid count, total count.
    """
    raw_arr = np.asarray(dh).ravel()
    total_count = int(raw_arr.size)
    finite_mask = np.isfinite(raw_arr)
    finite_count = int(np.sum(finite_mask))

    if finite_count == 0:
        return {
            "mean": float(np.nan),
            "median": float(np.nan),
            "std": float(np.nan),
            "rmse": float(np.nan),
            "nmad": float(np.nan),
            "mix": float(np.nan),
            "count": 0,
            "finite_count": 0,
            "total_count": total_count,
            "outlier_count": 0,
        }

    filtered_dh, inlier_mask = filter_outliers(
        raw_arr, perc_t=perc_t, d_lim=d_lim, std_t=std_t
    )
    valid_count = int(filtered_dh.size)
    outlier_count = finite_count - valid_count

    if valid_count == 0:
        return {
            "mean": float(np.nan),
            "median": float(np.nan),
            "std": float(np.nan),
            "rmse": float(np.nan),
            "nmad": float(np.nan),
            "mix": float(np.nan),
            "count": 0,
            "finite_count": finite_count,
            "total_count": total_count,
            "outlier_count": outlier_count,
        }

    mean_val = float(np.mean(filtered_dh))
    median_val = float(np.median(filtered_dh))
    std_val = float(np.std(filtered_dh))
    rmse_val = compute_rmse(filtered_dh)
    nmad_val = compute_nmad(filtered_dh)
    mix_val = (2.0 * nmad_val + rmse_val) / 3.0

    return {
        "mean": mean_val,
        "median": median_val,
        "std": std_val,
        "rmse": rmse_val,
        "nmad": nmad_val,
        "mix": mix_val,
        "count": valid_count,
        "finite_count": finite_count,
        "total_count": total_count,
        "outlier_count": outlier_count,
    }
