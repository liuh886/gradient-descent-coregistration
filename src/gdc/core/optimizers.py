"""Optimization routines for point-to-surface coregistration.

Includes:
- Compass Search (pattern search along coordinate directions, ideal for non-smooth/noisy objective functions)
- SciPy-based optimizers (L-BFGS-B, Nelder-Mead, Powell)
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
from scipy import optimize


@dataclass
class OptimizerResult:
    """Standardized result structure for coregistration optimization."""

    x: np.ndarray
    fun: float
    optimizer_name: str
    success: bool
    nit: int
    nfev: int
    message: str
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "x": self.x.tolist(),
            "fun": float(self.fun),
            "optimizer_name": self.optimizer_name,
            "success": bool(self.success),
            "nit": int(self.nit),
            "nfev": int(self.nfev),
            "message": str(self.message),
        }


def minimize_compass(
    func: Callable[[np.ndarray], float],
    x0: Sequence[float],
    bounds: Optional[Sequence[Tuple[float, float]]] = None,
    deltainit: float = 1.0,
    deltatol: float = 0.005,
    feps: float = 0.0001,
    redfactor: float = 2.0,
    maxiter: int = 200,
    disp: bool = False,
) -> OptimizerResult:
    """Pattern search / Compass search optimizer.

    Explores positive and negative steps along each coordinate direction.
    Robust to minor discontinuities and local noise in sampled DEM residuals.

    Parameters
    ----------
    func : callable
        Objective function to minimize: f(x) -> float.
    x0 : sequence of float
        Initial parameter guess.
    bounds : sequence of (min, max), optional
        Parameter bounds.
    deltainit : float, default 1.0
        Initial search step size (in pixel units).
    deltatol : float, default 0.005
        Termination step size threshold.
    feps : float, default 0.0001
        Minimum function improvement to accept a step.
    redfactor : float, default 2.0
        Reduction factor for step size when no descent direction is found.
    maxiter : int, default 200
        Maximum iterations.
    disp : bool, default False
        Whether to print search progression.

    Returns
    -------
    OptimizerResult
    """
    x = np.array(x0, dtype=np.float64)
    n_dim = len(x)

    if bounds is not None:
        b_arr = np.asarray(bounds, dtype=np.float64)
        x = np.clip(x, b_arr[:, 0], b_arr[:, 1])
    else:
        b_arr = None

    cache: Dict[Tuple[float, ...], float] = {}
    eval_count = 0

    def evaluate(pt: np.ndarray) -> float:
        nonlocal eval_count
        key = tuple(np.round(pt, decimals=8))
        if key not in cache:
            eval_count += 1
            val = float(func(pt))
            if np.isnan(val):
                val = 1e12
            cache[key] = val
        return cache[key]

    def clip_point(pt: np.ndarray) -> np.ndarray:
        if b_arr is not None:
            return np.clip(pt, b_arr[:, 0], b_arr[:, 1])
        return pt

    current_val = evaluate(x)
    delta = float(deltainit)
    nit = 0

    # Direction vectors: +e_i, -e_i for each dimension
    directions: List[np.ndarray] = []
    for i in range(n_dim):
        d_pos = np.zeros(n_dim)
        d_pos[i] = 1.0
        d_neg = np.zeros(n_dim)
        d_neg[i] = -1.0
        directions.extend([d_pos, d_neg])

    while delta >= deltatol and nit < maxiter:
        nit += 1
        improved = False
        best_candidate = x.copy()
        best_candidate_val = current_val

        for d in directions:
            x_test = clip_point(x + delta * d)
            if np.allclose(x_test, x, atol=1e-12):
                continue
            val_test = evaluate(x_test)
            if val_test < (best_candidate_val - feps):
                best_candidate = x_test
                best_candidate_val = val_test
                improved = True

        if improved:
            x = best_candidate
            current_val = best_candidate_val
            if disp:
                print(f"[Compass] iter={nit}, delta={delta:.4f}, x={x}, loss={current_val:.6f}")
        else:
            # Check midpoint test along coordinate directions before shrinking
            mid_improved = False
            for d in directions:
                x_test = clip_point(x + delta * d)
                x_mid = clip_point(0.5 * (x + x_test))
                if np.allclose(x_mid, x, atol=1e-12):
                    continue
                val_mid = evaluate(x_mid)
                if val_mid < (current_val - feps):
                    x = x_mid
                    current_val = val_mid
                    delta = max(delta / redfactor, deltatol)
                    mid_improved = True
                    break

            if not mid_improved:
                delta /= redfactor

    success = delta < deltatol or nit < maxiter
    return OptimizerResult(
        x=x,
        fun=current_val,
        optimizer_name="compass_search",
        success=success,
        nit=nit,
        nfev=eval_count,
        message=f"Terminated with final step delta={delta:.6f}",
    )


def minimize_scipy(
    func: Callable[[np.ndarray], float],
    x0: Sequence[float],
    method: str = "L-BFGS-B",
    bounds: Optional[Sequence[Tuple[float, float]]] = None,
    disp: bool = False,
    options: Optional[Dict[str, Any]] = None,
) -> OptimizerResult:
    """SciPy optimization wrapper for coregistration.

    Default method is 'L-BFGS-B'.
    """
    opts = options.copy() if options else {}
    if disp and "disp" not in opts:
        opts["disp"] = True

    eval_count = 0

    def wrapped_func(x_pt: np.ndarray) -> float:
        nonlocal eval_count
        eval_count += 1
        val = float(func(x_pt))
        if np.isnan(val):
            return 1e12
        return val

    res = optimize.minimize(
        wrapped_func,
        x0=np.asarray(x0, dtype=np.float64),
        method=method,
        bounds=bounds,
        options=opts,
    )

    fun_val = float(res.fun)
    # Consider successful if standard success is true, or loss is already near zero
    is_success = bool(res.success or fun_val < 1e-4)

    return OptimizerResult(
        x=np.asarray(res.x, dtype=np.float64),
        fun=fun_val,
        optimizer_name=f"scipy_{method.lower()}",
        success=is_success,
        nit=int(getattr(res, "nit", 0)),
        nfev=eval_count,
        message=str(getattr(res, "message", "")),
        details={"raw_status": getattr(res, "status", None)},
    )


def run_optimizer(
    func: Callable[[np.ndarray], float],
    x0: Sequence[float],
    optimizer: str = "lbfgsb",
    bounds: Optional[Sequence[Tuple[float, float]]] = None,
    disp: bool = False,
    **kwargs: Any,
) -> OptimizerResult:
    """Dispatcher for running either compass or scipy optimization."""
    opt_norm = optimizer.strip().lower().replace("-", "").replace("_", "")
    if opt_norm in {"compass", "compasssearch", "pattern"}:
        return minimize_compass(
            func=func,
            x0=x0,
            bounds=bounds,
            disp=disp,
            **{k: v for k, v in kwargs.items() if k in {"deltainit", "deltatol", "feps", "redfactor", "maxiter"}},
        )
    elif opt_norm in {"lbfgsb", "lbfgs", "scipylbfgsb"}:
        return minimize_scipy(func=func, x0=x0, method="L-BFGS-B", bounds=bounds, disp=disp, options=kwargs)
    elif opt_norm in {"neldermead", "simplex"}:
        return minimize_scipy(func=func, x0=x0, method="Nelder-Mead", bounds=bounds, disp=disp, options=kwargs)
    elif opt_norm in {"powell"}:
        return minimize_scipy(func=func, x0=x0, method="Powell", bounds=bounds, disp=disp, options=kwargs)
    else:
        raise ValueError(
            f"Unsupported optimizer '{optimizer}'. Supported: 'compass', 'lbfgsb', 'neldermead', 'powell'."
        )
