"""Optimization routines for point-to-surface coregistration.

Includes:
- Compass Search (pattern search along coordinate directions, ideal for non-smooth/noisy objective functions)
  with optional on-demand stochastic error control (`errorcontrol=True`).
- SciPy-based optimizers (L-BFGS-B, Nelder-Mead, Powell)
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
from scipy import optimize, stats


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
            "details": self.details or {},
        }


class AveragedFunction:
    """Manages stochastic function evaluations with strictly on-demand sample expansion.

    Unlike naive error-control that raises sample size N globally across all points,
    this class expands N incrementally ONLY when a specific comparison between candidate
    and current point is inconclusive (i.e. mean difference indicates descent, but
    statistical power is insufficient to confirm significance at level alpha).

    Baseline sample size (n_init) is preserved so that unambiguous steps take minimal evaluations.
    """

    def __init__(
        self,
        func: Callable[..., float],
        n_init: int = 8,
        max_n: int = 60,
        paired: bool = True,
        random_state: int = 42,
    ):
        self.func = func
        self.n_init = max(2, int(n_init))
        self.max_n = max(self.n_init, int(max_n))
        self.paired = paired
        self.cache: Dict[Tuple[float, ...], List[float]] = {}
        self.nev = 0
        rng = np.random.default_rng(random_state)
        # Sufficient pool of shared paired seeds
        self.seeds = list(rng.integers(0, 2**31 - 1, size=max_n * 4))

    def _ensure_samples(self, pt: np.ndarray, target_n: int) -> None:
        key = tuple(np.round(pt, decimals=8))
        if key not in self.cache:
            self.cache[key] = []
        curr = self.cache[key]
        n_curr = len(curr)
        if n_curr < target_n:
            needed = target_n - n_curr
            for i in range(needed):
                seed = int(self.seeds[(n_curr + i) % len(self.seeds)])
                try:
                    val = float(self.func(pt, seed=seed))
                except TypeError:
                    try:
                        val = float(self.func(pt))
                    except Exception:
                        val = 1e12
                val = val if np.isfinite(val) else 1e12
                curr.append(val)
                self.nev += 1

    def evaluate(self, pt: np.ndarray, n_samples: Optional[int] = None) -> Tuple[float, float]:
        """Return (mean, standard_error) for pt using at least n_samples."""
        n = n_samples or self.n_init
        self._ensure_samples(pt, n)
        key = tuple(np.round(pt, decimals=8))
        vals = np.array(self.cache[key][:n], dtype=np.float64)
        mean_val = float(np.mean(vals))
        se_val = float(np.std(vals, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        return mean_val, se_val

    def compare_candidates(
        self,
        xtest: np.ndarray,
        x: np.ndarray,
        alpha: float = 0.05,
        step_n: int = 4,
    ) -> bool:
        """Test if xtest is statistically significantly smaller than x.

        Increases N incrementally ONLY for this specific candidate pair when inconclusive.
        Never inflates the global baseline sample size for future candidates.
        """
        curr_n = self.n_init

        while True:
            self._ensure_samples(xtest, curr_n)
            self._ensure_samples(x, curr_n)

            key_t = tuple(np.round(xtest, decimals=8))
            key_x = tuple(np.round(x, decimals=8))
            vt = np.array(self.cache[key_t][:curr_n], dtype=np.float64)
            vx = np.array(self.cache[key_x][:curr_n], dtype=np.float64)

            diff = vt - vx
            mean_diff = float(np.mean(diff))

            # Case 1: Clearly worse or equal -> reject immediately without extra sampling
            if mean_diff >= 0.0:
                return False

            # Case 2: Mean difference is negative (candidate looks promising)
            if self.paired and curr_n > 1:
                diff_std = float(np.std(diff, ddof=1))
                if diff_std < 1e-12:
                    return True
                t_stat = mean_diff / (diff_std / np.sqrt(curr_n))
                df = curr_n - 1
            else:
                stat_res = stats.ttest_ind(vt, vx, equal_var=False)
                t_stat = float(stat_res.statistic)
                df = 2 * curr_n - 2

            p_val = float(stats.t.cdf(t_stat, df))

            # Case 2a: Statistically significant descent confirmed!
            if p_val < alpha:
                return True

            # Case 2b: Inconclusive (mean is lower, but p-value >= alpha due to noise).
            # Incrementally add samples ONLY for this comparison:
            if curr_n + step_n <= self.max_n:
                curr_n += step_n
            else:
                # Sample budget for this pair exhausted without reaching significance
                return False


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
    errorcontrol: bool = False,
    alpha: float = 0.05,
    funcNinit: int = 8,
    max_N: int = 60,
    step_N: int = 4,
    paired: bool = True,
    **kwargs: Any,
) -> OptimizerResult:
    """Pattern search / Compass search optimizer with optional on-demand stochastic error control.

    Parameters
    ----------
    func : callable
        Objective function: f(x) -> float, or f(x, seed=int) -> float when errorcontrol=True.
    x0 : sequence of float
        Initial parameter guess.
    bounds : sequence of (min, max), optional
        Parameter bounds.
    deltainit : float, default 1.0
        Initial search step size (in pixel units).
    deltatol : float, default 0.005
        Termination step size threshold.
    feps : float, default 0.0001
        Minimum function improvement to accept a step (used in deterministic mode).
    redfactor : float, default 2.0
        Reduction factor for step size when no descent direction is found.
    maxiter : int, default 200
        Maximum iterations.
    disp : bool, default False
        Whether to print search progression.
    errorcontrol : bool, default False
        If True, enables on-demand sample-size averaging and hypothesis testing to suppress
        stochastic evaluation noise (e.g. mini-batch sampling, photon jitter, non-robust losses).
    alpha : float, default 0.05
        Significance level for descent tests.
    funcNinit : int, default 8
        Base sample count per candidate point.
    max_N : int, default 60
        Maximum sample repetitions allocated per inconclusive comparison.
    step_N : int, default 4
        Sample increment per undecided test step.
    paired : bool, default True
        If True, evaluates test candidates with identical random seeds for paired difference testing.

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

    def clip_point(pt: np.ndarray) -> np.ndarray:
        if b_arr is not None:
            return np.clip(pt, b_arr[:, 0], b_arr[:, 1])
        return pt

    # Direction vectors: +e_i, -e_i for each dimension
    directions: List[np.ndarray] = []
    for i in range(n_dim):
        d_pos = np.zeros(n_dim)
        d_pos[i] = 1.0
        d_neg = np.zeros(n_dim)
        d_neg[i] = -1.0
        directions.extend([d_pos, d_neg])

    delta = float(deltainit)
    nit = 0

    if errorcontrol:
        # On-demand stochastic evaluation mode
        funcm = AveragedFunction(
            func=func,
            n_init=funcNinit,
            max_n=max_N,
            paired=paired,
        )
        # Bonferroni correction for multi-directional search
        alpha_corr = alpha / len(directions)

        while delta >= deltatol and nit < maxiter:
            nit += 1
            improved = False
            best_candidate = x.copy()

            for d in directions:
                x_test = clip_point(x + delta * d)
                if np.allclose(x_test, x, atol=1e-12):
                    continue

                # On-demand paired hypothesis testing: N expands ONLY if borderline
                if funcm.compare_candidates(x_test, x, alpha=alpha_corr, step_n=step_N):
                    best_candidate = x_test
                    improved = True
                    break

            if improved:
                x = best_candidate
                if disp:
                    mean_f, se_f = funcm.evaluate(x)
                    print(f"[Compass-EC] iter={nit}, delta={delta:.4f}, x={x}, loss={mean_f:.6f}±{se_f:.6f}")
            else:
                # Test midpoint along directions before step halving
                mid_improved = False
                for d in directions:
                    x_test = clip_point(x + delta * d)
                    x_mid = clip_point(0.5 * (x + x_test))
                    if np.allclose(x_mid, x, atol=1e-12):
                        continue
                    if funcm.compare_candidates(x_mid, x, alpha=alpha_corr, step_n=step_N):
                        x = x_mid
                        delta = max(delta / redfactor, deltatol)
                        mid_improved = True
                        if disp:
                            print(f"[Compass-EC] Midpoint step accepted: delta={delta:.4f}, x={x}")
                        break

                if not mid_improved:
                    delta /= redfactor

        final_val, final_se = funcm.evaluate(x)
        details = {
            "errorcontrol": True,
            "funse": float(final_se),
            "paired": bool(paired),
            "n_init": int(funcNinit),
            "max_N": int(max_N),
        }
        eval_count = funcm.nev
    else:
        # Standard deterministic evaluation mode (fast path)
        cache: Dict[Tuple[float, ...], float] = {}
        eval_count = 0

        def evaluate(pt: np.ndarray) -> float:
            nonlocal eval_count
            key = tuple(np.round(pt, decimals=8))
            if key not in cache:
                eval_count += 1
                try:
                    val = float(func(pt))
                except TypeError:
                    val = float(func(pt, seed=0))
                if np.isnan(val):
                    val = 1e12
                cache[key] = val
            return cache[key]

        current_val = evaluate(x)

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

        final_val = current_val
        details = {"errorcontrol": False}

    success = delta < deltatol or nit < maxiter
    opt_label = "compass_search(errorcontrol=True)" if errorcontrol else "compass_search"

    return OptimizerResult(
        x=x,
        fun=final_val,
        optimizer_name=opt_label,
        success=success,
        nit=nit,
        nfev=eval_count,
        message=f"Terminated with final step delta={delta:.6f}",
        details=details,
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
        try:
            val = float(func(x_pt))
        except TypeError:
            val = float(func(x_pt, seed=0))
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
    errorcontrol: bool = False,
    **kwargs: Any,
) -> OptimizerResult:
    """Dispatcher for running either compass (with optional errorcontrol) or scipy optimization."""
    opt_norm = optimizer.strip().lower().replace("-", "").replace("_", "")
    if opt_norm in {"compass", "compasssearch", "pattern"}:
        return minimize_compass(
            func=func,
            x0=x0,
            bounds=bounds,
            disp=disp,
            errorcontrol=errorcontrol,
            **kwargs,
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
