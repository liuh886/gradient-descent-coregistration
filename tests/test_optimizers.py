import numpy as np
import pytest

from gdc.core.optimizers import minimize_compass, minimize_scipy, run_optimizer


def test_compass_search_quadratic():
    # True minimum at (1.2, -0.7)
    target = np.array([1.2, -0.7])

    def bowl(x):
        return np.sum((x - target) ** 2)

    res = minimize_compass(
        func=bowl,
        x0=[0.0, 0.0],
        bounds=[(-3.0, 3.0), (-3.0, 3.0)],
        deltainit=1.0,
        deltatol=0.005,
    )

    assert res.success
    assert res.optimizer_name == "compass_search"
    assert pytest.approx(target[0], abs=0.02) == res.x[0]
    assert pytest.approx(target[1], abs=0.02) == res.x[1]
    assert res.fun < 0.001


def test_compass_search_with_errorcontrol_on_noisy_function():
    """Verify errorcontrol=True suppresses stochastic noise and converges to true minimum."""
    target = np.array([1.2, -0.7])

    def noisy_bowl(x, seed=0):
        rng = np.random.default_rng(seed)
        noise = rng.normal(0.0, 0.15)
        return float(np.sum((x - target) ** 2) + noise)

    res = minimize_compass(
        func=noisy_bowl,
        x0=[0.0, 0.0],
        bounds=[(-3.0, 3.0), (-3.0, 3.0)],
        deltainit=1.0,
        deltatol=0.02,
        errorcontrol=True,
        funcNinit=8,
        paired=True,
    )

    assert res.success
    assert "errorcontrol=True" in res.optimizer_name
    assert res.details.get("errorcontrol") is True
    assert pytest.approx(target[0], abs=0.08) == res.x[0]
    assert pytest.approx(target[1], abs=0.08) == res.x[1]
    # Verify N was expanded on-demand only (evaluations kept frugal, avoiding global blowout)
    assert res.nfev < 600


def test_scipy_lbfgsb_quadratic():
    target = np.array([1.2, -0.7])

    def bowl(x):
        return np.sum((x - target) ** 2)

    res = minimize_scipy(
        func=bowl,
        x0=[0.0, 0.0],
        method="L-BFGS-B",
        bounds=[(-3.0, 3.0), (-3.0, 3.0)],
    )

    assert res.success
    assert res.optimizer_name == "scipy_l-bfgs-b"
    assert pytest.approx(target[0], abs=1e-4) == res.x[0]
    assert pytest.approx(target[1], abs=1e-4) == res.x[1]


def test_optimizer_dispatcher():
    def simple_loss(x):
        return (x[0] - 2.0) ** 2

    res_compass = run_optimizer(simple_loss, x0=[0.0], optimizer="compass", bounds=[(-5.0, 5.0)])
    assert pytest.approx(2.0, abs=0.02) == res_compass.x[0]

    res_scipy = run_optimizer(simple_loss, x0=[0.0], optimizer="lbfgsb", bounds=[(-5.0, 5.0)])
    assert pytest.approx(2.0, abs=1e-4) == res_scipy.x[0]
