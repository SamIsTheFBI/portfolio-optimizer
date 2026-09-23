import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .data import DataStore

TRADING_DAYS = 252


def _covariance_matrix(returns_matrix: pd.DataFrame) -> np.ndarray:
    return returns_matrix.cov().values * TRADING_DAYS


def _annualized_returns(returns_matrix: pd.DataFrame) -> np.ndarray:
    cumulative = (1 + returns_matrix).prod()
    n_days = len(returns_matrix)
    return (cumulative ** (TRADING_DAYS / n_days) - 1).values


def _portfolio_volatility(weights: np.ndarray, cov: np.ndarray) -> float:
    return float(np.sqrt(weights @ cov @ weights))


def _portfolio_return(weights: np.ndarray, ann_returns: np.ndarray) -> float:
    return float(weights @ ann_returns)


def equal_weights(
    tickers: list[str],
    returns_matrix: pd.DataFrame,
    store: DataStore,
    constraints: dict | None = None,
) -> np.ndarray:
    n = len(tickers)
    return np.full(n, 100.0 / n)


def risk_parity(
    tickers: list[str],
    returns_matrix: pd.DataFrame,
    store: DataStore,
    constraints: dict | None = None,
) -> np.ndarray:
    n = len(tickers)
    cov = _covariance_matrix(returns_matrix)

    def risk_contribution_error(weights: np.ndarray) -> float:
        port_vol = _portfolio_volatility(weights, cov)
        marginal = cov @ weights
        risk_contrib = weights * marginal / port_vol
        target = port_vol / n
        return float(np.sum((risk_contrib - target) ** 2))

    w0 = np.full(n, 1.0 / n)
    bounds = [(0.0, 1.0)] * n
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    result = minimize(
        risk_contribution_error,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"maxiter": 1000, "ftol": 1e-15},
    )

    return result.x * 100.0


def minimize_volatility(
    tickers: list[str],
    returns_matrix: pd.DataFrame,
    store: DataStore,
    constraints: dict | None = None,
) -> np.ndarray:
    n = len(tickers)
    cov = _covariance_matrix(returns_matrix)

    w0 = np.full(n, 1.0 / n)
    bounds = [(0.0, 1.0)] * n
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    result = minimize(
        lambda w: _portfolio_volatility(w, cov),
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"maxiter": 1000, "ftol": 1e-15},
    )

    return result.x * 100.0


def maximize_sharpe_ratio(
    tickers: list[str],
    returns_matrix: pd.DataFrame,
    store: DataStore,
    constraints: dict | None = None,
) -> np.ndarray:
    n = len(tickers)
    cov = _covariance_matrix(returns_matrix)
    ann_ret = _annualized_returns(returns_matrix)

    def neg_sharpe(weights: np.ndarray) -> float:
        port_ret = _portfolio_return(weights, ann_ret)
        port_vol = _portfolio_volatility(weights, cov)
        if port_vol < 1e-10:
            return 0.0
        return -port_ret / port_vol

    w0 = np.full(n, 1.0 / n)
    bounds = [(0.0, 1.0)] * n
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    result = minimize(
        neg_sharpe,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"maxiter": 1000, "ftol": 1e-15},
    )

    return result.x * 100.0


def _max_drawdown(returns_series: np.ndarray) -> float:
    cumulative = np.cumprod(1 + returns_series)
    peak = np.maximum.accumulate(cumulative)
    drawdown = (peak - cumulative) / peak
    return float(np.max(drawdown))


def minimize_drawdown(
    tickers: list[str],
    returns_matrix: pd.DataFrame,
    store: DataStore,
    constraints: dict | None = None,
) -> np.ndarray:
    n = len(tickers)
    returns_array = returns_matrix.values

    def objective(weights: np.ndarray) -> float:
        portfolio_returns = returns_array @ weights
        return _max_drawdown(portfolio_returns)

    w0 = np.full(n, 1.0 / n)
    bounds = [(0.0, 1.0)] * n
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    best_result = None
    rng = np.random.default_rng(42)
    for i in range(10):
        if i == 0:
            x0 = w0.copy()
        else:
            x0 = rng.dirichlet(np.ones(n))

        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={"maxiter": 1000, "ftol": 1e-15},
        )

        if best_result is None or result.fun < best_result.fun:
            best_result = result

    return best_result.x * 100.0
