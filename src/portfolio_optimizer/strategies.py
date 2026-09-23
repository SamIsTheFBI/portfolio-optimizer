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


def _max_drawdown(returns_series: np.ndarray) -> float:
    cumulative = np.cumprod(1 + returns_series)
    peak = np.maximum.accumulate(cumulative)
    drawdown = (peak - cumulative) / peak
    return float(np.max(drawdown))


def _build_bounds_and_constraints(
    n: int,
    tickers: list[str],
    returns_matrix: pd.DataFrame,
    store: DataStore,
    constraints: dict | None,
) -> tuple[list[tuple[float, float]], list[dict]]:
    lo = 0.0
    hi = 1.0
    if constraints:
        if constraints.get("min_weight") is not None:
            lo = constraints["min_weight"] / 100.0
        if constraints.get("max_weight") is not None:
            hi = constraints["max_weight"] / 100.0

    bounds = [(lo, hi)] * n
    cons: list[dict] = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    if not constraints:
        return bounds, cons

    if constraints.get("min_dividend_yield") is not None:
        target = constraints["min_dividend_yield"] / 100.0
        yields = np.array([store.get_dividend_yield(t) for t in tickers])
        cons.append({
            "type": "ineq",
            "fun": lambda w, y=yields, t=target: float(w @ y - t),
        })

    cov = _covariance_matrix(returns_matrix)
    ann_ret = _annualized_returns(returns_matrix)

    if constraints.get("min_cagr") is not None:
        target = constraints["min_cagr"] / 100.0
        cons.append({
            "type": "ineq",
            "fun": lambda w, r=ann_ret, t=target: _portfolio_return(w, r) - t,
        })

    if constraints.get("min_volatility") is not None:
        target = constraints["min_volatility"] / 100.0
        cons.append({
            "type": "ineq",
            "fun": lambda w, c=cov, t=target: _portfolio_volatility(w, c) - t,
        })

    if constraints.get("max_volatility") is not None:
        target = constraints["max_volatility"] / 100.0
        cons.append({
            "type": "ineq",
            "fun": lambda w, c=cov, t=target: t - _portfolio_volatility(w, c),
        })

    if constraints.get("max_drawdown") is not None:
        target = constraints["max_drawdown"] / 100.0
        returns_array = returns_matrix.values
        cons.append({
            "type": "ineq",
            "fun": lambda w, ra=returns_array, t=target: t - _max_drawdown(ra @ w),
        })

    return bounds, cons


def _check_feasibility(result, constraints: dict | None) -> None:
    if not result.success and "infeasible" in result.message.lower():
        raise ValueError(
            "Optimization infeasible: the given constraints cannot all be satisfied simultaneously. "
            "Try relaxing min/max weight bounds or portfolio-level constraints."
        )


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
    bounds, cons = _build_bounds_and_constraints(n, tickers, returns_matrix, store, constraints)

    def risk_contribution_error(weights: np.ndarray) -> float:
        port_vol = _portfolio_volatility(weights, cov)
        marginal = cov @ weights
        risk_contrib = weights * marginal / port_vol
        target = port_vol / n
        return float(np.sum((risk_contrib - target) ** 2))

    w0 = np.full(n, 1.0 / n)
    result = minimize(
        risk_contribution_error,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"maxiter": 1000, "ftol": 1e-15},
    )
    _check_feasibility(result, constraints)
    return result.x * 100.0


def minimize_volatility(
    tickers: list[str],
    returns_matrix: pd.DataFrame,
    store: DataStore,
    constraints: dict | None = None,
) -> np.ndarray:
    n = len(tickers)
    cov = _covariance_matrix(returns_matrix)
    bounds, cons = _build_bounds_and_constraints(n, tickers, returns_matrix, store, constraints)

    w0 = np.full(n, 1.0 / n)
    result = minimize(
        lambda w: _portfolio_volatility(w, cov),
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"maxiter": 1000, "ftol": 1e-15},
    )
    _check_feasibility(result, constraints)
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
    bounds, cons = _build_bounds_and_constraints(n, tickers, returns_matrix, store, constraints)

    def neg_sharpe(weights: np.ndarray) -> float:
        port_ret = _portfolio_return(weights, ann_ret)
        port_vol = _portfolio_volatility(weights, cov)
        if port_vol < 1e-10:
            return 0.0
        return -port_ret / port_vol

    w0 = np.full(n, 1.0 / n)
    result = minimize(
        neg_sharpe,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"maxiter": 1000, "ftol": 1e-15},
    )
    _check_feasibility(result, constraints)
    return result.x * 100.0


def minimize_drawdown(
    tickers: list[str],
    returns_matrix: pd.DataFrame,
    store: DataStore,
    constraints: dict | None = None,
) -> np.ndarray:
    n = len(tickers)
    returns_array = returns_matrix.values
    bounds, cons = _build_bounds_and_constraints(n, tickers, returns_matrix, store, constraints)

    def objective(weights: np.ndarray) -> float:
        portfolio_returns = returns_array @ weights
        return _max_drawdown(portfolio_returns)

    w0 = np.full(n, 1.0 / n)

    best_result = None
    rng = np.random.default_rng(42)
    for i in range(10):
        if i == 0:
            x0 = w0.copy()
        else:
            x0 = rng.dirichlet(np.ones(n))
            x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])
            x0 = x0 / x0.sum()

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

    _check_feasibility(best_result, constraints)
    return best_result.x * 100.0
