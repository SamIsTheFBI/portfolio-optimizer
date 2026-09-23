from contextlib import asynccontextmanager
from typing import AsyncIterator

import numpy as np
from fastapi import FastAPI, HTTPException

from .data import DataStore
from .schemas import (
    AllocationChange,
    FactorBetas,
    FactorBetasResponse,
    OptimizeRequest,
    OptimizeResponse,
)
from .strategies import (
    compute_factor_betas,
    equal_weights,
    maximize_sharpe_ratio,
    minimize_drawdown,
    minimize_volatility,
    optimize_factor_exposure,
    risk_parity,
)

store: DataStore | None = None

STRATEGY_MAP = {
    "equal_weights": equal_weights,
    "risk_parity": risk_parity,
    "minimize_volatility": minimize_volatility,
    "maximize_sharpe_ratio": maximize_sharpe_ratio,
    "minimize_drawdown": minimize_drawdown,
    "optimize_factor_exposure": optimize_factor_exposure,
}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global store
    store = DataStore()
    yield


app = FastAPI(title="Portfolio Optimizer API", lifespan=lifespan)


@app.post("/optimize", response_model=OptimizeResponse)
def optimize(request: OptimizeRequest) -> OptimizeResponse:
    assert store is not None

    tickers = [s.ticker for s in request.securities]
    current_weights = [s.current_weight for s in request.securities]

    for ticker in tickers:
        if ticker not in store.valid_tickers:
            raise HTTPException(status_code=400, detail=f"Unknown ticker: {ticker}")

    strategy_fn = STRATEGY_MAP.get(request.strategy)
    if strategy_fn is None:
        raise HTTPException(
            status_code=400,
            detail=f"Strategy '{request.strategy}' is not yet implemented",
        )

    returns_matrix = store.get_returns_matrix(tickers)
    if returns_matrix.empty:
        raise HTTPException(status_code=400, detail="No overlapping return data for selected tickers")

    try:
        kwargs: dict = dict(
            tickers=tickers,
            returns_matrix=returns_matrix,
            store=store,
            constraints=request.constraints.model_dump() if request.constraints else None,
        )
        if request.strategy == "optimize_factor_exposure":
            if not request.factor_target:
                raise HTTPException(
                    status_code=400,
                    detail="factor_target is required for optimize_factor_exposure strategy",
                )
            kwargs["factor_target"] = request.factor_target

        optimized_weights = strategy_fn(**kwargs)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    allocation_changes = []
    for i, ticker in enumerate(tickers):
        allocation_changes.append(
            AllocationChange(
                ticker=ticker,
                security_name=store.get_fund_name(ticker),
                current_weight=round(current_weights[i], 2),
                optimized_weight=round(float(optimized_weights[i]), 2),
                change=round(float(optimized_weights[i]) - current_weights[i], 2),
            )
        )

    current_weights_arr = np.array(current_weights)
    optimized_weights_arr = np.array([float(w) for w in optimized_weights])

    current_betas = compute_factor_betas(current_weights_arr, tickers, returns_matrix, store)
    optimized_betas = compute_factor_betas(optimized_weights_arr, tickers, returns_matrix, store)

    factor_betas_response = FactorBetasResponse(
        current_portfolio=FactorBetas(**current_betas),
        optimized_portfolio=FactorBetas(**optimized_betas),
    )

    return OptimizeResponse(
        optimization_strategy=request.strategy,
        allocation_changes=allocation_changes,
        factor_betas=factor_betas_response,
    )
