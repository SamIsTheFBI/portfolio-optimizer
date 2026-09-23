from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException

from .data import DataStore
from .schemas import AllocationChange, OptimizeRequest, OptimizeResponse
from .strategies import (
    equal_weights,
    maximize_sharpe_ratio,
    minimize_drawdown,
    minimize_volatility,
    risk_parity,
)

store: DataStore | None = None

STRATEGY_MAP = {
    "equal_weights": equal_weights,
    "risk_parity": risk_parity,
    "minimize_volatility": minimize_volatility,
    "maximize_sharpe_ratio": maximize_sharpe_ratio,
    "minimize_drawdown": minimize_drawdown,
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
        optimized_weights = strategy_fn(
            tickers=tickers,
            returns_matrix=returns_matrix,
            store=store,
            constraints=request.constraints.model_dump() if request.constraints else None,
        )
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

    return OptimizeResponse(
        optimization_strategy=request.strategy,
        allocation_changes=allocation_changes,
    )
