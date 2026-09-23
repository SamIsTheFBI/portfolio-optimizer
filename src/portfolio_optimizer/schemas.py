from pydantic import BaseModel, field_validator


class SecurityInput(BaseModel):
    ticker: str
    current_weight: float


class Constraints(BaseModel):
    min_weight: float | None = None
    max_weight: float | None = None
    min_cagr: float | None = None
    min_volatility: float | None = None
    max_volatility: float | None = None
    max_drawdown: float | None = None
    min_dividend_yield: float | None = None


VALID_STRATEGIES = [
    "equal_weights",
    "risk_parity",
    "minimize_volatility",
    "maximize_sharpe_ratio",
    "minimize_drawdown",
    "optimize_factor_exposure",
]


class OptimizeRequest(BaseModel):
    securities: list[SecurityInput]
    strategy: str
    constraints: Constraints | None = None
    factor_target: str | None = None

    @field_validator("strategy")
    @classmethod
    def validate_strategy(cls, v: str) -> str:
        if v not in VALID_STRATEGIES:
            raise ValueError(f"Unsupported strategy: {v}. Must be one of {VALID_STRATEGIES}")
        return v

    @field_validator("securities")
    @classmethod
    def validate_securities(cls, v: list[SecurityInput]) -> list[SecurityInput]:
        if len(v) < 2:
            raise ValueError("At least 2 securities required")
        return v


class AllocationChange(BaseModel):
    ticker: str
    security_name: str
    current_weight: float
    optimized_weight: float
    change: float


class FactorBetas(BaseModel):
    value: float
    momentum: float
    size: float


class FactorBetasResponse(BaseModel):
    current_portfolio: FactorBetas
    optimized_portfolio: FactorBetas


class OptimizeResponse(BaseModel):
    optimization_strategy: str
    allocation_changes: list[AllocationChange]
    factor_betas: FactorBetasResponse | None = None
