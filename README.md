# Portfolio Optimizer API

A REST API that computes optimized portfolio weights for a set of ETFs using various optimization strategies. Built with FastAPI and scipy.

## Setup

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager

### Installation

```bash
git clone <repo-url>
cd portfolio-optimizer
uv sync
```

### Running the API

```bash
uv run uvicorn portfolio_optimizer:app --reload
```

The API will be available at `http://127.0.0.1:8000`. Interactive docs at `http://127.0.0.1:8000/docs`.

## API Endpoint

### `POST /optimize`

Accepts a portfolio and returns optimized weights.

#### Request Body

```json
{
  "securities": [
    {"ticker": "SPY", "current_weight": 60.0},
    {"ticker": "AGG", "current_weight": 30.0},
    {"ticker": "GLD", "current_weight": 10.0}
  ],
  "strategy": "minimize_volatility",
  "constraints": {
    "min_weight": 5.0,
    "max_weight": 40.0,
    "min_dividend_yield": 2.50
  },
  "factor_target": "momentum"
}
```

#### Parameters

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `securities` | array | Yes | List of tickers with current weights (%) |
| `strategy` | string | Yes | Optimization strategy name |
| `constraints` | object | No | Optional weight and portfolio constraints |
| `factor_target` | string | No | Required for `optimize_factor_exposure` (`momentum`, `value`, `size`) |

#### Available Strategies

| Strategy | Description |
|----------|-------------|
| `equal_weights` | Equal allocation across all securities |
| `risk_parity` | Equal risk contribution from each security |
| `minimize_volatility` | Minimum portfolio variance |
| `maximize_sharpe_ratio` | Maximum risk-adjusted return (risk-free rate = 0%) |
| `minimize_drawdown` | Minimize worst historical peak-to-trough decline |
| `optimize_factor_exposure` | Maximize/minimize exposure to a risk factor |

#### Constraints

| Field | Description |
|-------|-------------|
| `min_weight` | Minimum weight per security (%) |
| `max_weight` | Maximum weight per security (%) |
| `min_cagr` | Minimum portfolio CAGR (%) |
| `min_volatility` | Minimum portfolio volatility (%) |
| `max_volatility` | Maximum portfolio volatility (%) |
| `max_drawdown` | Maximum portfolio drawdown (%) |
| `min_dividend_yield` | Minimum weighted dividend yield (%) |

#### Response

```json
{
  "optimization_strategy": "minimize_volatility",
  "allocation_changes": [
    {
      "ticker": "SPY",
      "security_name": "State Street SPDR S&P 500 ETF Trust",
      "current_weight": 60.0,
      "optimized_weight": 6.92,
      "change": -53.08
    }
  ],
  "factor_betas": {
    "current_portfolio": {"value": 0.16, "momentum": 0.13, "size": -0.06},
    "optimized_portfolio": {"value": 0.08, "momentum": 0.10, "size": -0.09}
  }
}
```

## Available Tickers

| Ticker | Fund Name |
|--------|-----------|
| IEFA | iShares Core MSCI EAFE ETF |
| GLD | SPDR Gold Shares |
| AGG | iShares Core US Aggregate Bond ETF |
| VEA | Vanguard Developed Markets Index Fund ETF |
| SPY | State Street SPDR S&P 500 ETF Trust |

## Example Requests

```bash
# Equal Weights
curl -X POST http://127.0.0.1:8000/optimize \
  -H "Content-Type: application/json" \
  -d '{"securities": [{"ticker": "IEFA", "current_weight": 25}, {"ticker": "SPY", "current_weight": 75}], "strategy": "equal_weights"}'

# Risk Parity
curl -X POST http://127.0.0.1:8000/optimize \
  -H "Content-Type: application/json" \
  -d '{"securities": [{"ticker": "VEA", "current_weight": 25}, {"ticker": "AGG", "current_weight": 75}], "strategy": "risk_parity"}'

# Maximize Sharpe with constraints
curl -X POST http://127.0.0.1:8000/optimize \
  -H "Content-Type: application/json" \
  -d '{"securities": [{"ticker": "IEFA", "current_weight": 20}, {"ticker": "GLD", "current_weight": 20}, {"ticker": "AGG", "current_weight": 20}, {"ticker": "VEA", "current_weight": 20}, {"ticker": "SPY", "current_weight": 20}], "strategy": "maximize_sharpe_ratio", "constraints": {"min_dividend_yield": 2.5, "min_weight": 5, "max_weight": 40}}'

# Factor Exposure (Maximize Momentum)
curl -X POST http://127.0.0.1:8000/optimize \
  -H "Content-Type: application/json" \
  -d '{"securities": [{"ticker": "IEFA", "current_weight": 20}, {"ticker": "GLD", "current_weight": 20}, {"ticker": "AGG", "current_weight": 20}, {"ticker": "VEA", "current_weight": 20}, {"ticker": "SPY", "current_weight": 20}], "strategy": "optimize_factor_exposure", "factor_target": "momentum"}'
```

## Project Structure

```
src/portfolio_optimizer/
├── __init__.py        # Package entry point
├── app.py             # FastAPI application and endpoint
├── data.py            # Excel data loading and access
├── schemas.py         # Pydantic request/response models
└── strategies.py      # Optimization strategy implementations
```

## Technical Details

- **Covariance matrix** annualized using 252 trading days
- **Annualized returns** computed via compound growth: `(∏(1+r))^(252/N) - 1`
- **Optimization** via `scipy.optimize.minimize` with SLSQP method
- **Minimize Drawdown** uses 10 random starting points to handle non-convexity
- **Factor betas** computed via OLS regression against Momentum, Value, Size factor returns
