import numpy as np
import pandas as pd

from .data import DataStore


def equal_weights(
    tickers: list[str],
    returns_matrix: pd.DataFrame,
    store: DataStore,
    constraints: dict | None = None,
) -> np.ndarray:
    n = len(tickers)
    return np.full(n, 100.0 / n)
