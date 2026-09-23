from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parent.parent.parent / "Data.xlsx"


def load_fund_info() -> pd.DataFrame:
    df = pd.read_excel(DATA_PATH, sheet_name="Fund Info", engine="openpyxl")
    df = df.dropna(subset=["ticker"])
    df["dividend_yield"] = df["dividend_yield"].fillna(0.0)
    return df


def load_fund_returns() -> pd.DataFrame:
    df = pd.read_excel(DATA_PATH, sheet_name="Fund Returns", engine="openpyxl")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    return df


def load_factor_returns() -> pd.DataFrame:
    df = pd.read_excel(DATA_PATH, sheet_name="Factor Returns", engine="openpyxl")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["index_ticker", "date"]).reset_index(drop=True)
    return df


class DataStore:
    def __init__(self) -> None:
        self.fund_info = load_fund_info()
        self.fund_returns = load_fund_returns()
        self.factor_returns = load_factor_returns()

    def get_fund_name(self, ticker: str) -> str:
        row = self.fund_info[self.fund_info["ticker"] == ticker]
        if row.empty:
            raise KeyError(f"Unknown ticker: {ticker}")
        return row.iloc[0]["fund_name"]

    def get_dividend_yield(self, ticker: str) -> float:
        row = self.fund_info[self.fund_info["ticker"] == ticker]
        if row.empty:
            raise KeyError(f"Unknown ticker: {ticker}")
        return float(row.iloc[0]["dividend_yield"])

    def get_returns_matrix(self, tickers: list[str]) -> pd.DataFrame:
        df = self.fund_returns[self.fund_returns["ticker"].isin(tickers)]
        pivot = df.pivot(index="date", columns="ticker", values="total_return")
        pivot = pivot[tickers]
        pivot = pivot.dropna()
        return pivot

    @property
    def valid_tickers(self) -> list[str]:
        return self.fund_info["ticker"].tolist()
