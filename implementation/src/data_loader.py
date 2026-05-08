"""
Load and merge news + quantitative CSVs into clean DataFrames.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from src.config import (
    NEWS_CSV, QUANT_CSV, TICKER_DIR, TICKERS, TICKER_TO_IDX, CACHE_DIR,
)


def _normalise_ticker(raw: str) -> str:
    return raw.strip().upper()


def load_news() -> pd.DataFrame:
    """Load the news CSV, normalise tickers and dates."""
    df = pd.read_csv(NEWS_CSV)
    df["Stock_symbol"] = df["Stock_symbol"].apply(_normalise_ticker)
    df["Date"] = pd.to_datetime(df["Date"], utc=True)
    df = df.sort_values("Date").reset_index(drop=True)
    text_col = "Article"
    df["text"] = df[text_col].fillna(df["Article_title"])
    return df


def load_quant_all() -> pd.DataFrame:
    """Load the combined OHLCV CSV for all tickers."""
    df = pd.read_csv(QUANT_CSV)
    df["Ticker"] = df["Ticker"].apply(_normalise_ticker)
    df["Date"] = pd.to_datetime(df["Date"], utc=True)
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    return df


def load_per_ticker() -> dict[str, pd.DataFrame]:
    """Load per-ticker CSVs (with sentiment) into a dict keyed by ticker."""
    out: dict[str, pd.DataFrame] = {}
    for fp in sorted(TICKER_DIR.iterdir()):
        if fp.suffix != ".csv":
            continue
        ticker = fp.stem.upper()
        if ticker == "BRK-B":
            ticker = "BRK-B"
        df = pd.read_csv(fp)
        df["Date"] = pd.to_datetime(df["Date"], utc=True)
        df = df.sort_values("Date").reset_index(drop=True)
        df["Ticker"] = ticker
        df["Return"] = df["Close"].pct_change()
        out[ticker] = df
    return out


def build_returns_matrix(per_ticker: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Pivot per-ticker data into a (dates x tickers) returns matrix."""
    frames = []
    for ticker, df in per_ticker.items():
        s = df.set_index("Date")["Return"].rename(ticker)
        frames.append(s)
    mat = pd.concat(frames, axis=1).sort_index()
    return mat


def build_sentiment_matrix(per_ticker: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Pivot per-ticker data into a (dates x tickers) sentiment matrix."""
    frames = []
    for ticker, df in per_ticker.items():
        col = "Scaled_sentiment" if "Scaled_sentiment" in df.columns else "Sentiment_gpt"
        s = df.set_index("Date")[col].rename(ticker)
        frames.append(s)
    mat = pd.concat(frames, axis=1).sort_index()
    return mat


def build_event_sequence(
    news_df: pd.DataFrame,
    mention_map: dict[int, list[str]],
) -> list[dict]:
    """
    Build a chronologically sorted event sequence for the Hawkes process.
    Each event: {time, tickers, article_idx, embedding_idx}
    """
    events = []
    t0 = news_df["Date"].min()
    for idx, row in news_df.iterrows():
        tickers_hit = mention_map.get(idx, [row["Stock_symbol"]])
        ticker_indices = [
            TICKER_TO_IDX[t] for t in tickers_hit if t in TICKER_TO_IDX
        ]
        if not ticker_indices:
            continue
        dt = (row["Date"] - t0).total_seconds() / 3600.0  # hours from start
        events.append({
            "time": dt,
            "tickers": ticker_indices,
            "article_idx": idx,
        })
    events.sort(key=lambda e: e["time"])
    return events
