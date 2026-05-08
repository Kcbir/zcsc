"""
Build the contagion graph from three real-data edge sources:
  1. Co-mention edges (entity extraction)
  2. Semantic proximity edges (embedding similarity)
  3. Price correlation edges (daily returns)
"""
import numpy as np
import networkx as nx
import pandas as pd
from src.config import (
    TICKERS, TICKER_TO_IDX, N_TICKERS, SECTOR_MAP,
    SEMANTIC_SIM_THRESHOLD,
)


def _normalise_adjacency(A: np.ndarray) -> np.ndarray:
    """Min-max normalise to [0, 1], zero diagonal."""
    A = A.copy().astype(np.float64)
    np.fill_diagonal(A, 0)
    mx = A.max()
    if mx > 0:
        A /= mx
    return A.astype(np.float32)


def build_comention_adj(comention: np.ndarray) -> np.ndarray:
    return _normalise_adjacency(comention)


def build_semantic_adj(
    sim_matrix: np.ndarray, threshold: float = SEMANTIC_SIM_THRESHOLD
) -> np.ndarray:
    A = sim_matrix.copy()
    A[A < threshold] = 0.0
    np.fill_diagonal(A, 0)
    return A


def build_correlation_adj(returns_df: pd.DataFrame) -> np.ndarray:
    """
    Absolute pairwise Pearson correlation of daily returns.
    Only tickers present in both returns_df and our universe are filled.
    """
    A = np.zeros((N_TICKERS, N_TICKERS), dtype=np.float32)
    common = [t for t in TICKERS if t in returns_df.columns]
    if len(common) < 2:
        return A
    corr = returns_df[common].corr().abs().fillna(0).values
    idx_map = {t: returns_df[common].columns.get_loc(t) for t in common}
    for i, ti in enumerate(common):
        for j, tj in enumerate(common):
            if ti == tj:
                continue
            gi, gj = TICKER_TO_IDX[ti], TICKER_TO_IDX[tj]
            A[gi, gj] = corr[idx_map[ti], idx_map[tj]]
    np.fill_diagonal(A, 0)
    return A


def combine_adjacency(
    comention: np.ndarray,
    semantic: np.ndarray,
    correlation: np.ndarray,
    weights: tuple[float, float, float] = (0.4, 0.35, 0.25),
) -> np.ndarray:
    """Weighted combination of the three normalised adjacency sources."""
    w_cm, w_sem, w_corr = weights
    cm = _normalise_adjacency(comention)
    sem = _normalise_adjacency(semantic)
    cor = _normalise_adjacency(correlation)
    combined = w_cm * cm + w_sem * sem + w_corr * cor
    np.fill_diagonal(combined, 0)
    return combined


def adjacency_to_networkx(adj: np.ndarray) -> nx.DiGraph:
    """Convert adjacency matrix to a directed NetworkX graph with attributes."""
    G = nx.DiGraph()
    for i, t in enumerate(TICKERS):
        G.add_node(t, sector=SECTOR_MAP.get(t, "Other"), idx=i)
    for i in range(N_TICKERS):
        for j in range(N_TICKERS):
            if adj[i, j] > 1e-6:
                G.add_edge(TICKERS[i], TICKERS[j], weight=float(adj[i, j]))
    return G


def get_neighbourhood(adj: np.ndarray, ticker_idx: int, top_k: int = 10) -> list[int]:
    """Return indices of the top-k most connected neighbours."""
    row = adj[ticker_idx].copy()
    row[ticker_idx] = -1
    return list(np.argsort(row)[-top_k:][::-1])
