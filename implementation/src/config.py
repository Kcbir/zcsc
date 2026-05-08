"""
Central configuration: paths, ticker universe, hyperparameters.
"""
import os
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data"
CACHE_DIR = ROOT / "cache"
FIGURES_DIR = ROOT / "figures"
TICKER_DIR = DATA_DIR / "July_2022_Quant_Per_Ticker"

NEWS_CSV = DATA_DIR / "july_2022_news.csv"
QUANT_CSV = DATA_DIR / "july_2022_quant_all_stocks.csv"

CACHE_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Ticker universe (canonical upper-case) derived from per-ticker quant files
# ---------------------------------------------------------------------------
TICKERS = sorted([
    "AAPL", "ABBV", "AMD", "BABA", "BRK-B", "C", "COST", "CVX", "DIS", "GE",
    "GOOG", "INTC", "KO", "MSFT", "QQQ", "T", "TSLA", "TSM", "WFC", "WMT",
    "AAL", "AMGN", "BHP", "BIDU", "BIIB", "CAT", "CMCSA", "CMG", "COP", "CRM",
    "EBAY", "GILD", "GLD", "GSK", "MRK", "MU", "NKE", "NVDA", "ORCL", "PEP",
    "PYPL", "QCOM", "TGT", "TM", "USO", "V", "XLF",
])

TICKER_TO_IDX = {t: i for i, t in enumerate(TICKERS)}
N_TICKERS = len(TICKERS)

# Sector mapping (approximate GICS-style)
SECTOR_MAP = {
    "AAPL": "Technology", "AMD": "Semiconductors", "BABA": "E-Commerce",
    "GOOG": "Technology", "INTC": "Semiconductors", "MSFT": "Technology",
    "NVDA": "Semiconductors", "ORCL": "Technology", "QCOM": "Semiconductors",
    "MU": "Semiconductors", "TSM": "Semiconductors", "BIDU": "Technology",
    "CRM": "Technology", "EBAY": "E-Commerce", "PYPL": "Fintech",
    "TSLA": "EV/Auto", "TM": "EV/Auto", "CAT": "Industrials",
    "GE": "Industrials", "NKE": "Consumer", "CMG": "Consumer",
    "COST": "Consumer", "KO": "Consumer", "PEP": "Consumer",
    "TGT": "Consumer", "WMT": "Consumer", "DIS": "Media",
    "CMCSA": "Media", "T": "Telecom",
    "ABBV": "Pharma", "AMGN": "Pharma", "BIIB": "Pharma",
    "GILD": "Pharma", "GSK": "Pharma", "MRK": "Pharma",
    "C": "Finance", "WFC": "Finance", "V": "Finance", "XLF": "Finance",
    "BRK-B": "Finance",
    "BHP": "Materials", "CVX": "Energy", "COP": "Energy",
    "GLD": "Commodities", "USO": "Commodities",
    "AAL": "Airlines", "QQQ": "ETF",
}

# ---------------------------------------------------------------------------
# Model hyper-parameters
# ---------------------------------------------------------------------------
EMBED_DIM = 384          # MiniLM-L6-v2 output dimension
HIDDEN_DIM = 64          # c-LSTM hidden state
LATENT_DIM = 16          # bilinear projection space
LEARNING_RATE = 1e-3
N_EPOCHS = 50
BATCH_SIZE = 32
EDGE_PRUNE_THRESHOLD = 0.01
SEMANTIC_SIM_THRESHOLD = 0.35
DECAY_INIT = 0.1         # initial exponential decay rate


def set_seed(seed: int = 42) -> None:
    """Set all RNG seeds for full reproducibility. Call before any model init."""
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
