"""
Regex-based entity extraction: find which of the 47 tickers are
mentioned in each article's full text.  Produces a mention matrix
and a per-article ticker list.
"""
import re
import numpy as np
import pandas as pd
from src.config import TICKERS, TICKER_TO_IDX, N_TICKERS, CACHE_DIR


# ── Aliases: full company names (case-insensitive match) ──────────────────
# Only substantial company names go here; no short ambiguous words.
_COMPANY_NAMES: dict[str, str] = {
    "Apple":                 "AAPL",
    "iPhone":                "AAPL",
    "iPad":                  "AAPL",
    "MacBook":               "AAPL",
    "Microsoft":             "MSFT",
    "Windows":               "MSFT",
    "Azure":                 "MSFT",
    "Alphabet":              "GOOG",
    "Google":                "GOOG",
    "YouTube":               "GOOG",
    "Oracle":                "ORCL",
    "Salesforce":            "CRM",
    "Baidu":                 "BIDU",
    "eBay":                  "EBAY",
    "NVIDIA":                "NVDA",
    "Nvidia":                "NVDA",
    "GeForce":               "NVDA",
    "Advanced Micro Devices": "AMD",
    "Intel":                 "INTC",
    "Qualcomm":              "QCOM",
    "Snapdragon":            "QCOM",
    "Micron":                "MU",
    "Micron Technology":     "MU",
    "TSMC":                  "TSM",
    "Taiwan Semiconductor":  "TSM",
    "Tesla":                 "TSLA",
    "Toyota":                "TM",
    "Caterpillar":           "CAT",
    "General Electric":      "GE",
    "Nike":                  "NKE",
    "Chipotle":              "CMG",
    "Costco":                "COST",
    "Coca-Cola":             "KO",
    "Coca Cola":             "KO",
    "PepsiCo":               "PEP",
    "Pepsi":                 "PEP",
    "Target Corp":           "TGT",
    "Walmart":               "WMT",
    "Walt Disney":           "DIS",
    "Disney":                "DIS",
    "Comcast":               "CMCSA",
    "AT&T":                  "T",
    "AbbVie":                "ABBV",
    "Amgen":                 "AMGN",
    "Biogen":                "BIIB",
    "Gilead":                "GILD",
    "Gilead Sciences":       "GILD",
    "GlaxoSmithKline":       "GSK",
    "Merck":                 "MRK",
    "Citigroup":             "C",
    "Citibank":              "C",
    "Wells Fargo":           "WFC",
    "Visa":                  "V",
    "Berkshire Hathaway":    "BRK-B",
    "Berkshire":             "BRK-B",
    "PayPal":                "PYPL",
    "BHP Group":             "BHP",
    "BHP Billiton":          "BHP",
    "Chevron":               "CVX",
    "ConocoPhillips":        "COP",
    "Conoco":                "COP",
    "American Airlines":     "AAL",
    "Alibaba":               "BABA",
}

# Tickers that are safe to match as uppercase word-bounded symbols
# (not common English words)
_SAFE_TICKERS = {
    "AAPL", "ABBV", "AMD", "AMGN", "BABA", "BIDU", "BIIB",
    "CMCSA", "CMG", "CRM", "CVX", "EBAY", "GILD",
    "GOOG", "GOOGL", "GSK", "INTC", "MSFT", "NVDA", "ORCL",
    "PYPL", "QCOM", "TSLA", "TSM", "WFC", "WMT", "AAL",
    "BRK-B",
}


def _build_patterns() -> list[tuple[re.Pattern, str]]:
    """Build compiled regex patterns, longest-first."""
    patterns = []

    # Company names: case-insensitive, word-bounded
    for name, ticker in sorted(_COMPANY_NAMES.items(), key=lambda x: -len(x[0])):
        pat = re.compile(r"\b" + re.escape(name) + r"\b", re.IGNORECASE)
        patterns.append((pat, ticker))

    # Safe ticker symbols: exact uppercase, word-bounded
    for sym in sorted(_SAFE_TICKERS, key=lambda x: -len(x)):
        pat = re.compile(r"\b" + re.escape(sym) + r"\b")
        patterns.append((pat, sym))

    return patterns

_PATTERNS = _build_patterns()


def extract_tickers_from_text(text: str) -> set[str]:
    """Return set of canonical tickers mentioned in *text*."""
    found: set[str] = set()
    if not isinstance(text, str):
        return found
    for pat, ticker in _PATTERNS:
        if pat.search(text):
            found.add(ticker)
    return found


def build_mention_map(news_df: pd.DataFrame) -> dict[int, list[str]]:
    """
    For every article row, extract all tickers mentioned in the full text.
    Returns {row_index: [ticker, ...]}.
    """
    mention_map: dict[int, list[str]] = {}
    for idx, row in news_df.iterrows():
        text = str(row.get("text", "")) + " " + str(row.get("Article_title", ""))
        found = extract_tickers_from_text(text)
        tagged = row.get("Stock_symbol", "")
        if tagged in TICKER_TO_IDX:
            found.add(tagged)
        mention_map[idx] = sorted(found)
    return mention_map


def build_mention_matrix(
    mention_map: dict[int, list[str]], n_articles: int
) -> np.ndarray:
    """
    Sparse binary matrix (n_articles x N_TICKERS).
    M[i, j] = 1 iff article i mentions ticker j.
    """
    M = np.zeros((n_articles, N_TICKERS), dtype=np.float32)
    for idx, tickers in mention_map.items():
        for t in tickers:
            if t in TICKER_TO_IDX:
                M[idx, TICKER_TO_IDX[t]] = 1.0
    return M


def build_comention_matrix(mention_matrix: np.ndarray) -> np.ndarray:
    """
    Co-mention adjacency: C[i,j] = number of articles that mention
    both ticker i and ticker j.  Diagonal set to zero.
    """
    C = mention_matrix.T @ mention_matrix  # (N_TICKERS x N_TICKERS)
    np.fill_diagonal(C, 0)
    return C
