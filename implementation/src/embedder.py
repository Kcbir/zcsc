"""
Lightweight sentence embeddings using all-MiniLM-L6-v2 (22.7M params, 384-dim).
Caches results to disk so embedding is a one-time cost.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from src.config import CACHE_DIR, TICKERS, TICKER_TO_IDX, N_TICKERS

_EMBED_CACHE = CACHE_DIR / "article_embeddings.npy"
_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def embed_articles(texts: list[str], force: bool = False) -> np.ndarray:
    """
    Embed a list of texts and cache to disk.
    Returns (n_texts, 384) float32 array.
    """
    if _EMBED_CACHE.exists() and not force:
        arr = np.load(_EMBED_CACHE)
        if arr.shape[0] == len(texts):
            return arr

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(_MODEL_NAME)
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,  # unit-norm for fast cosine sim
    )
    np.save(_EMBED_CACHE, embeddings)
    return embeddings


def compute_ticker_centroids(
    embeddings: np.ndarray,
    mention_map: dict[int, list[str]],
) -> dict[str, np.ndarray]:
    """
    Average embedding for each ticker (across all articles mentioning it).
    """
    accum = {t: [] for t in TICKERS}
    for idx, tickers in mention_map.items():
        if idx >= embeddings.shape[0]:
            continue
        for t in tickers:
            if t in accum:
                accum[t].append(embeddings[idx])

    centroids = {}
    for t, vecs in accum.items():
        if vecs:
            c = np.mean(vecs, axis=0)
            c = c / (np.linalg.norm(c) + 1e-9)
            centroids[t] = c
        else:
            centroids[t] = np.zeros(embeddings.shape[1], dtype=np.float32)
    return centroids


def compute_semantic_similarity_matrix(
    centroids: dict[str, np.ndarray],
) -> np.ndarray:
    """
    (N_TICKERS x N_TICKERS) cosine similarity matrix from ticker centroids.
    """
    mat = np.zeros((N_TICKERS, N_TICKERS), dtype=np.float32)
    vecs = np.stack([centroids[t] for t in TICKERS])
    mat = vecs @ vecs.T  # cosine sim (already unit-normalised)
    np.fill_diagonal(mat, 0.0)
    return mat
