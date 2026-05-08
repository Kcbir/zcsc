#!/usr/bin/env python3
"""
End-to-end data preprocessing pipeline:
  1. Load news + quant CSVs from data/
  2. Entity extraction → cross-company mention map
  3. Embed articles with MiniLM
  4. Build contagion graph
  5. Save all artefacts to cache/

Run: python load_fnspid.py
Prerequisites: data/ populated per data/README_data.md
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pickle
from src.config import set_seed, CACHE_DIR, TICKERS, TICKER_TO_IDX
from src.data_loader import (
    load_news, load_per_ticker, build_returns_matrix, build_event_sequence,
)
from src.entity_extractor import (
    build_mention_map, build_mention_matrix, build_comention_matrix,
)
from src.embedder import embed_articles, compute_ticker_centroids, compute_semantic_similarity_matrix
from src.graph_builder import (
    build_comention_adj, build_semantic_adj, build_correlation_adj,
    combine_adjacency,
)


def main():
    set_seed(42)
    print("=" * 60)
    print("STEP 1 / 5 — Loading data")
    print("=" * 60)
    news_df = load_news()
    print(f"  News articles: {len(news_df)}")
    print(f"  Date range: {news_df['Date'].min()} → {news_df['Date'].max()}")

    per_ticker = load_per_ticker()
    print(f"  Tickers with quant data: {len(per_ticker)}")
    returns_df = build_returns_matrix(per_ticker)
    print(f"  Returns matrix: {returns_df.shape}")

    print()
    print("=" * 60)
    print("STEP 2 / 5 — Entity extraction")
    print("=" * 60)
    mention_map = build_mention_map(news_df)
    mention_matrix = build_mention_matrix(mention_map, len(news_df))
    comention = build_comention_matrix(mention_matrix)

    n_cross = sum(1 for v in mention_map.values() if len(v) > 1)
    tickers_hit = set()
    for v in mention_map.values():
        tickers_hit.update(v)
    print(f"  Articles with cross-company mentions: {n_cross}/{len(news_df)}")
    print(f"  Unique tickers found via NER: {len(tickers_hit)}")
    print(f"  Tickers: {sorted(tickers_hit)}")

    mention_counts = {}
    for t in TICKERS:
        idx = TICKER_TO_IDX[t]
        mention_counts[t] = int(mention_matrix[:, idx].sum())
    print(f"  Top mentioned: {sorted(mention_counts.items(), key=lambda x: -x[1])[:10]}")

    print()
    print("=" * 60)
    print("STEP 3 / 5 — Embedding articles (MiniLM-L6-v2)")
    print("=" * 60)
    texts = news_df["text"].tolist()
    embeddings = embed_articles(texts)
    print(f"  Embeddings shape: {embeddings.shape}")

    centroids = compute_ticker_centroids(embeddings, mention_map)
    sem_sim = compute_semantic_similarity_matrix(centroids)
    n_nonzero_sem = (sem_sim > 0.35).sum()
    print(f"  Semantic similarity edges (>0.35): {n_nonzero_sem}")

    print()
    print("=" * 60)
    print("STEP 4 / 5 — Building contagion graph")
    print("=" * 60)
    cm_adj = build_comention_adj(comention)
    sem_adj = build_semantic_adj(sem_sim)
    corr_adj = build_correlation_adj(returns_df)
    combined_adj = combine_adjacency(cm_adj, sem_adj, corr_adj)

    n_edges = (combined_adj > 0.01).sum()
    print(f"  Co-mention edges: {(cm_adj > 0.01).sum()}")
    print(f"  Semantic edges:   {(sem_adj > 0.01).sum()}")
    print(f"  Correlation edges: {(corr_adj > 0.01).sum()}")
    print(f"  Combined edges:   {n_edges}")

    print()
    print("=" * 60)
    print("STEP 5 / 5 — Building event sequence & saving")
    print("=" * 60)
    events = build_event_sequence(news_df, mention_map)
    print(f"  Event sequence length: {len(events)}")
    print(f"  Time span: {events[0]['time']:.1f}h → {events[-1]['time']:.1f}h")

    # Save all artefacts
    np.save(CACHE_DIR / "mention_matrix.npy", mention_matrix)
    np.save(CACHE_DIR / "comention_matrix.npy", comention)
    np.save(CACHE_DIR / "semantic_similarity.npy", sem_sim)
    np.save(CACHE_DIR / "combined_adjacency.npy", combined_adj)
    np.save(CACHE_DIR / "correlation_adj.npy", corr_adj)
    np.save(CACHE_DIR / "embeddings.npy", embeddings)

    with open(CACHE_DIR / "mention_map.pkl", "wb") as f:
        pickle.dump(mention_map, f)
    with open(CACHE_DIR / "mention_counts.pkl", "wb") as f:
        pickle.dump(mention_counts, f)
    with open(CACHE_DIR / "events.pkl", "wb") as f:
        pickle.dump(events, f)

    returns_df.to_csv(CACHE_DIR / "returns_matrix.csv")

    print(f"\n  All artefacts saved to {CACHE_DIR}/")
    print("  Done.")


if __name__ == "__main__":
    main()
