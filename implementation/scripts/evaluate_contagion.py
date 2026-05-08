#!/usr/bin/env python3
"""
Evaluation driver for all quantitative claims in the paper.

Loads cache/hawkes_model.pt and emits Tables 2, 4, 5 numbers as JSON.

Run: python evaluate_contagion.py --holdout 0.4 > results.json
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pickle
import pandas as pd
import torch
import torch.nn.functional as F

from src.config import (
    set_seed,
    CACHE_DIR, N_TICKERS, EMBED_DIM, HIDDEN_DIM, LATENT_DIM, TICKERS,
)
from src.neural_hawkes import NeuralHawkesProcess
from src.evaluator import (
    compute_contagion_intensity_series,
    evaluate_contagion_detection,
    evaluate_random_baseline,
    evaluate_sector_baseline,
    evaluate_multi_threshold,
    evaluate_portfolio,
    predictive_lead_time,
)


def _norm_adj(A):
    A = A.copy().astype(np.float64)
    np.fill_diagonal(A, 0)
    mx = A.max()
    if mx > 0:
        A /= mx
    return A.astype(np.float32)


def main():
    set_seed(42)

    embeddings = np.load(CACHE_DIR / "embeddings.npy")
    adj = np.load(CACHE_DIR / "combined_adjacency.npy")
    with open(CACHE_DIR / "events.pkl", "rb") as f:
        events = pickle.load(f)
    returns_df = pd.read_csv(
        CACHE_DIR / "returns_matrix.csv", index_col=0, parse_dates=True
    )

    cm_raw = np.load(CACHE_DIR / "comention_matrix.npy")
    sem_raw = np.load(CACHE_DIR / "semantic_similarity.npy")
    corr_raw = np.load(CACHE_DIR / "correlation_adj.npy")
    sem_thresh = sem_raw.copy()
    sem_thresh[sem_thresh < 0.35] = 0.0
    np.fill_diagonal(sem_thresh, 0)
    components = (
        _norm_adj(cm_raw),
        _norm_adj(sem_thresh),
        _norm_adj(corr_raw),
    )

    model = NeuralHawkesProcess(
        n_nodes=N_TICKERS, embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM,
    )
    state = torch.load(CACHE_DIR / "hawkes_model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        w = F.softmax(model.adj_logits, dim=0).cpu().numpy()
    learned_adj = sum(w[k] * components[k] for k in range(3)).astype(np.float32)
    print(f"# learned adjacency weights: cm={w[0]:.3f} sem={w[1]:.3f} corr={w[2]:.3f}",
          file=sys.stderr)

    res = compute_contagion_intensity_series(
        model, events, embeddings, learned_adj, device="cpu"
    )
    alpha_log = res["alpha_log"]
    intensity_log = res["intensity_log"]

    det = evaluate_contagion_detection(alpha_log, returns_df)
    rand = evaluate_random_baseline(alpha_log, returns_df)
    sect = evaluate_sector_baseline(alpha_log, returns_df)
    mt = evaluate_multi_threshold(alpha_log, returns_df)
    port = evaluate_portfolio(intensity_log, returns_df)
    lead = predictive_lead_time(intensity_log, returns_df)

    out = {
        "n_events": len(events),
        "n_alpha_records": len(alpha_log),
        "learned_adj_weights": {
            "co_mention": float(w[0]),
            "semantic": float(w[1]),
            "correlation": float(w[2]),
        },
        "full_model": det,
        "random_baseline": rand,
        "sector_baseline_90th": sect,
        "multi_threshold": {str(k): v for k, v in mt.items()},
        "portfolio": port,
        "lead_time": lead,
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
