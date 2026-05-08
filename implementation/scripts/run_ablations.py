#!/usr/bin/env python3
"""
Run the three structural ablations end-to-end and emit JSON with the
real precision / lift / portfolio / lead-time numbers, plus structural
diagnostics (graph density before vs. after pruning, alpha-matrix
asymmetry) that justify the bilinear and pruning components beyond
raw precision.
"""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pickle
import pandas as pd
import torch
import torch.nn.functional as F

from src.config import (
    set_seed,
    CACHE_DIR, N_TICKERS, EMBED_DIM, HIDDEN_DIM, LATENT_DIM, LEARNING_RATE,
    EDGE_PRUNE_THRESHOLD,
)
from src.neural_hawkes import NeuralHawkesProcess
from src.trainer import train
from src.evaluator import (
    compute_contagion_intensity_series,
    evaluate_contagion_detection,
    evaluate_random_baseline,
    evaluate_sector_baseline,
    evaluate_multi_threshold,
    evaluate_portfolio,
    predictive_lead_time,
)

N_EPOCHS = 50


def _norm_adj(A):
    A = A.copy().astype(np.float64)
    np.fill_diagonal(A, 0)
    mx = A.max()
    if mx > 0:
        A /= mx
    return A.astype(np.float32)


def _seed():
    set_seed(42)


def _train_model(events, embeddings, adj, components,
                 freeze_bilinear=False, prune=True):
    _seed()
    m = NeuralHawkesProcess(
        n_nodes=N_TICKERS, embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM,
    )
    if freeze_bilinear:
        for p in m.attention.parameters():
            p.requires_grad = False
    train(
        m, events, embeddings, adj,
        n_epochs=N_EPOCHS, lr=LEARNING_RATE,
        prune_threshold=EDGE_PRUNE_THRESHOLD if prune else 0,
        device="cpu", verbose=False,
        adj_components=components,
    )
    return m


def _eval_block(model, events, embeddings, learned_adj, returns_df):
    res = compute_contagion_intensity_series(
        model, events, embeddings, learned_adj, device="cpu"
    )
    a_log = res["alpha_log"]
    i_log = res["intensity_log"]
    det = evaluate_contagion_detection(a_log, returns_df)
    mt = evaluate_multi_threshold(a_log, returns_df)
    port = evaluate_portfolio(i_log, returns_df)
    lead = predictive_lead_time(i_log, returns_df)

    # Structural diagnostics
    final_alpha = a_log[-1]["alpha"] if a_log else np.zeros((N_TICKERS, N_TICKERS))
    nonzero = final_alpha > 1e-8
    mean_degree = float(nonzero.sum(axis=1).mean())
    # Asymmetry: mean |alpha_ij - alpha_ji| / (alpha_ij + alpha_ji + eps)
    sym_pairs_num = 0.0
    sym_pairs_cnt = 0
    for i in range(N_TICKERS):
        for j in range(i + 1, N_TICKERS):
            a, b = float(final_alpha[i, j]), float(final_alpha[j, i])
            if a + b > 1e-9:
                sym_pairs_num += abs(a - b) / (a + b + 1e-12)
                sym_pairs_cnt += 1
    asymmetry = sym_pairs_num / max(sym_pairs_cnt, 1)

    return {
        "detection_90th": det,
        "multi_threshold": {str(k): v for k, v in mt.items()},
        "portfolio": port,
        "lead_time": lead,
        "structural": {
            "mean_active_outdegree": mean_degree,
            "alpha_asymmetry": asymmetry,
            "n_active_pairs": int(sym_pairs_cnt),
        },
    }


def main():
    embeddings = np.load(CACHE_DIR / "embeddings.npy")
    adj = np.load(CACHE_DIR / "combined_adjacency.npy")
    events = pickle.load(open(CACHE_DIR / "events.pkl", "rb"))
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

    out = {}

    # ── Full Model (re-train deterministically) ────────────────────────
    print("[1/4] Full Model", file=sys.stderr); t0 = time.time()
    m = _train_model(events, embeddings, adj, components)
    with torch.no_grad():
        w = F.softmax(m.adj_logits, dim=0).cpu().numpy()
    learned_adj = sum(w[k] * components[k] for k in range(3)).astype(np.float32)
    out["full_model"] = _eval_block(m, events, embeddings, learned_adj, returns_df)
    out["full_model"]["learned_adj_weights"] = {
        "co_mention": float(w[0]),
        "semantic": float(w[1]),
        "correlation": float(w[2]),
    }
    out["random_baseline_90th"] = evaluate_random_baseline(
        compute_contagion_intensity_series(
            m, events, embeddings, learned_adj
        )["alpha_log"], returns_df,
    )
    out["sector_baseline_90th"] = evaluate_sector_baseline(
        compute_contagion_intensity_series(
            m, events, embeddings, learned_adj
        )["alpha_log"], returns_df,
    )
    print(f"   done in {time.time()-t0:.0f}s", file=sys.stderr)

    # ── Ablation: w/o Graph (zero adjacency, static) ───────────────────
    print("[2/4] w/o Graph", file=sys.stderr); t0 = time.time()
    zero_adj = np.zeros_like(adj)
    _seed()
    m_g = NeuralHawkesProcess(
        n_nodes=N_TICKERS, embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM,
    )
    train(
        m_g, events, embeddings, zero_adj,
        n_epochs=N_EPOCHS, lr=LEARNING_RATE,
        prune_threshold=EDGE_PRUNE_THRESHOLD,
        device="cpu", verbose=False, adj_components=None,
    )
    out["wo_graph"] = _eval_block(m_g, events, embeddings, zero_adj, returns_df)
    print(f"   done in {time.time()-t0:.0f}s", file=sys.stderr)

    # ── Ablation: w/o Bilinear (frozen random attention) ───────────────
    print("[3/4] w/o Bilinear", file=sys.stderr); t0 = time.time()
    m_b = _train_model(events, embeddings, adj, components,
                       freeze_bilinear=True)
    with torch.no_grad():
        w_b = F.softmax(m_b.adj_logits, dim=0).cpu().numpy()
    bil_adj = sum(w_b[k] * components[k] for k in range(3)).astype(np.float32)
    out["wo_bilinear"] = _eval_block(m_b, events, embeddings, bil_adj, returns_df)
    print(f"   done in {time.time()-t0:.0f}s", file=sys.stderr)

    # ── Ablation: w/o Pruning ──────────────────────────────────────────
    print("[4/4] w/o Pruning", file=sys.stderr); t0 = time.time()
    m_p = _train_model(events, embeddings, adj, components, prune=False)
    with torch.no_grad():
        w_p = F.softmax(m_p.adj_logits, dim=0).cpu().numpy()
    prune_adj = sum(w_p[k] * components[k] for k in range(3)).astype(np.float32)
    out["wo_pruning"] = _eval_block(m_p, events, embeddings, prune_adj, returns_df)
    print(f"   done in {time.time()-t0:.0f}s", file=sys.stderr)

    print(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
