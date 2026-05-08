#!/usr/bin/env python3
"""
Train the Neural Hawkes model, run ablations, generate all figures.

Run: python train_hawkes.py --seed 42
Outputs: cache/hawkes_model.pt, figures/*.pdf
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pickle
import torch
import pandas as pd

from src.config import (
    set_seed,
    CACHE_DIR, FIGURES_DIR, N_TICKERS, EMBED_DIM, HIDDEN_DIM, LATENT_DIM,
    LEARNING_RATE, TICKERS, TICKER_TO_IDX, EDGE_PRUNE_THRESHOLD,
)
from src.neural_hawkes import NeuralHawkesProcess
from src.trainer import train
from src.evaluator import (
    BASELINES, compute_contagion_intensity_series,
    evaluate_contagion_detection,
    evaluate_random_baseline, evaluate_sector_baseline,
    evaluate_multi_threshold,
    evaluate_portfolio, predictive_lead_time,
)
from src.data_loader import load_per_ticker, build_sentiment_matrix
from src.visualizer import (
    plot_contagion_network, plot_propagation_cascade,
    plot_attention_heatmap, plot_latent_space,
    plot_performance_comparison, plot_ablation_study,
    plot_training_loss,
)

N_EPOCHS = 50  # converges by ~40


def _norm_adj(A):
    """Min-max normalise to [0,1] with zero diagonal."""
    A = A.copy().astype(np.float64)
    np.fill_diagonal(A, 0)
    mx = A.max()
    if mx > 0:
        A /= mx
    return A.astype(np.float32)


def load_artefacts():
    set_seed(42)

    embeddings = np.load(CACHE_DIR / "embeddings.npy")
    adj = np.load(CACHE_DIR / "combined_adjacency.npy")
    with open(CACHE_DIR / "events.pkl", "rb") as f:
        events = pickle.load(f)
    with open(CACHE_DIR / "mention_counts.pkl", "rb") as f:
        mention_counts = pickle.load(f)
    returns_df = pd.read_csv(CACHE_DIR / "returns_matrix.csv", index_col=0, parse_dates=True)

    # Individual adjacency components for learnable weighting
    cm_raw = np.load(CACHE_DIR / "comention_matrix.npy")
    sem_raw = np.load(CACHE_DIR / "semantic_similarity.npy")
    corr_raw = np.load(CACHE_DIR / "correlation_adj.npy")
    sem_thresh = sem_raw.copy()
    sem_thresh[sem_thresh < 0.35] = 0.0
    np.fill_diagonal(sem_thresh, 0)
    adj_components = (_norm_adj(cm_raw), _norm_adj(sem_thresh), _norm_adj(corr_raw))

    return embeddings, adj, events, mention_counts, returns_df, adj_components


def train_model(events, embeddings, adj, n_epochs, device="cpu", verbose=True,
                prune=True, freeze_bilinear=False, use_adj=None,
                adj_components=None):
    """Unified training function for full model and ablations."""
    model = NeuralHawkesProcess(
        n_nodes=N_TICKERS, embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM,
    )
    if freeze_bilinear:
        for p in model.attention.parameters():
            p.requires_grad = False

    a = use_adj if use_adj is not None else adj
    losses = train(
        model, events, embeddings, a,
        n_epochs=n_epochs, lr=LEARNING_RATE,
        prune_threshold=EDGE_PRUNE_THRESHOLD if prune else 0,
        device=device, verbose=verbose,
        adj_components=adj_components,
    )
    return model, losses


def full_evaluate(model, events, embeddings, adj, returns_df, sentiment_df, label=""):
    """Run all evaluation metrics including detection baselines."""
    results = compute_contagion_intensity_series(model, events, embeddings, adj)

    det = evaluate_contagion_detection(results["alpha_log"], returns_df)
    rand_det = evaluate_random_baseline(results["alpha_log"], returns_df)
    sect_det = evaluate_sector_baseline(results["alpha_log"], returns_df)
    port = evaluate_portfolio(results["intensity_log"], returns_df)
    lead = predictive_lead_time(results["intensity_log"], returns_df)

    print(f"\n  [{label}] Detection (top-3, next-day, 90th pct):")
    print(f"    Precision:  {det.get('precision', 'N/A')} | "
          f"Spearman: {det.get('spearman_corr', 'N/A')} (p={det.get('spearman_p', 'N/A')}) | "
          f"Fires: {det.get('n_fires', 0)}")
    print(f"    Random:     {rand_det.get('precision', 'N/A')} | "
          f"Sector:  {sect_det.get('precision', 'N/A')}")
    print(f"  [{label}] Portfolio: Sharpe={port.get('sharpe_ratio', 'N/A')} | "
          f"Win={port.get('win_rate', 'N/A')} | n={port.get('n_days', '?')} days")
    print(f"  [{label}] Lead: {lead.get('mean_lead_hours', 0):.0f}h mean / "
          f"{lead.get('median_lead_hours', 0):.0f}h median")

    return {"detection": det,
            "random_detection": rand_det, "sector_detection": sect_det,
            "portfolio": port, "lead": lead, "raw": results}


def main():
    device = "cpu"
    embeddings, adj, events, mention_counts, returns_df, adj_components = load_artefacts()

    per_ticker = load_per_ticker()
    sentiment_df = build_sentiment_matrix(per_ticker)

    # ── Full model ─────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  TRAINING — Full Model (Graph + Bilinear + Edge Pruning)")
    print("=" * 65)
    model, losses = train_model(events, embeddings, adj, N_EPOCHS, device,
                                adj_components=adj_components)
    plot_training_loss(losses)
    torch.save(model.state_dict(), CACHE_DIR / "hawkes_model.pt")

    # Report learned adjacency weights
    with torch.no_grad():
        w = torch.nn.functional.softmax(model.adj_logits, dim=0)
        print(f"\n  Learned adjacency weights: "
              f"co-mention={w[0]:.3f}, semantic={w[1]:.3f}, correlation={w[2]:.3f}")

    # Reconstruct learned adjacency for evaluation
    with torch.no_grad():
        w = torch.nn.functional.softmax(model.adj_logits, dim=0)
        learned_adj = sum(w[k].item() * adj_components[k] for k in range(3))

    print("\n" + "=" * 65)
    print("  EVALUATION — Full Model")
    print("=" * 65)
    full_res = full_evaluate(model, events, embeddings, learned_adj,
                             returns_df, sentiment_df, "Full")

    # ── Ablation 1: No graph ───────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  ABLATION 1 — Without Graph (Isolated Nodes)")
    print("=" * 65)
    zero_adj = np.zeros_like(adj)
    abl1_model, _ = train_model(events, embeddings, adj, N_EPOCHS, device,
                                verbose=False, use_adj=zero_adj)
    abl1_res = full_evaluate(abl1_model, events, embeddings, zero_adj,
                             returns_df, sentiment_df, "No Graph")

    # ── Ablation 2: No bilinear ────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  ABLATION 2 — Without Bilinear (Frozen Random Attention)")
    print("=" * 65)
    abl2_model, _ = train_model(events, embeddings, adj, N_EPOCHS, device,
                                verbose=False, freeze_bilinear=True,
                                adj_components=adj_components)
    with torch.no_grad():
        w2 = torch.nn.functional.softmax(abl2_model.adj_logits, dim=0)
        abl2_adj = sum(w2[k].item() * adj_components[k] for k in range(3))
    abl2_res = full_evaluate(abl2_model, events, embeddings, abl2_adj,
                             returns_df, sentiment_df, "No Bilinear")

    # ── Ablation 3: No pruning ─────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  ABLATION 3 — Without Edge Pruning")
    print("=" * 65)
    abl3_model, _ = train_model(events, embeddings, adj, N_EPOCHS, device,
                                verbose=False, prune=False,
                                adj_components=adj_components)
    with torch.no_grad():
        w3 = torch.nn.functional.softmax(abl3_model.adj_logits, dim=0)
        abl3_adj = sum(w3[k].item() * adj_components[k] for k in range(3))
    abl3_res = full_evaluate(abl3_model, events, embeddings, abl3_adj,
                             returns_df, sentiment_df, "No Pruning")

    # ── Generate all figures ───────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  GENERATING PUBLICATION FIGURES")
    print("=" * 65)

    # Fig 1: Network
    plot_contagion_network(
        learned_adj, mention_counts,
        highlight_path=["AAPL", "NVDA", "TSM"],
    )

    # Fig 2: Cascade
    plot_propagation_cascade(
        full_res["raw"]["alpha_log"],
        source_ticker="AAPL",
        target_tickers=["MSFT", "NVDA", "TSM", "GOOG", "INTC", "AMD"],
        event_label="Semiconductor Supply Chain Contagion",
    )

    # Fig 3: Heatmap
    plot_attention_heatmap(full_res["raw"]["alpha_log"])

    # Fig 4: Latent space
    plot_latent_space(full_res["raw"]["hidden_log"])

    # Fig 5: Multi-panel performance comparison
    plot_performance_comparison(
        full_res,
        {
            "w/o Graph": abl1_res,
            "w/o Bilinear": abl2_res,
            "w/o Pruning": abl3_res,
        },
    )

    # Fig 6: Ablation (contagion precision)
    plot_ablation_study(
        full_res["detection"].get("precision", 0),
        {
            "w/o Graph\n(Isolated)": abl1_res["detection"].get("precision", 0),
            "w/o Bilinear\n(Static Corr.)": abl2_res["detection"].get("precision", 0),
            "w/o Edge\nPruning": abl3_res["detection"].get("precision", 0),
        },
    )

    # ── Final summary ─────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  RESULTS SUMMARY")
    print("=" * 65)

    print("\n  ── Contagion Detection (top-3, next-day, 90th-pct, holdout) ──")
    print(f"  {'Configuration':<30} {'Precision':>10} {'Spearman':>10} {'p-value':>12}")
    print("  " + "-" * 64)
    for name, res in [("Full Model", full_res), ("No Graph", abl1_res),
                      ("No Bilinear", abl2_res), ("No Pruning", abl3_res)]:
        d = res["detection"]
        print(f"  {name:<30} {d.get('precision', 0):>10.4f} "
              f"{d.get('spearman_corr', 0):>10.4f} {d.get('spearman_p', 1.0):>12.6f}")
    r = full_res["random_detection"]
    s = full_res["sector_detection"]
    print(f"  {'Random Baseline':<30} {r.get('precision', 0):>10.4f} "
          f"{r.get('spearman_corr', 0):>10.4f} {'N/A':>12}")
    print(f"  {'Sector Baseline':<30} {s.get('precision', 0):>10.4f} "
          f"{s.get('spearman_corr', 0):>10.4f} {'N/A':>12}")

    # Multi-threshold precision lift
    print("\n  ── Precision vs Threshold (model / random → lift) ──")
    mt = evaluate_multi_threshold(full_res["raw"]["alpha_log"], returns_df)
    for pct, vals in sorted(mt.items()):
        print(f"    {int(pct*100):>3}th pct: {vals['model_precision']:.4f} / "
              f"{vals['random_precision']:.4f} → {vals['lift']:.2f}x lift")

    print(f"\n  Predictive Lead: "
          f"{full_res['lead'].get('mean_lead_hours', 0):.0f}h mean / "
          f"{full_res['lead'].get('median_lead_hours', 0):.0f}h median")

    print(f"\n  Figures saved to: {FIGURES_DIR}/")
    print("  Done.")


if __name__ == "__main__":
    main()
