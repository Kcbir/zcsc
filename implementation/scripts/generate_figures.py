#!/usr/bin/env python3
"""
Regenerate all publication figures from cached model + artefacts.
No retraining required.
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
    TICKERS, TICKER_TO_IDX, EDGE_PRUNE_THRESHOLD,
)
from src.neural_hawkes import NeuralHawkesProcess
from src.trainer import train
from src.evaluator import (
    compute_contagion_intensity_series,
    evaluate_contagion_detection, evaluate_multi_threshold,
    evaluate_random_baseline, evaluate_sector_baseline,
    evaluate_portfolio, predictive_lead_time,
)
from src.data_loader import load_per_ticker, build_sentiment_matrix
from src.visualizer import (
    plot_contagion_network, plot_propagation_cascade, plot_cascade_overlay,
    plot_attention_heatmap, plot_latent_space,
    plot_performance_comparison, plot_ablation_study,
    plot_training_loss,
)


def load_artefacts():
    set_seed(42)
    embeddings = np.load(CACHE_DIR / "embeddings.npy")
    adj = np.load(CACHE_DIR / "combined_adjacency.npy")
    with open(CACHE_DIR / "events.pkl", "rb") as f:
        events = pickle.load(f)
    with open(CACHE_DIR / "mention_counts.pkl", "rb") as f:
        mention_counts = pickle.load(f)
    returns_df = pd.read_csv(CACHE_DIR / "returns_matrix.csv", index_col=0, parse_dates=True)
    return embeddings, adj, events, mention_counts, returns_df


def evaluate(model, events, embeddings, adj, returns_df, sentiment_df, label=""):
    results = compute_contagion_intensity_series(model, events, embeddings, adj)
    det = evaluate_contagion_detection(results["alpha_log"], returns_df)
    port = evaluate_portfolio(results["intensity_log"], returns_df)
    lead = predictive_lead_time(results["intensity_log"], returns_df)
    return {"detection": det, "portfolio": port,
            "lead": lead, "raw": results}


def train_fresh(events, embeddings, adj, n_epochs=50, prune=True,
                freeze_bilinear=False, use_adj=None):
    model = NeuralHawkesProcess(N_TICKERS, EMBED_DIM, HIDDEN_DIM, LATENT_DIM)
    if freeze_bilinear:
        for p in model.attention.parameters():
            p.requires_grad = False
    a = use_adj if use_adj is not None else adj
    train(model, events, embeddings, a, n_epochs=n_epochs,
          prune_threshold=EDGE_PRUNE_THRESHOLD if prune else 0, verbose=False)
    return model


def main():
    device = "cpu"
    embeddings, adj, events, mention_counts, returns_df = load_artefacts()
    per_ticker = load_per_ticker()
    sentiment_df = build_sentiment_matrix(per_ticker)

    print("Training full model...")
    model = train_fresh(events, embeddings, adj)
    full_res = evaluate(model, events, embeddings, adj, returns_df, sentiment_df)

    print("Training ablation 1 (no graph)...")
    zero_adj = np.zeros_like(adj)
    abl1 = train_fresh(events, embeddings, adj, use_adj=zero_adj)
    abl1_res = evaluate(abl1, events, embeddings, zero_adj, returns_df, sentiment_df)

    print("Training ablation 2 (no bilinear)...")
    abl2 = train_fresh(events, embeddings, adj, freeze_bilinear=True)
    abl2_res = evaluate(abl2, events, embeddings, adj, returns_df, sentiment_df)

    print("Training ablation 3 (no pruning)...")
    abl3 = train_fresh(events, embeddings, adj, prune=False)
    abl3_res = evaluate(abl3, events, embeddings, adj, returns_df, sentiment_df)

    print("\nGenerating figures...")

    plot_contagion_network(adj, mention_counts,
                           highlight_path=["AAPL", "NVDA", "TSM"])

    plot_propagation_cascade(
        full_res["raw"]["alpha_log"],
        source_ticker="AAPL",
        target_tickers=["MSFT", "NVDA", "TSM", "GOOG", "INTC"],
        event_label="Contagion From AAPL to Supply Chain",
    )

    plot_cascade_overlay(
        full_res["raw"]["alpha_log"],
        source_ticker="AAPL",
        target_tickers=["MSFT", "NVDA", "TSM", "GOOG", "INTC", "AMD", "QCOM"],
    )
    plot_attention_heatmap(full_res["raw"]["alpha_log"])
    plot_latent_space(full_res["raw"]["hidden_log"])

    plot_performance_comparison(
        full_res,
        {"w/o Graph": abl1_res, "w/o Bilinear": abl2_res, "w/o Pruning": abl3_res},
    )

    plot_ablation_study(
        full_res["detection"].get("precision", 0),
        {
            "w/o Graph\n(Isolated)": abl1_res["detection"].get("precision", 0),
            "w/o Bilinear\n(Static Corr.)": abl2_res["detection"].get("precision", 0),
            "w/o Edge\nPruning": abl3_res["detection"].get("precision", 0),
        },
    )

    print(f"\nAll figures saved to {FIGURES_DIR}/")

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    # Detection baselines
    rand_det = evaluate_random_baseline(full_res['raw']['alpha_log'], returns_df)
    sect_det = evaluate_sector_baseline(full_res['raw']['alpha_log'], returns_df)

    print(f"\n  Full Model:")
    print(f"    Detection Precision:  {full_res['detection']['precision']:.2%}")
    print(f"    Spearman Correlation: {full_res['detection']['spearman_corr']:.4f} "
          f"(p={full_res['detection'].get('spearman_p', 'N/A')})")
    print(f"    Random Baseline:      {rand_det['precision']:.2%}")
    print(f"    Sector Baseline:      {sect_det['precision']:.2%}")
    print(f"    Portfolio Sharpe:     {full_res['portfolio']['sharpe_ratio']:.2f}")
    print(f"    Portfolio Return:     {full_res['portfolio']['cumulative_return']:.2%}")
    print(f"    Lead Time:            {full_res['lead']['mean_lead_hours']:.0f}h mean")

    print(f"\n  Ablation Precision Drops:")
    for name, res in [("No Graph", abl1_res), ("No Bilinear", abl2_res), ("No Pruning", abl3_res)]:
        prec = res['detection']['precision']
        delta = full_res['detection']['precision'] - prec
        print(f"    {name}: {prec:.2%} (delta: {delta:+.2%})")


if __name__ == "__main__":
    main()
