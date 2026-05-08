"""
Publication-quality figures for ACM SIGMOD FinDS.

All figures: 300 DPI, grayscale-friendly, LaTeX-compatible labels.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import networkx as nx
from pathlib import Path
from src.config import TICKERS, TICKER_TO_IDX, N_TICKERS, SECTOR_MAP, FIGURES_DIR

# ── Style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

SECTOR_COLORS = {
    "Technology":     "#2171b5",
    "Semiconductors": "#6a3d9a",
    "E-Commerce":     "#e6550d",
    "EV/Auto":        "#e31a1c",
    "Industrials":    "#636363",
    "Consumer":       "#31a354",
    "Media":          "#fd8d3c",
    "Telecom":        "#756bb1",
    "Pharma":         "#de2d26",
    "Finance":        "#3182bd",
    "Materials":      "#8c6d31",
    "Energy":         "#843c39",
    "Commodities":    "#d6616b",
    "Airlines":       "#7b4173",
    "Fintech":        "#17becf",
    "ETF":            "#bcbd22",
    "Other":          "#999999",
}


def _sector_color(ticker: str) -> str:
    return SECTOR_COLORS.get(SECTOR_MAP.get(ticker, "Other"), "#999999")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  FIGURE 1 — Contagion Network Graph                                     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def plot_contagion_network(
    adj: np.ndarray,
    mention_counts: dict[str, int] | None = None,
    highlight_path: list[str] | None = None,
    out_path: Path | None = None,
):
    """
    Force-directed graph of all tickers.
    Nodes sized by article count, coloured by sector.
    Edges weighted by contagion strength.
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    G = nx.DiGraph()

    for i, t in enumerate(TICKERS):
        G.add_node(t)
    for i in range(N_TICKERS):
        for j in range(N_TICKERS):
            w = adj[i, j]
            if w > 0.05:
                G.add_edge(TICKERS[i], TICKERS[j], weight=w)

    pos = nx.spring_layout(G, k=2.5, iterations=80, seed=42)

    # Node sizes
    if mention_counts:
        sizes = [300 + mention_counts.get(t, 0) * 40 for t in G.nodes()]
    else:
        sizes = [400] * len(G.nodes())

    node_colors = [_sector_color(t) for t in G.nodes()]

    # Draw edges
    edges = G.edges(data=True)
    edge_weights = [d["weight"] for _, _, d in edges]
    max_w = max(edge_weights) if edge_weights else 1
    edge_widths = [0.3 + 3.0 * (w / max_w) for w in edge_weights]
    edge_alphas = [0.15 + 0.6 * (w / max_w) for w in edge_weights]

    for (u, v, d), width, alpha in zip(edges, edge_widths, edge_alphas):
        ax.annotate(
            "", xy=pos[v], xytext=pos[u],
            arrowprops=dict(
                arrowstyle="-|>", color="#555555",
                lw=width, alpha=alpha,
                connectionstyle="arc3,rad=0.1",
            ),
        )

    # Highlight specific path
    if highlight_path:
        for k in range(len(highlight_path) - 1):
            u, v = highlight_path[k], highlight_path[k + 1]
            if u in pos and v in pos:
                ax.annotate(
                    "", xy=pos[v], xytext=pos[u],
                    arrowprops=dict(
                        arrowstyle="-|>", color="#e31a1c",
                        lw=3.5, alpha=0.9,
                        connectionstyle="arc3,rad=0.15",
                    ),
                )

    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=sizes,
                           node_color=node_colors, edgecolors="white",
                           linewidths=1.2, alpha=0.92)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=7, font_weight="bold",
                            font_color="white")

    # Legend
    unique_sectors = sorted(set(SECTOR_MAP.get(t, "Other") for t in TICKERS))
    patches = [mpatches.Patch(color=SECTOR_COLORS.get(s, "#999"),
                              label=s) for s in unique_sectors]
    ax.legend(handles=patches, loc="lower left", ncol=2, framealpha=0.8,
              fontsize=7)

    ax.set_title("Semantic Contagion Network — July 2022 Equity Universe")
    ax.axis("off")

    out = out_path or FIGURES_DIR / "fig1_contagion_network.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    print(f"  Saved: {out}")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  FIGURE 2 — Propagation Cascade                                         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def plot_propagation_cascade(
    alpha_log: list[dict],
    source_ticker: str = "AAPL",
    target_tickers: list[str] | None = None,
    event_label: str = "",
    out_path: Path | None = None,
    max_pairwise: int = 3,
    active_fraction: float = 0.20,
    mean_active_outdegree: float = 0.96,
):
    """
    Shows how contagion propagates FROM a source ticker TO its top receivers.

    Only channels whose peak smoothed excitation exceeds `active_fraction` of the
    strongest channel are shown (max `max_pairwise` panels). All pairwise panels
    share a common y-axis so active spikes and flat inactive channels are
    visually comparable on the same scale.
    """
    if not alpha_log:
        return
    src_idx = TICKER_TO_IDX.get(source_ticker, 0)

    if target_tickers is None:
        final_alpha = alpha_log[-1]["alpha"]
        top_targets = np.argsort(final_alpha[src_idx])[-8:][::-1]
        target_tickers = [TICKERS[i] for i in top_targets if final_alpha[src_idx, i] > 1e-8]

    if not target_tickers:
        return

    times  = [e["time"] for e in alpha_log]
    window = min(20, max(1, len(times) // 5))
    kernel = np.ones(window) / window if window > 1 else None

    def _smooth(series):
        arr = np.array(series, dtype=float)
        return np.convolve(arr, kernel, mode="same") if kernel is not None else arr

    # ── Compute smoothed series and peak for every candidate channel ─────────
    series_map: dict[str, np.ndarray] = {}
    for ticker in target_tickers:
        tgt_idx = TICKER_TO_IDX[ticker]
        series_map[ticker] = _smooth(
            [e["alpha"][src_idx, tgt_idx] for e in alpha_log]
        )

    if not series_map:
        return

    # Use the global max across ALL candidates (not just active) so the
    # shared y-axis reflects the true dynamic range.
    global_max = max(s.max() for s in series_map.values())
    threshold  = active_fraction * global_max if global_max > 1e-10 else 1e-10

    # Keep only channels strictly above threshold, ordered by peak descending,
    # capped at max_pairwise.
    active = sorted(
        [t for t in target_tickers if series_map[t].max() > threshold],
        key=lambda t: series_map[t].max(),
        reverse=True,
    )[:max_pairwise]

    n_total_candidates = len(target_tickers)
    n_hidden = n_total_candidates - len(active)

    n = len(active)
    fig, axes = plt.subplots(n + 1, 1, figsize=(12, 2.4 * (n + 1)), sharex=True)
    if n + 1 == 1:
        axes = [axes]

    # ── Top panel: total outgoing excitation ─────────────────────────────────
    total_out    = [e["alpha"][src_idx].sum() for e in alpha_log]
    total_smooth = _smooth(total_out)

    axes[0].fill_between(times, total_smooth, alpha=0.3, color="#2171b5")
    axes[0].plot(times, total_smooth, color="#2171b5", lw=2,
                 label=f"Total $\\alpha({source_ticker} \\to *)$")
    src_sector = SECTOR_MAP.get(source_ticker, "")
    axes[0].set_title(f"{source_ticker} ({src_sector}) — Total Outgoing Excitation",
                      fontsize=10, fontweight="bold")
    axes[0].set_ylabel("$\\sum_j \\alpha$")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(alpha=0.15)

    # ── Per-target panels (active channels only, shared y-axis) ──────────────
    # shared_max is the peak of the *strongest* active channel; all panels are
    # scaled to this so inactive channels appear visibly flat by comparison.
    cmap       = plt.cm.Set1
    shared_max = max((series_map[t].max() for t in active), default=global_max)

    for row, ticker in enumerate(active):
        ax           = axes[row + 1]
        alpha_smooth = series_map[ticker]
        color        = cmap(row % 9)

        ax.fill_between(times, alpha_smooth, alpha=0.25, color=color)
        ax.plot(times, alpha_smooth, color=color, lw=1.5,
                label=f"$\\alpha({source_ticker} \\to {ticker})$")

        tgt_sector = SECTOR_MAP.get(ticker, "")
        ax.set_title(f"{source_ticker} $\\to$ {ticker} ({tgt_sector})", fontsize=9)
        ax.set_ylabel("$\\alpha$")
        ax.set_ylim(0, shared_max * 1.25)
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(alpha=0.15)

    axes[-1].set_xlabel("Time (hours from start of July 2022)")

    # Caption note: filtering threshold + sparsity finding
    pct = int(round(active_fraction * 100))
    note = (
        f"Only channels with peak excitation >{pct}% of the strongest receiver are shown "
        f"({n_hidden} near-zero pair(s) omitted). "
        f"Remaining pairs exhibit near-zero directed excitation, consistent with the "
        f"graph's sparsity (mean active out-degree {mean_active_outdegree:.2f})."
    )
    fig.text(0.5, -0.01, note, ha="center", fontsize=7.5, color="#555555",
             style="italic", wrap=True)

    if event_label:
        fig.suptitle(f"Contagion Propagation: {event_label}", fontsize=12, y=1.01)

    fig.tight_layout()
    out = out_path or FIGURES_DIR / "fig2_propagation_cascade.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    print(f"  Saved: {out}")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  FIGURE 3 — Attention Decay Heatmap                                     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def plot_attention_heatmap(
    alpha_log: list[dict],
    snapshot_idx: int = -1,
    out_path: Path | None = None,
):
    """
    47x47 heatmap of alpha(i,j) at a given event snapshot.
    Clustered by sector.
    """
    if not alpha_log:
        return
    snapshot = alpha_log[snapshot_idx]
    alpha = snapshot["alpha"]

    # Sort tickers by sector for block structure
    sector_order = sorted(TICKERS, key=lambda t: (SECTOR_MAP.get(t, "ZZZ"), t))
    idx_order = [TICKER_TO_IDX[t] for t in sector_order]
    alpha_sorted = alpha[np.ix_(idx_order, idx_order)]

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(
        alpha_sorted, cmap="YlOrRd", aspect="auto",
        interpolation="nearest",
    )

    ax.set_xticks(range(N_TICKERS))
    ax.set_yticks(range(N_TICKERS))
    ax.set_xticklabels(sector_order, rotation=90, fontsize=6)
    ax.set_yticklabels(sector_order, fontsize=6)

    # Sector boundaries
    prev_sector = None
    for i, t in enumerate(sector_order):
        s = SECTOR_MAP.get(t, "Other")
        if s != prev_sector and prev_sector is not None:
            ax.axhline(i - 0.5, color="black", lw=0.5, alpha=0.5)
            ax.axvline(i - 0.5, color="black", lw=0.5, alpha=0.5)
        prev_sector = s

    cbar = fig.colorbar(im, ax=ax, shrink=0.7, label="$\\alpha(i,j)$ excitation")
    ax.set_title(f"Bilinear Attention Heatmap — t = {snapshot['time']:.1f}h")

    out = out_path or FIGURES_DIR / "fig3_attention_heatmap.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    print(f"  Saved: {out}")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  FIGURE 4 — Latent Space (UMAP)                                         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def plot_latent_space(
    hidden_log: list[dict],
    snapshot_idx: int = -1,
    out_path: Path | None = None,
):
    """
    UMAP projection of hidden states h_i(t), coloured by sector.
    """
    if not hidden_log:
        return
    snapshot = hidden_log[snapshot_idx]
    h = snapshot["hidden"]  # (N_TICKERS, HIDDEN_DIM)

    from umap import UMAP
    reducer = UMAP(n_components=2, random_state=42, n_neighbors=min(10, N_TICKERS - 1))
    coords = reducer.fit_transform(h)

    fig, ax = plt.subplots(figsize=(10, 8))
    for i, ticker in enumerate(TICKERS):
        color = _sector_color(ticker)
        ax.scatter(coords[i, 0], coords[i, 1], c=color, s=120,
                   edgecolors="white", linewidths=0.8, zorder=3)
        ax.annotate(ticker, (coords[i, 0], coords[i, 1]),
                    fontsize=6.5, fontweight="bold",
                    textcoords="offset points", xytext=(5, 5))

    unique_sectors = sorted(set(SECTOR_MAP.get(t, "Other") for t in TICKERS))
    patches = [mpatches.Patch(color=SECTOR_COLORS.get(s, "#999"),
                              label=s) for s in unique_sectors]
    ax.legend(handles=patches, loc="best", fontsize=7, ncol=2, framealpha=0.8)
    ax.set_title(f"Latent State Space (UMAP) — t = {snapshot['time']:.1f}h")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.grid(alpha=0.15)

    out = out_path or FIGURES_DIR / "fig4_latent_space.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    print(f"  Saved: {out}")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  FIGURE 5 — Performance Comparison                                      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def plot_performance_comparison(
    full_metrics: dict,
    ablation_metrics: dict[str, dict],
    out_path: Path | None = None,
):
    """
    Multi-threshold dot/line plot: detection precision across percentile thresholds
    for Full Model and each ablation. Lead-time panel is omitted — mention in text.

    Expects each metrics dict to optionally contain a 'multi_threshold' key whose
    value is {threshold_float_or_str: {'model_precision': float, ...}}.
    Falls back to a single-point dot chart using 'detection'/'precision' if absent.
    """
    configs = ["Full Model"] + list(ablation_metrics.keys())
    all_m   = [full_metrics]  + list(ablation_metrics.values())

    # Colour/marker scheme — grayscale-friendly
    palette = ["#2171b5", "#e6550d", "#31a354", "#756bb1"]
    markers = ["o", "s", "^", "D"]
    lw      = 1.8
    ms      = 7

    # ── Try to gather multi-threshold data ──────────────────────────────────
    def _get_mt(m):
        return m.get("multi_threshold") or {}

    sample_mt = next((_get_mt(m) for m in all_m if _get_mt(m)), {})
    has_multi = bool(sample_mt)

    fig, ax = plt.subplots(figsize=(9, 5))

    if has_multi:
        # Normalise keys to float; sort ascending
        thresh_keys = sorted({float(k) for k in sample_mt.keys()})
        x_labels = [f"{int(round(t * 100))}th" for t in thresh_keys]
        x_pos    = list(range(len(thresh_keys)))

        for i, (cfg, m) in enumerate(zip(configs, all_m)):
            mt = _get_mt(m)
            prec_vals = []
            for t in thresh_keys:
                # Try both float key and rounded-string variants
                entry = mt.get(t) or mt.get(str(t)) or mt.get(f"{t:.2f}") or {}
                if isinstance(entry, dict):
                    prec_vals.append(entry.get("model_precision", entry.get("precision", 0)))
                else:
                    prec_vals.append(0.0)

            color  = palette[i % len(palette)]
            marker = markers[i % len(markers)]
            # Dashed + hollow for ablations, solid + filled for full model
            ls = "-" if i == 0 else "--"
            ax.plot(x_pos, prec_vals, color=color, marker=marker,
                    lw=lw, ms=ms, ls=ls, label=cfg, zorder=3)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels)
        ax.set_xlabel("Detection Threshold (Percentile)")

    else:
        # Fallback: single dot per config at 90th percentile
        prec_vals = [m.get("detection", {}).get("precision", 0) for m in all_m]
        for i, (cfg, val) in enumerate(zip(configs, prec_vals)):
            color  = palette[i % len(palette)]
            marker = markers[i % len(markers)]
            ax.plot([i], [val], color=color, marker=marker, ms=ms + 2,
                    ls="none", label=cfg, zorder=3)
            ax.text(i, val + 0.003, f"{val:.3f}", ha="center",
                    fontsize=8, fontweight="bold", color=color)
        ax.set_xticks(range(len(configs)))
        ax.set_xticklabels(configs, rotation=20, ha="right", fontsize=8)
        ax.set_xlabel("Ablation Configuration")

    ax.set_ylabel("Contagion Detection Precision")
    ax.set_title("Ablation Study — Detection Precision Across Thresholds")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", framealpha=0.9, fontsize=9)
    ax.grid(alpha=0.15)

    fig.tight_layout()
    out = out_path or FIGURES_DIR / "fig5_performance_comparison.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    print(f"  Saved: {out}")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  FIGURE 6 — Ablation Study                                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def plot_ablation_study(
    full_prec: float,
    ablation_results: dict[str, float],
    out_path: Path | None = None,
):
    """
    Bar chart showing detection precision when each component is removed.
    """
    labels = ["Full Model"] + list(ablation_results.keys())
    values = [full_prec] + list(ablation_results.values())

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = ["#2171b5"] + ["#fc9272"] * len(ablation_results)
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, edgecolor="white", width=0.55)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Detection Precision (90th %ile)")
    ax.set_title("Ablation: Contagion Detection Precision")
    ax.set_ylim(0, max(values) * 1.4 if max(values) > 0 else 0.3)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{val:.3f}", ha="center", fontsize=8, fontweight="bold")

    fig.tight_layout()
    out = out_path or FIGURES_DIR / "fig6_ablation_study.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_cascade_overlay(
    alpha_log: list[dict],
    source_ticker: str = "AAPL",
    target_tickers: list[str] | None = None,
    out_path: Path | None = None,
):
    """
    Overlay of outgoing alpha from a source ticker to multiple targets,
    showing relative propagation strength.
    """
    if not alpha_log:
        return
    src_idx = TICKER_TO_IDX.get(source_ticker, 0)

    if target_tickers is None:
        final_alpha = alpha_log[-1]["alpha"]
        top_targets = np.argsort(final_alpha[src_idx])[-8:][::-1]
        target_tickers = [TICKERS[i] for i in top_targets if final_alpha[src_idx, i] > 1e-8]

    fig, ax = plt.subplots(figsize=(13, 5.5))
    times = [e["time"] for e in alpha_log]
    cmap = plt.cm.Set1
    window = min(20, len(times) // 5)

    for i, ticker in enumerate(target_tickers):
        tgt_idx = TICKER_TO_IDX[ticker]
        alpha_series = np.array([e["alpha"][src_idx, tgt_idx] for e in alpha_log])
        if window > 1:
            kernel = np.ones(window) / window
            alpha_series = np.convolve(alpha_series, kernel, mode="same")

        mx = alpha_series.max()
        if mx > 1e-8:
            alpha_norm = alpha_series / mx
        else:
            alpha_norm = alpha_series

        color = cmap(i % 9)
        sector = SECTOR_MAP.get(ticker, "")
        ax.plot(times, alpha_norm, color=color, lw=2, alpha=0.85,
                label=f"$\\to$ {ticker} ({sector})")

    ax.set_xlabel("Time (hours from start of July 2022)")
    ax.set_ylabel("Normalised $\\alpha$ Excitation")
    ax.set_title(f"Contagion From {source_ticker} — Receiver Comparison")
    ax.legend(loc="upper right", fontsize=8, ncol=2, framealpha=0.85)
    ax.grid(alpha=0.15)

    fig.tight_layout()
    out = out_path or FIGURES_DIR / "fig2b_cascade_overlay.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_training_loss(losses: list[float], out_path: Path | None = None):
    """Training loss curve."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(losses, color="#2171b5", lw=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Negative Log-Likelihood")
    ax.set_title("Training Convergence")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    out = out_path or FIGURES_DIR / "training_loss.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    print(f"  Saved: {out}")
