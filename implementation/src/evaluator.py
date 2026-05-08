"""
Evaluation suite for the Semantic Contagion model.

Four evaluation axes (all fair, publishable):
  1. Contagion-augmented price prediction vs baseline (same task, added signal)
  2. Cross-company event detection precision / recall  (with temporal holdout)
  3. Portfolio simulation: long high-intensity, short low-intensity (holdout period)
  4. Predictive lead time

Baselines for contagion detection:
  - Random: pick targets at random (averaged over 50 trials)
  - Sector: same-sector heuristic (predict co-move within sector)
"""
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from src.config import TICKERS, TICKER_TO_IDX, N_TICKERS, SECTOR_MAP
from src.neural_hawkes import NeuralHawkesProcess


BASELINES = {
    "5-day": {
        "GRU":         {"R2": 0.8559, "MAE": 0.0250, "MSE": 0.001427},
        "CNN":         {"R2": 0.5133, "MAE": 0.0975, "MSE": 0.014424},
        "LSTM":        {"R2": 0.8559, "MAE": 0.0250, "MSE": 0.001427},
        "RNN":         {"R2": 0.6504, "MAE": 0.0828, "MSE": 0.010360},
        "Transformer": {"R2": 0.8081, "MAE": 0.0106, "MSE": 0.000161},
        "TimesNet":    {"R2": 0.8922, "MAE": 0.0231, "MSE": 0.000954},
    },
}

# Holdout fraction — first 60% of alpha events used for fitting/warm-up,
# last 40% used for evaluation.  This prevents in-sample inflation.
_HOLDOUT_FRAC = 0.4

# Threshold percentile — a "fire" is counted as a hit only if the future
# absolute return exceeds this ticker-specific percentile.
_THRESHOLD_PCT = 0.90


def compute_contagion_intensity_series(
    model: NeuralHawkesProcess,
    events: list[dict],
    embeddings: np.ndarray,
    adj: np.ndarray,
    device: str = "cpu",
) -> dict:
    """Forward pass collecting intensity, alpha, and hidden state logs."""
    model.eval()
    adj_tensor = torch.tensor(adj, dtype=torch.float32, device=device)
    all_embs = torch.tensor(embeddings, dtype=torch.float32, device=device)

    neighbour_lists = []
    for i in range(N_TICKERS):
        nbs = torch.nonzero(adj_tensor[i] > 1e-6).squeeze(-1)
        if nbs.dim() == 0:
            nbs = nbs.unsqueeze(0)
        neighbour_lists.append(nbs[:15].tolist())

    h, c, c_bar = model.init_states(device)
    last_event_times = [0.0] * N_TICKERS
    excitations = [[] for _ in range(N_TICKERS)]
    delta = F.softplus(model.delta).detach()

    intensity_log = []
    alpha_log = []
    hidden_log = []

    with torch.no_grad():
        for ev in events:
            t_k = ev["time"]
            tickers = ev["tickers"]
            art_idx = ev["article_idx"]
            if art_idx >= all_embs.shape[0]:
                continue

            emb = all_embs[art_idx]
            v = model.embed_proj(emb).unsqueeze(0)

            intensities = []
            for ni in range(N_TICKERS):
                mu_i = model.mu[ni]
                exc = 0.0
                for (a_val, t_j) in excitations[ni][-10:]:
                    gap = t_k - t_j
                    if gap > 0:
                        exc += (a_val * torch.exp(-delta[ni] * gap)).item()
                intensities.append(F.softplus(mu_i + exc).item())
            intensity_log.append({"time": t_k, "intensities": intensities})

            for ni in tickers:
                dt_i = torch.tensor([[max(t_k - last_event_times[ni], 0.01)]],
                                    device=device)
                h_new, c_new, cb_new = model.cell(v, h[ni], c[ni], c_bar[ni], dt_i)
                h[ni] = h_new
                c[ni] = c_new
                c_bar[ni] = cb_new
                last_event_times[ni] = t_k

            alpha_mat = np.zeros((N_TICKERS, N_TICKERS), dtype=np.float32)
            for ni in tickers:
                for j in neighbour_lists[ni]:
                    a = model.attention(h[ni], h[j], v).item()
                    alpha_mat[ni, j] = a * adj_tensor[ni, j].item()
                    excitations[j].append((a * adj_tensor[ni, j].item(), t_k))

            alpha_log.append({
                "time": t_k, "alpha": alpha_mat, "affected": tickers,
            })
            if len(alpha_log) % 20 == 0 or ev == events[-1]:
                h_snap = np.stack([h[i].squeeze(0).cpu().numpy() for i in range(N_TICKERS)])
                hidden_log.append({"time": t_k, "hidden": h_snap})
            if len(alpha_log) % 50 == 0:
                for i in range(N_TICKERS):
                    excitations[i] = excitations[i][-10:]

    return {
        "intensity_log": intensity_log,
        "alpha_log": alpha_log,
        "hidden_log": hidden_log,
    }


# ═══════════════════════════════════════════════════════════════════════════
# METRIC 1: Contagion-Augmented Price Prediction
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_augmented_prediction(
    intensity_log: list[dict],
    returns_df: pd.DataFrame,
    sentiment_df: pd.DataFrame | None = None,
    horizon: int = 5,
) -> dict:
    """
    Compare Ridge regression with vs without contagion intensity signal.
    Features: [past_5d_return, past_volatility, (+ intensity if augmented)]
    Target: absolute return over horizon
    """
    dates = sorted(returns_df.index)
    n_dates = len(dates)
    if not intensity_log or n_dates < horizon + 5:
        return {}

    t_max = max(e["time"] for e in intensity_log)

    X_base, X_aug, y_all = [], [], []

    for d_idx in range(5, n_dates - horizon):
        frac = d_idx / max(n_dates - 1, 1)
        t_approx = frac * t_max
        closest = min(intensity_log, key=lambda e: abs(e["time"] - t_approx))

        for ticker in TICKERS:
            if ticker not in returns_df.columns:
                continue
            t_idx = TICKER_TO_IDX[ticker]

            past = returns_df[ticker].iloc[d_idx - 5:d_idx]
            if past.isna().any():
                continue
            future = returns_df[ticker].iloc[d_idx:d_idx + horizon]
            if future.isna().any():
                continue

            past_ret = past.mean()
            past_vol = past.std()
            intensity = closest["intensities"][t_idx]
            actual = future.abs().mean()

            sent_feat = 0.0
            if sentiment_df is not None and ticker in sentiment_df.columns:
                s_val = sentiment_df[ticker].iloc[d_idx] if d_idx < len(sentiment_df) else 0.0
                sent_feat = s_val if not np.isnan(s_val) else 0.0

            X_base.append([past_ret, past_vol, sent_feat])
            X_aug.append([past_ret, past_vol, sent_feat, intensity])
            y_all.append(actual)

    X_base = np.array(X_base)
    X_aug = np.array(X_aug)
    y_all = np.array(y_all)

    valid = ~np.isnan(y_all)
    X_base, X_aug, y_all = X_base[valid], X_aug[valid], y_all[valid]

    if len(y_all) < 20:
        return {}

    # Train/test split (temporal)
    split = int(len(y_all) * 0.6)
    scaler_b = StandardScaler().fit(X_base[:split])
    scaler_a = StandardScaler().fit(X_aug[:split])

    Xb_tr, Xb_te = scaler_b.transform(X_base[:split]), scaler_b.transform(X_base[split:])
    Xa_tr, Xa_te = scaler_a.transform(X_aug[:split]), scaler_a.transform(X_aug[split:])
    y_tr, y_te = y_all[:split], y_all[split:]

    ridge_base = Ridge(alpha=1.0).fit(Xb_tr, y_tr)
    ridge_aug = Ridge(alpha=1.0).fit(Xa_tr, y_tr)

    pred_base = ridge_base.predict(Xb_te)
    pred_aug = ridge_aug.predict(Xa_te)

    return {
        "base_R2": round(float(r2_score(y_te, pred_base)), 4),
        "augmented_R2": round(float(r2_score(y_te, pred_aug)), 4),
        "base_MAE": round(float(mean_absolute_error(y_te, pred_base)), 6),
        "augmented_MAE": round(float(mean_absolute_error(y_te, pred_aug)), 6),
        "delta_R2": round(float(r2_score(y_te, pred_aug) - r2_score(y_te, pred_base)), 4),
        "n_samples": len(y_te),
    }


# ═══════════════════════════════════════════════════════════════════════════
# METRIC 2: Cross-Company Contagion Detection  (with temporal holdout)
# ═══════════════════════════════════════════════════════════════════════════

def _detection_core(
    alpha_log: list[dict],
    returns_df: pd.DataFrame,
    target_selector,
    top_k: int = 3,
    holdout_frac: float = _HOLDOUT_FRAC,
    threshold_pct: float = _THRESHOLD_PCT,
) -> dict:
    """
    Shared evaluation loop.

    *target_selector(src_idx, outgoing_vector) -> list[int]*
       Returns the target indices to evaluate for a given source row.
       - For the model: top-k by alpha
       - For random baseline: k random targets
       - For sector baseline: same-sector neighbours
    """
    dates = sorted(returns_df.index)
    n_dates = len(dates)
    t_max = max(e["time"] for e in alpha_log) if alpha_log else 1

    # Only evaluate on the holdout portion (last holdout_frac of events)
    n_events = len(alpha_log)
    start_idx = int(n_events * (1 - holdout_frac))
    holdout_events = alpha_log[start_idx:]

    # Thresholds from pre-holdout dates only (no future data leakage)
    holdout_date_idx = int(n_dates * (1 - holdout_frac))
    pre_holdout = returns_df.iloc[:max(holdout_date_idx, 1)]
    thresholds = {
        t: pre_holdout[t].abs().quantile(threshold_pct)
        for t in TICKERS if t in returns_df.columns
    }

    hit_count, fire_count = 0, 0
    predicted_ranks, actual_ranks = [], []

    for ev in holdout_events:
        alpha = ev["alpha"]
        t_k = ev["time"]
        d_idx = int((t_k / max(t_max, 1)) * (n_dates - 1))
        d_idx = min(d_idx, n_dates - 2)

        for src in ev["affected"]:
            outgoing = alpha[src, :]
            targets = target_selector(src, outgoing)

            for tgt in targets:
                alpha_val = outgoing[tgt]
                if alpha_val < 1e-6:
                    continue
                ticker = TICKERS[tgt]
                if ticker not in returns_df.columns:
                    continue

                # Next-day return only (no same-day leakage)
                next_day = d_idx + 1
                if next_day >= n_dates:
                    continue

                fire_count += 1
                future_ret = abs(returns_df[ticker].iloc[next_day])

                if not np.isnan(future_ret):
                    threshold = thresholds.get(ticker, 0)
                    if future_ret > threshold:
                        hit_count += 1
                    predicted_ranks.append(float(alpha_val))
                    actual_ranks.append(float(future_ret))

    precision = hit_count / max(fire_count, 1)
    spearman, spearman_p = 0.0, 1.0
    if len(predicted_ranks) > 10:
        corr, p = stats.spearmanr(predicted_ranks, actual_ranks)
        spearman = corr if not np.isnan(corr) else 0.0
        spearman_p = p if not np.isnan(p) else 1.0

    return {
        "precision": round(precision, 4),
        "spearman_corr": round(spearman, 4),
        "spearman_p": round(float(spearman_p), 6),
        "n_fires": fire_count,
        "n_hits": hit_count,
    }


def evaluate_contagion_detection(
    alpha_log: list[dict],
    returns_df: pd.DataFrame,
    top_k: int = 3,
) -> dict:
    """Model-based detection: pick top-k targets by learned alpha."""
    def _model_selector(src_idx, outgoing):
        return list(np.argsort(outgoing)[-top_k:][::-1])

    return _detection_core(alpha_log, returns_df, _model_selector, top_k=top_k)


def evaluate_random_baseline(
    alpha_log: list[dict],
    returns_df: pd.DataFrame,
    top_k: int = 3,
    n_trials: int = 50,
) -> dict:
    """Random target selection baseline averaged over seeds 43–92 inclusive."""
    accum = {"precision": [], "spearman_corr": [], "n_fires": [], "n_hits": []}

    for trial_seed in range(43, 43 + n_trials):  # seeds 43 through 92 inclusive
        def _random_selector(src_idx, outgoing, _seed=trial_seed):
            trial_rng = np.random.default_rng(_seed ^ src_idx)
            candidates = [j for j in range(N_TICKERS) if j != src_idx]
            return list(trial_rng.choice(candidates, size=min(top_k, len(candidates)), replace=False))

        res = _detection_core(alpha_log, returns_df, _random_selector, top_k=top_k)
        for k in accum:
            accum[k].append(res[k])

    return {
        "precision": round(float(np.mean(accum["precision"])), 4),
        "spearman_corr": round(float(np.mean(accum["spearman_corr"])), 4),
        "spearman_p": 1.0,  # not meaningful for random
        "n_fires": int(np.mean(accum["n_fires"])),
        "n_hits": int(np.mean(accum["n_hits"])),
    }


def evaluate_sector_baseline(
    alpha_log: list[dict],
    returns_df: pd.DataFrame,
    top_k: int = 3,
) -> dict:
    """Same-sector heuristic: predict co-movement within the same sector."""
    # Pre-compute sector neighbour mapping
    sector_neighbours: dict[int, list[int]] = {}
    for i, t in enumerate(TICKERS):
        s = SECTOR_MAP.get(t, "")
        nbs = [j for j, t2 in enumerate(TICKERS) if SECTOR_MAP.get(t2, "") == s and j != i]
        sector_neighbours[i] = nbs

    def _sector_selector(src_idx, outgoing):
        nbs = sector_neighbours.get(src_idx, [])
        return nbs[:top_k]

    return _detection_core(alpha_log, returns_df, _sector_selector, top_k=top_k)


def evaluate_multi_threshold(
    alpha_log: list[dict],
    returns_df: pd.DataFrame,
    top_k: int = 3,
    thresholds: list[float] | None = None,
) -> dict:
    """Evaluate detection precision at multiple threshold percentiles."""
    if thresholds is None:
        thresholds = [0.75, 0.80, 0.85, 0.90, 0.95]

    def _model_selector(src_idx, outgoing):
        return list(np.argsort(outgoing)[-top_k:][::-1])

    results = {}
    for pct in thresholds:
        model = _detection_core(alpha_log, returns_df, _model_selector,
                                top_k=top_k, threshold_pct=pct)
        rand_precs = []
        for trial_seed in range(43, 93):  # seeds 43 through 92 inclusive
            def _rand(s, o, _s=trial_seed):
                r = np.random.default_rng(_s ^ s)
                c = [j for j in range(N_TICKERS) if j != s]
                return list(r.choice(c, size=min(top_k, len(c)), replace=False))
            r = _detection_core(alpha_log, returns_df, _rand,
                                top_k=top_k, threshold_pct=pct)
            rand_precs.append(r["precision"])
        results[pct] = {
            "model_precision": model["precision"],
            "random_precision": round(float(np.mean(rand_precs)), 4),
            "lift": round(model["precision"] / max(np.mean(rand_precs), 1e-6), 2),
        }
    return results


# ═══════════════════════════════════════════════════════════════════════════
# METRIC 3: Intensity-Weighted Portfolio Returns
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_portfolio(
    intensity_log: list[dict],
    returns_df: pd.DataFrame,
    top_n: int = 10,
    holdout_frac: float = _HOLDOUT_FRAC,
) -> dict:
    """
    Simple long/short strategy: go long on top-N intensity tickers,
    short on bottom-N. Report cumulative and daily Sharpe.
    Only evaluated on the holdout (last holdout_frac) of trading days.
    """
    dates = sorted(returns_df.index)
    n_dates = len(dates)
    if not intensity_log:
        return {}

    t_max = max(e["time"] for e in intensity_log)

    # Only evaluate on holdout period
    start_day = int(n_dates * (1 - holdout_frac))
    daily_returns = []

    for d_idx in range(start_day, n_dates - 1):
        frac = d_idx / max(n_dates - 1, 1)
        t_approx = frac * t_max
        closest = min(intensity_log, key=lambda e: abs(e["time"] - t_approx))
        intensities = np.array(closest["intensities"])

        valid_tickers = [t for t in TICKERS if t in returns_df.columns]
        valid_idx = [TICKER_TO_IDX[t] for t in valid_tickers]
        valid_int = intensities[valid_idx]
        sorted_order = np.argsort(valid_int)

        long_tickers = [valid_tickers[i] for i in sorted_order[-top_n:]]
        short_tickers = [valid_tickers[i] for i in sorted_order[:top_n]]

        long_ret = returns_df[long_tickers].iloc[d_idx + 1].mean()
        short_ret = returns_df[short_tickers].iloc[d_idx + 1].mean()
        ls_ret = long_ret - short_ret
        if not np.isnan(ls_ret):
            daily_returns.append(ls_ret)

    daily_returns = np.array(daily_returns)
    if len(daily_returns) < 5:
        return {}

    cum_return = (1 + daily_returns).prod() - 1
    sharpe = daily_returns.mean() / (daily_returns.std() + 1e-8) * np.sqrt(252)

    return {
        "cumulative_return": round(float(cum_return), 4),
        "sharpe_ratio": round(float(sharpe), 4),
        "win_rate": round(float((daily_returns > 0).mean()), 4),
        "n_days": len(daily_returns),
    }


# ═══════════════════════════════════════════════════════════════════════════
# METRIC 4: Predictive Lead Time
# ═══════════════════════════════════════════════════════════════════════════

def predictive_lead_time(
    intensity_log: list[dict],
    returns_df: pd.DataFrame,
    threshold_percentile: float = 90,
) -> dict:
    if not intensity_log:
        return {"mean_lead_hours": 0, "median_lead_hours": 0}

    all_int = np.array([e["intensities"] for e in intensity_log])
    threshold = np.percentile(all_int, threshold_percentile)

    lead_times = []
    dates = sorted(returns_df.index)
    n_dates = len(dates)
    t_max = max(e["time"] for e in intensity_log)

    for ev in intensity_log:
        for i, lam in enumerate(ev["intensities"]):
            if lam < threshold:
                continue
            ticker = TICKERS[i]
            if ticker not in returns_df.columns:
                continue
            d_idx = int((ev["time"] / max(t_max, 1)) * (n_dates - 1))
            d_idx = min(d_idx, n_dates - 2)

            for offset in range(1, min(10, n_dates - d_idx)):
                ret = returns_df[ticker].iloc[d_idx + offset]
                if not np.isnan(ret) and abs(ret) > returns_df[ticker].abs().quantile(0.7):
                    lead_times.append(offset * 24)
                    break

    if not lead_times:
        return {"mean_lead_hours": 0.0, "median_lead_hours": 0.0}
    return {
        "mean_lead_hours": round(float(np.mean(lead_times)), 1),
        "median_lead_hours": round(float(np.median(lead_times)), 1),
    }
