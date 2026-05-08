"""
Optimised MLE training for the Neural Hawkes Process.

Uses per-node state lists to avoid in-place tensor mutation.
Edge pruning follows Proposition 1 in the paper:
    S_ij = Σ_k α(i→j, t_k) · exp(−δ_j · (T − t_k))
Edges with S_ij < prune_threshold are removed (subject to a minimum-
neighbour safety floor).

Loss scaling heuristic
----------------------
The raw MLE objective is  L = −log_lik + integral.  In early training the
integral term often dominates because μ is initialised uniformly and
excitation is still close to zero.  We rescale the integral term by
  scale = |log_lik| / |integral| × 0.3
so that early gradients are driven primarily by the log-likelihood signal.
As training progresses and excitation grows, the two terms naturally
equilibrate. This adaptive scaling achieves the same effect as
alternating-phase training without the implementation complexity.
"""
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import trange
from src.config import N_TICKERS, LEARNING_RATE, N_EPOCHS, EDGE_PRUNE_THRESHOLD
from src.neural_hawkes import NeuralHawkesProcess

# Wait this many epochs before activating edge pruning so the model
# can learn reasonable alpha values first.
_PRUNE_WARMUP_EPOCHS = 10

# Every node keeps at least this many neighbours even if S_ij is tiny.
_MIN_NEIGHBOURS = 3


def train(
    model: NeuralHawkesProcess,
    events: list[dict],
    embeddings: np.ndarray,
    adj: np.ndarray,
    n_epochs: int = N_EPOCHS,
    lr: float = LEARNING_RATE,
    prune_threshold: float = EDGE_PRUNE_THRESHOLD,
    device: str = "cpu",
    verbose: bool = True,
    adj_components: tuple[np.ndarray, ...] | None = None,
) -> list[float]:
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    all_embs = torch.tensor(embeddings, dtype=torch.float32, device=device)

    # Adjacency: learnable mixture weights or static
    learnable_adj = adj_components is not None
    if learnable_adj:
        _comp_tensors = tuple(
            torch.tensor(a, dtype=torch.float32, device=device)
            for a in adj_components
        )
    else:
        _static_adj = torch.tensor(adj, dtype=torch.float32, device=device)

    prune_mask = torch.ones(N_TICKERS, N_TICKERS, device=device)

    def _current_adj():
        """Differentiable adjacency when using learnable mixture weights."""
        if learnable_adj:
            w = F.softmax(model.adj_logits, dim=0)
            base = w[0] * _comp_tensors[0] + w[1] * _comp_tensors[1] + w[2] * _comp_tensors[2]
        else:
            base = _static_adj
        return base * prune_mask

    losses = []
    pbar = trange(n_epochs, desc="Training", disable=not verbose)

    for epoch in pbar:
        model.train()
        optimizer.zero_grad()

        # Recompute adjacency (tracks adj_logits gradient when learnable)
        adj_tensor = _current_adj()
        neighbour_lists = []
        with torch.no_grad():
            _det = adj_tensor.detach()
            for _i in range(N_TICKERS):
                _nbs = torch.nonzero(_det[_i] > 1e-6).squeeze(-1)
                if _nbs.dim() == 0:
                    _nbs = _nbs.unsqueeze(0)
                neighbour_lists.append(_nbs[:15].tolist())

        h, c, c_bar = model.init_states(device)
        last_event_times = [0.0] * N_TICKERS

        # Per-node running excitation (list of (alpha_value, event_time))
        excitations = [[] for _ in range(N_TICKERS)]

        # Per-edge excitation records for S_ij computation
        # key: (i, j), value: list of (alpha_float, event_time)
        edge_excitations: dict[tuple[int, int], list[tuple[float, float]]] = {}

        log_lik = torch.tensor(0.0, device=device)
        integral = torch.tensor(0.0, device=device)
        delta = F.softplus(model.delta)

        prev_time = 0.0
        n_processed = 0

        for ev_idx, ev in enumerate(events):
            t_k = ev["time"]
            tickers = ev["tickers"]
            art_idx = ev["article_idx"]
            if art_idx >= all_embs.shape[0]:
                continue

            emb = all_embs[art_idx]
            v = model.embed_proj(emb).unsqueeze(0)  # (1, embed_dim)

            # ── Log-intensity at event time for affected tickers ──
            for ni in tickers:
                mu_i = model.mu[ni]
                exc = torch.tensor(0.0, device=device)
                for (a_val, t_j) in excitations[ni][-10:]:
                    gap = t_k - t_j
                    if gap > 0:
                        exc = exc + a_val * torch.exp(-delta[ni] * gap)
                lam = F.softplus(mu_i + exc)
                log_lik = log_lik + torch.log(lam + 1e-8)

            # ── Integral penalty (sampled every 10 events) ──
            if ev_idx % 10 == 0:
                gap = t_k - prev_time
                if gap > 0.1:
                    for ni in range(N_TICKERS):
                        mu_i = model.mu[ni]
                        exc = torch.tensor(0.0, device=device)
                        for (a_val, t_j) in excitations[ni][-3:]:
                            g = (prev_time + t_k) / 2 - t_j
                            if g > 0:
                                exc = exc + a_val * torch.exp(-delta[ni] * g)
                        integral = integral + F.softplus(mu_i + exc) * gap

            # ── Update c-LSTM states for affected nodes ──
            for ni in tickers:
                dt_i = torch.tensor([[max(t_k - last_event_times[ni], 0.01)]],
                                    device=device)
                h_new, c_new, cb_new = model.cell(v, h[ni], c[ni], c_bar[ni], dt_i)
                h[ni] = h_new
                c[ni] = c_new
                c_bar[ni] = cb_new
                last_event_times[ni] = t_k

            # ── Compute alphas for outgoing edges ──
            for ni in tickers:
                for j in neighbour_lists[ni]:
                    alpha_ij = model.attention(h[ni], h[j], v).squeeze()
                    edge_w = adj_tensor[ni, j]
                    weighted_alpha = alpha_ij * edge_w
                    excitations[j].append((weighted_alpha, t_k))

                    # Record for S_ij (detached float for memory safety)
                    key = (ni, j)
                    a_float = weighted_alpha.item()
                    if key not in edge_excitations:
                        edge_excitations[key] = []
                    edge_excitations[key].append((a_float, t_k))

            prev_time = t_k
            n_processed += 1

            # Truncate history to prevent memory blowup
            if ev_idx % 50 == 0:
                for i in range(N_TICKERS):
                    if len(excitations[i]) > 15:
                        excitations[i] = [
                            (a.detach(), t) for a, t in excitations[i][-10:]
                        ]

        # ── Loss scaling heuristic (see docstring) ──
        if integral.abs().item() > 1e-6:
            scale = log_lik.abs().item() / integral.abs().item() * 0.3
        else:
            scale = 1.0
        loss = -log_lik + integral * scale

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        loss_val = loss.item()
        losses.append(loss_val)
        if verbose:
            pbar.set_postfix(loss=f"{loss_val:.2f}", ll=f"{log_lik.item():.2f}")

        # ── S_ij-based edge pruning (Proposition 1) ──
        if (prune_threshold > 0
                and epoch >= _PRUNE_WARMUP_EPOCHS
                and epoch % 5 == 0):
            T_final = prev_time if prev_time > 0 else 1.0
            delta_np = delta.detach().cpu().numpy()

            # Compute S_ij for every recorded edge
            s_ij: dict[tuple[int, int], float] = {}
            for (i, j), records in edge_excitations.items():
                s = 0.0
                for (a_val, t_k_rec) in records:
                    gap = T_final - t_k_rec
                    s += a_val * np.exp(-delta_np[j] * gap)
                s_ij[(i, j)] = s

            # Build per-target incoming edge lists from S_ij
            incoming_edges: dict[int, list[tuple[int, float]]] = {}
            for (i, j), s_val in s_ij.items():
                if j not in incoming_edges:
                    incoming_edges[j] = []
                incoming_edges[j].append((i, s_val))

            # For each node, rank incoming edges and prune weak ones
            for j in range(N_TICKERS):
                incoming = incoming_edges.get(j, [])
                if not incoming:
                    continue
                incoming.sort(key=lambda x: x[1], reverse=True)
                for rank_idx, (i, s_val) in enumerate(incoming):
                    if rank_idx >= _MIN_NEIGHBOURS and s_val < prune_threshold:
                        prune_mask[i, j] = 0.0

        # Detach all states for next epoch
        h = [hi.detach() for hi in h]
        c = [ci.detach() for ci in c]
        c_bar = [ci.detach() for ci in c_bar]
        for i in range(N_TICKERS):
            excitations[i] = [(a.detach(), t) for a, t in excitations[i][-10:]]

    return losses
