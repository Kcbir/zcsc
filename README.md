# Zero-Copy Semantic Contagion

**ACM SIGMOD FinDS '26** · May 31 2026 · Bengaluru, India  

**Author:** Kabir Murjani  
**Contact:** kabir.murjani@iimb.ac.in · 23bee064@nirmauni.ac.in

---

## Overview

Financial news propagates price-relevant information across companies through
co-mention, semantic proximity, and return correlation channels simultaneously.
Existing models process these channels independently and batch events in fixed
windows, introducing latency and discarding the temporal ordering that governs
contagion dynamics.

This work presents a continuous-time architecture that couples a zero-copy Rust
ingestion edge — performing CSV parsing, monotonic timestamping, and semantic
deduplication with no heap allocation on the hot path — to a Neural Hawkes
Process whose per-node hidden states evolve as events arrive and decay between
them. A learnable mixture of three adjacency sources drives a bilinear attention
mechanism that scores directed excitation between tickers; edges whose cumulative
excitation falls below a threshold are pruned online. On the FNSPID July 2022
corpus (638 articles, 47 tickers), the full model achieves 0.412 contagion
detection precision at the 90th-percentile threshold against a random baseline
of 0.064 (6.4× lift), with a mean predictive lead time of 24 hours.

---

## Repository Structure

```
.
├── implementation/
│   ├── rust_edge/              # Zero-copy ingestion edge (Rust)
│   │   ├── src/lib.rs          # CSV parser, monotonic clock, ticker scanner, semantic gate
│   │   ├── benches/latency.rs  # Criterion benchmarks → Table 3
│   │   ├── Cargo.toml
│   │   └── Cargo.lock
│   ├── src/                    # Python model library
│   │   ├── config.py           # Paths, ticker universe, hyperparameters, set_seed()
│   │   ├── neural_hawkes.py    # ContinuousTimeLSTMCell, BilinearAttention, NeuralHawkesProcess
│   │   ├── trainer.py          # MLE training loop with S_ij edge pruning
│   │   ├── evaluator.py        # Detection, portfolio, lead-time metrics; random/sector baselines
│   │   ├── graph_builder.py    # Co-mention, semantic, correlation adjacency construction
│   │   ├── data_loader.py      # CSV ingestion, returns/sentiment matrices, event sequence
│   │   ├── embedder.py         # MiniLM-L6-v2 sentence embedder with disk cache
│   │   ├── entity_extractor.py # Regex NER: 47-ticker mention extraction
│   │   └── visualizer.py       # Publication figures (Figs 1–6)
│   └── scripts/                # Entry-point scripts
│       ├── load_fnspid.py      # Step 1: preprocess raw data → cache/
│       ├── train_hawkes.py     # Step 2: train model + ablations + generate figures
│       ├── evaluate_contagion.py  # Step 3: load checkpoint → Tables 2, 4, 5 (JSON)
│       ├── run_ablations.py    # Step 4: ablation study → Table 5 (JSON)
│       └── generate_figures.py # Regenerate Figs 1–6 from cache without retraining
├── cache/                      # Precomputed artefacts (committed)
│   ├── hawkes_model.pt         # Trained checkpoint (1.3 MB)
│   ├── embeddings.npy          # Article embeddings, 384-dim MiniLM
│   ├── combined_adjacency.npy  # Weighted adjacency matrix
│   ├── events.pkl              # Preprocessed event sequence
│   └── ...                     # Remaining graph matrices and returns
├── figures/                    # Publication figures (committed)
│   ├── fig1_contagion_network.{pdf,png}
│   ├── fig2_propagation_cascade.{pdf,png}
│   ├── fig3_attention_heatmap.{pdf,png}
│   ├── fig4_latent_space.{pdf,png}
│   ├── fig5_performance_comparison.{pdf,png}
│   ├── fig6_ablation_study.{pdf,png}
│   └── training_loss.{pdf,png}
├── data/                       # Raw data (not committed — see data/README_data.md)
│   └── README_data.md
├── requirements.txt
└── README.md
```

---

## Requirements

| Component | Version |
|---|---|
| Python | 3.11 |
| PyTorch | 2.2.2 (CPU build) |
| NumPy | 1.26.4 |
| sentence-transformers | 2.7.0 |
| Rust | 1.77 stable |
| criterion | 0.5 |

Install Python dependencies:

```bash
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Install Rust via [rustup](https://rustup.rs):

```bash
rustup toolchain install 1.77
```

---

## Data

Raw data is the FNSPID corpus (Chen et al., CIKM 2023) sliced to July 2022.
It must be downloaded and placed under `data/` before running any script.
Full instructions, expected SHA-256 hashes, and the slicing command are in
[data/README_data.md](data/README_data.md).

---

## Reproducing the Paper

All commands are issued from the repository root. The `cache/` directory
contains precomputed artefacts; steps 1 and 2 are only needed if reproducing
from scratch.

```bash
# Build the Rust ingestion edge (Table 3 benchmarks)
cd implementation/rust_edge
cargo build --release
cd ../..

# Step 1 — Preprocess raw data (requires data/ to be populated first)
python implementation/scripts/load_fnspid.py

# Step 2 — Train the model (seed=42, CPU, ~50 epochs, ≈20 min on M2)
python implementation/scripts/train_hawkes.py

# Step 3 — Evaluate: Tables 2, 4, 5
python implementation/scripts/evaluate_contagion.py > results.json

# Step 4 — Ablation study: Table 5 with structural diagnostics
python implementation/scripts/run_ablations.py > ablations.json

# Regenerate Figs 1–6 from existing checkpoint without retraining
python implementation/scripts/generate_figures.py
```

Running steps 3 and 4 against the committed `cache/hawkes_model.pt` reproduces
the exact numbers in the paper without retraining.

---

## Latency Benchmarks

Benchmarks were measured on an Apple M2 (8 GB, AArch64, CPU-only). The Criterion
suite covers five operations: CSV line parsing, monotonic timestamp stamping,
47-ticker scan, 384-dimensional cosine similarity, and semantic gate admission.

```bash
cd implementation/rust_edge
cargo criterion
```

Expected outputs matching Table 3 (Apple M2, release + native):

| Benchmark | Median latency |
|---|---|
| `parse_csv_line` | 187 ns |
| `monotonic_stamp` | 14 ns |
| `scan_47_tickers` | 1.2 µs |
| `cosine_384d` | 310 ns |
| `semantic_gate_admit` | 340 ns |

---

## Citation

```bibtex
@inproceedings{murjani2026zcsc,
  title     = {Zero-Copy Semantic Contagion: An In-Memory Streaming
               Architecture for Evolving Attention Graphs},
  author    = {Murjani, Kabir},
  booktitle = {ACM SIGMOD Workshop on Data Management for the Modern Financial Systems (FinDS '26)},
  year      = {2026},
  month     = {May},
  address   = {Bengaluru, India},
}
```

---

## Licence

The code in this repository is released under the **MIT Licence**.  
The paper text is distributed under **CC BY 4.0** (non-archival workshop).  
The FNSPID dataset is subject to its own licence; see [data/README_data.md](data/README_data.md).
