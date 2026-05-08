use criterion::{black_box, criterion_group, criterion_main, Criterion};
use contagion_edge::{parse_csv_line, MonotonicClock, scan_tickers, cosine_similarity, SemanticGate};
use std::time::Duration;

fn bench_parse_csv_line(c: &mut Criterion) {
    let line = "2022-07-05,AAPL,Apple reports strong Q3 earnings beating Wall Street estimates,Apple Inc (NASDAQ: AAPL) reported quarterly revenue of $83 billion beating analyst expectations of $82.8 billion. The company saw strong demand for iPhone 14 series particularly in emerging markets while services revenue reached an all-time high.";
    c.bench_function("parse_csv_line", |b| {
        b.iter(|| parse_csv_line(black_box(line)))
    });
}

fn bench_monotonic_stamp(c: &mut Criterion) {
    let mut clock = MonotonicClock::new();
    c.bench_function("monotonic_stamp", |b| {
        b.iter(|| clock.stamp())
    });
}

fn bench_ticker_scan(c: &mut Criterion) {
    let tickers: Vec<&str> = vec![
        "AAPL", "ABBV", "AMD", "AMGN", "BABA", "BHP", "BIDU", "BIIB",
        "BRK-B", "C", "CAT", "CMCSA", "CMG", "COP", "COST", "CRM",
        "CVX", "DIS", "EBAY", "GE", "GILD", "GLD", "GOOG", "GSK",
        "INTC", "KO", "MRK", "MSFT", "MU", "NKE", "NVDA", "ORCL",
        "PEP", "PYPL", "QCOM", "QQQ", "T", "TGT", "TM", "TSLA",
        "TSM", "USO", "V", "WFC", "WMT", "XLF", "AAL",
    ];
    let text = "Apple (AAPL) and NVIDIA (NVDA) both rallied on semiconductor supply chain news. \
                Taiwan Semiconductor (TSM) confirmed new capacity plans while Intel (INTC) \
                announced delays. Microsoft and Google also moved on the broader tech sentiment.";
    c.bench_function("scan_47_tickers", |b| {
        b.iter(|| scan_tickers(black_box(text), black_box(&tickers)))
    });
}

fn bench_cosine_384(c: &mut Criterion) {
    let a: Vec<f32> = (0..384).map(|i| (i as f32 * 0.01).sin()).collect();
    let b: Vec<f32> = (0..384).map(|i| (i as f32 * 0.02).cos()).collect();
    c.bench_function("cosine_384d", |b_| {
        b_.iter(|| cosine_similarity(black_box(&a), black_box(&b)))
    });
}

fn bench_semantic_gate(c: &mut Criterion) {
    let mut gate = SemanticGate::new(0.35, 50);
    let vecs: Vec<Vec<f32>> = (0..100)
        .map(|i| (0..384).map(|j| ((i * 384 + j) as f32 * 0.001).sin()).collect())
        .collect();
    c.bench_function("semantic_gate_admit", |b| {
        let mut idx = 0;
        b.iter(|| {
            gate.admit(black_box(&vecs[idx % 100]));
            idx += 1;
        })
    });
}

fn configured_criterion() -> Criterion {
    Criterion::default()
        .warm_up_time(Duration::from_secs(3))
        .measurement_time(Duration::from_secs(5))
}

criterion_group!(
    name = benches;
    config = configured_criterion();
    targets =
        bench_parse_csv_line,
        bench_monotonic_stamp,
        bench_ticker_scan,
        bench_cosine_384,
        bench_semantic_gate,
);
criterion_main!(benches);
