//! Zero-copy CSV/FIX ingestion edge for the Semantic Contagion pipeline.
//!
//! The parser returns byte-slice references into the original input buffer,
//! avoiding heap allocation on the hot path.  A monotonic clock enforces
//! strictly increasing event timestamps.

use std::time::Instant;

// ── Zero-copy CSV record ────────────────────────────────────────────────

/// A parsed CSV row referencing the original buffer (no allocation).
#[derive(Debug)]
pub struct NewsRecord<'a> {
    pub date: &'a str,
    pub ticker: &'a str,
    pub title: &'a str,
    pub article: &'a str,
}

/// Parse a single CSV line into borrowed slices.  Returns `None` on
/// malformed input.  Assumes columns: Date, Stock_symbol, Article_title,
/// Article (at minimum 4 comma-separated fields).
pub fn parse_csv_line(line: &str) -> Option<NewsRecord<'_>> {
    let mut fields = Vec::with_capacity(4);
    let mut start = 0;
    let mut in_quote = false;

    for (i, ch) in line.char_indices() {
        match ch {
            '"' => in_quote = !in_quote,
            ',' if !in_quote => {
                fields.push(&line[start..i]);
                start = i + 1;
                if fields.len() == 3 {
                    // Everything remaining is the article body
                    fields.push(&line[start..]);
                    break;
                }
            }
            _ => {}
        }
    }
    if fields.len() < 4 {
        // Fewer than 4 commas — push whatever is left
        fields.push(&line[start..]);
    }
    if fields.len() < 4 {
        return None;
    }

    Some(NewsRecord {
        date: fields[0].trim().trim_matches('"'),
        ticker: fields[1].trim().trim_matches('"'),
        title: fields[2].trim().trim_matches('"'),
        article: fields[3].trim().trim_matches('"'),
    })
}

/// Parse an entire CSV buffer (header + data rows) into records.
/// Zero-copy: all string references point into `buf`.
pub fn parse_csv_buffer(buf: &str) -> Vec<NewsRecord<'_>> {
    let mut records = Vec::new();
    let mut lines = buf.lines();
    let _header = lines.next(); // skip header row
    for line in lines {
        if line.is_empty() {
            continue;
        }
        if let Some(rec) = parse_csv_line(line) {
            records.push(rec);
        }
    }
    records
}

// ── Monotonic timestamp clock ───────────────────────────────────────────

/// A monotonic clock that guarantees strictly increasing timestamps.
/// In deployment, this would use `clock_gettime(CLOCK_MONOTONIC)` or
/// ARM architectural counters; here we wrap `std::time::Instant`.
pub struct MonotonicClock {
    epoch: Instant,
    last_ns: u64,
}

impl MonotonicClock {
    pub fn new() -> Self {
        Self {
            epoch: Instant::now(),
            last_ns: 0,
        }
    }

    /// Returns a strictly increasing nanosecond timestamp.
    pub fn stamp(&mut self) -> u64 {
        let ns = self.epoch.elapsed().as_nanos() as u64;
        let ts = if ns > self.last_ns { ns } else { self.last_ns + 1 };
        self.last_ns = ts;
        ts
    }
}

impl Default for MonotonicClock {
    fn default() -> Self {
        Self::new()
    }
}

// ── Entity extraction (branchless ticker match) ─────────────────────────

/// Fast ticker mention scanner.  Returns indices into `tickers` that
/// appear as uppercase word-bounded tokens in `text`.
pub fn scan_tickers<'a>(text: &str, tickers: &'a [&str]) -> Vec<&'a str> {
    let upper = text.to_uppercase();
    let bytes = upper.as_bytes();
    let mut found = Vec::new();

    for &tk in tickers {
        let tk_bytes = tk.as_bytes();
        let mut pos = 0;
        while pos + tk_bytes.len() <= bytes.len() {
            if let Some(idx) = memchr::memmem::find(&bytes[pos..], tk_bytes) {
                let abs = pos + idx;
                let before_ok =
                    abs == 0 || !bytes[abs - 1].is_ascii_alphanumeric();
                let after_ok = abs + tk_bytes.len() >= bytes.len()
                    || !bytes[abs + tk_bytes.len()].is_ascii_alphanumeric();
                if before_ok && after_ok {
                    found.push(tk);
                    break;
                }
                pos = abs + 1;
            } else {
                break;
            }
        }
    }
    found
}

// ── Semantic dedup gate ─────────────────────────────────────────────────

/// Cosine similarity between two f32 slices (no SIMD, portable).
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom < 1e-9 { 0.0 } else { dot / denom }
}

/// Rolling centroid gate: returns `true` if the vector is novel enough
/// (cosine similarity to all active centroids < threshold).
pub struct SemanticGate {
    centroids: Vec<Vec<f32>>,
    threshold: f32,
    max_centroids: usize,
}

impl SemanticGate {
    pub fn new(threshold: f32, max_centroids: usize) -> Self {
        Self {
            centroids: Vec::new(),
            threshold,
            max_centroids,
        }
    }

    /// Returns `true` if the embedding is novel (should be forwarded).
    pub fn admit(&mut self, embedding: &[f32]) -> bool {
        for c in &self.centroids {
            if cosine_similarity(c, embedding) >= self.threshold {
                return false;
            }
        }
        if self.centroids.len() >= self.max_centroids {
            self.centroids.remove(0);
        }
        self.centroids.push(embedding.to_vec());
        true
    }
}

// ── PyO3 bridge (enabled with `--features python`) ─────────────────────

#[cfg(feature = "python")]
mod pybridge {
    use super::*;
    use pyo3::prelude::*;

    /// Python-accessible parsed record (owned strings for safety across FFI).
    #[pyclass]
    #[derive(Clone)]
    pub struct PyNewsRecord {
        #[pyo3(get)]
        pub date: String,
        #[pyo3(get)]
        pub ticker: String,
        #[pyo3(get)]
        pub title: String,
        #[pyo3(get)]
        pub article: String,
        #[pyo3(get)]
        pub timestamp_ns: u64,
    }

    #[pyfunction]
    fn parse_csv(path: &str) -> PyResult<Vec<PyNewsRecord>> {
        let buf = std::fs::read_to_string(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let mut clock = MonotonicClock::new();
        let records = parse_csv_buffer(&buf);
        Ok(records
            .into_iter()
            .map(|r| PyNewsRecord {
                date: r.date.to_string(),
                ticker: r.ticker.to_string(),
                title: r.title.to_string(),
                article: r.article.to_string(),
                timestamp_ns: clock.stamp(),
            })
            .collect())
    }

    #[pyfunction]
    fn scan_tickers_py(text: &str, tickers: Vec<String>) -> Vec<String> {
        let refs: Vec<&str> = tickers.iter().map(|s| s.as_str()).collect();
        scan_tickers(text, &refs)
            .into_iter()
            .map(|s| s.to_string())
            .collect()
    }

    #[pymodule]
    fn contagion_edge(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(parse_csv, m)?)?;
        m.add_function(wrap_pyfunction!(scan_tickers_py, m)?)?;
        m.add_class::<PyNewsRecord>()?;
        Ok(())
    }
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_simple_line() {
        let line = "2022-07-01,AAPL,Apple rises,Apple Inc stock rose 3% today";
        let rec = parse_csv_line(line).unwrap();
        assert_eq!(rec.date, "2022-07-01");
        assert_eq!(rec.ticker, "AAPL");
        assert_eq!(rec.title, "Apple rises");
        assert!(rec.article.contains("Apple Inc"));
    }

    #[test]
    fn parse_quoted_csv() {
        let line = r#""2022-07-01","NVDA","Nvidia Q2","Revenue beat, margins up""#;
        let rec = parse_csv_line(line).unwrap();
        assert_eq!(rec.ticker, "NVDA");
    }

    #[test]
    fn monotonic_clock_strictly_increasing() {
        let mut clock = MonotonicClock::new();
        let t1 = clock.stamp();
        let t2 = clock.stamp();
        let t3 = clock.stamp();
        assert!(t2 > t1);
        assert!(t3 > t2);
    }

    #[test]
    fn ticker_scan_finds_matches() {
        let tickers = &["AAPL", "NVDA", "MSFT", "GOOG"];
        let text = "Apple (AAPL) and NVDA both rallied. GOOG fell.";
        let found = scan_tickers(text, tickers);
        assert!(found.contains(&"AAPL"));
        assert!(found.contains(&"NVDA"));
        assert!(found.contains(&"GOOG"));
        assert!(!found.contains(&"MSFT"));
    }

    #[test]
    fn ticker_scan_no_partial_match() {
        let tickers = &["AAL", "AA"];
        let text = "American Airlines AAL rose. AA not mentioned separately.";
        let found = scan_tickers(text, tickers);
        assert!(found.contains(&"AAL"));
        // "AA" appears as substring of "AAL" but should also match standalone
        assert!(found.contains(&"AA"));
    }

    #[test]
    fn semantic_gate_dedup() {
        let mut gate = SemanticGate::new(0.95, 10);
        let v1 = vec![1.0, 0.0, 0.0];
        let v2 = vec![0.0, 1.0, 0.0];
        let v1_near = vec![0.999, 0.001, 0.0];

        assert!(gate.admit(&v1));       // novel
        assert!(gate.admit(&v2));       // novel (orthogonal)
        assert!(!gate.admit(&v1_near)); // duplicate of v1
    }

    #[test]
    fn cosine_self() {
        let v = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 1e-5);
    }
}
