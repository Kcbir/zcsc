# Data Setup

The raw data used in this paper is the **FNSPID** corpus (Chen et al., CIKM 2023),
restricted to July 2022. It must not be committed to this repository due to the
dataset's redistribution licence.

---

## 1. Download Source

**Dataset:** FNSPID — Financial News and Stock Price Integration Dataset  
**Paper:** Zihan Chen, Yijia Zhao, Qian Zhao, Zhaowei Liu, Junjie Han.
"FNSPID: A Comprehensive Financial News Dataset in Time Series."
*Proc. 32nd ACM Int. Conf. on Information & Knowledge Management (CIKM 2023).*  
**DOI:** 10.1145/3583780.3615283  
**Repository:** https://github.com/Zihan1004/FNSPID

---

## 2. Expected Files and SHA-256 Hashes

After slicing to July 2022, the following files must be placed in this directory:

| File | Description | SHA-256 |
|---|---|---|
| `july_2022_news.csv` | 638 news articles, July 2022 | `2edf12709f3ed907c9794ab66012b18e965f8d63a10684ba8222a2711b47ecc5` |
| `july_2022_quant_all_stocks.csv` | Combined OHLCV for all 47 tickers | `8c9fd415c37a95f284f9b4016b975ad608af318061c3d48d9284013c87294b4f` |
| `July_2022_Quant_Per_Ticker/<TICKER>.csv` | Per-ticker OHLCV + GPT sentiment (47 files) | — |

Verify with:

```bash
shasum -a 256 july_2022_news.csv july_2022_quant_all_stocks.csv
```

---

## 3. Preprocessing Command Sequence

```bash
# 1. Clone FNSPID
git clone https://github.com/Zihan1004/FNSPID.git fnspid_raw

# 2. Slice to July 2022 and place files in data/
python - <<'EOF'
import pandas as pd, shutil, os
from pathlib import Path

raw = Path("fnspid_raw")
out = Path("data")

# News slice
news = pd.read_csv(raw / "Stock_news" / "Stock_news.csv")
news["Date"] = pd.to_datetime(news["Date"], utc=True)
july = news[(news["Date"].dt.year == 2022) & (news["Date"].dt.month == 7)]
july.to_csv(out / "july_2022_news.csv", index=False)
print(f"News articles: {len(july)}")

# Combined OHLCV
quant = pd.read_csv(raw / "Stock_price" / "all_stocks.csv")
quant["Date"] = pd.to_datetime(quant["Date"], utc=True)
july_q = quant[(quant["Date"].dt.year == 2022) & (quant["Date"].dt.month == 7)]
july_q.to_csv(out / "july_2022_quant_all_stocks.csv", index=False)

# Per-ticker files (copy July rows)
ticker_dir = out / "July_2022_Quant_Per_Ticker"
ticker_dir.mkdir(exist_ok=True)
for fp in sorted((raw / "Stock_price" / "per_ticker").iterdir()):
    if fp.suffix != ".csv":
        continue
    df = pd.read_csv(fp)
    df["Date"] = pd.to_datetime(df["Date"], utc=True)
    df_july = df[(df["Date"].dt.year == 2022) & (df["Date"].dt.month == 7)]
    if len(df_july):
        df_july.to_csv(ticker_dir / fp.name, index=False)
print("Done.")
EOF

# 3. Verify hashes
shasum -a 256 data/july_2022_news.csv data/july_2022_quant_all_stocks.csv
```

---

## 4. Note on Sentiment Scores

No GPT API calls are made by this codebase. The `Sentiment_gpt` and
`Scaled_sentiment` columns are consumed directly as provided in FNSPID.
The GPT-4 scoring was performed by the original dataset authors and is
part of the distributed corpus.

---

## 5. After Data Placement

Run the preprocessing pipeline:

```bash
cd implementation/scripts
python load_fnspid.py
```

This reads the raw files from `data/`, builds all cache artefacts, and
writes them to `cache/`. Expected output:

```
STEP 1 / 5 — Loading data
  News articles: 638
  Date range: 2022-07-01 00:00:00+00:00 → 2022-07-29 00:00:00+00:00
  Tickers with quant data: 47
...
  All artefacts saved to .../cache/
```
